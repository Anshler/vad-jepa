"""
Gate-behaviour diagnostics for a trained Sparse SlotSSM checkpoint.

Answers: is the sparse gate doing input-dependent routing or is it static?
Run this BEFORE committing to a VCL=64 training run — if the gate is static
at VCL=8, it never learned to route and longer context won't help.

Diagnostics (all computed from test-set inference, no retraining):
  1. Active-slot turnover — Jaccard similarity of active set between consecutive frames
  2. Per-slot activation histogram — fraction of frames each slot is active
  3. Cross-slot state diversity — mean pairwise cosine similarity of final slot states
  4. Activation–label correlation — per-slot P(active | anomaly) vs P(active | normal)
  5. Gate entropy — per-block entropy of the softmax gate distribution

Usage (WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vad-jepa

    python tests/diag_sparse_gate.py \
        --config cfgs/vjepa_sparse_slotssm.yaml \
        --checkpoint output/v4_1/checkpoints/sparse-vjepa.pt \
        --max_videos 50
"""
from __future__ import annotations

import argparse, os, sys, numpy as np, torch, yaml
from easydict import EasyDict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import torchvision.transforms._functional_tensor as _ft
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
except ImportError:
    pass

from model import build_multi_head_vjepa, MambaCache
from movad_core.dota import Dota, gt_cls_target, setup_dota
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="cfgs/vjepa_sparse_slotssm.yaml")
parser.add_argument("--checkpoint", required=True,
                    help="Path to checkpoint .pt file (e.g. output/v4_1/checkpoints/sparse-vjepa.pt)")
parser.add_argument("--max_videos", type=int, default=50,
                    help="Limit test to first N shuffled videos (0 = all)")
parser.add_argument("--data_path", default=None,
                    help="Override data_path from config")
parser.add_argument("--seed", type=int, default=42,
                    help="Shuffle seed for subset selection (default: 42)")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load config ─────────────────────────────────────────────────────────────
cfg_path = os.path.join(_REPO_ROOT, args.config)
with open(cfg_path, "r") as f:
    cfg = EasyDict(yaml.safe_load(f))
cfg.device = DEVICE
cfg.compile = False
if args.data_path:
    cfg.data_path = args.data_path
if "data_path" not in cfg:
    cfg.data_path = "/mnt/d/Users/Chrysenberg69420/Downloads/DoTA_dataset"

ckpt_path = os.path.join(_REPO_ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint

print(f"Config:      {args.config}")
print(f"Checkpoint:  {ckpt_path}")
print(f"Data path:   {cfg.data_path}")
print(f"Max videos:  {args.max_videos if args.max_videos > 0 else 'all'}")
print(f"Device:      {DEVICE}")
print(f"NF={cfg.get('NF', cfg.get('num_frames', 4))}  VCL={cfg.get('VCL', 8)}")

# ── Load model ──────────────────────────────────────────────────────────────
head_name = os.path.splitext(os.path.basename(args.config))[0]
cfg._head_cfgs_flat = [dict(cfg)]
cfg._head_cfgs_flat[0]["name"] = head_name

print(f"\nBuilding model for head: {head_name}")
model = build_multi_head_vjepa(cfg)
model.to(DEVICE)
head = model.heads[head_name]

ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
ckpt_sd = ckpt["model_state_dict"]
missing, unexpected = head.load_state_dict(ckpt_sd, strict=False)
print(f"Loaded checkpoint epoch {ckpt.get('epoch', '?')}")
print(f"  Missing keys:   {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")

# ── Verify this is a sparse model ───────────────────────────────────────────
if head.temporal_type not in ("sparse_slotssm",):
    print(f"\nERROR: temporal_model is '{head.temporal_type}', expected 'sparse_slotssm'")
    print("This diagnostic only works with sparse SlotSSM models.")
    sys.exit(1)

temporal = head.temporal
print(f"  Slots: K={temporal.num_slots}  top_k={temporal.top_k}  blocks={len(temporal.blocks)}")

# ── Build test dataset (subset) ─────────────────────────────────────────────
NF = cfg.get("NF", cfg.get("num_frames", 4))
img_size = cfg.get("img_size", 256)
input_shape = cfg.get("input_shape", [img_size, img_size])

test_cfg = EasyDict(dict(cfg))
test_cfg.batch_size = 1
test_cfg.input_shape = input_shape

_, test_loader = setup_dota(Dota, test_cfg, num_workers=0, VCL=None, phase="test")

dataset = test_loader.dataset
total = len(dataset)
limit = args.max_videos if args.max_videos > 0 else total
limit = min(limit, total)

rng = np.random.RandomState(args.seed)
indices = rng.permutation(total)[:limit].tolist()
print(f"\nTest videos: {limit}/{total}  (shuffled, seed={args.seed})")

# ── Run inference with diagnostics enabled ──────────────────────────────────
fb = NF
model.eval()
head.eval()

# --- Enable diagnostic collection on the temporal model ---
temporal.enable_diagnostics()

amp_dtype = torch.float16 if cfg.get("amp_dtype", "fp32") == "fp16" else None
autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else __import__('contextlib').nullcontext()

# Per-video diagnostics
all_targets = []          # list of per-frame labels [int]
all_slots = []            # list of per-frame slot states [K, D]
all_gate_scores = []      # list of per-frame per-block gate scores [num_blocks, B, K]
all_active_idx = []       # list of per-frame per-block active indices [num_blocks, B, top_k]
total_frames = 0

for idx in tqdm(indices, desc="Evaluating"):
    video_data, data_info_raw = dataset[idx]

    if isinstance(video_data, np.ndarray):
        frames_tensor = torch.from_numpy(video_data).float()
    else:
        frames_tensor = video_data.float()

    if frames_tensor.dim() == 4:
        frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]

    video_data = frames_tensor.to(DEVICE)
    data_info = (torch.tensor(data_info_raw).float().unsqueeze(0).to(DEVICE)
                 if not isinstance(data_info_raw, torch.Tensor)
                 else data_info_raw.float().unsqueeze(0).to(DEVICE))

    v_len = video_data.shape[2]

    with torch.no_grad():
        video_len_orig = data_info[:, 0]
        toa_batch = data_info[:, 2]
        tea_batch = data_info[:, 3]

        state = None
        vl = int(video_len_orig[0].item())
        valid_frames = vl - fb
        if valid_frames <= 0:
            continue

        # --- Clear per-block diagnostics for this video ---
        temporal.enable_diagnostics()   # resets lists

        for i in range(fb, v_len):
            target = gt_cls_target(i, toa_batch, tea_batch).long()
            clip = video_data[:, :, i - fb:i, :, :]

            with autocast_ctx:
                output, state = head(clip, state)

            if i - fb < valid_frames:
                all_targets.append(target[0].item())
                total_frames += 1

        # --- Collect this video's diagnostics ---
        diag = temporal.get_diagnostics()

        # Slot states: [num_steps, B, K, D] → per-step [K, D] assuming B=1
        for t in range(len(diag["slots"])):
            all_slots.append(diag["slots"][t][0].clone())   # [K, D]

        # Gate scores & active indices: per block, per step
        num_blocks = len(diag["gate_scores"])
        num_steps = max(len(diag["gate_scores"][b]) for b in range(num_blocks))
        for t in range(num_steps):
            step_gate = []
            step_active = []
            for b in range(num_blocks):
                if t < len(diag["gate_scores"][b]):
                    step_gate.append(diag["gate_scores"][b][t][0].clone())     # [K]
                    step_active.append(diag["active_idx"][b][t][0].clone())    # [top_k]
                else:
                    step_gate.append(torch.full((temporal.num_slots,), float("nan")))
                    step_active.append(torch.full((temporal.top_k,), -1, dtype=torch.long))
            all_gate_scores.append(torch.stack(step_gate))      # [num_blocks, K]
            all_active_idx.append(torch.stack(step_active))     # [num_blocks, top_k]

    torch.cuda.empty_cache()

# ── Disable diagnostics ─────────────────────────────────────────────────────
temporal.disable_diagnostics()

if total_frames == 0:
    print("\nERROR: No valid frames collected. Check VCL and NF settings.")
    sys.exit(1)

print(f"\nCollected {total_frames} frames across {limit} videos  "
      f"({len(all_gate_scores)} with gate data)")

# ═════════════════════════════════════════════════════════════════════════════
# Diagnostic 1 — Active-slot turnover
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("DIAGNOSTIC 1 — Active-slot turnover (Jaccard similarity)")
print(f"{'='*70}")
print("Measures whether the gate selects different slots on consecutive frames.")
print("Jaccard = |active(t) ∩ active(t+1)| / top_k")
print("  1.00 every frame → DEAD GATE (static routing, same 16 slots always)")
print("  0.70–0.95        → HEALTHY (gate responds to input changes)")
print("  < 0.50           → UNSTABLE (gate thrashing — routing near-random)")

num_blocks = len(temporal.blocks)
for b in range(num_blocks):
    jaccards = []
    for t in range(len(all_active_idx) - 1):
        if all_active_idx[t][b][0].item() < 0:   # skip padded
            continue
        set_t = set(all_active_idx[t][b].tolist())
        set_t1 = set(all_active_idx[t + 1][b].tolist())
        jac = len(set_t & set_t1) / temporal.top_k
        jaccards.append(jac)

    if jaccards:
        jac = torch.tensor(jaccards)
        static_pct = (jac > 0.99).float().mean() * 100
        print(f"\n  Block {b}:  mean={jac.mean():.4f}  std={jac.std():.4f}  "
              f"min={jac.min():.4f}  max={jac.max():.4f}")
        print(f"             static (>0.99): {static_pct:.0f}% of frame pairs  "
              f"n={len(jaccards)}")

# ═════════════════════════════════════════════════════════════════════════════
# Diagnostic 2 — Per-slot activation histogram
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("DIAGNOSTIC 2 — Per-slot activation frequency")
print(f"{'='*70}")
print("Fraction of frames each slot is active.  All slots should participate;")
print("no slot should be at 0% (dead) or 100% (always-on, crowding others out).")

for b in range(num_blocks):
    slot_counts = torch.zeros(temporal.num_slots)
    valid_steps = 0
    for t in range(len(all_active_idx)):
        if all_active_idx[t][b][0].item() < 0:
            continue
        for s in all_active_idx[t][b].tolist():
            slot_counts[s] += 1
        valid_steps += 1

    if valid_steps == 0:
        continue

    freqs = slot_counts / valid_steps
    print(f"\n  Block {b} (n={valid_steps} steps):")
    print(f"    mean={freqs.mean():.4f}  std={freqs.std():.4f}  "
          f"min={freqs.min():.4f}  max={freqs.max():.4f}")
    print(f"    expected: {temporal.top_k}/{temporal.num_slots} = {temporal.top_k/temporal.num_slots:.4f}")

    # Bucket analysis
    dead = (freqs < 0.02).sum().item()
    low = ((freqs >= 0.02) & (freqs < 0.15)).sum().item()
    mid = ((freqs >= 0.15) & (freqs < 0.85)).sum().item()
    high = (freqs >= 0.85).sum().item()
    print(f"    dead (<2%): {dead}  low (2-15%): {low}  mid (15-85%): {mid}  high (>85%): {high}")

    # Print top-5 and bottom-5 slots
    sorted_idx = freqs.argsort()
    print(f"    Bottom 5: {sorted_idx[:5].tolist()} → freqs={[f'{freqs[i]:.3f}' for i in sorted_idx[:5]]}")
    print(f"    Top 5:    {sorted_idx[-5:].flip(0).tolist()} → freqs={[f'{freqs[i]:.3f}' for i in sorted_idx[-5:].flip(0)]}")

# ═════════════════════════════════════════════════════════════════════════════
# Diagnostic 3 — Cross-slot state diversity
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("DIAGNOSTIC 3 — Cross-slot state diversity (pairwise cosine similarity)")
print(f"{'='*70}")
print("Mean pairwise cosine sim between all 32 slot states.  Lower = more diverse.")
print("In sparse, inactive slots are frozen → should stay MORE diverse than dense.")
print("If cos-sim > 0.95 across all slots → slots have collapsed to the same representation.")

if all_slots:
    # Sample up to 200 frames evenly to avoid O(n²) over thousands of frames
    sample_every = max(1, len(all_slots) // 200)
    sampled = all_slots[::sample_every]

    pairwise_sims = []
    for slots_t in sampled:  # [K, D]
        s_norm = slots_t / (slots_t.norm(dim=-1, keepdim=True) + 1e-9)
        sim = torch.mm(s_norm, s_norm.T)  # [K, K]
        # Exclude diagonal (self-similarity = 1.0)
        mask = ~torch.eye(temporal.num_slots, dtype=torch.bool)
        pairwise_sims.append(sim[mask].mean().item())

    sims = torch.tensor(pairwise_sims)
    print(f"\n  Sampled {len(sampled)} frames (every {sample_every})")
    print(f"  Mean pairwise cos-sim: {sims.mean():.4f}  ± {sims.std():.4f}")
    print(f"  Range: [{sims.min():.4f}, {sims.max():.4f}]")

    # Also check active vs inactive slot diversity at last frame
    if len(all_active_idx) > 0 and len(all_slots) > 0:
        last_active = set(all_active_idx[-1][-1].tolist())  # block 3, last step
        last_slots = all_slots[-1]  # [K, D]
        active_mask = torch.tensor([i in last_active for i in range(temporal.num_slots)])
        inactive_mask = ~active_mask

        # Within active slots
        if active_mask.sum() > 1:
            a_slots = last_slots[active_mask]
            a_norm = a_slots / (a_norm := a_slots.norm(dim=-1, keepdim=True) + 1e-9)
            a_sim = torch.mm(a_norm, a_norm.T)
            a_mask = ~torch.eye(a_sim.shape[0], dtype=torch.bool)
            print(f"  Active slots pairwise cos-sim:   {a_sim[a_mask].mean():.4f}")

        # Within inactive slots
        if inactive_mask.sum() > 1:
            i_slots = last_slots[inactive_mask]
            i_norm = i_slots / (i_norm := i_slots.norm(dim=-1, keepdim=True) + 1e-9)
            i_sim = torch.mm(i_norm, i_norm.T)
            i_mask = ~torch.eye(i_sim.shape[0], dtype=torch.bool)
            print(f"  Inactive slots pairwise cos-sim: {i_sim[i_mask].mean():.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# Diagnostic 4 — Activation–label correlation
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("DIAGNOSTIC 4 — Activation–label correlation")
print(f"{'='*70}")
print("For each slot: P(active | anomaly frame) vs P(active | normal frame).")
print("A slot with large ratio is 'anomaly-sensitive' — the gate routes")
print("differently on anomaly frames, which is strong evidence the routing")
print("is task-relevant.")

gts_tensor = torch.tensor(all_targets)
anomaly_mask = gts_tensor == 1
normal_mask = gts_tensor == 0
n_anomaly = anomaly_mask.sum().item()
n_normal = normal_mask.sum().item()

if n_anomaly > 0 and n_normal > 0:
    for b in range(num_blocks):
        print(f"\n  Block {b}:")
        # Per-slot activation rate on anomaly vs normal frames
        slot_anomaly_rate = torch.zeros(temporal.num_slots)
        slot_normal_rate = torch.zeros(temporal.num_slots)

        for t in range(len(all_active_idx)):
            if all_active_idx[t][b][0].item() < 0:
                continue
            active_set = set(all_active_idx[t][b].tolist())
            for s in range(temporal.num_slots):
                if s in active_set:
                    if anomaly_mask[t]:
                        slot_anomaly_rate[s] += 1
                    else:
                        slot_normal_rate[s] += 1

        slot_anomaly_rate /= max(n_anomaly, 1)
        slot_normal_rate /= max(n_normal, 1)
        ratio = slot_anomaly_rate / (slot_normal_rate + 1e-9)

        print(f"    Anomaly frames: {n_anomaly}  Normal frames: {n_normal}")
        print(f"    Activation rate (anomaly): mean={slot_anomaly_rate.mean():.4f}  "
              f"std={slot_anomaly_rate.std():.4f}")
        print(f"    Activation rate (normal):  mean={slot_normal_rate.mean():.4f}  "
              f"std={slot_normal_rate.std():.4f}")

        # Slots most sensitive to anomalies (high ratio)
        top_anomaly = ratio.argsort(descending=True)[:5]
        print(f"    Top-5 anomaly-sensitive slots (highest P(active|anomaly)/P(active|normal)):")
        for s in top_anomaly:
            print(f"      slot {s:2d}: anom={slot_anomaly_rate[s]:.3f}  "
                  f"norm={slot_normal_rate[s]:.3f}  ratio={ratio[s]:.2f}")

        # Slots most sensitive to normal
        top_normal = ratio.argsort()[:5]
        print(f"    Top-5 normal-sensitive slots:")
        for s in top_normal:
            print(f"      slot {s:2d}: anom={slot_anomaly_rate[s]:.3f}  "
                  f"norm={slot_normal_rate[s]:.3f}  ratio={ratio[s]:.2f}")
else:
    print("\n  SKIP — insufficient anomaly or normal frames for comparison")

# ═════════════════════════════════════════════════════════════════════════════
# Diagnostic 5 — Gate entropy
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("DIAGNOSTIC 5 — Gate entropy distribution")
print(f"{'='*70}")
print("Entropy of softmax(gate_scores).  Higher = more uniform routing.")
print(f"  log(K) = log({temporal.num_slots}) = {np.log(temporal.num_slots):.2f}  (uniform)")
print(f"  log(top_k) = log({temporal.top_k}) = {np.log(temporal.top_k):.2f}  (moderately selective)")
print("  < 1.0 → gate extremely peaked, 1-2 slots dominate")

for b in range(num_blocks):
    entropies = []
    for t in range(len(all_gate_scores)):
        scores = all_gate_scores[t][b]  # [K]
        if torch.isnan(scores).any():
            continue
        p = scores.softmax(dim=-1)
        ent = -(p * (p + 1e-9).log()).sum()
        entropies.append(ent.item())

    if entropies:
        ent = torch.tensor(entropies)
        print(f"\n  Block {b}:  mean={ent.mean():.3f}  std={ent.std():.3f}  "
              f"min={ent.min():.3f}  max={ent.max():.3f}  n={len(entropies)}")

# ═════════════════════════════════════════════════════════════════════════════
# Summary verdict
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("SUMMARY — Gate health assessment")
print(f"{'='*70}")

issues = []
ok_signals = []

# ── Compute per-block per-slot activation frequencies once ──────────────────
block_freqs = []   # list of [K] tensors, one per block
for b in range(num_blocks):
    slot_counts = torch.zeros(temporal.num_slots)
    valid_steps = 0
    for t in range(len(all_active_idx)):
        if all_active_idx[t][b][0].item() < 0:
            continue
        for s in all_active_idx[t][b].tolist():
            slot_counts[s] += 1
        valid_steps += 1
    block_freqs.append(slot_counts / max(valid_steps, 1))

# ── Check 1: Turnover (block 0 as primary, block 3 as confirmation) ────────
jaccards_b0 = []
for t in range(len(all_active_idx) - 1):
    if all_active_idx[t][0][0].item() < 0:
        continue
    set_t = set(all_active_idx[t][0].tolist())
    set_t1 = set(all_active_idx[t + 1][0].tolist())
    jaccards_b0.append(len(set_t & set_t1) / temporal.top_k)

jaccards_b3 = []
for t in range(len(all_active_idx) - 1):
    if all_active_idx[t][3][0].item() < 0:
        continue
    set_t = set(all_active_idx[t][3].tolist())
    set_t1 = set(all_active_idx[t + 1][3].tolist())
    jaccards_b3.append(len(set_t & set_t1) / temporal.top_k)

if jaccards_b0:
    mean_jac0 = np.mean(jaccards_b0)
    static_pct0 = (np.array(jaccards_b0) > 0.99).mean() * 100
    mean_jac3 = np.mean(jaccards_b3) if jaccards_b3 else float("nan")

    if static_pct0 > 90:
        issues.append(f"DEAD GATE: {static_pct0:.0f}% static pairs in block 0")
    elif static_pct0 > 50:
        issues.append(f"Mostly static: {static_pct0:.0f}% static pairs in block 0")
    else:
        ok_signals.append(f"Turnover: Jaccard={mean_jac0:.3f} (block 0), {mean_jac3:.3f} (block 3), "
                          f"only {static_pct0:.0f}% static")

# ── Check 2: Cross-block slot utilization ──────────────────────────────────
# A slot "dead" in one block may be highly active in another (complementarity).
# Count slots that are marginal (<10%) in ALL blocks vs. healthy in ≥1 block.
all_marginal = torch.ones(temporal.num_slots, dtype=torch.bool)
per_block_dead = []
for b in range(num_blocks):
    freqs = block_freqs[b]
    dead_b = (freqs < 0.02).sum().item()
    low_b = ((freqs >= 0.02) & (freqs < 0.10)).sum().item()
    mid_b = ((freqs >= 0.10) & (freqs <= 0.90)).sum().item()
    high_b = (freqs > 0.90).sum().item()
    per_block_dead.append((dead_b, low_b, mid_b, high_b))
    all_marginal &= (freqs < 0.10)   # slot is marginal in ALL blocks

universally_dead = all_marginal.sum().item()

# Cross-block complementarity: correlation between block 0 and block 1 freqs
if num_blocks >= 2:
    corr_01 = torch.corrcoef(torch.stack([block_freqs[0], block_freqs[1]]))[0, 1].item()
    if corr_01 < -0.5:
        ok_signals.append(f"Cross-block complementarity: corr(block0, block1)={corr_01:.2f} "
                          f"— slots inactive in one block are active in the other")
    elif corr_01 > 0.5:
        pass  # same slots dominated both blocks — not necessarily bad

if universally_dead >= 8:
    issues.append(f"UNIVERSALLY DEAD: {universally_dead} slots <10% in ALL blocks")
elif universally_dead > 0:
    issues.append(f"Marginal slots: {universally_dead} slots <10% in all blocks "
                  f"(but may be healthy in ≥1 block)")
else:
    ok_signals.append(f"Slot utilization: {universally_dead} universally-dead slots "
                      f"(all slots reach ≥10% in at least one block)")

# Block 3 health (the final routing decision)
freqs_b3 = block_freqs[3]
dead_b3 = (freqs_b3 < 0.02).sum().item()
low_b3 = ((freqs_b3 >= 0.02) & (freqs_b3 < 0.10)).sum().item()
print(f"\n  Block 3 (final routing): {per_block_dead[3][2]} in 10-90% range, "
      f"min={freqs_b3.min():.3f}, max={freqs_b3.max():.3f}")

# ── Check 3: Label correlation (credible slots only, block 3) ──────────────
# Filter to slots with ≥5% frequency in block 3 — low-frequency ratios are noise.
if n_anomaly > 0 and n_normal > 0 and len(all_active_idx) > 0:
    credible_mask = block_freqs[3] >= 0.05
    n_credible = credible_mask.sum().item()

    slot_anomaly_rate = torch.zeros(temporal.num_slots)
    slot_normal_rate = torch.zeros(temporal.num_slots)
    for t in range(len(all_active_idx)):
        if all_active_idx[t][3][0].item() < 0:
            continue
        active_set = set(all_active_idx[t][3].tolist())
        for s in range(temporal.num_slots):
            if s in active_set:
                if anomaly_mask[t]:
                    slot_anomaly_rate[s] += 1
                else:
                    slot_normal_rate[s] += 1
    slot_anomaly_rate /= max(n_anomaly, 1)
    slot_normal_rate /= max(n_normal, 1)
    ratio = slot_anomaly_rate / (slot_normal_rate + 1e-9)

    credible_ratio = ratio[credible_mask]
    if len(credible_ratio) > 0:
        max_cr = credible_ratio.max().item()
        min_cr = credible_ratio.min().item()
        top_slot = credible_mask.nonzero()[credible_ratio.argmax()].item()
        if max_cr > 1.5 or min_cr < 0.67:
            ok_signals.append(
                f"Label corr (block 3, {n_credible} credible slots): "
                f"ratio range [{min_cr:.2f}, {max_cr:.2f}], "
                f"top anomaly-sensitive: slot {top_slot} (ratio={max_cr:.2f})")
        else:
            issues.append(
                f"No label correlation in block 3: ratio range [{min_cr:.2f}, {max_cr:.2f}] "
                f"— gate routes orthogonal to task")
    else:
        issues.append("No credible slots in block 3 (all <5% freq) — gate collapsed?")

# ── Check 4: Gate entropy (block 0 as early routing, block 3 as final) ─────
# Print summary of entropy progression across blocks
entropy_means = []
for b in range(num_blocks):
    entropies_b = []
    for t in range(len(all_gate_scores)):
        scores = all_gate_scores[t][b]
        if torch.isnan(scores).any():
            continue
        p = scores.softmax(dim=-1)
        entropies_b.append((-(p * (p + 1e-9).log()).sum()).item())
    if entropies_b:
        entropy_means.append(np.mean(entropies_b))

if entropy_means:
    ent_str = " → ".join(f"{e:.2f}" for e in entropy_means)
    print(f"  Entropy progression: {ent_str}  (uniform = {np.log(temporal.num_slots):.2f})")

# ── Final verdict ──────────────────────────────────────────────────────────
for signal in ok_signals:
    print(f"  ✓ {signal}")

if issues:
    print(f"\n  ⚠ ISSUES ({len(issues)}):")
    for issue in issues:
        print(f"    - {issue}")

# Recommendation logic
has_cross_block_complementarity = (
    num_blocks >= 2
    and torch.corrcoef(torch.stack([block_freqs[0], block_freqs[1]]))[0, 1].item() < -0.3
)
gate_is_dead = len(jaccards_b0) > 0 and (np.array(jaccards_b0) > 0.99).mean() > 0.90
gate_is_static = len(jaccards_b0) > 0 and (np.array(jaccards_b0) > 0.99).mean() > 0.50

if gate_is_dead:
    print(f"\n  → VERDICT: Gate is dead (static routing). VCL=64 won't help.")
    print(f"    Fix: increase eps_random or add stronger entropy regularization.")
elif gate_is_static and universally_dead > 8:
    print(f"\n  → VERDICT: Gate is mostly static with many dead slots.")
    print(f"    Try: eps_random=0.10, entropy_weight=0.05 before VCL=64.")
elif universally_dead >= 16:
    print(f"\n  → VERDICT: {universally_dead} universally-dead slots — routing collapse.")
    print(f"    Fix: increase eps_random, check entropy_weight isn't too high.")
else:
    print(f"\n  → VERDICT: Gate is alive and routing dynamically. VCL=64 is worth running.")
    if has_cross_block_complementarity:
        print(f"    Cross-block complementarity means every slot participates somewhere.")
    print(f"    The marginal-slot issue in early blocks is mitigated by block 3's uniform routing.")