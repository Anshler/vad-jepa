"""
Quick validation of a loaded checkpoint on a small subset of test data.

Usage (WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vad-jepa

    # Swin+LSTM (remapped MOVAD checkpoint)
    python tests/quick_val.py --config cfgs/swin_lstm.yaml \
        --checkpoint output/v4_1/checkpoints/model-640_remapped.pt \
        --max_videos 50

    # V-JEPA+Mamba (your trained model)
    python tests/quick_val.py --config cfgs/vjepa_mamba.yaml \
        --checkpoint output/vjepa_mamba_VCL_8/checkpoints/model-100.pt \
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

from model import build_multi_head_vjepa
from movad_core.dota import Dota, gt_cls_target, setup_dota
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="cfgs/swin_lstm.yaml")
parser.add_argument("--checkpoint", required=True,
                    help="Path to checkpoint .pt file")
parser.add_argument("--max_videos", type=int, default=0,
                    help="Limit test to first N videos (0 = all)")
parser.add_argument("--data_path", default=None,
                    help="Override data_path from config")
parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for subset selection (default: 42)")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load config ─────────────────────────────────────────────────────────
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

# ── Load model ──────────────────────────────────────────────────────────
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
if missing:
    for k in missing:
        print(f"    - {k}")

# ── Build test dataset (subset) ─────────────────────────────────────────
NF = cfg.get("NF", cfg.get("num_frames", 4))
img_size = cfg.get("img_size", 256)
input_shape = cfg.get("input_shape", [img_size, img_size])

test_cfg = EasyDict(dict(cfg))
test_cfg.batch_size = 1  # one video at a time for simplicity
test_cfg.input_shape = input_shape

_, test_loader = setup_dota(Dota, test_cfg, num_workers=0, VCL=None, phase="test")

dataset = test_loader.dataset
total = len(dataset)
limit = args.max_videos if args.max_videos > 0 else total
limit = min(limit, total)

# Shuffle indices with fixed seed for reproducible random subset
rng = np.random.RandomState(args.seed)
indices = rng.permutation(total)[:limit].tolist()
print(f"\nTest videos: {limit}/{total}  (shuffled, seed={args.seed})")

# ── Run validation ──────────────────────────────────────────────────────
fb = NF
model.eval()
head.eval()

targets_all, outputs_all, toas_all, teas_all = [], [], [], []
idxs_all, info_all, frames_counter = [], [], []
raw_logits_all = []  # collect raw logits (before softmax) to diagnose training regime

amp_dtype = torch.float16 if cfg.get("amp_dtype", "fp32") == "fp16" else None
autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else __import__('contextlib').nullcontext()

for idx in tqdm(indices, desc="Evaluating"):
    video_data, data_info_raw = dataset[idx]

    # Format: frames → tensor [1, C, T, H, W]
    if isinstance(video_data, np.ndarray):
        frames_tensor = torch.from_numpy(video_data).float()
    else:
        frames_tensor = video_data.float()

    # dataset returns [T, C, H, W] → need [1, C, T, H, W]
    if frames_tensor.dim() == 4:
        frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]

    video_data = frames_tensor.to(DEVICE)
    data_info = torch.tensor(data_info_raw).float().unsqueeze(0).to(DEVICE) if not isinstance(data_info_raw, torch.Tensor) else data_info_raw.float().unsqueeze(0).to(DEVICE)

    v_len = video_data.shape[2]
    B = video_data.shape[0]

    with torch.no_grad():
        video_len_orig = data_info[:, 0]
        idx_batch = data_info[:, 1]
        toa_batch = data_info[:, 2]
        tea_batch = data_info[:, 3]
        info_batch = data_info[:, 7:11]

        video_targets = []
        video_outputs = []
        state = None
        vl = int(video_len_orig[0].item())
        valid_frames = vl - fb
        if valid_frames <= 0:
            continue

        for i in range(fb, v_len):
            target = gt_cls_target(i, toa_batch, tea_batch).long()
            clip = video_data[:, :, i - fb:i, :, :]   # [B, C, NF, H, W]

            with autocast_ctx:
                output, state = head(clip, state)       # full forward: encoder → temporal → classifier

            flt = i >= video_len_orig
            target[flt] = -100
            output[flt] = -100

            # Capture raw logits BEFORE softmax for diagnosis
            if i - fb < valid_frames:
                raw_logits_all.append(output[0].detach().cpu().clone())

            if cfg.get("apply_softmax", False):
                output = output.softmax(dim=1)

            # Store class-1 probability (not raw logit) so threshold 0.5 is meaningful
            prob = output if cfg.get("apply_softmax", False) else output.softmax(dim=1)
            if i - fb < valid_frames:
                video_targets.append(target[0].item())
                video_outputs.append(prob[0, 1].item())

    targets_all.append(video_targets)
    outputs_all.append(video_outputs)
    toas_all.append(toa_batch[0].item())
    teas_all.append(tea_batch[0].item())
    idxs_all.append(idx_batch[0].item())
    info_all.append(info_batch[0].tolist())
    frames_counter.append(vl)

    # Periodically flush CUDA cache
    if idx % 10 == 0:
        torch.cuda.empty_cache()

# ── Compute metrics (flat, main metrics only — skip per-class for quick runs) ─
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

preds = np.array([s for video in outputs_all for s in video])
gts = np.array([t for video in targets_all for t in video])

auc_roc = roc_auc_score(gts, preds)
auc_pr = average_precision_score(gts, preds)

# F1 at threshold 0.5
preds_bin = (preds > 0.5).astype(int)
tp = ((preds_bin == 1) & (gts == 1)).sum()
fp = ((preds_bin == 1) & (gts == 0)).sum()
fn = ((preds_bin == 0) & (gts == 1)).sum()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy = accuracy_score(gts, preds_bin)

print(f"\n{'='*60}")
print(f"RESULTS  (config={args.config}  videos={limit})")
print(f"{'='*60}")
print(f"  AUC-ROC:     {auc_roc:.4f}")
print(f"  AUC-PR:      {auc_pr:.4f}")
print(f"  F1 @0.5:     {f1:.4f}")
print(f"  Accuracy:    {accuracy:.4f}")
print(f"  Precision:   {precision:.4f}")
print(f"  Recall:      {recall:.4f}")
print(f"  Anomaly %:   {gts.mean():.4f}  ({gts.sum():.0f}/{len(gts)} frames)")
print(f"{'='*60}")

# ── Raw logit diagnosis ─────────────────────────────────────────────────
# Double-softmax: loss = CE(softmax(softmax(logits)), target)
#   The inner softmax squashes logits → [0,1]; the outer softmax then
#   softmax([p, 1-p]) = sigmoid(2p-1) ≈ [0.27, 0.73] max confidence.
#   Gradient dL/d(b-a) = (sigmoid(sigmoid(b-a)) - t) * sigmoid'(b-a) * sigmoid'(sigmoid(b-a))
#   At |b-a| = 2: gradient ≈ 0.02×  (90% dead)
#   At |b-a| = 4: gradient ≈ 0.0001× (dead)
#   The model CANNOT push |b-a| past ~3 — there's no gradient signal.
#
# Standard CE: loss = CE(logits, target)
#   Gradient dL/d(b-a) = (sigmoid(b-a) - t) — never vanishes.
#   |b-a| grows unbounded as the model gains confidence.
#
# So the diagnostic is |diff| = |b-a|:
#   |diff|_max < 3   → consistent with double-softmax training
#   |diff|_max > 6   → physically impossible under double-softmax
if raw_logits_all:
    raw = torch.stack(raw_logits_all)  # [N, 2]
    class0 = raw[:, 0]
    class1 = raw[:, 1]
    diff = class1 - class0  # sigmoid(diff) = softmax(logits)[:, 1]
    d_abs_max = diff.abs().max().item()
    print(f"\n{'='*60}")
    print("RAW LOGIT DIAGNOSIS (before softmax)")
    print(f"{'='*60}")
    print(f"  Samples:           {len(raw_logits_all)}")
    print(f"  class-0 logit:     [{class0.min():.2f}, {class0.max():.2f}]  mean={class0.mean():.2f}  std={class0.std():.2f}")
    print(f"  class-1 logit:     [{class1.min():.2f}, {class1.max():.2f}]  mean={class1.mean():.2f}  std={class1.std():.2f}")
    print(f"  diff = b-a:        [{diff.min():.2f}, {diff.max():.2f}]  mean={diff.mean():.2f}  std={diff.std():.2f}")
    # Anti-correlation check: if a ≈ -b then softmax on/off same ranking
    corr_ab = (class0 * class1).mean() / (class0.std() * class1.std() + 1e-8)
    print(f"  corr(a, b):        {corr_ab:.3f}  (≈ -1.0 → a = -b → softmax on/off same AUC)")
    print()
    if d_abs_max > 6:
        print(f"  |b-a|_max = {d_abs_max:.1f} > 6 → gradient-dead under double-softmax")
        print(f"  ✓ CHECKPOINT WAS TRAINED WITHOUT DOUBLE-SOFTMAX")
    elif d_abs_max > 3:
        print(f"  |b-a|_max = {d_abs_max:.1f} > 3 → unlikely under double-softmax")
        print(f"  ⚠ CHECKPOINT WAS LIKELY TRAINED WITHOUT DOUBLE-SOFTMAX")
    else:
        print(f"  |b-a|_max = {d_abs_max:.1f} < 3 → compatible with double-softmax")
        print(f"  → Inconclusive — could be either regime")
    print(f"{'='*60}")