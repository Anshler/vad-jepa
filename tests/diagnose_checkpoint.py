"""
Diagnose a downloaded MOVAD checkpoint — compare against the swin_lstm model. Then convert to current format

Usage (WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vad-jepa
    python tests/diagnose_checkpoint.py
"""
from __future__ import annotations

import os, sys, torch, yaml
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

CKPT_PATH = os.path.join(_REPO_ROOT, "output", "v4_1", "checkpoints", "model-640.pt")
CFG_PATH = os.path.join(_REPO_ROOT, "cfgs", "swin_lstm.yaml")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Checkpoint: {CKPT_PATH}")
print(f"Config:     {CFG_PATH}")

# ── 1. Load checkpoint ──────────────────────────────────────────────────
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
ckpt_sd = ckpt.get("model_state_dict", ckpt)
ckpt_keys = sorted(ckpt_sd.keys())
print(f"\nEpoch: {ckpt.get('epoch', 'N/A')}  |  State dict: {len(ckpt_keys)} keys")

# ── 2. Build model from config ──────────────────────────────────────────
with open(CFG_PATH, "r") as f:
    cfg = EasyDict(yaml.safe_load(f))
cfg.device = DEVICE
cfg.compile = False

head_name = os.path.splitext(os.path.basename(CFG_PATH))[0]
cfg._head_cfgs_flat = [dict(cfg)]
cfg._head_cfgs_flat[0]["name"] = head_name

model = build_multi_head_vjepa(cfg)
model.to(DEVICE)
head = model.heads[head_name]
model_keys = sorted(head.state_dict().keys())

# ── 3. Show ALL checkpoint keys (not just first 2) grouped by component ─
print(f"\n{'='*70}")
print("CHECKPOINT KEYS (all)")
print(f"{'='*70}")

# Group by top-level prefix
from collections import defaultdict
groups = defaultdict(list)
for k in ckpt_keys:
    prefix = k.split(".")[0]
    groups[prefix].append(k)

for prefix in sorted(groups):
    keys = groups[prefix]
    print(f"\n  [{prefix}]  ({len(keys)} keys)")
    for k in keys[:8]:
        shape = tuple(ckpt_sd[k].shape)
        print(f"    {k:<65s}  {shape}")
    if len(keys) > 8:
        print(f"    ... +{len(keys)-8} more")

# ── 4. Show ALL model keys grouped by component ─────────────────────────
print(f"\n{'='*70}")
print("MODEL KEYS (all)")
print(f"{'='*70}")

from collections import defaultdict
mgroups = defaultdict(list)
for k in model_keys:
    prefix = k.split(".")[0]
    mgroups[prefix].append(k)

for prefix in sorted(mgroups):
    keys = mgroups[prefix]
    print(f"\n  [{prefix}]  ({len(keys)} keys)")
    for k in keys[:8]:
        shape = tuple(head.state_dict()[k].shape)
        print(f"    {k:<65s}  {shape}")
    if len(keys) > 8:
        print(f"    ... +{len(keys)-8} more")

# ── 5. Build the mapping ────────────────────────────────────────────────
print(f"\n{'='*70}")
print("PROPOSED MAPPING")
print(f"{'='*70}")

mapping = {}
unmapped_ckpt = []
unmapped_model = set(model_keys)

for ck in ckpt_keys:
    ck_shape = tuple(ckpt_sd[ck].shape)

    # Rule 1: model.* → encoder.swin.*  (Swin backbone)
    if ck.startswith("model."):
        new_key = "encoder.swin." + ck.removeprefix("model.")
        if new_key in unmapped_model:
            mapping[ck] = new_key
            unmapped_model.discard(new_key)
            continue

    # Rule 2: rnn.* → temporal.rnn.*  (LSTM)
    if ck.startswith("rnn."):
        new_key = "temporal." + ck
        if new_key in unmapped_model:
            mapping[ck] = new_key
            unmapped_model.discard(new_key)
            continue

    # Rule 3: rnn_bn.* → temporal.norm.*  (LSTM layer norm)
    if ck.startswith("rnn_bn."):
        suffix = ck.removeprefix("rnn_bn.")
        new_key = "temporal.norm." + suffix
        if new_key in unmapped_model:
            mapping[ck] = new_key
            unmapped_model.discard(new_key)
            continue

    # Rule 4: Exact name match  (bn, lin1, lin2, lin3)
    if ck in unmapped_model:
        mapping[ck] = ck
        unmapped_model.discard(ck)
        continue

    unmapped_ckpt.append(ck)

print(f"  Mapped:   {len(mapping)} keys")
print(f"  Unmapped checkpoint: {len(unmapped_ckpt)}")
print(f"  Unmapped model:      {len(unmapped_model)}")

if unmapped_ckpt:
    print(f"\n  Unmapped checkpoint keys:")
    for k in unmapped_ckpt[:20]:
        shape = tuple(ckpt_sd[k].shape)
        print(f"    {k:<65s}  {shape}")

if unmapped_model:
    print(f"\n  Unmapped model keys (unused by Swin path — safe to leave random):")
    for k in sorted(unmapped_model):
        shape = tuple(head.state_dict()[k].shape)
        print(f"    {k:<65s}  {shape}")

# ── 6. Shape verification of mapped keys ────────────────────────────────
print(f"\n{'='*70}")
print("SHAPE VERIFICATION (mapped keys)")
print(f"{'='*70}")

mismatches = 0
for ck, mk in sorted(mapping.items()):
    cs = tuple(ckpt_sd[ck].shape)
    ms = tuple(head.state_dict()[mk].shape)
    if cs != ms:
        print(f"  ✗ SHAPE MISMATCH: {ck} {cs} → {mk} {ms}")
        mismatches += 1

if mismatches == 0:
    print(f"  ✓ All {len(mapping)} mapped keys have matching shapes")

# ── 7. Remap and save ───────────────────────────────────────────────────
print(f"\n{'='*70}")
print("REMAP & SAVE")
print(f"{'='*70}")

remapped_sd = {mapping[ck]: ckpt_sd[ck] for ck in ckpt_keys}

# Verify: load into model
missing, unexpected = head.load_state_dict(remapped_sd, strict=False)
print(f"  Missing keys:     {len(missing)}")
print(f"  Unexpected keys:  {len(unexpected)}")

if len(missing) > 0:
    print(f"\n  Missing (will keep random init):")
    for k in missing:
        print(f"    {k}")
if len(unexpected) > 0:
    print(f"\n  Unexpected (in remapped but not model — shouldn't happen):")
    for k in unexpected:
        print(f"    {k}")

# Save remapped checkpoint
remapped_path = CKPT_PATH.replace(".pt", "_remapped.pt")
remapped_ckpt = {
    "epoch": ckpt.get("epoch", 0),
    "model_state_dict": remapped_sd,
    "optimizer_state_dict": ckpt.get("optimizer_state_dict", {}),
}
torch.save(remapped_ckpt, remapped_path)
print(f"\n  Remapped checkpoint saved → {remapped_path}")
print(f"  Epoch: {remapped_ckpt['epoch']}")
print(f"  Keys:  {len(remapped_sd)}")

# ── 8. Quick inference check ────────────────────────────────────────────
print(f"\n{'='*70}")
print("INFERENCE CHECK")
print(f"{'='*70}")

head.eval()
dummy = torch.randn(1, 3, 8, 256, 256, device=DEVICE)
with torch.no_grad():
    out = head(dummy)
out_tensor = out[0] if isinstance(out, tuple) else out
print(f"  Input:  {tuple(dummy.shape)}")
print(f"  Output: {tuple(out_tensor.shape)}")
print(f"  Values: [{out_tensor.min().item():.4f}, {out_tensor.max().item():.4f}]")
print(f"  ✓ Forward pass succeeded — model loaded correctly")
print(f"{'='*70}")