"""
Verify Swin checkpoint loads deterministically — no randomly-initialized weights.

Strategy: build the model twice with different random seeds, load the same
checkpoint into both. If all weights come from the checkpoint, both instances
will be identical bit-for-bit. Any randomly-initialized parameter will differ.

Usage (from WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad
    python tests/verify_swin_checkpoint.py
"""

from __future__ import annotations

import os
import sys

import torch
import yaml
from easydict import EasyDict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import build_cls_vjepa

CFG_DIR = os.path.join(_REPO_ROOT, "cfgs")
CHECKPOINT = "pretrained/swin_base_patch244_window1677_sthv2.pth"


def load_and_get_encoder_state(seed: int):
    """Build model with given seed, load checkpoint, return encoder state_dict."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    cfg_path = os.path.join(CFG_DIR, "swin_lstm.yaml")
    with open(cfg_path, "r") as fh:
        cfg = EasyDict(yaml.safe_load(fh))
    cfg.device = "cpu"

    model = build_cls_vjepa(cfg)
    model.eval()
    return model.encoder.state_dict()


def main():
    print(f"Checkpoint: {CHECKPOINT}")
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: checkpoint not found at '{CHECKPOINT}'")
        sys.exit(1)

    print("Building model with seed=42 ...")
    state_a = load_and_get_encoder_state(42)

    print("Building model with seed=123 ...")
    state_b = load_and_get_encoder_state(123)

    # Compare every key
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    if only_a:
        print(f"\n⚠  Keys only in seed=42: {sorted(only_a)}")
    if only_b:
        print(f"\n⚠  Keys only in seed=123: {sorted(only_b)}")

    common = sorted(keys_a & keys_b)
    mismatched = []
    matched = 0

    for k in common:
        t_a = state_a[k]
        t_b = state_b[k]
        if not torch.equal(t_a, t_b):
            max_diff = (t_a.float() - t_b.float()).abs().max().item()
            mismatched.append((k, tuple(t_a.shape), max_diff))
        else:
            matched += 1

    total = len(common)

    print(f"\n{'='*70}")
    if mismatched:
        print(f"❌  MISMATCH: {len(mismatched)}/{total} parameters differ — "
              f"some weights are RANDOMLY INITIALIZED")
        print(f"\nRandomly-initialized parameters:")
        for name, shape, diff in mismatched:
            print(f"    {name:<60s}  shape={str(shape):<20s}  max_diff={diff:.6e}")
    else:
        print(f"✓  ALL {matched}/{total} parameters IDENTICAL across seeds")
        print(f"   The checkpoint fully determines every weight — nothing is random.")

    # Also verify: keys in state_dict vs keys in checkpoint
    print(f"\n{'='*70}")
    print("Cross-check: checkpoint keys vs model state_dict ...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    ckpt_sd = ckpt.get("state_dict", ckpt)
    # Strip "backbone." prefix if present
    ckpt_keys = set()
    for k in ckpt_sd:
        if k.startswith("backbone."):
            ckpt_keys.add(k[9:])
        else:
            ckpt_keys.add(k)

    model_keys = set(state_a.keys())
    # SwinEncoder wraps SwinTransformer3D as self.swin → model keys are "swin.layers..."
    # Checkpoint keys are "layers..." (no prefix). Strip "swin." for comparison.
    swin_model_keys_raw = {k for k in model_keys if k.startswith("swin.")}
    swin_model_keys = {k[5:] for k in swin_model_keys_raw}  # strip "swin." prefix

    in_ckpt_not_model = ckpt_keys - swin_model_keys
    in_model_not_ckpt = swin_model_keys - ckpt_keys

    if in_ckpt_not_model:
        print(f"  In checkpoint but NOT in model.swin: {len(in_ckpt_not_model)} keys")
        for k in sorted(in_ckpt_not_model)[:10]:
            print(f"    {k}")
        if len(in_ckpt_not_model) > 10:
            print(f"    ... and {len(in_ckpt_not_model) - 10} more")

    if in_model_not_ckpt:
        print(f"\n  ⚠  In model.swin but NOT in checkpoint: {len(in_model_not_ckpt)} keys")
        for k in sorted(in_model_not_ckpt):
            print(f"    {k}  — RANDOMLY INITIALIZED")
    else:
        print(f"  ✓  All {len(swin_model_keys)} model.swin keys are covered by the checkpoint")

    # Summary
    print(f"\n{'='*70}")
    if mismatched or in_model_not_ckpt:
        print("VERDICT: ❌  Some encoder weights are randomly initialized")
    else:
        print("VERDICT: ✓  Encoder is fully deterministic from checkpoint — zero random init")


if __name__ == "__main__":
    main()