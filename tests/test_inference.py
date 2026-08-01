"""
Inference smoke tests — loads actual YAML configs via build_cls_vjepa,
using the pretrained V-JEPA 2.1 ViT-B checkpoint when available.

Usage (from WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad
    python tests/test_inference.py
    python tests/test_inference.py --amp fp16
    python tests/test_inference.py --amp bf16
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext

import torch
import yaml
from easydict import EasyDict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import _HAS_MAMBA_SSM, build_cls_vjepa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG_DIR = os.path.join(_REPO_ROOT, "cfgs")

# Pretrained V-JEPA 2.1 ViT-B checkpoint — strict-loads cleanly.
_VJEPA2_CKPT = os.path.expanduser("~/vjepa2-checkpoints/vjepa2_1_vitb_dist_vitG_384.pt")
_HAS_CKPT = os.path.exists(_VJEPA2_CKPT)

_AMP_CHOICES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}

parser = argparse.ArgumentParser()
parser.add_argument("--amp", default="fp32", choices=list(_AMP_CHOICES),
                    help="AMP dtype (default: fp32)")
amp_dtype = _AMP_CHOICES[parser.parse_args().amp]

print(f"Device: {DEVICE}")
print(f"mamba_ssm: {_HAS_MAMBA_SSM}")
print(f"checkpoint: {'found' if _HAS_CKPT else 'not found'} ({_VJEPA2_CKPT})")
print(f"amp: {parser.parse_args().amp}")


def load_cfg(name):
    path = os.path.join(CFG_DIR, name)
    with open(path, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.device = DEVICE
    # Wire in the actual checkpoint as ViT-B (matching the pretrained weights).
    if _HAS_CKPT:
        cfg.model_name = "vit_base"
        cfg.checkpoint_path = _VJEPA2_CKPT
    return cfg


def _autocast():
    return torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()


def run_streaming_test(cfg, tag, num_steps=4):
    """Build from config, run stride-1 streaming, verify state carry-forward."""
    print(f"\n  [{tag}] loading {cfg.temporal_model} from config …")
    model = build_cls_vjepa(cfg)
    model.eval()

    shape = (2, 3, cfg.num_frames, cfg.img_size, cfg.img_size)
    state = None
    for step in range(num_steps):
        x = torch.randn(*shape).to(DEVICE)
        with torch.no_grad(), _autocast():
            logits, state = model(x, state)
        prob = logits.softmax(dim=1)[0, 1].item()
        print(f"    step {step+1}: out={tuple(logits.shape)}  anomaly_prob={prob:.4f}")

    # Verify state carries forward
    if cfg.temporal_model == "lstm":
        assert isinstance(state, tuple) and len(state) == 2
        print(f"    state: LSTM  h={tuple(state[0].shape)} c={tuple(state[1].shape)}")
    else:
        assert state is not None
        print(f"    state: MambaCache  seqlen_offset={state.seqlen_offset}")

    # Entropy for sparse models
    ent = getattr(getattr(model, "temporal", None), "_entropy", None)
    if ent is not None and hasattr(ent, "item") and ent.item() > 0:
        print(f"    gate entropy: {ent.item():.2f}")

    print(f"    ✓ {tag} OK")
    return model


# ===========================================================================
print("=" * 70)
print("V-JEPA 2.1 + MOVAD — config-based smoke tests")
print("=" * 70)

# --- LSTM (always available) ---
print("\n[1/4] LSTM baseline")
cfg_lstm = load_cfg("vjepa_v1.yaml")
run_streaming_test(cfg_lstm, "LSTM")

# --- Mamba ---
print("\n[2/4] Mamba")
if _HAS_MAMBA_SSM:
    cfg_mamba = load_cfg("vjepa_mamba.yaml")
    run_streaming_test(cfg_mamba, "Mamba")
else:
    print("  (skipped — mamba_ssm not installed)")

# --- SlotSSM ---
print("\n[3/4] SlotSSM")
if _HAS_MAMBA_SSM:
    cfg_slotssm = load_cfg("vjepa_slotssm.yaml")
    run_streaming_test(cfg_slotssm, "SlotSSM")
else:
    print("  (skipped — mamba_ssm not installed)")

# --- Sparse SlotSSM ---
print("\n[4/4] Sparse SlotSSM")
if _HAS_MAMBA_SSM:
    cfg_sparse = load_cfg("vjepa_sparse_slotssm.yaml")
    run_streaming_test(cfg_sparse, "SparseSlotSSM")
else:
    print("  (skipped — mamba_ssm not installed)")

# ===========================================================================
# Resolution / frame-count flexibility (LSTM)
# ===========================================================================
print("\n" + "=" * 70)
print("Resolution + frame-count flexibility")
print("=" * 70)

model_lstm = build_cls_vjepa(cfg_lstm)
model_lstm.eval()
for frames, res, tag in [(16, 256, "16f@256"), (32, 256, "32f@256"), (16, 384, "16f@384")]:
    x = torch.randn(2, 3, frames, res, res).to(DEVICE)
    with torch.no_grad(), _autocast():
        out, _ = model_lstm(x)
    print(f"  {tag}: {tuple(x.shape)} -> {tuple(out.shape)}  ✓")

# ===========================================================================
# Multi-head smoke test — one encoder, multiple temporal models
# ===========================================================================
print("\n" + "=" * 70)
print("Multi-head — shared encoder, multiple temporal models")
print("=" * 70)

if _HAS_MAMBA_SSM and _HAS_CKPT:
    from model import build_multi_head_vjepa

    # Build a multi-head cfg combining 6 existing configs
    head_files = [
        "vjepa_v1.yaml",          # LSTM
        "vjepa_mamba.yaml",       # Mamba
        "vjepa_slotssm.yaml",    # SlotSSM
        "vjepa_sparse_slotssm.yaml",  # Sparse SlotSSM
    ]
    head_names = [os.path.splitext(f)[0] for f in head_files]

    all_cfgs = []
    for fpath in head_files:
        with open(os.path.join(CFG_DIR, fpath), "r") as f:
            all_cfgs.append(EasyDict(yaml.safe_load(f)))

    # Master = first config
    cfg_multi = all_cfgs[0]
    cfg_multi.device = DEVICE
    if _HAS_CKPT:
        cfg_multi.checkpoint_path = _VJEPA2_CKPT

    cfg_multi._head_cfgs_flat = [
        {**dict(hc), "name": hn} for hc, hn in zip(all_cfgs, head_names)
    ]

    model_multi = build_multi_head_vjepa(cfg_multi)
    model_multi.eval()

    print(f"\n  Heads: {list(model_multi.heads.keys())}")

    # Test: encode once → each head processes independently (streaming)
    shape = (2, 3, cfg_multi.num_frames, cfg_multi.img_size, cfg_multi.img_size)
    x = torch.randn(*shape).to(DEVICE)
    for name, head in model_multi.heads.items():
        state = None
        for step in range(3):
            with torch.no_grad(), _autocast():
                logits, state = head(x, state)
            prob = logits.softmax(dim=1)[0, 1].item()
            if step == 0:
                print(f"  [{name}] step 1: anomaly_prob={prob:.4f}")
        print(f"    ✓ {name} streaming OK (3 steps)")

    # Test: per-clip encoding across multiple clip indices
    video = torch.randn(2, 3, 16, cfg_multi.img_size, cfg_multi.img_size).to(DEVICE)
    with torch.no_grad(), _autocast():
        clip = video[:, :, :4, :, :]   # first 4-frame clip
        out, _ = model_multi.heads[next(iter(model_multi.heads.keys()))](clip, None)
    print(f"\n  per-clip encoding: clip {tuple(clip.shape)} → output {tuple(out.shape)}")
    print(f"    ✓ per-clip encode OK")

    print("\n  All heads pass streaming smoke tests ✓")
else:
    if not _HAS_MAMBA_SSM:
        print("  (skipped — mamba_ssm not installed)")
    if not _HAS_CKPT:
        print("  (skipped — checkpoint not found)")

print("\n" + "=" * 70)
print("All config-based smoke tests passed!")
print("=" * 70)
