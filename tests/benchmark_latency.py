"""
Latency benchmark — ViT-B compiled (default) with temporal model comparison.
Uses AMP fp16 by default (configurable via --amp flag or AMP_DTYPE env var).

Usage (from WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vad-jepa
    python tests/benchmark_latency.py
    python tests/benchmark_latency.py --amp fp32
    python tests/benchmark_latency.py --checkpoint ~/vjepa2-checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext

import torch
import yaml
from easydict import EasyDict

# Ensure the repo root is on sys.path for sibling imports
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import _HAS_MAMBA_SSM, build_cls_vjepa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG_DIR = os.path.join(_REPO_ROOT, "cfgs")

WARMUP = 30
MEASURE = 200
B, F, H, W = 1, 4, 384, 384
TARGET_FPS = 10.0

_AMP_CHOICES = {
    "fp32": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_DEFAULT_AMP = os.environ.get("AMP_DTYPE", "fp32")

CONFIGS = [
    ("vjepa_v1.yaml",                "LSTM"),
    ("vjepa_linear_probe.yaml",      "None"),
    ("vjepa_mamba.yaml",             "Mamba"),
    ("vjepa_slotssm.yaml",           "SlotSSM"),
    ("vjepa_sparse_slotssm.yaml",    "SpSlotSSM"),
]


def load_cfg(name):
    path = os.path.join(CFG_DIR, name)
    with open(path, "r") as fh:
        cfg = EasyDict(yaml.safe_load(fh))
    cfg.device = DEVICE
    return cfg


def _warmup_steps(total):
    return min(30, total // 3)


def measure_latency(model, x, steps, amp_dtype=None):
    warmup = _warmup_steps(steps)
    measure_n = steps - warmup
    state = None
    ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()

    for _ in range(warmup):
        with torch.no_grad(), ctx:
            _, state = model(x, state)

    state = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(measure_n):
        with torch.no_grad(), ctx:
            _, state = model(x, state)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    return dt / measure_n


# ===========================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--amp", default=_DEFAULT_AMP, choices=list(_AMP_CHOICES),
                    help=f"AMP dtype (default: {_DEFAULT_AMP})")
parser.add_argument("--checkpoint", default=None,
                    help="Path to a pretrained V-JEPA checkpoint (.pt)")
args = parser.parse_args()
amp_dtype = _AMP_CHOICES[args.amp]

gpu_name = torch.cuda.get_device_name(0)
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
amp_tag = f"AMP {args.amp}" if amp_dtype else "fp32"
print(f"GPU: {gpu_name}  ({mem_gb:.1f} GB)")
print(f"Target: <= {1000 / TARGET_FPS:.0f} ms/step for {TARGET_FPS} FPS online")
print(f"ViT-B + torch.compile  |  {amp_tag}  |  WARMUP={WARMUP}  MEASURE={MEASURE}  Batch={B}\n")

print(f"{'Model':<22} {'Total':>8}  {'FPS':>7}  {'RT?':>5}")
print(f"{'-'*22} {'-'*8}  {'-'*7}  {'-'*5}")

for name, tag in CONFIGS:
    if not _HAS_MAMBA_SSM and "mamba" in name:
        continue
    if not _HAS_MAMBA_SSM and name not in ("vjepa_v1.yaml", "vjepa_linear_probe.yaml"):
        continue

    cfg = load_cfg(name)
    cfg.model_name = "vit_base"
    if args.checkpoint is not None:
        cfg.checkpoint_path = args.checkpoint
    model = build_cls_vjepa(cfg)
    model.eval()

    x = torch.randn(B, 3, F, H, W, device=DEVICE)
    ms = measure_latency(model, x, WARMUP + MEASURE, amp_dtype=amp_dtype)
    fps = 1000 / ms
    status = "✓" if fps >= TARGET_FPS else "✗"
    print(f"{tag:<22} {ms:7.1f}ms  {fps:6.1f}   {status}")
