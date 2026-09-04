"""
Diagnostic: time the encoder alone vs the full model, with and without autocast.
Helps isolate whether the overhead is in the encoder backbone or elsewhere.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from easydict import EasyDict
from model import build_cls_vjepa

DEVICE = torch.device("cuda")
CFG_DIR = _REPO_ROOT / "cfgs"

B, F, H, W = 1, 4, 384, 384
WARMUP = 30
MEASURE = 200

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None,
                    help="Path to a pretrained V-JEPA checkpoint (.pt)")
args = parser.parse_args()

cfg = EasyDict({k: v for k, v in __import__("yaml").safe_load(
    (CFG_DIR / "vjepa_v1.yaml").open()).items()})
cfg.device = DEVICE
cfg.model_name = "vit_base"
cfg.compile = False
if args.checkpoint is not None:
    cfg.checkpoint_path = args.checkpoint

model = build_cls_vjepa(cfg)
model.eval()

x = torch.randn(B, 3, F, H, W, device=DEVICE)

for amp_dtype, label in [(None, "fp32"), (torch.bfloat16, "bf16")]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()

    # ── Time the encoder alone ──
    state = None
    for _ in range(WARMUP):
        with torch.no_grad(), ctx:
            z = model.encoder(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE):
        with torch.no_grad(), ctx:
            z = model.encoder(x)
    torch.cuda.synchronize()
    enc_ms = (time.perf_counter() - t0) / MEASURE * 1000
    print(f"  Encoder alone:     {enc_ms:>7.2f}ms  ({1000/enc_ms:>6.1f} FPS)")

    # ── Time the full model ──
    state = None
    for _ in range(WARMUP):
        with torch.no_grad(), ctx:
            out, state = model(x, state)
    state = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE):
        with torch.no_grad(), ctx:
            out, state = model(x, state)
    torch.cuda.synchronize()
    full_ms = (time.perf_counter() - t0) / MEASURE * 1000
    print(f"  Full model:        {full_ms:>7.2f}ms  ({1000/full_ms:>6.1f} FPS)")
    print(f"  Non-encoder:       {full_ms - enc_ms:>7.2f}ms")
    print(f"  Encoder fraction:  {enc_ms/full_ms*100:>5.1f}%")

print(f"\ntorch: {torch.__version__}, CUDA: {torch.version.cuda}")