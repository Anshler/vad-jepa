"""
Quick test to see if torch.compile fixes the bf16 regression on L4.
Run on the L4 where bf16 is slower:
    python tests/diag_compile_test.py --checkpoint /path/to/checkpoint.pt
"""
from __future__ import annotations

import importlib
import sys
import time
from contextlib import nullcontext
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml
from easydict import EasyDict
from model import build_cls_vjepa

DEVICE = torch.device("cuda")

cfg = EasyDict(yaml.safe_load((_REPO_ROOT / "cfgs" / "vjepa_v1.yaml").open()))
cfg.device = DEVICE
cfg.model_name = "vit_base"

# Build without compile
cfg.compile = False
model_no_compile = build_cls_vjepa(cfg)
model_no_compile.eval()

# Build with compile
cfg.compile = True
model_compile = build_cls_vjepa(cfg)
model_compile.eval()

x = torch.randn(1, 3, 4, 384, 384, device=DEVICE)

for label, model in [("no compile", model_no_compile), ("compile", model_compile)]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for amp, amp_label in [(None, "fp32"), (torch.bfloat16, "bf16")]:
        ctx = torch.amp.autocast("cuda", dtype=amp) if amp else nullcontext()
        state = None
        for _ in range(30):
            with torch.no_grad(), ctx:
                _, state = model(x, state)
        state = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            with torch.no_grad(), ctx:
                _, state = model(x, state)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 100 * 1000
        print(f"  {amp_label:>5s}: {ms:>6.2f}ms  ({1000/ms:>5.1f} FPS)")