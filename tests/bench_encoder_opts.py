"""
Encoder latency benchmark — tests fp32, autocast fp16, and fewer frames at fixed 256².
All configs use torch.compile (the default in build_cls_vjepa).

Usage (from WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vad-jepa
    python tests/bench_encoder_opts.py
"""
from __future__ import annotations

import os, sys, time, torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vjepa_encoder import load_pretrained_encoder, VJEPA2Encoder

CKPT = os.path.expanduser("~/vjepa2-checkpoints/vjepa2_1_vitb_dist_vitG_384.pt")
DEV = "cuda"
WARM, MEAS = 20, 100


def med(seq):
    s = sorted(seq)
    return s[len(s) // 2]


def bench_model(model, x, autocast_dtype=None, warm=WARM, meas=MEAS):
    """Median latency in ms."""
    for _ in range(warm):
        with torch.no_grad():
            if autocast_dtype:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    model(x)
            else:
                model(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(meas):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if autocast_dtype:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    model(x)
            else:
                model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return med(times)


def run_one(frames, res, autocast_dtype, label):
    raw = load_pretrained_encoder("vit_base", res, frames, 2, 16, CKPT, device=DEV)
    enc = VJEPA2Encoder(raw)
    enc = enc.to(device=DEV)
    enc.encoder = torch.compile(enc.encoder, mode="reduce-overhead")
    enc.eval()
    x = torch.randn(1, 3, frames, res, res, device=DEV, dtype=torch.float32)
    ms = bench_model(enc, x, autocast_dtype=autocast_dtype)
    fps = 1000 / ms
    return label, ms, fps


if __name__ == "__main__":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Checkpoint: {CKPT}")
    print(f"Warmup={WARM}  Measure={MEAS}  Batch=1")
    print("All configs: torch.compile(reduce-overhead) on encoder\n")

    results = []

    configs = [
        # (frames, res, autocast, label)
        (16, 256, None,           "fp32 16f@256  (baseline)"),
        (16, 256, torch.float16,  "amp  16f@256            "),
        (8,  256, None,           "fp32  8f@256            "),
        (8,  256, torch.float16,  "amp   8f@256            "),
        (4,  256, None,           "fp32  4f@256            "),
        (4,  256, torch.float16,  "amp   4f@256            "),
    ]

    has_ckpt = os.path.exists(CKPT)
    if not has_ckpt:
        print("WARNING: checkpoint not found, using random init. "
              "Latency will be correct but features are random.\n")

    for frames, res, ac_dtype, label in configs:
        if not has_ckpt:
            raw = load_pretrained_encoder("vit_base", res, frames, 2, 16, None, device=DEV)
            enc = VJEPA2Encoder(raw)
            enc.encoder = torch.compile(enc.encoder, mode="reduce-overhead")
            enc.eval()
            x = torch.randn(1, 3, frames, res, res, device=DEV, dtype=torch.float32)
            ms = bench_model(enc, x, ac_dtype)
        else:
            _, ms, _ = run_one(frames, res, ac_dtype, label)

        fps = 1000 / ms
        ok = "OK" if fps >= 10 else "SLOW"
        results.append((label.strip(), ms, fps))
        print("%-32s %7.1fms  %6.1f FPS  [%s]" % (label.strip(), ms, fps, ok))

    # Summary
    if results:
        best = min(r[1] for r in results)
        print(f"\n{'='*60}")
        print("Knob effects (relative to fp32 16f@256 compiled baseline):")
        print(f"{'='*60}")
        baseline_ms = next(r[1] for r in results if "baseline" in r[0])
        for label, ms, fps in sorted(results, key=lambda r: r[1]):
            speedup = baseline_ms / ms
            print("%-32s %7.1fms  %6.1f FPS  %5.2fx" % (label, ms, fps, speedup))
