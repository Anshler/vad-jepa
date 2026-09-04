"""
Diagnostic: pure matmul microbenchmark at ViT-B sizes.
Run from WSL with your conda env active.

Reveals whether the bf16 vs fp32 difference is in the matmuls themselves
or in the full-model overhead (layernorms, casts, dispatches).
"""
from __future__ import annotations

import subprocess
import time

import torch

sizes = [
    ("QKV   [1152,768]×[768,2304]", 1152, 768, 2304),
    ("Score [1152,1152]", 1152, 1152, 1152),
    ("Attn  [1152,64]×[64,1152]", 1152, 64, 1152),
    ("FFN1  [1152,768]×[768,3072]", 1152, 768, 3072),
    ("FFN2  [1152,3072]×[3072,768]", 1152, 3072, 768),
]
N_REPEAT = 500
WARMUP = 20

for dtype, dtype_label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
    print(f"\n{'='*60}")
    print(f"  {dtype_label}")
    print(f"{'='*60}")
    print(f"  {'op':30s} {'ms':>8s}  {'TFLOPS':>8s}  {'vs fp32':>8s}")
    print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*8}")
    fp32_times = []
    for i, (name, M, K, N) in enumerate(sizes):
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        if dtype == torch.bfloat16:
            a = a.bfloat16()
            b = b.bfloat16()
        for _ in range(WARMUP):
            c = a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_REPEAT):
            c = a @ b
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / N_REPEAT * 1000
        tflops = 2 * M * K * N / ms / 1e9
        if dtype == torch.float32:
            fp32_ms = ms
        else:
            ratio = fp32_ms / ms
        print(f"  {name:30s} {ms:>7.3f}  {tflops:>7.1f}"
              + (f"  {ratio:>7.2f}×" if dtype == torch.bfloat16 else "")
        )
        if dtype == torch.float32:
            fp32_times.append(ms)

# nvidia-smi info
print()
try:
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.bandwidth,driver_version,power.limit",
            "--format=csv,noheader",
        ]
    )
except FileNotFoundError:
    pass
print(f"torch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
print(f"TF32 cuDNN:  {torch.backends.cudnn.allow_tf32}")
print(f"float32 matmul precision: {torch.get_float32_matmul_precision()}")