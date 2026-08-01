"""
Component-level latency benchmark.

Measures latency broken down by model component for each temporal variant:
  ┌─────────────────────────────────────────────────────┐
  │  Encoder   │  Spatial Conv  │  Temporal   │  Cls   │
  │  (ViT)     │  (depthwise)   │  (LSTM/…)   │  (MLP) │
  └─────────────────────────────────────────────────────┘

Uses torch.cuda.Event for precise GPU-side timing, and takes care to
handle the different forward path structures:
  - Standard:  encoder → bn → lin1 → temporal → lin2 → lin3
  - SlotSSM:   encoder(patch) → spatial_pool → SlotSSM → slot_attn → MLP
  - Swin:      encoder(patch) → flatten → bn → lin1 → temporal → …

Usage
-----
  conda activate vjepa2-312
  cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vad-jepa

  # All configs (requires checkpoint + mamba_ssm where needed)
  python tests/bench_component_latency.py

  # Only standard temporal models (no mamba_ssm dependency)
  python tests/bench_component_latency.py --standard

  # Single model
  python tests/bench_component_latency.py --model vjepa_v1.yaml

  # Spatial grid variant (V-JEPA spatial pooling)
  python tests/bench_component_latency.py --model vjepa_v1.yaml --spatial-grid 6 6

  # Override AMP / frames / resolution
  python tests/bench_component_latency.py --amp fp16 --frames 8 --resolution 256

  # Plot results
  python tests/bench_component_latency.py --plot component_latency.png
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from easydict import EasyDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch.nn as nn
import torch.nn.functional as F

from model import (
    _HAS_MAMBA_SSM,
    ClsVJEPA,
    build_cls_vjepa,
    VJEPA2Encoder,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG_DIR = _REPO_ROOT / "cfgs"

WARMUP = 30
MEASURE = 200
DEFAULT_FRAMES = 4
DEFAULT_RES = 384
BATCH = 1

_AMP_CHOICES = {
    "fp32": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_DEFAULT_AMP = os.environ.get("AMP_DTYPE", "fp32")

_HEADER_FMT = (
    "{model:<18} {enc:>8} {spat:>8} {proj:>8} {temp:>8} {cls:>8}  {total:>8}  {fps:>6}"
)
_ROW_FMT = (
    "{model:<18} {enc:>7.1f} {spat:>7.1f} {proj:>7.1f} {temp:>7.1f} {cls:>7.1f}  {total:>7.1f}  {fps:>5.1f}"
)


def load_cfg(name: str) -> EasyDict:
    path = CFG_DIR / name
    with open(str(path), "r") as fh:
        cfg = EasyDict(yaml.safe_load(fh))
    cfg.device = DEVICE
    return cfg


def make_event():
    return torch.cuda.Event(enable_timing=True)


# ===========================================================================
# Component-level forward runner
# ===========================================================================
class ComponentTimer:

    def __init__(self, model: ClsVJEPA, amp_dtype=None):
        self.model = model
        self.amp_dtype = amp_dtype
        self.ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
        self._is_slot = model._slot_based
        self._is_swin = model._is_swin
        self._has_spatial_grid = model._use_spatial_grid
        self._state = None

    def _record_call(self, fn, ev_start, ev_end, *args, **kwargs):
        """Call ``fn(*args, **kwargs)`` bracketed by two CUDA events."""
        ev_start.record()
        with torch.no_grad(), self.ctx:
            out = fn(*args, **kwargs)
        ev_end.record()
        return out

    def __call__(self, x: torch.Tensor):
        """Run full forward, return ``{name: ms, ...}`` dict.

        CUDA events are recorded per-component without intermediate syncs.
        A single ``torch.cuda.synchronize()`` at the end reads all times.
        """
        times = {}
        m = self.model
        ev = {}  # name -> (start_event, end_event)

        torch.cuda.synchronize()

        # ---------------------------------------------------------------
        # 1. Encoder
        # ---------------------------------------------------------------
        s, e = make_event(), make_event()
        ev["encoder"] = (s, e)
        if self._is_slot or self._is_swin or self._has_spatial_grid:
            z = self._record_call(m.encoder, s, e, x, return_patches=True)
        else:
            z = self._record_call(m.encoder, s, e, x)

        # ---------------------------------------------------------------
        # 2. Spatial Conv (V-JEPA spatial grid only)
        # ---------------------------------------------------------------
        if self._has_spatial_grid and not self._is_swin:
            s, e = make_event(), make_event()
            ev["spatial_conv"] = (s, e)
            patch_tokens = z.clone() if self._is_slot else z
            with torch.no_grad(), self.ctx:
                s.record()
                pooled = m._spatial_pool_tokens(patch_tokens, m._vjepa_n_temp)
                e.record()
            if self._is_slot:
                z = pooled
            else:
                z = pooled.reshape(x.shape[0], -1)
        else:
            ev["spatial_conv"] = None

        # ---------------------------------------------------------------
        # 3. Projection (bn → lin1 → drop) — non-slot only
        # ---------------------------------------------------------------
        if not self._is_slot:
            s, e = make_event(), make_event()
            ev["projection"] = (s, e)
            with torch.no_grad(), self.ctx:
                s.record()
                if self._is_swin:
                    h = z.transpose(1, 2).flatten(1)
                else:
                    h = z
                h = m.bn(h)
                h = F.relu(m.lin1(h))
                h = m.drop(h)
                e.record()
        else:
            ev["projection"] = None

        # ---------------------------------------------------------------
        # 4. Temporal model
        # ---------------------------------------------------------------
        s, e = make_event(), make_event()
        ev["temporal"] = (s, e)
        inp = z if self._is_slot else h
        with torch.no_grad(), self.ctx:
            s.record()
            slots_or_feats, self._state = m.temporal(inp, self._state)
            e.record()

        # ---------------------------------------------------------------
        # 5. Classifier (slot-attn + MLP or lin2 + lin3)
        # ---------------------------------------------------------------
        s, e = make_event(), make_event()
        ev["classifier"] = (s, e)
        if self._is_slot:
            with torch.no_grad(), self.ctx:
                s.record()
                D = slots_or_feats.shape[-1]
                scores = (slots_or_feats * m.slot_query).sum(dim=-1) / (D ** 0.5)
                attn = scores.softmax(dim=-1)
                pooled = (attn.unsqueeze(-1) * slots_or_feats).sum(dim=1)
                logits = m.classifier(pooled)
                e.record()
        else:
            with torch.no_grad(), self.ctx:
                s.record()
                logits = F.relu(m.lin2(slots_or_feats))
                logits = m.drop(logits)
                logits = m.lin3(logits)
                e.record()

        # ---------------------------------------------------------------
        # Single sync → read all times
        # ---------------------------------------------------------------
        torch.cuda.synchronize()

        for name, se in ev.items():
            if se is None:
                times[name] = 0.0
            else:
                times[name] = se[0].elapsed_time(se[1])

        times["total"] = sum(v for v in times.values())
        return times


# ===========================================================================
def measure_component_latency(
    model: ClsVJEPA,
    x: torch.Tensor,
    amp_dtype,
    warmup: int = WARMUP,
    measure: int = MEASURE,
) -> dict[str, float]:
    """Return median-per-component latencies in ms."""
    timer = ComponentTimer(model, amp_dtype=amp_dtype)
    accum: dict[str, list[float]] = {}

    for step in range(warmup + measure):
        times = timer(x)
        if step >= warmup:
            for k, v in times.items():
                accum.setdefault(k, []).append(v)

    median = {}
    for k, vals in accum.items():
        s = sorted(vals)
        median[k] = s[len(s) // 2]
    return median


# ===========================================================================
def _fmt(label: str, median: dict, show_spatial: bool) -> str:
    enc = median.get("encoder", 0)
    spat = median.get("spatial_conv", 0)
    proj = median.get("projection", 0)
    temp = median.get("temporal", 0)
    cls = median.get("classifier", 0)
    total = median.get("total", 0)
    fps = 1000 / total if total > 0 else 0
    return _ROW_FMT.format(
        model=label, enc=enc, spat=spat, proj=proj,
        temp=temp, cls=cls, total=total, fps=fps,
    )


# ===========================================================================
def plot_results(results: list[tuple], output_path: str):
    """Generate a stacked-bar plot of component latencies."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return

    labels = [r[0] for r in results]
    enc = [r[1].get("encoder", 0) for r in results]
    spat = [r[1].get("spatial_conv", 0) for r in results]
    proj = [r[1].get("projection", 0) for r in results]
    temp = [r[1].get("temporal", 0) for r in results]
    cls = [r[1].get("classifier", 0) for r in results]

    components = [
        ("Encoder", enc, "#4C72B0"),
        ("Spatial Conv", spat, "#DD8452"),
        ("Projection", proj, "#55A868"),
        ("Temporal", temp, "#C44E52"),
        ("Classifier", cls, "#8172B3"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for name, vals, color in components:
        bars = ax.bar(x, vals, bottom=bottom, label=name, color=color, width=0.55)
        # Annotate non-zero values inside bars
        for xi, (b, v) in enumerate(zip(bars, vals)):
            if v > 1.0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + v / 2,
                        f"{v:.0f}ms", ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Component-level latency  |  {BATCH}×{NUM_FRAMES}@{RES}  |  "
                 f"AMP {args.amp}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # FPS line on secondary axis
    ax2 = ax.twinx()
    fps_vals = [1000 / r[1]["total"] for r in results]
    ax2.plot(x, fps_vals, "o-", color="black", linewidth=1.5, markersize=5, label="FPS")
    ax2.set_ylabel("FPS", color="black")
    for xi, f in enumerate(fps_vals):
        ax2.annotate(f"{f:.0f}", (xi, f), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved: {output_path}")


# ===========================================================================
parser = argparse.ArgumentParser(
    description="Component-level latency benchmark for temporal variants"
)
parser.add_argument("--amp", default=_DEFAULT_AMP, choices=list(_AMP_CHOICES),
                    help=f"AMP dtype (default: {_DEFAULT_AMP})")
parser.add_argument("--checkpoint", default=None,
                    help="Path to a pretrained V-JEPA checkpoint (.pt)")
parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                    help=f"Number of input frames (default: {DEFAULT_FRAMES})")
parser.add_argument("--resolution", type=int, default=DEFAULT_RES,
                    help=f"Spatial resolution (default: {DEFAULT_RES})")
parser.add_argument("--batch", type=int, default=BATCH,
                    help=f"Batch size (default: {BATCH})")
parser.add_argument("--model", default=None,
                    help="Run a single config file by name")
parser.add_argument("--standard", action="store_true",
                    help="Only run models that don't require mamba_ssm")
parser.add_argument("--plot", default=None,
                    help="Output path for stacked-bar plot (requires matplotlib)")
parser.add_argument("--spatial-grid", type=int, nargs=2, default=None,
                    metavar=("H", "W"),
                    help="Enable V-JEPA spatial grid (e.g. --spatial-grid 6 6)")
args = parser.parse_args()

amp_dtype = _AMP_CHOICES[args.amp]
B = args.batch
NUM_FRAMES = args.frames
RES = args.resolution
WARMUP = 30 if NUM_FRAMES <= 16 else 10
MEASURE = 200 if NUM_FRAMES <= 16 else 80

gpu_name = torch.cuda.get_device_name(0)
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
amp_tag = f"AMP {args.amp}" if amp_dtype else "fp32"

print(f"GPU: {gpu_name}  ({mem_gb:.1f} GB)")
print(f"Config: WARMUP={WARMUP}  MEASURE={MEASURE}  Batch={B}  "
      f"{B}×{NUM_FRAMES}@{RES}")
print(f"Dtype: {amp_tag}")
if args.spatial_grid:
    print(f"Spatial grid: {args.spatial_grid[0]}×{args.spatial_grid[1]} "
          f"(V-JEPA depthwise Conv2d)")
print()

# Gather model configs
if args.model is not None:
    CONFIGS = [(args.model, Path(args.model).stem)]
else:
    CONFIGS = [
        ("vjepa_v1.yaml",                "LSTM"),
        ("vjepa_linear_probe.yaml",      "None"),
    ]
    if not args.standard:
        CONFIGS += [
            ("vjepa_mamba.yaml",             "Mamba"),
            ("vjepa_slotssm.yaml",           "SlotSSM"),
            ("vjepa_sparse_slotssm.yaml",    "SpSlotSSM"),
        ]

results: list[tuple[str, dict]] = []

# Print column header
print(" " * 18 + "──Encoder── ─Spatial── ─Proj─── ─Temporal ──Cls───  ──Total──  ──FPS──")
print(_HEADER_FMT.format(
    model="Model", enc="Enc(ms)", spat="Spat(ms)", proj="Proj(ms)",
    temp="Temp(ms)", cls="Cls(ms)", total="Total(ms)", fps="FPS",
))
print("-" * 90)

for fname, tag in CONFIGS:
    if not _HAS_MAMBA_SSM and "mamba" in fname:
        continue
    if not _HAS_MAMBA_SSM and fname not in ("vjepa_v1.yaml", "vjepa_linear_probe.yaml"):
        continue

    cfg = load_cfg(fname)
    cfg.model_name = "vit_base"

    if args.checkpoint is not None:
        cfg.checkpoint_path = args.checkpoint
    if args.spatial_grid is not None:
        cfg.vjepa_spatial_grid = list(args.spatial_grid)

    model = build_cls_vjepa(cfg)
    model.eval()

    x = torch.randn(B, 3, NUM_FRAMES, RES, RES, device=DEVICE)
    median = measure_component_latency(model, x, amp_dtype)

    results.append((tag, median))
    print(_fmt(tag, median, args.spatial_grid is not None))

    # Clean up to avoid OOM across configs
    del model
    torch.cuda.empty_cache()

# Print relative breakdown
if len(results) > 1:
    print("\n" + "=" * 90)
    print("Relative breakdown (% of total):")
    print("=" * 90)
    print(" " * 18 + "  Encoder   Spatial   Proj    Temporal   Cls")
    for label, med in results:
        t = med["total"]
        if t > 0:
            print(f"  {label:<18}  "
                  f"{med['encoder']/t*100:>5.1f}%   "
                  f"{med['spatial_conv']/t*100:>5.1f}%   "
                  f"{med['projection']/t*100:>5.1f}%   "
                  f"{med['temporal']/t*100:>5.1f}%   "
                  f"{med['classifier']/t*100:>5.1f}%")

if args.plot:
    plot_results(results, args.plot)