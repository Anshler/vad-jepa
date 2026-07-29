"""
Verify parameter counts — detailed hierarchical breakdown.

Usage (from WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad
    python tests/verify_params.py
    python tests/verify_params.py --cfg cfgs/vjepa_v1.yaml
    python tests/verify_params.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import yaml
from easydict import EasyDict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import (
    ClsVJEPA,
    LSTMTemporalModel,
    MambaTemporalModel,
    NoTemporalModel,
    SlotSSMBlock,
    SlotSSMTemporalModel,
    _HAS_MAMBA_SSM,
    build_cls_vjepa,
)

CFG_DIR = os.path.join(_REPO_ROOT, "cfgs")

ALL_CONFIGS = [
    "vjepa_v1.yaml",
    "vjepa_mamba.yaml",
    "vjepa_slotssm.yaml",
    "vjepa_slotssm_inv.yaml",
    "vjepa_sparse_slotssm.yaml",
    "vjepa_sparse_slotssm_inv.yaml",
    "vjepa_linear_probe.yaml",
    "swin_lstm.yaml",
    "swin_mamba.yaml",
    "swin_mamba3.yaml",
]

# ---------------------------------------------------------------------------
# Parameter counting helpers
# ---------------------------------------------------------------------------


def _fmt(n: int) -> str:
    """Format a param count human-readably."""
    if n >= 1e6:
        return f"{n / 1e6:7.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:7.1f}K"
    else:
        return f"{n:8d}"


def _trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _frozen(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if not p.requires_grad)


def _total(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_param(module: nn.Parameter) -> int:
    return module.numel()


# ---------------------------------------------------------------------------
# Per-component breakdown
# ---------------------------------------------------------------------------


def _count_named_children(mod: nn.Module, prefix: str = "", depth: int = 0) -> list[tuple[str, int, int]]:
    """Recursively count trainable params for every named sub-module and direct parameter.

    Returns list of (name, trainable_count, total_count).
    """
    results: list[tuple[str, int, int]] = []

    # Direct parameters (not inside a child module) — e.g. nn.Parameter attrs
    direct_trainable = 0
    direct_total = 0
    for pname, p in mod.named_parameters(recurse=False):
        n = p.numel()
        direct_total += n
        if p.requires_grad:
            direct_trainable += n

    if direct_total > 0:
        label = f"{prefix}" if prefix else "(direct params)"
        results.append((label, direct_trainable, direct_total))

    # Named children
    for cname, child in mod.named_children():
        child_prefix = f"{prefix}.{cname}" if prefix else cname
        child_trainable = _trainable(child)
        child_total = _total(child)
        if child_total == 0:
            continue

        # Check if this child has further named children we can drill into
        grandchildren = list(child.named_children())
        if grandchildren:
            # Drill down recursively
            results.extend(_count_named_children(child, child_prefix, depth + 1))
        else:
            results.append((child_prefix, child_trainable, child_total))

    return results


# ---------------------------------------------------------------------------
# ClsVJEPA-aware breakdown
# ---------------------------------------------------------------------------


def breakdown_cls_vjepa(model: ClsVJEPA) -> dict:
    """Return a structured breakdown of a ClsVJEPA model's parameters."""
    info: dict = OrderedDict()

    # ---- Encoder ----
    info["encoder"] = {
        "total": _total(model.encoder),
        "trainable": _trainable(model.encoder),
        "frozen": _frozen(model.encoder),
    }

    # ---- Spatial pool (optional) ----
    sp = getattr(model, "vjepa_spatial_pool", None)
    if sp is not None:
        info["spatial_pool (learned depthwise Conv2d)"] = {
            "total": _total(sp),
            "trainable": _trainable(sp),
        }

    temporal = model.temporal

    # ---- Temporal breakdown ----
    temporal_info: dict = OrderedDict()
    temporal_info["_total"] = _total(temporal)
    temporal_info["_trainable"] = _trainable(temporal)

    if isinstance(temporal, SlotSSMTemporalModel):
        temporal_info["slots_init"] = {"trainable": count_param(temporal.slots_init)}

        for i, blk in enumerate(temporal.blocks):
            blk_info: dict = OrderedDict()
            blk_info["input_proj"] = {"trainable": _trainable(blk.input_proj)}
            blk_info["cross_attn_input_norm"] = {"trainable": _trainable(blk.cross_attn_input_norm)}
            blk_info["cross_attn_ref_norm"] = {"trainable": _trainable(blk.cross_attn_ref_norm)}
            blk_info["cross_attn"] = {"trainable": _trainable(blk.cross_attn)}
            blk_info["time_mixer_norm"] = {"trainable": _trainable(blk.time_mixer_norm)}
            blk_info["mamba"] = {"trainable": _trainable(blk.mamba)}
            blk_info["space_attn_norm"] = {"trainable": _trainable(blk.space_attn_norm)}
            blk_info["self_attn"] = {"trainable": _trainable(blk.self_attn)}

            if blk.top_k is not None:
                blk_info["gate (sparse)"] = {"trainable": _trainable(blk.gate)}

            blk_total = sum(v["trainable"] for v in blk_info.values())
            blk_info["_block_total"] = blk_total
            temporal_info[f"block[{i}]"] = blk_info

    elif isinstance(temporal, MambaTemporalModel):
        for i, (blk, norm) in enumerate(zip(temporal.blocks, temporal.norms)):
            temporal_info[f"block[{i}].mamba"] = {"trainable": _trainable(blk)}
            temporal_info[f"block[{i}].norm"] = {"trainable": _trainable(norm)}

    elif isinstance(temporal, LSTMTemporalModel):
        temporal_info["rnn"] = {"trainable": _trainable(temporal.rnn)}
        temporal_info["norm"] = {"trainable": _trainable(temporal.norm)}

    elif isinstance(temporal, NoTemporalModel):
        temporal_info["(identity)"] = {"trainable": 0}

    info["temporal"] = temporal_info

    # ---- Slot query (SlotSSM only) ----
    sq = getattr(model, "slot_query", None)
    if sq is not None:
        info["slot_query (learned attention-pool)"] = {
            "trainable": count_param(sq),
        }

    # ---- Classifier / projections ----
    if model._slot_based:
        info["classifier"] = {
            "total": _total(model.classifier),
            "trainable": _trainable(model.classifier),
            "children": _count_named_children(model.classifier),
        }
    else:
        # Standard path — separate pre-temporal projection from post-temporal classifier
        pre_info: OrderedDict = OrderedDict()
        pre_info["bn (LayerNorm)"] = {"trainable": _trainable(model.bn)}
        pre_info["lin1 (input → latent)"] = {"trainable": _trainable(model.lin1)}
        pre_info["_pre_total"] = _trainable(model.bn) + _trainable(model.lin1)

        post_info: OrderedDict = OrderedDict()
        post_info["lin2 (temporal_out → latent)"] = {"trainable": _trainable(model.lin2)}
        post_info["lin3 (latent → 2)"] = {"trainable": _trainable(model.lin3)}
        post_info["_post_total"] = _trainable(model.lin2) + _trainable(model.lin3)

        info["pre-temporal projection"] = pre_info
        info["post-temporal classifier"] = post_info

    # ---- Totals ----
    trainable_total = 0
    frozen_total = 0

    for key, val in info.items():
        if isinstance(val, dict):
            if "total" in val and key.startswith("encoder"):
                frozen_total += val.get("frozen", val["total"])
            elif "trainable" in val and "_" not in key:
                trainable_total += val["trainable"]
            elif "_total" in val:
                # Embedded temporal — use _trainable
                trainable_total += val.get("_trainable", 0)

    # Recursively sum trainable from all nested dicts with "trainable" key
    def _sum_trainable(d):
        s = 0
        if isinstance(d, dict):
            if "trainable" in d and not any(k.startswith("_") for k in d.keys() if k != "trainable"):
                pass  # handled via parent
            for k, v in d.items():
                if k == "trainable" and isinstance(v, (int, float)):
                    s += v
                elif isinstance(v, dict):
                    s += _sum_trainable(v)
        return s

    # Don't double-count — we already summed via the top-level keys.  The more
    # reliable approach: take the encoder frozen count + sum of trainable from
    # the full PyTorch model.
    info["_summary"] = {
        "encoder_frozen": _frozen(model.encoder),
        "encoder_total": _total(model.encoder),
        "trainable_from_pytorch": _trainable(model),
        "frozen_from_pytorch": _frozen(model),
        "total_from_pytorch": _total(model),
    }

    return info


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_breakdown(info: dict, indent: int = 0):
    """Pretty-print a nested parameter breakdown."""

    def _print(indent: int, *args):
        print(" " * indent + " ".join(str(a) for a in args))

    IND = 2

    for section, data in info.items():
        if section.startswith("_"):
            continue
        if section == "encoder":
            frozen_n = data.get("frozen", 0)
            label = "encoder (frozen)" if frozen_n > 0 else "encoder (trainable)"
            _print(indent, f"┌─ {label}")
            _print(indent + IND, f"{_fmt(data['total'])} total  ({_fmt(data['frozen'])} frozen, {_fmt(data['trainable'])} trainable)")
            continue

        if section == "spatial_pool (learned depthwise Conv2d)":
            _print(indent, f"├─ {section}")
            _print(indent + IND, f"{_fmt(data['trainable'])} trainable")
            continue

        if section == "temporal":
            _print(indent, f"├─ temporal ({data.get('_total', 0) / 1e6:.2f}M total, {data.get('_trainable', 0) / 1e6:.2f}M trainable)")
            # Print sub-components
            for sub, subdata in data.items():
                if sub.startswith("_"):
                    continue
                if sub == "slots_init":
                    _print(indent + IND, f"├─ {sub}: {_fmt(subdata['trainable'])}")
                elif sub.startswith("block[") and "_block_total" in subdata:
                    # Container block (SlotSSM — has sub-components)
                    blk_total = subdata.get("_block_total", 0)
                    _print(indent + IND, f"├─ {sub}: {_fmt(blk_total)}")
                    for comp, compdata in subdata.items():
                        if comp.startswith("_"):
                            continue
                        _print(indent + IND * 2, f"├─ {comp}: {_fmt(compdata['trainable'])}")
                elif sub.startswith("block["):
                    # Leaf block component (Mamba/LSTM block element)
                    _print(indent + IND, f"├─ {sub}: {_fmt(subdata.get('trainable', 0))}")
                else:
                    _print(indent + IND, f"├─ {sub}: {_fmt(subdata.get('trainable', 0))}")
            continue

        if section == "slot_query (learned attention-pool)":
            _print(indent, f"├─ {section}: {_fmt(data['trainable'])}")
            continue

        if section == "classifier":
            _print(indent, f"├─ classifier ({_fmt(data['total'])} total, {_fmt(data['trainable'])} trainable)")
            children = data.get("children", [])
            for cname, ctrain, ctotal in children:
                _print(indent + IND, f"├─ {cname}: {_fmt(ctrain)} trainable  ({_fmt(ctotal)} total)")
            continue

        if section == "pre-temporal projection":
            _print(indent, f"├─ pre-temporal projection")
            for sub, subdata in data.items():
                if sub.startswith("_"):
                    continue
                _print(indent + IND, f"├─ {sub}: {_fmt(subdata['trainable'])}")
            _print(indent + IND, f"  = {_fmt(data.get('_pre_total', 0))}")
            continue

        if section == "post-temporal classifier":
            _print(indent, f"├─ post-temporal classifier")
            for sub, subdata in data.items():
                if sub.startswith("_"):
                    continue
                _print(indent + IND, f"├─ {sub}: {_fmt(subdata['trainable'])}")
            _print(indent + IND, f"  = {_fmt(data.get('_post_total', 0))}")
            continue

    # Summary
    summary = info.get("_summary", {})
    print()
    print(f"  {'─' * 58}")
    enc_frozen = summary.get("encoder_frozen", 0)
    enc_total = summary.get("encoder_total", 0)
    enc_label = "Encoder (frozen)" if enc_frozen > 0 else "Encoder (trainable)"
    print(f"  {enc_label}:      {_fmt(enc_frozen if enc_frozen > 0 else enc_total)}  ({_fmt(enc_total)} total)")
    print(f"  Trainable (from torch): {_fmt(summary.get('trainable_from_pytorch', 0))}")
    print(f"  Total (from torch):     {_fmt(summary.get('total_from_pytorch', 0))}")


# ---------------------------------------------------------------------------
# Cross-check: verify manual breakdown matches PyTorch
# ---------------------------------------------------------------------------


def cross_check(model: nn.Module, info: dict) -> bool:
    """Verify that the manual breakdown sums to PyTorch's own counts."""
    summary = info.get("_summary", {})
    pt_trainable = summary.get("trainable_from_pytorch", 0)
    pt_total = summary.get("total_from_pytorch", 0)

    # Sum all trainable params from breakdown
    def _sum_trainable(d):
        s = 0
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "trainable" and isinstance(v, (int, float)):
                    s += v
                elif isinstance(v, dict):
                    s += _sum_trainable(v)
        return s

    breakdown_trainable = _sum_trainable(info)

    # The breakdown only counts trainable in non-encoder sections + the temporal._trainable etc.
    # Instead, check total against PyTorch for the trainable subset.
    pt_trainable_set = set(id(p) for p in model.parameters() if p.requires_grad)
    pt_frozen_set = set(id(p) for p in model.parameters() if not p.requires_grad)

    # Verify no overlap
    assert len(pt_trainable_set & pt_frozen_set) == 0, "Overlap between trainable and frozen params!"

    ok = True
    if abs(breakdown_trainable - pt_trainable) > 10:  # allow tiny rounding
        print(f"  ⚠  WARNING: breakdown trainable sum ({breakdown_trainable:,}) != PyTorch trainable ({pt_trainable:,})")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Verify parameter counts for MOVAD models")
    parser.add_argument("--cfg", type=str, default=None, help="Single config file to check")
    parser.add_argument("--all", action="store_true", help="Check all configs")
    parser.add_argument("--train-encoder", action="store_true", help="Unfreeze encoder (count as trainable)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    if args.cfg:
        configs = [args.cfg]
    elif args.all:
        configs = ALL_CONFIGS
    else:
        # Default: all V-JEPA + LSTM/Mamba configs (skip SlotSSM if mamba_ssm not installed)
        configs = ALL_CONFIGS

    for cfg_name in configs:
        # Accept direct paths (e.g. "cfgs/swin_lstm.yaml") or bare names
        if os.path.exists(cfg_name):
            cfg_path = cfg_name
        else:
            cfg_path = os.path.join(CFG_DIR, cfg_name)
        if not os.path.exists(cfg_path):
            print(f"  SKIP: {cfg_path} not found")
            continue

        with open(cfg_path, "r") as fh:
            cfg = EasyDict(yaml.safe_load(fh))
        cfg.device = args.device
        if args.train_encoder:
            cfg.train_encoder = True

        # Skip SlotSSM configs if mamba_ssm not installed
        tm = cfg.get("temporal_model", "")
        if tm in ("slotssm", "sparse_slotssm") and not _HAS_MAMBA_SSM:
            print(f"\n{'=' * 60}")
            print(f"  SKIP: {cfg_name}  (mamba_ssm not installed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {cfg_name}  —  {cfg.get('model_name', '?')}  +  {tm}")
        print(f"{'=' * 60}")

        model = build_cls_vjepa(cfg)
        if args.train_encoder:
            for p in model.encoder.parameters():
                p.requires_grad = True
            model.train_encoder = True
        model.eval()

        info = breakdown_cls_vjepa(model)
        print_breakdown(info)
        cross_check(model, info)

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()