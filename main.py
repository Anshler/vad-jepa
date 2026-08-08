"""
Entry point: V-JEPA 2.1 + MOVAD training / evaluation.

Supports five temporal model variants via config key ``temporal_model``:
  - ``lstm`` (default)
  - ``mamba``
  - ``mamba3``
  - ``slotssm``
  - ``sparse_slotssm``

Usage
-----
Train:
    python main.py --config cfgs/vjepa_v1.yaml --phase train --epochs 200

Test:
    python main.py --config cfgs/vjepa_v1.yaml --phase test --epoch 190

Resume:
    python main.py --config cfgs/vjepa_v1.yaml --phase train --epoch 50
"""
from __future__ import annotations

import argparse
import copy

import gc
import os
import random
import sys
from contextlib import nullcontext

import numpy as np
import torch
import yaml
from easydict import EasyDict
from movad_core.wandb_utils import init_wandb_for_head
from tqdm import tqdm

# Ensure the repo root is on sys.path for sibling imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Compatibility: torchvision >=0.20 moved functional_tensor to _functional_tensor
# pytorchvideo 0.1.5 still imports the old path.
try:
    import torchvision.transforms._functional_tensor as _ft
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
except ImportError:
    pass

from movad_core.dota import Dota, gt_cls_target, setup_dota
from movad_core.losses import build_loss
from movad_core.metrics import evaluation, print_results
from movad_core.optim import build_optimizer
from movad_core import utils as movad_utils
from model import build_multi_head_vjepa


# ---------------------------------------------------------------------------
# CLI
#
# Single-model (backward-compatible):
#   python main.py --config cfgs/vjepa_v1.yaml --phase train
#
# Multi-head (one encoder → multiple temporal models):
#   python main.py --config cfgs/vjepa_v1.yaml cfgs/vjepa_mamba.yaml cfgs/vjepa_slotssm.yaml --phase train
#
# The first config is the *master* — its encoder, data, training, augmentation,
# and output sections are shared.  Each config contributes its temporal-model
# variant as an independent head.  Head name = config basename minus ``.yaml``.
# ---------------------------------------------------------------------------
def parse_configs():
    parser = argparse.ArgumentParser(description="V-JEPA 2.1 + MOVAD anomaly detection")
    _DEFAULT_CONFIGS = [
        "cfgs/vjepa_slotssm.yaml",
        "cfgs/vjepa_sparse_slotssm.yaml",
        "cfgs/vjepa_mamba.yaml",
    ]
    parser.add_argument("--config", nargs="+", default=_DEFAULT_CONFIGS,
                        help="YAML config(s). First = master (encoder/data/training/...); "
                             "rest = temporal-model heads reusing the master's shared settings.")
    parser.add_argument("--phase", default="train", choices=["train", "test"], help="train or test")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--snapshot_interval", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=-1, help="Resume from (train) or eval (test)")
    parser.add_argument("--enable_validation", action="store_true", default=None,
                        help="Run validation during training (overrides config)")
    parser.add_argument("--no-enable_validation", action="store_false", dest="enable_validation",
                        help="Disable validation during training")
    parser.add_argument("--validation_epoch_step", type=int, default=None,
                        help="Validate every N epochs (overrides config)")
    parser.add_argument("--num_frames", type=int, default=None, help="Frames per encoder clip / NF (overrides config)")
    parser.add_argument("--VCL", type=int, default=None, help="Video clip length in frames (overrides config)")
    parser.add_argument("--checkpoint_path", default=None, help="Pretrained encoder checkpoint (overrides config)")
    parser.add_argument("--data_path", default=None, help="Dataset root directory (overrides config)")
    parser.add_argument("--output", default=None, help="Output directory (default: from first config)")
    parser.add_argument("--val_batch_size", type=int, default=2, help="Batch size for validation/test (default: 2)")
    parser.add_argument("--train_encoder", action="store_true", default=False,
                        help="Unfreeze the V-JEPA encoder and train it jointly with the temporal head(s)")
    args = parser.parse_args()

    # Load all configs
    all_cfgs = []
    for path in args.config:
        with open(path, "r") as f:
            all_cfgs.append(EasyDict(yaml.safe_load(f)))

    # Master config = first one
    cfg = all_cfgs[0]

    # Derive output from master config before CLI args clobber it
    _yaml_output = cfg.get("output", "./output")
    _yaml_workers = cfg.get("num_workers", 0)
    _yaml_enable_val = cfg.get("enable_validation", True)
    _yaml_val_step = cfg.get("validation_epoch_step", 10)
    _yaml_ckpt = cfg.get("checkpoint_path", None)
    _yaml_data = cfg.get("data_path", "./data/dota")
    _yaml_vcl = cfg.get("VCL", 16)
    _yaml_nf = cfg.get("num_frames", cfg.get("NF", 16))
    _yaml_batch = cfg.get("batch_size", 8)
    _yaml_lr = cfg.get("lr", 0.0001)

    cfg.update(vars(args))

    if cfg.output is None:
        cfg.output = _yaml_output
    # Keep YAML num_workers unless explicitly overridden via CLI
    if args.num_workers == 0 and _yaml_workers > 0:
        cfg.num_workers = _yaml_workers
    # Keep YAML validation settings unless explicitly overridden via CLI
    if args.enable_validation is None:
        cfg.enable_validation = _yaml_enable_val
    if args.validation_epoch_step is None:
        cfg.validation_epoch_step = _yaml_val_step
    if args.checkpoint_path is None:
        cfg.checkpoint_path = _yaml_ckpt
    if args.data_path is None:
        cfg.data_path = _yaml_data
    if args.VCL is None:
        cfg.VCL = _yaml_vcl
    if args.batch_size is None:
        cfg.batch_size = _yaml_batch
    if args.lr is None:
        cfg.lr = _yaml_lr
    if args.num_frames is None:
        cfg.num_frames = _yaml_nf

    # NF and num_frames are the same thing — keep them in sync
    cfg.NF = cfg.num_frames

    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build head configs — always, even for 1 head
    import pathlib

    encoder_status = "finetuned" if cfg.get("train_encoder", False) else "frozen"
    cfg._head_names = [
        f"{pathlib.Path(p).stem}_VCL_{cfg.VCL}_NF_{cfg.NF}_{encoder_status}"
        for p in args.config
    ]
    head_cfgs_flat = []
    for name, hc in zip(cfg._head_names, all_cfgs):
        entry = dict(hc)
        entry["name"] = name
        head_cfgs_flat.append(entry)
    cfg._head_cfgs_flat = head_cfgs_flat

    return cfg


def set_deterministic(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Checkpoint helpers — strip frozen encoder weights (saves ~600MB per ckpt)
# ---------------------------------------------------------------------------
def _head_state_dict_without_encoder(head, train_encoder: bool = False):
    """Return ``head.state_dict()``, excluding encoder weights unless training them."""
    if train_encoder:
        return head.state_dict()
    return {k: v for k, v in head.state_dict().items() if not k.startswith("encoder.")}


def _load_head_state_dict(head, state_dict, strict=False):
    """Load state dict into a head.

    When ``strict=False`` (default), encoder keys are silently skipped (frozen
    encoder loaded separately).  When ``strict=True`` (``--train_encoder``),
    every key must match — the checkpoint should contain the full model.
    """
    head.load_state_dict(state_dict, strict=strict)


# ---------------------------------------------------------------------------
# Training loop — one encode → N independent temporal trainings
#
# Works for 1 head or N heads — same code path either way.
# Each head has its own optimizer, checkpoint directory, and wandb
# writer.  Losses are never summed — ``.backward()`` flows only through that
# head's temporal + classifier parameters.  The shared encoder is frozen.
# ---------------------------------------------------------------------------
def train(cfg, model, traindata_loader, begin_epoch,
          testdata_loader=None, validation_epoch_step=10,
          opt_state_dicts=None, wandb_resume_id=None):
    """Train a MultiHeadVJEPA: encode once per batch, then loop heads sequentially.

    Heads are trained **independently** — different losses, different
    optimizers, different checkpoints.  Each head's ``.backward()`` only
    touches that head's parameters.

    If ``testdata_loader`` is provided, validation runs every
    ``validation_epoch_step`` epochs and metrics are logged to wandb.
    """
    head_names = list(model.heads.keys())

    # -------------------------------------------------------------------
    # Per-head infrastructure
    # -------------------------------------------------------------------
    head_cfgs: dict[str, dict] = {}
    criterion: dict[str, torch.nn.Module] = {}
    optimizer: dict[str, torch.optim.Optimizer] = {}
    output_dirs: dict[str, str] = {}
    autocast_ctx: dict[str, object] = {}

    for name in head_names:
        hc = model.head_configs.get(name, {})
        head_cfgs[name] = dict(cfg)
        for k in ("dim_latent", "dropout", "rnn_state_size", "rnn_cell_num",
                   "mamba_d_state", "mamba_d_conv", "mamba_expand", "mamba_version",
                   "num_slots", "slot_dim", "num_ssm_blocks", "top_k", "eps_random",
                   "use_inverted_attention", "entropy_weight"):
            if k in hc:
                head_cfgs[name][k] = hc[k]

        output_dir = os.path.join(cfg.output, name)
        output_dirs[name] = output_dir
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        with open(os.path.join(output_dir, "cfg.yml"), "w") as f:
            yaml.dump(head_cfgs[name], f, default_flow_style=False)

        # Build head-specific loss
        head_easy = EasyDict(head_cfgs[name])
        head_easy.device = cfg.device
        criterion[name] = build_loss(head_easy)

        # Build head-specific optimizer (only head params, not shared encoder)
        opt, _ = build_optimizer(
            EasyDict({"lr": cfg.lr}),
            model.heads[name],
            None,
        )
        optimizer[name] = opt

        # Restore optimizer state when resuming from a checkpoint
        opt_sd = (opt_state_dicts or {}).get(name)
        if opt_sd is not None:
            opt.load_state_dict(opt_sd)

        _amp_cfg = head_cfgs[name].get("amp_dtype", "fp32")
        _amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_amp_cfg)
        autocast_ctx[name] = torch.amp.autocast("cuda", dtype=_amp_dtype) if _amp_dtype else nullcontext()

    # -------------------------------------------------------------------
    # Shared wandb writer — ONE run for all heads with metric names
    # prefixed by head.  wandb.init(reinit=True) finishes the previous
    # run, so per-head writers silently killed each other (head 2's init
    # would finish head 1's run → "Run is finished" on next log).
    # -------------------------------------------------------------------
    merged_cfg = dict(cfg)
    merged_cfg["heads"] = list(head_names)
    shared_writer = init_wandb_for_head(
        "-".join(head_names) if len(head_names) > 1 else head_names[0],
        merged_cfg, cfg.output, begin_epoch,
        resume_id=wandb_resume_id,
    )
    _wandb_run_id = shared_writer.id  # stored once, used for all head checkpoints

    # -------------------------------------------------------------------
    # Per-video temporal loop — MOVAD-style: each clip passes through
    # the full model (encoder → projection → temporal → classifier).
    # -------------------------------------------------------------------
    def _run_temporal_loop(head, video_data, data_info, fb, v_len, head_name, opt):
        """MOVAD-style temporal training loop.

        Each clip ``[B, C, NF, H, W]`` passes through the full model:
        ``head(clip, state)`` handles Swin→pool→proj→LSTM→classifier
        (or equivalent ViT path) in one call.

        Backprop cadence is controlled by the ``bptt_horizon`` head config:
          * ``1`` (default)          — per-frame truncated BPTT, hidden state
                                       detached every clip-step (original MOVAD
                                       behavior; starves a recurrent model of
                                       temporal gradient).
          * ``window`` / ``VCL - NF``— full-window BPTT: unroll the whole
                                       window and backprop through all clip-
                                       positions at once, detaching the hidden
                                       state only at the window boundary.  This
                                       is what lets an LSTM actually learn.
          * integer N                — chunked BPTT: backward every N clip-steps.
        """
        toa_batch = data_info[:, 2]
        tea_batch = data_info[:, 3]
        video_len_orig = data_info[:, 0]

        hc = head_cfgs[head_name]
        bptt = hc.get("bptt_horizon", 1)
        grad_clip = hc.get("grad_clip", None)
        n_clip_steps = v_len - fb
        if bptt in (None, "window", float("inf"), "full"):
            bptt = n_clip_steps
        bptt = max(int(bptt), 1)

        # Recurrent models keep (h, c) on the autograd graph between clip-steps
        # so backward can unroll through time; detachment happens at chunk
        # boundaries below instead of inside the model every frame.
        if bptt > 1 and getattr(head.temporal, "detach_every_step", None) is not None:
            head.temporal.detach_every_step = False

        state = None
        slot_diag = {}
        total_loss_val = 0.0
        frame_count = 0

        acc_loss = None
        steps_since_bwd = 0

        for i in range(fb, v_len):
            target = gt_cls_target(i, toa_batch, tea_batch).long()
            clip = video_data[:, :, i - fb:i, :, :]   # [B, C, NF, H, W]

            with autocast_ctx[head_name]:
                output, state = head(clip, state)

            flt = i >= video_len_orig
            target[flt] = -100
            output[flt] = -100

            if head_cfgs[head_name].get("apply_softmax", False):
                output = output.softmax(dim=1)

            loss = criterion[head_name](output, target)

            entropy_weight = head_cfgs[head_name].get("entropy_weight", 0.0)
            if entropy_weight > 0 and hasattr(head, "temporal") and hasattr(head.temporal, "_entropy"):
                ent = head.temporal._entropy
                if isinstance(ent, torch.Tensor) and ent.item() > 0:
                    loss = loss + entropy_weight * (-ent)

            if hasattr(head, "temporal") and hasattr(head.temporal, "_slot_mass_min"):
                _mass_min = head.temporal._slot_mass_min
                if not (isinstance(_mass_min, torch.Tensor) and torch.isnan(_mass_min)):
                    slot_diag["mass_min"] = min(slot_diag.get("mass_min", 999.0), float(_mass_min))
                    slot_diag["mass_mean"] = slot_diag.get("mass_mean", 0.0) + float(head.temporal._slot_mass_mean)
                    slot_diag["usage_frac"] = min(slot_diag.get("usage_frac", 999.0), float(head.temporal._slot_usage_frac))
                    slot_diag["_count"] = slot_diag.get("_count", 0) + 1

            # Accumulate the loss across clip-steps; backprop through time
            # (chunked BPTT) only at boundaries.
            acc_loss = loss if acc_loss is None else acc_loss + loss
            total_loss_val += loss.detach().item()
            frame_count += 1
            steps_since_bwd += 1

            boundary = (steps_since_bwd >= bptt) or (i == v_len - 1)
            if boundary:
                opt.zero_grad()
                acc_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in head.parameters() if p.requires_grad], grad_clip)
                opt.step()
                # Detach the recurrent state at the chunk boundary (LSTM: (h, c)).
                # SSMs like Mamba/Slot carry their own cache and need no detach.
                if isinstance(state, tuple):
                    state = (state[0].detach(), state[1].detach())
                acc_loss = None
                steps_since_bwd = 0

        return total_loss_val, frame_count, slot_diag

    # -------------------------------------------------------------------
    # Epoch loop
    # -------------------------------------------------------------------
    model.train(True)

    for e in range(begin_epoch, cfg.epochs):
        epoch_losses = {n: 0.0 for n in head_names}
        epoch_frames = {n: 0 for n in head_names}

        loader_iter = iter(traindata_loader)

        video_data, data_info = next(loader_iter)
        video_data = video_data.to(cfg.device, non_blocking=True)
        data_info = data_info.to(cfg.device, non_blocking=True)

        n_batches = len(traindata_loader)
        pbar = tqdm(range(n_batches), desc=f"Epoch {e+1}/{cfg.epochs}")
        for j in pbar:
            video_data = torch.swapaxes(video_data, 1, 2)
            v_len = video_data.shape[2]

            if j < n_batches - 1:
                next_video, next_info = next(loader_iter)
                next_video = next_video.to(cfg.device, non_blocking=True)
                next_info = next_info.to(cfg.device, non_blocking=True)

            # Train each head sequentially — MOVAD-style per-frame full forward
            postfix_parts = []
            for name in head_names:
                head = model.heads[name]

                total_loss, f_count, slot_diag = _run_temporal_loop(
                    head, video_data, data_info, cfg.NF, v_len, name, optimizer[name],
                )

                epoch_losses[name] += total_loss
                epoch_frames[name] += max(f_count, 1)

                avg_loss = total_loss / max(f_count, 1)
                shared_writer.log({f"{name}/train/loss_step": avg_loss},
                                  step=e * 1000 + int(j * 1000 / n_batches))
                postfix_parts.append(f"{name}:{avg_loss:.3f}")

                if slot_diag:
                    n = slot_diag.pop("_count", 1)
                    shared_writer.log({
                        f"{name}/slots/mass_min": slot_diag["mass_min"],
                        f"{name}/slots/mass_mean": slot_diag["mass_mean"] / n,
                        f"{name}/slots/usage_frac": slot_diag["usage_frac"],
                    }, step=e * 1000 + int(j * 1000 / n_batches))

            pbar.set_postfix_str(" ".join(postfix_parts))

            if j < n_batches - 1:
                video_data = next_video
                data_info = next_info

        # End-of-epoch logging
        print(f"  Epoch {e+1}/{cfg.epochs}")
        for name in head_names:
            e_loss = epoch_losses[name] / max(epoch_frames[name], 1)
            shared_writer.log({f"{name}/train/epoch_loss": e_loss}, step=e * 1000 + 999)
            print(f"    {name}: avg_loss={e_loss:.4f}  ({epoch_frames[name]} frames)")

        # Checkpoint
        if (e + 1) % cfg.snapshot_interval == 0:
            for name in head_names:
                ckpt_path = os.path.join(output_dirs[name], "checkpoints", f"model-{e+1:02d}.pt")
                torch.save(
                    {
                        "epoch": e,
                        "model_state_dict": _head_state_dict_without_encoder(
                            model.heads[name], train_encoder=cfg.train_encoder
                        ),
                        "optimizer_state_dict": optimizer[name].state_dict(),
                        "wandb_run_id": _wandb_run_id,
                    },
                    ckpt_path,
                )
                print(f"    {name} checkpoint → {ckpt_path}")

        # Validation — free training tensors first to avoid stacking
        if testdata_loader is not None and (e + 1) % validation_epoch_step == 0:
            # Shut down training DataLoader workers to reclaim CPU RAM.
            # torch.cuda.empty_cache() only frees GPU VRAM — the worker pool
            # (num_workers=4) still holds prefetched video batches in system RAM.
            # Deleting the iterator triggers worker-shutdown; gc.collect() forces
            # immediate reclamation before eval loads its own data.
            del loader_iter
            gc.collect()
            torch.cuda.empty_cache()

            print(f"\n  === Validation at epoch {e+1} ===")
            _evaluate_model(cfg, model, testdata_loader, e + 1, writer=shared_writer, autocast_ctx=autocast_ctx)

            # After eval: reclaim eval tensors, then recreate training iterator
            # for the next epoch.
            torch.cuda.empty_cache()
            gc.collect()
            model.train(True)
            # Training iterator is recreated at the top of the next epoch

    shared_writer.finish()

    # Save final checkpoint if not already saved at the last epoch
    if cfg.epochs % cfg.snapshot_interval != 0:
        for name in head_names:
            ckpt_path = os.path.join(output_dirs[name], "checkpoints", f"model-{cfg.epochs:02d}.pt")
            torch.save(
                {
                    "epoch": cfg.epochs - 1,
                    "model_state_dict": _head_state_dict_without_encoder(
                        model.heads[name], train_encoder=cfg.train_encoder
                    ),
                    "optimizer_state_dict": optimizer[name].state_dict(),
                    "wandb_run_id": _wandb_run_id,
                },
                ckpt_path,
            )
            print(f"    {name} final checkpoint → {ckpt_path}")


# ---------------------------------------------------------------------------
# Shared evaluation helper — one encode → evaluate all heads sequentially
# Used by both train-time validation and standalone testing.
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _evaluate_model(cfg, model, testdata_loader, epoch, writer=None, autocast_ctx=None):
    """Run validation/test inference and compute metrics for all heads.

    Each clip passes through the full model (encoder → temporal → classifier)
    matching the original MOVAD evaluation pattern exactly.

    Parameters
    ----------
    cfg : EasyDict
    model : MultiHeadVJEPA
    testdata_loader : DataLoader
    epoch : int — current epoch number (used for result filenames and logging step)
    writer : wandb.Run | None — if provided, metrics are logged to wandb
             under ``<head>/val/<metric>`` for each head.

    Returns
    -------
    dict[str, dict] — per-head metrics dict as returned by ``evaluation()``.
    """
    fb = cfg.NF
    head_names = list(model.heads.keys())

    per_head = {
        n: {
            "targets_all": [], "outputs_all": [], "toas_all": [],
            "teas_all": [], "idxs_all": [], "info_all": [], "frames_counter": [],
        }
        for n in head_names
    }

    model.eval()

    for batch_idx, batch in enumerate(tqdm(testdata_loader, desc=f"Val epoch {epoch}")):
        video_data, data_info = batch
        video_data = video_data.to(cfg.device, non_blocking=True)
        data_info = data_info.to(cfg.device, non_blocking=True)
        video_data = torch.swapaxes(video_data, 1, 2)
        B = video_data.shape[0]
        v_len_max = video_data.shape[2]         # already padded to max in bucket

        video_len_orig = data_info[:, 0]            # [B] — truth for masking
        idx_batch = data_info[:, 1]
        toa_batch = data_info[:, 2]
        tea_batch = data_info[:, 3]
        info_batch = data_info[:, 7:11]

        for name in head_names:
            head = model.heads[name]

            t_shape = (B, v_len_max - fb)
            targets = torch.full(t_shape, -100).to(cfg.device)
            outputs = torch.full(t_shape, -100, dtype=torch.float).to(cfg.device)

            state = None
            for i in range(fb, v_len_max):
                target = gt_cls_target(i, toa_batch, tea_batch).long()
                clip = video_data[:, :, i - fb:i, :, :]   # [B, C, NF, H, W]
                _ac = autocast_ctx.get(name, nullcontext()) if autocast_ctx else nullcontext()
                with _ac:
                    output, state = head(clip, state)       # full forward: encoder → temporal → classifier

                # Mask padded positions — identical pattern to training loop
                flt = i >= video_len_orig
                target[flt] = -100
                output[flt] = -100

                if cfg.get("apply_softmax", False):
                    output = output.softmax(dim=1)

                targets[:, i - fb] = target.clone()
                # Store class-1 probability (not raw logit) so threshold 0.5 is meaningful
                prob = output if cfg.get("apply_softmax", False) else output.softmax(dim=1)
                outputs[:, i - fb] = prob[:, 1].clone()

            res = per_head[name]
            # Append per-video results, filtering out padded frames
            for b in range(B):
                vl = int(video_len_orig[b].item())
                valid_frames = vl - fb
                if valid_frames <= 0:
                    continue
                res["targets_all"].append(targets[b, :valid_frames].tolist())
                res["outputs_all"].append(outputs[b, :valid_frames].tolist())
                res["toas_all"].append(toa_batch[b].item())
                res["teas_all"].append(tea_batch[b].item())
                res["idxs_all"].append(idx_batch[b].item())
                res["info_all"].append(info_batch[b].tolist())
                res["frames_counter"].append(vl)

        # Periodically flush the CUDA caching allocator to release any
        # cached-but-unusable blocks back to the OS.
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    # Save and evaluate per-head
    import pickle

    all_metrics = {}
    for name in head_names:
        res = per_head[name]
        head_output = os.path.join(cfg.output, name)
        eval_dir = os.path.join(head_output, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        filename = os.path.join(eval_dir, f"results-{epoch:02d}.pkl")

        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "targets": res["targets_all"],
                    "outputs": res["outputs_all"],
                    "toas": np.array(res["toas_all"]).reshape(-1),
                    "teas": np.array(res["teas_all"]).reshape(-1),
                    "idxs": np.array(res["idxs_all"]).reshape(-1),
                    "info": np.array(res["info_all"]).reshape(-1, 4),
                    "frames_counter": np.array(res["frames_counter"]).reshape(-1),
                },
                f,
            )

        content = movad_utils.load_results(filename)
        (auc_roc, auc_pr, f1_one, f1_mean, accuracy,
         report, eval_per_class, eval_per_class_ego) = evaluation(FPS=cfg.FPS, **content)
        all_metrics[name] = {
            "auc_roc": auc_roc, "auc_pr": auc_pr, "f1": f1_one,
            "f1_mean": f1_mean, "accuracy": accuracy,
        }

        # Log to wandb if writer provided (train-time validation)
        if writer is not None:
            try:
                writer.log({
                    f"{name}/val/auc_roc": auc_roc,
                    f"{name}/val/auc_pr": auc_pr,
                    f"{name}/val/f1": f1_one,
                    f"{name}/val/f1_mean": f1_mean,
                    f"{name}/val/accuracy": accuracy,
                }, step=epoch * 1000 - 1)
            except Exception:
                pass

        print(f"  [{name}] validation results (epoch {epoch}):")
        print_results(cfg, auc_roc, auc_pr, f1_one, f1_mean, accuracy,
                      report, eval_per_class, eval_per_class_ego)

    return all_metrics


# ---------------------------------------------------------------------------
# Standalone testing — load checkpoints and evaluate
# ---------------------------------------------------------------------------
@torch.inference_mode()
def test(cfg, model, testdata_loader, epoch):
    head_names = list(model.heads.keys())
    autocast_ctx: dict[str, object] = {}
    for name in head_names:
        hc = model.head_configs.get(name, {})
        _amp_cfg = hc.get("amp_dtype", cfg.get("amp_dtype", "fp32"))
        _amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_amp_cfg)
        autocast_ctx[name] = torch.amp.autocast("cuda", dtype=_amp_dtype) if _amp_dtype else nullcontext()
    return _evaluate_model(cfg, model, testdata_loader, epoch, writer=None, autocast_ctx=autocast_ctx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = parse_configs()
    set_deterministic(cfg.seed)

    # NF already synced to num_frames in parse_configs

    # --- Data --------------------------------------------------------------
    if cfg.phase == "train":
        # Training loader — always raw video
        traindata_loader, _ = setup_dota(
            Dota, cfg, num_workers=cfg.num_workers,
            VCL=cfg.get("VCL", None), phase="train",
        )

        # Validation loader — always raw video (per-clip full-model forward)
        testdata_loader = None
        if cfg.get("enable_validation", False):
            test_cfg = copy.deepcopy(cfg)
            test_cfg.batch_size = cfg.val_batch_size
            _, testdata_loader = setup_dota(
                Dota, test_cfg, num_workers=cfg.num_workers,
                VCL=None, phase="test",
            )

    elif cfg.phase == "test":
        traindata_loader = None
        test_cfg = copy.deepcopy(cfg)
        test_cfg.batch_size = cfg.val_batch_size
        _, testdata_loader = setup_dota(
            Dota, test_cfg, num_workers=cfg.num_workers,
            VCL=None, phase="test",
        )

    # --- Model -------------------------------------------------------------
    epoch_val = 0
    model = build_multi_head_vjepa(cfg)

    # Resume / load checkpoints per head
    opt_state_dicts = {}
    wandb_resume_id = None
    if cfg.epoch != -1:
        for name, head in model.heads.items():
            head_output = os.path.join(cfg.output, name)
            try:
                ckpt_cfg = EasyDict({
                    "output": head_output,
                    "epoch": cfg.epoch,
                    "device": cfg.device,
                })
                ckpt = movad_utils.load_checkpoint(ckpt_cfg)
                _load_head_state_dict(head, ckpt["model_state_dict"],
                                      strict=cfg.train_encoder)
                opt_state_dicts[name] = ckpt.get("optimizer_state_dict")
                if wandb_resume_id is None:
                    wandb_resume_id = ckpt.get("wandb_run_id")
                ep = ckpt["epoch"] + 1
                print(f"  [{name}] resumed from epoch {ep}")
            except FileNotFoundError:
                print(f"  [{name}] no checkpoint at epoch {cfg.epoch} — starting fresh")
        epoch_val = cfg.epoch if cfg.epoch > 0 else 0

    if cfg.phase == "train":
        val_step = cfg.get("validation_epoch_step", 10)
        if testdata_loader is not None:
            val_step = min(val_step, cfg.epochs)
        train(cfg, model, traindata_loader, epoch_val,
              testdata_loader=testdata_loader,
              validation_epoch_step=val_step,
              opt_state_dicts=opt_state_dicts,
              wandb_resume_id=wandb_resume_id)

    elif cfg.phase == "test":
        if cfg.epoch == -1:
            cfg.epoch = 0
        # Load each head's checkpoint
        for name, head in model.heads.items():
            head_output = os.path.join(cfg.output, name)
            try:
                ckpt_cfg = EasyDict({
                    "output": head_output,
                    "epoch": cfg.epoch,
                    "device": cfg.device,
                })
                ckpt = movad_utils.load_checkpoint(ckpt_cfg)
                _load_head_state_dict(head, ckpt["model_state_dict"])
                ep = ckpt["epoch"] + 1
                print(f"  [{name}] loaded checkpoint epoch {ep}")
            except FileNotFoundError:
                print(f"  [{name}] WARNING: no checkpoint at epoch {cfg.epoch}")
        test(cfg, model, testdata_loader, cfg.epoch)
