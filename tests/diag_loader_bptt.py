"""
Dataloader-based smoke test: CHUNKED-BPTT fix on the REAL DoTA dataloader.

Unlike diag_loop_bptt.py (which loads every chosen video onto the GPU at
once), this streams videos through the actual ``Dota`` dataset / DataLoader
one at a time, so GPU memory is bounded by a single video — you can scale
``--nvideos`` without blowing up VRAM or RAM.

  * builds the full model via build_multi_head_vjepa
  * iterates the DoTA loader (batch_size=1) over the N shortest videos
  * for each video, runs the REAL ``head(clip, state)`` forward over its
    clip-positions with ``detach_every_step = False`` (the flag the new
    main.py loop sets when bptt_horizon > 1) and full-window BPTT
  * one grad step per video (state reset between videos)
  * if the fix holds, the mean loss should drop well below ~0.59 (chance)
    and keep falling

Usage (WSL, repo root):
  python tests/diag_loader_bptt.py --nvideos 10 --lr 0.003 --epochs 60
  python tests/diag_loader_bptt.py --nvideos 10 --lr 0.001 --epochs 60
  python tests/diag_loader_bptt.py --phase val           # eval val split
  python tests/diag_loader_bptt.py --keep_dropout        # real 0.3 dropout
"""
from __future__ import annotations

import argparse, os, sys, copy, numpy as np, torch, torch.nn.functional as F, yaml
from contextlib import nullcontext
from easydict import EasyDict
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    import torchvision.transforms._functional_tensor as _ft
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
except ImportError:
    pass

from model import build_multi_head_vjepa
from movad_core.dota import Dota, gt_cls_target, pad_frames
from movad_core.data_transform import pad_collate_videos
from movad_core.losses import build_loss
from movad_core.optim import build_optimizer

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="cfgs/vjepa_v1.yaml")
parser.add_argument("--nvideos", type=int, default=10,
                    help="train over the N shortest anomaly videos")
parser.add_argument("--batch", type=int, default=1,
                    help="windows per optimizer step (real training uses 32; "
                         "batch=1 is much noisier and can hide the fix)")
parser.add_argument("--phase", default="train", choices=["train", "val"])
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--clip", type=float, default=0.0, help="grad-norm clip; 0 disables")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--keep_dropout", action="store_true", default=False)
parser.add_argument("--log_every", type=int, default=1)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[loader {os.path.basename(args.config)}][{args.phase}] nvideos={args.nvideos} "
      f"lr={args.lr} clip={args.clip} workers={args.workers} device={DEVICE} epochs={args.epochs}")

torch.manual_seed(0); torch.cuda.manual_seed(0)

cfg = EasyDict(yaml.safe_load(open(os.path.join(_REPO_ROOT, args.config))))
cfg.device = DEVICE
cfg.compile = False
cfg.lr = args.lr
cfg.train_encoder = False
cfg._head_cfgs_flat = [dict(cfg)]
cfg._head_cfgs_flat[0]["name"] = "h"

_amp = cfg.get("amp_dtype", "fp32")
_dt = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_amp)
autocast_ctx = torch.amp.autocast("cuda", dtype=_dt) if _dt else nullcontext()

NF = cfg.get("num_frames", 4)
VCL = cfg.get("VCL", 8)   # mimic real training: sample a VCL-length window per video

# ---- real DoTA dataset + transform, VCL-window sampling (no AugMix for a clean signal)
mean = cfg.get("data_mean", [0.218, 0.220, 0.209])
std = cfg.get("data_std", [0.277, 0.280, 0.277])
transform = T.Compose([
    pad_frames(cfg.get("input_shape", [384, 384])),
    T.Lambda(lambda x: torch.tensor(x)),
    T.Lambda(lambda x: x.permute(0, 3, 1, 2)),          # [T,H,W,C] -> [T,C,H,W]
    T.Lambda(lambda x: x / 255.0),
    T.Normalize(mean, std),
])
data = Dota(cfg.get("data_path",
                    "/mnt/d/Users/Chrysenberg69420/Downloads/DoTA_dataset"),
            args.phase, transforms={"image": transform}, VCL=None,
            vertical_flip_prob=0., horizontal_flip_prob=0.)

# N shortest (anomaly-trimmed) videos via Subset
order = sorted(range(len(data)),
               key=lambda i: data.metadata[data.keys[i]]["num_frames"])
subset = Subset(data, order[: args.nvideos])
pin_memory = False  # keep it simple / avoids pinned-memory surprises on WSL
loader = DataLoader(subset, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=pin_memory,
                    collate_fn=pad_collate_videos, drop_last=False)
print(f"  dataset={len(data)} videos  ->  sampled {len(subset)} shortest "
      f"(VCL={VCL}, {VCL - NF} clip-positions/window, batch={args.batch}):")
for i in order[: args.nvideos]:
    k = data.keys[i]
    print(f"    {k}: {data.metadata[k]['num_frames']} frames")

# ---- full model (frozen encoder), real forward path ----
model = build_multi_head_vjepa(cfg)
head_cfg = EasyDict(dict(cfg)); head_cfg.device = DEVICE; head_cfg.name = "h"
criterion = build_loss(head_cfg)

head = next(iter(model.heads.values()))
if not args.keep_dropout:
    for m in head.modules():
        if isinstance(m, torch.nn.Dropout): m.p = 0.0   # deterministic diagnostics

by_bptt = 10**9  # full-window BPTT: one grad step per video
head.temporal.detach_every_step = False
print(f"  temporal_type={head.temporal_type}  detach_every_step=False (window BPTT)")

opt, _ = build_optimizer(EasyDict({"lr": args.lr}), head, None)
model.to(DEVICE); model.train()

for ep in range(args.epochs):
    total_loss = 0.0; frames = 0
    for video_data, info in loader:
        # video_data: [B, T, C, H, W] -> [B, C, T, H, W]
        video_data = video_data.permute(0, 2, 1, 3, 4).to(DEVICE)
        info = info.to(DEVICE)
        Tlen = video_data.shape[2]
        toa, tea, vlo = info[:, 2], info[:, 3], info[:, 0]
        B = video_data.shape[0]

        state = None; acc = None; since = 0
        opt.zero_grad()
        for i in range(NF, Tlen):
            target = gt_cls_target(i, toa, tea).long()          # [B]
            with autocast_ctx:
                output, state = head(video_data[:, :, i - NF:i], state)
            flt = i >= vlo
            target[flt] = -100; output[flt] = -100
            loss = criterion(output, target)
            acc = loss if acc is None else acc + loss
            total_loss += loss.detach().item() * B
            frames += B; since += 1

            if (since >= by_bptt) or (i == Tlen - 1):     # end-of-window boundary
                acc.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in head.parameters() if p.requires_grad], args.clip)
                opt.step(); opt.zero_grad()
                if isinstance(state, tuple):
                    state = (state[0].detach(), state[1].detach())
                acc = None; since = 0

    if ep == 0 or (ep + 1) % args.log_every == 0 or ep == args.epochs - 1:
        print(f"  ep {ep+1:3d}: loss={total_loss/max(frames,1):.5f}", flush=True)

print(f"\n=> loader {args.phase} nvideos={args.nvideos} lr={args.lr} final={total_loss/max(frames,1):.5f}")
