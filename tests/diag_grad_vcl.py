"""
WHY does LSTM fail (49.6% AUC) while Mamba learns, on the SAME VCL-window
dataloader / same LR / same randomness?

Measures, per epoch, the gradient a model ACTUALLY receives and how much its
weights actually move, on the exact failing setup:
  * real DoTA dataloader, VCL-length random windows, batch_size=1
  * full-window BPTT (unroll = VCL-NF clip-steps), lr from a --lr toggle
  * identical for --model lstm and --model mamba

If Mamba's gradient/weight-movement is ~100x larger than LSTM's at the same
lr, that is the intrinsic cause (gate-attenuated LSTM gradient starvation),
independent of batch size or data.

Usage (WSL, repo root):
  python tests/diag_grad_vcl.py --model lstm  --lr 0.003 --nvideos 8 --epochs 4
  python tests/diag_grad_vcl.py --model mamba --lr 0.003 --nvideos 8 --epochs 4
"""
from __future__ import annotations

import argparse, os, sys, json, numpy as np, torch, torch.nn.functional as F, yaml
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
parser.add_argument("--config", default=None)   # None -> pick from --model
parser.add_argument("--model", default="lstm", choices=["lstm", "mamba"])
parser.add_argument("--nvideos", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--workers", type=int, default=2)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg_file = args.config or ("cfgs/vjepa_v1.yaml" if args.model == "lstm"
                           else "cfgs/vjepa_mamba.yaml")
print(f"[grad-vcl {args.model}] {cfg_file} lr={args.lr} nvideos={args.nvideos} "
      f"epochs={args.epochs} device={DEVICE}")

torch.manual_seed(0); torch.cuda.manual_seed(0)

cfg = EasyDict(yaml.safe_load(open(os.path.join(_REPO_ROOT, cfg_file))))
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
VCL = cfg.get("VCL", 8)

mean = cfg.get("data_mean", [0.218, 0.220, 0.209])
std = cfg.get("data_std", [0.277, 0.280, 0.277])
transform = T.Compose([
    pad_frames(cfg.get("input_shape", [384, 384])),
    T.Lambda(lambda x: torch.tensor(x)),
    T.Lambda(lambda x: x.permute(0, 3, 1, 2)),
    T.Lambda(lambda x: x / 255.0),
    T.Normalize(mean, std),
])
data = Dota(cfg.get("data_path",
                    "/mnt/d/Users/Chrysenberg69420/Downloads/DoTA_dataset"),
            "train", transforms={"image": transform}, VCL=cfg.get("VCL", 8),
            vertical_flip_prob=0., horizontal_flip_prob=0.)
order = sorted(range(len(data)),
               key=lambda i: data.metadata[data.keys[i]]["num_frames"])
subset = Subset(data, order[: args.nvideos])
loader = DataLoader(subset, batch_size=1, shuffle=True,
                    num_workers=args.workers, pin_memory=False,
                    collate_fn=pad_collate_videos)

model = build_multi_head_vjepa(cfg)
head_cfg = EasyDict(dict(cfg)); head_cfg.device = DEVICE; head_cfg.name = "h"
criterion = build_loss(head_cfg)
head = next(iter(model.heads.values()))
for m in head.modules():
    if isinstance(m, torch.nn.Dropout): m.p = 0.0
head.temporal.detach_every_step = False   # full-window BPTT (same as prod)

temporal_pars = list(head.temporal.parameters())
opt, _ = build_optimizer(EasyDict({"lr": args.lr}), head, None)
model.to(DEVICE); model.train()

print(f"  temporal_type={head.temporal_type}  temporal params="
      f"{sum(p.numel() for p in temporal_pars):,}")

# capture starting weights (after .to(DEVICE)) to measure movement
p0 = {id(p): p.detach().clone() for p in head.parameters()}
def weight_move():
    return sum((p.detach() - p0[id(p)]).norm().item() for p in head.parameters())

for ep in range(args.epochs):
    t_loss = 0.0; frames = 0; steps = 0
    g_tot, g_temp, g_lin1 = [], [], []
    for video_data, info in loader:
        video_data = video_data.permute(0, 2, 1, 3, 4).to(DEVICE)
        info = info.to(DEVICE)
        Tlen = video_data.shape[2]
        toa, tea, vlo = info[:, 2], info[:, 3], info[:, 0]
        state = None; acc = None
        opt.zero_grad()
        for i in range(NF, Tlen):
            target = gt_cls_target(i, toa, tea).long()
            with autocast_ctx:
                output, state = head(video_data[:, :, i - NF:i], state)
            flt = i >= vlo
            target[flt] = -100; output[flt] = -100
            loss = criterion(output, target)
            acc = loss if acc is None else acc + loss
            t_loss += loss.detach().item() * video_data.shape[0]
            frames += video_data.shape[0]
            if i == Tlen - 1:                            # end-of-window BPTT
                acc.backward()
                g_tot.append(sum(p.grad.norm().item() for p in head.parameters()
                                 if p.grad is not None))
                g_temp.append(sum(p.grad.norm().item() for p in temporal_pars
                                  if p.grad is not None))
                gl1 = head.lin1.weight.grad
                g_lin1.append(gl1.norm().item() if gl1 is not None else 0.0)
                opt.step(); opt.zero_grad()
                if isinstance(state, tuple):
                    state = (state[0].detach(), state[1].detach())
                acc = None; steps += 1
    print(f"  ep {ep+1}: loss={t_loss/max(frames,1):.4f}  "
          f"grad(total)={np.mean(g_tot):.3e}  "
          f"grad(temp)={np.mean(g_temp):.3e}  "
          f"grad(lin1)={np.mean(g_lin1):.3e}  weight_move={weight_move():.3e}")

print(f"=> {args.model} at lr={args.lr}: "
      f"per-frame loss={t_loss/max(frames,1):.4f} (chance ~0.69)")
