"""
Diagnostic: why does the LSTM head stall at lr=5e-5 while Mamba learns?

Compares LSTM vs Mamba on the SAME real DoTA video, measuring per-layer
gradient norms and LSTM cell/hidden state norms over training steps.

Usage (from WSL, repo root):
    python tests/diag_lstm_grad.py --config cfgs/vjepa_v1.yaml   --lr 5e-5 --epochs 3
    python tests/diag_lstm_grad.py --config cfgs/vjepa_mamba.yaml --lr 5e-5 --epochs 3
    ... --lr 0.01          (to see the contrast that makes LSTM overfit)
"""
from __future__ import annotations

import argparse, os, sys, json, numpy as np, torch, torch.nn.functional as F, yaml
from contextlib import nullcontext
from easydict import EasyDict
from PIL import Image
import torchvision.transforms.functional as TF

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    import torchvision.transforms._functional_tensor as _ft
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
except ImportError:
    pass

from model import build_multi_head_vjepa
from movad_core.dota import gt_cls_target
from movad_core.losses import build_loss
from movad_core.optim import build_optimizer

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="cfgs/vjepa_v1.yaml")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--epochs", type=int, default=3)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{os.path.basename(args.config)}]  device={DEVICE}  lr={args.lr}  epochs={args.epochs}")

torch.manual_seed(0); torch.cuda.manual_seed(0)

cfg = EasyDict(yaml.safe_load(open(args.config)))
cfg.device = DEVICE
cfg.compile = False
cfg.lr = args.lr
cfg.train_encoder = False
cfg._head_cfgs_flat = [dict(cfg)]
cfg._head_cfgs_flat[0]["name"] = "test_head"

_amp = cfg.get("amp_dtype", "fp32")
_dt = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_amp)
autocast_ctx = torch.amp.autocast("cuda", dtype=_dt) if _dt else nullcontext()
print(f"  amp_dtype={_amp}")

NF = cfg.get("num_frames", 4)
DATA_PATH = cfg.get("data_path", "/mnt/d/Users/Chrysenberg69420/Downloads/DoTA_dataset")

# ---- shortest real DoTA anomaly video ----
meta_path = os.path.join(DATA_PATH, "metadata", "metadata_val.json")
metadata = json.load(open(meta_path))
shortest = None
for key, info in metadata.items():
    nf = info["num_frames"]; astart = info.get("anomaly_start", -1)
    if astart < 0 or nf <= NF + 2: continue
    if shortest is None or nf < shortest[0]:
        shortest = (nf, key, astart, info["anomaly_end"])
n_frames, video_key, a_start, a_end = shortest
frames_dir = os.path.join(DATA_PATH, "frames", video_key, "images")
frame_files = sorted(os.listdir(frames_dir))[:n_frames]
frames_np = np.array([np.asarray(Image.open(os.path.join(frames_dir, f))) for f in frame_files]).astype(np.float32)
input_shape = cfg.get("input_shape", [384, 384])
H_in, W_in = input_shape
frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
frames_t = TF.resize(frames_t, [H_in, W_in], antialias=True)
frames_t = (frames_t / 255.0 - 0.5) / 0.5
vd = frames_t.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)   # [1, C, T, H, W]
v_len = n_frames
print(f"  video={video_key} frames={v_len} anomaly=[{a_start},{a_end}]")

di = torch.zeros(1, 11, device=DEVICE)
di[:, 0] = v_len; di[:, 2] = a_start; di[:, 3] = a_end; di[:, 4] = 1

# ---- build model (frozen encoder) ----
model = build_multi_head_vjepa(cfg)
head = next(iter(model.heads.values()))
head_cfg = EasyDict(dict(cfg)); head_cfg.device = DEVICE; head_cfg.name = "test_head"
criterion = build_loss(head_cfg)
opt, _ = build_optimizer(EasyDict({"lr": args.lr}), head, None)
model.to(DEVICE); model.train()
temporal_pars = list(head.temporal.parameters())
print(f"  temporal_type={head.temporal_type}  temporal params={sum(p.numel() for p in temporal_pars):,}")

def snap(label, t):
    with torch.no_grad():
        _min = float(t.abs().min()); _max = float(t.abs().max()); _mean = float(t.abs().mean())
    print(f"    {label}: abs-min={_min:.3e}  abs-mean={_mean:.3e}  abs-max={_max:.3e}")

toa, tea, vlo = di[:, 2], di[:, 3], di[:, 0]

for ep in range(args.epochs):
    state = None
    ep_loss, frames = 0.0, 0
    g_tot = []
    g_lin1 = []
    g_lstm = []
    hnorm = cnorm = None
    for i in range(NF, v_len):
        target = gt_cls_target(i, toa, tea).long()
        clip = vd[:, :, i - NF:i, :, :]
        with autocast_ctx:
            output, state = head(clip, state)
        flt = i >= vlo
        target[flt] = -100; output[flt] = -100
        loss = criterion(output, target)
        opt.zero_grad(); loss.backward()

        # gradient norms
        glin1 = head.lin1.weight.grad
        gtot = sum(p.grad.norm().item() for p in head.parameters() if p.grad is not None)
        glstm = sum(p.grad.norm().item() for p in temporal_pars if p.grad is not None)
        g_tot.append(gtot); g_lin1.append(glin1.norm().item()); g_lstm.append(glstm)

        # LSTM state norms
        if hasattr(state, "__len__") and isinstance(state, tuple):
            hnorm = state[0].abs().mean().item(); cnorm = state[1].abs().mean().item()

        opt.step()
        ep_loss += loss.detach().item(); frames += 1

    if isinstance(head.temporal_type, str) and head.temporal_type == "lstm":
        print(f"\n[epoch {ep}]  loss={ep_loss/frames:.5f}  (LSTM h={hnorm:.3f} c={cnorm:.3f})")
    else:
        print(f"\n[epoch {ep}]  loss={ep_loss/frames:.5f}")
    print(f"    grad-norm over {frames} frames:")
    print(f"      temporal-params : mean={np.mean(g_lstm):.3e}  min={min(g_lstm):.3e}  max={max(g_lstm):.3e}")
    print(f"      lin1 (upstream)  : mean={np.mean(g_lin1):.3e}  min={min(g_lin1):.3e}  max={max(g_lin1):.3e}")
    print(f"      TOTAL            : mean={np.mean(g_tot):.3e}  min={min(g_tot):.3e}  max={max(g_tot):.3e}")
