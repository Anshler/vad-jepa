"""
Fast single-config overfit test.
Usage:
    python tests/test_overfit.py --train_encoder --lr 0.005 --epochs 100
    python tests/test_overfit.py                    # frozen, lr=0.01
    python tests/test_overfit.py --train_encoder    # trainable, lr=0.01
    python tests/test_overfit.py --softmax          # with old double-softmax behavior
    python tests/test_overfit.py --real_data        # shortest real video from DoTA
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
parser.add_argument("--config", default="cfgs/videomae_s_mamba.yaml",
                    help="Path to config YAML (e.g. cfgs/videomae_s_mamba.yaml or cfgs/vjepa_mamba.yaml)")
parser.add_argument("--checkpoint", default=None,
                    help="Override checkpoint path (uses config value if not set)")
parser.add_argument("--epochs", type=int, default=100)
# Default: use the config's LR (all head cfgs train at 5e-5; the test must not
# invent its own — an LR 100× too high diverges the Mamba head into NaN).
parser.add_argument("--lr", type=float, default=None, help="Override config LR (default: use config's `lr` value)")
parser.add_argument("--train_encoder", action="store_true", default=False)
parser.add_argument("--softmax", action="store_true", default=False,
                    help="Enable double-softmax (for comparing with old behavior)")
parser.add_argument("--real_data", action="store_true", default=False,
                    help="Use the shortest real video from DoTA instead of synthetic data")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
apply_softmax = args.softmax
print(f"Device: {DEVICE}  |  train_encoder={args.train_encoder}  |  lr={args.lr or 'cfg'}"
      f"  |  epochs={args.epochs}  |  softmax={apply_softmax}  |  real_data={args.real_data}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# --- Build model ---
cfg_path = os.path.join(_REPO_ROOT, args.config)
cfg = EasyDict(yaml.safe_load(open(cfg_path)))
cfg.device = DEVICE
cfg.checkpoint_path = args.checkpoint or cfg.get("checkpoint_path")
cfg.compile = False
# Effective LR: CLI override, else the config's value (all head cfgs train at 5e-5).
lr = args.lr if args.lr is not None else cfg.get("lr", 5e-5)
print(f"  effective lr={lr}")
cfg.lr = lr
cfg.train_encoder = args.train_encoder
cfg._head_cfgs_flat = [dict(cfg)]
cfg._head_cfgs_flat[0]["name"] = "test_head"

_amp_cfg = cfg.get("amp_dtype", "fp32")
_amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_amp_cfg)
autocast_ctx = torch.amp.autocast("cuda", dtype=_amp_dtype) if _amp_dtype else nullcontext()
print(f"  amp_dtype={_amp_cfg}")

NF = cfg.get("num_frames", 4)
config_VCL = cfg.get("VCL", 8)
input_shape = cfg.get("input_shape", [256, 256])

model = build_multi_head_vjepa(cfg)
head_name = next(iter(model.heads.keys()))
head = model.heads[head_name]

head_cfg = EasyDict(dict(cfg))
head_cfg.device = DEVICE
head_cfg.name = head_name
criterion = build_loss(head_cfg)
opt, _ = build_optimizer(EasyDict({"lr": lr}), head, None)

model.to(DEVICE)
model.train()

# --- Data ---
if args.real_data:
    # Load metadata, find the shortest video with an anomaly window
    DATA_PATH = cfg.get("data_path", "D:/Users/Chrysenberg69420/Downloads/DoTA_dataset")
    meta_path = os.path.join(DATA_PATH, "metadata", "metadata_val.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    shortest = None
    for key, info in metadata.items():
        nf = info["num_frames"]
        astart = info.get("anomaly_start", -1)
        if astart < 0 or nf <= NF + 2:    # need at least a few valid frames
            continue
        if shortest is None or nf < shortest[0]:
            shortest = (nf, key, astart, info["anomaly_end"])

    assert shortest is not None, "No anomaly video found!"
    n_frames, video_key, a_start, a_end = shortest
    print(f"\nShortest anomaly video: {video_key}")
    print(f"  Frames: {n_frames}  Anomaly window: [{a_start}, {a_end}]  ({a_end - a_start + 1}f)")

    # Load raw frames
    frames_dir = os.path.join(DATA_PATH, "frames", video_key, "images")
    frame_files = sorted(os.listdir(frames_dir))[:n_frames]
    frames_np = np.array([np.asarray(Image.open(os.path.join(frames_dir, f)))
                           for f in frame_files]).astype(np.float32)
    # frames_np: [T, H, W, C]

    # Preprocess: resize + normalize using the config's data_mean/data_std
    # (matching the encoder's assumed input stats — MOVAD uses [-1,1] i.e.
    #  mean/std 0.5; VideoMAE uses ImageNet stats).
    input_shape = cfg.get("input_shape", [240, 320])
    _mean = torch.tensor(cfg.get("data_mean", [0.5, 0.5, 0.5]), dtype=torch.float32)
    _std = torch.tensor(cfg.get("data_std", [0.5, 0.5, 0.5]), dtype=torch.float32)
    H_in, W_in = input_shape
    frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()  # [T, C, OH, OW]
    frames_t = TF.resize(frames_t, [H_in, W_in], antialias=True)         # [T, C, H_in, W_in]
    frames_t = (frames_t / 255.0 - _mean.view(1, 3, 1, 1)) / _std.view(1, 3, 1, 1)

    # → [1, C, T, H, W]
    vd = frames_t.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
    v_len = n_frames
    VCL = v_len   # use full video (the overfit loop iterates from NF to VCL)
    B = 1

    di = torch.zeros(B, 11, device=DEVICE)
    di[:, 0] = v_len
    di[:, 1] = 0
    di[:, 2] = a_start
    di[:, 3] = a_end
    di[:, 4] = 1

    print(f"  Tensor: {tuple(vd.shape)}  VCL={VCL}  valid_frames={VCL - NF}")
else:
    # --- Synthetic data ---
    VCL = config_VCL
    B = 2
    vd = torch.randn(B, 3, VCL, input_shape[0], input_shape[1], device=DEVICE)
    di = torch.zeros(B, 11, device=DEVICE)
    di[:, 0] = VCL
    di[:, 1] = torch.arange(B)
    di[:, 2] = -1
    di[:, 3] = -1
    di[:, 4] = -1
    for b in range(B // 2, B):
        di[b, 2] = 4
        di[b, 3] = 6
        di[b, 4] = 1

    print(f"\nSynthetic: {B} videos, {VCL} frames, labels={di[:, 4].int().tolist()}")
    v_len = VCL

n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
print(f"Trainable params in head: {n_params:,}")

# --- Feature diversity check ---
MAX_PAIRWISE = 2000

def _is_vjepa(enc):
    # V-JEPA ViT returns temporally-multi-packed tokens ([B, n_temp*spat*spat, D])
    # and exposes _vjepa_n_temp on the head.  Swin / VideoMAE encoders return
    # plain spatial token grids (no temporal dim to fold), so they pass through.
    from vjepa_encoder import VJEPA2Encoder
    return isinstance(enc, VJEPA2Encoder)

def _temporal_avg(feat):
    enc = head.encoder
    if not _is_vjepa(enc):
        return feat
    n_temp = head._vjepa_n_temp
    return feat.reshape(feat.shape[0], n_temp, -1, feat.shape[-1]).mean(dim=1)

def _sim_report(t, label):
    flat = t.reshape(-1, t.shape[-1])
    n = flat.shape[0]
    if n > MAX_PAIRWISE:
        idx = torch.randperm(n, device=flat.device)[:MAX_PAIRWISE]
        flat = flat[idx]
        n_label = f"sampled {MAX_PAIRWISE}/{n}"
    else:
        n_label = str(n)
    sims = []
    chunk = 128
    for i in range(0, flat.shape[0], chunk):
        q = flat[i : i + chunk]
        s = F.cosine_similarity(q.unsqueeze(1), flat.unsqueeze(0), dim=-1)
        sims.append(s)
    sims = torch.cat(sims, dim=0)
    mask = ~torch.eye(sims.shape[0], dtype=torch.bool, device=sims.device)
    off = sims[mask]
    if off.numel() == 0:
        print(f"  {label}: tokens={n_label}  sim=N/A (single token — no pairs)")
        return
    print(f"  {label}: tokens={n_label}  sim=[{off.min():.4f}, {off.max():.4f}]  mean={off.mean():.4f}")

with torch.no_grad():
    # Encode first clip to check feature diversity
    clip_check = vd[:, :, :NF, :, :]
    patches_check = head.encoder(clip_check, return_patches=True)
    patches_check = _temporal_avg(patches_check)
    _sim_report(patches_check, "patch tokens")
    _sim_report(patches_check.mean(dim=1, keepdim=True), "pooled (mean)")

# --- Overfit ---
toa_batch = di[:, 2]
tea_batch = di[:, 3]
video_len_orig = di[:, 0]

for epoch in range(args.epochs):
    state = None
    epoch_loss = 0.0
    frame_count = 0

    for i in range(NF, VCL):
        target = gt_cls_target(i, toa_batch, tea_batch).long()

        # MOVAD-style: full clip → full model forward in one call
        clip = vd[:, :, i - NF:i, :, :]          # [B, C, NF, H, W]
        with autocast_ctx:
            output, state = head(clip, state)      # encoder → projection → temporal → classifier

        flt = i >= video_len_orig
        target[flt] = -100
        output[flt] = -100

        if apply_softmax:
            output = output.softmax(dim=1)

        loss = criterion(output, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.detach().item()
        frame_count += 1

    avg_loss = epoch_loss / max(frame_count, 1)

    if epoch == 0 or (epoch + 1) % 10 == 0:
        with torch.no_grad():
            state_p = None
            preds = []
            for i in range(NF, VCL):
                clip = vd[:, :, i - NF:i, :, :]
                with autocast_ctx:
                    out, state_p = head(clip, state_p)
                if cfg.get("apply_softmax", False):
                    out = out.softmax(dim=1)
                preds.append(out[:, 1].cpu())
            all_preds = torch.stack(preds, dim=1)

        grads_ok = 0
        for p in head.parameters():
            if p.grad is not None and p.grad.norm().item() > 1e-10:
                grads_ok += 1
        print(f"  ep {epoch+1:3d}: loss={avg_loss:.6f}"
              f"  pred_range=[{all_preds.min():.4f}, {all_preds.max():.4f}]"
              f"  mean={all_preds.mean():.4f}  n_frames={all_preds.shape[1]}  params_w_grad={grads_ok}")

final_loss = avg_loss
learned = abs(final_loss - 0.693) > 0.02 and final_loss < 0.65
status = "✓ LEARNED" if learned else "✗ STUCK"
print(f"\n=> {status}  final_loss={final_loss:.6f}")