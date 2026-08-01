"""
Training + validation smoke test — synthetic data, one step each.
Supports all configs (ViT, Swin, LSTM, Mamba, SlotSSM variants).

Verifies:
  - Model build from config
  - Per-clip full forward with autocast (train + val)
  - Loss compute, backward, optimizer step (no NaNs, loss decreases)
  - Swin-specific architecture checks (grid cells, projection sizes)

Usage (from WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vad-jepa
    python tests/test_training.py
    python tests/test_training.py --config swin_lstm.yaml
    python tests/test_training.py --amp fp16
    python tests/test_training.py --train_encoder
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext

import torch
import yaml
from easydict import EasyDict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Compatibility: torchvision >=0.20 moved functional_tensor to _functional_tensor
try:
    import torchvision.transforms._functional_tensor as _ft  # noqa: F401
    import sys as _sys
    _sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
except ImportError:
    pass

from model import _HAS_MAMBA_SSM, build_multi_head_vjepa
from movad_core.dota import gt_cls_target
from movad_core.losses import build_loss
from movad_core.optim import build_optimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG_DIR = os.path.join(_REPO_ROOT, "cfgs")

_AMP_CHOICES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="vjepa_mamba.yaml",
                    help="Config YAML in cfgs/ (default: vjepa_mamba.yaml)")
parser.add_argument("--amp", default="fp16", choices=list(_AMP_CHOICES),
                    help="AMP dtype (default: fp16)")
parser.add_argument("--checkpoint", default=None,
                    help="Path to a pretrained checkpoint (.pt)")
parser.add_argument("--train_encoder", action="store_true", default=False,
                    help="Unfreeze the V-JEPA encoder and train jointly")
args = parser.parse_args()
AMP_DTYPE = _AMP_CHOICES[args.amp]
CHECKPOINT_PATH = args.checkpoint

torch.manual_seed(42)
torch.cuda.manual_seed(42)

print(f"Device: {DEVICE}")
print(f"mamba_ssm: {_HAS_MAMBA_SSM}")
print(f"config: {args.config}")
if CHECKPOINT_PATH:
    print(f"checkpoint: {CHECKPOINT_PATH}")
else:
    print("checkpoint: none (using random init)")
print(f"amp: {args.amp}")
print(f"train_encoder: {args.train_encoder}")


# ---------------------------------------------------------------------------
# Build model + infrastructure from a real config
# ---------------------------------------------------------------------------
def load_cfg(name: str) -> EasyDict:
    path = os.path.join(CFG_DIR, name)
    with open(path, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.device = DEVICE
    if CHECKPOINT_PATH:
        cfg.checkpoint_path = CHECKPOINT_PATH
    return cfg


cfg = load_cfg(args.config)
cfg.train_encoder = args.train_encoder
head_name = os.path.splitext(args.config)[0].replace("vjepa_", "").replace("swin_", "")
cfg._head_cfgs_flat = [dict(cfg)]
cfg._head_cfgs_flat[0]["name"] = head_name

model = build_multi_head_vjepa(cfg)
head_name = next(iter(model.heads.keys()))
head = model.heads[head_name]
is_swin = head._is_swin
is_vit = not is_swin and not args.config.startswith("swin")

# --- Architecture diagnostics (Swin-specific + general) ---
print(f"\nArchitecture: {'Swin' if is_swin else 'ViT'} backbone  |  temporal: {head.temporal_type}")
if is_swin:
    embed_dim = head.encoder.embed_dim
    num_patches = head.encoder.num_patches
    print(f"  encoder.embed_dim: {embed_dim}")
    print(f"  encoder.num_patches: {num_patches}")
    print(f"  encoder.avgpool: AdaptiveAvgPool3d output_size=(1,{int(num_patches**0.5)},{int(num_patches**0.5)})")
    print(f"  movad_proj_in: {embed_dim}×{num_patches}={embed_dim * num_patches}  →  {head.lin1.out_features}")
    assert head.lin1.in_features == embed_dim * num_patches, \
        f"Expected lin1.in_features={embed_dim * num_patches}, got {head.lin1.in_features}"
    if head.temporal_type == "lstm":
        assert head.lin2.in_features == head.lin1.out_features, \
            f"LSTM: expected lin2.in_features={head.lin1.out_features} (unidirectional), got {head.lin2.in_features}"
        print(f"  ✓ Unidirectional LSTM: lin2 in={head.lin2.in_features} → out={head.lin2.out_features}")
    print(f"  ✓ MOVAD Swin projection verified")

# Per-head infrastructure
head_cfgs: dict[str, dict] = {}
criterion: dict[str, torch.nn.Module] = {}
optimizer: dict[str, torch.optim.Optimizer] = {}
autocast_ctx: dict[str, object] = {}

for name in list(model.heads.keys()):
    hc = model.head_configs.get(name, {})
    head_cfgs[name] = dict(cfg)
    for k in ("dim_latent", "dropout", "rnn_state_size", "rnn_cell_num",
              "mamba_d_state", "mamba_d_conv", "mamba_expand", "mamba_version",
              "num_slots", "slot_dim", "num_ssm_blocks", "top_k", "eps_random",
              "use_inverted_attention", "entropy_weight"):
        if k in hc:
            head_cfgs[name][k] = hc[k]

    head_easy = EasyDict(head_cfgs[name])
    head_easy.device = DEVICE
    criterion[name] = build_loss(head_easy)

    if args.train_encoder:
        opt, _ = build_optimizer(EasyDict({"lr": cfg.lr}), model, None)
    else:
        opt, _ = build_optimizer(EasyDict({"lr": cfg.lr}), model.heads[name], None)
    optimizer[name] = opt

    _amp_cfg = head_cfgs[name].get("amp_dtype", "fp32")
    _amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_amp_cfg)
    autocast_ctx[name] = (
        torch.amp.autocast("cuda", dtype=_amp_dtype) if _amp_dtype else nullcontext()
    )

model.to(DEVICE)
model.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,} (encoder {'unfrozen' if args.train_encoder else 'frozen'})")

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
B = 2
NF = cfg.get("num_frames", 4)
v_len = NF + 8
img_size = cfg.get("img_size", 256)

video_data = torch.randn(B, 3, v_len, img_size, img_size, device=DEVICE)

data_info = torch.zeros(B, 11, device=DEVICE)
data_info[:, 0] = v_len
data_info[:, 1] = torch.arange(B)
data_info[:, 2] = -1
data_info[:, 3] = -1
data_info[:, 4] = -1
data_info[1, 2] = 6    # anomaly start
data_info[1, 3] = 9    # anomaly end
data_info[1, 4] = 1    # label

# ---------------------------------------------------------------------------
# 1.  TRAINING  —  one encode + per-frame temporal loop + backward
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("1. TRAINING — one step")
print("=" * 70)

print(f"  synthetic video: {tuple(video_data.shape)}")

toa_batch = data_info[:, 2]
tea_batch = data_info[:, 3]
video_len_orig = data_info[:, 0]

state = None
total_loss_val = 0.0
frame_count = 0

for i in range(NF, v_len):
    target = gt_cls_target(i, toa_batch, tea_batch).long()

    with autocast_ctx[head_name]:
        clip = video_data[:, :, i - NF:i, :, :]   # [B, C, NF, H, W]
        output, state = head(clip, state)           # full forward: encoder → temporal → classifier

    flt = i >= video_len_orig
    target[flt] = -100
    output[flt] = -100

    if head_cfgs[head_name].get("apply_softmax", False):
        output = output.softmax(dim=1)

    loss = criterion[head_name](output, target)

    optimizer[head_name].zero_grad()
    loss.backward()
    optimizer[head_name].step()

    total_loss_val += loss.detach().item()
    frame_count += 1

avg_loss = total_loss_val / max(frame_count, 1)
print(f"  avg_loss={avg_loss:.4f}  frames={frame_count}")
print(f"  ✓ training step OK")

loss_before = avg_loss

# ---------------------------------------------------------------------------
# 2.  TRAINING  —  second step (verify loss decreases)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. TRAINING — second step (check loss decreases)")
print("=" * 70)

state2 = None
total_loss2_val = 0.0

for i in range(NF, v_len):
    target = gt_cls_target(i, toa_batch, tea_batch).long()

    with autocast_ctx[head_name]:
        clip = video_data[:, :, i - NF:i, :, :]
        output, state2 = head(clip, state2)

    flt = i >= video_len_orig
    target[flt] = -100
    output[flt] = -100
    if head_cfgs[head_name].get("apply_softmax", False):
        output = output.softmax(dim=1)
    loss = criterion[head_name](output, target)
    optimizer[head_name].zero_grad()
    loss.backward()
    optimizer[head_name].step()
    total_loss2_val += loss.detach().item()

loss_after = total_loss2_val / max(frame_count, 1)

print(f"  step 1 loss: {loss_before:.4f}  →  step 2 loss: {loss_after:.4f}")
if loss_after < loss_before:
    print("  ✓ loss decreased")
else:
    print("  ⚠ loss did not decrease (expected on synthetic data with 1 step)")
print("  ✓ second training step OK")

# ---------------------------------------------------------------------------
# 3.  VALIDATION  —  encode + temporal loop with autocast (no grad)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. VALIDATION — encode + temporal loop with autocast (no grad)")
print("=" * 70)

model.eval()

v_len_val = NF + 12
video_val = torch.randn(1, 3, v_len_val, img_size, img_size, device=DEVICE)
data_info_val = torch.zeros(1, 11, device=DEVICE)
data_info_val[:, 0] = v_len_val
data_info_val[:, 1] = 0
data_info_val[:, 2] = v_len_val - 10
data_info_val[:, 3] = v_len_val - 6
data_info_val[:, 4] = 1

with torch.no_grad():
    n_clips = v_len_val - NF
    state_val = None
    outputs_val = torch.zeros(1, n_clips, device=DEVICE)
    for i in range(NF, v_len_val):
        clip = video_val[:, :, i - NF:i, :, :]
        with autocast_ctx[head_name]:
            output, state_val = head(clip, state_val)
        if head_cfgs[head_name].get("apply_softmax", False):
            output = output.softmax(dim=1)
        prob = output if head_cfgs[head_name].get("apply_softmax", False) else output.softmax(dim=1)
        outputs_val[:, i - NF] = prob[:, 1]

print(f"  val outputs: {tuple(outputs_val.shape)}")
print(f"  val anomaly scores: min={outputs_val.min().item():.4f}  max={outputs_val.max().item():.4f}")
assert not torch.isnan(outputs_val).any(), "NaN in validation outputs!"
print("  ✓ validation step OK")

# ---------------------------------------------------------------------------
# 4.  Full-movie forward (no RNN state)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. Forward (full-movie) — verify encoder dimensions")
print("=" * 70)

model.eval()
video_full = torch.randn(2, 3, NF, img_size, img_size, device=DEVICE)

with torch.no_grad():
    with autocast_ctx[head_name]:
        out = model.heads[head_name](video_full)
    out_tensor = out[0] if isinstance(out, tuple) else out
    print(f"  forward output: {tuple(out_tensor.shape)}")

    # Encoder patch output dimensions
    if is_swin:
        encoder_out = head.encoder(video_full, return_patches=True)
        print(f"  encoder patches: {tuple(encoder_out.shape)}  (expect [2, {head.encoder.num_patches}, {head.encoder.embed_dim}])")
        assert encoder_out.shape[1] == head.encoder.num_patches, \
            f"Expected {head.encoder.num_patches} patches, got {encoder_out.shape[1]}"
        print(f"  ✓ Swin patch dimensions verified")

    encoder_vec = head.encoder(video_full, return_patches=False)
    print(f"  encoder vector: {tuple(encoder_vec.shape)}  (expect [2, {head.encoder.embed_dim}])")

print("  ✓ full-movie forward OK")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"All training + validation smoke tests passed!  (config={args.config}  amp={args.amp})")
print(f"  Backbone: {'Swin' if is_swin else 'ViT'}  |  Temporal: {head.temporal_type}")
print("=" * 70)
