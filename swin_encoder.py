"""
Swin Transformer 3D encoder for vad-jepa — MOVAD paper parity.

Architecture matches the MOVAD paper (Video Swin-B + AdaptiveAvgPool3d projection):
    Swin → AdaptiveAvgPool3d((1,6,6)) → [B, C, 1, 6, 6]
    ─ return_patches=True  → flatten spatial → [B, 36, C]  (grid cells as "patches")
    ─ return_patches=False → flatten → LN → Linear(C×36→C) → ReLU → Dropout → [B, C]

Usage (in config YAML):
    model_name: swin_base_ssv2
    input_shape: [240, 320]
    num_frames: 4
    checkpoint_path: pretrained/swin_base_patch244_window1677_sthv2.pth
    dropout: 0.3
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from video_swin_transformer import SwinTransformer3D


# ---------------------------------------------------------------------------
# Swin variant presets  (matching MOVAD src/models.py build_model_cfg)
# ---------------------------------------------------------------------------
_SWIN_PRESETS = {
    "swin_t": {
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "patch_size": (4, 4, 4),
        "window_size": (7, 7, 7),
        "drop_path_rate": 0.1,
        "patch_norm": True,
    },
    "swin_s": {
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
        "patch_size": (4, 4, 4),
        "window_size": (7, 7, 7),
        "drop_path_rate": 0.2,
        "patch_norm": True,
    },
    "swin_b": {
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "patch_size": (4, 4, 4),
        "window_size": (7, 7, 7),
        "drop_path_rate": 0.3,
        "patch_norm": True,
    },
    "swin_l": {
        "embed_dim": 192,
        "depths": [2, 2, 18, 2],
        "num_heads": [6, 12, 24, 48],
        "patch_size": (4, 4, 4),
        "window_size": (7, 7, 7),
        "drop_path_rate": 0.3,
        "patch_norm": True,
    },
    # Pretrained on Something-Something V2 (the main MOVAD backbone)
    "swin_base_patch244_window1677_sthv2": {
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "patch_size": (2, 4, 4),           # temporal stride 2, spatial stride 4
        "window_size": (16, 7, 7),         # large temporal window for SSv2
        "drop_path_rate": 0.3,
        "patch_norm": True,
    },
    "swin_base_patch4_window7_224_22k": {
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "patch_size": (4, 4, 4),
        "window_size": (7, 7, 7),
        "drop_path_rate": 0.3,
        "patch_norm": True,
    },
}


def _resolve_preset(model_name: str) -> tuple[str, dict]:
    """Map a short ``model_name`` (e.g. ``swin_base_ssv2``) to a ``transformer_type``
    and its kwargs dict."""
    # Direct passthrough — if model_name matches a transformer_type key
    if model_name in _SWIN_PRESETS:
        return model_name, _SWIN_PRESETS[model_name]

    # Shortcut aliases
    aliases = {
        "swin_tiny":    "swin_t",
        "swin_small":   "swin_s",
        "swin_base_ssv2": "swin_base_patch244_window1677_sthv2",
    }
    tt = aliases.get(model_name)
    if tt is not None and tt in _SWIN_PRESETS:
        return tt, _SWIN_PRESETS[tt]

    raise ValueError(
        f"Unknown Swin model_name '{model_name}'. "
        f"Available: {list(_SWIN_PRESETS.keys())} | aliases: {list(aliases.keys())}"
    )


# =============================================================================
# Swin Encoder
# =============================================================================
class SwinEncoder(nn.Module):
    """Swin Transformer 3D encoder matching the MOVAD paper (Video Swin).

    Input
    -----
    x : ``[B, 3, T, H, W]``  video clip.

    Output (when ``return_patches=True``)
    -------------------------------------
    ``[B, 36, embed_dim]``  AdaptiveAvgPool3d((1,6,6)) grid cells as "patches"
    for the MOVAD projection path in ClsVJEPA.

    Output (when ``return_patches=False``)
    --------------------------------------
    ``[B, embed_dim]``  global feature via full MOVAD projection:
    AdaptiveAvgPool3d((1,6,6)) → flatten → LN → Linear(C×36→C) → ReLU → Dropout.
    """

    def __init__(self, swin: SwinTransformer3D, dropout: float = 0.3):
        super().__init__()
        self.swin = swin
        self.embed_dim: int = swin.num_features
        self.add_module("swin", swin)

        # MOVAD-style adaptive pooling: preserves a 6×6 spatial grid
        self.avgpool = nn.AdaptiveAvgPool3d((1, 6, 6))

        # MOVAD projection layers (used in return_patches=False)
        self.proj_norm = nn.LayerNorm(self.embed_dim * 36)
        self.proj = nn.Linear(self.embed_dim * 36, self.embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def num_patches(self) -> int:
        """Number of spatial grid cells (matches MOVAD: 36)."""
        return 36

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        # x: [B, 3, T, H, W]
        feat = self.swin(x)                          # [B, C, D, H, W]
        feat = self.avgpool(feat)                    # [B, C, 1, 6, 6]

        if return_patches:
            B, C, D, H, W = feat.shape
            # Flatten D×H×W to grid cells → [B, 36, C]
            return feat.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)

        # MOVAD projection: avgpool → flatten → LN → Linear → ReLU → Dropout
        feat = feat.flatten(1)                       # [B, C×36]
        feat = self.proj_norm(feat)                  # [B, C×36]
        feat = F.relu(self.proj(feat))               # [B, C]
        feat = self.proj_drop(feat)                  # [B, C]
        return feat


# =============================================================================
# Builder
# =============================================================================
def build_swin_encoder(cfg) -> SwinEncoder:
    """Build a SwinEncoder from a MOVAD-style EasyDict config.

    Required config keys
    --------------------
    model_name : str
        One of the Swin presets (e.g. ``swin_base_ssv2``).
    input_shape : list[int] | None
        ``[H, W]`` spatial resolution. Defaults to ``[256, 256]`` if not set.
    num_frames : int
        Frames per clip / NF.  Must be ≥ 16 for Swin-B with 4 stages.
    dropout : float
        Drop path rate propagated to ``drop_path_rate``.  Default 0.3.
    pretrained : str | None
        Path to pretrained checkpoint.
    pretrained2d : bool
        If True, inflate 2D weights to 3D.  Default False.
    transformer_type : str | None
        Override the preset key (e.g. ``swin_base_patch244_window1677_sthv2``).
    """
    model_name = cfg.get("model_name", "swin_base_ssv2")
    tt, preset = _resolve_preset(model_name)

    # Allow explicit override of the preset key
    tt = cfg.get("transformer_type", tt)

    # Build the Swin backbone
    swin = SwinTransformer3D(
        pretrained=None,       # we load manually below
        pretrained2d=False,
        patch_size=preset["patch_size"],
        in_chans=3,
        embed_dim=preset["embed_dim"],
        depths=preset["depths"],
        num_heads=preset["num_heads"],
        window_size=preset["window_size"],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=cfg.get("dropout", 0.3),  # MOVAD wires dropout → drop_path_rate
        norm_layer=nn.LayerNorm,
        patch_norm=preset["patch_norm"],
        frozen_stages=-1,
        use_checkpoint=False,
    )

    # Load pretrained 3D checkpoint (Something-Something V2 format)
    pretrained_path = cfg.get("checkpoint_path", None)
    if pretrained_path is not None:
        import os
        if os.path.isfile(pretrained_path):
            _load_swin_checkpoint(swin, pretrained_path)
        else:
            print(f"[SwinEncoder] WARNING: pretrained checkpoint not found at "
                  f"'{pretrained_path}' — using random initialization.")

    return SwinEncoder(swin, dropout=cfg.get("dropout", 0.3))


def _load_swin_checkpoint(swin: SwinTransformer3D, path: str) -> None:
    """Load a 3D Swin checkpoint, stripping a ``backbone.`` prefix if present.

    The Something-Something V2 pretrained checkpoint uses:
        {'state_dict': {'backbone.patch_embed.proj.weight': ..., ...}}
    """
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            k = k[9:]   # strip "backbone." prefix
        new_state_dict[k] = v

    missing, unexpected = swin.load_state_dict(new_state_dict, strict=False)
    n_missing = len(missing)
    n_unexpected = len(unexpected)
    if n_missing > 0 or n_unexpected > 0:
        print(f"[SwinEncoder] Loaded pretrained from {path} "
              f"(missing={n_missing}, unexpected={n_unexpected})")
    else:
        print(f"[SwinEncoder] Loaded pretrained from {path}")
