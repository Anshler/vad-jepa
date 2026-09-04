"""
VideoMAE ViT-Small encoder (backbone only) — vendored from simple-tad.

Mirrors ``modeling_pretrain.PretrainVisionTransformerEncoder`` (VideoMAE
architecture) so that a DAPT checkpoint's `encoder.*` weights load 1:1.

VideoMAE tokenizes a video ``[B, 3, T, H, W]`` with a tubelet Conv3d
(tubelet_size, patch, patch) → ``[B, N, E]`` tokens, adds a *fixed sinusoidal*
positional embedding (no CLS token), runs ``depth`` vanilla transformer
blocks, and applies a final LayerNorm.  No MAE decoder — encoder only (for
MOVAD, the decoder head is unused/discarded).

Sizes for ViT-Small: embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.
Dependencies: torch + numpy only (same as the rest of movad).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    """Fixed sin-cos positional embedding table (VideoMAE / BEiT style).

    Returns ``[1, n_position, d_hid]`` (batch dim prepended), non-learnable.
    """
    def get_position_angle_vec(position: int) -> list[float]:
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])   # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])   # dim 2i+1
    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False
    ).unsqueeze(0)


class DropPath(nn.Module):
    """Stochastic depth (drop a whole residual path with prob ``drop_prob``)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        binary_tensor = random_tensor.floor_()
        return x / keep_prob * binary_tensor


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """VideoMAE fused-bias attention.

    When ``qkv_bias=True`` the q and v projections get *separate* bias vectors
    (``q_bias``, ``v_bias``), concatenated onto the (bias-free) ``qkv.weight``
    at runtime — the VideoMAE pretrain state_dict layout:
        blocks.N.attn.qkv.weight, blocks.N.attn.q_bias, blocks.N.attn.v_bias
    This differs from a plain ``nn.Linear(dim, 3*dim, bias=True)`` layout, hence
    the manual reconstruction below.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _qkv_bias(self) -> torch.Tensor | None:
        if self.q_bias is None:
            return None
        return torch.cat(
            (self.q_bias, torch.zeros_like(self.v_bias), self.v_bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        bias = self._qkv_bias()
        qkv = F.linear(x, weight=self.qkv.weight, bias=bias) \
            if bias is not None else self.qkv(x)
        qkv = (
            qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Conv3d tubelet patch embedding for videos.

    ``[B, C, T, H, W]`` → ``[B, N, E]``  where
    ``N = (T/tubelet) * (H/patch) * (W/patch)``.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        num_frames: int = 16,
        tubelet_size: int = 2,
    ):
        super().__init__()
        self.tubelet_size = int(tubelet_size)
        num_patches = (
            (img_size // patch_size)
            * (img_size // patch_size)
            * (num_frames // self.tubelet_size)
        )
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size, patch_size),
            stride=(self.tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, N, E]
        return x


class VideoMAEVisionEncoder(nn.Module):
    """VideoMAE ViT-Small encoder (backbone only, no MAE decoder).

    Mirrors ``PretrainVisionTransformerEncoder`` field-for-field so a DAPT
    checkpoint's ``encoder.*`` submodule loads straight in::

        state_dict["model"].encoder.patch_embed.proj.weight  → self.patch_embed.proj.weight
        state_dict["model"].encoder.blocks.N.*               → self.blocks.N.*
        state_dict["model"].encoder.norm.*                   → self.norm.*
        state_dict["model"].encoder.pos_embed                → self.pos_embed (buffer)

    Input :  ``[B, 3, T, H, W]``  video clip (0..1 or 0..255, normalized upstream).
    Output:  ``[B, N, E]``  patch tokens (post LayerNorm), or pooled ``[B, E]``.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        num_frames: int = 16,
        tubelet_size: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, num_frames=num_frames, tubelet_size=tubelet_size,
        )
        self.num_patches = self.patch_embed.num_patches

        # Fixed sinusoidal positional embedding (VideoMAE/BEiT).  Registered as a
        # buffer so it appears in state_dict (key "pos_embed") and loads from ckpt.
        self.register_buffer(
            "pos_embed",
            get_sinusoid_encoding_table(self.num_patches, embed_dim),
            persistent=False,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 3, T, H, W]`` → ``[B, N, E]`` patch tokens (post-LayerNorm)."""
        x = self.patch_embed(x)                       # [B, N, E]
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                              # LayerNorm over feature dim
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pool over tokens → ``[B, E]`` (VideoMAE ``fc_norm``-style)."""
        return self.forward_tokens(x).mean(dim=1)