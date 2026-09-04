"""
Frozen DAPT-VideoMAE-S encoder wrapper.

Loads a simple-tad DAPT checkpoint (VideoMAE ViT-Small, domain-adapted on
BDD100K + CAP-DATA — dashcam, never fine-tuned on DoTA), keeps only the
encoder backbone (discards the MAE decoder), and exposes a clean
``(clip) -> pooled_features | patch_tokens`` interface, matching the
``VJEPA2Encoder`` / ``SwinEncoder`` conventions used elsewhere in movad.

Checkpoint key layout (saved by ``run_mae_double_pretraining.py``)::

    checkpoint["model"].encoder.patch_embed.*   -> encoder.patch_embed.*
    checkpoint["model"].encoder.blocks.*        -> encoder.blocks.*
    checkpoint["model"].encoder.norm.*          -> encoder.norm.*
    checkpoint["model"].encoder.pos_embed       -> encoder.pos_embed (buffer)
    checkpoint["model"].decoder.* / encoder_to_decoder / mask_token (discarded)

We strip ``module.``/``model.``/``encoder.`` prefixes (the checkpoint is
already nested under ``encoder.*``, mirroring ``PretrainVisionTransformer``).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from video_mae_transformer import VideoMAEVisionEncoder


def load_pretrained_videomae(
    img_size: int = 224,
    num_frames: int = 16,
    tubelet_size: int = 2,
    patch_size: int = 16,
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    checkpoint_path: str | None = None,
    checkpoint_key: str = "model",
    device: str | torch.device = "cuda",
    **model_kwargs,
) -> VideoMAEVisionEncoder:
    """Build a VideoMAE ViT-Small encoder and load DAPT pretrained weights.

    The checkpoint is a full ``PretrainVisionTransformer`` (encoder + decoder).
    Only the encoder submodule is loaded; decoder/`encoder_to_decoder`/
    `mask_token` weights are dropped.
    """
    encoder = VideoMAEVisionEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        **model_kwargs,
    )

    if checkpoint_path is not None:
        print(f"Loading DAPT-VideoMAE-S checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt
        if isinstance(ckpt, dict) and checkpoint_key in ckpt:
            state_dict = ckpt[checkpoint_key]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif not isinstance(ckpt, dict):
            raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

        # Strip leading prefixes common to saved DAPT / DDP checkpoints.
        clean = {}
        for k, v in state_dict.items():
            for prefix in ("module.encoder.", "encoder.", "module.", "model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            # Keep only encoder-backbone keys (drop any leftover nesting).
            if k.startswith("patch_embed.") or k.startswith("blocks.") \
                    or k.startswith("norm.") or k == "pos_embed":
                clean[k] = v

        # Strict first — fail loudly rather than silently random-init.
        try:
            missing, unexpected = encoder.load_state_dict(clean, strict=True)
            if missing or unexpected:
                print(f"  strict load: missing={missing}\n"
                      f"               unexpected={unexpected}")
            else:
                print("  loaded with strict=True")
        except RuntimeError as e:
            print(f"  strict load failed: {e}")
            print("  falling back to shape-matched loading …")
            encoder_state = encoder.state_dict()
            for k, v in encoder_state.items():
                if k not in clean:
                    print(f'    key "{k}" not found — keeping random init')
                elif clean[k].shape != v.shape:
                    print(f'    key "{k}" shape mismatch: '
                          f'checkpoint {clean[k].shape} vs model {v.shape}')
            msg = encoder.load_state_dict(clean, strict=False)
            print(f"  loaded with msg: {msg}")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


class VideoMAEEncoder(nn.Module):
    """
    Frozen DAPT-VideoMAE-S encoder.

    Input : ``[B, 3, T, H, W]`` video clip.

    ``return_patches=False`` → ``[B, embed_dim]``   mean-pooled feature vector.
    ``return_patches=True``  → ``[B, N, embed_dim]`` full patch tokens (SlotSSM).
    """

    def __init__(self, encoder: VideoMAEVisionEncoder, pool: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.embed_dim: int = encoder.embed_dim
        self.pool = pool
        self.add_module("encoder", encoder)

    @property
    def num_patches(self) -> int:
        return self.encoder.num_patches

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        z = self.encoder.forward_tokens(x)        # [B, N, embed_dim]
        if return_patches:
            return z
        if self.pool == "mean":
            return z.mean(dim=1)
        if self.pool == "cls":
            return z[:, 0, :]
        raise ValueError(f"Unknown pool mode: {self.pool}")


def build_videomae_encoder(cfg) -> VideoMAEEncoder:
    """Build a VideoMAEEncoder from a MOVAD-style EasyDict config."""
    raw = load_pretrained_videomae(
        img_size=cfg.get("img_size", 224),
        num_frames=cfg.get("num_frames", 16),
        tubelet_size=cfg.get("tubelet_size", 2),
        patch_size=cfg.get("patch_size", 16),
        embed_dim=cfg.get("embed_dim", 384),
        depth=cfg.get("depth", 12),
        num_heads=cfg.get("num_heads", 6),
        checkpoint_path=cfg.get("checkpoint_path", None),
        checkpoint_key=cfg.get("checkpoint_key", "model"),
        device=cfg.get("device", "cuda"),
    )
    return VideoMAEEncoder(raw, pool=cfg.get("pool", "mean"))