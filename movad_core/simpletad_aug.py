"""
simple-tad DoTA augmentation — vendored, per-frame, on ``[T, H, W, C]`` float arrays.

Mirrors ``run_frame_finetuning`` + ``dota.py`` for DoTA training:

  1. ``_aug_frame`` ("DRIVE_TRANSFORMS", Cinemagraphic style) — RandAugment over
     auto_augment policy, applied to PIL frames converted back to per-frame.
  2. RandomHorizontalFlip / RandomVerticalFlip (MOVAD's existing flippers).
  3. ColorJitter / AutoAugment (VideoMAE timm recipe).
  4. RandomErasing.

The recipe is faithful to simple-tad's DoTA config::

    --aa rand-m6-n3-mstd0.5-inc1    (rand augment)
    --color_jitter 0.4              (used if auto_augment is None)
    --reprob 0.25 --remode pixel --recount 1   (random erasing)
    mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]

Operates on float32 ``[T,H,W,C]`` frame arrays (the same format movad's
``Dota.__getitem__`` produces).  Each frame is treated as an independent image,
which is what simple-tad does per-view.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Hand-rolled pieces (avoid importing the whole simple-tad video_transforms.py)
# ---------------------------------------------------------------------------
def _pad_wide_clips(h: int, w: int, crop_size: int):
    """Pad the two-dimensional video (vertical) to square then resize to
    ``crop_size`` — simple-tad's ``video_transforms.pad_wide_clips``.
    Operates on a single frame ``[H, W, C]``.
    """
    _PAD_MODES = (
        None, None, None, None, None,
        "black", "black",
        "color",
        "reflect", "reflect",
        "replicate", "replicate",
    )

    def _do_pad(x, alpha, pad_top, pad_bottom):
        reflect_padded = cv2.copyMakeBorder(
            x, pad_top, pad_bottom, 0, 0, cv2.BORDER_REFLECT)
        black_padded = cv2.copyMakeBorder(
            x, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        blended = cv2.addWeighted(reflect_padded, alpha, black_padded, 1 - alpha, 0)
        return cv2.resize(
            blended, dsize=(crop_size, crop_size), interpolation=cv2.INTER_CUBIC)

    choice = torch.randint(0, len(_PAD_MODES), (1,)).item()
    padding_mode = _PAD_MODES[choice]
    h_to_sq = w - h
    if padding_mode is not None and h_to_sq > 0:
        pad_top = int(round(torch.rand(1).item() * 0.5 * h_to_sq))
        pad_bottom = int(round(torch.rand(1).item() * 0.5 * h_to_sq))
        alpha = torch.rand(1).item() * 0.7
        if padding_mode == "reflect":
            return lambda x: _do_pad(x, alpha, pad_top, pad_bottom)
        if padding_mode == "replicate":
            return lambda x: cv2.resize(
                cv2.copyMakeBorder(x, pad_top, pad_bottom, 0, 0, cv2.BORDER_REPLICATE),
                dsize=(crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        # black / color
        color = torch.randint(0, 256, (3,)).tolist() if padding_mode == "color" else [0, 0, 0]
        return lambda x: cv2.resize(
            cv2.copyMakeBorder(x, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=color),
            dsize=(crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
    else:
        return lambda x: cv2.resize(
            x, dsize=(crop_size, crop_size), interpolation=cv2.INTER_CUBIC)


_DRIVE_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
]


def _pil_interp(method: str):
    m = method.lower()
    if m == "bicubic":
        return Image.BICUBIC
    if m == "lanczos":
        return Image.LANCZOS
    if m == "hamming":
        return Image.HAMMING
    return Image.BILINEAR


def _tensor_normalize(t: torch.Tensor, mean, std) -> torch.Tensor:
    """In-place ImageNet normalize on a float tensor of any shape ``(..., C)``."""
    mean = torch.tensor(mean, dtype=t.dtype, device=t.device)
    std = torch.tensor(std, dtype=t.dtype, device=t.device)
    return t.sub_(mean).div_(std)


class SimpleTadAugmentTrain(torch.nn.Module):
    """Full simple-tad DoTA **training** (augmentation) transform pipeline.

    Input :  ``[T, H, W, C]`` float32 frame array (uint8/255 normalized later).
    Output:  ``[T, C, H, W]`` float32 tensor that the model consumes directly
    (already ImageNet-normalized), i.e. the same shape movad expects after its
    own ``transforms`` + ``permute``.
    """

    def __init__(
        self,
        crop_size: int = 224,
        auto_augment: str = "rand-m6-n3-mstd0.5-inc1",
        color_jitter: float = 0.4,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        re_prob: float = 0.25,
    ):
        super().__init__()
        self.crop_size = int(crop_size)
        self.auto_augment = auto_augment
        self.color_jitter = float(color_jitter)
        self.mean = mean
        self.std = std
        self.re_prob = re_prob

        import timm
        import timm.data.auto_augment as _timm_aa

        self._rand_aug = None
        if auto_augment and str(auto_augment).startswith("rand"):
            aa_params = {"translate_const": int(crop_size * 0.45)}
            # timm expects a Compose; build once (frame-agnostic policy).
            policy = _timm_aa.rand_augment_transform(
                auto_augment, aa_params, _DRIVE_TRANSFORMS)
            self._rand_aug = policy
        elif color_jitter is not None:
            self._color_jitter = transforms.ColorJitter(
                color_jitter, color_jitter, color_jitter, 0.0)

        # torchvision v1 RandomErasing has no `mode`/`max_count`. simple-tad's
        # remode=pixel/recount=1 → erase with a constant 0 (pixel), one patch.
        self._erase = transforms.RandomErasing(
            p=re_prob, scale=(0.02, 0.1), ratio=(0.3, 3.3),
            value=0.0, inplace=False,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``[T, H, W, C]`` float32 → ``[T, C, H, W]`` normalized."""
        T, H, W, C = frames.shape
        crop_size = self.crop_size
        pad_fn = _pad_wide_clips(H, W, crop_size)

        # --- per-frame pad-to-square + RandAugment, then ToTensor ---
        out = []
        for i in range(T):
            f = frames[i].numpy()            # [H, W, C] float32 0..255
            f = pad_fn(f)                     # [crop, crop, C]
            pil = Image.fromarray(f.astype(np.uint8))
            if self._rand_aug is not None:
                t = self._rand_aug(pil)
            else:
                t = self._color_jitter(pil)
            t = transforms.ToTensor()(t)      # [C, H, W] 0..1
            out.append(t)
        x = torch.stack(out, dim=0)           # [T, C, H, W] 0..1

        # --- transpose → [T, H, W, C], normalize, transpose back ---
        x = x.permute(0, 2, 3, 1).contiguous()
        x = _tensor_normalize(x, self.mean, self.std)
        x = x.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]

        # --- random erasing (applied on the T,C,H,W + first-frame axis) ---
        if self.re_prob > 0:
            x = self._erase(x)

        return x


class SimpleTadAugmentVal(torch.nn.Module):
    """Val/test pipeline: single square crop to ``crop_size`` + ImageNet norm.
    Matches simple-tad ``Resize((crop,crop))`` + ``ClipToTensor`` + ``Normalize``.
    """

    def __init__(self, crop_size: int = 224, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.crop_size = int(crop_size)
        self.mean = mean
        self.std = std

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``[T, H, W, C]`` float32 → ``[T, C, H, W]`` normalized."""
        T, H, W, C = frames.shape
        resize = _pad_wide_clips(H, W, self.crop_size)
        out = []
        for i in range(T):
            f = cv2.resize(
                frames[i].numpy(), dsize=(self.crop_size, self.crop_size),
                interpolation=cv2.INTER_CUBIC)
            t = Image.fromarray(f.astype(np.uint8))
            t = transforms.ToTensor()(t)
            out.append(t)
        x = torch.stack(out, dim=0)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = _tensor_normalize(x, self.mean, self.std)
        return x.permute(0, 3, 1, 2).contiguous()