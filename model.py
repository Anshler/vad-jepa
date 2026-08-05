"""
MOVAD anomaly classifier with a frozen V-JEPA 2.1 encoder backbone.

Supports five temporal model variants:
  - ``lstm``           — 3-layer LSTM (original MOVAD design)
  - ``mamba``          — 3 Mamba SSM blocks (mamba_ssm package)
  - ``mamba3``         — 3 Mamba3 SSM blocks with RoPE (mamba_ssm package)
  - ``slotssm``        — modular slots, per-slot Mamba, cross+self-attention
  - ``sparse_slotssm`` — SlotSSM + top-k sparse gating

SlotSSM architecture follows the reference repo (NeurIPS 2024) 1:1:
  block = Norm→CrossAttn(inverted,multi-head)→+res → Norm→Mamba→+res → Norm→SelfAttn→+res

All variants target ~15–25M trainable parameters with a frozen V-JEPA encoder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa_encoder import VJEPA2Encoder, build_vjepa2_encoder
from swin_encoder import SwinEncoder, build_swin_encoder

# ---------------------------------------------------------------------------
# Mamba_ssm import
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba, Mamba2, Mamba3

    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False
    Mamba = None
    Mamba2 = None
    Mamba3 = None


def _require_mamba():
    if not _HAS_MAMBA_SSM:
        raise ImportError(
            "mamba_ssm is required for this temporal model. "
            "Install with: pip install mamba-ssm causal-conv1d"
        )


# ---------------------------------------------------------------------------
# Flash-attn import (optional — speeds up SlotSSM self/cross attention).
# Requires a flash-attn wheel built against the EXACT PyTorch + CUDA version.
# Pre-built wheels: https://github.com/Dao-AILab/flash-attention/releases
# ---------------------------------------------------------------------------
try:
    from flash_attn.modules.mha import MHA as FlashMHA

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False
    FlashMHA = None

# ---------------------------------------------------------------------------
# MambaCache — identical to the SlotSSM repo.
# ---------------------------------------------------------------------------
@dataclass
class MambaCache:
    seqlen_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MultiHeadAttention — from the SlotSSM reference repo (NeurIPS 2024).
#
# Supports *inverted* attention where softmax runs over the slot dimension
# instead of the feature dimension, forcing input features to compete for
# slot assignment.  This encourages slot specialization without auxiliary
# losses.  The reference repo uses this as the default (train.py:120).
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional inverted softmax.

    Standard:  softmax over source (features) — each slot picks features.
    Inverted:  softmax over target (slots) — features compete for slots.
    """

    def __init__(
        self, d_model, num_heads, dropout=0.0, inverted=False, bias=True,
        norm_over_input=True, epsilon=1e-5,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.inverted = inverted
        self.norm_over_input = norm_over_input
        self.epsilon = epsilon

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.proj_q = nn.Linear(d_model, d_model, bias=bias)
        self.proj_k = nn.Linear(d_model, d_model, bias=bias)
        self.proj_v = nn.Linear(d_model, d_model, bias=bias)
        self.proj_o = nn.Linear(d_model, d_model, bias=bias)

        # Populated during forward when inverted=True (see diagnostics in forward)
        self._slot_mass_min = torch.tensor(float("nan"))
        self._slot_mass_mean = torch.tensor(float("nan"))
        self._slot_usage_frac = torch.tensor(float("nan"))

    def forward(self, q, k, v):
        B, T, _ = q.shape
        _, S, _ = k.shape

        q_proj = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k_proj = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v_proj = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)

        q_proj = q_proj * (q_proj.shape[-1] ** (-0.5))
        attn = torch.matmul(q_proj, k_proj.transpose(-1, -2))

        if self.inverted:
            # Softmax over (head * target) → features compete over slots
            attn = F.softmax(attn.flatten(start_dim=1, end_dim=2), dim=1).reshape(
                B, self.num_heads, T, S,
            )
            # --- diagnostics: per-slot mass BEFORE re-normalization ----------
            # Raw softmax mass per slot (summed over heads and patches).
            # Each of S patches distributes 1.0 across h×T entries by inverted
            # softmax, so fair share = S/T.  Normalize to fraction of fair share:
            # 1.0 = exactly fair share, < 0.05 = at risk, < 1e-4 = dead.
            pre_norm_mass = attn.detach().sum(dim=(1, -1))           # [B, T]
            fair = S / T                                               # fair-share mass per slot
            mass_norm = pre_norm_mass / fair                           # [B, T], fair share = 1.0
            self._slot_mass_min = mass_norm.min()                      # worst slot fraction
            self._slot_mass_mean = mass_norm.mean()                    # avg across slots
            # Fraction of slots receiving at least 15% of fair share
            self._slot_usage_frac = (mass_norm > 0.15).float().mean()
            # -----------------------------------------------------------------
            if self.norm_over_input:
                attn = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)
        output = torch.matmul(attn, v_proj).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)
        return output


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------
def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _count_frozen(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if not p.requires_grad)


def _print_param_summary(encoder, temporal, classifier, extra_parts=None, spatial_pool=None):
    frozen = _count_frozen(encoder)
    sp_count = _count(spatial_pool) if spatial_pool is not None else 0
    trainable = _count(temporal) + _count(classifier) + sp_count
    if extra_parts:
        for _tag, mod in extra_parts:
            trainable += _count(mod) if mod is not None else 0

    print(f"  Frozen (encoder):     {frozen / 1e6:.1f}M")
    print(f"  Trainable (temporal): {_count(temporal) / 1e6:.2f}M")
    cls_label = f"  Trainable (classifier+proj{'+spatial_pool' if sp_count else ''}): {(_count(classifier) + sp_count) / 1e6:.2f}M"
    print(cls_label)
    if extra_parts:
        for tag, mod in extra_parts:
            n = _count(mod) if mod is not None else 0
            print(f"  Trainable ({tag}): {n / 1e6:.2f}M")
    print("  ---")
    print(f"  Total trainable:      {trainable / 1e6:.2f}M")
    print(f"  Total frozen:         {frozen / 1e6:.1f}M")


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------
def _weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)


# ===========================================================================
# Temporal model: identity (no temporal modelling — pure per-frame MLP probe)
# ===========================================================================
class NoTemporalModel(nn.Module):
    """Identity passthrough that returns no state.

    Used for diagnostic linear probing — replaces any recurrent/SSM model so
    each frame is classified independently.  If features carry discriminative
    signal, even this simple per-frame MLP will beat random.
    """

    def forward(self, x, state=None):
        return x, None


# ===========================================================================
# Temporal model: LSTM
# ===========================================================================
class LSTMTemporalModel(nn.Module):
    """Multi-layer LSTM — matches MOVAD's ``nn.LSTM(dim, hidden, num_layers)``.

    MOVAD uses unidirectional LSTM (default).  Set ``bidirectional=True`` if
    you want bidirectional (output dim becomes ``2 * hidden_size``, and
    ``ClsVJEPA.lin2`` is resized accordingly).
    """

    def __init__(
        self,
        dim: int,
        hidden_size: int,
        num_layers: int = 3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            dim, hidden_size, num_layers, bidirectional=bidirectional
        )
        self.norm = nn.LayerNorm(dim)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        # When True (default), (h, c) are detached EVERY clip-step — this is
        # truncated BPTT with horizon 1, which starves the LSTM of temporal
        # gradient.  The training loop sets this to False and detaches at
        # chunk boundaries instead, enabling real backprop-through-time.
        self.detach_every_step = True

    @property
    def output_dim(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, x, state=None):
        x = self.norm(x).unsqueeze(0)
        x, new_state = self.rnn(x, state)
        x = x.squeeze(0)
        hx, cx = new_state
        if self.detach_every_step:
            hx, cx = hx.detach(), cx.detach()
        return x, (hx, cx)


# ===========================================================================
# Temporal model: Mamba / Mamba2 / Mamba3 (streaming via MambaCache)
# ===========================================================================
class MambaTemporalModel(nn.Module):
    """Multi-block Mamba temporal model with streaming inference.

    Supports three Mamba versions via ``mamba_version``:

    - ``"mamba1"`` — original Mamba (Mamba-1 SSM with conv + SiLU)
    - ``"mamba2"`` — Mamba-2 SSM (structured state-space duality)
    - ``"mamba3"`` — Mamba-3 SSM (RoPE, no conv, 4 state tensors per layer)

    All versions share the same streaming interface: each frame ``[B, D]``
    passes through residual Mamba blocks, with state held in a ``MambaCache``
    (``Mamba3.step()`` is called directly for subsequent frames to work around
    a shape mismatch in its own ``forward()``).
    """

    def __init__(
        self, dim: int, expand: int = 2, d_state: int = 128, d_conv: int = 4,
        num_blocks: int = 3, mamba_version: str = "mamba2",
        # --- Mamba3-specific (ignored for mamba1/mamba2) ---
        headdim: int = 64,
        ngroups: int = 1,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        is_outproj_norm: bool = False,
    ):
        super().__init__()
        _require_mamba()
        self.mamba_version = mamba_version
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_blocks):
            if mamba_version == "mamba3":
                blk = Mamba3(
                    d_model=dim, d_state=d_state, expand=expand,
                    headdim=headdim, ngroups=ngroups,
                    is_mimo=is_mimo, mimo_rank=mimo_rank,
                    chunk_size=chunk_size, layer_idx=i,
                    is_outproj_norm=is_outproj_norm,
                )
            else:
                mamba_cls = Mamba2 if mamba_version == "mamba2" else Mamba
                kw = dict(d_model=dim, d_state=d_state, d_conv=d_conv,
                          expand=expand, layer_idx=i)
                if mamba_version == "mamba2":
                    kw["headdim"] = 64
                    assert (dim * expand / 64) % 8 == 0, (
                        f"Mamba2 requires (d_model * expand / headdim) %% 8 == 0, "
                        f"got ({dim} * {expand} / 64) = {dim * expand / 64}"
                    )
                blk = mamba_cls(**kw)
            self.blocks.append(blk)
            self.norms.append(nn.LayerNorm(dim))

    def forward(self, x, cache: MambaCache | None = None):
        # x: [B, D] — one frame at a time (streaming)
        if cache is None:
            cache = MambaCache()

        h = x.unsqueeze(1)                                     # [B, 1, D]

        if self.mamba_version == "mamba3":
            # Mamba3's step() expects squeezed [B, D], but its own forward()
            # passes [B, 1, D] when seqlen_offset > 0 — call step() directly.
            is_first_step = cache.seqlen_offset == 0
            for i, (blk, norm) in enumerate(zip(self.blocks, self.norms)):
                h_normed = norm(h.squeeze(1)).unsqueeze(1)     # [B, 1, D]
                if is_first_step:
                    # Full scan on seqlen=1 → populates the 4 cache state tensors
                    out = blk(h_normed, inference_params=cache)
                else:
                    angle_state, ssm_state, k_state, v_state = (
                        cache.key_value_memory_dict[i]
                    )
                    out, _, _, _, _ = blk.step(
                        h_normed.squeeze(1),                   # [B, D]
                        angle_state, ssm_state, k_state, v_state,
                    )
                    out = out.unsqueeze(1)
                h = out + h
        else:
            # Mamba1 / Mamba2: forward() handles step() transparently
            for blk, norm in zip(self.blocks, self.norms):
                h = blk(norm(h.squeeze(1)).unsqueeze(1),
                        inference_params=cache) + h

        cache.seqlen_offset += 1
        return h.squeeze(1), cache


# ===========================================================================
# SlotSSM — follows the reference repo (NeurIPS 2024) block-for-block.
#
#   ref (raw V-JEPA patches)  →  each block projects independently
#   slots                     →  cross-attn (inverted, multi-head)
#                             →  +res → Mamba(per-slot) → +res
#                             →  +res → SelfAttn(slots) → +res
# ===========================================================================


class SlotSSMBlock(nn.Module):
    """
    One SlotSSM block — matches the reference repo 1:1.

        ref → input_proj → ref_norm
        slots → slot_norm ─┤
        cross-attn(standard or inverted, multi-head) → +residual
        → [sparse gate: top-k mask]
        → Mamba(per-slot, streaming)     → +residual
        → SelfAttn(across slots)         → +residual

    Dense (``top_k=None``): all K slots update every step (reference behaviour).

    Sparse (``top_k=int``): only top-k slots are *active* per timestep.
    Inactive slots are truly frozen — no cross-attn update, no Mamba, no
    self-attn update.  Their representation is preserved bit-for-bit across
    steps, serving as long-term memory.  Active slots can still *read* from
    inactive slots via self-attention (inactive slots are KV-only).

    Cross-attention mode
    --------------------
    ``use_inverted_attention=False`` (default):
        Standard FlashMHA cross-attention.  Each slot independently picks
        which features to attend to via softmax over the feature dimension.
        Multiple slots can attend to the same features — no competition.

    ``use_inverted_attention=True``:
        Inverted softmax (from the SlotSSM reference repo, module.py).
        Softmax runs over (head × slot) dimensions, so each feature token
        competes to be claimed by a slot.  Single-head by default to
        encourage object-level segmentation (reference repo, train.py:122).
        Uses eager MultiHeadAttention — FlashMHA doesn't support inverted.
    """

    def __init__(
        self, slot_dim: int, input_dim: int, top_k: int | None = None,
        mamba_d_state: int = 128, mamba_d_conv: int = 4, mamba_expand: int = 2,
        mamba_version: str = "mamba2", num_heads: int = 4, block_idx: int = 0,
        eps_random: float = 0.0,
        use_inverted_attention: bool = False,
    ):
        super().__init__()
        _require_mamba()
        self.top_k = top_k
        self.eps_random = eps_random
        self.use_inverted_attention = use_inverted_attention
        mamba_cls = Mamba2 if mamba_version == "mamba2" else Mamba

        self.input_proj = nn.Linear(input_dim, slot_dim, bias=False)

        # Cross-attention
        #  - inverted:   always eager MultiHeadAttention (FlashMHA incompatible)
        #  - standard:  FlashMHA when available, else nn.MultiheadAttention
        self.cross_attn_input_norm = nn.LayerNorm(slot_dim)
        self.cross_attn_ref_norm = nn.LayerNorm(slot_dim)
        if use_inverted_attention:
            # Single head matches the reference repo default (train.py:122):
            #   encoder_attn_num_heads=1  # for inverted attn to encourage object segmentation
            self.cross_attn = MultiHeadAttention(
                d_model=slot_dim, num_heads=num_heads, inverted=True,
            )
            self._cross_attn_inverted = True
        elif _HAS_FLASH_ATTN:
            self.cross_attn = FlashMHA(embed_dim=slot_dim, num_heads=num_heads, cross_attn=True)
            self._cross_attn_inverted = False
        else:
            self.cross_attn = nn.MultiheadAttention(slot_dim, num_heads, batch_first=True)
            self._cross_attn_inverted = False

        # Diagnostics populated during forward (only meaningful for inverted path)
        self._slot_mass_min = torch.tensor(float("nan"))
        self._slot_mass_mean = torch.tensor(float("nan"))
        self._slot_usage_frac = torch.tensor(float("nan"))

        # Sparse gate — only when top_k is set
        if top_k is not None:
            self.gate = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, 1, bias=False),
            )
        self._gate_entropy = torch.tensor(0.0)  # accumulated per forward pass

        # --- Diagnostics (populated during forward when _diag_enabled=True) ---
        self._diag_enabled = False
        self._diag_gate_scores: list = []   # [B, K] per step
        self._diag_active_idx: list = []    # [B, top_k] per step

        # Per-slot Mamba
        kw = dict(d_model=slot_dim, d_state=mamba_d_state, d_conv=mamba_d_conv,
                  expand=mamba_expand, layer_idx=block_idx)
        if mamba_version == "mamba2":
            kw["headdim"] = 64
            # Reference asserts this constraint (slotssm.py:153)
            assert (slot_dim * mamba_expand / kw["headdim"]) % 8 == 0, (
                f"Mamba2 requires (d_model * expand / headdim) %% 8 == 0, "
                f"got ({slot_dim} * {mamba_expand} / {kw['headdim']}) = "
                f"{slot_dim * mamba_expand / kw['headdim']}"
            )
        self.mamba = mamba_cls(**kw)
        self.time_mixer_norm = nn.LayerNorm(slot_dim)

        # Slot self-attention
        # - Dense: self-attn (Q=KV=slots), FlashMHA when available.
        # - Sparse: active slots query, all slots as KV (Q≠KV).
        #   FlashMHA self-attn can't do this, so sparse always uses eager
        #   nn.MultiheadAttention.  At 32×32 with batch=1 it's tiny (0.47ms
        #   vs 0.21ms FlashMHA) — not worth a dedicated FlashMHA cross-attn
        #   instance (4.2M params across 4 blocks).
        self.space_attn_norm = nn.LayerNorm(slot_dim)
        if _HAS_FLASH_ATTN and top_k is None:
            self.self_attn = FlashMHA(embed_dim=slot_dim, num_heads=num_heads)
        else:
            self.self_attn = nn.MultiheadAttention(slot_dim, num_heads, batch_first=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cross_attn(self, slots, ref_raw):
        ref_proj = self.input_proj(ref_raw)                       # [B, N, D]
        q = self.cross_attn_input_norm(slots)                     # [B, K, D]
        kv = self.cross_attn_ref_norm(ref_proj)                   # [B, N, D]
        if self._cross_attn_inverted:
            out = self.cross_attn(q, kv, kv)
            # Forward diagnostics from the inverted MultiHeadAttention
            self._slot_mass_min = self.cross_attn._slot_mass_min
            self._slot_mass_mean = self.cross_attn._slot_mass_mean
            self._slot_usage_frac = self.cross_attn._slot_usage_frac
        elif _HAS_FLASH_ATTN:
            out = self.cross_attn(x=q, x_kv=kv)
        else:
            out = self.cross_attn(q, kv, kv)[0]
        return out

    def _mamba_step(self, slots, cache):
        """Run Mamba on all slots. Dense path only."""
        x = self.time_mixer_norm(slots).reshape(-1, 1, slots.shape[-1])
        return self.mamba(x, inference_params=cache).reshape_as(slots)

    def _mamba_step_sparse(self, slots, active_flat, cache):
        """Run Mamba only on active slots.

        First call (cache not yet initialised for this block): runs the full
        scan path on all slots to initialise the Mamba cache.  Matches the
        prior behaviour where inactive slots advanced once before being frozen
        from step 2 onward.

        Subsequent calls: extracts active-slot states from the cache as
        contiguous tensors, calls Mamba2.step() directly on the active subset,
        then scatters the updated states and outputs back.  Inactive slots are
        never touched — no clone, no restore, ~50 % less Mamba compute.
        """
        B, K, D = slots.shape
        layer_idx = self.mamba.layer_idx
        x = self.time_mixer_norm(slots).reshape(-1, 1, D)

        has_prev = layer_idx in cache.key_value_memory_dict
        if not has_prev:
            # First call: scan path initialises the cache for all slots and
            # advances every state by one timestep (same as the original
            # save/restore path, which also had no previous state to restore).
            full_out = self.mamba(x, inference_params=cache).reshape(B, K, D)
            return full_out   # caller applies the mask

        # --- Subsequent calls: active-slot-only step ------------------------
        kv = cache.key_value_memory_dict[layer_idx]

        # Extract active slots as contiguous tensors (required by the fused
        # CUDA / Triton kernels inside Mamba2.step).
        conv_active = kv[0][active_flat].contiguous()          # [n_active, C, d_conv]
        ssm_active = kv[1][active_flat].contiguous()           # [n_active, H, hdim, d_state]
        x_active = x[active_flat]                              # [n_active, 1, D]

        # step() mutates conv_active & ssm_active in-place.
        out_active, _, _ = self.mamba.step(x_active, conv_active, ssm_active)

        # Scatter the updated states back into the cache.
        kv[0][active_flat] = conv_active
        kv[1][active_flat] = ssm_active

        # Scatter the output into a zeroed tensor (inactive slots → 0).
        out = torch.zeros(B * K, D, device=slots.device, dtype=out_active.dtype)
        out[active_flat] = out_active.squeeze(1)               # remove seqlen dim
        return out.reshape(B, K, D)

    def _self_attn_all(self, slots):
        x = self.space_attn_norm(slots)
        result = self.self_attn(x) if _HAS_FLASH_ATTN else self.self_attn(x, x, x)[0]
        return result

    def _self_attn_sparse(self, slots, active_flat):
        """Active slots query; all slots serve as KV (read-only memory).

        Always uses eager nn.MultiheadAttention — FlashMHA self-attn can't
        do Q≠KV, and the 16×32 attn at batch=1 is below FlashMHA's breakeven.
        """
        B, K, D = slots.shape
        x = self.space_attn_norm(slots)                        # [B, K, D]
        kv = x                                                  # [B, K, D]
        x_flat = x.reshape(B * K, D)                            # [B*K, D]
        q = x_flat[active_flat].reshape(B, -1, D)              # [B, active_slots, D]

        out_active = self.self_attn(q, kv, kv)[0]

        out = torch.zeros(B, K, D, device=slots.device, dtype=out_active.dtype)
        out_flat = out.reshape(B * K, D)
        out_flat[active_flat] = out_active.reshape(-1, D)
        return out

    # ------------------------------------------------------------------
    def forward(self, slots, ref_raw, cache: MambaCache):
        if self.top_k is None:
            # === Dense: all slots update (reference SlotSSM) ===
            slots = slots + self._cross_attn(slots, ref_raw)
            slots = slots + self._mamba_step(slots, cache)
            slots = slots + self._self_attn_all(slots)
            return slots

        # === Sparse: only top-k active slots update ===
        B, K, D = slots.shape
        layer_idx = self.mamba.layer_idx
        has_prev = layer_idx in cache.key_value_memory_dict

        # 1. Cross-attn for all K slots — needed so the gate can see the
        #    current input.  Each slot queries the scene from its frozen
        #    state; the resulting cross_out encodes "what this slot sees."
        cross_out = self._cross_attn(slots, ref_raw)

        # 2. Gate on input-informed representation (RIMs-style: the input
        #    drives activation, not just pre-existing slot state).
        informed = slots + cross_out
        gate_scores = self.gate(informed).squeeze(-1)                # [B, K]

        if self.training and self.eps_random > 0 and torch.rand(1).item() < self.eps_random:
            active_idx = torch.stack([torch.randperm(K, device=slots.device)[:self.top_k]
                                       for _ in range(B)])
            self._gate_entropy = gate_scores.new_tensor(float(math.log(K)))
        else:
            _, active_idx = gate_scores.topk(self.top_k, dim=1)      # [B, top_k]
            p = gate_scores.softmax(dim=-1)                           # [B, K]
            entropy = -(p * (p + 1e-9).log()).sum(dim=-1).mean()     # scalar
            self._gate_entropy = entropy.detach()

        # --- Diagnostics: capture gate scores and active indices ---
        if self._diag_enabled:
            self._diag_gate_scores.append(gate_scores.detach().cpu())
            self._diag_active_idx.append(active_idx.detach().cpu())

        # --- First step: Mamba cache not yet initialised for this block.
        #     Run scan path on all K slots to populate the cache, then
        #     fall through to the mask-based path (same as before).
        if not has_prev:
            mask = torch.zeros(B, K, device=slots.device, dtype=slots.dtype).scatter_(1, active_idx, 1.0)
            mask_3d = mask.unsqueeze(-1)
            active_flat = mask.reshape(-1).bool()
            slots = slots + cross_out * mask_3d
            x_all = self.time_mixer_norm(slots).reshape(-1, 1, D)
            full_out = self.mamba(x_all, inference_params=cache).reshape(B, K, D)
            slots = slots + full_out * mask_3d
            slots = slots + self._self_attn_sparse(slots, active_flat)
            return slots

        # === Subsequent steps: compact active-slot path -------------------
        # Operate on a dense [B, top_k, D] tensor using integer advanced
        # indexing, which produces contiguous views.  This eliminates all
        # mask multiplications and gather/scatter in attention, keeping the
        # CUDA graph fused and avoiding the ~1 ms/block sync overhead.

        tk = self.top_k
        batch_idx = torch.arange(B, device=slots.device).unsqueeze(1)   # [B, 1]

        # 3. Compact active slots — integer indexing → contiguous
        compact_slots = slots[batch_idx, active_idx]                     # [B, tk, D]
        compact_cross = cross_out[batch_idx, active_idx]                 # [B, tk, D]

        # 4. Cross-attn update (dense on compact, no mask)
        compact_slots = compact_slots + compact_cross

        # 5. Mamba on compact active slots
        x_compact = self.time_mixer_norm(compact_slots).reshape(-1, 1, D)  # [B*tk, 1, D]
        idx_flat = active_idx.reshape(-1)                                   # [B*tk]
        kv = cache.key_value_memory_dict[layer_idx]
        conv_active = kv[0][idx_flat]       # [B*tk, C, d_conv] — integer idx → contiguous
        ssm_active = kv[1][idx_flat]        # [B*tk, H, hdim, d_state]
        out_active, _, _ = self.mamba.step(x_compact, conv_active, ssm_active)
        kv[0][idx_flat] = conv_active       # scatter updated states back
        kv[1][idx_flat] = ssm_active
        compact_slots = compact_slots + out_active.squeeze(1).reshape(B, tk, D)

        # 6. Self-attn: compact Q queries full slots as KV (read-only memory)
        compact_q = self.space_attn_norm(compact_slots)                # [B, tk, D]
        full_kv = self.space_attn_norm(slots)                           # [B, K, D]
        sa_out = self.self_attn(compact_q, full_kv, full_kv)[0]        # [B, tk, D]
        compact_slots = compact_slots + sa_out

        # 7. Scatter compact back into full slots (only active indices change).
        #     Clone to avoid in-place on a leaf-variable view.
        slots = slots.clone()
        slots[batch_idx, active_idx] = compact_slots
        return slots


class SlotSSMTemporalModel(nn.Module):
    """
    SlotSSM: K modular slots with independent Mamba dynamics.

    When ``top_k`` is None → dense (all slots update every step).
    When ``top_k`` is int  → sparse (only top-k active; inactive slots freeze).

    Follows the reference repo: initial slots are learnable, ref (raw V-JEPA
    patches) is passed to every block, each block projects independently.
    """

    def __init__(
        self, num_slots: int = 32, slot_dim: int = 512, input_dim: int = 1408,
        num_blocks: int = 4, top_k: int | None = None,
        mamba_d_state: int = 128, mamba_d_conv: int = 4, mamba_expand: int = 2,
        mamba_version: str = "mamba2", num_heads: int = 4,
        eps_random: float = 0.0,
        use_inverted_attention: bool = False,
    ):
        super().__init__()
        _require_mamba()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.top_k = top_k

        self.slots_init = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)

        self.blocks = nn.ModuleList([
            SlotSSMBlock(
                slot_dim=slot_dim, input_dim=input_dim, top_k=top_k,
                mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand, mamba_version=mamba_version,
                num_heads=num_heads, block_idx=i,
                eps_random=eps_random if top_k is not None else 0.0,
                use_inverted_attention=use_inverted_attention,
            )
            for i in range(num_blocks)
        ])
        self._entropy = torch.tensor(0.0)  # populated during forward
        self._slot_mass_min = torch.tensor(float("nan"))
        self._slot_mass_mean = torch.tensor(float("nan"))
        self._slot_usage_frac = torch.tensor(float("nan"))

        # --- Diagnostic collection (opt-in, off by default) ---
        self._diag_slots: list | None = None   # final slot states [B, K, D] per step

    def enable_diagnostics(self):
        """Enable per-step collection of gate scores, active indices, and slot states."""
        for blk in self.blocks:
            blk._diag_enabled = True
            blk._diag_gate_scores = []
            blk._diag_active_idx = []
        self._diag_slots = []

    def disable_diagnostics(self):
        """Disable diagnostic collection and free stored data."""
        for blk in self.blocks:
            blk._diag_enabled = False
            blk._diag_gate_scores = []
            blk._diag_active_idx = []
        self._diag_slots = []

    def get_diagnostics(self) -> dict:
        """Return collected diagnostics after a forward pass.

        Returns
        -------
        dict with keys:
            gate_scores : list[list[Tensor]]  — per-block list of [B, K] gate logits
            active_idx  : list[list[Tensor]]  — per-block list of [B, top_k] active indices
            slots       : list[Tensor]        — [B, K, D] final slot state per step
            entropy     : list[float]         — gate entropy per block (sum across steps)
            mass_min    : float               — minimum slot mass (inverted attn only, else NaN)
            mass_mean   : float
            usage_frac  : float
        """
        return {
            "gate_scores": [list(blk._diag_gate_scores) for blk in self.blocks],
            "active_idx": [list(blk._diag_active_idx) for blk in self.blocks],
            "slots": list(self._diag_slots) if self._diag_slots is not None else [],
            "entropy": [float(blk._gate_entropy) for blk in self.blocks],
            "mass_min": float(self._slot_mass_min),
            "mass_mean": float(self._slot_mass_mean),
            "usage_frac": float(self._slot_usage_frac),
        }

    def forward(self, patches, cache: MambaCache | None = None):
        B = patches.shape[0]
        if cache is None:
            cache = MambaCache()

        slots = self.slots_init.expand(B, -1, -1)
        ent = 0.0
        for blk in self.blocks:
            slots = blk(slots, patches, cache)
            if blk.top_k is not None:
                ent = ent + blk._gate_entropy
        self._entropy = ent  # training loop reads this

        # Aggregate inverted cross-attn diagnostics across blocks (worst-case)
        self._slot_mass_min = min(blk._slot_mass_min for blk in self.blocks)
        self._slot_mass_mean = (sum(blk._slot_mass_mean for blk in self.blocks) / len(self.blocks))
        self._slot_usage_frac = min(blk._slot_usage_frac for blk in self.blocks)

        # --- Diagnostics: store final slot states ---
        if self._diag_slots is not None:
            self._diag_slots.append(slots.detach().cpu())

        cache.seqlen_offset += 1
        return slots, cache                                          # [B, K, D]


# ===========================================================================
# Main model
# ===========================================================================
class ClsVJEPA(nn.Module):
    """
    V-JEPA 2.1 encoder → (pool or patches) → temporal model → binary classifier.
    """

    def __init__(
        self, encoder: VJEPA2Encoder, embed_dim: int,
        dim_latent: int = 1024, dropout: float = 0.5, temporal_model: str = "lstm",
        # LSTM
        rnn_state_size: int = 1024, rnn_cell_num: int = 3,
        # Mamba / SSM
        mamba_d_state: int = 128, mamba_d_conv: int = 4,
        mamba_expand: int = 2, mamba_version: str = "mamba2",
        # SlotSSM
        num_slots: int = 32, slot_dim: int = 512, num_ssm_blocks: int = 4,
        # Sparse SlotSSM
        top_k: int = 16,
        eps_random: float = 0.0,
        # Inverted attention (SlotSSM reference repo style)
        use_inverted_attention: bool = False,
        train_encoder: bool = False,
        # V-JEPA spatial-grid mode (keep patch tokens, pool spatially like Swin)
        vjepa_spatial_grid: tuple | None = None,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        img_size: int = 384,
        verbose: bool = True,
    ):
        super().__init__()
        self.temporal_type = temporal_model
        self.train_encoder = train_encoder

        # ---- Layout-agnostic setup ------------------------------------------
        self._is_swin = isinstance(encoder, SwinEncoder)

        # SwinEncoder always creates proj_norm + proj + proj_drop for its
        # return_patches=False path, but ClsVJEPA only ever calls it with
        # return_patches=True (it provides its own bn + lin1 projection).
        # Drop the dead layers so parameter counts and memory match reality.
        if self._is_swin:
            for _attr in ("proj_norm", "proj", "proj_drop"):
                if hasattr(encoder, _attr):
                    delattr(encoder, _attr)

        # V-JEPA spatial-grid mode: keep patch tokens, pool spatially (like Swin).
        # Applies to ALL temporal models — it's an encoder-postprocessing step.
        # e.g. vjepa_spatial_grid=[6, 6] pools 24×24×2 → 6×6×1 = 36 grid cells.
        self._tubelet_size = tubelet_size
        self._vjepa_n_temp = num_frames // tubelet_size

        self._use_spatial_grid = (
            vjepa_spatial_grid is not None
            and not self._is_swin
        )
        if self._use_spatial_grid:
            self._grid_size = vjepa_spatial_grid[0] * vjepa_spatial_grid[1]
            n_spat = img_size // patch_size

            # Learned spatial downsampling via depthwise Conv2d.  The temporal
            # dimension is always 1 (averaged upstream in _spatial_pool_tokens),
            # so 3D convolution is unnecessary.
            self.vjepa_spatial_pool = nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=(n_spat // vjepa_spatial_grid[0],
                             n_spat // vjepa_spatial_grid[1]),
                stride=(n_spat // vjepa_spatial_grid[0],
                        n_spat // vjepa_spatial_grid[1]),
                groups=embed_dim,   # depthwise: each channel learns its own filter
                bias=False,
            )

            if verbose:
                k_h = n_spat // vjepa_spatial_grid[0]
                k_w = n_spat // vjepa_spatial_grid[1]
                print(
                    f"[ClsVJEPA] V-JEPA spatial pool (learned depthwise): "
                    f"{self._vjepa_n_temp}×{n_spat}×{n_spat}"
                    f" → 1×{vjepa_spatial_grid[0]}×{vjepa_spatial_grid[1]}"
                    f" = {self._grid_size} grid cells × {embed_dim}D"
                    f"  (Conv2d k={k_h}×{k_w}, groups={embed_dim})"
                )

        # ---- Slot-based path ------------------------------------------------
        if temporal_model in ("slotssm", "sparse_slotssm"):
            _require_mamba()
            self._slot_based = True
            is_sparse = temporal_model == "sparse_slotssm"
            self.temporal = SlotSSMTemporalModel(
                num_slots=num_slots, slot_dim=slot_dim, input_dim=embed_dim,
                num_blocks=num_ssm_blocks, top_k=top_k if is_sparse else None,
                mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand, mamba_version=mamba_version,
                eps_random=eps_random if is_sparse else 0.0,
                use_inverted_attention=use_inverted_attention,
            )

            # Learned attention-pool over slots (640 params — negligible).
            # A learnable query attends to slots via dot-product, letting the
            # classifier upweight anomaly-relevant slots instead of blending
            # all 32 equally.  Then a post-temporal MLP mirroring the standard
            # path's lin2 ensures both architectures have the same depth.
            self.slot_query = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
            D = slot_dim
            self.classifier = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, dim_latent),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_latent, dim_latent),   # mirrors lin2
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_latent, 2),
            )
            self.apply(_weights_init)
            self.encoder = encoder   # attach AFTER weight init — preserves pretrained weights

            mode = "sparse" if is_sparse else "dense"
            if verbose:
                sp = getattr(self, "vjepa_spatial_pool", None)
                print(f"\n[ClsVJEPA] SlotSSM ({mode}) — Parameter summary:")
                _print_param_summary(encoder, self.temporal, self.classifier, spatial_pool=sp)
            return

        # ---- Standard path --------------------------------------------------
        self._slot_based = False
        self._keep_patches = self._use_spatial_grid
        if self._keep_patches:
            in_features = embed_dim * self._grid_size
        elif self._is_swin:
            in_features = embed_dim * encoder.num_patches
        else:
            in_features = embed_dim

        self.bn = nn.LayerNorm(in_features)
        self.lin1 = nn.Linear(in_features, dim_latent)
        self.lin2 = nn.Linear(dim_latent, dim_latent)
        self.lin3 = nn.Linear(dim_latent, 2)
        self.drop = nn.Dropout(dropout)

        if temporal_model == "lstm":
            self.temporal = LSTMTemporalModel(dim_latent, rnn_state_size, rnn_cell_num)
            temporal_out = self.temporal.output_dim   # 2 * rnn_state_size when bidirectional
            self.lin2 = nn.Linear(temporal_out, dim_latent)   # override: bidirectional → 2× hidden
        elif temporal_model == "mamba":
            _require_mamba()
            self.temporal = MambaTemporalModel(
                dim=dim_latent, expand=mamba_expand,
                d_state=mamba_d_state, d_conv=mamba_d_conv,
                num_blocks=rnn_cell_num, mamba_version=mamba_version,
            )
        elif temporal_model == "none":
            self.temporal = NoTemporalModel()
        else:
            raise ValueError(f"Unknown temporal_model: {temporal_model}")

        self.apply(_weights_init)
        self.encoder = encoder   # attach AFTER weight init — preserves pretrained weights

        if verbose:
            sp = getattr(self, "vjepa_spatial_pool", None)
            print("\n[ClsVJEPA] Parameter summary:")
            _print_param_summary(
                encoder, self.temporal,
                nn.ModuleList([self.lin1, self.lin2, self.lin3, self.bn]),
                spatial_pool=sp,
            )

    def _spatial_pool_tokens(
        self, x: torch.Tensor, n_temp: int | None = None,
    ) -> torch.Tensor:
        """Reshape patch tokens → temporal average → spatial grid → learned Conv2d → tokens.

        Temporal averaging is done *before* the learned depthwise Conv2d.
        Since convolution is linear,
        :math:`\\text{mean}(\\text{conv}(x_t)) = \\text{conv}(\\text{mean}(x_t))`
        — the two operations commute, so averaging early is lossless and
        shrinks the Conv2d input by :math:`T\\times`.

        ``[B, N, D]`` → ``[B, grid_size, D]``  (preserves token structure).
        ``n_spat`` is derived from the actual token count so this works with
        any input resolution, not just the config-specified one.

        Args:
            x: ``[B, N, D]`` patch tokens from V-JEPA encoder.
            n_temp: Temporal patches.  Defaults to 1 (input is already
                temporally averaged).  Callers that pass raw encoder
                output (``forward()``) should pass ``self._vjepa_n_temp``
                explicitly.
        """
        B, N, D = x.shape
        t = n_temp if n_temp is not None else 1
        h = int((N / t) ** 0.5)
        # Temporal average (lossless — see docstring), then feed as NCHW to Conv2d.
        x = x.reshape(B, t, h, h, D).mean(dim=1)                # [B, H, W, D]
        x = x.permute(0, 3, 1, 2)                                # [B, D, H, W]
        x = self.vjepa_spatial_pool(x)                            # [B, D, h, w]  (learned spatial)
        x = x.flatten(2).transpose(1, 2)                         # [B, grid_size, D]
        return x

    def forward(self, x, state=None):
        if self._slot_based:
            patches = self.encoder(x, return_patches=True)    # [B, N, embed_dim]
            if self._use_spatial_grid:
                patches = self._spatial_pool_tokens(patches, self._vjepa_n_temp)
            slots, new_state = self.temporal(patches, state)  # [B, K, D]
            # Learned attention-pool: query attends to slots
            D = slots.shape[-1]
            scores = (slots * self.slot_query).sum(dim=-1) / (D ** 0.5)  # [B, K]
            attn = scores.softmax(dim=-1)                       # [B, K]
            pooled = (attn.unsqueeze(-1) * slots).sum(dim=1)   # [B, D]
            return self.classifier(pooled), new_state

        if self._keep_patches:
            x = self.encoder(x, return_patches=True)           # [B, N, D]
            x = self._spatial_pool_tokens(x, self._vjepa_n_temp).reshape(x.shape[0], -1)
        elif self._is_swin:
            x = self.encoder(x, return_patches=True)           # [B, 36, embed_dim]
            x = x.transpose(1, 2).flatten(1)                   # [B, C*36]  (spatial-fast → MOVAD-compatible order)
        else:
            x = self.encoder(x)                                # [B, embed_dim]
        x = self.bn(x)
        x = F.relu(self.lin1(x))
        x = self.drop(x)
        x, new_state = self.temporal(x, state)
        x = F.relu(self.lin2(x))
        x = self.drop(x)
        x = self.lin3(x)
        return x, new_state


# ===========================================================================
# Multi-Head Wrapper — shared frozen encoder, multiple independent temporal
# heads trained on the same encoded features from a single ViT pass.
# ===========================================================================
class MultiHeadVJEPA(nn.Module):
    """One V-JEPA encoder → multiple temporal + classifier heads.

    Each head is a complete MOVAD-style model: encoder → projection → temporal
    → classifier.  Each clip ``[B, C, NF, H, W]`` passes through the full model
    in a single ``head(clip, state)`` call, matching the original MOVAD pattern
    exactly — Swin+pool+proj+LSTM+classifier all in one forward.

    Each head trains independently (its own optimizer, checkpoint, and
    wandb writer).  Losses are **never summed** — each head's ``.backward()``
    flows only through its own parameters.

    When ``train_encoder=True`` the encoder is unfrozen and trained jointly.
    This mode assumes a **single head** — the encoder gradients come from one
    temporal model only.

    Usage
    -----
    >>> model = build_multi_head_vjepa(cfg)
    >>> for i in range(NF, VCL):
    >>>     clip = video[:, :, i - NF:i, :, :]   # [B, C, NF, H, W]
    >>>     output, state = model.heads[name](clip, state)
    >>>     loss = criterion(output, target)
    >>>     loss.backward()
    >>>     opt.step()
    """

    def __init__(self, encoder: VJEPA2Encoder, heads_configs: list[dict],
                 train_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.train_encoder = train_encoder

        if self.train_encoder:
            if len(heads_configs) > 1:
                raise ValueError(
                    f"train_encoder=True only supports a single head, got {len(heads_configs)}. "
                    "Multiple heads would produce conflicting encoder gradients."
                )
            # load_pretrained_encoder() froze these — reverse it
            for p in self.encoder.parameters():
                p.requires_grad = True

        self.heads = nn.ModuleDict()
        self.head_configs: dict[str, dict] = {}

        for head_cfg in heads_configs:
            name = head_cfg["name"]
            if name in self.heads:
                raise ValueError(f"Duplicate head name: {name}")
            self.head_configs[name] = dict(head_cfg)

            self.heads[name] = ClsVJEPA(
                encoder=encoder,
                embed_dim=encoder.embed_dim,
                dim_latent=head_cfg.get("dim_latent", 1024),
                dropout=head_cfg.get("dropout", 0.5),
                temporal_model=head_cfg["temporal_model"],
                rnn_state_size=head_cfg.get("rnn_state_size", 1024),
                rnn_cell_num=head_cfg.get("rnn_cell_num", 3),
                mamba_d_state=head_cfg.get("mamba_d_state", 128),
                mamba_d_conv=head_cfg.get("mamba_d_conv", 4),
                mamba_expand=head_cfg.get("mamba_expand", 2),
                mamba_version=head_cfg.get("mamba_version", "mamba2"),
                num_slots=head_cfg.get("num_slots", 32),
                slot_dim=head_cfg.get("slot_dim", 512),
                num_ssm_blocks=head_cfg.get("num_ssm_blocks", 4),
                top_k=head_cfg.get("top_k", 16),
                eps_random=head_cfg.get("eps_random", 0.0),
                use_inverted_attention=head_cfg.get("use_inverted_attention", False),
                train_encoder=train_encoder,
                vjepa_spatial_grid=head_cfg.get("vjepa_spatial_grid", None),
                patch_size=head_cfg.get("patch_size", 16),
                num_frames=head_cfg.get("num_frames", 16),
                tubelet_size=head_cfg.get("tubelet_size", 2),
                img_size=head_cfg.get("img_size", 384),
                verbose=False,
            )

        # Summarise
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        total_trainable = 0
        enc_label = "Trainable" if self.train_encoder else "Frozen"
        print(f"\n[MultiHeadVJEPA] {len(self.heads)} heads — shared encoder, independent temporal models")
        print(f"  {enc_label} (encoder): {enc_params / 1e6:.1f}M")
        for name, head in self.heads.items():
            tp = _count(head)
            total_trainable += tp
            print(f"  Head '{name}' ({head.temporal_type}): {tp / 1e6:.2f}M trainable")
        print("  ---")
        print(f"  Total trainable (all heads): {total_trainable / 1e6:.2f}M")

    def train(self, mode: bool = True):
        """Set training mode — temporal heads follow ``mode``, encoder stays eval
        unless ``train_encoder=True``."""
        super().train(mode)
        if not self.train_encoder:
            self.encoder.eval()
        return self


# ===========================================================================
# Factories
# ===========================================================================
def _build_encoder(cfg):
    """Build the spatial encoder (V-JEPA ViT or Swin 3D) based on ``model_name``."""
    model_name = cfg.get("model_name", "vit_base")
    if model_name.startswith("swin"):
        return build_swin_encoder(cfg)
    return build_vjepa2_encoder(cfg)


def build_cls_vjepa(cfg) -> ClsVJEPA:
    encoder = _build_encoder(cfg)

    if cfg.get("compile", True) and hasattr(torch, "compile"):
        backbone = getattr(encoder, "encoder", None) or getattr(encoder, "swin", None)
        if backbone is not None:
            compiled = torch.compile(backbone, mode="default")
            if hasattr(encoder, "encoder"):
                encoder.encoder = compiled
            else:
                encoder.swin = compiled

    model = ClsVJEPA(
        encoder=encoder,
        embed_dim=encoder.embed_dim,
        dim_latent=cfg.get("dim_latent", 1024),
        dropout=cfg.get("dropout", 0.5),
        temporal_model=cfg.get("temporal_model", "lstm"),
        rnn_state_size=cfg.get("rnn_state_size", 1024),
        rnn_cell_num=cfg.get("rnn_cell_num", 3),
        mamba_d_state=cfg.get("mamba_d_state", 128),
        mamba_d_conv=cfg.get("mamba_d_conv", 4),
        mamba_expand=cfg.get("mamba_expand", 2),
        mamba_version=cfg.get("mamba_version", "mamba2"),
        num_slots=cfg.get("num_slots", 32),
        slot_dim=cfg.get("slot_dim", 512),
        num_ssm_blocks=cfg.get("num_ssm_blocks", 4),
        top_k=cfg.get("top_k", 16),
        eps_random=cfg.get("eps_random", 0.0),
        use_inverted_attention=cfg.get("use_inverted_attention", False),
        train_encoder=cfg.get("train_encoder", False),
        vjepa_spatial_grid=cfg.get("vjepa_spatial_grid", None),
        patch_size=cfg.get("patch_size", 16),
        num_frames=cfg.get("num_frames", 16),
        tubelet_size=cfg.get("tubelet_size", 2),
        img_size=cfg.get("img_size", 384),
    ).to(cfg.device)

    return model


def build_multi_head_vjepa(cfg) -> MultiHeadVJEPA:
    """Build a MultiHeadVJEPA from a multi-config CLI invocation.

    The CLI produces ``cfg._head_cfgs_flat`` — a list of dicts, each the
    full parsed YAML from one ``--config`` path, with a ``"name"`` field
    derived from the file basename.  The first config is the master (encoder,
    data, training settings); each subsequent config contributes its
    ``temporal_model`` settings.

    Every head inherits shared defaults from the master config (``dim_latent``,
    ``dropout``, etc.) but can override them per-head.
    """
    encoder = _build_encoder(cfg)

    if cfg.get("compile", True) and hasattr(torch, "compile"):
        backbone = getattr(encoder, "encoder", None) or getattr(encoder, "swin", None)
        if backbone is not None:
            compiled = torch.compile(backbone, mode="default")
            if hasattr(encoder, "encoder"):
                encoder.encoder = compiled
            else:
                encoder.swin = compiled

    head_configs = []
    for hc in cfg._head_cfgs_flat:
        merged = dict(hc)   # full YAML from that config file
        head_configs.append(merged)

    model = MultiHeadVJEPA(
        encoder=encoder, heads_configs=head_configs,
        train_encoder=cfg.get("train_encoder", False),
    ).to(cfg.device)

    return model

