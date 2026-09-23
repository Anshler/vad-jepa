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
        norm_over_input=True, epsilon=1e-5, logit_scale=1.0,
        logit_scale_learnable=False, num_slots=None, slot_bias_gamma=0.0,
        slot_bias_cap=5.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.inverted = inverted
        self.norm_over_input = norm_over_input
        self.epsilon = epsilon
        # Multiplies the attention logits before the softmax. 1.0 is neutral and
        # reproduces the original behaviour exactly.
        #
        # This is the INVERSE of the conventional softmax temperature:
        # softmax(z*T) == softmax(z / (1/T)), so T=4 means tau=0.25. If you
        # implement it as softmax(logits / tau), you want tau = 1/T.
        #
        # Why it matters here: the standard softmax runs over patches, and its
        # logits span only ~0.49 sd across the 36 patches. The softmax's natural
        # scale is 1, so that is nearly flat -- each slot effectively averages
        # over ~32 of 36 patches, i.e. it receives the patch mean regardless of
        # its query. Every slot therefore gets the same input, so slots become
        # near-copies and the sparse gate has nothing to discriminate.
        # Scaling the logits to ~2 sd makes the top patches dominate, and slots
        # with different queries then bind different patches.
        # Measured (tests/check_inverted_attn.py): out_sim 0.78 at 1.0,
        # 0.45 at 2.0, 0.15 at 4.0, 0.06 at 8.0.
        self.logit_scale = float(logit_scale)
        # Learnable variant: one scalar PER HEAD, initialized to logit_scale.
        # Preferred to a hand-picked constant because it lets the model report
        # where it wants to sit rather than us guessing -- and a learned value
        # near the init is evidence the flatness was never a problem.
        # NOTE: the training optimizer is plain SGD with momentum and NO weight
        # decay, so this parameter is not pulled toward 0. If weight decay is ever
        # added, exclude this parameter from it.
        self.logit_scale_learnable = bool(logit_scale_learnable)
        if self.logit_scale_learnable:
            self.logit_scale_param = nn.Parameter(
                torch.full((num_heads,), float(logit_scale)))
        else:
            self.logit_scale_param = None

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
        # DIFFERENTIABLE slot-usage penalty, inverted path only. 0 when the
        # patch mass is spread evenly over the slots; grows as it concentrates.
        # The caller weights it into the loss (config `attn_usage_weight`).
        self._usage_penalty = torch.tensor(0.0)

        # --- gradient-FREE slot-usage bias (inverted path only) --------------
        # A per-slot additive bias on the LOGITS, nudged by observed usage and
        # never differentiated. This is the mechanism that can actually work,
        # and the gradient-based penalty above provably cannot.
        #
        # Measured, 48 videos / 10 epochs, scale-16 inverted, otherwise identical:
        #
        #     weight    usage_frac   live slots   penalty
        #     0.0          0.031          1         3.51
        #     0.2          0.062          2         2.90
        #     2.0          0.031          1         3.16
        #
        # A 10x larger weight does NOTHING -- usage_frac is identical to the
        # control and the penalty stays ~3 at every weight. The model cannot
        # reduce the penalty at any price, because the softmax has SATURATED:
        # once a slot's logits sit far below the max, d(softmax_s)/d(z_s) =
        # exp(z_s - z_max) -> 0. The gradient that would lift a starved slot
        # vanishes exactly when the slot is starved. So the term is not weak, it
        # is unreachable.
        #
        # A bias needs no derivative: it shifts a starved slot's logits directly.
        #
        #   b_s += gamma * sign(1 - usage_s)      # usage relative to fair share
        #
        # detached, training-only (eval must be deterministic), clamped so it
        # cannot run away. Updated from the same `mass_norm` the diagnostic uses.
        # Note this acts on the patch-to-slot ASSIGNMENT, unlike the gate's
        # `balance_weight`, which acts on routing.
        self.slot_bias_gamma = float(slot_bias_gamma)
        self.slot_bias_cap = float(slot_bias_cap)
        if inverted and self.slot_bias_gamma > 0 and num_slots:
            # A buffer, not a Parameter: never differentiated, and only created
            # when enabled so existing checkpoints load unchanged.
            self.register_buffer("slot_bias", torch.zeros(int(num_slots)))
        else:
            self.slot_bias = None

    def forward(self, q, k, v):
        B, T, _ = q.shape
        _, S, _ = k.shape

        q_proj = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k_proj = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v_proj = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)

        q_proj = q_proj * (q_proj.shape[-1] ** (-0.5))
        # Per-head scale when learnable, else the fixed scalar.
        scale = (self.logit_scale_param.view(1, -1, 1, 1)
                 if self.logit_scale_param is not None else self.logit_scale)
        attn = torch.matmul(q_proj, k_proj.transpose(-1, -2)) * scale

        # Gradient-free usage bias, inverted path only. Applied to the SLOT axis
        # (dim 2) BEFORE the softmax, so it shifts which slot a patch prefers.
        # Under standard attention the softmax runs over patches and a per-slot
        # bias would be a constant within each row -- no effect -- so it is not
        # applied there.
        #
        # Applied BEFORE `_last_logits` is recorded, so the diagnostic and the
        # spread measurement both see the logits the softmax actually saw.
        if self.inverted and self.slot_bias is not None:
            attn = attn + self.slot_bias.view(1, 1, -1, 1)

        # Keep the pre-softmax logits (detached, diagnostic-only). The spread of
        # this tensor ACROSS TARGETS is what decides how selective the softmax
        # is: the softmax's natural scale is 1, so a spread well below 1 means
        # near-uniform attention regardless of the queries. That is the quantity
        # logit_scale acts on, and it depends on the learned projection weights,
        # so it has to be measured per model rather than assumed.
        self._last_logits = attn.detach()

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

            # --- gradient-free usage bias update -----------------------------
            # Raised for under-used slots, lowered for over-used ones, by a fixed
            # step. `mass_norm` is relative to fair share, so the target is 1.0.
            #
            # TRAINING ONLY. Eval must be deterministic, and the bias is state
            # that should be frozen at whatever training settled on -- otherwise
            # two eval passes over the same data would disagree.
            #
            # Averaged over the batch before the update: a per-slot step taken
            # per batch item would be `batch_size` times larger and depend on the
            # batch composition.
            if self.slot_bias is not None and self.training:
                with torch.no_grad():
                    self.slot_bias.add_(
                        self.slot_bias_gamma
                        * torch.sign(1.0 - mass_norm.mean(dim=0)))
                    self.slot_bias.clamp_(-self.slot_bias_cap,
                                          self.slot_bias_cap)
            # -----------------------------------------------------------------
            # DIFFERENTIABLE version of the same quantity, for the loss.
            #
            # Sharpening this softmax starves slots with no counter-pressure:
            # measured at init, usage_frac is 1.00 / 0.99 / 0.73 / 0.52 / 0.45
            # at logit_scale 1 / 2 / 4 / 8 / 16, and mass_min hits exactly 0.000
            # by scale 8 -- at least one slot receives nothing. Training then
            # drives usage_frac to ~0, which is the collapse.
            #
            # Nothing else pushes back: the classifier POOLS over slots, so a
            # dead slot costs it nothing, and this repo has no reconstruction
            # objective to make a dead slot expensive (which is how the
            # reference repo survives the same sharpening).
            #
            # `mass_norm` sums to T across slots by construction (mean 1.0), so
            # the squared deviation from 1 is the variance of the per-slot mass:
            # 0 at an even spread, large when it concentrates. Per batch item,
            # i.e. per frame -- the collapse that matters is within a frame,
            # since that is what the pool reads.
            # ONE-SIDED: charge only the DEFICIT, not the excess.
            #
            # A symmetric `(mass - 1)^2` is dominated by the over-used slots -- a
            # slot at 3.0 contributes 4.0 while a starved slot at 0.0 contributes
            # 1.0 -- so minimising it lowers the PEAK rather than lifting the
            # FLOOR. Measured: descending it for 300 steps at the training lr cut
            # the summed penalty 15.70 -> 9.33 while the worst block's
            # `_slot_usage_frac` fell 0.44 -> 0.27, i.e. it made starvation
            # worse while looking like it was working.
            #
            # Starvation is the floor, and it is what kills routing: a frame with
            # two live slots has nothing for a top-16 gate to choose between. So
            # only the deficit is charged, and a slot above fair share costs
            # nothing -- concentrating attention is allowed, killing slots is not.
            #
            # 0 at an even spread; rises as slots fall below fair share.
            live_mass = attn.sum(dim=(1, -1)) / fair                 # [B, T]
            deficit = torch.clamp(1.0 - live_mass, min=0.0)
            self._usage_penalty = (deficit ** 2).mean()
            if self.norm_over_input:
                attn = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)
        else:
            attn = F.softmax(attn, dim=-1)

        # Keep the map for offline inspection (detached, diagnostic-only).  The
        # inverted path's whole claim is that patches compete for slots, which is
        # a statement about this tensor; without it the claim can only be
        # inferred from the mass scalars above.
        self._last_attn = attn.detach()

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
def _restore_rng(cpu_state, cuda_state=None):
    """Rewind the global RNG to a snapshot.

    Used to make the weight-init pass reproducible independently of how many
    draws the construction phase happened to consume -- see ClsVJEPA.__init__.
    """
    torch.set_rng_state(cpu_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.MultiheadAttention):
        # in_proj_weight is a raw Parameter, not inside an nn.Linear, so the
        # branch above never sees it and it keeps nn.MultiheadAttention's own
        # xavier_uniform_. For the FUSED [3D, D] matrix that uses fan_out = 3D, so
        # the bound is sqrt(6/(D+3D)) instead of the sqrt(6/(D+D)) a separate
        # [D, D] Linear would get -- a factor sqrt(2) smaller per projection, and
        # therefore 0.5 in the logit spread, since a logit is a product of two
        # projections.
        #
        # Measured: logit_sd 0.489 here vs 0.983 for the reference repo's
        # separate proj_q/proj_k Linears. That is the whole difference between
        # this repo's cross-attention and the reference's, and it is an INIT
        # difference only -- the two compute the same function on the same
        # weights (tests/_check_mha_equiv.py, bit-identical).
        #
        # gain=sqrt(2) restores the scale the nn.Linear branch intends, so the
        # fused parameter no longer silently gets a different effective gain.
        # Biases are already zeroed by nn.MultiheadAttention._reset_parameters;
        # repeated here so this branch fully owns the parameter.
        torch.nn.init.xavier_uniform_(m.in_proj_weight, gain=math.sqrt(2))
        torch.nn.init.constant_(m.in_proj_bias, 0)
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


def _attn_reduce(w, grid=None):
    """Reduce an averaged-over-heads attention map ``[B, T, S]`` to per-query stats.

    Returns ``(entropy, max_weight, centroid)`` where entropy/max are ``[B, T]``
    and centroid is ``[B, T, 2]`` (row, col) when ``grid`` is given and matches
    ``S``, else ``None``.  Full maps are never kept — a [K, N] map per step per
    block over thousands of frames is hundreds of MB.
    """
    p = w.clamp_min(0)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
    ent = -(p * (p + 1e-12).log()).sum(dim=-1)
    mx = p.max(dim=-1).values
    cent = None
    if grid is not None and grid[0] * grid[1] == p.shape[-1]:
        gh, gw = grid
        ys = torch.arange(gh, dtype=p.dtype, device=p.device).repeat_interleave(gw)
        xs = torch.arange(gw, dtype=p.dtype, device=p.device).repeat(gh)
        cent = torch.stack([(p * ys).sum(-1), (p * xs).sum(-1)], dim=-1)
    return ent, mx, cent


def _slot_sim(s):
    """Mean off-diagonal cosine similarity of [B, K, D] slot states.

    High = slots are redundant (interchangeable); low = slots have specialised.
    Returned per batch item so active/inactive subsets can be compared.
    """
    x = s / (s.norm(dim=-1, keepdim=True) + 1e-9)
    sim = torch.bmm(x, x.transpose(1, 2))                      # [B, K, K]
    K = sim.shape[-1]
    if K < 2:
        return s.new_zeros(s.shape[0])
    off = sim - torch.eye(K, device=sim.device, dtype=sim.dtype).unsqueeze(0)
    return off.sum(dim=(1, 2)) / (K * (K - 1))                 # [B]


def _sincos_2d_pos_embed(n_tokens: int, dim: int, device, dtype):
    """Fixed 2D sin/cos positional embedding for a grid of ``n_tokens`` cells.

    Shape ``[1, n_tokens, dim]``. Same construction as the SlotSSM reference repo
    (`src/models/encoder.py:get_2d_sincos_pos_embed`, itself HuggingFace's), and
    as the ViTs it is copied from: the dimension is split into four equal
    quarters holding sin(y), cos(y), sin(x), cos(x).

    A SQUARE grid is assumed (``h = round(sqrt(n))``); a non-square token count
    falls back to a 1 x n strip rather than raising, so an odd resolution cannot
    crash a run. At the 6x6 = 36 grid this model uses, `dim=512 -> 4*128 = 512`
    exactly, so no padding is ever needed here.

    FIXED, not learned -- so it is a constant, not a parameter, and the flag adds
    NO state-dict keys. That is deliberate: a learned table would be a bare
    `nn.Parameter`, which `_weights_init` cannot see (the trap that caught
    `in_proj_weight` in 15.9.1, `state_gate`, and the pooling `slot_query`), and
    it would also make the mode checkpoint-incompatible for no benefit.
    """
    h = int(round(n_tokens ** 0.5))
    if h * h == n_tokens:
        gw = h
    else:
        h, gw = 1, n_tokens
    d4 = max(dim // 4, 1)
    omega = 1.0 / (10000 ** (torch.arange(d4, dtype=torch.float32,
                                          device=device) / d4))
    gy = torch.arange(h, dtype=torch.float32, device=device).repeat_interleave(gw)
    gx = torch.arange(gw, dtype=torch.float32, device=device).repeat(h)
    emb = torch.cat([torch.sin(gy[:, None] * omega),
                     torch.cos(gy[:, None] * omega),
                     torch.sin(gx[:, None] * omega),
                     torch.cos(gx[:, None] * omega)], dim=1)
    if emb.shape[1] < dim:                       # only if dim is not divisible by 4
        emb = torch.cat([emb, torch.zeros(n_tokens, dim - emb.shape[1],
                                          device=device)], dim=1)
    return emb[:, :dim].unsqueeze(0).to(dtype)


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
        balance_weight: float = 0.0,
        num_slots: int = 32,
        recency_lambda: float = 0.0,
        recency_rho: float = 0.9,
        gumbel_sigma: float = 0.0,
        logit_scale: float = 1.0,
        logit_scale_learnable: bool = False,
        slot_input_gain: bool = False,
        slot_input_proj_rank: int = 0,
        slot_state_decay: bool = False,
        gate_reads_state: bool = False,
        gate_state_init: float = 0.0,
        slot_bias_gamma: float = 0.0,
        slot_bias_cap: float = 5.0,
        # Add a FIXED 2D sin/cos positional embedding to the tokens the
        # cross-attention reads, so slots can in principle key on WHERE a patch
        # is rather than only on what it contains. Off by default: it changes the
        # forward, so it is opt-in like every other flag here. Fixed rather than
        # learned, so it adds no parameters and no state-dict keys.
        slot_pos_pe: bool = False,
    ):
        super().__init__()
        _require_mamba()
        self.top_k = top_k
        self.num_slots = num_slots
        self.eps_random = eps_random
        self.use_inverted_attention = use_inverted_attention
        # Multiplies the cross-attention logits before the softmax. 1.0 is
        # neutral. Applies to BOTH the dense and sparse paths, because the
        # cross-attention is shared -- the flat attention it corrects is upstream
        # of the top_k split, so it cannot be made sparse-only. See findings 15.9.
        self.logit_scale = float(logit_scale)
        # MoE-style load-balancing weight; 0 disables it (no behaviour change).
        self.balance_weight = balance_weight
        mamba_cls = Mamba2 if mamba_version == "mamba2" else Mamba

        self.input_proj = nn.Linear(input_dim, slot_dim, bias=False)

        # Cross-attention
        #  - inverted:   always eager MultiHeadAttention (FlashMHA incompatible)
        #  - standard:  FlashMHA when available, else nn.MultiheadAttention
        self.cross_attn_input_norm = nn.LayerNorm(slot_dim)
        self.cross_attn_ref_norm = nn.LayerNorm(slot_dim)
        # The custom module is required whenever logit_scale != 1: neither
        # FlashMHA nor nn.MultiheadAttention exposes the logits, so the scale
        # cannot be applied through them. It also returns the attention map, which
        # keeps the cross-attention diagnostics working on this path.
        self.logit_scale_learnable = bool(logit_scale_learnable)
        self._cross_attn_custom = (use_inverted_attention or logit_scale != 1.0
                                   or logit_scale_learnable)
        if self._cross_attn_custom:
            # Single head matches the reference repo default (train.py:122):
            #   encoder_attn_num_heads=1  # for inverted attn to encourage object segmentation
            self.cross_attn = MultiHeadAttention(
                d_model=slot_dim, num_heads=num_heads,
                inverted=use_inverted_attention, logit_scale=logit_scale,
                logit_scale_learnable=logit_scale_learnable,
                # num_slots + the usage bias: a gradient-free per-slot logit
                # bias, the only mechanism that can lift a saturated slot.
                num_slots=num_slots, slot_bias_gamma=slot_bias_gamma,
                slot_bias_cap=slot_bias_cap,
            )
            self._cross_attn_inverted = use_inverted_attention
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

        # --- Per-slot differentiation ---------------------------------------
        # Why: the recurrent state homogenises across slots (cross-slot cos-sim
        # 0.60 on real video, against the slot state's 0.28) and attention
        # sharpening does NOT reach it (0.610 -> 0.607 over scale 1->3). The
        # cause looks like the SHARED dynamics: every slot runs the same
        # recurrence with the same weights, so states converge to a common
        # trajectory. So each slot needs its own dynamics.
        #
        # Deliberately implemented WITHOUT touching the Mamba: no monkeypatching,
        # no overriding mamba internals, no hooks on in_proj, no A_log/kernel
        # changes. Everything below is an ordinary tensor op on a tensor this
        # block already owns, applied to the Mamba's input or to the state the
        # caller already writes back.
        #
        # Two constraints on where the modulation can go:
        #  * AFTER time_mixer_norm -- LayerNorm normalises each slot's vector
        #    independently, so anything applied before it is normalised away.
        #  * it must change the recurrence's BALANCE (history vs current input),
        #    not just an output scale: cosine similarity is invariant to positive
        #    scaling, so a pure magnitude change is invisible to the metric that
        #    measures differentiation.
        #
        # All three are initialised DIFFERENTIATED, not at the neutral value.
        # A neutral init would rely on the task gradient to create the
        # difference, and the task gradient has no reason to (findings 15.8).
        # A differentiated init is also more likely to survive: the init fix
        # showed training preserves the init's regime (0.978 -> 0.993).
        self.slot_input_gain = bool(slot_input_gain)
        self.slot_input_proj_rank = int(slot_input_proj_rank)
        self.slot_state_decay = bool(slot_state_decay)

        if self.slot_input_gain:
            # Per-slot, per-channel input gain. Lognormal (mean 1, ~+-35%),
            # drawn per slot so slots start with different effective timescales.
            self.slot_gain = nn.Parameter(
                torch.exp(0.3 * torch.randn(num_slots, slot_dim)))
        else:
            self.slot_gain = None

        if self.slot_input_proj_rank > 0:
            # Per-slot low-rank map into the recurrence: x <- x + U_s (V x).
            # A matmul, so allowed; the rank keeps it affordable (~0.54M per
            # block at r=32 vs 8.4M for a full per-slot [D, D]).
            r = self.slot_input_proj_rank
            self.slot_U = nn.Parameter(torch.randn(num_slots, slot_dim, r) * 0.2)
            self.slot_V = nn.Parameter(torch.randn(r, slot_dim) / r ** 0.5)
        else:
            self.slot_U = None
            self.slot_V = None

        if self.slot_state_decay:
            # Per-slot decay on the recurrent state, bounded to (0.5, 1.0) so the
            # optimizer cannot take the degenerate route of killing the state
            # (decay -> 0 destroys memory while looking like differentiation).
            # Raw init is spread so slots start with different horizons.
            self.slot_decay_raw = nn.Parameter(
                torch.empty(num_slots).uniform_(-1.5, 1.5))
        else:
            self.slot_decay_raw = None

        # Sparse gate — only when top_k is set
        if top_k is not None:
            self.gate = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, 1, bias=False),
            )

        # --- Gate reads the recurrent state --------------------------------
        # An extra additive term on the gate score, read from the SSM state the
        # block wrote at the PREVIOUS step (the cache, not the current slots):
        #
        #     gate_scores += state_gate( mean_{H,hdim} ssm_state )     [B, K]
        #
        # Rationale: the gate currently scores `slots + cross_out`, which is
        # instantaneous -- it cannot express "this slot's memory is what is
        # arriving". A readout of the state can, in principle.
        #
        # Measured precondition (findings 16.7): on the trained baseline the
        # best SINGLE direction through the state re-ranks slots by 0.0021
        # against the gate's ~0.7 score spread (0.3%) -- nothing to read. On
        # `slot_input_proj_rank: 32` it is 0.2422 (34.6%), which IS enough to
        # re-rank a top-16. So this flag is only worth enabling alongside `proj`.
        #
        # Init is ZERO by default, so the arm starts bit-identical to `proj`
        # alone and the term only exists if the gradient puts it there. That is
        # the honest test: findings 15.8 showed the gradient flattens the gate
        # when there is nothing to choose between, so a readout that stays at
        # zero is itself the answer. `gate_state_init` (the weight sd) opens the
        # other arm -- a random readout that perturbs selection from step 0 --
        # at the cost of injecting the state's STATIC per-slot component, which
        # is the `frozen` intervention findings 15.7 measured as doing nothing.
        #
        # Sparse-only: the term multiplies a gate score, and dense has no gate.
        self.gate_reads_state = bool(gate_reads_state) and top_k is not None
        self.gate_state_init = float(gate_state_init)

        # `proj` is a MEASURED PRECONDITION of the readout, so refuse the
        # combination that cannot work rather than train a guaranteed-null arm
        # for 50 epochs and discover it later.  The numbers are in the message
        # because this encodes an empirical fact, not a logical one: if a later
        # measurement shows the baseline state does carry a usable slot x time
        # signal, this check is what should be deleted.
        if self.gate_reads_state and slot_input_proj_rank <= 0:
            raise ValueError(
                "gate_reads_state=True requires slot_input_proj_rank > 0. "
                "The SSM state is only worth reading once the slots have their "
                "own dynamics: the best single readout direction through it "
                "re-ranks slots by 0.0021 against the gate's ~0.7 score spread "
                "(0.3%) on the baseline, and 0.2422 (34.6%) on `proj` "
                "(findings 16.7). Set slot_input_proj_rank: 32, or turn "
                "gate_reads_state off.")
        if self.gate_state_init != 0.0 and not self.gate_reads_state:
            raise ValueError(
                "gate_state_init only applies when gate_reads_state=True; "
                f"got gate_state_init={self.gate_state_init} with the readout "
                "off, where it would silently do nothing.")

        if self.gate_reads_state:
            # A raw Parameter, NOT nn.Linear, and deliberately so: ClsVJEPA
            # applies `_weights_init` to the whole model, and its nn.Linear
            # branch would xavier this away -- destroying the zero init that
            # makes the arm a clean ablation.  Nothing else in _weights_init
            # sees a bare Parameter, so this stays where it is put.
            w = (torch.randn(1, mamba_d_state) * self.gate_state_init
                 if self.gate_state_init != 0.0
                 else torch.zeros(1, mamba_d_state))
            self.state_gate = nn.Parameter(w)
            # Normalise the readout's input, as every other projection in this
            # block already does (`cross_attn_input_norm`, `time_mixer_norm`,
            # `space_attn_norm`).  Measured why: the gate reads a LayerNorm'd
            # vector of per-slot norm sqrt(512) = 22.6, while the raw state
            # summary has norm ~0.37, so the readout's gradient came out 60x
            # below the gate's (ratio 0.0165) and the weight needed ~350k SGD
            # steps to reach an audible scale against a real run's ~20-80k.
            # The 60x is very nearly the input-norm ratio, so normalising should
            # recover most of it.  LayerNorm is NOT touched by `_weights_init`
            # (it handles Linear / MultiheadAttention / LSTMCell only), so it
            # keeps weight=1, bias=0 and adds no new confound.
            self.state_norm = nn.LayerNorm(mamba_d_state)
        else:
            self.state_gate = None
            self.state_norm = None

        self._gate_entropy = torch.tensor(0.0)  # accumulated per forward pass
        self._gate_balance = torch.tensor(0.0)  # load-balancing term (0 if disabled)
        # Slot-usage penalty from the inverted cross-attention (0 unless the
        # attention is inverted). Differentiable; weighted into the loss by the
        # training loop via config `attn_usage_weight`.
        self._attn_usage = torch.tensor(0.0)

        # --- DeepSeek-style aux-loss-free load balancing -------------------
        # A detached per-slot bias added to the gate scores before top-k, nudged
        # by  b_i += gamma * sign(target - f_i).  The bias affects SELECTION only
        # and is never differentiated, which is what lets it act on TEMPORAL
        # usage: the SSM state carries no gradient across frames (the in-place
        # step() kernels are inference-only), so a loss-based balancer cannot
        # reach a multi-frame statistic at all.
        # --- Recency penalty (least-recently-used bias) --------------------
        # Suppress slots that were active recently, forcing turnover.  Unlike a
        # constant load-balancing bias, this is STATE-DEPENDENT, which is what
        # lets it correct a state-driven concentration: measured on a sticky
        # synthetic sequence it raised effective_K from 17.9 to 29.6 of 32 and
        # cut static_frac from 0.70 to 0.09.  Gradient-free, so it works despite
        # the SSM state carrying no gradient across frames.
        self.recency_lambda = recency_lambda
        self.recency_rho = recency_rho
        self._recency = None      # transient per-sample EMA, never checkpointed

        # Gumbel exploration scale (0 = off).  Training-only; see Fix B below.
        self.gumbel_sigma = gumbel_sigma

        # Reproduce the pre-fix forward (bfaa73a) at inference: build the
        # self-attn KV from PRE-update slot states.  This is the ONLY
        # inference-relevant difference between bfaa73a and HEAD in the sparse
        # path -- the STE, eps_random, gumbel, recency and balance changes are
        # gradient-only or training-only, and the write-back form was measured
        # to have exactly zero effect (see step 8 in forward()).  Off everywhere
        # except the compatibility evaluation of a pre-fix checkpoint.
        # Verified faithful to bfaa73a by tests/check_prefix_forward.py, which
        # diffs it against the real bfaa73a:model.py at one bf16 ULP.
        self.legacy_stale_kv = False

        # --- Positional embedding on the cross-attention's KV tokens --------
        # WHY THIS EXISTS (measured 2026-09-23): the slots do not use position at
        # all. Re-running findings 6.7's centroid test on the trained scale sweep
        # reproduced its baseline exactly (sparse scale 1 spread 0.060 cells on a
        # 6x6 grid, inside 6.7's 0.04-0.07 band) and showed what sharpening does
        # to it: the spread grows ~10x (0.06 -> 0.60 cells by scale 16) but every
        # slot's temporal-mean centroid stays within 0.25 cells of the grid
        # CENTRE, and the between-slot / within-slot ratio stays BELOW 1 at every
        # scale (0.57-0.85) -- a slot's centroid drifts more over time than slots
        # differ from each other. So the slots still carry no stable spatial
        # identity, at any scale, in either architecture.
        #
        # The input has no position to use: `_spatial_pool_tokens` averages the
        # NF frames and then applies a DEPTHWISE 4x4 conv, so output cell (i,j)
        # depends only on input block (i,j) and nothing mixes across cells. Any
        # position information in the pooled tokens is whatever survived the
        # frozen encoder's own layernorm stack, implicitly.
        #
        # The reference repo gets position a different way and this is NOT how it
        # does it: its ViT encoder adds fixed sincos embeddings before its
        # transformer (`encoder.py:160`) and the slot model then reads those
        # position-tagged tokens. This flag adds the embedding at the slot
        # boundary instead, which is a DEVIATION from the reference -- worth
        # stating in a write-up.
        #
        # Note what it can and cannot do at INIT: the 32 slots start from
        # `slots_init`, which is per-slot distinct but highly similar (cos 0.85
        # at dense scale 1), so an identical positional signal added to the keys
        # makes every slot's attention pattern position-driven and therefore
        # stable over time, but cannot by itself create BETWEEN-slot
        # specialisation. That needs training to lock different slots onto
        # different positions.
        self.slot_pos_pe = bool(slot_pos_pe)
        self._pe_cache: dict = {}

        # --- Diagnostics (populated during forward when _diag_enabled=True) ---
        self._diag_enabled = False
        self._diag_gate_scores: list = []   # [B, K] per step
        self._diag_active_idx: list = []    # [B, top_k] per step
        self._diag_cross: list = []         # per step: (entropy [B,K], max [B,K], centroid [B,K,2])
        self._diag_self: list = []          # per step: (entropy [B,tk], read_mass [B,K])
        self._diag_slot_delta: list = []    # per step: [B, K] L2 change of each slot
        self._diag_slot_sim: list = []      # per step: (all, active, inactive) mean cos-sim
        # Optional callable(gate_scores, active_idx) -> active_idx, used by
        # routing interventions (random / round-robin / frozen).  None = learned
        # top-k, i.e. exactly the production path.
        self._route_override = None
        # Spatial grid (rows, cols) of the patch tokens, for attention centroids.
        self._diag_grid = None

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

    def _pos_pe(self, x):
        """Cached fixed sincos table for `x`'s token count, device and dtype."""
        key = (x.shape[1], x.dtype, str(x.device))
        pe = self._pe_cache.get(key)
        if pe is None:
            pe = _sincos_2d_pos_embed(x.shape[1], x.shape[-1], x.device, x.dtype)
            self._pe_cache[key] = pe
        return pe

    def _cross_attn(self, slots, ref_raw):
        ref_proj = self.input_proj(ref_raw)                       # [B, N, D]
        # Optional FIXED positional embedding on the tokens the slots read, added
        # in slot space just before the LayerNorm -- a constant per position, so
        # it survives the norm as a direction change and adds no parameters.
        if self.slot_pos_pe:
            ref_proj = ref_proj + self._pos_pe(ref_proj)
        q = self.cross_attn_input_norm(slots)                     # [B, K, D]
        kv = self.cross_attn_ref_norm(ref_proj)                   # [B, N, D]
        if self._cross_attn_custom:
            # Custom module: positional args, returns a tensor, and keeps its map
            # as _last_attn (detached).  Used for the inverted path AND whenever
            # logit_scale != 1, since FlashMHA/nn.MultiheadAttention cannot apply
            # a logit scale.
            out = self.cross_attn(q, kv, kv)
            if self._cross_attn_inverted:
                # Forward diagnostics from the inverted MultiHeadAttention
                self._slot_mass_min = self.cross_attn._slot_mass_min
                self._slot_mass_mean = self.cross_attn._slot_mass_mean
                self._slot_usage_frac = self.cross_attn._slot_usage_frac
            if self._diag_enabled:
                # Recover the same reduced stats the nn.MultiheadAttention branch
                # produces, from the map the custom module kept.  Head-averaged to
                # match that branch's [B, T, S] shape, so the numbers stay
                # comparable across paths.
                w = getattr(self.cross_attn, '_last_attn', None)
                if w is not None:
                    ent, mx, cent = _attn_reduce(w.mean(dim=1),
                                                 grid=self._diag_grid)
                    self._diag_cross.append(
                        (ent.cpu(), mx.cpu(),
                         cent.cpu() if cent is not None else None))
        elif _HAS_FLASH_ATTN:
            out = self.cross_attn(x=q, x_kv=kv)
        elif self._diag_enabled:
            # nn.MultiheadAttention already computes the map; keep only reduced
            # per-slot stats so memory stays flat over long videos.
            out, w = self.cross_attn(q, kv, kv, need_weights=True)
            ent, mx, cent = _attn_reduce(w.detach(), grid=self._diag_grid)
            self._diag_cross.append((ent.cpu(), mx.cpu(),
                                     cent.cpu() if cent is not None else None))
        else:
            out = self.cross_attn(q, kv, kv)[0]
        # Slot-usage penalty, inverted path only. 0.0 otherwise, and the
        # standard module has no `_usage_penalty` at all, hence the getattr.
        # Consumed by the training loop via SlotSSMTemporalModel._attn_usage.
        self._attn_usage = getattr(self.cross_attn, '_usage_penalty',
                                   torch.tensor(0.0))
        return out

    def slot_decay(self):
        """Per-slot decay in (0.5, 1.0). Only defined when slot_state_decay."""
        return 0.5 + 0.5 * torch.sigmoid(self.slot_decay_raw)

    def _slot_modulate(self, x_norm, idx=None):
        """Per-slot input modulation, applied AFTER time_mixer_norm.

        x_norm : [B, K, D] (idx=None, all slots in order) or [B, tk, D] with idx
                 giving the slot id per position.
        Ordinary tensor ops only -- a multiply and a matmul. No Mamba internals.
        """
        if self.slot_gain is not None:
            g = self.slot_gain if idx is None else self.slot_gain[idx]
            x_norm = x_norm * g
        if self.slot_U is not None:
            u = self.slot_U if idx is None else self.slot_U[idx]
            x_norm = x_norm + torch.einsum('...d,...dr->...r', x_norm, u) @ self.slot_V
        return x_norm

    def _mamba_step(self, slots, cache):
        """Run Mamba on all slots. Dense path only."""
        x = self._slot_modulate(self.time_mixer_norm(slots)).reshape(
            -1, 1, slots.shape[-1])
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
        x = self._slot_modulate(self.time_mixer_norm(slots)).reshape(-1, 1, D)

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
        if self._diag_enabled and not _HAS_FLASH_ATTN:
            # Dense read profile.  Same reduction as _self_attn_sparse, so the
            # "how much mass does slot j receive" column is directly comparable
            # between the dense and sparse paths.  FlashMHA cannot return
            # weights, so this only exists for the eager module — which is what
            # dense instantiates when flash-attn is absent (see __init__).
            out, w = self.self_attn(x, x, x, need_weights=True)
            w = w.detach()                                         # [B, K, K]
            ent, _, _ = _attn_reduce(w)
            self._diag_self.append((ent.cpu(), w.sum(dim=1).cpu()))
            return out
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

        if self._diag_enabled:
            out_active, w = self.self_attn(q, kv, kv, need_weights=True)
            w = w.detach()                                     # [B, active, K]
            ent, _, _ = _attn_reduce(w)
            self._diag_self.append((ent.cpu(), w.sum(dim=1).cpu()))
        else:
            out_active = self.self_attn(q, kv, kv)[0]

        out = torch.zeros(B, K, D, device=slots.device, dtype=out_active.dtype)
        out_flat = out.reshape(B * K, D)
        out_flat[active_flat] = out_active.reshape(-1, D)
        return out

    @staticmethod
    def _sim_triple(slots, active_mask):
        """(all, active-only, inactive-only) mean pairwise cos-sim, as CPU tensors."""
        with torch.no_grad():
            all_s = _slot_sim(slots)
            m = active_mask
            # Boolean-mask indexing on [B,K,D] flattens to [n, D], so re-add the
            # batch dim before computing pairwise similarity.
            act = (_slot_sim(slots[m].unsqueeze(0)) if m.sum() > 1
                   else slots.new_zeros(1))
            inact = (_slot_sim(slots[~m].unsqueeze(0)) if (~m).sum() > 1
                     else slots.new_zeros(1))
        return (all_s.cpu(), act.cpu(), inact.cpu())

    # ------------------------------------------------------------------
    def forward(self, slots, ref_raw, cache: MambaCache):
        if self.top_k is None:
            # === Dense: all slots update (reference SlotSSM) ===
            prev_slots = slots      # input state, for the per-slot delta measure
            slots = slots + self._cross_attn(slots, ref_raw)
            slots = slots + self._mamba_step(slots, cache)
            slots = slots + self._self_attn_all(slots)
            if self._diag_enabled:
                # Record the same two quantities the sparse path records, so
                # dense and sparse are directly comparable.  Dense has no gate,
                # so every slot is "active": the active column equals the all
                # column and the inactive set is empty (recorded as 0).
                self._diag_slot_delta.append(
                    (slots - prev_slots).detach().norm(dim=-1).cpu())
                all_active = torch.ones(slots.shape[:2], device=slots.device,
                                        dtype=torch.bool)
                self._diag_slot_sim.append(self._sim_triple(slots, all_active))
            return slots

        # === Sparse: only top-k active slots update ===
        B, K, D = slots.shape
        prev_slots = slots          # input state, for the per-slot delta measure
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

        # Optional third term: read the recurrent state the block wrote at the
        # PREVIOUS step.  `has_prev` is False on the first step (no cache yet),
        # so the term simply does not exist there -- the gate falls back to the
        # instantaneous score, which is the only thing it could use anyway.
        #
        # This is read BEFORE any of this step's writes, so it is genuinely the
        # state as of step t-1, and inactive slots contribute their (older)
        # state unchanged -- which is the point: age is part of what it reads.
        #
        # DETACHED, and it has to be. The step-0 scan (Mamba2.forward with
        # seqlen_offset=0) is differentiable and stores its output state in the
        # cache, so on step 1 this tensor is still attached to step 0's graph --
        # and after the training loop's per-step .backward() that graph is freed,
        # so reading it differentiably raises "backward through the graph a
        # second time". Every later step is written by the in-place step()
        # kernels and carries no graph at all, so the attachment is a step-0
        # artefact. Detaching also keeps the design honest: the readout is what
        # learns, the state is an input. (This is NOT the fix for §3's missing
        # temporal gradient -- that needs a differentiable pass over the whole
        # sequence, not a gradient into one stale step.)
        if self.state_gate is not None and has_prev:
            ssm = cache.key_value_memory_dict[layer_idx][1].detach()
            s = ssm.float().mean(dim=tuple(range(1, ssm.dim() - 1)))  # [B*K, d_state]
            s = self.state_norm(s.view(B, K, -1))
            assert s.shape[-1] == self.state_gate.shape[-1], (
                f"ssm_state last dim {s.shape[-1]} != gate readout "
                f"{self.state_gate.shape[-1]}")
            gate_scores = gate_scores + (
                s.to(gate_scores.dtype) @ self.state_gate.t()).squeeze(-1)

        # ``p`` is the differentiable gate distribution; the *selection* stays
        # hard top-k.  The gate is trained through two paths:
        #   (1) the straight-through mask below (forward = hard 0/1, backward
        #       flows through p), and
        #   (2) the entropy term, which is no longer detached.
        # Previously gate_scores reached only .topk() (indices, no gradient) and
        # a detached entropy, so the gate received ZERO gradient and stayed at
        # its xavier init — it was a fixed random projection, not a learned gate.
        p = gate_scores.softmax(dim=-1)                              # [B, K]

        if self.training and self.eps_random > 0 and torch.rand(1).item() < self.eps_random:
            active_idx = torch.stack([torch.randperm(K, device=slots.device)[:self.top_k]
                                       for _ in range(B)])
            self._gate_entropy = gate_scores.new_tensor(float(math.log(K)))
            self._gate_balance = gate_scores.new_tensor(0.0)
            # Fix A: KEEP the STE gradient on eps_random steps.  The random
            # subset's update IS computed and IS evaluated, so the loss reduction
            # genuinely says whether those slots were useful.  Flowing gradient
            # here turns this from wasted noise into eps-greedy EXPLORATION with
            # learning: the gate discovers slots it would never have picked.
            #
            # This is the fix for rich-get-richer.  The STE can only reach
            # SELECTED slots (an unselected slot's delta is never computed, so
            # there is nothing to evaluate), which is why the gate previously
            # only ever learned about slots it already chose.  eps_random is the
            # one mechanism that evaluates a slot the gate did not choose.
            _ste = True
        else:
            # Corrections steer SELECTION only; p (used for the STE mask and
            # the entropy diagnostic) stays the unbiased gate distribution, so
            # the gate MLP learns from the task loss independently of them.
            shift = None
            if self.recency_lambda > 0 and self._recency is not None:
                r = -self.recency_lambda * self._recency
                shift = r if shift is None else shift + r
            biased = gate_scores if shift is None else gate_scores + shift
            # Fix B: Gumbel exploration.  Perturb the SELECTION scores so slots
            # just below the cutoff get tried — far more informative per step than
            # eps_random's uniformly random subset.  Training-only, and the STE
            # still flows through the clean p, so this explores without biasing
            # the learning signal.  sigma is additive; the gate score spread is
            # ~0.7 (from the entropy), so keep sigma well below that.
            if self.training and self.gumbel_sigma > 0 and self._route_override is None:
                u = torch.rand_like(biased).clamp_min(1e-20)
                biased = biased + self.gumbel_sigma * (-torch.log(-torch.log(u)))
            _, active_idx = biased.topk(self.top_k, dim=1)           # [B, top_k]
            entropy = -(p * (p + 1e-9).log()).sum(dim=-1).mean()     # scalar
            # Diagnostic-only (reported by get_diagnostics and the mechanism
            # script), so detached: as a LOSS term it was useless — a near-uniform
            # softmax still picks the same top-k, since top-k depends on ORDER not
            # magnitude, so it cannot see slot starvation (93% of max entropy at
            # 0.64 coverage).  See findings §1e.
            self._gate_entropy = entropy.detach()
            # MoE-style load balancing (Switch/GShard form), so slots are not
            # starved: f from the hard mask (no grad), P differentiable, so
            # gradients reach the gate only through P — as in the literature.
            #
            # Normalised to the excess over the uniform minimum.  That minimum
            # is top_k, NOT 1: at uniform routing each of K slots is selected by
            # a fraction top_k/K of frames, so K*sum(f*P) = top_k.  It equals K
            # at full collapse.  So the raw form ranges [top_k, K], and dividing
            # by (K - top_k) maps 0 = balanced .. 1 = collapsed for any top_k.
            #
            # Normalising also matters for scale: the raw form's gradient
            # (measured 54 on gate[1].weight, top_k=8) swamps the task gradient
            # (~0.006) by ~90x at weight 0.01.
            if self.balance_weight > 0 and self.top_k < K:
                with torch.no_grad():
                    hard_oh = torch.zeros_like(p).scatter_(1, active_idx, 1.0)
                f = hard_oh.mean(dim=0)
                raw = K * (f * p.mean(dim=0)).sum()
                self._gate_balance = (raw - self.top_k) / (K - self.top_k)
            else:
                self._gate_balance = gate_scores.new_tensor(0.0)
            _ste = True

        # --- Routing interventions (diagnostics only; None = learned top-k) ---
        if self._route_override is not None:
            active_idx = self._route_override(gate_scores, active_idx, self)

        # Straight-through mask: forward value is the hard top-k mask, backward
        # passes through p, so the hard selection is trainable.  Built AFTER the
        # override so it matches the selection actually used.  Detached only
        # under a diagnostic override, where the selection is the script's, not
        # the gate's.  (eps_random steps deliberately DO carry gradient — Fix A.)
        with torch.no_grad():
            hard = torch.zeros_like(p).scatter_(1, active_idx, 1.0)
        mask_ste = (hard + p - p.detach()
                    if (_ste and self._route_override is None) else hard)

        # --- aux-loss-free balancing: accumulate per-sample TEMPORAL usage ---
        # f_i = mean over this sample's frames of the fraction of batch items
        # that activated slot i.  A temporal statistic — which is the actual
        # failure mode (within-video starvation).  The weighted aux loss could
        # only ever see cross-video diversity, since its f was averaged over the
        # batch, which here is one frame from each of B different videos.
        # seqlen_offset == 0 marks a fresh sample (the loop resets state per
        # video), so that is where the previous sample's usage is applied.
        # Recency EMA: reset at a sample boundary (seqlen_offset == 0 marks the
        # first frame, since the loop passes a fresh state per video).  Runs at
        # eval too — this is an inference-time scheduling mechanism, not a
        # trained quantity — but not under a diagnostic routing override.
        if self.recency_lambda > 0 and _ste and self._route_override is None:
            cur = hard.mean(dim=0)
            if (self._recency is None or cache.seqlen_offset == 0
                    or self._recency.device != slots.device):
                self._recency = cur
            else:
                self._recency = (self.recency_rho * self._recency
                                 + (1.0 - self.recency_rho) * cur)


        # --- Diagnostics: capture gate scores and active indices ---
        if self._diag_enabled:
            self._diag_gate_scores.append(gate_scores.detach().cpu())
            self._diag_active_idx.append(active_idx.detach().cpu())

        # --- First step: Mamba cache not yet initialised for this block.
        #     Run scan path on all K slots to populate the cache, then
        #     fall through to the mask-based path (same as before).
        if not has_prev:
            mask = torch.zeros(B, K, device=slots.device, dtype=slots.dtype).scatter_(1, active_idx, 1.0)
            active_flat = mask.reshape(-1).bool()
            # STE mask: same forward value as `mask`, gradient flows to the gate.
            mask_3d = mask_ste.unsqueeze(-1)
            slots = slots + cross_out * mask_3d
            x_all = self._slot_modulate(self.time_mixer_norm(slots)).reshape(-1, 1, D)
            full_out = self.mamba(x_all, inference_params=cache).reshape(B, K, D)
            slots = slots + full_out * mask_3d
            slots = slots + self._self_attn_sparse(slots, active_flat)
            if self._diag_enabled:
                # Record here too, so slot_delta stays index-aligned with
                # gate_scores / active_idx (this branch returns early).
                self._diag_slot_delta.append(
                    (slots - prev_slots).detach().norm(dim=-1).cpu())
                self._diag_slot_sim.append(self._sim_triple(slots, mask.bool()))
            return slots

        # === Subsequent steps: compact active-slot path -------------------
        # Operate on a dense [B, top_k, D] tensor using integer advanced
        # indexing, which produces contiguous views.  This eliminates all
        # mask multiplications and gather/scatter in attention, keeping the
        # CUDA graph fused and avoiding the ~1 ms/block sync overhead.

        tk = self.top_k
        batch_idx = torch.arange(B, device=slots.device).unsqueeze(1)   # [B, 1]

        # 3. Compact active slots — integer indexing → contiguous
        prev_active = slots[batch_idx, active_idx]                       # [B, tk, D]
        compact_cross = cross_out[batch_idx, active_idx]                 # [B, tk, D]

        # 4. Cross-attn update (dense on compact, no mask)
        compact_slots = prev_active + compact_cross

        # 5. Mamba on compact active slots
        x_compact = self._slot_modulate(
            self.time_mixer_norm(compact_slots), active_idx).reshape(-1, 1, D)
        idx_flat = active_idx.reshape(-1)                                   # [B*tk]
        kv = cache.key_value_memory_dict[layer_idx]
        conv_active = kv[0][idx_flat]       # [B*tk, C, d_conv] — integer idx → contiguous
        ssm_active = kv[1][idx_flat]        # [B*tk, H, hdim, d_state]
        out_active, _, _ = self.mamba.step(x_compact, conv_active, ssm_active)
        kv[0][idx_flat] = conv_active       # scatter updated states back
        kv[1][idx_flat] = ssm_active
        compact_slots = compact_slots + out_active.squeeze(1).reshape(B, tk, D)

        # 6. Write the compact update back BEFORE self-attn, so the KV reflects
        #    post-update slots exactly as the dense path does.  Inactive slots
        #    are untouched, so only the active rows were stale.  Doing this
        #    scatter *after* self-attn (as before) fed it PRE-update KV, which
        #    made the sparse path a different function from dense even at
        #    top_k=K — verified: top_k=32 then differed from dense by 0.6
        #    relative, and matches to 0 once the KV is fresh.
        kv_source = slots        # pre-update states, for legacy_stale_kv below
        slots = torch.index_put(slots, (batch_idx, active_idx), compact_slots)

        # 7. Self-attn: compact Q queries full slots as KV (read-only memory)
        compact_q = self.space_attn_norm(compact_slots)                # [B, tk, D]
        # ``legacy_stale_kv`` reproduces the pre-fix forward exactly (bfaa73a):
        # it built this KV from the PRE-update slot states, so the active rows
        # were stale.  This exists only so a checkpoint trained under that
        # forward can be evaluated under the code it was trained with.  Without
        # it, such a checkpoint is a train/test forward mismatch and its AUC
        # under the fixed code is comparable to nothing — see findings §1.5.
        # It is never set during training or normal evaluation.
        full_kv = self.space_attn_norm(kv_source if self.legacy_stale_kv
                                       else slots)                      # [B, K, D]
        if self._diag_enabled:
            sa_out, w = self.self_attn(compact_q, full_kv, full_kv,
                                       need_weights=True)              # w: [B, tk, K]
            w = w.detach()
            ent, _, _ = _attn_reduce(w)
            read_mass = w.sum(dim=1)                                   # [B, K] mass each slot receives
            self._diag_self.append((ent.cpu(), read_mass.cpu()))
        else:
            sa_out = self.self_attn(compact_q, full_kv, full_kv)[0]    # [B, tk, D]
        compact_slots = compact_slots + sa_out

        # 8. Final write-back, weighted by the straight-through mask so the gate
        #    receives gradient.  ``w_ste`` is exactly 1.0 in the forward pass, so
        #    this is identical to the hard write while letting gradients reach p.
        #
        #    The pre-fix forward wrote ``compact_slots`` directly instead of
        #    round-tripping through prev + (new - prev) * 1.0.  That looked like
        #    it should matter under bf16 (catastrophic cancellation), so it was
        #    implemented as a separate legacy flag and measured: it changes the
        #    result by EXACTLY nothing -- both forms give 0.7583 pooled AUC on
        #    the seed-42 40-video subset, to four decimals, with identical mean
        #    video AUC.  So the branch was removed rather than kept as dead code.
        w_ste = mask_ste.gather(1, active_idx).unsqueeze(-1)           # [B, tk, 1]
        slots = torch.index_put(
            slots, (batch_idx, active_idx),
            prev_active + (compact_slots - prev_active) * w_ste)
        if self._diag_enabled:
            # Per-slot L2 movement this step.  Inactive slots must read exactly
            # 0 for the frozen-memory claim to hold.
            self._diag_slot_delta.append(
                (slots - prev_slots).detach().norm(dim=-1).cpu())
            self._diag_slot_sim.append(self._sim_triple(slots, hard.bool()))
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
        balance_weight: float = 0.0,
        recency_lambda: float = 0.0,
        recency_rho: float = 0.9,
        gumbel_sigma: float = 0.0,
        logit_scale: float = 1.0,
        logit_scale_learnable: bool = False,
        slot_input_gain: bool = False,
        slot_input_proj_rank: int = 0,
        slot_state_decay: bool = False,
        gate_reads_state: bool = False,
        gate_state_init: float = 0.0,
        # Gradient-free per-slot logit bias for inverted attention; 0 = off.
        slot_bias_gamma: float = 0.0,
        slot_bias_cap: float = 5.0,
        # Fixed 2D sincos positional embedding on the cross-attention's tokens.
        # See SlotSSMBlock for why it exists and what it can/cannot do at init.
        slot_pos_pe: bool = False,
    ):
        super().__init__()
        _require_mamba()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.top_k = top_k
        # NOTE: not gated on top_k, unlike eps_random/gumbel/balance/recency.
        # Those live in the sparse gate, so they can be sparse-only. The flat
        # cross-attention this corrects is in the shared block, upstream of the
        # dense/sparse split, so it applies to both. See findings 15.9.
        self.logit_scale = float(logit_scale)
        self.logit_scale_learnable = bool(logit_scale_learnable)
        # Per-slot differentiation flags (see SlotSSMBlock for the rationale).
        # NOT gated on top_k: the recurrent state homogenises on the dense path
        # too, so these apply to both.
        self.slot_input_gain = bool(slot_input_gain)
        self.slot_input_proj_rank = int(slot_input_proj_rank)
        self.slot_state_decay = bool(slot_state_decay)
        # Gate-reads-state is the ONE flag here that is sparse-only: it adds a
        # term to a gate score, and dense has no gate. See SlotSSMBlock.
        self.gate_reads_state = bool(gate_reads_state)
        self.gate_state_init = float(gate_state_init)

        self.slots_init = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)

        self.blocks = nn.ModuleList([
            SlotSSMBlock(
                slot_dim=slot_dim, input_dim=input_dim, top_k=top_k,
                mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand, mamba_version=mamba_version,
                num_heads=num_heads, block_idx=i,
                eps_random=eps_random if top_k is not None else 0.0,
                use_inverted_attention=use_inverted_attention,
                balance_weight=balance_weight if top_k is not None else 0.0,
                num_slots=num_slots,
                recency_lambda=recency_lambda if top_k is not None else 0.0,
                recency_rho=recency_rho,
                gumbel_sigma=gumbel_sigma if top_k is not None else 0.0,
                logit_scale=logit_scale,
                logit_scale_learnable=logit_scale_learnable,
                slot_input_gain=slot_input_gain,
                slot_input_proj_rank=slot_input_proj_rank,
                slot_state_decay=slot_state_decay,
                gate_reads_state=gate_reads_state,
                gate_state_init=gate_state_init,
                slot_bias_gamma=slot_bias_gamma,
                slot_bias_cap=slot_bias_cap,
                slot_pos_pe=slot_pos_pe,
            )
            for i in range(num_blocks)
        ])
        self._entropy = torch.tensor(0.0)  # populated during forward
        self._balance = torch.tensor(0.0)  # populated during forward
        self._slot_mass_min = torch.tensor(float("nan"))
        self._slot_mass_mean = torch.tensor(float("nan"))
        self._slot_usage_frac = torch.tensor(float("nan"))

        # --- Diagnostic collection (opt-in, off by default) ---
        self._diag_slots: list | None = None   # final slot states [B, K, D] per step

    def enable_diagnostics(self, grid=None, collect_slots=False):
        """Enable per-step collection of gate scores, active indices, and slot states.

        ``grid`` is the (rows, cols) spatial layout of the patch tokens, needed
        for attention centroids.  Pass None to skip centroid/Self-attn capture.

        ``collect_slots`` stores the full ``[B, K, D]`` slot state every step,
        which is ~1 GB over a few thousand frames at K=32/D=512 — it defaults to
        off, since ``slot_delta`` already captures per-slot movement.
        """
        for blk in self.blocks:
            blk._diag_enabled = True
            blk._diag_gate_scores = []
            blk._diag_active_idx = []
            blk._diag_cross = []
            blk._diag_self = []
            blk._diag_slot_delta = []
            blk._diag_slot_sim = []
            blk._diag_grid = grid
        self._diag_slots = [] if collect_slots else None

    def disable_diagnostics(self):
        """Disable diagnostic collection and free stored data."""
        for blk in self.blocks:
            blk._diag_enabled = False
            blk._diag_gate_scores = []
            blk._diag_active_idx = []
            blk._diag_cross = []
            blk._diag_self = []
            blk._diag_slot_delta = []
            blk._diag_slot_sim = []
        self._diag_slots = []

    def set_route_override(self, fn):
        """Install ``fn(gate_scores, active_idx, block) -> active_idx`` on every block.

        Used by routing interventions (random / round-robin / frozen).  The block
        is passed so an override can keep per-block state.  ``None`` restores the
        learned top-k path.
        """
        for blk in self.blocks:
            blk._route_override = fn

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
            # Added by diag_sparse_gate_topk.py — empty when grid was not passed
            "cross": [list(blk._diag_cross) for blk in self.blocks],
            "self_attn": [list(blk._diag_self) for blk in self.blocks],
            "slot_delta": [list(blk._diag_slot_delta) for blk in self.blocks],
            "slot_sim": [list(blk._diag_slot_sim) for blk in self.blocks],
        }

    def forward(self, patches, cache: MambaCache | None = None):
        B = patches.shape[0]
        if cache is None:
            cache = MambaCache()

        slots = self.slots_init.expand(B, -1, -1)
        ent = 0.0
        bal = 0.0
        usage = torch.tensor(0.0)
        for blk in self.blocks:
            slots = blk(slots, patches, cache)
            if blk.slot_state_decay:
                # Per-slot decay on the recurrent state, applied ONCE here rather
                # than in each of the block's three Mamba call sites. The cache
                # state is [B*K, H, hdim, d_state] with slots flattened as (b, k),
                # so repeat the [K] decay B times to line up.
                #
                # Replaces the dict entry rather than mul_()ing in place: the
                # first-step branch runs the differentiable scan, so the cached
                # tensor may carry a graph and an in-place op on it would either
                # error or silently corrupt it.
                kv = cache.key_value_memory_dict.get(blk.mamba.layer_idx)
                if kv is not None:
                    d = blk.slot_decay().repeat(B).view(-1, 1, 1, 1)
                    cache.key_value_memory_dict[blk.mamba.layer_idx] = (kv[0],
                                                                       kv[1] * d)
            if blk.top_k is not None:
                ent = ent + blk._gate_entropy
                bal = bal + blk._gate_balance
            usage = usage + blk._attn_usage
        self._entropy = ent    # training loop reads this
        self._balance = bal    # load-balancing term (0 unless balance_weight > 0)
        # Summed over blocks, matching `_balance`. 0 unless the attention is
        # inverted. The training loop weights it by `attn_usage_weight`.
        self._attn_usage = usage

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
        balance_weight: float = 0.0,
        recency_lambda: float = 0.0,
        recency_rho: float = 0.9,
        gumbel_sigma: float = 0.0,
        # Inverted attention (SlotSSM reference repo style)
        use_inverted_attention: bool = False,
        # Multiplies the cross-attention logits before the softmax. 1.0 = neutral.
        # Applies to dense and sparse alike (shared block). See findings 15.9.
        logit_scale: float = 1.0,
        # Learn one logit scale PER HEAD instead of fixing it, initialized to
        # logit_scale. Lets the model report where it wants to sit.
        logit_scale_learnable: bool = False,
        # Per-slot differentiation (see SlotSSMBlock). Off by default so nothing
        # existing changes.
        slot_input_gain: bool = False,
        slot_input_proj_rank: int = 0,
        slot_state_decay: bool = False,
        # Let the sparse gate score also read the previous step's SSM state.
        # Sparse-only (dense has no gate). See SlotSSMBlock for the measurement
        # that says this is worth doing only alongside `slot_input_proj_rank`.
        gate_reads_state: bool = False,
        gate_state_init: float = 0.0,
        # Gradient-free per-slot logit bias for inverted attention. Inverted
        # only; 0 = off. See MultiHeadAttention.
        slot_bias_gamma: float = 0.0,
        slot_bias_cap: float = 5.0,
        # Fixed 2D sincos positional embedding on the cross-attention's tokens.
        # Off by default; see SlotSSMBlock for the measurement behind it.
        slot_pos_pe: bool = False,
        # --- Slot pooling: how the classifier reads the 32 slots -----------
        # 'dot'  (default) -- the historical behaviour, kept BIT-IDENTICAL: a
        #        bare `nn.Parameter` query, `softmax(slots . q / sqrt(D))`.
        # 'attn' -- LayerNorm the slots, then score them with attention pooling
        #        (Ilse et al. 2018, ABMIL), gated or not by `slot_pool_gated`:
        #            plain (Eq. 8): a_k ~ exp( w^T tanh(V h_k) )
        #            gated (Eq. 9): a_k ~ exp( w^T ( tanh(V h_k) * sigmoid(U h_k) ) )
        #        See the measurement at the construction site for why.
        slot_pool: str = 'dot',
        slot_pool_hidden: int = 128,
        # Gating on top of 'attn': the two forms then differ by exactly one
        # projection (U), which is what makes them a clean pair. Default True --
        # gated is the form the literature recommends for the regime we measured
        # (tanh near-linear on [-1, 1]) -- but a run should set it explicitly.
        slot_pool_gated: bool = True,
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

        _init_rng = torch.get_rng_state()
        _init_cuda_rng = (torch.cuda.get_rng_state_all()
                          if torch.cuda.is_available() else None)

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
                balance_weight=balance_weight if is_sparse else 0.0,
                recency_lambda=recency_lambda if is_sparse else 0.0,
                recency_rho=recency_rho,
                gumbel_sigma=gumbel_sigma if is_sparse else 0.0,
                logit_scale=logit_scale,
                logit_scale_learnable=logit_scale_learnable,
                slot_input_gain=slot_input_gain,
                slot_input_proj_rank=slot_input_proj_rank,
                slot_state_decay=slot_state_decay,
                gate_reads_state=gate_reads_state,
                gate_state_init=gate_state_init,
                slot_bias_gamma=slot_bias_gamma,
                slot_bias_cap=slot_bias_cap,
                slot_pos_pe=slot_pos_pe,
            )

            # --- Slot pooling: how the classifier reads the 32 slots ---------
            #
            # `slot_pool = 'dot'` IS THE HISTORY AND IS BIT-IDENTICAL to the
            # pre-change model. It is also a measured FAILURE, which is why
            # `'attn'` exists:
            #
            #   The query set the softmax logit scale directly, because it was a
            #   bare Parameter multiplied into the slots, with no projection and
            #   no normalisation: logit = q.s/sqrt(D) with ||q|| = 0.02*sqrt(512)
            #   = 0.4525 against a slot RMS of ~1.8, i.e. logits spanning about
            #   +-0.04. A softmax over 32 slots with a 0.04 spread is uniform by
            #   construction, and it measured uniform: `eff = exp(entropy)` came
            #   out at 31.95-31.99 of a maximum of 32, with max weight 0.032-0.035
            #   against the uniform 1/32 = 0.03125, in all ten arms of the
            #   logit-scale sweep, both architectures, every scale. So the
            #   classifier's input was an UNWEIGHTED MEAN of all 32 slots and the
            #   "attention-pool" was an arithmetic mean.
            #   It never recovered: ||slot_query|| after 50 epochs was 0.968-1.005x
            #   its init value in every one of those arms, despite carrying
            #   optimizer state. `_weights_init` cannot rescue it either -- that
            #   function only visits nn.Modules, so a bare Parameter keeps
            #   whatever it was handed. That is the same bug class as the fused
            #   `in_proj_weight` of 15.9.1 (half the intended logit scale) and the
            #   reason `state_gate` had to be designed around it: THIS IS THE
            #   THIRD INSTANCE.
            #
            # `'attn'` is the STANDARD fix from the MIL literature, not an
            # invention: attention pooling over a bag of instances, Ilse et al.
            # 2018 (ABMIL, ICML). Both of its forms are available, differing by
            # exactly one projection so they pair cleanly:
            #
            #     plain (Eq. 8):  a_k ~ exp( w^T tanh(V h_k) )
            #     gated (Eq. 9):  a_k ~ exp( w^T ( tanh(V h_k) * sigmoid(U h_k) ) )
            #
            # Gated is the DEFAULT and the form the paper recommends for our
            # regime, but the pair is meant to be run explicitly -- measured at
            # init on a trained temporal model they are NOT equivalent, and the
            # gated one is the FLATTER of the two (eff 28.5 vs 21.1 of 32), because
            # sigmoid(U h) ~ 0.5 at init ATTENUATES the score rather than spreading
            # it. `tests/_check_slot_pool.py` reports both.
            #
            # Two properties matter, and both are things 'dot' structurally lacked:
            #
            #  * The logit scale is set by the INIT SCHEME rather than by a
            #    hand-picked constant, because the score goes through PROJECTIONS
            #    (V, U, w are nn.Linear) instead of a raw dot product with a bare
            #    vector. Xavier on those gives logits of order 1 -- measured
            #    (tests/_check_slot_pool.py) at sd 0.95 against 'dot''s 0.038 --
            #    which is the regime where the softmax actually selects.
            #  * The LayerNorm makes the pool invariant to slot magnitude, which
            #    under 'dot' drifted with `logit_scale` and left the pool's
            #    effective temperature an uncontrolled function of the
            #    cross-attention sharpening.
            #
            # GATED rather than the plain tanh form, for a reason the paper states
            # and our own measurement independently lands on. Ilse et al. add the
            # gate because "the tanh(.) non-linearity could be inefficient to
            # learn complex relations", since "tanh(x) is approximately linear for
            # x in [-1, 1]", which "could limit the final expressiveness"; the gate
            # "introduces a learnable non-linearity that potentially removes the
            # troublesome linearity in tanh(.)". Our pool logits measure at sd
            # 0.95, so the pre-tanh activations sit INSIDE [-1, 1] -- tanh is in
            # its near-linear region, which is exactly the inefficiency the gate
            # exists to fix. So this is the cited answer to the regime we measured,
            # and it is one mechanism rather than a choice needing an ablation.
            #
            # Deliberately NO temperature on the pooling softmax. Contrastive
            # learning learns one (CLIP's logit_scale, the idiom this repo's
            # `logit_scale` copies), but MIL pooling does not, and a tuned constant
            # here would be a non-standard addition requiring its own ablation.
            #
            # NOTE: the modes have different parameter names, so checkpoints are
            # NOT portable between them (documented, like `logit_scale_learnable`).
            self.slot_pool = str(slot_pool)
            self.slot_pool_hidden = int(slot_pool_hidden)
            self._last_pool_attn = None      # the pool's weights, for diagnostics
            if self.slot_pool == 'attn':
                # Gated is a FLAG on 'attn', not a separate mode, so the plain
                # and gated arms differ by exactly one projection (U) and nothing
                # else -- which is what makes them a clean ablation pair.
                #   plain: a_k ~ exp( w^T tanh(V h_k) )
                #   gated: a_k ~ exp( w^T ( tanh(V h_k) * sigmoid(U h_k) ) )
                self.slot_pool_norm = nn.LayerNorm(slot_dim)
                self.slot_pool_V = nn.Linear(slot_dim, self.slot_pool_hidden)
                self.slot_pool_w = nn.Linear(self.slot_pool_hidden, 1)
                self.slot_pool_gated = bool(slot_pool_gated)
                self.slot_pool_U = (nn.Linear(slot_dim, self.slot_pool_hidden)
                                    if self.slot_pool_gated else None)
                self.slot_query = None
            elif self.slot_pool == 'dot':
                self.slot_pool_gated = False
                self.slot_pool_U = None
                self.slot_query = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
            else:
                raise ValueError(
                    f"slot_pool must be 'dot' or 'attn', got {slot_pool!r}")
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
            _restore_rng(_init_rng, _init_cuda_rng)
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

        _restore_rng(_init_rng, _init_cuda_rng)
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
            # Learned attention-pool: query attends to slots. See __init__ for
            # why 'dot' measures as an arithmetic mean and 'attn' is the fix.
            if self.slot_pool == 'attn':
                h = self.slot_pool_norm(slots)                    # [B, K, D]
                # Attention pooling, Ilse et al. 2018: Eq. 8 plain, Eq. 9 gated.
                g = torch.tanh(self.slot_pool_V(h))
                if self.slot_pool_U is not None:
                    g = g * torch.sigmoid(self.slot_pool_U(h))
                scores = self.slot_pool_w(g).squeeze(-1)          # [B, K]
            else:
                D = slots.shape[-1]
                scores = (slots * self.slot_query).sum(dim=-1) / (D ** 0.5)
            attn = scores.softmax(dim=-1)                       # [B, K]
            # Kept for diagnostics, like MultiHeadAttention._last_attn: the pool
            # weights are the only way to see WHICH slots reach the classifier.
            self._last_pool_attn = attn.detach()
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

        # --- every head starts from the SAME init ---------------------------
        # `set_deterministic(cfg.seed)` runs ONCE at the start of the run, and
        # each `ClsVJEPA(...)` below draws from that same global RNG stream. So
        # head 0 consumes draws 1..N1, head 1 consumes N1+1..N1+N2, and so on:
        # the run is reproducible, but the heads within it are NOT identical
        # (measured: `slots_init` up to 0.125 apart, tests/_head_init_check.py).
        #
        # That makes a multi-head run a fair EXPERIMENT but not a controlled
        # ABLATION -- two arms would differ by the flag AND by a random weight
        # draw, and that draw is worth ~0.02 AUC (findings 16.4), the same size
        # as the effects being chased. It is what made the gate arms look
        # significantly worse than their twins when a same-weights A/B showed
        # the readout moves only 0.7-6% of routes.
        #
        # Saving the state before the loop and restoring it before each head
        # gives every head draws 1..N, so all arms start identical.
        #
        # BLAST RADIUS -- read before re-deriving any run from its seed.
        #
        # This loop is the head-level half and is a no-op for a SINGLE head
        # (restoring before the only head restores to where it already was).
        # The other half is in ClsVJEPA.__init__, which rewinds the RNG before
        # `self.apply(_weights_init)`; construction consumes draws, so that
        # rewind changes the xavier values for EVERY model, single head
        # included. Measured: the stock module's init logit_sd moved
        # 0.971 -> 0.986 (tests/_check_init_fix.py).
        #
        # So: every run started after this change gets different weights than
        # it would have before it, at the same seed. Existing checkpoints are
        # unaffected -- init cannot touch a loaded model -- but a pre-change run
        # cannot be re-derived from its seed, only from its checkpoint.
        _rng = torch.get_rng_state()
        _cuda_rng = (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None)

        for head_cfg in heads_configs:
            name = head_cfg["name"]
            if name in self.heads:
                raise ValueError(f"Duplicate head name: {name}")
            self.head_configs[name] = dict(head_cfg)

            torch.set_rng_state(_rng)
            if _cuda_rng is not None:
                torch.cuda.set_rng_state_all(_cuda_rng)

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
                balance_weight=head_cfg.get("balance_weight", 0.0),
                recency_lambda=head_cfg.get("recency_lambda", 0.0),
                recency_rho=head_cfg.get("recency_rho", 0.9),
                gumbel_sigma=head_cfg.get("gumbel_sigma", 0.0),
                use_inverted_attention=head_cfg.get("use_inverted_attention", False),
                logit_scale=head_cfg.get("logit_scale", 1.0),
                logit_scale_learnable=head_cfg.get("logit_scale_learnable", False),
                slot_input_gain=head_cfg.get("slot_input_gain", False),
                slot_input_proj_rank=head_cfg.get("slot_input_proj_rank", 0),
                slot_state_decay=head_cfg.get("slot_state_decay", False),
                gate_reads_state=head_cfg.get("gate_reads_state", False),
                gate_state_init=head_cfg.get("gate_state_init", 0.0),
                slot_bias_gamma=head_cfg.get("slot_bias_gamma", 0.0),
                slot_bias_cap=head_cfg.get("slot_bias_cap", 5.0),
                slot_pos_pe=head_cfg.get("slot_pos_pe", False),
                slot_pool=head_cfg.get("slot_pool", "dot"),
                slot_pool_hidden=head_cfg.get("slot_pool_hidden", 128),
                slot_pool_gated=head_cfg.get("slot_pool_gated", True),
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
        balance_weight=cfg.get("balance_weight", 0.0),
        recency_lambda=cfg.get("recency_lambda", 0.0),
        recency_rho=cfg.get("recency_rho", 0.9),
        gumbel_sigma=cfg.get("gumbel_sigma", 0.0),
        use_inverted_attention=cfg.get("use_inverted_attention", False),
        logit_scale=cfg.get("logit_scale", 1.0),
        logit_scale_learnable=cfg.get("logit_scale_learnable", False),
        slot_input_gain=cfg.get("slot_input_gain", False),
        slot_input_proj_rank=cfg.get("slot_input_proj_rank", 0),
        slot_state_decay=cfg.get("slot_state_decay", False),
        gate_reads_state=cfg.get("gate_reads_state", False),
        gate_state_init=cfg.get("gate_state_init", 0.0),
        slot_bias_gamma=cfg.get("slot_bias_gamma", 0.0),
        slot_bias_cap=cfg.get("slot_bias_cap", 5.0),
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

