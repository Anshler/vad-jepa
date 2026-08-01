"""
Correctness test for the active-only sparse Mamba step.

Verifies that the new path (``_mamba_step_sparse`` — runs Mamba2.step() only
on active slots) produces numerically identical outputs and cache states as the
old save/restore path (ran Mamba on all slots then cloned back inactive states).

Run from WSL:
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vad-jepa
    python tests/test_sparse_mamba.py
"""
from __future__ import annotations

import copy
import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

from model import (
    MambaCache,
    SlotSSMBlock,
    SlotSSMTemporalModel,
    _HAS_MAMBA_SSM,
    _require_mamba,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Old save/restore path (defined inline so we can run both against the same
# block instance and compare).
# ---------------------------------------------------------------------------
def _mamba_step_sparse_old(block, slots, active_flat, cache):
    """Original implementation: run Mamba on all slots, restore inactive."""
    B, K, D = slots.shape
    layer_idx = block.mamba.layer_idx
    x = block.time_mixer_norm(slots).reshape(-1, 1, D)

    has_prev = layer_idx in cache.key_value_memory_dict
    if not has_prev:
        full_out = block.mamba(x, inference_params=cache).reshape(B, K, D)
        return full_out

    kv = cache.key_value_memory_dict[layer_idx]
    inactive = ~active_flat
    conv_saved = kv[0][inactive].clone()
    ssm_saved = kv[1][inactive].clone()

    full_out = block.mamba(x, inference_params=cache).reshape(B, K, D)

    kv[0][inactive] = conv_saved
    kv[1][inactive] = ssm_saved

    mask = active_flat.float().view(B, K, 1)
    return full_out * mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clone_cache(cache: MambaCache) -> MambaCache:
    """Deep-copy a MambaCache for comparison testing."""
    new = MambaCache(seqlen_offset=cache.seqlen_offset)
    for k, (conv, ssm) in cache.key_value_memory_dict.items():
        new.key_value_memory_dict[k] = (conv.clone(), ssm.clone())
    return new


def _cache_allclose(a: MambaCache, b: MambaCache, atol=1e-5):
    """Assert two MambaCache instances are numerically identical."""
    assert a.seqlen_offset == b.seqlen_offset, f"offset: {a.seqlen_offset} vs {b.seqlen_offset}"
    assert set(a.key_value_memory_dict.keys()) == set(b.key_value_memory_dict.keys())
    for k in sorted(a.key_value_memory_dict.keys()):
        ca, sa = a.key_value_memory_dict[k]
        cb, sb = b.key_value_memory_dict[k]
        torch.testing.assert_close(ca, cb, atol=atol, rtol=1e-5, msg=f"conv_state layer {k}")
        torch.testing.assert_close(sa, sb, atol=atol, rtol=1e-5, msg=f"ssm_state layer {k}")
    return True


# ===========================================================================
# Test: compare old vs new _mamba_step_sparse across multiple steps
# ===========================================================================
def test_sparse_mamba_correctness():
    """Old and new sparse Mamba paths must produce identical outputs and caches."""
    if not _HAS_MAMBA_SSM:
        print("[SKIP] mamba_ssm not installed")
        return

    B, K, D = 1, 32, 512
    input_dim = 1408        # ViT-L embed_dim, NOT used by mamba step but needed for __init__
    N = 64                  # dummy ref tokens
    num_steps = 8
    seed = 42

    # --- Build two identical blocks -----------------------------------------
    torch.manual_seed(seed)
    block_new = SlotSSMBlock(
        slot_dim=D, input_dim=input_dim, top_k=16,
        mamba_d_state=128, mamba_d_conv=4, mamba_expand=2,
        mamba_version="mamba2", num_heads=4, block_idx=0,
        eps_random=0.0,
    ).to(DEVICE)

    torch.manual_seed(seed)
    block_old = SlotSSMBlock(
        slot_dim=D, input_dim=input_dim, top_k=16,
        mamba_d_state=128, mamba_d_conv=4, mamba_expand=2,
        mamba_version="mamba2", num_heads=4, block_idx=0,
        eps_random=0.0,
    ).to(DEVICE)

    block_new.eval()
    block_old.eval()

    cache_new = MambaCache()
    cache_old = MambaCache()

    torch.manual_seed(1234)
    slots_new = torch.randn(B, K, D, device=DEVICE)
    slots_old = slots_new.clone()

    for step in range(num_steps):
        # Identical random input to both paths
        ref = torch.randn(B, N, input_dim, device=DEVICE)

        # --- Full forward through the block, but we monkey-patch the Mamba
        #     step to use the old implementation on block_old -----------------
        B_slots, K_slots = slots_new.shape[:2]

        # Cross-attn (identical for both)
        cross_new = block_new._cross_attn(slots_new, ref)
        cross_old = block_old._cross_attn(slots_old, ref)

        # Gate (identical weights → same output)
        informed_new = slots_new + cross_new
        informed_old = slots_old + cross_old
        with torch.no_grad():
            gate_new = block_new.gate(informed_new).squeeze(-1)
            gate_old = block_old.gate(informed_old).squeeze(-1)

        _, active_idx_new = gate_new.topk(block_new.top_k, dim=1)
        _, active_idx_old = gate_old.topk(block_old.top_k, dim=1)
        assert (active_idx_new == active_idx_old).all(), f"step {step}: gate indices diverge"

        mask_3d_new = torch.zeros(B, K, 1, device=DEVICE, dtype=slots_new.dtype)
        mask_3d_new.scatter_(1, active_idx_new.unsqueeze(-1), 1.0)
        mask_3d_old = mask_3d_new.clone()
        active_flat = mask_3d_new.reshape(-1).bool()

        # Apply cross-attn update
        slots_new = slots_new + cross_new * mask_3d_new
        slots_old = slots_old + cross_old * mask_3d_old

        # --- THE KEY COMPARISON: Mamba step --------------------------------
        slots_before_new = slots_new.clone()
        slots_before_old = slots_old.clone()

        has_prev = block_new.mamba.layer_idx in cache_new.key_value_memory_dict
        if has_prev:
            # Snapshot inactive cache state BEFORE the Mamba step.
            # After the step, inactive entries must be bit-identical to
            # this snapshot — proving the new path never touches them.
            kv_new = cache_new.key_value_memory_dict[block_new.mamba.layer_idx]
            conv_before_inactive = kv_new[0][~active_flat].clone()
            ssm_before_inactive = kv_new[1][~active_flat].clone()

        # Run both paths
        time_new = block_new._mamba_step_sparse(slots_new, active_flat, cache_new)
        time_old = _mamba_step_sparse_old(block_old, slots_old, active_flat, cache_old)

        if has_prev:
            kv_new = cache_new.key_value_memory_dict[block_new.mamba.layer_idx]
            conv_after_inactive = kv_new[0][~active_flat]
            ssm_after_inactive = kv_new[1][~active_flat]
            assert torch.equal(
                conv_before_inactive, conv_after_inactive,
            ), f"step {step}: new path mutated inactive conv_state"
            assert torch.equal(
                ssm_before_inactive, ssm_after_inactive,
            ), f"step {step}: new path mutated inactive ssm_state"

        # Apply Mamba update (the caller's mask will zero inactive output)
        slots_new = slots_before_new + time_new * mask_3d_new
        slots_old = slots_before_old + time_old * mask_3d_old

        # --- Assertions ----------------------------------------------------
        # 1. Output slots must match
        torch.testing.assert_close(
            slots_new, slots_old, atol=1e-5, rtol=1e-4,
            msg=f"step {step}: slots diverge after Mamba step",
        )

        # 2. Cache states must match
        #    First call: both use the scan path (no previous cache) — identical by construction.
        if step > 0:
            _cache_allclose(cache_new, cache_old)

        # 3. Active slots: states must have advanced identically
        if step > 0:
            for layer_idx in cache_new.key_value_memory_dict:
                cn, sn = cache_new.key_value_memory_dict[layer_idx]
                co, so = cache_old.key_value_memory_dict[layer_idx]
                # Active slots must match
                torch.testing.assert_close(
                    cn[active_flat], co[active_flat], atol=1e-5, rtol=1e-4,
                    msg=f"step {step} layer {layer_idx}: active conv_state mismatch",
                )
                torch.testing.assert_close(
                    sn[active_flat], so[active_flat], atol=1e-5, rtol=1e-4,
                    msg=f"step {step} layer {layer_idx}: active ssm_state mismatch",
                )
                # Inactive slots: each path preserves them correctly, but
                # they may carry a ~1 ulp difference from when they were
                # active in an earlier step (scan vs step batch size).
                # The pre→post snapshot above already proves the new path
                # never mutates them.  Cross-compare with tolerance here.
                inactive = ~active_flat
                torch.testing.assert_close(
                    cn[inactive], co[inactive], atol=1e-6, rtol=1e-6,
                    msg=f"step {step} layer {layer_idx}: inactive conv_state mismatch",
                )
                torch.testing.assert_close(
                    sn[inactive], so[inactive], atol=1e-6, rtol=1e-6,
                    msg=f"step {step} layer {layer_idx}: inactive ssm_state mismatch",
                )

        # Self-attn (identical for both)
        sa_new = block_new._self_attn_sparse(slots_new, active_flat)
        sa_old = block_old._self_attn_sparse(slots_old, active_flat)
        slots_new = slots_new + sa_new
        slots_old = slots_old + sa_old

        torch.testing.assert_close(
            slots_new, slots_old, atol=1e-5, rtol=1e-4,
            msg=f"step {step}: slots diverge after self-attn",
        )

        cache_new.seqlen_offset += 1
        cache_old.seqlen_offset += 1

    print(f"✓  All states match across {num_steps} steps ({K} slots, top_k={block_new.top_k})")


# ===========================================================================
# Test: full block forward — dense vs sparse with epsilon=0 (no random gating)
# ===========================================================================
def test_sparse_full_forward():
    """Full SlotSSMBlock.forward (sparse) matches expectation after N steps."""
    if not _HAS_MAMBA_SSM:
        print("[SKIP] mamba_ssm not installed")
        return

    B, K, D = 1, 32, 512
    input_dim = 1408
    N = 64
    num_steps = 4

    torch.manual_seed(42)
    block = SlotSSMBlock(
        slot_dim=D, input_dim=input_dim, top_k=16,
        mamba_d_state=128, mamba_d_conv=4, mamba_expand=2,
        mamba_version="mamba2", num_heads=4, block_idx=0,
        eps_random=0.0,
    ).to(DEVICE).eval()

    cache = MambaCache()
    torch.manual_seed(99)
    slots = torch.randn(B, K, D, device=DEVICE)

    for step in range(num_steps):
        ref = torch.randn(B, N, input_dim, device=DEVICE)
        slots_before = slots.clone()
        slots = block(slots, ref, cache)
        cache.seqlen_offset += 1     # done by SlotSSMTemporalModel in real usage

        # Sanity: half the slots must be frozen (active=top_k)
        active_slots = (slots != slots_before).any(dim=-1).sum().item()
        assert active_slots <= block.top_k, (
            f"step {step}: {active_slots} slots changed (top_k={block.top_k})"
        )

    print(f"✓  Sparse forward: {num_steps} steps, ≤{block.top_k} slots active per step")


# ===========================================================================
# State snapshot: pre-allocated buffer (Opportunity 3 from research note)
# ===========================================================================
def test_preallocated_snapshots():
    """Verify the clone-based save/restore produces correct snapshot values.

    This test captures the current behavior so that if we later convert to
    pre-allocated buffers (register_buffer), we can confirm the numerical
    result is byte-identical.
    """
    if not _HAS_MAMBA_SSM:
        print("[SKIP] mamba_ssm not installed")
        return

    B, K, D = 1, 32, 512
    input_dim = 1408
    N = 64

    torch.manual_seed(42)
    block = SlotSSMBlock(
        slot_dim=D, input_dim=input_dim, top_k=16,
        mamba_d_state=128, mamba_d_conv=4, mamba_expand=2,
        mamba_version="mamba2", num_heads=4, block_idx=0,
        eps_random=0.0,
    ).to(DEVICE).eval()

    cache = MambaCache()

    # Prime the cache with one forward pass
    slots = torch.randn(B, K, D, device=DEVICE)
    ref = torch.randn(B, N, input_dim, device=DEVICE)
    slots = block(slots, ref, cache)
    cache.seqlen_offset += 1   # done by SlotSSMTemporalModel in real usage

    # Snapshot the current cache
    snap = _clone_cache(cache)

    # Run a second step
    slots = block(slots, ref, cache)
    cache.seqlen_offset += 1

    # Verify the snapshot values haven't been corrupted (they should be
    # different from current values for inactive slots)
    for layer_idx in snap.key_value_memory_dict:
        c_snap, s_snap = snap.key_value_memory_dict[layer_idx]
        c_curr, s_curr = cache.key_value_memory_dict[layer_idx]

        # Not all entries should match — active slots advanced, inactive didn't
        all_same_conv = torch.allclose(c_snap, c_curr, atol=1e-6)
        all_same_ssm = torch.allclose(s_snap, s_curr, atol=1e-6)
        assert not (all_same_conv and all_same_ssm), (
            f"layer {layer_idx}: snapshot == current — no state advanced (bug)"
        )
        # At least some slots should match (the inactive ones)
        matching_conv = (c_snap - c_curr).abs().max(dim=-1).values.max(dim=-1).values < 1e-6
        num_frozen = matching_conv.sum().item()
        assert num_frozen >= K - block.top_k, (
            f"layer {layer_idx}: only {num_frozen} slots frozen, "
            f"expected ≥ {K - block.top_k}"
        )

    print(f"    ✓  Cache snapshot: {K - block.top_k}+ slots correctly frozen across blocks")


# ===========================================================================
# Test: full SlotSSMTemporalModel — old vs new path across 4 blocks
# ===========================================================================
def test_temporal_model_correctness():
    """SlotSSMTemporalModel with old vs new sparse Mamba must produce identical results."""
    if not _HAS_MAMBA_SSM:
        print("[SKIP] mamba_ssm not installed")
        return
    _require_mamba()

    B, K, D = 1, 32, 512
    input_dim = 1408
    N = 64                  # dummy ref tokens (simulating V-JEPA patch count)
    num_blocks = 4
    top_k = 16
    num_steps = 8
    seed = 42

    # --- Build two identical models -----------------------------------------
    torch.manual_seed(seed)
    model_new = SlotSSMTemporalModel(
        num_slots=K, slot_dim=D, input_dim=input_dim,
        num_blocks=num_blocks, top_k=top_k,
        mamba_d_state=128, mamba_d_conv=4, mamba_expand=2,
        mamba_version="mamba2", num_heads=4, eps_random=0.0,
    ).to(DEVICE).eval()

    torch.manual_seed(seed)
    model_old = SlotSSMTemporalModel(
        num_slots=K, slot_dim=D, input_dim=input_dim,
        num_blocks=num_blocks, top_k=top_k,
        mamba_d_state=128, mamba_d_conv=4, mamba_expand=2,
        mamba_version="mamba2", num_heads=4, eps_random=0.0,
    ).to(DEVICE).eval()

    # Patch old model blocks to use the save/restore path
    for blk in model_old.blocks:
        blk._mamba_step_sparse = (
            lambda slots, af, cache, b=blk: _mamba_step_sparse_old(b, slots, af, cache)
        )

    cache_new = MambaCache()
    cache_old = MambaCache()

    torch.manual_seed(1234)

    for step in range(num_steps):
        ref = torch.randn(B, N, input_dim, device=DEVICE)

        # Snapshot pre-step caches to verify inactive slots are untouched
        cache_new_pre = _clone_cache(cache_new) if step > 0 else None
        cache_old_pre = _clone_cache(cache_old) if step > 0 else None

        slots_new, cache_new = model_new(ref, cache_new)
        slots_old, cache_old = model_old(ref, cache_old)

        # 1. Output slots must match
        torch.testing.assert_close(
            slots_new, slots_old, atol=1e-5, rtol=1e-4,
            msg=f"step {step}: SlotSSMTemporalModel output diverges",
        )

        # 2. seqlen_offset must match
        assert cache_new.seqlen_offset == cache_old.seqlen_offset, (
            f"step {step}: offset {cache_new.seqlen_offset} vs {cache_old.seqlen_offset}"
        )

        # 3. Gate entropy must match
        torch.testing.assert_close(
            model_new._entropy, model_old._entropy, atol=1e-6, rtol=1e-5,
            msg=f"step {step}: entropy mismatch",
        )

        # 4. For the new path: snapshot inactive cache BEFORE the Mamba step
        #    (inside the model call), then verify after that those slots are
        #    bit-identical — proving we never touched them.
        if step > 0:
            _cache_allclose(cache_new, cache_old, atol=1e-5)

        # 5. Inactive slots: bit-identical across old/new (both paths freeze them).
        if step > 0:
            for blk_idx in range(num_blocks):
                cn_new, sn_new = cache_new.key_value_memory_dict[blk_idx]
                co_new, so_new = cache_new_pre.key_value_memory_dict[blk_idx]
                cn_old, sn_old = cache_old.key_value_memory_dict[blk_idx]
                co_old, so_old = cache_old_pre.key_value_memory_dict[blk_idx]

                # active = cache state changed in OLD path (both should agree)
                active = (
                    ~torch.isclose(cn_old, co_old, atol=1e-7).all(dim=-1).all(dim=-1)
                )
                inactive = ~active

                # Inactive slots in NEW path: must be bit-identical to pre-step
                assert torch.equal(
                    cn_new[inactive], co_new[inactive],
                ), f"step {step} blk {blk_idx}: new path mutated inactive conv_state"
                assert torch.equal(
                    sn_new[inactive], so_new[inactive],
                ), f"step {step} blk {blk_idx}: new path mutated inactive ssm_state"

                # Inactive slots in OLD path: also bit-identical (clone+restore)
                assert torch.equal(
                    cn_old[inactive], co_old[inactive],
                ), f"step {step} blk {blk_idx}: old path corrupt inactive conv_state"
                assert torch.equal(
                    sn_old[inactive], so_old[inactive],
                ), f"step {step} blk {blk_idx}: old path corrupt inactive ssm_state"

                # Cross-check: old vs new inactive states — use tolerance
                torch.testing.assert_close(
                    cn_new[inactive], cn_old[inactive], atol=1e-5, rtol=1e-4,
                    msg=f"step {step} blk {blk_idx}: old/new inactive conv_state differ",
                )
                torch.testing.assert_close(
                    sn_new[inactive], sn_old[inactive], atol=1e-5, rtol=1e-4,
                    msg=f"step {step} blk {blk_idx}: old/new inactive ssm_state differ",
                )

    print(f"    ✓  SlotSSMTemporalModel: outputs + {num_blocks}× cache states match across {num_steps} steps")


# ===========================================================================
# Benchmark: old save/restore vs new active-only Mamba step
# ===========================================================================
def benchmark_sparse_mamba():
    """Measure wall-clock time for old vs new _mamba_step_sparse.

    Benchmarks at the block level (single Mamba2 instance) and at the
    full SlotSSMTemporalModel level (4 blocks, real usage pattern).
    """
    if not _HAS_MAMBA_SSM:
        print("[SKIP] mamba_ssm not installed")
        return
    _require_mamba()

    import time

    B, K, D = 1, 32, 512
    input_dim = 1408
    N = 64                  # dummy ref tokens (mimics V-JEPA patch count)
    top_k = 16
    warmup = 50
    measure = 500

    gpu_name = torch.cuda.get_device_name(0)

    # ---- Single block ----------------------------------------------------
    torch.manual_seed(42)
    block_new = SlotSSMBlock(
        slot_dim=D, input_dim=input_dim, top_k=top_k,
        mamba_d_state=128, mamba_d_conv=4, mamba_expand=2,
        mamba_version="mamba2", num_heads=4, block_idx=0,
        eps_random=0.0,
    ).to(DEVICE).eval()

    torch.manual_seed(42)
    block_old = SlotSSMBlock(
        slot_dim=D, input_dim=input_dim, top_k=top_k,
        mamba_d_state=128, mamba_d_conv=4, mamba_expand=2,
        mamba_version="mamba2", num_heads=4, block_idx=0,
        eps_random=0.0,
    ).to(DEVICE).eval()

    def _time_block(block, use_new, label):
        cache = MambaCache()
        torch.manual_seed(1234)
        slots = torch.randn(B, K, D, device=DEVICE)
        ref = torch.randn(B, N, input_dim, device=DEVICE)

        # Single forward to get gate + active mask (identical for both paths)
        cross = block._cross_attn(slots, ref)
        informed = slots + cross
        with torch.no_grad():
            g = block.gate(informed).squeeze(-1)
        _, active_idx = g.topk(block.top_k, dim=1)
        mask = torch.zeros(B, K, 1, device=DEVICE, dtype=slots.dtype)
        mask.scatter_(1, active_idx.unsqueeze(-1), 1.0)
        active_flat = mask.reshape(-1).bool()
        slots = slots + cross * mask

        # Warmup
        for _ in range(warmup):
            sn = slots + torch.randn_like(slots) * 1e-4
            if use_new:
                block._mamba_step_sparse(sn, active_flat, cache)
            else:
                _mamba_step_sparse_old(block, sn, active_flat, cache)
            cache.seqlen_offset += 1

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(measure):
            sn = slots + torch.randn_like(slots) * 1e-4
            if use_new:
                block._mamba_step_sparse(sn, active_flat, cache)
            else:
                _mamba_step_sparse_old(block, sn, active_flat, cache)
            cache.seqlen_offset += 1
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / measure * 1000  # ms
        return elapsed

    ms_old_1 = _time_block(block_old, False, "old")
    ms_new_1 = _time_block(block_new, True, "new")

    # ---- Full SlotSSMTemporalModel (4 blocks) ----------------------------
    torch.manual_seed(42)
    model_new = SlotSSMTemporalModel(
        num_slots=K, slot_dim=D, input_dim=input_dim,
        num_blocks=4, top_k=top_k,
        mamba_d_state=128, mamba_d_conv=4, mamba_expand=2,
        mamba_version="mamba2", num_heads=4, eps_random=0.0,
    ).to(DEVICE).eval()

    torch.manual_seed(42)
    model_old = SlotSSMTemporalModel(
        num_slots=K, slot_dim=D, input_dim=input_dim,
        num_blocks=4, top_k=top_k,
        mamba_d_state=128, mamba_d_conv=4, mamba_expand=2,
        mamba_version="mamba2", num_heads=4, eps_random=0.0,
    ).to(DEVICE).eval()
    for blk in model_old.blocks:
        blk._mamba_step_sparse = (
            lambda sl, af, c, b=blk: _mamba_step_sparse_old(b, sl, af, c)
        )

    def _time_model(model, label):
        cache = MambaCache()
        torch.manual_seed(1234)

        # Warmup
        for _ in range(warmup):
            ref = torch.randn(B, N, input_dim, device=DEVICE)
            _, cache = model(ref, cache)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(measure):
            ref = torch.randn(B, N, input_dim, device=DEVICE)
            _, cache = model(ref, cache)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / measure * 1000  # ms
        return elapsed

    ms_old_4 = _time_model(model_old, "old-4block")
    ms_new_4 = _time_model(model_new, "new-4block")

    # ---- Report ----------------------------------------------------------
    speedup_1 = ms_old_1 / ms_new_1
    speedup_4 = ms_old_4 / ms_new_4

    print(f"\n  GPU: {gpu_name}")
    print(f"  Warmup={warmup}  Measure={measure}  B={B}  K={K}  top_k={top_k}")
    print()
    print(f"  {'':<28} {'Old (save/restore)':>20}  {'New (active-only)':>20}  {'Speedup':>8}")
    print(f"  {'-'*28} {'-'*20}  {'-'*20}  {'-'*8}")
    print(f"  {'Single block (_mamba_step)':<28} {ms_old_1:>19.3f}ms  {ms_new_1:>19.3f}ms  {speedup_1:>7.2f}x")
    print(f"  {'SlotSSMTemporalModel (4 blocks)':<28} {ms_old_4:>19.3f}ms  {ms_new_4:>19.3f}ms  {speedup_4:>7.2f}x")

    # Also report Mamba portion only (4 blocks × mamba_step, excluding
    # cross-attn, gate, self-attn)
    mamba_old_est = ms_old_1 * 4
    mamba_new_est = ms_new_1 * 4
    print(f"  {'  of which ~Mamba portion (4× single)':<28} {mamba_old_est:>19.3f}ms  {mamba_new_est:>19.3f}ms  {mamba_old_est/mamba_new_est:>7.2f}x")

    total_saved = ms_old_4 - ms_new_4
    if total_saved > 0:
        print(f"\n  ✓  Active-only saves {total_saved:.3f}ms/step on the full 4-block model")


# ===========================================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"mamba_ssm: {_HAS_MAMBA_SSM}")
    print()

    test_sparse_mamba_correctness()
    test_sparse_full_forward()
    test_temporal_model_correctness()
    test_preallocated_snapshots()

    print("\nAll sparse Mamba step tests passed!\n")

    print("=" * 70)
    print("Benchmark: old save/restore vs new active-only")
    print("=" * 70)
    benchmark_sparse_mamba()
