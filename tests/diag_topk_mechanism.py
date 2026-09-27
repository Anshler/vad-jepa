"""
Mechanism diagnostics for Sparse SlotSSM checkpoints that differ only in ``top_k``.

``diag_sparse_gate.py`` answers "is the gate alive?" with fixed thresholds that
are implicitly calibrated for top_k=16 — its Jaccard bands, usage buckets and
"credible slot" filter all assume top_k/K = 0.5, so at top_k=4 or 8 it reports
thrashing and routing collapse for healthy models.  This script answers the
comparison question instead: *what does changing top_k actually change?*

All routing statistics are either chance-corrected or scale-free, so they are
comparable across top_k:

  * turnover      — Jaccard excess over the chance value top_k/K
  * gate-input    — corr(1 - Jaccard, ||d patch||): is routing input-driven?
  * effective K   — exp(H(slot usage)); top_k-invariant
  * gate margin   — (s[top_k] - s[top_k+1]) / sd(s): how arbitrary the cutoff is
  * MI(active;label) against a shuffled-label null (per-slot counts are tiny at
    low top_k, so an un-nulled MI is mostly finite-sample bias)
  * spatial       — per-slot cross-attention entropy + centroid spread
  * read mass     — self-attention mass the active set routes to frozen slots
  * frozen audit  — fraction of inactive-slot updates that are exactly zero

Routing interventions (``--mode``) isolate *whether selection matters* from
*how much sparsity*: learned / random / roundrobin / frozen-first-frame, each
scored with the same AUC and FPR-matched event recall.

Usage (WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad

    python tests/diag_topk_mechanism.py \
        --config cfgs/vjepa_sparse_slotssm_topk_4.yaml \
        --checkpoint output/_supersede/vjepa_sparse_slotssm_topk_4_VCL_64_NF_4_finetuned/checkpoints/model-30.pt \
        --max_videos 60 --mode learned

    # then repeat with --mode random / roundrobin / frozen
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import torchvision.transforms._functional_tensor as _ft
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
except ImportError:
    pass

import yaml  # noqa: E402
from easydict import EasyDict  # noqa: E402

from model import build_multi_head_vjepa  # noqa: E402
from movad_core.dota import Dota, gt_cls_target, setup_dota  # noqa: E402
from analyze_topk_sweep import event_metrics, pooled_auc, per_video_auc  # noqa: E402

# ---------------------------------------------------------------------------
# Routing interventions.  Each returns an active_idx replacement; the block is
# passed so an override can keep per-block state.
# ---------------------------------------------------------------------------
def make_override(mode, top_k, num_slots, seed=0):
    if mode == 'learned':
        return None

    if mode == 'random':
        gen = torch.Generator().manual_seed(seed)

        def _random(gate_scores, active_idx, blk):
            B, K = gate_scores.shape
            return torch.stack([
                torch.randperm(K, generator=gen)[:top_k] for _ in range(B)
            ]).to(active_idx.device)
        return _random

    if mode == 'roundrobin':
        def _roundrobin(gate_scores, active_idx, blk):
            off = getattr(blk, '_rr_off', 0)
            blk._rr_off = off + top_k
            idx = [(off + i) % num_slots for i in range(top_k)]
            B = gate_scores.shape[0]
            return torch.tensor(idx, device=active_idx.device).unsqueeze(0).expand(B, -1)
        return _roundrobin

    if mode == 'frozen':
        def _frozen(gate_scores, active_idx, blk):
            first = getattr(blk, '_frozen_idx', None)
            if first is None:
                blk._frozen_idx = active_idx.clone()
                return active_idx
            return blk._frozen_idx
        return _frozen

    raise ValueError(f'unknown mode {mode!r}')


def make_knockout(top_k, num_slots, drop_slots):
    """Zero out the given slots from every selection (slot-knockout ablation)."""
    drop = set(int(s) for s in drop_slots)

    def _knockout(gate_scores, active_idx, blk):
        g = gate_scores.clone()
        for s in drop:
            g[:, s] = -1e9
        _, idx = g.topk(top_k, dim=1)
        return idx
    return _knockout


# ---------------------------------------------------------------------------
# Derived statistics
# ---------------------------------------------------------------------------
def _entropy(p, axis=-1):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    return -(p * np.log(p)).sum(axis=axis)


def _gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return float('nan')
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def _shift_labels(lab, seg_len, rng):
    """Null labels with temporal structure preserved.

    A global ``rng.permutation(lab)`` whitens the label series, but both the
    labels (accident windows) and the slot activations (Jaccard 0.74-0.84) are
    strongly autocorrelated.  Two independent autocorrelated series have a
    LARGER correlation variance than one autocorrelated and one white series,
    so permuting shrinks the null variance and inflates every z-score.  Rolling
    each video's label segment by a random offset keeps the class balance and
    the run-length structure, giving a null with the right variance.
    """
    y = np.array(lab, dtype=np.float64)
    if not seg_len:
        return rng.permutation(lab)
    pos = 0
    for L in seg_len:
        L = int(L)
        if L > 1 and pos + L <= y.size:
            sh = int(rng.randint(1, L))
            y[pos:pos + L] = np.roll(lab[pos:pos + L], sh)
        pos += L
    return y


def _mi_bernoulli(a, y):
    """MI between boolean mask ``a`` and label vector ``y``, in nats."""
    tot = 0.0
    for av, yv in ((True, 1.0), (True, 0.0), (False, 1.0), (False, 0.0)):
        pxy = float(((a == av) & (y == yv)).mean())
        px = float((a == av).mean())
        py = float((y == yv).mean())
        if pxy > 0 and px > 0 and py > 0:
            tot += pxy * np.log(pxy / (px * py))
    return tot


def routing_stats(block_diags, top_k, num_slots, labels, toas, patch_delta,
                  seg_len=None, nf=4, smooth=5):
    """Every chance-corrected routing statistic for one block."""
    gate = np.asarray(block_diags['gate_scores'], dtype=np.float64)   # [T, K]
    active = np.asarray(block_diags['active_idx'], dtype=np.int64)    # [T, top_k]
    if gate.size == 0 or active.size == 0:
        return {}

    T, K = gate.shape
    chance_jac = top_k / num_slots

    # --- seam mask: pairs (t, t+1) that belong to the same video -----------
    # Only the boundaries *between* videos matter; the last segment has no
    # following pair, so its end index would fall outside the (T-1) array.
    same = np.ones(max(T - 1, 0), dtype=bool)
    if seg_len:
        pos = 0
        for L in seg_len[:-1]:
            if 0 < L < T and pos + L - 1 < same.size:
                same[pos + L - 1] = False      # pair straddling this seam
            pos += L

    # --- 1. turnover, chance-corrected ------------------------------------
    jac_all = np.array([
        len(set(active[t].tolist()) & set(active[t + 1].tolist())) / top_k
        for t in range(T - 1)
    ])
    jac = jac_all[same] if same.size == jac_all.size else jac_all
    static_frac = float((jac > 0.99).mean()) if jac.size else float('nan')

    # --- 2. gate-input correlation (scale-free) ---------------------------
    gate_in_corr = float('nan')
    if jac.size and patch_delta is not None and len(patch_delta) >= T:
        dd = np.abs(np.asarray(patch_delta[:T], dtype=np.float64)[1:])
        dd = dd[same] if same.size == dd.size else dd
        if dd.size == jac.size and dd.std() > 0 and jac.std() > 0:
            gate_in_corr = float(np.corrcoef(1.0 - jac, dd)[0, 1])

    # --- 3. slot usage, effective count, chance-corrected CV --------------
    counts = np.zeros(num_slots)
    for t in range(T):
        for s in active[t].tolist():
            counts[s] += 1
    usage = counts / max(T, 1)    # usage sums to top_k (each step activates top_k slots), so normalize before
    # taking an entropy — exp(H) is the *effective* slot count in [1, K].
    usage_p = usage / max(usage.sum(), 1e-12)
    eff_k = float(np.exp(_entropy(usage_p + 1e-12)))
    expect = top_k / num_slots
    chance_cv = np.sqrt((1 - expect) / max(expect * T, 1e-9))
    usage_cv = float(usage.std() / usage.mean()) if usage.mean() > 0 else float('nan')

    # --- 4. gate entropy (softmax over all K -> comparable across top_k) --
    p = np.exp(gate - gate.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    gate_ent = _entropy(p, axis=1)

    # --- 5. margin at the cutoff ------------------------------------------
    srt = np.sort(gate, axis=1)[:, ::-1]
    # top_k == K (dense) has no cutoff to measure: there is no (k+1)-th slot.
    margin = ((srt[:, top_k - 1] - srt[:, top_k]) / (gate.std(axis=1) + 1e-9)
              if top_k < K else np.full(T, np.nan))

    # --- 6. label association, per slot -----------------------------------
    # Reported as max AND mean over slots: the label information is
    # concentrated in a few slots, so a mean over 32 dilutes it ~32x and makes
    # a real effect look like nothing.  The max is the statistic
    # diag_sparse_gate.py implicitly uses (it prints the top-5 ratios) — but it
    # needs a null, because a max over 32 slots on ~40 active frames is noise.
    lab = np.asarray(labels[:T], dtype=np.float64)
    mi_mean_raw = mi_mean_excess = mi_max_raw = mi_max_excess = float('nan')
    n_slots_sig = 0
    ratio_max = ratio_max_all = float('nan')
    ratio_argmax_z = float('nan')
    top_slots = []
    keep: list = []
    if lab.size == T and 0 < lab.sum() < T:
        A = np.zeros((T, num_slots), dtype=bool)
        for t in range(T):
            A[t, active[t]] = True
        # Slots with too few active frames give a degenerate plug-in MI (cells
        # near zero can push it above H(y), which is impossible).
        keep = [s for s in range(num_slots)
                if A[:, s].sum() >= 30 and (~A[:, s]).sum() >= 30]
        if keep:
            def mi_vec(y):
                return np.array([_mi_bernoulli(A[:, s], y) for s in keep])

            obs = mi_vec(lab)
            rng = np.random.RandomState(0)
            n_perm = 200
            null = np.stack([mi_vec(_shift_labels(lab, seg_len, rng))
                             for _ in range(n_perm)])
            excess = obs - null.mean(axis=0)
            zs = excess / (null.std(axis=0) + 1e-12)
            mi_mean_raw = float(obs.mean())
            mi_mean_excess = float(excess.mean())
            mi_max_raw = float(obs.max())
            mi_max_excess = float(excess.max())
            n_slots_sig = int((zs > 2).sum())
            # The question "is the MOST label-sensitive slot real?" is a
            # max-over-slots question, so the null must be the max over slots
            # for each permutation — not one slot's own null mean.
            null_max = null.max(axis=1)
            mi_max_pvalue = float((null_max >= obs.max()).mean())
            mi_max_null_q95 = float(np.percentile(null_max, 95))
        else:
            mi_max_pvalue = float('nan')
            mi_max_null_q95 = float('nan')

        # the old diagnostic's statistic, on identical data, with significance
        na = float((lab == 1).sum())
        nn_ = float((lab == 0).sum())
        rows = []
        for s in range(num_slots):
            if A[:, s].sum() == 0:
                continue
            pa = float(A[lab == 1, s].mean())
            pn = float(A[lab == 0, s].mean())
            pbar = float(A[:, s].mean())
            se = np.sqrt(pbar * (1 - pbar) * (1 / na + 1 / nn_)) + 1e-12
            rows.append((s, pa, pn, pa / max(pn, 1e-9), (pa - pn) / se,
                         int(A[:, s].sum())))
        if rows:
            ratio_max_all = float(max(r[3] for r in rows))
            ratio_argmax_z = float(max(rows, key=lambda r: r[3])[4])
            if keep:
                kset = set(keep)
                ratio_max = float(max(r[3] for r in rows if r[0] in kset))
            top_slots = sorted(rows, key=lambda r: -r[3])[:5]

    # --- 7. spatial specialisation from cross-attention -------------------
    cross = block_diags.get('cross') or []
    xent = cent_spread = cent_pairdist = float('nan')
    if cross:
        ents = [np.asarray(e).reshape(-1) for (e, _mx, _c) in cross[:T]]
        cents = [np.asarray(c).reshape(-1, 2) for (_e, _mx, c) in cross[:T]
                 if c is not None]
        if ents:
            xent = float(np.mean([e.mean() for e in ents]))
        if cents:
            C = np.stack(cents, axis=0)                       # [T', K, 2]
            cent_spread = float(np.mean(C.std(axis=1)))
            if C.shape[1] > 1:
                d = np.linalg.norm(C[:, :, None, :] - C[:, None, :, :], axis=-1)
                iu = np.triu_indices(C.shape[1], k=1)
                cent_pairdist = float(d[:, iu[0], iu[1]].mean())

    # --- 8. frozen audit + read mass --------------------------------------
    sd = block_diags.get('slot_delta') or []
    self_d = block_diags.get('self_attn') or []
    frozen_frac = act_delta = inact_delta = float('nan')
    read_inactive_frac = read_gini = float('nan')
    act_mask = np.zeros((T, num_slots), dtype=bool)
    for t in range(T):
        act_mask[t, active[t]] = True
    # These lists must be step-aligned with gate/active; a shorter list means an
    # instrumentation gap and would silently compare the wrong steps.
    if sd and len(sd) != T:
        print(f"    [warn] slot_delta len {len(sd)} != steps {T}; skipping frozen audit")
        sd = []
    if self_d and len(self_d) != T:
        print(f"    [warn] self_attn len {len(self_d)} != steps {T}; skipping read mass")
        self_d = []
    if sd:
        S = np.asarray(sd[:T], dtype=np.float64)
        m = act_mask[:S.shape[0]]
        if m.any():
            frozen_frac = (float((S[~m] == 0).mean()) if (~m).any()
                           else float('nan'))       # dense: no inactive set
            act_delta = float(S[m].mean())
            if (~m).any():
                inact_delta = float(S[~m].mean())
    if self_d:
        rm = np.asarray([np.asarray(r).reshape(-1) for (_e, r) in self_d[:T]],
                        dtype=np.float64)
        if rm.size:
            m = act_mask[:rm.shape[0]]
            tot = rm.sum(axis=1, keepdims=True) + 1e-12
            read_inactive_frac = float((rm * (~m)).sum() / tot.sum())
            read_gini = float(np.mean([_gini(rm[t]) for t in range(rm.shape[0])]))

    # --- 8b. cross-slot state similarity (redundancy vs specialisation) ----
    # High = slots are interchangeable; low = they have specialised.  The
    # active/inactive split matters: inactive slots are frozen at their init,
    # active ones are all pulled by the same cross-attn/Mamba input.
    ss = block_diags.get('slot_sim') or []
    sim_all = sim_act = sim_inact = float('nan')
    sim_all_last = sim_act_last = sim_inact_last = float('nan')
    sim_act_std = sim_all_std = float('nan')
    if ss:
        arr = np.asarray(ss[:T], dtype=np.float64)
        sim_all = float(np.nanmean(arr[:, 0]))
        sim_act = float(np.nanmean(arr[:, 1]))
        sim_inact = float(np.nanmean(arr[:, 2]))
        # Diag 3 in diag_sparse_gate.py reports only the LAST frame's active/
        # inactive split, which is a single sample AND biased: at the end of a
        # video most slots have been touched recently, so the "inactive" pool is
        # dominated by recently-active slots.  Report both so the gap is visible.
        sim_all_last = float(arr[-1, 0])
        sim_act_last = float(arr[-1, 1])
        sim_inact_last = float(arr[-1, 2])
        sim_act_std = float(np.nanstd(arr[:, 1]))
        sim_all_std = float(np.nanstd(arr[:, 0]))

    # --- 8c. is long-dormant memory READ?  (tests persistent memory) ------
    # age_k(t) = steps since slot k was last active.  If the frozen slots act as
    # long-term memory, self-attn read mass should favour HIGH-age (dormant)
    # slots.  corr ~ 0 or negative means the active set only reads what it (or
    # its neighbours) just wrote, i.e. no long-range memory use.
    read_age_corr = float('nan')
    if self_d and len(self_d) == T:
        rm = np.asarray([np.asarray(r).reshape(-1) for (_e, r) in self_d[:T]],
                        dtype=np.float64)                       # [T, K]
        age = np.full((T, K), np.nan)
        last = np.full(K, -1)
        for t in range(T):
            seen = last >= 0
            age[t, seen] = t - last[seen]
            for sl in active[t].tolist():
                last[sl] = t
        m = np.isfinite(age) & np.isfinite(rm)
        if m.sum() > 10 and rm[m].std() > 0 and age[m].std() > 0:
            read_age_corr = float(np.corrcoef(rm[m], age[m])[0, 1])

    # --- 8d. read mass BY AGE BUCKET (does DORMANT memory get read?) ------
    # A global corr(read_mass, age) can read ~0 even if a small dormant
    # population IS read, because low-age pairs dominate the mass.  Bucketing
    # shows whether age>10 slots receive any read mass at all.
    read_by_age = {}
    read_share_gt10 = float('nan')
    if self_d and len(self_d) == T:
        rm2 = np.asarray([np.asarray(r).reshape(-1) for (_e, r) in self_d[:T]],
                         dtype=np.float64)
        age2 = np.full((T, K), np.nan)
        last2 = np.full(K, -1)
        for t in range(T):
            seen = last2 >= 0
            age2[t, seen] = t - last2[seen]
            for sl in active[t].tolist():
                last2[sl] = t
        m2 = np.isfinite(age2) & np.isfinite(rm2)
        tot = rm2[m2].sum()
        for lo, hi in ((0, 0), (1, 1), (2, 5), (6, 10), (11, 30), (31, 10 ** 9)):
            sel = m2 & (age2 >= lo) & (age2 <= hi)
            if sel.sum() == 0:
                continue
            read_by_age[f"{lo}-{'inf' if hi > 10 ** 8 else hi}"] = dict(
                n=int(sel.sum()),
                share=float(rm2[sel].sum() / tot) if tot > 0 else float('nan'),
                mean=float(rm2[sel].mean()))
        sel10 = m2 & (age2 >= 11)
        read_share_gt10 = float(rm2[sel10].sum() / tot) if tot > 0 else float('nan')

    # --- 8e. are DORMANT slots TRIGGERED by relevant input? --------------
    # Hypothesis: a dormant slot is a "backup" that stays quiet until the input
    # contains something relevant.  That is a claim about ACTIVATION, not
    # reading.  Prediction: activations of long-dormant slots are enriched on
    # anomaly frames and coincide with larger input change (||d patch||).
    trig = {}
    if patch_delta is not None and len(patch_delta) >= T:
        pdd = np.asarray(patch_delta[:T], dtype=np.float64)
        lab = np.asarray(labels[:T], dtype=np.float64)
        last = np.full(K, -1)
        ages, anoms, pds = [], [], []
        for t in range(T):
            for sl in active[t].tolist():
                if last[sl] >= 0:
                    ages.append(t - last[sl]); anoms.append(lab[t]); pds.append(pdd[t])
            for sl in active[t].tolist():
                last[sl] = t
        if len(ages) > 30:
            ages = np.array(ages); anoms = np.array(anoms); pds = np.array(pds)
            trig['_base'] = dict(n=int(ages.size), anom_rate=float(lab.mean()),
                                 patch_delta=float(pdd.mean()))
            for lo, hi, name in ((1, 2, 'recent_1_2'), (3, 10, 'mid_3_10'),
                                 (11, 10 ** 9, 'dormant_gt10')):
                m = (ages >= lo) & (ages <= hi)
                if m.sum() < 10:
                    continue
                trig[name] = dict(n=int(m.sum()), anom_rate=float(anoms[m].mean()),
                                  patch_delta=float(pds[m].mean()))

    # per-slot update interval (the only source of a timescale hierarchy, since
    # one Mamba module is shared by all slots -> identical intrinsic dynamics)
    _u = usage[usage > 0]
    slow_iv = float(1.0 / _u.min()) if _u.size else float('nan')
    fast_iv = float(1.0 / _u.max()) if _u.size else float('nan')

    # --- 9. onset response (offset by NF: step t scores frame t+NF) --------
    onset_jac = base_jac = float('nan')
    if jac_all.size:
        idx = set()
        for k, toa in enumerate(toas):
            base_frame = int(sum(seg_len[:k])) if seg_len else 0
            for d in range(0, smooth + 1):
                g = base_frame + int(toa) - nf + d
                if 0 <= g < jac_all.size and same[g]:
                    idx.add(g)
        if idx:
            onset_jac = float(np.mean([jac_all[t] for t in sorted(idx)]))
            base_jac = float(jac.mean())

    return dict(
        n_steps=int(T), chance_jaccard=float(chance_jac),
        jaccard_mean=float(jac.mean()) if jac.size else float('nan'),
        jaccard_excess=float((jac.mean() - chance_jac) / (1 - chance_jac))
        if jac.size else float('nan'),
        static_frac=static_frac, gate_input_corr=gate_in_corr,
        usage_mean=float(usage.mean()), usage_cv=usage_cv,
        chance_cv=float(chance_cv),
        usage_cv_ratio=float(usage_cv / chance_cv) if chance_cv > 0 else float('nan'),
        effective_k=eff_k, dead_slots=int((usage < 0.02).sum()),
        # Per-slot usage and the dead-slot IDS, not just the count.  `dead_slots`
        # is PER BLOCK -- each SlotSSMBlock owns its own gate and selects
        # independently, so a slot dead in one block is routinely live in another.
        # Saving the ids is what makes the cross-block question (same slot? ever
        # written by ANY block?) answerable downstream; the count alone cannot
        # distinguish "never selected" from "selected 50 times, scattered", since
        # usage is aggregated over all T steps and has no temporal order.
        usage_per_slot=[float(u) for u in usage],
        dead_slot_ids=sorted(int(s) for s in np.nonzero(usage < 0.02)[0]),
        gate_entropy_mean=float(gate_ent.mean()),
        gate_entropy_uniform=float(np.log(num_slots)),
        gate_entropy_norm=float(gate_ent.mean() / np.log(num_slots)),
        margin_mean=float(np.mean(margin)),
        margin_frac_below_0p1=float((margin < 0.1).mean()),
        mi_mean_raw=mi_mean_raw, mi_mean_excess=mi_mean_excess,
        mi_max_raw=mi_max_raw, mi_max_excess=mi_max_excess,
        mi_max_pvalue=mi_max_pvalue, mi_max_null_q95=mi_max_null_q95,
        n_slots_sig=n_slots_sig, n_slots_kept=len(keep) if lab.size == T else 0,
        ratio_max=ratio_max, ratio_max_all=ratio_max_all,
        ratio_argmax_z=ratio_argmax_z,
        ratio_top5=[[int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                     float(r[4]), int(r[5])] for r in top_slots],
        cross_attn_entropy=xent, centroid_spread=cent_spread,
        centroid_pairdist=cent_pairdist,
        frozen_update_frac=frozen_frac, active_delta=act_delta,
        inactive_delta=inact_delta,
        read_inactive_frac=read_inactive_frac, read_gini=read_gini,
        slot_sim_all=sim_all, slot_sim_active=sim_act, slot_sim_inactive=sim_inact,
        slot_sim_all_last=sim_all_last, slot_sim_active_last=sim_act_last,
        slot_sim_inactive_last=sim_inact_last,
        slot_sim_active_std=sim_act_std, slot_sim_all_std=sim_all_std,
        read_age_corr=read_age_corr, slow_interval=slow_iv, fast_interval=fast_iv,
        read_by_age=read_by_age, read_share_age_gt10=read_share_gt10,
        trigger_by_age=trig,
        jaccard_at_onset=onset_jac, jaccard_baseline=base_jac,
    )


# ---------------------------------------------------------------------------
# Model / data
# ---------------------------------------------------------------------------
def build_and_load(cfg, ckpt_path, device):
    head_name = os.path.splitext(os.path.basename(cfg._config_name))[0]
    cfg._head_cfgs_flat = [dict(cfg)]
    cfg._head_cfgs_flat[0]["name"] = head_name

    model = build_multi_head_vjepa(cfg)
    model.to(device)
    head = model.heads[head_name]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    missing, unexpected = head.load_state_dict(sd, strict=False)
    # A silently-missing gate or encoder would invalidate every number below.
    if missing or unexpected:
        print(f"  !! CHECKPOINT MISMATCH  missing={len(missing)} unexpected={len(unexpected)}")
        for k in list(missing)[:8]:
            print(f"     missing: {k}")
        for k in list(unexpected)[:8]:
            print(f"     unexpected: {k}")
    else:
        print(f"  checkpoint loaded cleanly ({len(sd)} tensors, "
              f"epoch {ckpt.get('epoch', '?')})")
    return model, head, head_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max_videos", type=int, default=60)
    ap.add_argument("--mode", default="learned",
                    choices=["learned", "random", "roundrobin", "frozen", "knockout"])
    ap.add_argument("--knockout_slots", type=int, nargs="*", default=None,
                    help="Slot ids to remove from every selection (mode=knockout)")
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--seed", type=int, default=42,
                    help="Selects WHICH videos are scored (keep fixed to stay paired)")
    ap.add_argument("--route_seed", type=int, default=0,
                    help="Separate seed for the random-routing intervention, so "
                         "routing randomness can be varied without changing the "
                         "video subset. Use it to measure the spread of "
                         "'random routing' AUC across draws.")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--allow_dense", action="store_true",
                    help="Also accept a DENSE SlotSSM checkpoint (temporal_model "
                         "'slotssm'). Dense is the top_k = K degenerate case: no "
                         "gate and no selection, so it is forced to 'learned' mode "
                         "and the gate-based statistics are degenerate by "
                         "construction. The gate-free statistics (slot_sim, read "
                         "mass, read mass by age) are collected identically, which "
                         "is the point -- it puts dense and sparse on one axis.")
    ap.add_argument("--legacy_stale_kv", action="store_true",
                    help="Run the sparse path under the PRE-FIX forward (bfaa73a): "
                         "self-attn KV built from pre-update slot states. The "
                         "frozen VCL=64 sparse checkpoint was trained AND "
                         "evaluated under that forward, so this is the only way "
                         "to score it under the code it was trained with. "
                         "Without it, that checkpoint is a train/test forward "
                         "mismatch and its AUC is comparable to nothing. Affects "
                         "the sparse path only; the dense path is unchanged.")
    ap.add_argument("--content", default="none",
                    choices=["none", "breakpreserve", "scramble", "dense_ssm",
                             "dense_both"],
                    help="CONTENT intervention on the preserved rows, orthogonal to "
                         "--mode (which changes SELECTION). 'breakpreserve' copies "
                         "the freshly written active rows over the preserved ones: "
                         "same sparsity, same selection, nothing preserved -- the "
                         "causal test for 'the sparse advantage is preservation'. "
                         "'scramble' permutes the stale content among the preserved "
                         "rows: staleness and row count kept, slot-specific "
                         "association destroyed, which separates 'preserved set is "
                         "MEMORY' from 'preserved set is an un-overwritten BUFFER'. "
                         "'dense_ssm' advances EVERY slot's per-slot Mamba state "
                         "while the slot tensor still gets only top_k rows: the "
                         "complement, dense-ifying the RECURRENT preservation. "
                         "'dense_both' = dense_ssm + breakpreserve. "
                         "Sparse only. Under the row-level modes the frozen audit "
                         "(inactive_delta, frozen_update_frac) is expected to break "
                         "BY CONSTRUCTION -- ignore it, it is the intervention.")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = os.path.join(_REPO_ROOT, args.config)
    with open(cfg_path) as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.device = DEVICE
    cfg.compile = False
    cfg._config_name = args.config
    if args.data_path:
        cfg.data_path = args.data_path

    ckpt_path = (args.checkpoint if os.path.isabs(args.checkpoint)
                 else os.path.join(_REPO_ROOT, args.checkpoint))
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Mode:       {args.mode}")
    print(f"Device:     {DEVICE}")

    model, head, head_name = build_and_load(cfg, ckpt_path, DEVICE)

    if head.temporal_type != 'sparse_slotssm':
        # Dense SlotSSM is the top_k = K degenerate case.  It has no gate, so it
        # is only ever run in 'learned' mode -- the routing interventions have no
        # selection to override.  Everything that does not depend on the gate is
        # still collected, which is what makes dense comparable to sparse here.
        if head.temporal_type == 'slotssm' and args.allow_dense:
            pass
        else:
            print(f"ERROR: temporal_model is {head.temporal_type!r}, expected "
                  f"'sparse_slotssm' (or 'slotssm' with --allow_dense)")
            sys.exit(1)

    temporal = head.temporal
    is_dense = head.temporal_type == 'slotssm'
    num_slots = temporal.num_slots
    top_k = num_slots if is_dense else temporal.top_k
    if is_dense:
        args.mode = 'learned'
    grid = cfg.get('vjepa_spatial_grid', None)
    print(f"  K={num_slots}  top_k={top_k}  blocks={len(temporal.blocks)}  grid={grid}"
          + ("   [DENSE: no gate, all slots active]" if is_dense else ""))
    if args.legacy_stale_kv:
        if is_dense:
            print("  legacy_stale_kv: no-op for dense (that path has no compact "
                  "branch, so it is identical in both revisions)")
        else:
            for blk in temporal.blocks:
                blk.legacy_stale_kv = True
            print("  legacy_stale_kv: ON -- reproducing the pre-fix forward "
                  "(stale self-attn KV)")

    # --- routing intervention ---
    if args.mode == 'knockout':
        if not args.knockout_slots:
            print("ERROR: --mode knockout needs --knockout_slots")
            sys.exit(1)
        fn = make_knockout(top_k, num_slots, args.knockout_slots)
        print(f"  knockout: dropping slots {sorted(set(args.knockout_slots))}")
    else:
        fn = make_override(args.mode, top_k, num_slots, seed=args.route_seed)
    temporal.set_route_override(fn)

    # --- content intervention (orthogonal to selection) ---
    if args.content != 'none':
        if is_dense:
            print(f"  content={args.content}: NO-OP on dense (no preserved set). "
                  f"Reporting it anyway so the mode is on the record.")
        temporal.set_content_mode(args.content)
        if args.content == 'dense_ssm':
            print(f"  content=dense_ssm: EVERY slot's per-slot Mamba state is "
                  f"advanced each step while the slot tensor still receives only "
                  f"top_k rows. SELECTION untouched, and the ROW-level frozen "
                  f"audit stays valid (rows really are frozen).")
        else:
            print(f"  content={args.content}: the preserved rows' CONTENT is being "
                  f"altered while SELECTION is untouched. The frozen audit "
                  f"(inactive_delta / frozen_update_frac) WILL break -- that is the "
                  f"intervention, not a regression.")

    # --- data ---
    NF = cfg.get('NF', cfg.get('num_frames', 4))
    test_cfg = EasyDict(dict(cfg))
    test_cfg.batch_size = 1
    test_cfg.input_shape = cfg.get('input_shape', [cfg.get('img_size', 384)] * 2)
    _, test_loader = setup_dota(Dota, test_cfg, num_workers=0, VCL=None, phase='test')
    dataset = test_loader.dataset
    total = len(dataset)
    limit = min(args.max_videos if args.max_videos > 0 else total, total)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(total)[:limit].tolist()
    print(f"\nTest videos: {limit}/{total}  (shuffled, seed={args.seed})")

    model.eval()
    head.eval()
    temporal.enable_diagnostics(grid=grid, collect_slots=False)

    # Capture the patch tokens the temporal model sees, so gate changes can be
    # correlated against actual input change (a scale-free "is routing
    # input-driven?" measure, unlike raw Jaccard which depends on top_k).
    _patch_means: list = []

    def _grab_patches(_mod, inputs):
        p = inputs[0]
        _patch_means.append(p.detach().mean(dim=1).reshape(-1).float().cpu().numpy())

    _hook = temporal.register_forward_pre_hook(_grab_patches)

    amp = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(cfg.get('amp_dtype', 'fp32'))
    autocast_ctx = (torch.amp.autocast('cuda', dtype=amp) if amp
                    else __import__('contextlib').nullcontext())

    # Collected per-frame, per-block
    blocks = [dict(gate_scores=[], active_idx=[], cross=[], self_attn=[],
                   slot_delta=[], slot_sim=[])
              for _ in range(len(temporal.blocks))]
    all_targets, all_outputs, all_toas, all_teas = [], [], [], []
    all_labels, all_patch_delta, all_seg_len = [], [], []
    vid_keys: list = []
    frame_times = []
    n_frames = 0

    for vi, idx in enumerate(indices):
        video_data, data_info_raw = dataset[idx]
        frames = (torch.from_numpy(video_data).float()
                  if isinstance(video_data, np.ndarray) else video_data.float())
        if frames.dim() == 4:
            frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
        video = frames.to(DEVICE)
        info = (torch.tensor(data_info_raw).float().unsqueeze(0).to(DEVICE)
                if not isinstance(data_info_raw, torch.Tensor)
                else data_info_raw.float().unsqueeze(0).to(DEVICE))

        v_len = video.shape[2]
        vl = int(info[:, 0].item())
        toa_b = info[:, 2]
        tea_b = info[:, 3]
        valid = vl - NF
        if valid <= 0:
            continue

        temporal.enable_diagnostics(grid=grid, collect_slots=False)
        state = None
        _patch_means.clear()
        vid_targets, vid_outputs = [], []

        # Iterate only real frames (vl), not the padded bucket length, so the
        # number of temporal steps equals the number of scored frames and the
        # gate sequence stays index-aligned with the labels.
        for i in range(NF, vl):
            target = gt_cls_target(i, toa_b, tea_b).long()
            clip = video[:, :, i - NF:i, :, :]
            t0 = time.perf_counter()
            with torch.no_grad(), autocast_ctx:
                output, state = head(clip, state)
            frame_times.append((time.perf_counter() - t0) * 1000.0)

            p1 = float(output.softmax(dim=1)[:, 1].item())
            vid_targets.append(int(target[0].item()))
            vid_outputs.append(p1)
            all_labels.append(int(target[0].item()))
            n_frames += 1

        diag = temporal.get_diagnostics()
        nb = len(diag['gate_scores'])
        # AMP runs in bf16, and bf16 tensors have no numpy() — cast to fp32.
        for b in range(nb):
            if is_dense:
                # No gate, so synthesize the degenerate "every slot active"
                # record, one row per forward step.  routing_stats then reports
                # usage=uniform, eff_K=K, turnover=1, MI=0 -- all literally true
                # for dense -- while the sections we actually care about
                # (slot_sim, read mass, read mass by age) come from the real
                # forward pass and are computed by identical code.
                for _t in range(len(diag['slot_sim'][b])):
                    blocks[b]['gate_scores'].append(np.zeros(num_slots))
                    blocks[b]['active_idx'].append(np.arange(num_slots))
            for t in range(len(diag['gate_scores'][b])):
                blocks[b]['gate_scores'].append(
                    diag['gate_scores'][b][t][0].float().numpy())
                blocks[b]['active_idx'].append(
                    diag['active_idx'][b][t][0].long().numpy())
            for t in range(len(diag['cross'][b])):
                e, mx, c = diag['cross'][b][t]
                blocks[b]['cross'].append(
                    (e[0].float().numpy(), mx[0].float().numpy(),
                     c[0].float().numpy() if c is not None else None))
            for t in range(len(diag['self_attn'][b])):
                e, rm = diag['self_attn'][b][t]
                blocks[b]['self_attn'].append(
                    (e[0].float().numpy(), rm[0].float().numpy()))
            for t in range(len(diag['slot_delta'][b])):
                blocks[b]['slot_delta'].append(
                    diag['slot_delta'][b][t][0].float().numpy())
            for t in range(len(diag.get('slot_sim', [[]])[b])):
                a_, ac_, ia_ = diag['slot_sim'][b][t]
                blocks[b]['slot_sim'].append(
                    (a_[0].float().item(), ac_[0].float().item(), ia_[0].float().item()))

        all_targets.append(np.asarray(vid_targets))
        all_outputs.append(np.asarray(vid_outputs))
        all_toas.append(float(toa_b[0].item()))
        all_teas.append(float(tea_b[0].item()))
        vid_keys.append(int(info[0, 1].item()))     # data_info[1] = key index

        # Video boundaries: turnover between the last frame of one video and the
        # first of the next is meaningless (different scenes), so mark the seam.
        all_seg_len.append(len(vid_targets))
        # Patch-change aligned per step; 0 at each video start (no previous frame)
        pm = np.stack(_patch_means) if len(_patch_means) > 1 else None
        all_patch_delta.append(0.0)
        if pm is not None:
            all_patch_delta.extend(
                np.linalg.norm(np.diff(pm, axis=0), axis=1).tolist())

        if (vi + 1) % 10 == 0:
            print(f"  [{vi + 1}/{limit}] frames={n_frames} "
                  f"({np.mean(frame_times):.0f} ms/frame)")
        torch.cuda.empty_cache()

    temporal.disable_diagnostics()
    temporal.set_route_override(None)

    _hook.remove()
    if n_frames == 0:
        print("\nERROR: no frames collected.")
        sys.exit(1)

    # ---- detection metrics for this mode ----
    tgt = [t for t in all_targets if t.size]
    out = [o for o in all_outputs if o.size]
    auc = pooled_auc(tgt, out)
    pv = per_video_auc(tgt, out)
    em = event_metrics(np.asarray(all_toas), np.asarray(all_teas), tgt, out,
                       cfg.get('FPS', 10), smooth=args.smooth)

    print("\n" + "=" * 84)
    print(f"DETECTION — mode={args.mode}  top_k={top_k}  "
          f"({len(tgt)} videos, {n_frames} frames)")
    print("=" * 84)
    print(f"  pooled AUC      = {auc:.4f}")
    print(f"  mean video AUC  = {np.nanmean(pv):.4f}  (sd {np.nanstd(pv):.4f})")
    print(f"  latency         = {np.mean(frame_times):.1f} ms/frame "
          f"(median {np.median(frame_times):.1f}, p90 {np.percentile(frame_times, 90):.1f})")
    print(f"\n  {'FA/min':>7} {'realised':>9} {'recall':>8} {'med delay':>10} "
          f"{'mean delay':>11}")
    for m in em:
        print(f"  {m['fa_budget']:>7.1f} {m['fa_per_min_realised']:>9.4f} "
              f"{m['event_recall']:>8.4f} {m['median_delay']:>10.1f} "
              f"{m['mean_delay']:>11.1f}")

    # ---- routing diagnostics, per block ----
    stats = {}
    print("\n" + "=" * 84)
    print("ROUTING DIAGNOSTICS (chance-corrected; comparable across top_k)")
    print("=" * 84)
    for b in range(len(blocks)):
        st = routing_stats(blocks[b], top_k, num_slots, all_labels,
                           all_toas, all_patch_delta,
                           seg_len=all_seg_len, nf=NF, smooth=args.smooth)
        stats[f'block{b}'] = st
        if not st:
            continue
        print(f"\n  --- block {b} ({st['n_steps']} steps) ---")
        print(f"    turnover      Jaccard={st['jaccard_mean']:.4f}  "
              f"chance={st['chance_jaccard']:.4f}  "
              f"excess={st['jaccard_excess']:.4f}  static={st['static_frac']:.3f}")
        print(f"    gate-input    corr(1-Jaccard, |d patch|) = {st['gate_input_corr']:.4f}")
        print(f"    onset         Jaccard at toa={st['jaccard_at_onset']:.4f} "
              f"vs baseline={st['jaccard_baseline']:.4f}")
        print(f"    usage         mean={st['usage_mean']:.4f} (expect {top_k / num_slots:.4f})  "
              f"CV/chance={st['usage_cv_ratio']:.3f}  effective_K={st['effective_k']:.2f}"
              f"  dead={st['dead_slots']}")
        print(f"    gate entropy  mean={st['gate_entropy_mean']:.4f} / "
              f"uniform {st['gate_entropy_uniform']:.4f} "
              f"(norm {st['gate_entropy_norm']:.3f})")
        print(f"    cutoff margin mean={st['margin_mean']:.4f}  "
              f"frac<0.1={st['margin_frac_below_0p1']:.3f}")
        print(f"    label assoc   MI/slot: mean={st['mi_mean_raw']:.5f} "
              f"max={st['mi_max_raw']:.4f} nats  "
              f"(max excess over null={st['mi_max_excess']:.4f}, "
              f"{st['n_slots_sig']}/{st['n_slots_kept']} slots z>2)")
        print(f"                  max-statistic null: q95={st['mi_max_null_q95']:.4f} "
              f"p={st['mi_max_pvalue']:.3f}  "
              f"-> {'BEST SLOT BEATS CHANCE' if st['mi_max_pvalue'] < 0.05 else 'not distinguishable from chance'}")
        print(f"                  ratio: max(kept)={st['ratio_max']:.2f}  "
              f"max(ALL)={st['ratio_max_all']:.2f} "
              f"(z of that slot={st['ratio_argmax_z']:+.1f})")
        for (s, pa, pn, rt, zz, nact) in (st.get('ratio_top5') or [])[:3]:
            print(f"                    slot {s:2d}: anom={pa:.3f} norm={pn:.3f} "
                  f"ratio={rt:.2f} z={zz:+.1f} n_active={nact}")
        print(f"    spatial       cross-attn entropy={st['cross_attn_entropy']:.4f}  "
              f"centroid spread={st['centroid_spread']:.4f}  "
              f"pairwise dist={st['centroid_pairdist']:.4f}")
        print(f"    frozen audit  inactive-update-zero frac={st['frozen_update_frac']:.4f}  "
              f"active delta={st['active_delta']:.4f}  inactive delta={st['inactive_delta']:.4f}")
        print(f"    read mass     to-inactive frac={st['read_inactive_frac']:.4f}  "
              f"Gini={st['read_gini']:.4f}")
        _tg = st.get('trigger_by_age') or {}
        if _tg:
            b = _tg.get('_base', {})
            print(f"    TRIGGER by prior dormancy (base: anom_rate={b.get('anom_rate',float('nan')):.3f}, "
                  f"|d patch|={b.get('patch_delta',float('nan')):.3f}, n={b.get('n')}):")
            for k in ('recent_1_2', 'mid_3_10', 'dormant_gt10'):
                if k in _tg:
                    v = _tg[k]
                    print(f"      {k:<14} n={v['n']:<7} anom_rate={v['anom_rate']:.3f}  "
                          f"|d patch|={v['patch_delta']:.3f}")
        _rba = st.get('read_by_age') or {}
        print("    read mass by age: " + "  ".join(
            f"{k}:share={v['share']:.4f}(n={v['n']})" for k, v in _rba.items()))
        print(f"    read mass share to age>10 slots = {st['read_share_age_gt10']:.4f}")
        print(f"    persistence   corr(read_mass, slot age)={st['read_age_corr']:+.4f}"
              f"   update interval: slowest={st['slow_interval']:.1f} fastest={st['fast_interval']:.1f} steps")
        print(f"    cos-sim MEAN over frames: all={st['slot_sim_all']:.4f} "
              f"active={st['slot_sim_active']:.4f} inactive={st['slot_sim_inactive']:.4f}"
              f"   (active sd={st['slot_sim_active_std']:.3f})")
        print(f"    cos-sim LAST frame only: all={st['slot_sim_all_last']:.4f} "
              f"active={st['slot_sim_active_last']:.4f} inactive={st['slot_sim_inactive_last']:.4f}"
              f"   <- what diag_sparse_gate.py reports")
        print(f"    slot cos-sim  all={st['slot_sim_all']:.4f}  "
              f"active={st['slot_sim_active']:.4f}  "
              f"inactive={st['slot_sim_inactive']:.4f}   "
              f"(high = slots redundant)")

    # ---- cross-block view of slot usage ---------------------------------
    # `dead_slots` above is PER BLOCK.  Block is DEPTH, not time: the four
    # blocks run in sequence inside every timestep, threading one slot tensor,
    # and each owns its own gate.  So "dead in block 0" is a fact about one
    # gate's working set, not about the model, and it does NOT mean the slot
    # was later revived -- usage is aggregated over all T steps and carries no
    # temporal order at all.
    #
    # What the threading DOES make meaningful is the union: a slot is written
    # in step t if ANY block selected it, so the per-slot update rate at the
    # model level is the OR over blocks, which is >= every block's own rate.
    # Block-level "dead" therefore overstates deadness at the model level.
    xblock = {}
    _live = [st for st in (stats.get(f'block{b}') for b in range(len(blocks)))
             if st and st.get('usage_per_slot') is not None]
    if len(_live) > 1:
        U = np.asarray([st['usage_per_slot'] for st in _live], dtype=np.float64)
        dead_sets = [set(st['dead_slot_ids']) for st in _live]
        # --- EXACT model-level write rate --------------------------------
        # Per-step OR over blocks, from the per-step active sets.  This is the
        # quantity the question actually asks ("was slot k written this step"),
        # and it is NOT max(usage_b): max over AGGREGATES is only a LOWER bound
        # on it, tight only when one block's selections contain the others'.
        # Cheap: T * n_blocks * top_k index updates (~640k at T=9964).
        _A = [blocks[b]['active_idx'] for b in range(len(blocks))]
        _Tb = min(len(a) for a in _A)
        _or = np.zeros(num_slots)
        for t in range(_Tb):
            seen = set()
            for a in _A:
                seen.update(np.asarray(a[t]).reshape(-1).tolist())
            for k in seen:
                _or[int(k)] += 1
        or_rate = _or / max(_Tb, 1)
        any_usage = U.max(axis=0)          # LOWER bound on the true rate
        union = sorted(set().union(*dead_sets)) if dead_sets else []
        inter = sorted(set.intersection(*dead_sets)) if dead_sets else []
        xblock = dict(
            n_blocks=len(_live),
            dead_ids_per_block=[sorted(s) for s in dead_sets],
            dead_union_ids=union,
            dead_intersection_ids=inter,
            n_dead_pairs=len(sum([st['dead_slot_ids'] for st in _live], [])),
            any_block_usage_min=float(any_usage.min()),
            any_block_usage_mean=float(any_usage.mean()),
            dead_any_block=int((any_usage < 0.02).sum()),
            dead_every_block=int((U < 0.02).all(axis=0).sum()),
            effective_k_any_block=float(np.exp(_entropy(
                (any_usage / max(any_usage.sum(), 1e-12)) + 1e-12))),
            # the exact version of the two numbers above
            or_steps=int(_Tb),
            or_rate_min=float(or_rate.min()),
            or_rate_mean=float(or_rate.mean()),
            dead_or=int((or_rate < 0.02).sum()),
            effective_k_or=float(np.exp(_entropy(
                (or_rate / max(or_rate.sum(), 1e-12)) + 1e-12))),
            or_usage_per_slot=[float(u) for u in or_rate],
            eff_k_per_block=[float(st['effective_k']) for st in _live],
            dead_per_block=[int(st['dead_slots']) for st in _live],
        )
        print("\n" + "-" * 84)
        print("CROSS-BLOCK SLOT USAGE  (block = depth, not time)")
        print("-" * 84)
        print(f"  dead per block        : {xblock['dead_per_block']}")
        print(f"  eff_K per block       : "
              + "  ".join(f"{v:.2f}" for v in xblock['eff_k_per_block']))
        print(f"  dead ids per block    : {xblock['dead_ids_per_block']}")
        print(f"  union of dead ids     : {union}  ({len(union)} distinct)")
        print(f"  in EVERY block's dead : {inter}")
        print(f"  written by SOME block : min usage over slots = "
              f"{xblock['any_block_usage_min']:.4f} (max-of-aggregates LOWER bound), "
              f"dead(<0.02) at model level = {int(xblock['dead_any_block'])}")
        print(f"  EXACT per-step OR     : min rate over slots = "
              f"{xblock['or_rate_min']:.4f}  mean = {xblock['or_rate_mean']:.4f}  "
              f"dead(<0.02) = {int(xblock['dead_or'])}")
        print(f"  eff_K at model level  : {xblock['effective_k_or']:.2f} (exact OR)"
              f"   {xblock['effective_k_any_block']:.2f} (lower bound)"
              f"   vs {min(xblock['eff_k_per_block']):.2f}-"
              f"{max(xblock['eff_k_per_block']):.2f} per block")

    # ---- JSON ----
    payload = dict(
        config=args.config, checkpoint=args.checkpoint, mode=args.mode,
        top_k=int(top_k), num_slots=int(num_slots), seed=args.seed,
        route_seed=args.route_seed,
        smooth=args.smooth, n_videos=len(tgt), n_frames=int(n_frames),
        pooled_auc=float(auc), mean_video_auc=float(np.nanmean(pv)),
        # Per-video AUC lets modes be compared with a PAIRED test on identical
        # videos, instead of comparing two aggregate numbers whose difference is
        # the same size as the measurement noise.
        per_video_auc=[None if not np.isfinite(v) else float(v) for v in pv],
        # Key indices of the scored videos, so runs can be aligned for paired tests.
        video_keys=[int(k) for k in vid_keys],
        latency_ms_mean=float(np.mean(frame_times)),
        latency_ms_p90=float(np.percentile(frame_times, 90)),
        event=em, routing=stats,
        # Cross-block slot-usage view (per-block dead ids, the union, and the
        # exact per-step OR write rate).  Printed as well, but it belongs in the
        # json: it is the only model-level usage statistic here, and everything
        # under `routing` is PER BLOCK.
        xblock=xblock,
    )
    out_path = args.json_out or os.path.join(
        _REPO_ROOT, 'output',
        f"mech_{os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))}_"
        f"{args.mode}.json")
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
