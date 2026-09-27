"""
Threshold-free comparison of Sparse SlotSSM runs that differ only in ``top_k``.

Reads the ``results-*.pkl`` files written by ``_evaluate_model`` (main.py) and
computes metrics that do NOT depend on a 0.5 decision threshold, because these
models are badly miscalibrated around 0.5 (F1@0.5 is meaningless here).

What it measures
----------------
Per run, per epoch (all threshold-free):
  * pooled frame-level AUC-ROC and PR-AUC  (the numbers already reported)
  * mean within-video AUC  — separates normal from anomaly *inside one video*,
    so it is immune to cross-video score-scale drift that can move pooled AUC
    for reasons unrelated to ranking
  * score separation d' = (mu_anom - mu_norm) / pooled_sd, plus the two means
    (if top_k shifts the score scale, this is where it shows)
  * temporal jitter — mean |dp| between consecutive frames on normal segments

Detection quality at a *matched false-alarm budget* (the threshold-free way to
compare detectors that are not calibrated at 0.5):
  * threshold calibrated so that normal frames exceed it at a fixed FA/minute
  * event recall (any above-threshold frame inside the [toa, tea] window),
    median detection delay, achieved FA/min

Cross-run (paired — all runs score the identical videos in the same order):
  * paired bootstrap CI on the AUC *difference* (videos resampled, not frames)
  * Wilcoxon signed-rank on per-video AUC
  * per-class / per-ego / per-night AUC

Usage (WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad

    python tests/analyze_topk_sweep.py
    python tests/analyze_topk_sweep.py --n_boot 500 --smooth 5
    python tests/analyze_topk_sweep.py --epochs 20 30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

ANOMALIES = [
    'start_stop_or_stationary', 'moving_ahead_or_waiting', 'lateral',
    'oncoming', 'turning', 'pedestrian', 'obstacle',
    'leave_to_right', 'leave_to_left', 'unknown',
]

# info columns, from Dota.gather_info() -> data_info[7:11]
INFO_COLS = {'class': 0, 'ego': 1, 'night': 2, 'has_objects': 3}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _epoch_of(path: str) -> int:
    m = re.search(r'results-(\d+)\.pkl$', os.path.basename(path))
    return int(m.group(1))


def _scalar(text: str, key: str, cast=float):
    """Pull a top-level ``key: value`` scalar out of a dumped config text."""
    m = re.search(rf'^{re.escape(key)}:\s*([^\[\n#]+)', text, re.MULTILINE)
    if not m:
        return None
    try:
        return cast(m.group(1).strip())
    except (TypeError, ValueError):
        return None


def discover_runs(base: str, pattern: str = 'sparse_slotssm', epoch_stride: int = 10):
    """Find output dirs containing eval pickles; read top_k/VCL/FPS from cfg.yml.

    ``epoch_stride`` keeps only epochs that are multiples of it. An output dir
    can hold pickles from more than one run: a completed run validating every
    10 epochs does not overwrite the odd-epoch pickles left by an earlier,
    differently-configured run, and those stale files are not comparable.
    """
    runs = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        eval_dir = os.path.join(path, 'eval')
        if not os.path.isdir(eval_dir) or pattern not in name:
            continue
        all_pk = sorted(glob.glob(os.path.join(eval_dir, 'results-*.pkl')), key=_epoch_of)
        pk = [p for p in all_pk
              if epoch_stride <= 1 or _epoch_of(p) % epoch_stride == 0]
        if not pk:
            continue

        top_k, vcl, fps = None, None, 10
        cfg_path = os.path.join(path, 'cfg.yml')
        if os.path.exists(cfg_path):
            # cfg.yml is dumped with a !!python/object/apply:torch.device tag, so
            # yaml.safe_load refuses it. Scrape only the scalar keys we need.
            with open(cfg_path) as f:
                text = f.read()
            top_k = _scalar(text, 'top_k', int)
            vcl = _scalar(text, 'VCL', int)
            fps = _scalar(text, 'FPS', float) or 10
        if top_k is None:
            m = re.search(r'topk_(\d+)', name)
            top_k = int(m.group(1)) if m else 16

        runs.append({
            'name': name, 'path': path, 'top_k': int(top_k),
            'vcl': vcl, 'fps': fps,
            'pickles': {_epoch_of(p): p for p in pk},
            'n_dropped': len(all_pk) - len(pk),
            'dropped_epochs': sorted(_epoch_of(p) for p in all_pk
                                     if p not in set(pk)),
        })
    return runs


def load_pickle(path: str):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return {
        'targets': [np.asarray(t, dtype=np.float64) for t in d['targets']],
        'outputs': [np.asarray(o, dtype=np.float64) for o in d['outputs']],
        'toas': np.asarray(d['toas'], dtype=np.float64),
        'teas': np.asarray(d['teas'], dtype=np.float64),
        'idxs': np.asarray(d['idxs'], dtype=np.float64),
        'info': np.asarray(d['info'], dtype=np.float64),
        'n_frames': np.asarray(d['frames_counter'], dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Metric helpers — all threshold-free unless stated
# ---------------------------------------------------------------------------
def pooled_auc(targets, outputs, idx=None):
    if idx is None:
        y = np.concatenate(targets)
        s = np.concatenate(outputs)
    else:
        y = np.concatenate([targets[i] for i in idx])
        s = np.concatenate([outputs[i] for i in idx])
    return roc_auc_score(y, s)


def pooled_pr_auc(targets, outputs):
    y = np.concatenate(targets)
    s = np.concatenate(outputs)
    return average_precision_score(y, s)


def per_video_auc(targets, outputs):
    """AUC within each video. Returns (per_video_array, valid_mask)."""
    aucs = np.full(len(targets), np.nan)
    for v, (t, p) in enumerate(zip(targets, outputs)):
        if t.size == 0 or t.min() == t.max():
            continue
        aucs[v] = roc_auc_score(t, p)
    return aucs


def score_separation(targets, outputs):
    y = np.concatenate(targets)
    s = np.concatenate(outputs)
    n_mask, a_mask = y == 0, y == 1
    if not a_mask.any() or not n_mask.any():
        return dict(mu_norm=np.nan, mu_anom=np.nan, dprime=np.nan,
                    sd_norm=np.nan, sd_anom=np.nan)
    m_n, m_a = s[n_mask].mean(), s[a_mask].mean()
    sd_n, sd_a = s[n_mask].std(), s[a_mask].std()
    pooled_sd = np.sqrt((sd_n ** 2 + sd_a ** 2) / 2) + 1e-12
    return dict(mu_norm=float(m_n), mu_anom=float(m_a),
                sd_norm=float(sd_n), sd_anom=float(sd_a),
                dprime=float((m_a - m_n) / pooled_sd))


def temporal_jitter(targets, outputs):
    """Mean |dp| on consecutive frames whose *both* labels are normal."""
    diffs = []
    for t, p in zip(targets, outputs):
        if t.size < 2:
            continue
        d = np.abs(np.diff(p))
        both_normal = (t[:-1] == 0) & (t[1:] == 0)
        if both_normal.any():
            diffs.append(d[both_normal])
    if not diffs:
        return float('nan')
    return float(np.concatenate(diffs).mean())


def bootstrap_pooled(targets, outputs, n_boot, rng):
    """Bootstrap distribution of pooled AUC over *videos*."""
    n = len(targets)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        out[b] = pooled_auc(targets, outputs, idx)
    return out


def paired_bootstrap_diff(t_a, o_a, t_b, o_b, n_boot, rng):
    """Paired bootstrap over videos of AUC(a) - AUC(b). Same resample for both."""
    n = len(t_a)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        out[b] = pooled_auc(t_a, o_a, idx) - pooled_auc(t_b, o_b, idx)
    return out


def wilcoxon_paired(x, y):
    """Wilcoxon signed-rank p-value; falls back to NaN if scipy lacks it."""
    try:
        from scipy.stats import wilcoxon
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10 or np.allclose(x[m], y[m]):
            return float('nan')
        return float(wilcoxon(x[m], y[m]).pvalue)
    except Exception:
        return float('nan')


# ---------------------------------------------------------------------------
# FPR-matched detection metrics
# ---------------------------------------------------------------------------
def event_metrics(toas, teas, targets, outputs, fps, fa_budgets=(0.5, 1.0, 2.0, 5.0),
                  smooth=5):
    """Detection metrics at fixed false-alarm budgets (threshold-free comparison).

    The threshold is NOT 0.5 — it is calibrated so that normal frames exceed it
    at ``budget`` false alarms per minute of normal video time. Detection = at
    least one above-threshold frame inside [toa, tea]. Delay is measured from
    the first such frame.
    """
    norm_scores = np.concatenate([p[t == 0] for t, p in zip(targets, outputs)
                                  if (t == 0).any()])
    n_norm = norm_scores.size
    if n_norm == 0:
        return []
    norm_minutes = n_norm / (fps * 60.0)
    desc = np.sort(norm_scores)[::-1]

    results = []
    for budget in fa_budgets:
        n_allowed = int(round(budget * norm_minutes))
        n_allowed = min(max(n_allowed, 0), n_norm - 1)
        thr = desc[n_allowed]

        detected, delays, first_any = 0, [], []
        for toa, tea, t, p in zip(toas, teas, targets, outputs):
            ps = uniform_filter1d(p, size=smooth, mode='nearest')
            above = np.flatnonzero(ps > thr)
            if above.size:
                first_any.append(above[0] - toa)
            lo, hi = int(toa), int(tea)
            in_win = above[(above >= lo) & (above <= hi)]
            if in_win.size:
                detected += 1
                delays.append(in_win[0] - lo)
            else:
                delays.append(np.nan)

        # achieved FA/min: recompute the realised count, not the target
        realised = int((norm_scores > thr).sum())
        delays = np.asarray(delays, dtype=np.float64)
        results.append(dict(
            fa_budget=budget,
            threshold=float(thr),
            fa_per_min_realised=float(realised / norm_minutes),
            event_recall=float(detected / len(toas)),
            median_delay=float(np.nanmedian(delays)) if np.isfinite(delays).any() else float('nan'),
            mean_delay=float(np.nanmean(delays)) if np.isfinite(delays).any() else float('nan'),
            first_delay_any=float(np.median(first_any)) if first_any else float('nan'),
        ))
    return results


def f1_optimal(targets, outputs, n_grid=200):
    """Best F1 over a threshold sweep (threshold optimised, never fixed at 0.5)."""
    y = np.concatenate(targets)
    s = np.concatenate(outputs)
    qs = np.quantile(s, np.linspace(0.01, 0.99, n_grid))
    best = 0.0
    for thr in qs:
        pred = s > thr
        tp = float((pred & (y == 1)).sum())
        fp = float((pred & (y == 0)).sum())
        fn = float((~pred & (y == 1)).sum())
        if tp == 0:
            continue
        f1 = 2 * tp / (2 * tp + fp + fn)
        best = max(best, f1)
    return best


# ---------------------------------------------------------------------------
# Grouped AUC (class / ego / night)
# ---------------------------------------------------------------------------
def grouped_auc(targets, outputs, info, col, names=None):
    """AUC per group defined by info[:, col]. Groups with one class are skipped."""
    keys = np.unique(info[:, col])
    out = {}
    for k in keys:
        idx = np.flatnonzero(info[:, col] == k)
        if idx.size == 0:
            continue
        try:
            auc = pooled_auc(targets, outputs, idx)
        except ValueError:
            continue
        label = str(int(k))
        if names is not None and col == INFO_COLS['class']:
            i = int(k) - 1
            label = names[i] if 0 <= i < len(names) else label
        out[label] = dict(n_videos=int(idx.size), auc=float(auc))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _f(x, w=8, p=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ' ' * (w - 3) + 'n/a'
    return f'{x:>{w}.{p}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=os.path.join(_REPO_ROOT, 'output'))
    ap.add_argument('--pattern', default='sparse_slotssm',
                    help='Substring a run dir name must contain to be discovered. '
                         "Default 'sparse_slotssm'. Use 'vjepa_slotssm' for the "
                         "DENSE arms (it is not a substring of "
                         "'vjepa_sparse_slotssm...'), or point --output_dir at a "
                         'folder holding one architecture.')
    ap.add_argument('--runs', nargs='*', default=None,
                    help='Explicit run dir names (default: auto-discover)')
    ap.add_argument('--epochs', nargs='*', type=int, default=None,
                    help='Epochs to compare across runs (default: intersection)')
    ap.add_argument('--epoch_stride', type=int, default=10,
                    help='Keep only epochs that are multiples of this. 1 = keep all. '
                         'Guards against stale odd-epoch pickles from an earlier run.')
    ap.add_argument('--n_boot', type=int, default=300)
    ap.add_argument('--smooth', type=int, default=5,
                    help='Moving-average window (frames) for event detection')
    ap.add_argument('--event_epoch', type=int, default=None,
                    help='Force one epoch for the event table across all runs. '
                         'Default: each run\'s own peak-AUC epoch (not matched).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json_out', default=None)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    runs = discover_runs(args.output_dir, pattern=args.pattern,
                         epoch_stride=args.epoch_stride)
    if args.runs:
        runs = [r for r in runs if r['name'] in set(args.runs)]
    if not runs:
        print(f'No runs matching {args.pattern!r} with eval pickles under '
              f'{args.output_dir}')
        sys.exit(1)

    # ---- inventory + load ----
    print('=' * 100)
    print('RUN INVENTORY')
    print('=' * 100)
    print(f"epoch_stride={args.epoch_stride} "
          f"(non-multiples are dropped as stale leftovers from earlier runs)")
    print(f"{'run':<52} {'top_k':>6} {'VCL':>5} {'epochs':>24} {'dropped':>16}")
    for r in runs:
        ep = sorted(r['pickles'])
        ep_s = ','.join(str(e) for e in ep)
        if len(ep_s) > 23:
            ep_s = ep_s[:20] + '...'
        dr = ','.join(str(e) for e in r['dropped_epochs'])
        if len(dr) > 15:
            dr = dr[:12] + '...'
        print(f"{r['name']:<52} {r['top_k']:>6} {str(r['vcl']):>5} {ep_s:>24} {dr:>16}")

    print('\nLoading pickles...')
    for r in runs:
        r['data'] = {e: load_pickle(p) for e, p in sorted(r['pickles'].items())}
        print(f"  {r['name']:<52} {len(r['data'])} epoch(s) loaded")

    # ---- pairing check: identical videos in identical order? ----
    print('\n' + '=' * 100)
    print('PAIRING CHECK')
    print('=' * 100)
    ref = runs[0]
    ref_epoch = max(ref['data'])
    ref_d = ref['data'][ref_epoch]
    ref_key = (tuple(len(t) for t in ref_d['targets']), tuple(ref_d['idxs'].tolist()))
    paired = True
    for r in runs[1:]:
        e = max(r['data'])
        d = r['data'][e]
        key = (tuple(len(t) for t in d['targets']), tuple(d['idxs'].tolist()))
        same = key == ref_key
        paired &= same
        print(f"  {r['name']:<52} {'MATCH' if same else 'MISMATCH'}  "
              f"({len(d['targets'])} videos)")
    n_videos = len(ref_d['targets'])
    n_frames = sum(len(t) for t in ref_d['targets'])
    y_all = np.concatenate(ref_d['targets'])
    print(f"\n  videos={n_videos}  frames={n_frames}  "
          f"normal={int((y_all == 0).sum())}  anomaly={int((y_all == 1).sum())}")
    if not paired:
        print('  WARNING: video sets differ -> cross-run comparisons are NOT paired.')

    # ---- per-run epoch trajectory ----
    print('\n' + '=' * 100)
    print('TABLE A — per-run epoch trajectory (threshold-free)')
    print('=' * 100)
    print("'mvAUC' = mean within-video AUC (immune to cross-video score-scale drift)")
    print(f"{'run':<44} {'ep':>4} {'AUC':>9} {'PR-AUC':>9} {'mvAUC':>9} {'d-prime':>9} {'jitter':>9}")
    traj = {}
    for r in runs:
        traj[r['name']] = {}
        for e, d in sorted(r['data'].items()):
            auc = pooled_auc(d['targets'], d['outputs'])
            pr = pooled_pr_auc(d['targets'], d['outputs'])
            pv = per_video_auc(d['targets'], d['outputs'])
            sep = score_separation(d['targets'], d['outputs'])
            jit = temporal_jitter(d['targets'], d['outputs'])
            traj[r['name']][e] = dict(auc=float(auc), pr_auc=float(pr),
                                     mv_auc=float(np.nanmean(pv)),
                                     mv_auc_std=float(np.nanstd(pv)),
                                     dprime=sep['dprime'], jitter=jit,
                                     mu_norm=sep['mu_norm'], mu_anom=sep['mu_anom'])
            print(f"{r['name']:<44} {e:>4} {_f(auc, 9)} {_f(pr, 9)} "
                  f"{_f(np.nanmean(pv), 9)} {_f(sep['dprime'], 9)} {_f(jit, 9)}")
        best = max(traj[r['name']], key=lambda k: traj[r['name']][k]['auc'])
        print(f"{'  -> peak AUC at epoch ' + str(best):<44} {'':>4} "
              f"{_f(traj[r['name']][best]['auc'], 9)}")

    # ---- epochs to compare ----
    if args.epochs:
        epochs = [e for e in args.epochs if all(e in r['data'] for r in runs)]
    else:
        common = set(runs[0]['data'])
        for r in runs[1:]:
            common &= set(r['data'])
        epochs = sorted(common)
    if not epochs:
        print('\nNo epochs common to all runs — skipping cross-run tables.')
        epochs = []

    # ---- matched-epoch comparison ----
    if epochs:
        print('\n' + '=' * 100)
        print(f'TABLE B — matched-epoch comparison (epochs: {epochs})')
        print('=' * 100)
        hdr = f"{'top_k':>6} {'ep':>4} {'AUC':>9} {'[95% CI]':>20} {'PR-AUC':>9} " \
              f"{'mvAUC':>9} {'F1-opt':>8} {'d-prime':>9}"
        print(hdr)
        for e in epochs:
            for r in sorted(runs, key=lambda x: x['top_k']):
                d = r['data'][e]
                auc = pooled_auc(d['targets'], d['outputs'])
                boot = bootstrap_pooled(d['targets'], d['outputs'], args.n_boot, rng)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                pv = per_video_auc(d['targets'], d['outputs'])
                sep = score_separation(d['targets'], d['outputs'])
                f1o = f1_optimal(d['targets'], d['outputs'])
                print(f"{r['top_k']:>6} {e:>4} {_f(auc, 9)} "
                      f"{'[' + _f(lo, 7) + ',' + _f(hi, 7) + ']':>20} "
                      f"{_f(pooled_pr_auc(d['targets'], d['outputs']), 9)} "
                      f"{_f(np.nanmean(pv), 9)} {_f(f1o, 8)} {_f(sep['dprime'], 9)}")
            print()

        # ---- best-of on the common epoch grid ----
        print('=' * 100)
        print('TABLE F — best checkpoint per run, restricted to the COMMON epoch grid')
        print('=' * 100)
        print('This is the "report your best checkpoint" comparison, made fair by giving')
        print(f'every run the same {len(epochs)} draws ({epochs}). A run with more')
        print('validation points gets more chances to peak on noise, so an unrestricted')
        print('best-of favours whichever run validated most often.')
        print(f"\n{'top_k':>6} {'peak ep':>8} {'AUC':>9} {'[95% CI]':>20} "
              f"{'vs grid mean':>13} {'peak gain':>10}")
        best_of = {}
        for r in sorted(runs, key=lambda x: x['top_k']):
            e_best = max(epochs, key=lambda k: traj[r['name']][k]['auc'])
            d = r['data'][e_best]
            auc = pooled_auc(d['targets'], d['outputs'])
            boot = bootstrap_pooled(d['targets'], d['outputs'], args.n_boot, rng)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            grid_mean = np.mean([traj[r['name']][k]['auc'] for k in epochs])
            best_of[r['top_k']] = dict(run=r['name'], epoch=e_best, auc=float(auc),
                                       ci=[float(lo), float(hi)])
            print(f"{r['top_k']:>6} {e_best:>8} {_f(auc, 9)} "
                  f"{'[' + _f(lo, 7) + ',' + _f(hi, 7) + ']':>20} "
                  f"{_f(grid_mean, 13)} {_f(auc - grid_mean, 10)}")
        print('\n  "peak gain" = best minus mean over the grid: how much of the reported')
        print('  number is genuine level vs. selection on epoch noise.')

        ks = sorted(best_of)
        print('\n  Paired comparison of the best-of checkpoints:')
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                ka, kb = ks[j], ks[i]           # ka = higher top_k
                ra = next(r for r in runs if r['top_k'] == ka)
                rb = next(r for r in runs if r['top_k'] == kb)
                da = ra['data'][best_of[ka]['epoch']]
                db = rb['data'][best_of[kb]['epoch']]
                diff = paired_bootstrap_diff(da['targets'], da['outputs'],
                                             db['targets'], db['outputs'],
                                             args.n_boot, rng)
                lo, hi = np.percentile(diff, [2.5, 97.5])
                sig = 'SIG' if (lo > 0 or hi < 0) else '   '
                print(f"    top_k={ka} (ep {best_of[ka]['epoch']}) vs "
                      f"top_k={kb} (ep {best_of[kb]['epoch']}):  "
                      f"dAUC={best_of[ka]['auc'] - best_of[kb]['auc']:+.4f}  "
                      f"CI=[{lo:+.4f},{hi:+.4f}] {sig}")

        # ---- paired comparisons ----
        print('\n' + '=' * 100)
        print('TABLE C — paired comparisons at MATCHED epochs (same videos)')
        print('=' * 100)
        print('Answers "at a fixed training budget, does top_k change the achievable')
        print('AUC?" — it holds training dynamics constant, which best-of does not.')
        print('Delta = AUC(row) - AUC(col). CI is the paired bootstrap CI of the')
        print('difference: if it straddles 0 the two top_k are indistinguishable.')
        print('NOTE: this CI measures ranking consistency across the *same* test')
        print('videos. It is not a seed/epoch variance estimate — for that, compare')
        print('it against the epoch-to-epoch swing in TABLE A.')
        sorted_runs = sorted(runs, key=lambda x: x['top_k'])
        for i in range(len(sorted_runs)):
            for j in range(i + 1, len(sorted_runs)):
                ra, rb = sorted_runs[i], sorted_runs[j]
                print(f"\n  top_k={ra['top_k']} (higher)  vs  top_k={rb['top_k']}")
                for e in epochs:
                    da, db = ra['data'][e], rb['data'][e]
                    auc_a = pooled_auc(da['targets'], da['outputs'])
                    auc_b = pooled_auc(db['targets'], db['outputs'])
                    diff = paired_bootstrap_diff(da['targets'], da['outputs'],
                                                 db['targets'], db['outputs'],
                                                 args.n_boot, rng)
                    lo, hi = np.percentile(diff, [2.5, 97.5])
                    p = float((diff < 0).mean()) if hasattr(diff, 'mean') else float('nan')
                    pva = per_video_auc(da['targets'], da['outputs'])
                    pvb = per_video_auc(db['targets'], db['outputs'])
                    wp = wilcoxon_paired(pva, pvb)
                    sig = 'SIG' if (lo > 0 or hi < 0) else '   '
                    print(f"    ep {e:>3}: dAUC={auc_a - auc_b:+.4f}  "
                          f"CI=[{lo:+.4f},{hi:+.4f}] {sig}  "
                          f"P(row<col)={p:.3f}  "
                          f"dmvAUC={np.nanmean(pva) - np.nanmean(pvb):+.4f}  "
                          f"wilcoxon p={wp:.4f}")

        # ---- grouped AUC at peak epoch per run ----
        print('\n' + '=' * 100)
        print('TABLE D — grouped AUC at each run\'s peak-AUC epoch')
        print('=' * 100)
        for r in sorted(runs, key=lambda x: x['top_k']):
            best = max(r['data'], key=lambda k: traj[r['name']][k]['auc'])
            d = r['data'][best]
            print(f"\n  top_k={r['top_k']}  (peak epoch {best}, AUC={traj[r['name']][best]['auc']:.4f})")
            for col, title in (('class', 'per accident class'),
                               ('ego', 'per ego-involvement'),
                               ('night', 'per lighting (0=day,1=night)'),
                               ('has_objects', 'per has_objects')):
                g = grouped_auc(d['targets'], d['outputs'], d['info'],
                                INFO_COLS[col], names=ANOMALIES if col == 'class' else None)
                if not g:
                    continue
                items = sorted(g.items())
                line = '  '.join(f"{k}={v['auc']:.3f}" for k, v in items)
                print(f"    {title:<26}: {line}")

        # ---- event metrics ----
        print('\n' + '=' * 100)
        print('TABLE E — detection at matched false-alarm budgets (peak epoch per run)')
        print('=' * 100)
        print(f'Threshold is calibrated to the FA budget (NOT 0.5). Smoothing window '
              f'= {args.smooth} frames.')
        print('Detection = >=1 above-threshold frame inside [toa, tea].')
        print('CAUTION: compare runs at similar *realised* FA/min, not the target —')
        print('tied normal scores can make the calibrated threshold coarse.')
        print(f"\n{'top_k':>6} {'ep':>4} {'FA/min':>7} {'thr':>7} {'realised':>9} "
              f"{'recall':>8} {'med delay':>10} {'mean delay':>11}")
        for r in sorted(runs, key=lambda x: x['top_k']):
            if args.event_epoch is not None:
                ep_e = args.event_epoch if args.event_epoch in r['data'] else None
            else:
                ep_e = max(r['data'], key=lambda k: traj[r['name']][k]['auc'])
            if ep_e is None:
                continue
            d = r['data'][ep_e]
            em = event_metrics(d['toas'], d['teas'], d['targets'], d['outputs'],
                               r['fps'], smooth=args.smooth)
            for m in em:
                print(f"{r['top_k']:>6} {ep_e:>4} {m['fa_budget']:>7.1f} "
                      f"{_f(m['threshold'], 7)} "
                      f"{_f(m['fa_per_min_realised'], 9)} {_f(m['event_recall'], 8)} "
                      f"{_f(m['median_delay'], 10, 1)} {_f(m['mean_delay'], 11, 1)}")
            print()
        print('  delays are in frames (FPS=10 -> 10 frames = 1.0 s)')

        # ---- verdict ----
        print('\n' + '=' * 100)
        print('VERDICT')
        print('=' * 100)
        for e in epochs:
            spread = []
            for r in sorted_runs:
                d = r['data'][e]
                spread.append((r['top_k'], pooled_auc(d['targets'], d['outputs'])))
            lo_k, lo_v = min(spread, key=lambda x: x[1])
            hi_k, hi_v = max(spread, key=lambda x: x[1])
            # CI of the extreme pair
            ra = next(r for r in runs if r['top_k'] == hi_k)
            rb = next(r for r in runs if r['top_k'] == lo_k)
            da, db = ra['data'][e], rb['data'][e]
            diff = paired_bootstrap_diff(da['targets'], da['outputs'],
                                         db['targets'], db['outputs'],
                                         args.n_boot, rng)
            lo, hi = np.percentile(diff, [2.5, 97.5])
            verdict = 'SEPARATED' if (lo > 0 or hi < 0) else 'within noise'
            print(f"  epoch {e:>3}: best top_k={hi_k} ({hi_v:.4f}) vs worst top_k={lo_k} "
                  f"({lo_v:.4f})  spread={hi_v - lo_v:.4f}  "
                  f"CI=[{lo:+.4f},{hi:+.4f}] -> {verdict}")

    # ---- JSON ----
    out_path = args.json_out or os.path.join(_REPO_ROOT, 'output', 'topk_sweep_analysis.json')
    payload = {
        'n_videos': n_videos, 'n_frames': int(n_frames), 'paired': bool(paired),
        'trajectories': traj,
        'epochs_compared': list(epochs),
        'best_of_common_grid': best_of if epochs else {},
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, default=float)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
