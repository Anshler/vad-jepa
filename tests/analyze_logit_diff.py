"""The INTERNALS TABLE for a set of trained arms: what did the readout do, and
what else moved?

NOW POINTED AT THE SLOT_POOL ARMS: control / plain attn pool / gated attn pool,
at scale 1 and 4 -- six heads of ONE run (output/attention-pool), shared init, so
every comparison is exactly paired. The question is whether replacing
`slot_query` with attention pooling made the readout SELECT (`pool_eff`).

Previously pointed at the ten-arm scale sweep (output/more-logit-scale) and, before
that, the six arms of output/logit-scale; those tables are in findings 16.10.1,
16.10.6 and 16.10. Set ARMS and EPOCH for whichever folder you are reading.

1. THE INTERNALS TABLE. For each arm, on REAL frames:
     attn_out_sim   cross-slot cosine of the cross-attention OUTPUT. The
                    quantity that says whether slots receive different inputs.
     eff_patches    exp(mean per-slot attention entropy) = patches each slot
                    effectively averages, of 36.
     ssm_full       cross-slot cosine of the flattened SSM state -- findings
                    16.5/16.6's metric.
     ssm_read       same on mean over (H, hdim) -- what the gate's readout
                    actually consumes.
     slot_full      cross-slot cosine of the slot state the gate is built from.
     pool_eff       exp(entropy) over the 32 slots of the CLASSIFIER's pooling
                    softmax. THIS is the column that answers the run's question:
                    32 = the readout is a mean of all slots, 1 = all mass on one
                    slot. Read from the model's own `_last_pool_attn`, so it is
                    the distribution actually used, for both `dot` and `attn`.
     pool_max       the same distribution's largest weight.
     recv_eff       exp(entropy) over the 32 slots of the SELF-ATTENTION's
                    RECEIVED-MASS distribution (head-averaged w summed over the
                    query axis). NOT a selectivity measure: summing over queries
                    makes it near-uniform whenever different queries read
                    different slots, so it looks flat precisely when reading is
                    selective and diverse. Reading it as one produced a wrong
                    claim on 2026-09-23 ("the self-attention read is flat in all
                    ten arms"), which is why it is kept but labelled.

   THE PER-QUERY SELF-ATTENTION MEASURE DOES NOT EXIST YET, and `recv_eff` is not
   a substitute. Two attempts were made and both were flawed: one was unstable
   between runs on the same arm (dense s1 read eff 31.92 / slot cosine 0.995 in
   one run and 29.52 / 0.844 in another), and for SPARSE arms the forward hook
   captured the active-slot QUERY SUBSET (`_self_attn_sparse` passes `q`, not the
   full slot set), so the keys were wrong. A correct version needs the logits
   recomputed per head from `in_proj_weight` with the keys taken from
   `space_attn_norm(slots)` in full, on a pinned frame set. Do that before
   quoting anything about self-attention selectivity.

   The cross-attention map itself is `attn_entropy_table.py`, not this.

2. RANDOM vs REAL -- OFF by default (`RANDOM_PASS`). `check_logit_scale.py`, the
   "authoritative" logit_scale table of findings 15.9.3, feeds
   `torch.randn(1, 36, in_features)` as the patches; every other differentiation
   script streams real frames, because 16.5 showed random inputs invert the
   ordering between the STATE signals. Whether they also distort the ATTENTION
   metrics is what that pass tests -- on the six old arms. Set RANDOM_PASS = True
   to re-run it; it doubles the sweep's cost.

    python tests/analyze_logit_diff.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))
try:
    import torchvision.transforms._functional_tensor as _ft
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
except ImportError:
    pass

import yaml  # noqa: E402
from easydict import EasyDict  # noqa: E402
from model import MambaCache, build_multi_head_vjepa  # noqa: E402
from _cfgpool import pool_mode_from_ckpt  # noqa: E402

DATA = '/mnt/d/Users/Chrysenberg69420/Downloads/DoTA_dataset'
# Sample size, overridable so the internals can be re-measured with more coverage
# (the report's convention is 100 videos; the default here was 8, which is fine
# for a quick look and thin for any 'X is unchanged' claim).
STEPS = int(os.environ.get('ADI_STEPS', 150))
MAX_VIDEOS = int(os.environ.get('ADI_VIDEOS', 8))
D = 'output/vcl-64'

# THE FINETUNED VCL=64 SPARSE ARMS: the same architecture at logit_scale 1 and 4.
# Two SEPARATE runs (wandb vyq8ss7t / o0gw7ci9), so these do NOT share an init and
# any difference between them carries the ~0.02 floor (findings 16.4/18.1) -- which
# is exactly why the question here is INTERNAL ("does the scale do anything
# underneath") rather than an AUC comparison. The internals are functions of the
# trained weights, so they need no pairing to be readable.
#
# Both arms: top_k 16, VCL 64, NF 4, train_encoder TRUE, gated readout, no PE.
# EPOCH is 20 because only model-20.pt was kept.
#
# Run directories are named EXPLICITLY rather than built from a suffix pattern:
# this folder uses `_VCL_64_NF_4_finetuned`, and the previous caller of this script
# hardcoded `_VCL_8_NF_4_frozen`, which silently built a path that did not exist.
RANDOM_PASS = False

# (label, cfg, run dir, epoch) -- EPOCH IS PER ARM because these runs kept
# different snapshots: dense only has model-30.pt, the sparse arms only model-20.pt.
# Comparing internals across epochs is weaker than at a matched epoch, but the
# mechanism here is structural (frozen updates vs all-updates), so it is readable;
# say so rather than pretending the epochs match.
ARMS = [
    ('dense1 ', 'cfgs/vjepa_slotssm_VCL_64_NF_4_finetuned.yaml',
     f'{D}/vjepa_slotssm_VCL_64_NF_4_finetuned', 30),
    ('dense4 ', 'cfgs/vjepa_slotssm_ls4_VCL_64_NF_4_finetuned.yaml',
     f'{D}/vjepa_slotssm_ls4_VCL_64_NF_4_finetuned', 20),
    ('sparse1', 'cfgs/vjepa_sparse_slotssm_VCL_64_NF_4_finetuned.yaml',
     f'{D}/vjepa_sparse_slotssm_VCL_64_NF_4_finetuned', 20),
    ('sparse4', 'cfgs/vjepa_sparse_slotssm_ls4_VCL_64_NF_4_finetuned.yaml',
     f'{D}/vjepa_sparse_slotssm_ls4_VCL_64_NF_4_finetuned', 20),
]

# Run only a subset: ADI_ARMS=dense1,dense4. The arms are 150-step GPU passes,
# and adding one arm should not cost a rerun of the ones already measured
# (their numbers are in findings 25.4). Empty = all arms.
_ONLY = [s.strip() for s in os.environ.get('ADI_ARMS', '').split(',') if s.strip()]
if _ONLY:
    _unknown = sorted(set(_ONLY) - {a[0].strip() for a in ARMS})
    if _unknown:
        raise SystemExit(f'ADI_ARMS names not in ARMS: {_unknown}. '
                         f'Known: {[a[0].strip() for a in ARMS]}')
    ARMS = [a for a in ARMS if a[0].strip() in _ONLY]


def build(rel, device, seed=0):
    torch.manual_seed(seed)
    cfg = EasyDict(yaml.safe_load(open(os.path.join(ROOT, rel))))
    cfg.device = device
    cfg.compile = False
    cfg._config_name = rel
    name = os.path.splitext(os.path.basename(rel))[0]
    cfg._head_cfgs_flat = [dict(cfg)]
    cfg._head_cfgs_flat[0]['name'] = name
    return build_multi_head_vjepa(cfg).heads[name]


def load(head, run, device, epoch):
    """FULL head load. The pool lives on the head, so a temporal-only load would
    leave it at init -- which would make a trained readout look untrained.

    IT ALSO CHECKS THE POOL MODE AGAINST THE CHECKPOINT. On 2026-09-24 every
    `_ls*` config became gated attention pooling (findings 23), while the arms
    trained before that date used `dot`. Building a gated head and loading a `dot`
    checkpoint does NOT fail -- `strict=False` loads the temporal weights and
    leaves the pool at its initialisation -- so `pool_eff` would describe an
    untrained readout and look like a plausible measurement. `pool_mode_from_ckpt`
    reads the mode off the checkpoint's parameter names and this asserts instead.
    """
    p = os.path.join(ROOT, run, 'checkpoints', f'model-{epoch}.pt')
    ck = torch.load(p, map_location=device, weights_only=False)
    sd = ck['model_state_dict']
    trained = pool_mode_from_ckpt(sd)
    built = ('dot' if head.slot_query is not None
             else 'gated' if head.slot_pool_U is not None else 'attn-plain')
    if trained != built:
        raise SystemExit(
            f'{run}: checkpoint was trained with slot_pool={trained!r} but the '
            f'config builds {built!r}. The pool parameters differ by name, so the '
            f'readout would sit at init and pool_eff would be meaningless. Point '
            f'ARMS at the run\'s own cfg.yml, or pin slot_pool in the config.')
    m, u = head.load_state_dict(sd, strict=False)
    return len([k for k in m if not k.startswith('encoder.')]), len(u)


def real_patches(head, device, max_videos=8, nf=4):
    import contextlib
    from movad_core.dota import Dota, setup_dota
    cfg = EasyDict(yaml.safe_load(
        open(os.path.join(ROOT, 'cfgs/vjepa_sparse_slotssm.yaml'))))
    tc = EasyDict(dict(cfg))
    tc.batch_size = 1
    tc.input_shape = cfg.get('input_shape', [384, 384])
    tc.data_path = DATA
    _, loader = setup_dota(Dota, tc, num_workers=0, VCL=None, phase='test')
    ds = loader.dataset
    rng = np.random.RandomState(42)
    amp = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(
        cfg.get('amp_dtype', 'fp32'))
    ctx = (torch.amp.autocast('cuda', dtype=amp) if amp
           else contextlib.nullcontext())
    for idx in rng.permutation(len(ds))[:max_videos].tolist():
        vd, info_raw = ds[idx]
        fr = (torch.from_numpy(vd).float() if isinstance(vd, np.ndarray)
              else vd.float())
        if fr.dim() == 4:
            fr = fr.permute(1, 0, 2, 3).unsqueeze(0)
        video = fr.to(device)
        info = (torch.tensor(info_raw).float().unsqueeze(0).to(device)
                if not isinstance(info_raw, torch.Tensor)
                else info_raw.float().unsqueeze(0).to(device))
        vl = int(info[:, 0].item())
        for i in range(nf, vl):
            with torch.no_grad(), ctx:
                p = head.encoder(video[:, :, i - nf:i, :, :],
                                 return_patches=True)
                # MUST pool before the temporal model, exactly as ClsVJEPA.forward
                # does. Without this the temporal model is fed the raw
                # 24x24x2 = 1152 encoder tokens instead of the 6x6 = 36 grid
                # cells it actually sees -- which silently changes the attention
                # sequence length, and `eff = exp(entropy)` is bounded by it.
                if getattr(head, '_use_spatial_grid', False):
                    p = head._spatial_pool_tokens(p, head._vjepa_n_temp)
                yield p


def offdiag(X):
    if X.shape[0] < 2:
        return float('nan')
    X = X.reshape(X.shape[0], -1).float()
    X = X / (X.norm(dim=-1, keepdim=True) + 1e-12)
    G = X @ X.t()
    return float(G[~torch.eye(X.shape[0], dtype=torch.bool)].mean())


def measure(head, device, source):
    """source: 'real' (stream frames) or 'random' (iid patches, as 15.9.3 does)."""
    t = head.temporal
    K = t.num_slots
    cache = MambaCache()
    acc = {'attn': [], 'eff': [], 'ssm_full': [], 'ssm_read': [], 'slot': [],
           'pool_ent': [], 'pool_max': [], 'recv_ent': [], 'recv_max': []}
    captured = {}

    def mk(b):
        def hook(_m, _i, out):
            captured[b] = out[0] if isinstance(out, tuple) else out
        return hook
    handles = [blk.cross_attn.register_forward_hook(mk(b))
               for b, blk in enumerate(t.blocks)]
    t.enable_diagnostics(grid=None)
    torch.manual_seed(1234)
    try:
        for step in range(STEPS):
            if source == 'real':
                try:
                    p = next(src)
                except StopIteration:
                    break
            else:
                # 36 == the pooled grid size, matching what the real path feeds.
                p = torch.randn(1, 36, t.blocks[0].input_proj.in_features,
                                device=device)
            slots, cache = t(p, cache)
            if step == 0:
                continue
            acc['slot'].append(offdiag(slots[0].float().cpu()[:K]))

            # --- the CLASSIFIER's readout -------------------------------
            # ClsVJEPA pools the slots with an attention pool; the model records
            # its weights as `_last_pool_attn` (both `slot_pool` modes), so read
            # them rather than re-deriving the formula. Under `slot_pool='dot'`
            # this measured 31.95-31.99 of a maximum of 32 in every arm of the
            # scale sweep -- an unweighted mean -- which is what motivated the
            # `'attn'` mode. Under `'attn'` this is the check that the fix
            # actually made the pool selective.
            with torch.no_grad():
                # ClsVJEPA.pool_scores is the ONE implementation of this formula,
                # mode-aware. `_last_pool_attn` cannot be used here: measure()
                # drives the temporal model directly, so the HEAD's forward (which
                # sets it) never runs. Re-deriving the 'dot' form inline is what
                # crashed this on the attn arms, whose `slot_query` is None.
                _pa = head.pool_scores(slots).softmax(dim=-1).float()
                acc['pool_ent'].append(float(
                    -(_pa * (_pa + 1e-12).log()).sum(dim=-1).mean()))
                acc['pool_max'].append(float(_pa.max(dim=-1).values.mean()))

            # --- the SELF-ATTENTION's RECEIVED-MASS distribution ----------
            # `_diag_self` stores (entropy, read_mass) per block, where read_mass
            # = w.sum(dim=1) sums the head-AVERAGED attention over the QUERY axis:
            # how much total mass each slot receives from all the others.
            #
            # THIS IS NOT A SELECTIVITY MEASURE, and reading it as one produced a
            # wrong claim (2026-09-23: "the self-attention read is flat in all ten
            # arms, 31.8/32"). Summing over queries before normalising makes it
            # near-uniform WHENEVER different queries read DIFFERENT slots -- i.e.
            # it is near-uniform exactly when reading is selective and diverse.
            # It is the same flattening trap the cross-attention table warns about
            # for head-averaging ("an average of softmaxes is not a softmax").
            #
            # The correct measure is the per-QUERY, per-HEAD softmax over keys,
            # which needs the logits and so is not available from `_diag_self`
            # (which is head-averaged). Measured separately and reported by
            # `attn_entropy_table.py`'s self-attention section: dense s1 reads
            # at eff 31.92 of 32 -- uniform, because its slots are near-copies
            # (cos 0.995) -- while sparse s16 reads at eff 13.39 with slots at
            # cos 0.564. So the self-attention is NOT uniformly broken; it is
            # uniform exactly where the slots are copies, which is a consequence
            # of slot similarity rather than an init artefact.
            with torch.no_grad():
                _rs = [blk._diag_self[-1][1] for blk in t.blocks if blk._diag_self]
                if _rs:
                    _r = torch.stack([x.float() for x in _rs]).mean(dim=0)
                    _r = _r / (_r.sum(dim=-1, keepdim=True) + 1e-12)
                    acc['recv_ent'].append(float(
                        -(_r * (_r + 1e-12).log()).sum(dim=-1).mean()))
                    acc['recv_max'].append(float(_r.max(dim=-1).values.mean()))

            for b, blk in enumerate(t.blocks):
                if b in captured:
                    acc['attn'].append(offdiag(captured[b][0].float().cpu()))
                kv = cache.key_value_memory_dict.get(blk.mamba.layer_idx)
                if kv is not None:
                    ssm = kv[1].float().cpu()[:K]
                    acc['ssm_full'].append(offdiag(ssm))
                    acc['ssm_read'].append(offdiag(ssm.mean(dim=(1, 2))))
                if b < len(blk._diag_cross):
                    pass
            # per-slot attention entropy -> eff_patches
            ents = [blk._diag_cross[-1][0].mean().item()
                    for blk in t.blocks if blk._diag_cross]
            if ents:
                acc['eff'].append(math.exp(float(np.mean(ents))))
    finally:
        for h in handles:
            h.remove()
        t.disable_diagnostics()
    return {k: (float(np.mean(v)) if v else float('nan')) for k, v in acc.items()}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    sys.exit('needs CUDA')

print('=' * 104)
print('DIFFERENTIATION IN THE TRAINED LOGIT-SCALE ARMS')
print('=' * 104)
print('  REAL frames, 150 steps. Lower = more differentiated, for every column.')
print('  `pool_eff` is exp(entropy) over the 32 slots of the classifier\'s')
print('  pooling softmax, so 32.0 = uniform (no selection) and 1.0 = one slot.')
print('  `recv_eff` is the same on the self-attention RECEIVED-MASS')
print('  distribution. It is NOT a selectivity measure -- see the note above it.\n')
print(f"  {'arm':<10} {'attn_out':>9} {'eff/36':>7} {'ssm_full':>9} "
      f"{'ssm_read':>9} {'slot_full':>9} {'pool_eff':>9} {'recv_eff':>9} "
      f"{'pool_max':>9}")
print('  ' + '-' * 92)
real_res, rand_res = {}, {}
for label, rel, run, epoch in ARMS:
    head = build(rel, device)
    miss, unexp = load(head, run, device, epoch)
    head.eval()
    src = real_patches(head, device, max_videos=MAX_VIDEOS)
    real_res[label] = measure(head, device, 'real')
    r = real_res[label]
    print(f"  {label:<10} {r['attn']:>9.4f} {r['eff']:>7.2f} "
          f"{r['ssm_full']:>9.4f} {r['ssm_read']:>9.4f} {r['slot']:>9.4f} "
          f"{math.exp(r['pool_ent']):>9.2f} {math.exp(r['recv_ent']):>9.2f} "
          f"{r['pool_max']:>9.4f}")
    if RANDOM_PASS:
        rand_res[label] = measure(head, device, 'random')
    del head
    torch.cuda.empty_cache()

if not RANDOM_PASS:
    print("""
  THE RANDOM-PATCH TABLE IS OFF (RANDOM_PASS = False in this file). It compares
  findings 15.9.3's `torch.randn(1, 36, in_features)` table against real frames,
  on the six ARMS it was written for -- a settled question (16.5: random inputs
  invert the ordering between the STATE signals), and it would double the cost of
  a ten-arm sweep. Set RANDOM_PASS = True to re-run it.""")
else:

    print('\n' + '=' * 104)
    print('DOES THE RANDOM-PATCH TABLE LIE?  (findings 15.9.3 uses torch.randn)')
    print('=' * 104)
    print(f"  {'arm':<12} {'attn REAL':>10} {'attn RAND':>10} {'eff REAL':>9} "
          f"{'eff RAND':>9} {'ssm REAL':>9} {'ssm RAND':>9}")
    print('  ' + '-' * 72)
    for label, _rel, _run in ARMS:
        a, b = real_res[label], rand_res[label]
        print(f"  {label:<12} {a['attn']:>10.4f} {b['attn']:>10.4f} "
              f"{a['eff']:>9.2f} {b['eff']:>9.2f} {a['ssm_full']:>9.4f} "
              f"{b['ssm_full']:>9.4f}")

print('\n' + '=' * 104)
print('READING THIS')
print('=' * 104)
print('  attn_out_sim is the direct measure of "do slots receive different')
print('  inputs" -- the thing logit_scale is supposed to change.')
print('  ssm_full / ssm_read separate the two state collapses: the flattened')
print('  state vs the mean-over-(H,hdim) summary the gate readout consumes.')
print('  If the RAND columns differ materially from the REAL ones, then the')
print('  15.9.3 table -- which justifies the whole logit_scale sweep -- is')
print('  describing a different model from the one being trained.')