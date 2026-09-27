#!/usr/bin/env bash
# §25.11 accounting on the FULL test set (all 1402 videos).
#
# Why full test: the costs were measured at 100 videos, and so was the gap they
# were divided by (0.0155). But the matched-checkpoint FULL-TEST gap is 0.0102,
# so the subset inflates the denominator by 52% -- which is what makes the
# "share of the gap" swing between 41% and 63%. Both quantities must come off
# the same sample.
#
# Why `none` is included, when it looks redundant:
#   1. The paired test (bootstrap CI + Wilcoxon on per-video AUC) needs BOTH arms
#      on the same 1402 videos. `analyze_topk_sweep.py` does not save per-video
#      AUCs, so its full-test numbers cannot be paired against -- only compared
#      as levels.
#   2. It is the cross-harness check. `none` must reproduce the sweep's sparse
#      ep20 full-test AUC of 0.8572. If it does not, the two harnesses disagree
#      at full test and the intervention deltas are not readable.
#
# Colab-safe: no conda activate, no WSL paths. Run from the repo root.
# Edit ENC below (or the config's checkpoint_path) to wherever the V-JEPA 2.1
# encoder checkpoint lives in your Colab mount.
set -u

CFG=cfgs/vjepa_sparse_slotssm_VCL_64_NF_4_finetuned.yaml
CKPT=output/vcl-64/vjepa_sparse_slotssm_VCL_64_NF_4_finetuned/checkpoints/model-20.pt
DATA=/content/DoTA_dataset
OUT=output/vcl-64

echo "--- preflight ---"
[ -f "$CKPT" ] || { echo "MISSING checkpoint: $CKPT"; exit 1; }
[ -d "$DATA" ] || { echo "MISSING data dir: $DATA"; exit 1; }
grep -n '^checkpoint_path' "$CFG"
python -c "import yaml,sys; p=yaml.safe_load(open('$CFG'))['checkpoint_path']; sys.exit(0 if __import__('os').path.exists(p) else 1)" \
  || { echo "checkpoint_path in $CFG does not exist -- repoint it before running"; exit 1; }
echo "--- preflight OK ---"

# Order matters: dense_both is the headline numerator (the both-preserved-destroyed
# cost); none is the baseline it is paired against; dense_ssm gives the split
# between the row half and the recurrent half. Stop after any mode if you run out
# of time -- `none` + `dense_both` alone already give the headline number.
for mode in none dense_both dense_ssm; do
  echo "=== $(date +%H:%M:%S)  content=$mode  FULL TEST (all videos)"
  python tests/diag_topk_mechanism.py \
    --config "$CFG" --checkpoint "$CKPT" \
    --max_videos 0 --mode learned --content "$mode" --data_path "$DATA" \
    --json_out "$OUT/_content_${mode}_fulltest.json" \
    > "$OUT/_content_${mode}_fulltest.log" 2>&1
  echo "    exit=$?  $(date +%H:%M:%S)"
  grep -E "^  (pooled AUC|mean video AUC)" "$OUT/_content_${mode}_fulltest.log" || true
done
echo "FULLTEST CONTENT DONE $(date +%H:%M:%S)"
