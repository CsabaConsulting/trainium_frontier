#!/usr/bin/env bash
# Paired 900s spot-check for --ngram-ve-opt (train.py:4984, choices "rmsprop"/"adamw"). Locked
# default is "rmsprop" (train.py:703, "Recursive's choice": no first moment, betas=(0.0,
# NGRAM_VE_BETA2=0.999) -- see the train.py:3240-3266 comment for the full rationale). The
# "adamw" arm is "the fallback arm: identical group, but the same (0.8, EMBED_ADAMW_BETA2=0.95)
# Adam the unigram value-embedding tables use" (same comment). Never A/B'd on this stack at the
# current locked shape (592.9M total params, NGRAM_VE_LAYERS="1,-1", NGRAM_VE_TABLE_MULT=128).
#
# UNLIKE ngram-ve-lr/beta2 (pure scalar, no compute/tensor change), this axis swaps the update
# rule (adds a first-moment buffer + extra FMA per step for the 537M-param table group), and the
# 120-step calibration (research/scratch/ngramveopt_calib.sh, research/run-logs/probe-ngmveopt-
# {ctrl,arm}.log) found a REAL throughput cost: steady-state (steps 60-110) mean tok/s
# ctrl(rmsprop)=155,258 vs arm(adamw)=152,693, i.e. dt_arm/dt_ctrl = 1.0168 (+1.68% s/step,
# -1.65% tok/s). Small next to the tensor-shape-changing axes (dim256/mult256's ~19-23%) but real
# and worth step-matching rather than assuming away, per the task brief's explicit warning.
#
# Step-matching: control flat at 900s; arm's --max-train-seconds pre-scaled to
# 900 * 1.0168 = 915s so both land at (approximately) the same step count "by construction" --
# same convention as ngrammult256_900s.sh. Bracketed by two fresh same-commit controls (first and
# last), one 4-rank job owning the box, serial. Throughput verified rather than assumed: report
# actual training_seconds/steps/dt for all three runs, not just the calibration's prediction.
#
# Noise floor per task brief: |delta| < 0.001 bpb is noise; < 0.003 needs a same-day paired repeat
# before trusting; a real win >= noise floor earns a confirming 1800s pair before adoption, but
# ONLY if it clears the break-even bar computed from the measured throughput cost.
set -u
cd "$(dirname "$0")/../.." || exit 1
run() {
  local label="$1"; local desc="$2"; local secs="$3"; shift 3
  echo "=== $(date -u +%H:%M:%S) $label (max-train-seconds=$secs): $* ==="
  python research/run_experiment.py --label "$label" --solution T3.1 \
    --description "$desc" --num-steps 5000 --max-train-seconds "$secs" \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}

run ngmveopt-ctrl-900s-r1 \
  "fresh same-commit 900s control for the --ngram-ve-opt isolation A/B, flag at locked default (rmsprop, no first moment, betas=(0.0,0.999))" \
  900 --ngram-ve-opt rmsprop

run ngmveopt-arm-900s \
  "--ngram-ve-opt adamw (Recursive's stated fallback arm: identical group, betas=(0.8,EMBED_ADAMW_BETA2=0.95) like the unigram VE tables; max-train-seconds pre-scaled 900->915s per the calibration's measured +1.68% s/step cost to land at matched steps)" \
  915 --ngram-ve-opt adamw

run ngmveopt-ctrl-900s-r2 \
  "closing same-commit 900s control: measures this block's own drift/noise band" \
  900 --ngram-ve-opt rmsprop
echo "=== $(date -u +%H:%M:%S) done ==="
