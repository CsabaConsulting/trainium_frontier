#!/usr/bin/env bash
# Same-day confirming repeat for --ngram-ve-opt (see ngramveopt_900s.sh for full context/rationale).
# First pair: ctrl(rmsprop) mean val_bpb@2048=1.052890 (r1=1.052999, r2=1.052781, 1052/1054 steps)
# vs arm(adamw)=1.055048 (1053 steps) -- raw delta +0.002158 @2048 / +0.002089 @1024, i.e. adamw
# WORSE, matched steps. Control's own r1-vs-r2 spread was only 0.000218 (@2048) -- ~10x smaller
# than the arm delta -- but per the task's noise-floor rule (0.001-0.003 needs a same-day repeat
# before trusting) this must be reproduced before closing. Throughput cost also confirmed twice
# already (calibration +1.68% s/step, first spot-check +1.71% s/step, real training_seconds/steps)
# -- arm kept at max-train-seconds=915 (same scale as the first pair) since the two independent
# throughput measurements agree to within 0.5 points.
set -u
cd "$(dirname "$0")/../.." || exit 1
run() {
  local label="$1"; local desc="$2"; local secs="$3"; shift 3
  echo "=== $(date -u +%H:%M:%S) $label (max-train-seconds=$secs): $* ==="
  python research/run_experiment.py --label "$label" --solution T3.1 \
    --description "$desc" --num-steps 5000 --max-train-seconds "$secs" \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}

run ngmveopt-ctrl-900s-r3 \
  "same-day confirming-repeat control for --ngram-ve-opt, locked default (rmsprop)" \
  900 --ngram-ve-opt rmsprop

run ngmveopt-arm-900s-r2 \
  "same-day confirming-repeat arm for --ngram-ve-opt adamw, max-train-seconds pre-scaled 900->915s as in the first pair" \
  915 --ngram-ve-opt adamw

run ngmveopt-ctrl-900s-r4 \
  "closing confirming-repeat control: measures this block's own drift/noise band" \
  900 --ngram-ve-opt rmsprop
echo "=== $(date -u +%H:%M:%S) done ==="
