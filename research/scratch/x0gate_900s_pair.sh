#!/usr/bin/env bash
# Item 2: paired 900s A/B for Recursive's input-dependent x0 gate. Same commit, back to back.
#
#   x = resid_lambda*x + x0_lambda*(2*sigmoid(s * mean(x)) * x0)
#
# One learned scalar s per layer, zero-initialized, so 2*sigmoid(0) == 1.0 and the arm is
# bit-identical to the baseline at step 0 (verified over 5 optimizer steps in fp32 in
# research/scratch/qk_freeze_probe.py, which also checks causality_max_abs == 0.0: mean() is over
# each token's own 640 channels, never across positions).
#
# The control is FRESH, not a cross-day row: a 900s number is only comparable within a pair.
# Escalate to 1800s only if the delta is <= -0.002, and decompose it first -- the block-body
# scheduling anomaly means ANY block-body edit can collect +8-10% throughput for free, so a 900s
# "win" here has to be split into (steps) + (per-step residual) using -0.0705 bpb/ln(step) before
# it means anything. A win that is all throughput is not an x0-gate result at all.
set -u
cd "$(dirname "$0")/../.." || exit 1
run() {
  local label="$1"; local desc="$2"; shift 2
  echo "=== $(date -u +%H:%M:%S) $label : $* ==="
  python research/run_experiment.py --label "$label" --solution R1.2 \
    --description "$desc" --num-steps 5000 --max-train-seconds 900 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}
run x0gate-base-900s "fresh same-commit 900s control for the input-dependent x0 gate"
run x0gate-900s \
  "Recursive input-dependent x0 gate: x0 skip scaled by 2*sigmoid(s*mean(x)), one zero-init scalar per layer, bit-identical at step 0" \
  --x0-gate
echo "=== $(date -u +%H:%M:%S) done ==="
