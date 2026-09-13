#!/usr/bin/env bash
# Paired 1800s A/B for the frozen Q/K-shift throughput probe. Same commit, same day, back to back.
#
# The frozen arm (--qk-shift --qk-shift-beta 0.0 --qk-shift-freeze) is a MATHEMATICAL no-op:
# beta is fixed at exactly 0.0 and never enters an optimizer group, so `x + beta*causal_shift_1(x)`
# equals x bit-exactly at every step, and the backward contribution through the shifted branch is
# scaled by the same 0.0. Both arms therefore have identical optimizer groups and identical
# mathematics; the only difference is that the frozen arm's compiled block body carries three
# extra elementwise ops. The 120-step probe measured 0.84 s/step (base) vs 0.76 s/step (frozen),
# a +10.6% throughput gain, which by the -0.0705 bpb/ln(step) slope should be worth ~-0.007 bpb at
# the full budget IF the per-step quality really is unchanged.
set -u
cd "$(dirname "$0")/../.." || exit 1
run() {
  local label="$1"; local desc="$2"; shift 2
  echo "=== $(date -u +%H:%M:%S) $label : $* ==="
  python research/run_experiment.py --label "$label" --solution T1.x \
    --description "$desc" --num-steps 5000 --max-train-seconds 1800 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}
run qkfreeze-base-1800s "fresh same-commit 1800s control for the frozen qk-shift probe"
run qkfreeze-frozen-1800s \
  "frozen qk_shift_beta==0 (requires_grad=False, in no optimizer group): a mathematical no-op whose elementwise ops stay in the compiled block body" \
  --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze
echo "=== $(date -u +%H:%M:%S) done ==="
