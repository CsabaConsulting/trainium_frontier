#!/usr/bin/env bash
# T1.7 sizing probe: how much step time does the COMPILED Muon body actually buy at the
# locked config, and does it stack with the frozen qk-shift?
#
# A QUADRANT, not a pair, because the two changes are supposed to touch different parts of
# the step (optimizer window vs block-body compute) and "roughly additive" is a claim to be
# measured, not assumed -- the qk-shift-freeze/Engram interaction turned out SUPERadditive.
#
#   base        locked defaults (--nki-muon, the shipped Muon path)     the reference s/step
#   t17         --compile-muon  (mutually exclusive with --nki-muon by construction)
#   frozen      --qk-shift --qk-shift-beta 0 --qk-shift-freeze          re-confirm the budget
#   t17-frozen  both                                                    does it stack?
#
# Same 120-step fixed-step instrument the Engram probe used (validated to reproduce the
# 1800s steady-state s/step to three digits; lrm stays 1.000 throughout because the schedule
# is driven by elapsed/max_train_seconds=1800). Read the steady-state `dt`, flat from ~step 5.
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
FROZEN=(--qk-shift --qk-shift-beta 0.0 --qk-shift-freeze)
run() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/probe-$name" \
    --eval-seq-len 1024 "$@" > "research/run-logs/probe-$name.log" 2>&1
  grep -E "muon step: COMPILED" "research/run-logs/probe-$name.log" | head -6
  grep -E "^step 00(0[1-9]0|1[01]0) " "research/run-logs/probe-$name.log" | tail -12
}
run t17-base
run t17-compilemuon --compile-muon
run t17-frozen-only "${FROZEN[@]}"
run t17-compilemuon-frozen --compile-muon "${FROZEN[@]}"
echo "=== done ==="
