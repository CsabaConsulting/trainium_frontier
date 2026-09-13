#!/usr/bin/env bash
# Cheap fixed-step throughput probe for the block-body compiler-scheduling anomaly.
#
# Three arms, 120 steps each (lrm stays 1.000 for the whole probe because the schedule is driven by
# elapsed/max_train_seconds=1800, so every step is a steady-state step):
#
#   base   -- locked defaults
#   frozen -- --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze : a MATHEMATICAL IDENTITY (verified
#             bit-identical over 5 optimizer steps in research/scratch/qk_freeze_probe.py) whose
#             elementwise ops nevertheless stay in the compiled block graph
#   trained -- --qk-shift --qk-shift-beta 0.0 : the arm the 1800s run measured at +8.8% steps and
#             +0.0016 bpb, here as a positive control that the probe can see the speedup at all
#
# Read the steady-state `dt` from each log (they are flat from about step 5 onward).
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
run() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/probe-$name" \
    --eval-seq-len 1024 "$@" > "research/run-logs/probe-$name.log" 2>&1
  grep -E "^step 00(0[1-9]0|1[01]0) " "research/run-logs/probe-$name.log" | tail -12
}
run qkfreeze-base
run qkfreeze-frozen --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze
run qkfreeze-trained --qk-shift --qk-shift-beta 0.0
echo "=== done ==="
