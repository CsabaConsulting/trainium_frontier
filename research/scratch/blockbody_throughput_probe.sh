#!/usr/bin/env bash
# Fixed-step throughput probe for every block-body variant, against one shared control.
#
# 120 steps each with --max-train-seconds 1800, so lr_multiplier stays 1.000 for the whole probe
# (the schedule is driven by elapsed/max_train_seconds) and every step is a steady-state step with
# no cooldown recompile. Read the steady-state `dt` and `tok/s` from each log; they are flat from
# about step 5 onward. The `base` arm is the calibration: it must land on 0.83-0.84 s/step, which is
# what both probe-qkfreeze-base and the t39-base-1800s control measured.
#
# Arms, in the order they were found:
#
#   base           locked defaults.
#   qkfrozen       --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze. A verified mathematical
#                  identity (bit-identical over 5 optimizer steps in qk_freeze_probe.py) that
#                  measured 0.76 s/step, +10.6%. Re-run here so every arm shares one control.
#   nudge-qk       the MINIMAL identity: `q_in = k_in = x + nudge*x` with nudge frozen at 0.
#   nudge-resid    the same nudge on the residual stream instead, so the MLP sees it too.
#   normcse        --qkv-norm-cse: three identical norm() calls collapsed into one. Also an
#                  identity, but it REMOVES ops rather than adding them -- the opposite direction,
#                  which is what makes it the sharpest test of what the anomaly actually is.
#   parallel       --parallel-block: real architecture change (one norm, two independent branches).
#   par-vfc        --parallel-block --parallel-block-coalesce vfc (3 matmuls, no copies).
#   par-all        --parallel-block --parallel-block-coalesce all (1 matmul, 2 q/k copies).
#
# Only the first five are quality-neutral. `parallel*` change the mathematics from step 1 on, so
# their dt is only half the story and needs a paired A/B to price.
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
run() {
  local name="$1"; shift
  echo "=== $(date -u +%H:%M:%S) $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/probe-$name" \
    --eval-seq-len 1024 "$@" > "research/run-logs/probe-$name.log" 2>&1
  grep -E "^step 000(1|4|8)0 |^step 00110 " "research/run-logs/probe-$name.log"
}
run bb-base
run bb-qkfrozen --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze
run bb-nudge-qk --block-nudge --block-nudge-site qk
run bb-nudge-resid --block-nudge --block-nudge-site resid
run bb-normcse --qkv-norm-cse
run bb-parallel --parallel-block
run bb-par-vfc --parallel-block --parallel-block-coalesce vfc
run bb-par-all --parallel-block --parallel-block-coalesce all
echo "=== $(date -u +%H:%M:%S) done ==="
