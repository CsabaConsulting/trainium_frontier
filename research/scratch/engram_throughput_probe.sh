#!/usr/bin/env bash
# T3.2 sizing probe: does Engram-lite's ADDED compute fit inside the frozen qk-shift's freed budget?
#
# The pivot's whole arithmetic is "spend the free ~9.2%, do not go negative". This probe measures
# both sides of that trade at the same commit, in the same session, on the same box, using the
# 120-step fixed-step instrument (validated to reproduce 1800s steady-state s/step to three digits;
# lrm stays 1.000 throughout because the schedule is driven by elapsed/max_train_seconds=1800).
#
# A QUADRANT, not a pair, because the two changes could interact -- both touch block-level compute:
#
#   base           locked defaults                              (the reference s/step)
#   frozen         + --qk-shift --qk-shift-beta 0 --qk-shift-freeze : re-confirms the +9-10% budget
#                  AT the locked config, i.e. with the n-gram VE tables also on (item 1)
#   engram         + --engram                                    : the cost being bought
#   engram-frozen  + both                                        : whether the budget actually pays
#                  for it, and whether the two effects compose additively
#
# Read the steady-state `dt` from each log (flat from about step 5 onward). The trade works out iff
# dt(engram-frozen) <= dt(base): same wall-clock per step, strictly more capacity in it.
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
  grep -E "^step 00(0[1-9]0|1[01]0) " "research/run-logs/probe-$name.log" | tail -12
}
run engram-base
run engram-frozen-only "${FROZEN[@]}"
run engram-own --engram
run engram-own-frozen --engram "${FROZEN[@]}"
# Share mode costs no gather and no parameters, so it isolates how much of whatever the own-table
# arm costs is the GATHER as opposed to the gate/conv/residual-write mechanism.
run engram-share --engram --engram-share-ngram-tables
echo "=== done ==="
