#!/usr/bin/env bash
# Track: throughput->capacity, Probe 2 -- "launches per token".
#
# Hypothesis (track doc section 4): launch COUNT per token, not FLOPs, sets the step time, so a
# fatter micro-step (fewer, larger launches at equal total FLOPs) cuts step time. Our step is
# grad_accum=4 micro-steps of DEVICE_BATCH_SIZE*PACK_FACTOR = 1*4 = 4 rows x 2048 = 8192 tokens
# per rank. --pack-factor 8 -> 2 micro-steps of 16384; --pack-factor 16 -> 1 micro-step of 32768.
# TOTAL_BATCH_SIZE=131072 is held fixed by train.py, so total tokens/step is identical in all arms
# and this is a pure launch-count-vs-per-launch-work trade.
#
# TWO PRIORS AGAINST, both already in the notebook, which is why this is a cheap screen and not a
# day of work:
#   (1) T1.5's synthetic fwd+bwd microbenchmark on the real 4-rank/LNC=2 compiled shape measured
#       us/token by micro-batch ROW COUNT as 38.325 (1) / 22.322 (2) / 21.673 (4) / 23.800 (8),
#       i.e. 4 rows is the per-token optimum and 8 is already ~10% PAST it.
#   (2) Probe 1's native decomposition at the locked shape: 0.2544 s at ga=1 vs 0.7952 s at ga=4
#       => one micro-step 0.1803 s, collective+optimizer 0.0741 s = 9.3% of the step. The ENTIRE
#       prize for collapsing 4 micro-steps into 1 is bounded by that 9.3% minus whatever the fat
#       micro-step itself costs -- and prior (1) says the fat micro-step costs MORE per token.
# So the honest expectation is a small negative. Measured anyway because both priors are synthetic
# or indirect and the doc asks for the real number.
#
# Same 120-step fixed-step instrument as t17_throughput_probe.sh (validated to reproduce the 1800s
# steady-state s/step to three digits; lrm stays 1.000 because the schedule is driven by
# elapsed/max_train_seconds=1800). Read the steady-state `dt`, flat from ~step 5.
#
# Arms are run at the LOCKED model AND at the current best throughput stack (the other box's
# +13.67% budget flags), because PACK_FACTOR interacts with the block body (qk-shift-freeze) and
# with the optimizer window (compile-muon) in opposite directions: a fatter micro-step spends MORE
# in the block body and amortizes the optimizer window over fewer accumulations.
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
BUDGET=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze)
run() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/tc-probe-$name" \
    --eval-seq-len 1024 "$@" > "research/run-logs/tc-probe-$name.log" 2>&1
  grep -E "micro_tokens|num_params" "research/run-logs/tc-probe-$name.log" | head -3
  grep -E "^step 00(0[1-9]0|1[01]0) " "research/run-logs/tc-probe-$name.log" | tail -12
  grep -iE "out of memory|OOM|nrt_tensor_allocate|Error" "research/run-logs/tc-probe-$name.log" | head -5
}
# --- locked model, same-day control + the two fatter micro-steps ---
run pf4-locked
run pf8-locked  --pack-factor 8
run pf16-locked --pack-factor 16
# --- current best throughput stack ---
run pf4-budget  "${BUDGET[@]}"
run pf8-budget  "${BUDGET[@]}" --pack-factor 8
run pf16-budget "${BUDGET[@]}" --pack-factor 16
echo "=== done ==="
