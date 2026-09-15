#!/usr/bin/env bash
# Calibration for --ngram-ve-opt (train.py:4984, choices "rmsprop"/"adamw", locked default
# "rmsprop", train.py:703). Isolates the OPTIMIZER axis on the n-gram value-embedding tables'
# own param group (T3.1, train.py:3240-3266): control is the locked "rmsprop" (no first moment,
# betas=(0.0, NGRAM_VE_BETA2=0.999)); arm is "adamw" ("the fallback arm: identical group, but the
# same (0.8, EMBED_ADAMW_BETA2=0.95) Adam the unigram value-embedding tables use", per the
# train.py comment at line ~3255). This has never been A/B'd on this stack at the current locked
# shape (592.9M total params, NGRAM_VE_LAYERS="1,-1", NGRAM_VE_TABLE_MULT=128) -- Recursive's
# stated rationale for rmsprop (a hashed n-gram row is touched a fraction of a step on average;
# Adam's m/sqrt(v) damps a rare update by exactly the frequency factor it divides out of v) has
# never been empirically checked here.
#
# UNLIKE the ngram-ve-lr/beta2 axes (pure scalar hyperparameter, no tensor/compute change),
# switching optimizer kind changes the update rule and adds a first-moment buffer (same size as
# the 537M-param table) plus an extra elementwise FMA per step for that param group -- a real
# candidate for a throughput difference on a memory/compute-bound sparse-table update. This
# calibration measures that BEFORE assuming anything, per the task brief's explicit warning not
# to assume throughput parity here the way it held for the scalar lr/beta2 sweeps.
#
# 120-step fixed-step instrument, steady-state s/step + step-110 train loss from steps 100-110,
# same convention as ngrammult256_calib.sh / t17_l1_width_probe.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
run() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/probe-$name" \
    --eval-seq-len 1024 "$@" > "research/run-logs/probe-$name.log" 2>&1
  grep -E "num_params|^step 00(0[1-9]0|1[01]0) " "research/run-logs/probe-$name.log" | tail -13
}
run ngmveopt-ctrl --ngram-ve-opt rmsprop
run ngmveopt-arm --ngram-ve-opt adamw
echo "=== done ==="
