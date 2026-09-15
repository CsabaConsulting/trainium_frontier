#!/usr/bin/env bash
# Calibration for the ngram-ve-table-mult-256-on-top-of-locked-defaults spot-check. Isolates the
# TABLE WIDTH axis alone (--ngram-ve-table-mult 256, dim UNCHANGED at 128) at the CURRENT
# two-injection-site config (NGRAM_VE_LAYERS="1,-1", locked mult=128 baseline). This is the OTHER
# half of the 2-variable grid a concurrent box1 agent is probing with --ngram-ve-dim 256 (mult
# unchanged at 128) -- together the two isolate whether any gain from the historical
# "dim 256 + mult 128" precedent (pre-layer-1, single injection site) comes from table WIDTH, from
# embedding DIM, or needs both, now that there are two injection sites reading the tables instead
# of one.
#
# 120-step fixed-step instrument, steady-state s/step + step-110 train loss from steps 100-110,
# same convention as t17_l1_width_probe.sh / t17_spend_probe.sh. Locked defaults need NO extra
# flags: compile-muon, qk-shift-freeze, ngram-ve-layers "1,-1", table-mult 128 are all
# module-level defaults as of 2026-09-14 (train.py commit df82d08 here).
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
run ngm256-ctrl
run ngm256-arm --ngram-ve-table-mult 256
echo "=== done ==="
