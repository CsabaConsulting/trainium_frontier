#!/usr/bin/env bash
# Fresh calibration for the ngram-ve-dim-256-on-top-of-locked-defaults spot-check
# (dim=256 stacked on the CURRENT locked config: layers "1,-1", table_mult 128, dim 128).
# 120-step fixed-step instrument, steady-state s/step from steps 100-110, same convention as
# t17_l1_width_probe.sh. Locked defaults now need NO extra flags (compile-muon, qk-shift-freeze,
# ngram-ve-layers 1,-1, table-mult 128 are all module-level defaults as of 2026-09-14).
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
run ngd256-ctrl
run ngd256-arm --ngram-ve-dim 256
echo "=== done ==="
