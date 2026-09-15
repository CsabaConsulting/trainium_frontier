#!/usr/bin/env bash
# Fresh 120-step calibration for the depth=8-at-FIXED-width-640 spot-check.
# This is DECOUPLED from the earlier CONFOUNDED "768x8" result (--depth 8 alone bumps n_embd
# to round_up(8*96,128)=768 via ASPECT_RATIO -- see trn2-experiment-results.md Part A correction).
# Here the arm passes --depth 8 --n-embd 640 explicitly, so width is held fixed while depth
# goes 6->8 (+2 layers), orthogonal to every width point already tested.
# Same 120-step steady-state instrument convention as ngramdim256_calib.sh / tc_probe3f_width1024.sh.
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
run d8w640-ctrl
run d8w640-arm --depth 8 --n-embd 640
echo "=== done ==="
