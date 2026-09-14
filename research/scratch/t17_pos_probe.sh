#!/usr/bin/env bash
# Layer 1 turned out to be worth -0.0118 bpb and layer 3 is negative, which means POSITION is the
# dominant variable and only two of six positions have ever been tried at 4 ranks. 5 minutes per
# position on the 120-step instrument is cheap insurance against having picked the wrong one: the
# proxy has now called both known outcomes in the right direction and roughly the right size
# (ctrl 3.9608 -> l1 3.9402 = -0.0206 proxy for a real -0.0118; l3 3.9973 = positive for a real loss).
#
# Reference points, same instrument, both throughput flags on every arm:
#   layers "-1"      (locked)   0.7389 s/step   proxy 3.9608
#   layers "1,-1"               0.7848 s/step   proxy 3.9402   <- current best, real -0.0118 bpb
#   layers "3,-1"               0.7620 s/step   proxy 3.9973
#   layers "1,3,-1"             0.8122 s/step   proxy 3.9855
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
BUDGET=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze)
run() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/probe-$name" \
    --eval-seq-len 1024 "${BUDGET[@]}" "$@" > "research/run-logs/probe-$name.log" 2>&1
  grep -E "num_params|^step 001[01]0 " "research/run-logs/probe-$name.log" | tail -4
}
run pos-l0    --ngram-ve-layers "0,-1"
run pos-l2    --ngram-ve-layers "2,-1"
run pos-l12   --ngram-ve-layers "1,2,-1"
run pos-l0l1  --ngram-ve-layers "0,1,-1"
echo "=== done ==="
