#!/usr/bin/env bash
# Follow-up to t17_spend_probe.sh. Layer 1 turned out to be the big win (1.007376 @2139 steps vs the
# budget control's 1.019126 @2141, an exact step match, -0.0118 bpb), and it only costs 5.86% of the
# 13.67% budget. So there is budget LEFT OVER. This probe prices stacking the two spends that both
# worked -- an early n-gram layer AND a wider n-gram table -- so the follow-up 1800s arms can be
# step-matched at ~2180 steps (just under the ~2195 data wall) instead of guessed.
#
# Reference points from t17_spend_probe.sh (same instrument, both throughput flags on every arm):
#   budget control   0.7389 s/step   177394 tok/s
#   layers 1,-1      0.7848 s/step   167008 tok/s   (-5.86%)
#   dim 256          0.7674 s/step   170803 tok/s   (-3.72%)
#   mult 128         0.7667 s/step   170946 tok/s   (-3.64%)
#   dim256+mult128   0.8349 s/step   156990 tok/s   (-11.50%)
# Whether layer-1 and width costs compose additively is the thing being measured, not assumed: the
# layer-1 cost is a cross-pcore resharding effect on the residual stream and the width cost is table
# gather+matmul work, so they could easily interact in either direction.
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
BUDGET=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze --ngram-ve-layers "1,-1")
run() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/probe-$name" \
    --eval-seq-len 1024 "${BUDGET[@]}" "$@" > "research/run-logs/probe-$name.log" 2>&1
  grep -E "num_params|^step 00(0[1-9]0|1[01]0) " "research/run-logs/probe-$name.log" | tail -13
}
run l1w-dim256   --ngram-ve-dim 256
run l1w-m128     --ngram-ve-table-mult 128
run l1w-dim256-m128 --ngram-ve-dim 256 --ngram-ve-table-mult 128
run l1w-l13      --ngram-ve-layers "1,3,-1"
echo "=== done ==="
