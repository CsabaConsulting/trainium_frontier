#!/usr/bin/env bash
# Items 3+4: what does the +13.7% combined budget (T1.7 compiled Muon + frozen qk-shift) actually
# BUY? Same 120-step fixed-step instrument as t17_throughput_probe.sh, and every arm carries both
# throughput flags, so the reference is the quadrant's t17-compilemuon-frozen (0.74 s/step,
# ~177.5k tok/s) and every delta below is the marginal cost of the capacity being added.
#
# Item 3 -- put n-gram VE tables back on an EARLY layer. The prior is bad and specific: T3.1
# measured the cost of an early injection as a whole-graph LNC=2 resharding penalty
# (2.768 -> 3.711 s/step for {5} -> {1,5}, +34%), invariant to table geometry. If that still holds
# the 13.7% budget cannot pay for it and the item closes negative. Measured, not assumed.
#
# Item 4 -- buy WIDTH on the table that already works (the T3.2 lesson). Both axes were rejected in
# the 2026-09-12 sweep on THROUGHPUT grounds while being better PER STEP once normalised by the
# -0.077/ln(steps) slope: mult 128 -> 1.016534 equivalent-at-2139, dim 256 -> 1.015864, against the
# locked mult 64 / dim 128's 1.018985. The budget is exactly what those two lost on.
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
  grep -E "num_params|^step 00(0[1-9]0|1[01]0) " "research/run-logs/probe-$name.log" | tail -13
}
run spend-l1     --ngram-ve-layers "1,-1"
run spend-l3     --ngram-ve-layers "3,-1"
run spend-dim256 --ngram-ve-dim 256
run spend-m128   --ngram-ve-table-mult 128
run spend-dim256-m128 --ngram-ve-dim 256 --ngram-ve-table-mult 128
echo "=== done ==="
