#!/usr/bin/env bash
# Items 1-4, the real full-budget arms. Every arm is STEP-MATCHED at ~2150 steps by shortening its
# own --max-train-seconds to (2150 x its measured s/step from research/scratch/t17_spend_probe.sh).
#
# Why step-match instead of running all four for 1800 s: because of the data wall. At the locked
# config 1800 s buys 2139 of the ~2195 available steps, so the +13.67% combined budget (T1.7 +
# frozen qk-shift) would push a 1800 s run to ~2432 steps -- 237 steps of REPEATED corpus, which is
# measured NEGATIVE (+0.0023 bpb for a 9.2% speedup). Shortening the budget so the wall-clock LR
# schedule completes right at ~2150 steps is the "decline to use the speedup" move that scored
# 1.018967 for qkfreeze-frozen-1648s, and it makes every arm below a pure PER-STEP comparison
# against every other and against the locked 1.018985 / 2139 reference. No decomposition needed
# afterwards: the step counts are matched by construction.
#
#   ctrl          budget only                     190.1M   1589 s   is T1.7's non-bit-exact
#                                                                   Muon step neutral at 2150 steps?
#   dim256-m128   budget + dim 256 + mult 128     592.9M   1795 s   the whole budget spent as WIDTH
#   l1            budget + layers "1,-1"          324.5M   1687 s   item 3: an early n-gram layer
#   dim256        budget + dim 256                324.5M   1650 s   single-axis width control
set -u
cd "$(dirname "$0")/../.." || exit 1
BUDGET=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze)
run() {
  local label="$1"; local secs="$2"; local desc="$3"; shift 3
  echo "=== $(date -u +%H:%M:%S) $label (${secs}s) : $* ==="
  python research/run_experiment.py --label "$label" --solution T1.7 \
    --description "$desc" --num-steps 5000 --max-train-seconds "$secs" \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 \
    "${BUDGET[@]}" "$@"
}
run t17-ctrl-2150 1589 \
  "T1.7 compiled Muon step + frozen qk-shift, budget shortened to land at ~2150 steps (step-matched control)"
run t17-dim256-m128-2150 1795 \
  "T1.7 + frozen qk-shift budget spent entirely on n-gram VE WIDTH: dim 256 x mult 128, 592.9M params, ~2150 steps" \
  --ngram-ve-dim 256 --ngram-ve-table-mult 128
run t17-l1-2150 1687 \
  "T1.7 + frozen qk-shift budget spent on re-enabling n-gram VE on layer 1 (NGRAM_VE_LAYERS=1,-1), 324.5M params, ~2150 steps" \
  --ngram-ve-layers "1,-1"
run t17-dim256-2150 1650 \
  "T1.7 + frozen qk-shift budget spent on n-gram VE dim 256 only, 324.5M params, ~2150 steps" \
  --ngram-ve-dim 256
echo "=== $(date -u +%H:%M:%S) done ==="
