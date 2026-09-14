#!/usr/bin/env bash
# Track: throughput->capacity, Probe 3 step 4. One queue so the box never sits idle.
#
# (a) the missing screen arm. `--depth 8` alone does NOT give 640x8: n_embd defaults to
#     round_up(DEPTH*ASPECT_RATIO=96, HEAD_DIM=128), which is 640 at depth 6 and 768 at depth 8. So
#     the first screen's "640x8" arm was silently a duplicate of its "768x8" arm (identical
#     num_params=631,655,952, identical flops_per_token, 1.9195 vs 1.9247 s/step). That accident is
#     useful -- it is a same-shape repeat, and it puts this instrument's s/step reproducibility at
#     0.27% on the mean, comfortably under the 0.3% the notebook quotes. The real 640x8 needs
#     --depth 8 --n-embd 640 and has never been measured.
# (b) the payoff runs: fixed ~2135 steps, budget = 2135 * that arm's own measured s/step.
#     Control runs at the official 1800 s instead, because at 0.84565 s/step that IS 2128-2135 steps
#     and a legal-budget control is worth more than the <=2-step (<=0.00007 bpb) matching error.
set -u
cd "$(dirname "$0")/../.." || exit 1
BEST=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze
      --ngram-ve-layers "1,-1" --ngram-ve-table-mult 128)

screen() {
  local name="$1"; shift
  echo "=== SCREEN $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps 120 --max-train-seconds 1800 --out-dir "out/tc-probe-$name" \
    --eval-seq-len 1024 "${BEST[@]}" "$@" > "research/run-logs/tc-probe-$name.log" 2>&1
  grep -E "num_params=" "research/run-logs/tc-probe-$name.log" | head -1
  grep -E "^step 000[4-9]0 |^step 001[01]0 " "research/run-logs/tc-probe-$name.log" | tail -8
}

full() {
  local name="$1"; local secs="$2"; shift 2
  echo "=== FULL $name : budget=${secs}s : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --max-train-seconds "$secs" --out-dir "out/tc-$name" \
    "${BEST[@]}" "$@" > "research/run-logs/tc-$name.log" 2>&1
  grep -E "num_params=|public_val_bpb|causality|train_seconds|steps" "research/run-logs/tc-$name.log" | tail -12
}

screen sh-640x8-real --depth 8 --n-embd 640
full sh-640x6-2135 1800
full sh-768x6-2135 2115 --n-embd 768
full sh-896x6-2135 2493 --n-embd 896
echo "=== done ==="
