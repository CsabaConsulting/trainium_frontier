#!/usr/bin/env bash
# Track: throughput->capacity, Probe 3 step 4 -- PART B: the payoff experiment.
#
# Sweep WIDTH at a FIXED ~2135 steps. Each arm's --max-train-seconds is 2135 * (its own measured
# s/step from tc_probe3_shape_screen.sh), so the step counts are matched BY CONSTRUCTION and the
# comparison is pure quality-per-step with no data-wall confound (2135 < the ~2195-step one-epoch
# wall, so every arm sees one epoch and nothing more, and no post-hoc -0.077/ln(steps) correction is
# needed). 2135 rather than the doc's 2150 because 2135 is exactly what the CURRENT LOCKED CONFIG
# reaches in the official 1800 s -- so the control arm is the real submission run, not a shortened
# proxy for it, and every other arm's bpb is directly readable as "what this shape is worth at the
# locked config's own step count".
#
# Every arm carries the (now default) best config explicitly, so the arms survive a further defaults
# change on the other box. Reference: the other box's `t17-l1-m128` = 2135 steps / **1.004363** at this
# exact 640x6 shape and these exact flags, so arm sh-640x6-2135 is a cross-BOX reproduction check.
# If it does not land near 1.0044 nothing else in the table is trustworthy.
#
# DEPTH ARMS ARE DELIBERATELY NOT HERE. The 120-step screen killed both: 640x8 ran 1.91 s/step
# (-55.7% vs the control's 0.85) with a step-110 proxy loss of 3.9480 that is WORSE than the
# control's 3.9346, and 768x8 was worse still. A double negative -- slower AND worse per step -- has
# never been recovered by a full-budget run in this project's history, and at 1800 s depth 8 would
# reach ~942 steps for a +0.063 bpb step penalty. Recorded from the screen, not paid for again.
#
# Usage: SECS_640x6=... SECS_768x6=... SECS_896x6=... bash this.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
BEST=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze
      --ngram-ve-layers "1,-1" --ngram-ve-table-mult 128)
run() {
  local name="$1"; local secs="$2"; shift 2
  echo "=== $name : budget=${secs}s : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --max-train-seconds "$secs" --out-dir "out/tc-$name" \
    "${BEST[@]}" "$@" > "research/run-logs/tc-$name.log" 2>&1
  grep -E "num_params=|public_val_bpb|causality|train_steps|total_steps" "research/run-logs/tc-$name.log" | tail -10
  grep -E "^step 0(19|20|21|22)[0-9]0 " "research/run-logs/tc-$name.log" | tail -3
}
run sh-640x6-2135 "${SECS_640x6:?}"
run sh-768x6-2135 "${SECS_768x6:?}" --n-embd 768
run sh-896x6-2135 "${SECS_896x6:?}" --n-embd 896
echo "=== done ==="
