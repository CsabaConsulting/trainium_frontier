#!/usr/bin/env bash
# Track: throughput->capacity, Probe 3 step 4 -- PART A: measure s/step for every width/depth arm.
#
# This exists so PART B (the real fixed-step runs) can set each arm's --max-train-seconds to
# 2150 * (that arm's own s/step) and have the step counts matched BY CONSTRUCTION. That is the
# protocol the other box used for the T1.7 spend arms, and it is what makes the comparison a pure
# quality-per-step comparison with no data-wall confound: 2150 is below the ~2195-step wall, so
# every arm sees one epoch and nothing more, and no post-hoc -0.077/ln(steps) correction is needed.
#
# 120-step fixed-step instrument, steady-state dt flat from ~step 5.
#
# WHY EVERY ARM CARRIES `--ngram-ve-layers "1,-1"` AS WELL AS THE BUDGET FLAGS:
# the track doc's step 4 says to sweep width/depth first and re-add the paid-off features to the
# winner afterwards. That ordering costs one extra full-budget run per candidate and, worse, prices
# width against a control (1.018982) that is already superseded -- the other box's `t17-l1-2150`
# reached 1.007376 at 2139 steps with exactly these flags, which is already inside this track's
# <=1.010 gate. Sweeping ON TOP of that config instead means:
#   - arm sh-640x6 is a cross-BOX reproduction check of 1.007376 (box 2 already reproduced the
#     locked baseline's 0.84 s/step / ~156k tok/s to 3 digits, so this should land near 1.0074);
#   - every other arm answers the only question that matters, "does width or depth beat the best
#     known config at equal steps?", in ONE run rather than two;
#   - a winner is immediately a gate candidate with no follow-up run needed.
# The known hazard, recorded up front: layer-1 n-gram cost and n-gram TABLE width were measured
# sub-additive (they fight for the same resharding budget). Model width (n_embd) is a different
# axis, but sub-additivity is the prior, not additivity.
#
# n_embd must be a multiple of --head-dim (128): 640 -> 5 heads, 768 -> 6, 896 -> 7.
# At --depth 8, has_ve(i,8) is i%2==1, so VE layers are {1,3,5,7} and "1,-1" = {1,7} -- still
# "earliest + last", the same intervention as at depth 6's {1,5}.
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
BEST=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze --ngram-ve-layers "1,-1")
run() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/tc-probe-$name" \
    --eval-seq-len 1024 "${BEST[@]}" "$@" > "research/run-logs/tc-probe-$name.log" 2>&1
  grep -E "num_params" "research/run-logs/tc-probe-$name.log" | head -2
  grep -E "^step 00(0[1-9]0|1[01]0) " "research/run-logs/tc-probe-$name.log" | tail -12
  grep -iE "Traceback|Error|out of memory" "research/run-logs/tc-probe-$name.log" | head -3
}
run sh-640x6                                   # control == locked shape, best-known flags
run sh-768x6 --n-embd 768
run sh-896x6 --n-embd 896
run sh-640x8 --depth 8
run sh-768x8 --n-embd 768 --depth 8
echo "=== done ==="
