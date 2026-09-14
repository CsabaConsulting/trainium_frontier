#!/usr/bin/env bash
# Track: throughput->capacity, Probe 3 step 3 -- PAIRED SAME-DAY REPEAT of the one live candidate.
#
# The ablate screen produced exactly one constant-capacity throughput win: `--no-nki-rope-norm` ran
# +1.35% faster than the full stack (159,043 vs 156,917 tok/s at step 110). That is a small delta
# against a 0.27% instrument, and the project's own noise rule requires a paired same-day repeat for
# anything this size, so it is repeated here INTERLEAVED (full, norope, full, norope) rather than
# back-to-back, so a slow drift in box temperature//dev state cannot masquerade as the effect.
#
# WHY THIS IS WORTH A CONFIRMATION RUN EVEN THOUGH 1.35% << THE 9.46% BREAK-EVEN. It does NOT rescue
# width -- nothing does; that verdict is already final. But it is a FREE win at CONSTANT CAPACITY on
# the currently locked 640x6 config (proxy losses 3.9366 vs 3.9380 are inside the +/-0.0034 120-step
# noise floor), it retires a hand-written NKI kernel (a README tie-break plus for simplicity), and at
# 2138 steps it buys ~+29 steps = -0.0010 bpb off the -0.077/ln(steps) slope. That is at the noise
# floor on its own, so it is handed to the main track as a simplification-with-a-tailwind, NOT as a
# claimed bpb win.
#
# Arm `ab-norope-768` asks whether the win is shape-dependent: the rope+norm kernel's cost scales
# with n_embd, so if dropping it is worth MORE at 768 the break-even arithmetic shifts (it would need
# to be worth 9.46% there, which is not remotely plausible, but the number is cheap to have and it is
# the last quantitative gap in the width verdict).
#
# Reminder recorded here so it is not re-litigated: `--no-local-conv`'s +2.65% is NOT available. Its
# help text used to claim the conv was dead compute; T0.1 made local_conv.weight trainable, so
# --no-local-conv drops 7,680 trained params and cost +0.267 of step-110 loss. Help text now fixed.
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
BEST=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze
      --ngram-ve-layers "1,-1" --ngram-ve-table-mult 128)
run() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/tc-probe-$name" \
    --eval-seq-len 1024 "${BEST[@]}" "$@" > "research/run-logs/tc-probe-$name.log" 2>&1
  grep -E "num_params=" "research/run-logs/tc-probe-$name.log" | head -1
  grep -E "^step 000[4-9]0 |^step 001[01]0 " "research/run-logs/tc-probe-$name.log" | tail -8
  grep -iE "Traceback|Error|out of memory" "research/run-logs/tc-probe-$name.log" | head -3
}
run ab-full-r2
run ab-norope-r2   --no-nki-rope-norm
run ab-full-r3
run ab-norope-r3   --no-nki-rope-norm
run ab-norope-768  --no-nki-rope-norm --n-embd 768
echo "=== done ==="
