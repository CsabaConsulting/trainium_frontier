#!/usr/bin/env bash
# Track: throughput->capacity, width sweep -- ONE extra point at n_embd=1024, DEPTH=6.
#
# Extends the existing width-only curve (640/768/896 x 6, all measured by
# research/scratch/tc_probe3_shape_screen.sh on this same box) by one more clean multiple of
# HEAD_DIM=128. Same 120-step instrument, same flags, same parse, so the new row drops straight into
# the Part A table in research/trn2-experiment-results.md.
#
# WHY --depth 6 IS PASSED EXPLICITLY even though 6 is the default: the shape screen already burned a
# run on the `--depth 8` trap (n_embd = round_up(DEPTH * ASPECT_RATIO(96), HEAD_DIM(128)) when
# N_EMBD == 0, so --depth alone moves BOTH axes). Passing both axes on every arm makes the shape
# unambiguous from the command line alone.
#
# n_embd must be a multiple of --head-dim (128): 1024 -> 8 heads.
#
# WHY THE CONTROL IS RE-RUN TWICE, BRACKETING THE NEW ARM: (a) train.py has moved since the original
# screen (three new items landed, all default OFF) and the control re-run is the check that they cost
# nothing; (b) a bracketing pair gives this session its own s/step reproducibility band instead of
# borrowing the 0.27% figure from the accidental repeat in the original screen; (c) box drift cannot
# then masquerade as the width effect.
#
# Flags: none. Every flag the original screen passed explicitly (--compile-muon, --qk-shift
# --qk-shift-beta 0.0 --qk-shift-freeze, --ngram-ve-layers "1,-1", --ngram-ve-table-mult 128) has
# since been promoted to a module-level default, and USE_NKI_ROPE_NORM is still True, so the bare
# launch reproduces the screen's conditions exactly. Verified against the `hyperparameters` dump.
set -u
cd "$(dirname "$0")/../.." || exit 1
# This box's interpreter is the native_venv the whole track was measured in. Activate it explicitly
# so the driver is safe to launch detached (tmux/nohup), where an interactive profile is not sourced.
VENV=${VENV:-$HOME/workspace/native_venv}
[ -x "$VENV/bin/torchrun" ] || { echo "no torchrun in $VENV" >&2; exit 1; }
# shellcheck disable=SC1091
. "$VENV/bin/activate"
STEPS=${STEPS:-120}
run() {
  local name="$1"; shift
  echo "=== $(date -u +%H:%M:%S) $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/tc-probe-$name" \
    --eval-seq-len 1024 "$@" > "research/run-logs/tc-probe-$name.log" 2>&1
  grep -E "num_params=" "research/run-logs/tc-probe-$name.log" | head -1
  grep -E "^step 00(0[1-9]0|1[01]0) " "research/run-logs/tc-probe-$name.log" | tail -12
  grep -E "num_params_M|causality|public_val_bpb" "research/run-logs/tc-probe-$name.log" | head -6
  grep -iE "Traceback|Error|out of memory" "research/run-logs/tc-probe-$name.log" | head -3
}
run w1024-ctrl-r1 --n-embd 640  --depth 6
run w1024-1024x6  --n-embd 1024 --depth 6
run w1024-ctrl-r2 --n-embd 640  --depth 6
echo "=== $(date -u +%H:%M:%S) done ==="
