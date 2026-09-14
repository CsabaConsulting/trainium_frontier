#!/usr/bin/env bash
# Device smoke test for the three new flagged items (byte-feature wte init, parameter-matched
# coalesced SwiGLU, cautious WD + smoothed scalar betas) before any 900s device time is spent.
#
# WHY: the CPU checks (research/scratch/cautious_wd_equiv.py, plus the byte-init determinism/RMS and
# param-count checks) prove the MATH. They do not prove the compiled graph builds, that the new
# strided last-dim slices in the SwiGLU MLP do not fall back or fail to legalize, or that causality
# is still exact. On this stack those are separate questions from the math -- see the note at the top
# of tc_cleanup_smoke.sh.
#
# PASS for every arm: run completes, causality_passed True and causality_max_abs 0.0, no Traceback,
# and (for the arms that are supposed to be shape-neutral) num_params unchanged at 592,916,844.
set -u
cd "$(dirname "$0")/../.." 2>/dev/null || cd ~/trainium_frontier
BEST=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze
      --ngram-ve-layers "1,-1" --ngram-ve-table-mult 128)

run() {
  local name="$1"; shift
  echo "=== $name ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps 15 --max-train-seconds 1800 --out-dir "out/smoke-$name" \
    --eval-tokens 262144 --eval-seq-len 1024 "${BEST[@]}" "$@" \
    > "research/run-logs/smoke-$name.log" 2>&1
  grep -E "num_params=|byte-wte-init:|causality_passed|causality_max_abs|val_bpb" \
    "research/run-logs/smoke-$name.log" | head -8
  grep -iE "Traceback|Error|out of memory|nan" "research/run-logs/smoke-$name.log" | head -3
}

run ctrl
run bytewte  --byte-wte-init
run swiglu   --coalesced-swiglu
run cwd      --cautious-wd
run sbeta    --scalar-beta2 0.99 --scalar-lr 0.10
echo "=== done ==="
