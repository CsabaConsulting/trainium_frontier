#!/usr/bin/env bash
# Paired 900s A/B for --muon-lr-fp32 / USE_MUON_LR_FP32 (see the _muon_lr_g docstring, and the
# "side finding, recorded but NOT acted on" note in the 2026-09-15 sched-tricks section of
# trn2-experiment-results.md, which is where this candidate was flagged and explicitly deferred).
#
# The locked default path multiplies a 0-dim fp32 `lr` by the bf16 Newton-Schulz iterate `g`.
# PyTorch's 0-dim-vs-dimensioned type promotion rule casts `lr` DOWN to bf16 BEFORE the multiply,
# quantising every Muon group's effective lr by up to ~0.4% relative error, every step, uniformly
# per group per step (tiny lr jitter, not directional noise). --muon-lr-fp32 keeps the product in
# fp32 instead. Zero new tensors, zero new ops that survive to the shipped path, no expected
# throughput impact -- a precision-only change, so the decomposition below is mainly there to
# CONFIRM the throughput term really is zero, not to explain away a throughput regression.
#
# Bracketed by two fresh same-commit controls (first and last), same discipline as
# sched_tricks_900s.sh: one 4-rank job owns the whole box, serial, and a 900s number here is only
# comparable inside this block.
#
# DECOMPOSITION IS MANDATORY: delta = throughput term (-0.077 * ln(tokens_arm/tokens_ctrl)) +
# per-step residual. Noise floor: |delta| < 0.001 is noise; < 0.003 needs a same-day paired
# repeat before trusting. Escalate to the confirming 1800s pair only if the per-step residual is
# a real, reproducible win >= the noise floor.
set -u
cd "$(dirname "$0")/../.." || exit 1
run() {
  local label="$1"; local desc="$2"; shift 2
  echo "=== $(date -u +%H:%M:%S) $label : $* ==="
  python research/run_experiment.py --label "$label" --solution R1.2 \
    --description "$desc" --num-steps 5000 --max-train-seconds 900 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}

run mlrfp32-ctrl-900s-r1 "fresh same-commit 900s control for the --muon-lr-fp32 A/B, flag OFF (locked default: 0-dim fp32 lr gets cast to bf16 before the multiply against the bf16 Newton-Schulz iterate)"

run mlrfp32-arm-900s \
  "muon-lr-fp32 ON: keep the lr*g product in fp32 (g upcast, lr untouched, cast down to w's dtype only at the final subtract) instead of letting 0-dim type promotion cast lr to bf16 first. Zero-cost precision fix, no new tensors/ops, no expected throughput impact." \
  --muon-lr-fp32

run mlrfp32-ctrl-900s-r2 "closing same-commit 900s control: measures this block's own drift/noise band"
echo "=== $(date -u +%H:%M:%S) done ==="
