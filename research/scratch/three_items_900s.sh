#!/usr/bin/env bash
# Paired 900s A/B sequence for three independent items, all measured against ONE fresh same-commit
# control that is repeated at the END of the sequence so the run's own drift/noise is on the record
# (a 900s number is only comparable inside a pair, never against a cross-day row).
#
#   1. --byte-wte-init   plan 11.1 byte-feature wte init. Expected THROUGHPUT-NEUTRAL by
#                        construction (host-side, one-shot, 0.84s, zero ops in the training graph),
#                        so the whole delta should land in the per-step residual. If step count
#                        moves, that is noise or drift, not the arm.
#   2. --coalesced-swiglu  parameter-matched coalesced SwiGLU, hidden 1664, one Linear(640, 3328).
#                        592,425,324 params vs 592,916,844 (-0.08%). Prior 15-20%: the small-model
#                        ReLU^2 >= SwiGLU consensus is hardware-independent, so a real negative is
#                        the expected outcome, not a failure.
#   3. --cautious-wd     T2.3, and the three --scalar-beta2 0.99 arms are T2.4 (record #52's
#                        "smoothed scalars"), sweeping SCALAR_LR over {0.20 = current, 0.10, 0.05}.
#
# DECOMPOSITION IS MANDATORY before any verdict: split each delta into the throughput term
# (-0.077 * ln(tokens_arm / tokens_ctrl), the measured below-the-wall slope) and the per-step
# residual. Noise floor: |delta| < 0.001 is noise; < 0.003 needs a paired same-day repeat.
# Escalate to 1800s only for a per-step residual <= -0.002, and check any 1800s result against the
# ~2195-step data wall.
set -u
cd "$(dirname "$0")/../.." || exit 1
run() {
  local label="$1"; local desc="$2"; shift 2
  echo "=== $(date -u +%H:%M:%S) $label : $* ==="
  python research/run_experiment.py --label "$label" --solution R1.2 \
    --description "$desc" --num-steps 5000 --max-train-seconds 900 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}

run ti-ctrl-900s-r1 "fresh same-commit 900s control for the byte-wte-init / coalesced-SwiGLU / cautious-WD block"

run ti-bytewte-900s \
  "plan 11.1 byte-feature wte init: hashed signed bag of byte 1..3-grams of each token's own bytes (tokenizer decode_single_token_bytes only), column-centered, rescaled to the control's 0.8 row RMS; host-side one-shot, zero training-graph ops, no torch RNG consumed" \
  --byte-wte-init

run ti-swiglu-900s \
  "parameter-matched coalesced SwiGLU MLP: one Linear(640, 3328) split into gate/up halves at hidden 1664, silu(gate)*up instead of relu(c_fc(x))**2; 592.43M vs 592.92M params, same launch count" \
  --coalesced-swiglu

run ti-cwd-900s \
  "T2.3 cautious weight decay on the AdamW groups with wd != 0 (wte 0.001, lm_head 0.01): decay only where update*p > 0" \
  --cautious-wd

run ti-sbeta99-lr020-900s \
  "T2.4 smoothed scalars: scalar group Adam betas (0.9, 0.99) at the current SCALAR_LR 0.20" \
  --scalar-beta2 0.99

run ti-sbeta99-lr010-900s \
  "T2.4 smoothed scalars: scalar group betas (0.9, 0.99) with SCALAR_LR 0.10" \
  --scalar-beta2 0.99 --scalar-lr 0.10

run ti-sbeta99-lr005-900s \
  "T2.4 smoothed scalars: scalar group betas (0.9, 0.99) with SCALAR_LR 0.05" \
  --scalar-beta2 0.99 --scalar-lr 0.05

run ti-ctrl-900s-r2 "closing same-commit 900s control: measures this sequence's own drift/noise band"
echo "=== $(date -u +%H:%M:%S) done ==="
