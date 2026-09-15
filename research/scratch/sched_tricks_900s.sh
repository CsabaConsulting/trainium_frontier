#!/usr/bin/env bash
# Paired 900s A/B block for plan §11.1's three schedule-level tricks from Recursive's recipe
# (0.9109 on the nanochat autoresearch task -- the strongest external evidence in this competition,
# and these three are the last items of its "zero new ops" class that were never ported here).
#
# Bracketed by two fresh same-commit controls (first and last) so the sequence's own drift is on the
# record; a 900s number is only comparable inside a block, never against a cross-day row. Serial:
# one 4-rank job owns the whole box.
#
#   1. --demon-beta1                Adam "Demon": anneal the AdamW groups' beta1 down across the LR
#                                   warmdown. Recursive: 0.8 -> 0.55. Two arms: their window (the
#                                   warmdown, i.e. progress 0.25 -> 1, the default) and the whole
#                                   run (--demon-beta1-start-frac 0.0), because "across its
#                                   warmdown" is one sentence in a secondary source and the window
#                                   is the half of the spec most likely to have been paraphrased.
#   2. --muon-beta2-final           NorMuon beta2 ramp. Our tuned constant is 0.90, not Recursive's
#                                   0.95, so only the DIRECTION transfers. Three arms separate the
#                                   two things that could be doing the work: ramp to 0.95, ramp to
#                                   0.97 (Recursive's endpoint), and a FLAT 0.95 -- if flat 0.95
#                                   matches the ramp, the item is just "beta2 was mistuned" and the
#                                   ramp is decoration.
#   3. --muon-depth-lr-* / -mom-*   Per-depth LR (x1.15 bottom -> x0.85 top) and momentum (0.90 ->
#                                   0.97). Three arms: LR alone, momentum alone, both -- Recursive
#                                   changed them together, which is not evidence that either one
#                                   needs the other.
#
# All three are pure optimizer-schedule changes: no new tensors, no new ops, no new parameters, and
# param count must read 592,916,844 in every arm. The throughput term is therefore EXPECTED to be
# exactly zero, which is a prediction to check rather than an excuse to skip the decomposition --
# research/scratch/sched_tricks_smoke.sh measures dt directly, and any step-count movement here
# larger than the control pair's own is evidence the prediction is wrong.
#
# DECOMPOSITION IS MANDATORY before any verdict: delta = throughput term
# (-0.077 * ln(tokens_arm / tokens_ctrl), the measured below-the-wall slope) + per-step residual.
# Noise floor: |delta| < 0.001 is noise; < 0.003 needs a paired same-day repeat. Escalate to 1800s
# only for a per-step residual <= -0.002, and check any 1800s result against the ~2195-step data
# wall. Verdicts must reproduce at BOTH eval lengths (2048 and 1024) to within 0.0003.
set -u
cd "$(dirname "$0")/../.." || exit 1
run() {
  local label="$1"; local desc="$2"; shift 2
  echo "=== $(date -u +%H:%M:%S) $label : $* ==="
  python research/run_experiment.py --label "$label" --solution R1.2 \
    --description "$desc" --num-steps 5000 --max-train-seconds 900 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}

run st-ctrl-900s-r1 "fresh same-commit 900s control for the plan-11.1 schedule-tricks block (Demon beta1 / NorMuon beta2 ramp / per-depth Muon LR+momentum)"

run st-demon-900s \
  "plan 11.1 item 1, Adam Demon: anneal every AdamW group's beta1 down across the LR warmdown as a multiplicative shrink on its own initial beta1, calibrated so the 0.8 embedding family lands on 0.55 (Recursive's numbers). Starts at progress 1-WARMDOWN_RATIO=0.25. RMSProp n-gram group unaffected (no first moment)." \
  --demon-beta1

run st-demonfull-900s \
  "plan 11.1 item 1 with the anneal spread over the WHOLE run instead of the warmdown window: same 0.8 -> 0.55 endpoints, start_frac 0.0" \
  --demon-beta1 --demon-beta1-start-frac 0.0

run st-b2ramp95-900s \
  "plan 11.1 item 2, NorMuon beta2 ramp: Muon second-moment beta2 0.90 -> 0.95 linearly over elapsed/max_train_seconds. 1-beta2 reaches the compiled Muon body as a device tensor (fifth slot of the existing single staged copy), not a Python float." \
  --muon-beta2-final 0.95

run st-b2ramp97-900s \
  "plan 11.1 item 2 at Recursive's upper endpoint: Muon beta2 0.90 -> 0.97 over the run" \
  --muon-beta2-final 0.97

run st-b2flat95-900s \
  "plan 11.1 item 2 DISCRIMINATOR: flat Muon beta2 0.95, no ramp. Separates 'the ramp is the mechanism' from '0.90 was simply mistuned'." \
  --muon-beta2 0.95

run st-depthlr-900s \
  "plan 11.1 item 3a, per-depth Muon LR only: x1.15 at layer 0 ramping linearly to x0.85 at layer 5, as a per-parameter [K,1,1] operand inside the existing per-shape Muon groups (grouping, batched Newton-Schulz and compiled-graph count all unchanged). Non-block matrices (n-gram up-projections) stay at 1.0." \
  --muon-depth-lr-bottom 1.15 --muon-depth-lr-top 0.85

run st-depthmom-900s \
  "plan 11.1 item 3b, per-depth Muon momentum only: target 0.90 at layer 0 -> 0.97 at layer 5, applied as a multiplier on (1 - momentum) at reference momentum 0.95 so it rides on top of the global warmup/cooldown schedule" \
  --muon-depth-mom-bottom 0.90 --muon-depth-mom-top 0.97

run st-depthboth-900s \
  "plan 11.1 item 3, per-depth LR and momentum together, exactly as Recursive pairs them (LR x1.15->x0.85, momentum 0.90->0.97)" \
  --muon-depth-lr-bottom 1.15 --muon-depth-lr-top 0.85 \
  --muon-depth-mom-bottom 0.90 --muon-depth-mom-top 0.97

run st-ctrl-900s-r2 "closing same-commit 900s control: measures this sequence's own drift/noise band"
echo "=== $(date -u +%H:%M:%S) done ==="
