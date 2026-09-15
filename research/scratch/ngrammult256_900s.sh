#!/usr/bin/env bash
# Decisive step-matched 900s-class pair for --ngram-ve-table-mult 256 (dim UNCHANGED at 128),
# isolating the TABLE WIDTH axis alone at the CURRENT two-injection-site config
# (NGRAM_VE_LAYERS="1,-1", locked mult=128 baseline). Companion to the concurrent box1 agent's
# --ngram-ve-dim 256 isolation (mult unchanged at 128) -- together the two arms tell us whether any
# gain from the historical pre-layer-1 "dim 256 + mult 128" precedent comes from table WIDTH, from
# embedding DIM, or needs both.
#
# Calibration (research/scratch/ngrammult256_calib.sh, 120-step fixed instrument, this session,
# same box/commit):
#   ctrl (locked, mult=128)  s/step 0.8438   tok/s 155,327 (sd 0.26%)  592.9M params  step-110 loss 3.9372
#   arm  (mult=256)          s/step 1.0371   tok/s 126,380 (sd 0.32%)  1129.8M params step-110 loss 3.9351
# i.e. mult 256 costs +22.9% s/step / -18.6% tok/s on TWO injection sites -- far more than the old
# single-site mult-128 precedent (-3.64%) and ~9.5x the mult 64->128 upgrade's own two-site cost.
# This is NON-TRIVIAL, so per the standing methodology the decisive pair is STEP-MATCHED by scaling
# --max-train-seconds, not run flat at 900s each (a flat-900s comparison would be 1066 vs 868 steps,
# a 23% gap the -0.077/ln(steps) slope was never fit across in one hop).
#
# Target N = control's natural 900s step count (900 / 0.8438 = 1066). Control runs flat at the
# standard 900s. Arm's --max-train-seconds is pre-scaled to 1066 * 1.0371 = 1106s so both arms land
# at (approximately) the SAME step count -- the brief's per-step/throughput decomposition
# requirement satisfied by construction, same move as the 2026-09-14 "Item 4" capacity arms.
#
# Bracketed by two fresh same-commit controls (first and last), same discipline as
# muon_lr_fp32_900s.sh / sched_tricks_900s.sh -- one 4-rank job owns the whole box, serial. A
# number here is only comparable inside THIS block (do not borrow controls from other sessions).
#
# DECOMPOSITION: with steps already matched by construction, delta = pure per-step residual (no
# throughput term needed) -- but report the achieved step counts explicitly to confirm the match
# held (real full runs can drift slightly from the 120-step calibration's steady-state estimate).
# Noise floor: |delta| < 0.001 is noise; < 0.003 needs a same-day paired repeat before trusting.
set -u
cd "$(dirname "$0")/../.." || exit 1
run() {
  local label="$1"; local secs="$2"; local desc="$3"; shift 3
  echo "=== $(date -u +%H:%M:%S) $label (max-train-seconds=$secs): $* ==="
  python research/run_experiment.py --label "$label" --solution R1.2 \
    --description "$desc" --num-steps 5000 --max-train-seconds "$secs" \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}

run ngm256-ctrl-900s-r1 900 \
  "fresh same-commit 900s control for the --ngram-ve-table-mult 256 isolation A/B, flag at locked default (mult=128, dim=128, layers 1,-1)"

run ngm256-arm-900s 1106 \
  "table-mult 256 ON (dim unchanged at 128): isolates table WIDTH alone at the two-injection-site config. --max-train-seconds pre-scaled from the 120-step calibration (0.8438s/step ctrl, 1.0371s/step arm) to land at ~1066 steps, matching the control's natural 900s step count." \
  --ngram-ve-table-mult 256

run ngm256-ctrl-900s-r2 900 \
  "closing same-commit 900s control: measures this block's own drift/noise band"
echo "=== $(date -u +%H:%M:%S) done ==="
