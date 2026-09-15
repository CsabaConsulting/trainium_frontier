#!/usr/bin/env bash
# Paired ~900s spot-check: does depth=8 at FIXED width n_embd=640 (a pure depth increase,
# 6->8 layers, decoupled from the earlier CONFOUNDED "768x8" result where --depth 8 alone
# also bumped n_embd to 768 via round_up(DEPTH*ASPECT_RATIO,HEAD_DIM)) clear break-even?
# Step-matched via a fresh 120-step calibration
# (research/run-logs/probe-d8w640-{ctrl,arm}.log, same box/session/commit):
#   ctrl (locked defaults: depth=6, n_embd=640) dt=0.8336 s/step  592.9M params  step-110 loss 3.9350
#   arm  (--depth 8 --n-embd 640)               dt=1.6309 s/step  608.0M params  step-110 loss 4.0099
# Target N=552 steps so the ARM lands at ~900s; control's budget is shortened to the same N so
# both produce the same step count "by construction" -- same step-matching convention as
# ngramdim256_900s_pair.sh / tc_probe3f_width1024.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
python research/run_experiment.py --label depth8w640-ctrl-900 --solution T0.2 \
  --description "locked defaults (depth=6, n_embd=640) as-is, step-matched to ~552 steps for the depth=8-fixed-width-640 spot-check" \
  --num-steps 5000 --max-train-seconds 460 \
  --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024
python research/run_experiment.py --label depth8w640-arm-900 --solution T0.2 \
  --description "pure depth increase 6->8 at FIXED width n_embd=640 (explicit --n-embd override), decoupled from the confounded 768x8 result; ~900s, step-matched to ~552 steps" \
  --num-steps 5000 --max-train-seconds 900 \
  --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 --depth 8 --n-embd 640
echo "=== pair done ==="
