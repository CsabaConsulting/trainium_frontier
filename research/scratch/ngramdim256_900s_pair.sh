#!/usr/bin/env bash
# Paired ~900s spot-check: does --ngram-ve-dim 256 stacked on top of the CURRENT locked
# two-injection-site config (layers "1,-1", table_mult 128, dim 128 -- all module defaults now)
# clear break-even? Step-matched via a fresh 120-step calibration
# (research/run-logs/probe-ngd256-{ctrl,arm}.log, same box/session/commit):
#   ctrl (locked defaults, dim 128)   dt=0.8327 s/step  592.9M params  step-110 loss 3.9336
#   arm  (+ --ngram-ve-dim 256)       dt=1.0355 s/step  1130.1M params step-110 loss 3.9357
# Target N=869 steps so the ARM lands at ~900s; control's budget is shortened to the same N so
# both produce the same step count "by construction" (this week's step-matching convention).
set -u
cd "$(dirname "$0")/../.." || exit 1
python research/run_experiment.py --label ngd256-ctrl-900 --solution T3.1 \
  --description "current locked defaults (layers 1,-1 / mult 128 / dim 128), step-matched to 869 steps for the dim-256 spot-check" \
  --num-steps 5000 --max-train-seconds 724 \
  --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024
python research/run_experiment.py --label ngd256-arm-900 --solution T3.1 \
  --description "locked defaults + --ngram-ve-dim 256 on top of layers 1,-1 + mult 128, ~900s, step-matched to ~869 steps" \
  --num-steps 5000 --max-train-seconds 900 \
  --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 --ngram-ve-dim 256
echo "=== pair done ==="
