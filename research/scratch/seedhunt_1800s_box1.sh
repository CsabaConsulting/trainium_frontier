#!/usr/bin/env bash
# Real seed hunt at FULL 1800s budget, box 1, run sequentially in tmux.
#
# User-approved (explicit go-ahead) continuation of the seed-variance
# investigation: the 2026-09-15 3-seed check (42/43/44) found a real
# max-min 0.006136 bpb spread; the same-day 8-seed 300-step cheap screen
# (1,2,3,5,8,13,21,100) found a similarly-sized spread at a much cheaper
# budget. This script runs 6 MORE seeds at the real full 1800s budget,
# using the literal bare launch command (no run_experiment.py wrapper, no
# --no-eval-public, so public_val_bpb comes directly from the in-process
# eval -- same convention as the 3-seed check and the box-2 cross-box
# section in research/trn2-experiment-results.md).
#
# Seeds 5 and 13 are deliberately re-tests of the cheap screen's best/worst
# (of the 8 short-probe seeds) -- checking whether the 300-step proxy
# ranking predicts the full-1800s ranking. 34/55/89/144 are new coverage.
# Box 2 is concurrently running its own batch (1,2,3,5,8,13) -- seeds 5 and
# 13 therefore get run on BOTH boxes, an extra cross-box confirmation point.
#
# Run on the box, inside tmux, from the repo root:
#   tmux new-session -d -s seedhunt 'bash research/scratch/seedhunt_1800s_box1.sh > research/run-logs/seedhunt-box1-driver.log 2>&1'
set -u
cd ~/trainium_frontier || exit 1
mkdir -p research/run-logs

SEEDS="5 13 34 55 89 144"

for s in $SEEDS; do
  echo "=== $(date -u +%FT%TZ) seed=${s} START ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps 1000000 --seed "$s" \
    > "research/run-logs/seedhunt-box1-seed${s}-1800s.log" 2>&1
  rc=$?
  echo "=== $(date -u +%FT%TZ) seed=${s} DONE (exit ${rc}) ==="
done

echo "=== $(date -u +%FT%TZ) ALL SEEDS DONE ==="
