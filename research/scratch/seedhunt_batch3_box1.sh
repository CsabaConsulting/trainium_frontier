#!/bin/bash
set -uo pipefail
cd ~/trainium_frontier
source ~/workspace/native_venv/bin/activate
SEEDS='1000 1500 2000 2500 3000 3500'
for s in $SEEDS; do
  echo "=== STARTING seed $s at $(date) ==="
  LOG=research/run-logs/seedhunt-box1-batch3-seed${s}-1800s.log
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --num-steps 1000000 --seed $s > "$LOG" 2>&1
  echo "=== FINISHED seed $s at $(date) exit=$? ==="
  echo "DONE_SEED_$s" >> research/run-logs/seedhunt-box1-batch3.progress
done
echo ALL_DONE >> research/run-logs/seedhunt-box1-batch3.progress
