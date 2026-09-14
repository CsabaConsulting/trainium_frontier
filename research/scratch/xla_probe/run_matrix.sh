#!/usr/bin/env bash
# Probe 1 measurement matrix: native eager vs torch_xla graph mode on the SAME
# file (research/scratch/xla_probe/train_probe.py), at two identical shapes.
# One process at a time. Run under tmux from ~/trainium_frontier.
set -u

REPO=/home/ubuntu/trainium_frontier
NATIVE_VENV=/home/ubuntu/workspace/native_venv
XLA_VENV=/home/ubuntu/workspace/xla_venv
STEPS=${STEPS:-40}
WARM=${WARM:-10}

# Shape A: upstream starter defaults (depth 10, 640, seq 1024, dbs 1, batch 65536 -> ga 16)
SHAPE_A="--depth 10 --n-embd 640 --seq-len 1024 --device-batch-size 1 --total-batch-size 65536"
# Shape B: our locked-config shape (depth 6, 640, seq 2048, dbs 4, batch 131072 -> ga 4)
SHAPE_B="--depth 6 --n-embd 640 --seq-len 2048 --device-batch-size 4 --total-batch-size 131072"

run_native() {  # $1 label, $2.. args
  local label=$1; shift
  echo "=== BEGIN $label $(date -Is) ===" >> /tmp/tc-matrix.log
  ( source $NATIVE_VENV/bin/activate
    cd $REPO
    PYTHONPATH=$REPO NEURON_LOGICAL_NC_CONFIG=2 \
      torchrun --standalone --nproc_per_node=4 research/scratch/xla_probe/train_probe.py \
      --backend native --no-eval-public --num-steps $STEPS --warmup-measure-steps $WARM \
      --out-dir out/tc-$label --label $label "$@" ) > /tmp/tc-$label.log 2>&1
  echo "=== END $label rc=$? $(date -Is) ===" >> /tmp/tc-matrix.log
  grep -E '^probe_' /tmp/tc-$label.log >> /tmp/tc-matrix.log
}

run_xla() {  # $1 label, $2.. args
  local label=$1; shift
  echo "=== BEGIN $label $(date -Is) ===" >> /tmp/tc-matrix.log
  ( source $XLA_VENV/bin/activate
    cd $REPO
    PYTHONPATH=$REPO PJRT_DEVICE=NEURON NEURON_LOGICAL_NC_CONFIG=2 \
    NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
      torchrun --standalone --nproc_per_node=4 research/scratch/xla_probe/train_probe.py \
      --backend xla --no-eval-public --num-steps $STEPS --warmup-measure-steps $WARM \
      --out-dir out/tc-$label --label $label "$@" ) > /tmp/tc-$label.log 2>&1
  echo "=== END $label rc=$? $(date -Is) ===" >> /tmp/tc-matrix.log
  grep -E '^probe_' /tmp/tc-$label.log >> /tmp/tc-matrix.log
}

: > /tmp/tc-matrix.log
run_native p1-nat-A $SHAPE_A
run_xla    p1-xla-A $SHAPE_A
run_native p1-nat-B $SHAPE_B
run_xla    p1-xla-B $SHAPE_B
run_xla    p1-xla-B-1graph $SHAPE_B --no-mark-every-micro
echo "MATRIX_DONE" >> /tmp/tc-matrix.log
