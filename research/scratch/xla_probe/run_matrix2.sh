#!/usr/bin/env bash
# Probe 1 matrix, round 2. Round 1 (run_matrix.sh) measured shape A/B with
# grad_accum > 1, which is numerically INVALID under XLA: index_add_ across a
# mark_step boundary keeps only the last call, so every embedding gradient is
# wrong (see research/trn2-experiment-results.md). These arms pin grad_accum=1
# so both backends compute the same thing, plus one ga=4 XLA arm kept purely as
# a timing reference.
# One process at a time. Run under tmux from ~/trainium_frontier.
set -u

REPO=/home/ubuntu/trainium_frontier
NATIVE_VENV=/home/ubuntu/workspace/native_venv
XLA_VENV=/home/ubuntu/workspace/xla_venv
STEPS=${STEPS:-40}
WARM=${WARM:-10}
COOLDOWN=${COOLDOWN:-20}

# Locked-config shape (depth 6, n_embd 640, seq 2048, device batch 4).
# 4 ranks * 4 * 2048 = 32768 tokens per micro-step.
SHAPE_B1="--depth 6 --n-embd 640 --seq-len 2048 --device-batch-size 4 --total-batch-size 32768"
SHAPE_B4="--depth 6 --n-embd 640 --seq-len 2048 --device-batch-size 4 --total-batch-size 131072"

run_native() {  # $1 label, $2.. args
  local label=$1; shift
  echo "=== BEGIN $label $(date -Is) ===" >> /tmp/tc-matrix2.log
  ( source $NATIVE_VENV/bin/activate
    cd $REPO
    PYTHONPATH=$REPO NEURON_LOGICAL_NC_CONFIG=2 \
      torchrun --standalone --nproc_per_node=4 research/scratch/xla_probe/train_probe.py \
      --backend native --no-eval-public --num-steps $STEPS --warmup-measure-steps $WARM \
      --debug-numerics 2 --out-dir out/tc-$label --label $label "$@" ) > /tmp/tc-$label.log 2>&1
  echo "=== END $label rc=$? $(date -Is) ===" >> /tmp/tc-matrix2.log
  grep -E '^probe_|^dbg|^    (grads|params)' /tmp/tc-$label.log >> /tmp/tc-matrix2.log
  sleep $COOLDOWN
}

run_xla() {  # $1 label, $2.. args
  local label=$1; shift
  echo "=== BEGIN $label $(date -Is) ===" >> /tmp/tc-matrix2.log
  ( source $XLA_VENV/bin/activate
    cd $REPO
    PYTHONPATH=$REPO PJRT_DEVICE=NEURON NEURON_LOGICAL_NC_CONFIG=2 \
    NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
      torchrun --standalone --nproc_per_node=4 research/scratch/xla_probe/train_probe.py \
      --backend xla --no-eval-public --num-steps $STEPS --warmup-measure-steps $WARM \
      --debug-numerics 2 --out-dir out/tc-$label --label $label "$@" ) > /tmp/tc-$label.log 2>&1
  echo "=== END $label rc=$? $(date -Is) ===" >> /tmp/tc-matrix2.log
  grep -E '^probe_|^dbg|^    (grads|params)' /tmp/tc-$label.log >> /tmp/tc-matrix2.log
  sleep $COOLDOWN
}

: > /tmp/tc-matrix2.log
run_native p2-nat-B-ga1 $SHAPE_B1
run_xla    p2-xla-B-ga1 $SHAPE_B1
run_xla    p2-xla-B-ga4 $SHAPE_B4
echo "MATRIX2_DONE" >> /tmp/tc-matrix2.log
