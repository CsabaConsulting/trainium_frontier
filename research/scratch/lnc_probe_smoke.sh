#!/bin/bash
# LNC=1 investigation: smoke probes. Args: <name> <lnc> <nproc> <steps> [extra flags...]
set -u
NAME=$1; LNC=$2; NPROC=$3; STEPS=$4; shift 4
cd ~/trainium_frontier
export NEURON_LOGICAL_NC_CONFIG=$LNC
export NEURON_COMPILE_CACHE_DIR=/tmp/neuron_cache_lnc${LNC}
LOG=~/lnc_logs/${NAME}.log
mkdir -p ~/lnc_logs
echo "=== $NAME lnc=$LNC nproc=$NPROC steps=$STEPS extra=$* md5=$(md5sum train.py | cut -d" " -f1) $(date -u) ===" | tee $LOG
~/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=$NPROC train.py \
  --num-steps $STEPS --max-train-seconds 100000 "$@" 2>&1 | tee -a $LOG
echo "=== EXIT ${PIPESTATUS[0]} $(date -u) ===" | tee -a $LOG
