#!/bin/bash
# World-size axis at LNC=2 (the only LNC that runs): fixed-step arms so throughput
# differences cannot confound quality. usage: <name> <nproc> <steps>
set -u
cd ~/trainium_frontier
run () {
  NAME=$1; NPROC=$2; STEPS=$3
  LOG=~/lnc_logs/ws-${NAME}.log
  echo "### $NAME nproc=$NPROC steps=$STEPS md5=$(md5sum train.py | cut -d' ' -f1) $(date -u)" | tee $LOG
  NEURON_LOGICAL_NC_CONFIG=2 NEURON_COMPILE_CACHE_DIR=/tmp/neuron_cache_ws${NPROC} \
    ~/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=$NPROC train.py \
    --num-steps $STEPS --max-train-seconds 100000 >> $LOG 2>&1
  echo "### EXIT $? $(date -u)" | tee -a $LOG
  sleep 8
}
run ws4-400 4 400
run ws2-400 2 400
run ws4-400-v2 4 400
echo WSAXIS_DONE > ~/lnc_logs/wsaxis.done
