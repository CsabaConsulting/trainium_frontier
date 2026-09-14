#!/bin/bash
# Does LNC=1 come alive once the NEFFs are actually compiled with --lnc=1?
# train.py line 70 sets NEURON_CC_FLAGS via setdefault, so an external value wins.
set -u
cd ~/trainium_frontier
NAME=$1; NPROC=$2; WAIT=$3; STEPS=${4:-3}
LOG=~/lnc_logs/cc-${NAME}.log
echo "### $NAME nproc=$NPROC steps=$STEPS md5=$(md5sum train.py|cut -d' ' -f1) $(date -u)" | tee $LOG
NEURON_LOGICAL_NC_CONFIG=1 \
NEURON_CC_FLAGS="--optlevel=1 --lnc=1" \
NEURON_COMPILE_CACHE_DIR=/tmp/neuron_cache_cc_${NAME} \
  ~/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=$NPROC train.py \
  --num-steps $STEPS --max-train-seconds 100000 --no-clip-foreach >> $LOG 2>&1 &
PID=$!
for i in $(seq 1 $WAIT); do
  sleep 1
  kill -0 $PID 2>/dev/null || break
done
if kill -0 $PID 2>/dev/null; then
  RP=$(pgrep -f "[-]u train.py" | head -1)
  echo "--- rank0 stack at t=${WAIT}s ---" | tee -a $LOG
  sudo env "PATH=$PATH" ~/workspace/native_venv/bin/py-spy dump --pid $RP 2>&1 | tail -20 | tee -a $LOG
fi
if grep -q "world_size=" $LOG; then echo "VERDICT=INIT_OK" | tee -a $LOG; else echo "VERDICT=STUCK_OR_FAILED" | tee -a $LOG; fi
grep -E "^step|Error|error message|NRT|Mismatch" $LOG | sort -u | head -12
