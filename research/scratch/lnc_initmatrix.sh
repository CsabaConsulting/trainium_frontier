#!/bin/bash
# LNC init-matrix: launch train.py, sample rank-0 python stack, record whether
# init_process_group completes.
# usage: lnc_initmatrix.sh <name> <nproc> <wait_s> [ENV=VAL ...] [-- <train.py flags>]
set -u
NAME=$1; NPROC=$2; WAIT=$3; shift 3
ENVS=(); FLAGS=()
seen_sep=0
for a in "$@"; do
  if [ "$a" = "--" ]; then seen_sep=1; continue; fi
  if [ $seen_sep -eq 0 ]; then ENVS+=("$a"); else FLAGS+=("$a"); fi
done
cd ~/trainium_frontier
mkdir -p ~/lnc_logs
LOG=~/lnc_logs/init-${NAME}.log
: > $LOG
echo "### $NAME nproc=$NPROC env: ${ENVS[*]} flags: ${FLAGS[*]:-none} $(date -u)" | tee -a $LOG
env "${ENVS[@]}" NEURON_COMPILE_CACHE_DIR=/tmp/neuron_cache_${NAME} \
  ~/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=$NPROC train.py \
  --num-steps 3 --max-train-seconds 100000 ${FLAGS[@]+"${FLAGS[@]}"} >> $LOG 2>&1 &
LAUNCH=$!
sleep $WAIT
echo "--- rank0 stack at t=${WAIT}s ---" | tee -a $LOG
PID=$(ps -eo pid,args | grep "[-]u train.py" | awk '{print $1}' | head -1)
if [ -n "$PID" ]; then
  sudo env "PATH=$PATH" ~/workspace/native_venv/bin/py-spy dump --pid $PID 2>&1 | tee -a $LOG
else
  echo "no train.py process alive at t=${WAIT}s (crashed or finished)" | tee -a $LOG
fi
if grep -q "world_size=" $LOG; then echo "VERDICT=INIT_OK" | tee -a $LOG; else echo "VERDICT=STUCK_OR_FAILED" | tee -a $LOG; fi
grep -E "Error|error message|RuntimeError|assert" $LOG | grep -v OperatorEntry | head -5 | tee -a $LOG
kill $LAUNCH 2>/dev/null
pkill -9 -f "[-]u train.py" 2>/dev/null
sleep 8
echo "### END $NAME $(date -u)" | tee -a $LOG
