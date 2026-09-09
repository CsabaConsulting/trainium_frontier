#!/usr/bin/env bash
# run_phase.sh <lnc> <nproc> <tag> [extra phase_timer args...]
# Clean timing env: NO profiling/inspect/DGE vars (they add per-launch overhead
# and can overflow the SW notification queue).
set -euo pipefail
LNC="$1"; NPROC="$2"; TAG="$3"; shift 3
cd /home/ubuntu/prof2026sep08
source ~/workspace/native_venv/bin/activate

unset NEURON_RT_ENABLE_DGE_NOTIFICATIONS || true
unset NEURON_RT_INSPECT_ENABLE || true
unset NEURON_RT_INSPECT_DEVICE_PROFILE || true
unset NEURON_RT_INSPECT_SYSTEM_PROFILE || true
unset NEURON_FRAMEWORK_DEBUG || true
unset NEURON_RT_VISIBLE_CORES || true
export NEURON_LOGICAL_NC_CONFIG="$LNC"

if [ "$NPROC" = "1" ]; then
  export NEURON_RT_VISIBLE_CORES=0
  python phase_timer.py "$@" 2>&1 | tee "phase_${TAG}.log"
else
  torchrun --nproc_per_node="$NPROC" --nnodes=1 phase_timer.py "$@" 2>&1 | tee "phase_${TAG}.log"
fi
echo "PHASE_DONE $TAG"
