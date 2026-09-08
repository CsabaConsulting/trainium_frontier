#!/usr/bin/env bash
# Per-step cost of the RWKV time-mix as a function of RWKV_CHUNK, on the box.
#
# The pure-RWKV model runs at ~1.88 s/step eager (and 1.82 s/step compiled, after a
# 505 s compile bill that eats half a 900 s budget) against the transformer baseline's
# 0.90 s/step. RWKV_CHUNK trades intra-chunk work (O(T*Q) per head) against
# inter-chunk work (O(T*N*N/Q)), so there is a real optimum; 8 steps is enough since
# dt is already flat by step 2.
#
# Usage on the box:  bash research/scratch/rwkv_chunk_sweep.sh 2>&1 | tee /tmp/qsweep.log
set -u
export PATH=/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin
export VIRTUAL_ENV=/home/ubuntu/workspace/native_venv
export NEURON_LOGICAL_NC_CONFIG=2
cd /home/ubuntu/trainium_frontier || exit 1

for Q in 64 128 256 512; do
  echo "##### RWKV_CHUNK=$Q"
  timeout 1200 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps 8 --max-train-seconds 1800 --out-dir "out/qs$Q" \
    --eval-tokens 4096 --no-compile --tensor-lr-scalars --lrm-quantum 0 \
    --rwkv --rwkv-chunk "$Q" 2>&1 | grep -E '^step|Error|Traceback|assert'
  sleep 5
done
echo SWEEPDONE
