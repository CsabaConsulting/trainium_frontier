#!/usr/bin/env bash
# 2026-09-15 -- RESHARD HUNT, Block G: is layer 3 affordable in the config that would actually SHIP
# it, i.e. `--ngram-ve-layers "1,3,-1"` with the real table set?
#
# The brief's escalation question was "does removing the reshard cost make layer 3 affordable".
# Block A answered a narrower version of it: with ONE bigram table in isolation, layer 3 costs
# +1.5 ms/micro (~1%) over layer 5 at 4 ranks, against the +47.7 ms measured at 1 rank. So the
# throughput objection to layer 3 evaporated without anyone fixing anything.
#
# But Block F then showed that single-table isolation is NOT predictive of the shipped shape: layer 1
# alone costs +82 ms/micro, while layer 1 TOGETHER WITH layer 5 costs +1.2 ms. The graph's schedule
# depends on the SET of injection positions, not on any one of them. So "layer 3 is free" measured
# with one table at layer 3 does not license "layer 3 is free in {1,3,5} with four... six tables".
#
# This block measures the real thing. g1 is exactly NGRAM_VE_LAYERS="1,3,-1" at the locked geometry
# (bigram + trigram on layers 1, 3 and 5 = SIX tables); g2 is the shipped NGRAM_VE_LAYERS="1,-1"
# (four tables, num_params 592,916,844 = the locked config's own parameter count) as a same-session
# paired control. g3 adds the {1,3,5} bigram-only point so the marginal cost of the third POSITION
# can be separated from the marginal cost of two extra TABLES.
#
# This does not reopen layer 3 on quality -- that verdict is 120-step proxy 3.9973 alone / 3.9855
# with layer 1 against a 3.9608 control, and it is a loss regardless of throughput. It closes the
# brief's question with the number the question actually asked for.
set -u
cd "$(dirname "$0")/../.." || exit 1
source /home/ubuntu/workspace/native_venv/bin/activate
OUT=${OUT:-research/run-logs/reshard-g}
mkdir -p "$OUT" /home/ubuntu/prof/reshard
GEOT=(--ngram-ve --ngram-ve-trigram --ngram-ve-table-mult 128 --ngram-ve-dim 128
      --ngram-ve-layers "")
GEO=(--ngram-ve --no-ngram-ve-trigram --ngram-ve-table-mult 128 --ngram-ve-dim 128
     --ngram-ve-layers "")

run() {
  local name="$1"; shift
  echo "=== $(date -u +%H:%M:%S) $name : $* ===" | tee -a "$OUT/summary.txt"
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 \
    research/scratch/prof_driver.py \
    --out "/home/ubuntu/prof/reshard/$name" --window fwd --no-capture \
    --json-out "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
  grep -E "\[drv\] (num_params|profiled window|micro dts)" "$OUT/$name.log" | tail -4 \
    | tee -a "$OUT/summary.txt"
  grep -iE "Traceback|Error|out of memory" "$OUT/$name.log" | head -3 | tee -a "$OUT/summary.txt"
}

run g1-l135-tri "${GEOT[@]}" --ngram-layers "1,3,5"   # == NGRAM_VE_LAYERS "1,3,-1", 6 tables
run g2-l15-tri  "${GEOT[@]}" --ngram-layers "1,5"     # == the SHIPPED "1,-1", 4 tables, 592,916,844
run g3-l135-bi  "${GEO[@]}"  --ngram-layers "1,3,5"   # 3 tables: third position, no extra pair
echo "=== $(date -u +%H:%M:%S) done ===" | tee -a "$OUT/summary.txt"
