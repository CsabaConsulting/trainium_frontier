#!/usr/bin/env bash
# 2026-09-15 -- RESHARD HUNT, Block F: does the +83.7 ms layer-1 penalty survive in the config
# train.py ACTUALLY SHIPS?
#
# Block A measured single-table arms so that params/bytes were identical and only the injection
# POSITION moved. That isolation is what made the result readable, but it also means every arm in
# Block A is a config nobody ships. The locked config is NGRAM_VE_LAYERS="1,-1" -- tables on layer 1
# AND layer 5 (`-1` resolves to the last VE-eligible layer, 5 at DEPTH 6), bigram and trigram, so
# four tables. Block A says layer 1 alone costs +83.7 ms/micro over layer 5 alone; but the
# 2026-09-14 T1.7 probe measured the marginal cost of ADDING layer 1 to the shipped layer-5 config
# at only -5.86% throughput, i.e. roughly -9 ms, not +84. Those two facts are not compatible unless
# the layer-5 table's presence RESCUES the layer-1 injection.
#
# That distinction decides whether anything is left on the table at all:
#   if {1,5} ~= {5}      -> the penalty is an artefact of layer 1 being the ONLY injection, the
#                           shipped config never pays it, and the reshard hunt closes fully negative
#                           with no remaining target.
#   if {1,5} ~= {1}      -> the shipped config IS paying ~84 ms/micro (>35% of the micro-step) and
#                           there is a large, real, previously-unmeasured target -- worth far more
#                           than the dynamic address ever was.
# Either way this is the number that should have been measured first, and it costs 4 arms.
#
# f3/f4 add the trigram dimension (2 tables per position instead of 1) because the shipped config
# has both, and Probe 3 Part D's inversion was measured with the full table set: if the effect is
# about the NUMBER of injected tensors at a position rather than the position, f3 vs f1 shows it.
set -u
cd "$(dirname "$0")/../.." || exit 1
source /home/ubuntu/workspace/native_venv/bin/activate
OUT=${OUT:-research/run-logs/reshard-f}
mkdir -p "$OUT" /home/ubuntu/prof/reshard
GEO=(--ngram-ve --no-ngram-ve-trigram --ngram-ve-table-mult 128 --ngram-ve-dim 128
     --ngram-ve-layers "")
GEOT=(--ngram-ve --ngram-ve-trigram --ngram-ve-table-mult 128 --ngram-ve-dim 128
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

# Same-session controls, bigram only, so f1/f2 are directly paired with f3 rather than with
# Block A's cross-session numbers.
run f1-l5-only   "${GEO[@]}"  --ngram-layers "5"
run f2-l1-only   "${GEO[@]}"  --ngram-layers "1"
run f3-l15-both  "${GEO[@]}"  --ngram-layers "1,5"
# The shipped table set: bigram + trigram on both positions = 4 tables.
run f4-l15-tri   "${GEOT[@]}" --ngram-layers "1,5"
run f5-l5-tri    "${GEOT[@]}" --ngram-layers "5"
echo "=== $(date -u +%H:%M:%S) done ===" | tee -a "$OUT/summary.txt"
