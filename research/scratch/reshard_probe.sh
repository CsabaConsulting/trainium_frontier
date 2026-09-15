#!/usr/bin/env bash
# 2026-09-15 -- RESHARD HUNT, Block A: re-measure the LNC=2 dynamic-address resharding penalty at
# the REAL 4-rank shape, and price the headroom a fix could possibly recover.
#
# WHY THIS EXISTS. The whole "dynamic gather forces a cross-physical-core reshard of the residual
# stream at every DOWNSTREAM layer boundary" mechanism (trn2-experiment-results.md, 2026-09-12
# "ROOT CAUSE of the layer-position effect") was measured with prof_driver.py, which was written
# single-process. So every number in it -- +47.6 ms forward, 42.4M cross-pcore packets at 30.6
# B/packet, 78-80% DMA engine occupancy -- is a ONE-RANK number. On 2026-09-14 the T1.7 spend probe
# showed the *sign of affordability* flips at four ranks (layer 1 went from "+34% unaffordable" to
# "-5.86%, and the biggest bpb win in the repo"), with the stated reason being that a residual
# stream already sharded four ways makes the cross-pcore DMA far cheaper. Nobody has re-measured
# the MECHANISM at four ranks. Standing rule: a measured fact expires when the stack changes.
#
# THE ONE ARM THAT DECIDES THE TASK. `--ngram-gather static` replaces `tbl[rows]` with the
# contiguous slice `tbl[:b*t]`: same bytes out of HBM, same output shape, same downstream graph,
# same FLOPs -- the ONLY difference is that the addresses are compile-time known. It is
# numerically meaningless (a timing arm), but it is an exact upper bound on what ANY fix to the
# dynamic address can recover, whether that fix is a NKI kernel island, an eager pre-pass, or a
# one-hot matmul. At one rank it recovered ~88% of the layer-3 penalty (+5.5 ms instead of +47.6).
#   If static == dynamic at four ranks, there is nothing to fix and the task closes negative.
#   If static << dynamic at four ranks, the gap is the budget a NKI gather could compete for.
#
# ISOLATION. Every table arm carries EXACTLY ONE bigram table (`--no-ngram-ve-trigram` +
# `--ngram-only-layer N`) at the locked mult 128 / dim 128 geometry, so params, optimizer state,
# gradient all-reduce volume and table bytes are identical across l1/l3/l5 and the ONLY difference
# is the injection position, i.e. the number of downstream layer boundaries (4 / 2 / 0).
# `--ngram-ve-layers ""` is required for `--ngram-only-layer` to be authoritative (see the
# prof_driver docstring for the trap where train.py's own "1,-1" default silently intersects).
#
# --window fwd because the 2026-09-12 attribution put +47.0 of the +52.6 ms in the forward NEFF.
# --no-capture here: Block A only needs wall clock, and a 2.8 GB NTFF per arm is pure cost.
set -u
cd "$(dirname "$0")/../.." || exit 1
source /home/ubuntu/workspace/native_venv/bin/activate
OUT=${OUT:-research/run-logs/reshard-a}
mkdir -p "$OUT" /home/ubuntu/prof/reshard
GEO=(--ngram-ve --no-ngram-ve-trigram --ngram-ve-table-mult 128 --ngram-ve-dim 128
     --ngram-ve-layers "")

run() {
  local name="$1"; shift
  echo "=== $(date -u +%H:%M:%S) $name : $* ===" | tee -a "$OUT/summary.txt"
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 \
    research/scratch/prof_driver.py \
    --out "/home/ubuntu/prof/reshard/$name" --window fwd --no-capture \
    --json-out "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
  grep -E "\[drv\] (num_params|ngram_ve=|profiled window|timed step|micro)" "$OUT/$name.log" \
    | tail -12 | tee -a "$OUT/summary.txt"
  grep -iE "Traceback|Error|out of memory" "$OUT/$name.log" | head -3 | tee -a "$OUT/summary.txt"
}

# Baseline: no tables at all. Not the reference for the position effect (it has different params),
# but it is the reference for "what do the tables cost in total".
run a0-notab --no-ngram-ve
# The position triple, dynamic address (what train.py actually does today).
run a1-l5-dyn "${GEO[@]}" --ngram-only-layer 5
run a2-l3-dyn "${GEO[@]}" --ngram-only-layer 3
run a3-l1-dyn "${GEO[@]}" --ngram-only-layer 1
# The same triple with a STATIC address. a4 vs a1 is the static arm's own overhead/benefit with
# zero downstream boundaries, i.e. the control that says how much of a5/a6's gain is the address.
run a4-l5-sta "${GEO[@]}" --ngram-only-layer 5 --ngram-gather static
run a5-l3-sta "${GEO[@]}" --ngram-only-layer 3 --ngram-gather static
run a6-l1-sta "${GEO[@]}" --ngram-only-layer 1 --ngram-gather static
echo "=== $(date -u +%H:%M:%S) done ===" | tee -a "$OUT/summary.txt"
