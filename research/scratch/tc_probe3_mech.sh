#!/usr/bin/env bash
# Track: throughput->capacity, Probe 3 -- MECHANISM test for why the rope+norm NKI kernel went
# net-negative, and the clean price of the n-gram VE tables.
#
# THE OBSERVATION TO EXPLAIN. The two NKI kernels together are worth:
#   +21.1% on the stripped model  (ab-nofeat 223,815 vs ab-plain 184,825 tok/s; 0.1236 s/step saved)
#    +2.4% on the promoted stack  (ab-full 156,917 vs ab-nokernels 153,150 tok/s; 0.0206 s/step saved)
# and the rope+norm kernel ALONE has flipped from a documented +15-18% (when written, eager stack) to
# **-1.35%** (ab-norope 159,043 > ab-full 156,917). So the kernels' saving did not merely shrink as a
# FRACTION of a longer step -- it shrank 6x in ABSOLUTE seconds when the features were added back.
# A fixed per-step launch saving cannot do that. Something the features add is making the kernels
# actively expensive.
#
# HYPOTHESIS. The promoted `--ngram-ve-layers "1,-1"` injects an n-gram gather at layer 1, whose
# dynamic address forces LNC=2 to RE-SHARD the residual/V stream for every layer downstream (this is
# the same mechanism already recorded for why n-gram cost scales with depth). An NKI kernel island is
# opaque to that resharding decision, so the compiler must materialise a specific shard layout at
# each kernel boundary rather than keeping the stream in whatever layout the surrounding graph wants.
# That cost exists ONLY when resharding is in play -- i.e. only once the n-gram gather is there.
#
# PREDICTION, which is what these two arms test. With `--no-ngram-ve` (VE and local_conv left ON, so
# only the n-gram tables and their gather go away), the rope+norm kernel should be POSITIVE again:
#   mech-ng-off-kern-on  should be FASTER than  mech-ng-off-kern-off.
# If instead --no-nki-rope-norm is still faster with the n-gram gone, the resharding story is wrong
# and the kernel is simply obsolete against the current compiler -- an equally useful answer, and the
# one that would make retiring it unconditional rather than contingent.
#
# WHY THE MAIN TRACK NEEDS THIS EITHER WAY: it decides whether "retire the rope+norm kernel" is a
# permanent simplification or a fact about the CURRENT n-gram layer choice that must be re-measured
# the next time --ngram-ve-layers moves.
#
# SECOND THING THESE ARMS BUY, for free: mech-ng-off-kern-on is the first CLEAN price of the n-gram
# VE tables (ab-nofeat also dropped VE and local_conv, so it could not isolate them). That number is
# the exchange rate for the whole track's real conclusion -- the capacity purchase that DID pay off
# was embedding-table capacity, not width -- so it is worth having exactly.
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
BEST=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze
      --ngram-ve-layers "1,-1" --ngram-ve-table-mult 128)
run() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/tc-probe-$name" \
    --eval-seq-len 1024 "${BEST[@]}" "$@" > "research/run-logs/tc-probe-$name.log" 2>&1
  grep -E "num_params=" "research/run-logs/tc-probe-$name.log" | head -1
  grep -E "^step 000[4-9]0 |^step 001[01]0 " "research/run-logs/tc-probe-$name.log" | tail -8
  grep -iE "Traceback|Error|out of memory" "research/run-logs/tc-probe-$name.log" | head -3
}
run mech-ng-off-kern-on   --no-ngram-ve
run mech-ng-off-kern-off  --no-ngram-ve --no-nki-rope-norm
echo "=== done ==="
