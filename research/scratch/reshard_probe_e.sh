#!/usr/bin/env bash
# 2026-09-15 -- RESHARD HUNT, Block E: what the layer-1 injection penalty actually IS, now that
# Block A has proved it is NOT the dynamic address.
#
# BLOCK A's RESULT (4 ranks, LNC=2, one bigram table, mult 128 / dim 128, identical params):
#   position   fwd (dyn)  fwd (static)  micro (dyn)  micro (static)
#   layer 5      53.4 ms      53.0 ms     156.6 ms      156.7 ms
#   layer 3      54.3 ms      54.6 ms     156.2 ms      157.6 ms
#   layer 1     138.1 ms     137.5 ms     240.3 ms      239.9 ms
# So (i) the DYNAMIC ADDRESS is worth 0.0 +/- 0.6 ms in every position -- the entire LNC=2
# resharding mechanism documented on 2026-09-12 is a ONE-RANK artefact and does not exist at the
# competition shape; and (ii) there is nevertheless a large, real, +83.7 ms/micro (+53%) penalty
# for injecting at layer 1 rather than layer 5, with identical parameters and identical bytes.
#
# THE HYPOTHESIS THIS BLOCK TESTS. Probe 3 Part D (2026-09-14) found that the hand-written
# rope+QK-norm NKI kernel INVERTS in the presence of a layer-1 n-gram injection: worth +7.92%
# without it, -1.34% with it, a 9.4-point swing. The stated (and now falsified) reason was
# resharding making each NKI island a forced materialisation point. The observation survives its
# explanation: an early injection and an NKI island interact badly. If the kernel is the CARRIER
# of the layer-1 penalty, then `--no-nki-rope-norm` should collapse the 84 ms, and the layer-1 /
# layer-5 gap should close. If the gap survives with the kernel off, the penalty is intrinsic to
# the injection position and the kernel is innocent.
#
# 2x2, same instrument and same isolation as Block A (one bigram table, `--ngram-only-layer`, so
# params/bytes/optimizer/all-reduce are identical in all four arms and only the position and the
# kernel toggle move).
set -u
cd "$(dirname "$0")/../.." || exit 1
source /home/ubuntu/workspace/native_venv/bin/activate
OUT=${OUT:-research/run-logs/reshard-e}
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
  grep -E "\[drv\] (profiled window|micro dts)" "$OUT/$name.log" | tail -3 | tee -a "$OUT/summary.txt"
  grep -iE "Traceback|Error|out of memory" "$OUT/$name.log" | head -3 | tee -a "$OUT/summary.txt"
}

# The 2x2. e1/e3 reproduce Block A's a1/a3 (kernel ON, its shipped default) as same-session
# controls, so the kernel-off deltas are paired rather than cross-session.
run e1-l5-kon "${GEO[@]}" --ngram-only-layer 5
run e2-l5-koff "${GEO[@]}" --ngram-only-layer 5 --no-nki-rope-norm
run e3-l1-kon "${GEO[@]}" --ngram-only-layer 1
run e4-l1-koff "${GEO[@]}" --ngram-only-layer 1 --no-nki-rope-norm
# One more suspect, free to add: the softcap+CE kernel is the OTHER NKI island in the graph. If
# the carrier is "any NKI island x early injection" rather than the rope kernel specifically,
# removing this one should also move the layer-1 arm.
run e5-l1-nosc "${GEO[@]}" --ngram-only-layer 1 --no-nki-softcap-ce
run e6-l1-nokern "${GEO[@]}" --ngram-only-layer 1 --no-nki-rope-norm --no-nki-softcap-ce
echo "=== $(date -u +%H:%M:%S) done ===" | tee -a "$OUT/summary.txt"
