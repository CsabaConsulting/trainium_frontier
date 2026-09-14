#!/usr/bin/env bash
# Hardware smoke test of the dead-code-cleanup train.py (7267 -> 5191 lines), run on box 2 against
# this track's OWN same-day control so the comparison needs no cross-day assumption.
#
# WHY THIS IS NOT REDUNDANT WITH THE CPU BIT-EXACTNESS CHECK. The cleanup was verified bit-exact on
# CPU torch (identical loss, param/grad md5s, optimizer groups). That proves the MATH is unchanged. It
# does not prove the COMPILED GRAPH is unchanged, and on this stack that is a real distinction: this
# track just measured a 9.4-point throughput swing in one NKI kernel caused purely by the presence of
# one gather elsewhere in the graph, and a 10.2% SLOWDOWN from deleting 537M parameters. Graph-level
# effects here are large, non-additive, and not predictable from the math. So the question "does
# removing 2,076 lines change s/step or step-110 loss on device?" has to be measured.
#
# REFERENCE, from tc_probe3_ablate.sh + tc_probe3_repeat.sh on the pre-cleanup binary
# (md5 d0995051e0375541d6f9fc71abecf465), same box, same flags, 8-point means of steps 40-110:
#   ab-full x3 : 156945 / 155789 / 156235 tok/s  -> pooled 156323, spread 0.74%
#   step-110 loss: 3.9380 / 3.9380 / 3.9412
# PASS = tok/s inside that control spread and step-110 loss inside the +/-0.0034 120-step noise floor.
# A loss delta materially outside that floor means the cleanup was NOT behaviour-neutral on device and
# the submission binary must not change.
set -u
cd "$(dirname "$0")/../.." 2>/dev/null || cd ~/trainium_frontier
BEST=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze
      --ngram-ve-layers "1,-1" --ngram-ve-table-mult 128)
for name in clean-full-r1 clean-full-r2; do
  echo "=== $name ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps 120 --max-train-seconds 1800 --out-dir "out/tc-probe-$name" \
    --eval-seq-len 1024 "${BEST[@]}" > "research/run-logs/tc-probe-$name.log" 2>&1
  grep -E "num_params=" "research/run-logs/tc-probe-$name.log" | head -1
  grep -E "^step 001[01]0 " "research/run-logs/tc-probe-$name.log"
  grep -iE "Traceback|Error|out of memory" "research/run-logs/tc-probe-$name.log" | head -3
done
echo "=== done ==="
