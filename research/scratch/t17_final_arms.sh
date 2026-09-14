#!/usr/bin/env bash
# The two candidates for a new locked config, both built on the item-3 winner (n-gram VE on layer 1
# AND the last layer) and both step-matched just under the ~2195-step data wall.
#
# Where the numbers come from: t17-l1-2150 ran 2139 steps in 1687 s = 0.78868 s/step measured on the
# real 1800s loop (not the 120-step instrument), and scored 1.007376. It left 113 s of the 1800 s
# budget unspent, which is 46 steps of headroom before the wall. Two ways to spend it:
#
#   l1-2185     1723 s of pure extra steps.  2139 -> 2185 steps is worth -0.077*ln(2185/2139)
#               = -0.0016 bpb at the measured slope. Free, small, and the arithmetic is not in doubt.
#   l1-m128     spend it as n-gram table WIDTH instead (mult 64 -> 128, 592.9M params). The 120-step
#               instrument puts this at 0.832 s/step (157499 tok/s) vs 0.7848 for l1 alone, so the
#               FULL 1800 s budget lands at ~2153 steps -- no shortening needed at all. Its 120-step
#               loss proxy is 3.9343 vs l1's 3.9402; at the proxy-to-real ratio layer 1 itself
#               showed (-0.0206 proxy -> -0.0118 real, 0.57x) that is worth about -0.0034 bpb,
#               against a +0.0012 penalty for the 2185 -> 2153 step loss. Net prediction ~-0.002,
#               i.e. INSIDE the 0.003 threshold that requires a paired repeat -- which is why it is
#               run head to head against l1-2185 on the same day and the same commit rather than
#               compared to yesterday's number.
set -u
cd "$(dirname "$0")/../.." || exit 1
BUDGET=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze --ngram-ve-layers "1,-1")
run() {
  local label="$1"; local secs="$2"; local desc="$3"; shift 3
  echo "=== $(date -u +%H:%M:%S) $label (${secs}s) : $* ==="
  python research/run_experiment.py --label "$label" --solution T1.7 \
    --description "$desc" --num-steps 5000 --max-train-seconds "$secs" \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 \
    "${BUDGET[@]}" "$@"
}
run t17-l1-2185 1723 \
  "n-gram VE on layers 1,-1 + T1.7 compiled Muon + frozen qk-shift, budget spent to land at ~2185 steps (just under the data wall)"
run t17-l1-m128 1800 \
  "n-gram VE on layers 1,-1 with table mult 128 (592.9M) + T1.7 + frozen qk-shift, full 1800s budget lands at ~2153 steps" \
  --ngram-ve-table-mult 128
echo "=== $(date -u +%H:%M:%S) done ==="
