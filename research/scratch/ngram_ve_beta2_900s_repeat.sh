#!/usr/bin/env bash
# Same-day paired repeat for --ngram-ve-beta2 0.95, triggered by the standing noise-floor rule:
# the first pass (ngram_ve_beta2_900s.sh) found arm-095 WORSE than control by +0.00169 bpb, which
# is above the 0.001 noise floor but below the 0.003 threshold where a delta is trusted outright --
# i.e. squarely in the "needs a same-day paired repeat before trusting" band. This is also the
# surprising result (opposite sign from the EMBED_ADAMW_BETA2 precedent this was modeled on), so
# it gets the extra scrutiny rather than being taken on one pair.
#
# First-pass numbers (same box/session/commit):
#   ngmbeta2-ctrl-900s-r1   1063 steps  1.052188
#   ngmbeta2-arm-095-900s   1062 steps  1.053878   <- +0.00169 vs r1, +0.00155 vs r2
#   ngmbeta2-arm-099-900s   1064 steps  1.052365   <- +0.00011 vs mean control, clear noise
#   ngmbeta2-ctrl-900s-r2   1063 steps  1.052331   (control pair's own spread: 0.000143)
set -u
cd "$(dirname "$0")/../.." || exit 1
run() {
  local label="$1"; local desc="$2"; shift 2
  echo "=== $(date -u +%H:%M:%S) $label: $* ==="
  python research/run_experiment.py --label "$label" --solution T3.1 \
    --description "$desc" --num-steps 5000 --max-train-seconds 900 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}

run ngmbeta2-ctrl-900s-r3 \
  "fresh same-commit 900s control, same-day repeat pair for the --ngram-ve-beta2 0.95 result"

run ngmbeta2-arm-095-900s-r2 \
  "--ngram-ve-beta2 0.95, same-day repeat of the first-pass arm that measured +0.00169 bpb worse than control" \
  --ngram-ve-beta2 0.95
echo "=== $(date -u +%H:%M:%S) done ==="
