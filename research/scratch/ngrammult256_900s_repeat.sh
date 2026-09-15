#!/usr/bin/env bash
# Same-day paired repeat of ngrammult256_900s.sh: the first pair's step-normalised delta
# (-0.0022 bpb) sits in the 0.001-0.003 band that the standing noise-floor rule requires a repeat
# for before trusting. Same discipline: fresh control bracket, arm at the same pre-scaled
# --max-train-seconds (1106s, from the same calibration) to hold the step match.
set -u
cd "$(dirname "$0")/../.." || exit 1
run() {
  local label="$1"; local secs="$2"; local desc="$3"; shift 3
  echo "=== $(date -u +%H:%M:%S) $label (max-train-seconds=$secs): $* ==="
  python research/run_experiment.py --label "$label" --solution R1.2 \
    --description "$desc" --num-steps 5000 --max-train-seconds "$secs" \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}

run ngm256-ctrl-900s-r3 900 \
  "same-day repeat block: fresh same-commit 900s control, flag at locked default (mult=128)"

run ngm256-arm-900s-r2 1106 \
  "same-day repeat block: --ngram-ve-table-mult 256 ON (dim unchanged), same 1106s pre-scaled max-train-seconds as the first pair, confirming the -0.0022 bpb step-normalised delta before trusting it" \
  --ngram-ve-table-mult 256

run ngm256-ctrl-900s-r4 900 \
  "same-day repeat block: closing fresh same-commit 900s control"
echo "=== $(date -u +%H:%M:%S) done ==="
