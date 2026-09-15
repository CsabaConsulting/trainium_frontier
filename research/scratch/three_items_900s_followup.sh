#!/usr/bin/env bash
# Follow-up to three_items_900s.sh. The pure byte-feature wte init came in +0.00215 bpb at an exact
# step match against a control pair whose own spread was 0.000148 -- i.e. ~15x the pair spread, but
# still under the 0.003 "needs a paired repeat" threshold. Instead of a bare repeat, run the
# DOSE-RESPONSE: mix 0.5 and 0.25 blend the byte features with the normal_(0,0.8) init as
# mix*feat + sqrt(1-mix^2)*random. With mix 1.0 = +0.00215 and mix 0.0 = the control by definition,
# a monotone-in-mix penalty confirms the mechanism itself is harmful (and confirms the mix-1.0 read),
# while a flat or non-monotone pattern would say the mix-1.0 number was noise.
set -u
cd "$(dirname "$0")/../.." || exit 1
run() {
  local label="$1"; local desc="$2"; shift 2
  echo "=== $(date -u +%H:%M:%S) $label : $* ==="
  python research/run_experiment.py --label "$label" --solution R1.2 \
    --description "$desc" --num-steps 5000 --max-train-seconds 900 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}
run ti-bytewte-mix050-900s "byte-feature wte init at mix 0.5 (half byte features, half random, row RMS preserved) -- dose-response point for the mix-1.0 +0.00215 read" --byte-wte-init --byte-wte-init-mix 0.5
run ti-bytewte-mix025-900s "byte-feature wte init at mix 0.25 -- second dose-response point" --byte-wte-init --byte-wte-init-mix 0.25
echo "=== $(date -u +%H:%M:%S) done ==="
