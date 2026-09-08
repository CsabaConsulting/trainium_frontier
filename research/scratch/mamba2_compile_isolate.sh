#!/usr/bin/env bash
# Lever 2, experiment A: is the torch.compile anomaly on the Mamba2 path caused by
# the _SSDNKI torch.autograd.Function island, or by the mixer's ordinary ops?
#
# The 2x2 {compile, no-compile} x {nki-ssd, no-nki-ssd} at the REAL locked shape
# (4 ranks, lnc=2), plus a transformer control. Short runs; the only number wanted
# is the steady-state per-step dt, which train.py prints every 10 steps.
#
# WHY re-measure the two "known" cells: the previously-quoted compiled-Mamba2 figure
# (2.17 s/step) appears ONLY in two research/scratch docstrings -- it is not in
# research/experiment-log.jsonl and not in trn2-experiment-results.md. Unverifiable
# numbers get re-measured, not trusted.
#
# Run on the box from the repo root:
#   bash research/scratch/mamba2_compile_isolate.sh 2>&1 | tail -40
set -u

STEPS=${STEPS:-40}
OUT=${OUT:-/tmp/l2iso}
mkdir -p "$OUT"

run() {
  local label="$1"; shift
  echo "=== $label ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 600 --eval-tokens 65536 \
    --out-dir "out/l2iso-$label" "$@" > "$OUT/$label.log" 2>&1
  echo "exit=$?"
  grep -E "^step 000(1|2|3)0 |^step 0003[0-9] " "$OUT/$label.log" | tail -4
  grep -E "^(training_seconds|num_steps|num_params_M|compiled|causality_passed):" "$OUT/$label.log"
}

# The two cells that carry the whole claim
run m2-compile-nki    --mamba2 --nki-ssd
run m2-compile-eagerscan --mamba2 --no-nki-ssd
run m2-eager-nki      --mamba2 --nki-ssd --no-compile
run m2-eager-eagerscan --mamba2 --no-nki-ssd --no-compile
# controls
run xf-compile
run xf-eager          --no-compile

echo
echo "======== steady-state dt summary ========"
for f in "$OUT"/*.log; do
  lbl=$(basename "$f" .log)
  dts=$(grep -oE "dt [0-9]+\.[0-9]+s" "$f" | tail -3 | tr '\n' ' ')
  ts=$(grep -E "^training_seconds:" "$f" | awk '{print $2}')
  ns=$(grep -E "^num_steps:" "$f" | awk '{print $2}')
  printf "%-24s last_dt: %-28s training_s=%-8s steps=%-5s\n" "$lbl" "$dts" "${ts:-NA}" "${ns:-NA}"
done
