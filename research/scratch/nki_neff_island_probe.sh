#!/usr/bin/env bash
# Decisive test of the "NKI kernels are separate non-fusable NEFF islands under
# torch.compile" claim: compile the REAL graph at the REAL locked shape (LNC=2)
# once per dispatch path and count the NEFFs the runtime actually emits.
#
#   many NEFFs  -> the kernels really are separate islands (the premise holds)
#   few NEFFs, same count both ways -> they are already inside the compiled graph
#
# Run on the box from ~/trainium_frontier with the venv active.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

run_case() {
  local tag="$1"; shift
  local dir="out/neffprobe-${tag}"
  rm -rf "$dir"; mkdir -p "$dir"
  echo "=================================================================="
  echo "CASE: ${tag}   extra args: $*"
  echo "=================================================================="
  NEURON_RT_INSPECT_ENABLE=1 \
  NEURON_RT_INSPECT_DEVICE_PROFILE=1 \
  NEURON_RT_INSPECT_OUTPUT_DIR="$dir" \
  NEURON_RT_VISIBLE_CORES=0-1 \
  NEURON_LOGICAL_NC_CONFIG=2 \
  timeout 2400 python3 -u train.py --compile-only "$@" \
    > "${dir}/compile.log" 2>&1
  local rc=$?
  echo "exit=${rc}"
  grep -E "nki_custom_ops|nki_rope_norm|nki_softcap_ce|torch.compile enabled|WARNING|compile-only synthetic" \
      "${dir}/compile.log" | head -10
  local n
  n=$(find "$dir" -name '*.neff' 2>/dev/null | wc -l | tr -d ' ')
  echo "NEFF_COUNT[${tag}] = ${n}"
  find "$dir" -name '*.neff' -exec ls -la {} \; 2>/dev/null | awk '{print "   ", $5, $9}' | sort -k2
  if [ "$rc" != "0" ]; then
    echo "--- tail of failing log ---"; tail -30 "${dir}/compile.log"
  fi
  echo
}

run_case "autogradfn" --no-nki-custom-ops
run_case "customop"   --nki-custom-ops

echo "=================================================================="
echo "SUMMARY"
for tag in autogradfn customop; do
  printf "  %-12s NEFFs=%s\n" "$tag" "$(find out/neffprobe-${tag} -name '*.neff' 2>/dev/null | wc -l | tr -d ' ')"
done
