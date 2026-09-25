#!/usr/bin/env bash
# Sequential reproduction; keeps any earlier run's raw evidence.
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
cd /work
run_dir=$(mktemp -d /work/artifacts/replay.XXXXXX)
cp exactness_test.py "$run_dir/test.py"
python reconstruct.py > "$run_dir/reconstruct.stdout" 2> "$run_dir/reconstruct.stderr"
python audit_receipts.py > "$run_dir/receipt_audit.stdout" 2> "$run_dir/receipt_audit.stderr"
python audit_source.py > "$run_dir/source_audit.stdout" 2> "$run_dir/source_audit.stderr"
set +e
python exactness_test.py > "$run_dir/exactness.stdout" 2> "$run_dir/exactness.stderr"
test_rc=$?
set -e
cp artifacts/exactness_cases.json artifacts/exactness_failures.json artifacts/exactness_summary.json artifacts/extractions.json "$run_dir/"
printf '%s\n' "$test_rc" > "$run_dir/exit_code.txt"
cat "$run_dir/exactness_summary.json"
exit "$test_rc"
