#!/bin/bash
# Sequential CPU-only reproduction. Work directory must be fresh for exclusive artifacts.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python3 -B /work/scripts/prepare.py > /work/raw/prepare.stdout 2> /work/raw/prepare.stderr
python3 -B /work/scripts/review_test.py --output /work/raw/attempt-001 > /work/raw/attempt-001.stdout 2> /work/raw/attempt-001.stderr
python3 -B /work/scripts/run_attempt.py --name attempt-002-optimized --script /work/scripts/review_test.py --optimized
python3 -B /work/scripts/run_attempt.py --name attempt-003-boundaries --script /work/scripts/boundary_test.py
python3 -B /work/scripts/finalize_report.py
