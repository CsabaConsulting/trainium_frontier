#!/bin/bash
set -u
cd /work
test_command='env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -S /work/reference_test.py'
printf '%s\n' "$test_command" > /work/run-command.txt
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -S /work/reference_test.py > /work/test.stdout.txt 2> /work/test.stderr.txt
test_exit=$?
printf '%s\n' "$test_exit" > /work/test.exitcode.txt
printf 'exit_code=%s\n' "$test_exit" >> /work/run-command.txt
exit "$test_exit"
