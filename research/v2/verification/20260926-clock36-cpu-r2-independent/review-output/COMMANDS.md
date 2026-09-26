All outputs are within this review-output directory; no candidate/full-model import or remote action occurred.

Initial independent helper and binding attempt, exit 1 (missing neutral transport entries):

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B /tmp/trainium-clock36-cpu-r2-review-EB27oy/review-output/independent_test.py > /tmp/trainium-clock36-cpu-r2-review-EB27oy/review-output/initial.stdout.json 2> /tmp/trainium-clock36-cpu-r2-review-EB27oy/review-output/initial.stderr

After exact dependencies arrived, the script added dependency and full-context-diff matching checks and explicitly excluded the intentionally withheld narrative README. The independent helper fixtures did not change. Final normal attempt, exit 0:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B /tmp/trainium-clock36-cpu-r2-review-EB27oy/review-output/independent_test.py > /tmp/trainium-clock36-cpu-r2-review-EB27oy/review-output/final.stdout.json 2> /tmp/trainium-clock36-cpu-r2-review-EB27oy/review-output/final.stderr

Optimized repeat verifies rejection checks survive optimization, exit 0:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -O -B /tmp/trainium-clock36-cpu-r2-review-EB27oy/review-output/independent_test.py > /tmp/trainium-clock36-cpu-r2-review-EB27oy/review-output/optimized.stdout.json 2> /tmp/trainium-clock36-cpu-r2-review-EB27oy/review-output/optimized.stderr

Output hashes are in SHA256SUMS, excluding SHA256SUMS itself.
