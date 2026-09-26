All review commands ran with working directory
`/tmp/trainium-k2-registration-review-GS6xFb/native-packet`.
Runtime: `/home/poojith-devan/venv/bin/python`, Python 3.14.4. Tests are stdlib-only;
no Tensor or native module was loaded, and no host command/device operation ran.

```bash
set -o pipefail
/home/poojith-devan/venv/bin/python -I -B native-review-output/review_validator.py 2>&1 | tee native-review-output/validator-normal-attempt1.log
```

Exit 1: positive synthetic archive accepted; four false accepts among 30 mutations.

```bash
set -o pipefail
/home/poojith-devan/venv/bin/python -I -B native-review-output/review_mechanics.py 2>&1 | tee native-review-output/mechanics-normal-attempt1.log
```

Exit 0: static binding plus 12 driver, nine idle and seven inherited-lock mocks pass.

```bash
set -o pipefail
/home/poojith-devan/venv/bin/python -I -B -O native-review-output/review_validator.py 2>&1 | tee native-review-output/validator-optimized-attempt1.log
```

Exit 1: same four false accepts with Python assertions disabled.

```bash
set -o pipefail
/home/poojith-devan/venv/bin/python -I -B native-review-output/review_checker.py 2>&1 | tee native-review-output/checker-normal-attempt1.log
```

Exit 1. Source at execution is retained as `review_checker_attempt1.py`.

```bash
set -o pipefail
/home/poojith-devan/venv/bin/python -I -B native-review-output/review_checker.py 2>&1 | tee native-review-output/checker-normal-attempt2.log
```

Exit 1. Source at execution is retained as `review_checker_attempt2.py`.
Both harness errors and the test-only correction are in `FAILED_ATTEMPTS.md`.

```bash
set -o pipefail
/home/poojith-devan/venv/bin/python -I -B native-review-output/review_checker.py 2>&1 | tee native-review-output/checker-normal-attempt3.log
```

Exit 0: corrected mock argv handling; seven orchestration tests, safe nonfinite
comparison, signed-zero distinction and route-error recording pass.

```bash
set -o pipefail
/home/poojith-devan/venv/bin/python -I -B -O native-review-output/review_mechanics.py 2>&1 | tee native-review-output/mechanics-optimized-attempt1.log | rg '"kind": "(authenticated|static_binding|mechanics_summary)"|Traceback'
```

Exit 0.

```bash
set -o pipefail
/home/poojith-devan/venv/bin/python -I -B -O native-review-output/review_checker.py 2>&1 | tee native-review-output/checker-optimized-attempt1.log | rg '"kind": "(authenticated|helper_source_identity|disabled_checker|checker_summary)"|Traceback'
```

Exit 0. The displayed traceback text belongs to an intentionally mocked and
correctly recorded old-route exception, not an unhandled test failure.

Raw `mock-checker-*.log` files contain complete checker output under each virtual
scenario. The synthetic JSON baselines/counterexamples contain no native results.
Their source generators and duplicate optimized evidence are retained as emitted,
including JSON files without a trailing newline. Future re-execution should use
a fresh output directory because evidence files are created exclusively.
