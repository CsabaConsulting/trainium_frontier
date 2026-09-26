All commands ran with working directory
`/tmp/trainium-k2-registration-review-GS6xFb`. Nothing was executed from the
production repository or another worktree. These are CPU review commands only.

The exact review command invocations, in execution order, were:

```bash
set -o pipefail
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/tmp/trainium-k2-registration-review-GS6xFb/output /home/poojith-devan/venv/bin/python -I -B output/review_cpu.py ordered 2>&1 | tee output/own-ordered-attempt1.log
```

Exit 1. At this invocation the source was the version now preserved as
`review_cpu_attempt1.py`. Exact error and subsequent test-only correction are in
`FAILED_ATTEMPTS.md`. The candidate adapter was never edited.

```bash
set -o pipefail
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/tmp/trainium-k2-registration-review-GS6xFb/output /home/poojith-devan/venv/bin/python -I -B output/review_cpu.py ordered 2>&1 | tee output/own-ordered-attempt2.log | rg '"kind": "(authentication|disabled|runtime|archived_decorator|validator_negatives|registrations|dtype_summary|summary)"|Error|Traceback'
```

Exit 0: 70 exact routes, including 28 registered gradients and 42 controls.

```bash
set -o pipefail
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/tmp/trainium-k2-registration-review-GS6xFb/output /home/poojith-devan/venv/bin/python -I -B output/review_cpu.py zero 2>&1 | tee output/own-zero-attempt1.log | rg '"kind": "(runtime|dtype_summary|summary)"|Error|Traceback'
```

Exit 0: expected poison refusal, 20 registered mismatches and 42 exact controls.

```bash
set -o pipefail
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/tmp/trainium-k2-registration-review-GS6xFb/output /home/poojith-devan/venv/bin/python -I -B output/review_cpu.py stale 2>&1 | tee output/own-stale-attempt1.log | rg '"kind": "(runtime|dtype_summary|summary)"|Error|Traceback'
```

Exit 0: expected poison refusal, 20 registered mismatches and 42 exact controls.

```bash
set -o pipefail
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/tmp/trainium-k2-registration-review-GS6xFb/output /home/poojith-devan/venv/bin/python -I -B output/review_context.py 2>&1 | tee output/own-context-attempt1.log | rg '"kind": "(authentication|disabled|context_identity|context_summary)"|Error|Traceback'
```

Exit 0: 12 exact routes from unchanged source-context Function AST.

```bash
set -o pipefail
/home/poojith-devan/venv/bin/python -I -B output/adapter.py 2>&1 | tee output/own-disabled-cli.log
```

Exit 0: disabled; launch_authorized false; Torch and NKI not imported.

```bash
set -o pipefail
/home/poojith-devan/venv/bin/python -I -B output/review_addendum.py 2>&1 | tee output/own-addendum-attempt1.log
```

Exit 0: all nine additions authenticated, two Function wrapper ASTs identical,
historical compiled-custom zero-gradient failure independently audited on all
four supplied ranks. This command imported no Torch or NKI.

Runtime: Python 3.14.4, `/home/poojith-devan/venv/bin/python`, Torch 2.13.0+cpu;
Torch intra-op and inter-op threads both set to one. CPU fixtures use V8/T8/C4
and FP32 weights/dW. FP32 and BF16 outputs/upstreams are tested. Largest explicit
test-created tensor is 17x4 FP32 (272 bytes), with tiny fixed graph operations;
test tensor allocations remain below 64 MiB. The Python/Torch runtime itself
is outside this tensor limit. No model, downloaded data, weights, device access,
network call, package installation, training or evaluation was performed.

Read-only source inspection additionally used `sha256sum`, `cat`, `sed`, `rg`,
`jq`, `wc`, and `diff -u` inside the packet. The two checker source files differ,
so their `diff -u` returned the normal difference exit code 1. It was not a CPU
test failure. Original input files and both manifests were not modified.
