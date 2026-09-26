# Sequential source-bound reproduction

Run one command at a time, only after the Supervisor grants the laptop's
single heavy-process slot. This folder's source is frozen414cdd76. No SSH,
device call, training-data read or evaluator import is used by these checks.

Working directory for the commands below:
`/home/poojith-devan/Desktop/Trainium`.

## Existing CPU suite on this exact source

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/lanes/tables-runtime-select/test_fp32_reference.py
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/lanes/tables-runtime-select/test_tables.py --parent /home/poojith-devan/Desktop/Trainium/research/v2/lanes/k2-compiler-local-constants/train.py --repo /home/poojith-devan/Desktop/Trainium --index-dtype int32
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/lanes/tables-runtime-select/test_tables.py --parent /home/poojith-devan/Desktop/Trainium/research/v2/lanes/k2-compiler-local-constants/train.py --repo /home/poojith-devan/Desktop/Trainium --index-dtype int64
```

## Real exact-production table compiler

The isolated environment is the existing K2 diagnostic environment, Python
3.12.13, NKI0.5.0+28631259367.ga768afa6 and
neuronx-cc2.26.6360.0+6f180f47. No package is installed on a host or into the
competition runner. Compatibility libraries are the earlier local-only
Ubuntu extracts. The script enforces2GiB address space/900CPU seconds and
one-thread library settings. Compiler subprocesses inherit those limits.

```bash
PATH=/home/poojith-devan/Desktop/Trainium/research/v2/analysis/20260926-k2-parser-failure/.venv/bin:$PATH LD_LIBRARY_PATH=/home/poojith-devan/Desktop/Trainium/research/v2/analysis/20260926-k2-full-local-compile/scratch/compat/root/usr/lib/x86_64-linux-gnu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 900 /home/poojith-devan/Desktop/Trainium/research/v2/analysis/20260926-k2-parser-failure/.venv/bin/python -B research/v2/lanes/tables-runtime-select/check_backend.py compile-production
```

This uses the exact factory's authenticated `CompileKernel._compile_opts()`;
`NEURON_CC_FLAGS` inside the diagnostic process is optlevel1/auto-cast matmult
BF16 and target/LNC are Trn2/2. The real compiler command and PASS/FAIL are
retained in the artifact log. The wrapper exposes no nonproduction raw-options
mode. Source hashes, versions, production shapes, policy, timing and NEFF hash
are in unique timestamped JSON receipts. Compilation does not run the NEFF.

## Official NKI simulator, not custom CPU instruction emulation

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 300 /home/poojith-devan/Desktop/Trainium/research/v2/analysis/20260926-k2-parser-failure/.venv/bin/python -B research/v2/lanes/tables-runtime-select/check_backend.py simulate
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 300 /home/poojith-devan/Desktop/Trainium/research/v2/analysis/20260926-k2-parser-failure/.venv/bin/python -B research/v2/lanes/tables-runtime-select/check_backend.py simulate --fixture sparse
```

Both retain the entire524288x128 output and compare against independent
token-ordered FP32 reference accumulation. Equality is exact numerical array
equality, not a separate signed-zero bit assertion. These are simulator
checks, not native timing, distributed communication or optimizer evidence.
