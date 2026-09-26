# Serial local evidence plan — not executed at static handoff

Run from `/home/poojith-devan/Desktop/Trainium` only after the Supervisor
explicitly releases the one local heavy-process slot. Wait for each command
to exit before starting the next. Do not launch a second model/compiler or
simulator process in parallel. Stop and preserve the actual error if a test
fails; do not silently edit this frozen source.

CPU suite, exact accepted e1e2 base, both index dtypes, OFF3 and ON3:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 180 /home/poojith-devan/venv/bin/python -B research/v2/lanes/k2-runtime-groups/test_k2.py --base /home/poojith-devan/Desktop/Trainium/workflow/candidates/confirmed-clock36-116188/train.py
```

Optional separate parser diagnostic (the backend also executes the parser;
this is not an extra launch gate):

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 90 research/v2/analysis/20260926-k2-parser-failure/.venv/bin/python -B research/v2/lanes/k2-runtime-groups/test_parser.py
```

Real production backward compilation, exact authenticated existing SDK:

```sh
env PATH="/home/poojith-devan/Desktop/Trainium/research/v2/analysis/20260926-k2-parser-failure/.venv/bin:$PATH" LD_LIBRARY_PATH=/home/poojith-devan/Desktop/Trainium/research/v2/analysis/20260926-k2-full-local-compile/scratch/compat/root/usr/lib/x86_64-linux-gnu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 900 research/v2/analysis/20260926-k2-parser-failure/.venv/bin/python -B research/v2/lanes/k2-runtime-groups/check_backend.py compile-production
```

Official simulator fixture 1, full production shapes, four collisions per row:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 300 research/v2/analysis/20260926-k2-parser-failure/.venv/bin/python -B research/v2/lanes/k2-runtime-groups/check_backend.py simulate
```

Official simulator fixture 2, full production shapes, 4096 untouched rows,
eight collisions per active row:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 300 research/v2/analysis/20260926-k2-parser-failure/.venv/bin/python -B research/v2/lanes/k2-runtime-groups/check_backend.py simulate --fixture sparse
```

`check_backend.py` limits address space to 2 GiB, CPU time to 900 seconds,
library threads to one, target Trn2/LNC2 and flags to `--optlevel=1 --auto-cast
matmult --auto-cast-type bf16`. Real metadata shapes are in SPEC.md. Runtime
scratch goes only in this lane's ignored scratch/pip-tmp directories. No SDK
installation, model/data download or system change is part of these commands.

Each check writes a source-bound JSON receipt. Archive successful compiler
outputs with the exact returned receipt filename (not the literal placeholder):

```sh
/home/poojith-devan/venv/bin/python -B research/v2/lanes/k2-runtime-groups/archive.py --receipt research/v2/lanes/k2-runtime-groups/compile-production-result-<actual-time-ns>.json
```

Existing native framework-forward evidence remains accepted. Local compile and
simulation do not measure cold whole-model startup, native exactness, throughput
or BPB. The one actual native timing stays Supervisor-selected/Operator-owned.
