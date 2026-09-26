# Executed bounded CPU and composition commands

Working directory `/home/poojith-devan/Desktop/Trainium`. The local-heavy slot
was explicitly assigned by root; both CPU processes ran sequentially and
exited0. Do not run concurrently with another laptop compiler/model process.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 180 /home/poojith-devan/venv/bin/python -B research/v2/lanes/tables-compact-k2/test_tables.py --parent /home/poojith-devan/Desktop/Trainium/research/v2/lanes/k2-runtime-groups/train.py --repo /home/poojith-devan/Desktop/Trainium --index-dtype int32
```

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 180 /home/poojith-devan/venv/bin/python -B research/v2/lanes/tables-compact-k2/test_tables.py --parent /home/poojith-devan/Desktop/Trainium/research/v2/lanes/k2-runtime-groups/train.py --repo /home/poojith-devan/Desktop/Trainium --index-dtype int64
```

Then the lightweight stdlib-only binding check:

```sh
/home/poojith-devan/venv/bin/python -B research/v2/lanes/tables-compact-k2/check_composition.py
```

It checks exact factory bytes, bytes outside the factory, all other AST nodes,
fresh CPU source/test hashes and existing component receipt/artifact hashes.
It does not import Torch/NKI or run a compiler, simulator, native job or review.
Existing actual full-production compile/simulator receipts apply only to their
byte-identical components; no unnecessary new production execution is added.
