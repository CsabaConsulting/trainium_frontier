Local-only commands, run sequentially from the isolated worktree; no host or full-model calls.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python [-O] -B research/v2/proposals/20260926-confirmed36-clean-checkpoint-cpu/test_helper.py
    /usr/bin/python3 [-O] -B research/v2/proposals/20260926-confirmed36-clean-checkpoint-cpu/test_scope.py

Normal and optimized forms each exited0 (83 tiny cases and153 static checks respectively).
The pending CLI was also run in each mode with the absolute local INPUTS.pending.json and its SHA; both exited1 with REAL_INPUTS_PENDING, before any scientific source/model import.
Raw outputs are retained separately. Prior fixture failures and interim passing checks are not rewritten as final evidence.
All children exited. Runtime Python3.14.4/Torch2.13.0+cpu, one thread, maximum108 elements per test tensor.
