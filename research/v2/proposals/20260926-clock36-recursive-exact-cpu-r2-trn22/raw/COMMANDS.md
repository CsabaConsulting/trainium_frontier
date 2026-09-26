All commands ran locally, with no source/model/native/data imports. No full-size CPU run occurred.

Runtime: /home/poojith-devan/venv/bin/python; Torch 2.13.0+cpu.
For each of normal and optimized modes (-O only in the latter):

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python [-O] test_checkpoint_helper.py
    /usr/bin/python3 [-O] test_binding.py
    /usr/bin/python3 [-O] test_r2_scope.py

Working directory: this packet directory. Raw helper stdout/stderr and complete static test output are retained separately.
All six commands exited 0. No failed r2 helper attempt was omitted; none occurred.
No candidate source, optimizer update, G2 full model, hardware training or organizer causality evaluation was executed here.
