# Reproduce bounded tiny checks (not a host launcher)

Run only in an explicitly free shared laptop CPU slot. No full113M model on this laptop. Existing Torch runtime /home/poojith-devan/venv/bin/python is used; no install/download. OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS are each1, and PYTHONDONTWRITEBYTECODE=1.

TINY_CPU_PLAN.json records exact per-arm source pins and argv. Invoke the unchanged workflow/tools/preflight_dryrun.py from this worktree with --stub-prepare --steps3 --nproc4 --micro-rows1 --micro-tokens16 --timeout120 --keep, --train-py=<arm source> --out=<arm receipt>, then -- and that arm's exactargv. Arguments are separate tokens (for example --steps 3). PYTHONPATH is the packet test_support; it cannot read real Parquet. The harness itself injects its existing stub-prepare path first and records every CPU/fake-time/synthetic-input shim.

Run reference, default, explicit --no-recursive-schedule, then --recursive-schedule sequentially, never concurrently. Stop on any failure and retain rawreceipt/tempworkingdirectory. Do not treat a missing report as permission to retry or allocate a bigger model. No local test authorizes any host launch.

After all four, run python3 -B research/v2/lanes/architecture-recursive-stack/compare_tiny_receipts.py. It requires all12 losses and all65 finalrank0parameterdigests plus ownershipplan exactly match default/OFF to exact3e79. Every arm must pass its cross-rank update/reload checks. It reports ON parameter differences without requiring them as a speed/quality gate. Tests change compute/synthetic shape only through the already reviewed harness; production train.py is never rewritten.

Separate production gate remains exact8x1024/113256724parameters, all4ranks/3synthetic2x256steps, trusted prepare and sufficientRAM host, original common flock plus freshps/neuron-ls and independent binding/Operator assignment. This file does not provide a new host runner or permission.
