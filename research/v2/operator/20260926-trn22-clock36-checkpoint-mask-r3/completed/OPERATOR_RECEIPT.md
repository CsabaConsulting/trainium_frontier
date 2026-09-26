# Checkpoint-only r3: completed, independent result review pending

Exact assignment dcb8afb0; frozen packet 1ada51bd. One unchanged reviewed
mask-aware helper ran against the 16 pinned retained r2 inputs. No G2 rerun,
model-source change, checkpoint rewrite, cold/native/full training or eval.

Dispatch 2026-09-26 02:31:00.049030 UTC; helper launch 02:31:01.469640 UTC.
Preceding host-idle interval to helper launch was 15m27.123s, so the 15-minute
incident was real and reported. Terminal exit 0 at 02:31:55.620297 UTC with
`CPU_CHECKPOINT_ONLY_COMPLETE_PENDING_INDEPENDENT_REVIEW`. Original r2 remains
`FAILED_PRESERVED`; native launch remains unauthorized.

All four reference/default/OFF/ON ordinary unpatched loaders produced their
three synthetic outputs with byte-identical repeated calls and unchanged model
state. Each synthetic prefix comparison had max absolute difference 0. The
helper records 113,256,724 parameters and 65 states, validates full/short masks
at restored loader sequence length 8,192, and separately retains the historical
BF16 reference/default/OFF identity comparison. No ON/OFF equivalence or quality
claim is inferred. Independent acceptance is the Supervisor's responsibility.

All 16 inputs-after hashes exactly equal inputs-before. Original four trained
checkpoints stayed on the host unchanged. Stderr contains only the preserved
standard Neuron gather-registration warning. Full raw result SHA-256:
`d2c0c8ee06c97421e8a36f689348f3cf02a06f45ca6439d70892ddeddea83e13`.
Completed 21-file bundle SHA-256:
`25375c47e3629c0b2cf0f56d02b4dc96156dcf35be97055f0ad91765d6ccdc9d`.

Explicit postflight at 02:34:47.605284 UTC: all five actually observed
dispatch/flock/driver/timeout/helper PIDs absent; their live starttime, PPID,
PGID and SID retained. Full current census has no related workload/compiler
processes, no device holders, and idle neuron-ls. Original common lock
66305:832589 unchanged and free; both retired queue PIDs absent. No kills,
cleanup, reset or successor launch. Postflight SHA-256:
`c46119ba31d16ef5fa5c0053d4f21966a785886a01fb1a6d77f86a7fab9f0740`.

New idle threshold: 02:46:55.620297 UTC. The consumed r3 label must not be retried.
