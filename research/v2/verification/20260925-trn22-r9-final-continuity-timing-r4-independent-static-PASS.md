# Independent trn22 r2 final-continuity timing r4 review — STATIC PASS ONLY

Reviewed Implementer commit `384eb822c12c05cd994e98e81d2fe4f632cb6118`
on `origin/codex/20260925-trn22-r9-final-continuity-timing-r4-a764` in a
separate sparse worktree. I did not implement this package. I read the
repository contract, README, current handoff, experiment protocol and
decision log in their required order, then the workflow instructions,
current controller brief, runbook, pinned r2 writer and its independent HOLD.
This review involved **no SSH/EC2 contact, host write, source stage, native
run, evaluation, cleanup, breaker reset, AWS action or submission**.

**Verdict: PASS for the exact disabled, read-only source package only.**
Its checked-in contract has `dispatch_authorized: false`. The separate
`TIMING_REQUEST.ENABLED.json`, `INDEPENDENT_TIMING_REVIEW.PASS.json`,
`SUPERVISOR_TIMING_ASSIGNMENT.json` and fixed raw result are all absent.
This receipt is not the named final review of a future enabled request and
does not assign or permit a timing attempt. The underlying r2 source-stage
writer remains independently **HOLD**.

## Checked evidence

| Exact subject | SHA-256 |
| --- | --- |
| Disabled contract | `d42b304b71c7bce075efe393a8de6f17cb0decbc2ae8ccdac255f33362a16fc6` |
| Runbook | `7096835fb9f294c27f4600fd32b28944a5750db72e63e2a6952b84d450ea2557` |
| Local collector | `0def60cc66c280ec2043fb6209e7b91a640fab212b361f50e957ca3a10ee1a67` |
| Remote diagnostic | `09b811018356dca32e52c82db38840f31626e53005f596d222ed3560c7b0f0fe` |
| Supplied tests | `68232940f32d70e98e9639281cab27f84b8dfa42f9a227aab7006ed18368a240` |
| Pinned r2 writer | `2b84ef670811b3f0015b30e619cb3ec5b23e914326cdd3e74523fdc5b68ee7a7` |
| Independent tests | `a1a4cbf122182d14020a6ac7095c845306ba1e019018dad1878d0de3d0182902` |

The commit adds only five files under the new timing package. It does not
change `train.py`, `prepare.py`, data, evaluation or the r2 writer. Its
disabled contract's pinned collector, diagnostic and r2-writer hashes match
the actual bytes. `git diff --check` passes. The local `_authorize()`
fails on the absent enabled request before SSH identity examination, result
creation or host contact; an existing exclusive result prevents replay.

The diagnostic's measured final sequence matches the held r2 writer's
`_final_no_write_continuity()` ordering: activity; two broad label scans;
direct STOP/bundle/target checks; monotonic age at a **clock-only**
`would_open` point. The diagnostic uses read-only, no-follow opens for the
existing cooperative lock and source hashing. Its child commands are
read-only `git` checks, `fuser`, `neuron-ls` and `find`; there is no target
open, writer call, file creation/deletion or arbitrary process signal.
The SSH command uses pinned identity/host-key checks, `/dev/null` stdin,
isolated Python, and a bounded diagnostic-owned timeout. Each complete
trial is flushed as canonical JSONL before the next begins, so a later
timeout retains the completed strict prefix in the exclusive local receipt.
An incomplete timing or failed final probe is failed closed.

Commands rerun independently:

```text
python3 -B -m unittest -v test_timing_readonly.py             14/14 PASS
python3 -O -B -m unittest -v test_timing_readonly.py          14/14 PASS
python3 -B -m unittest -v research/v2/verification/test_20260925_trn22_r9_timing_r4_independent.py     6/6 PASS
python3 -O -B -m unittest -v research/v2/verification/test_20260925_trn22_r9_timing_r4_independent.py  6/6 PASS
```

Independent witnesses cover an absent authority before SSH/output, the
inclusive 1-second edge and one-nanosecond miss, a target appearing after
both broad scans, retention of completed timing records when the final
probe fails, one-attempt local result exclusivity, and the absence of
remote filesystem-write primitives. These are local synthetic checks,
not a live trn22 timing measurement.

## Boundaries that remain

The child timeouts are shorter than the writer's; a timeout is inconclusive,
not evidence of an unsafe or safe writer latency. A measured `would_open`
clock is **not** the actual r2 `O_EXCL`, so even twenty sub-second trials
would not close the r2 independent HOLD for late process/Neuron/disk changes
or the check-to-open scheduling gap. A future distinct enabled read-only
request needs its own exact independent review and one-shot Supervisor
assignment before an Operator may perform fresh local and host checks.
Any r3 stage repair needs its **separate** independent source review.
