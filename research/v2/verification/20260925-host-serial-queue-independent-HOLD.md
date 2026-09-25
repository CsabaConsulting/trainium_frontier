# Independent operational review: host serial queue

Verdict: **HOLD — one reproduced freeze-boundary defect.**

Reviewed at 2026-09-25 20:54 IST by a Verifier separate from the Implementer and
Operator. Candidate: `51ea5c4d9efa008bd48c64c67dcb65382793db1b`.
`workflow/tools/host_queue.py` SHA-256:
`7aa69c05d892f1e190e53cacde9a0dd65ec48b8db9141374137788d60e0f81a7`.
No host contact, launcher/model edit, cleanup or AWS action occurred.

## Reproduced defect

`QueueRunner.guards()` (line 203) rejects the research freeze on every call.
`phase()` calls `wait_idle()` after a successful training process, which calls
that same guard. Therefore a run already training when the freeze arrives is
marked failed before its mandatory fresh evaluation can start. Its successful
checkpoint remains unscored. The operational requirement is to enforce the
freeze between jobs, while completing/archive-evaluating an in-flight job.

Reproducer, using only the supplied local fake runner:

```python
import datetime
from test_host_queue import QueueTests

t = QueueTests()
t.setUp()
try:
    t.cfg["fake_sleep"] = 0.5
    t.cfg["freeze_utc"] = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(seconds=0.2)
    ).isoformat()
    runner = t.runner()
    try:
        runner.run()
    except RuntimeError as error:
        print(str(error))
    print((runner.root / "job-1/training-exit.json").exists())
    print((runner.root / "job-1/fresh-public20m-launch.json").exists())
    print(t.terminal("job-1")["status"])
finally:
    t.tearDown()
```

Observed: `RESEARCH_FREEZE`, `True`, `False`, `failed`.
No second job started.

Minimal requested repair: apply the freeze gate while waiting for and before
starting a new job; retain source, STOP, disk and quiescence guards during an
existing job, but allow its training-to-evaluation-to-archive sequence to finish
after the freeze. Add one local regression proving job 1 is evaluated and job 2
does not start when the freeze arrives during job 1.

## Checks that passed

- All seven supplied tests pass via
  `python3 -m unittest discover -s workflow/tools -p test_host_queue.py -v`.
- The real SIGKILL test proves the surviving phase retains the common device
  lock; restarting does not overlap it or relaunch a consumed label. The source
  holds the lock across training, evaluation and durable receipt creation.
- Both currently running manual A1/A2 drivers use the documented
  `/home/ubuntu/.cache/trainium_freshstart/device0.lock` path. Operator manifests
  must use that exact path, as documented.
- Every one of the twelve actual unconditional proposals in bundle `00facd13`
  has a matching recorded SHA and exact `queue_binding` (host, label, source,
  argv, kind, arm and base). The twelve explicit argv arrays satisfy the
  seed-42, full-budget, context-1024, isolated-output requirements.
- Both archived real A0 training logs and public-20M evaluation logs parse
  successfully. Trn21: 3,675 steps, 0.9914739939934344; trn22: 3,633 steps,
  0.991513958330512. Both evaluations record 20,971,520 tokens and no timeout.
- Training uses four ranks/LNC2 with seed 42 and 1,800 seconds, exact c5c source
  bytes and exact unchanged evaluator bytes. Fresh evaluation uses context
  1024/public-20M in a separate process. No shell-evaluated argv is introduced.
- STOP/disk/source/idle checks, prelaunch receipts, durable phase exit/error
  logs, completed-label deduplication and PAUSE_NEXT between jobs are present.
  The runner reports only `evaluated_pending_review`; it does not adopt or
  silently insert scientific verdicts.
- No secrets are inherited into the explicit child environment; no cleanup,
  host SSH, AWS mutation, submission, or evaluator change exists in this patch.
  No obsolete flag-only prelaunch independent-review ceremony is reinstated.

This HOLD concerns only the reproduced freeze defect. It is not a request for
additional unused abstractions, staging checks or repeated scientific review.
Once the narrow repair and regression pass, the Operator still supplies the
actual host paths/STOP paths, validates the manifest and performs deployment
preflight; this review does not itself deploy or start anything.
