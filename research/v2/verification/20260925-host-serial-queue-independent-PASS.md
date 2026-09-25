# Independent operational review: host serial queue successor

Verdict: **PASS for Operator deployment of the verified-file flag-only queue.**

Reviewed at 2026-09-25 20:56 IST. Candidate:
`2630351211e477a7028a9b34ff7f5b489fcb77fd`, successor to `51ea5c4d`.

Exact `workflow/tools/host_queue.py` SHA-256:
`47a2d5ea1fabdc9c18c8f03585dac37fd7923cfe51ce8596cb0d130183ad0cd8`.
Exact `workflow/tools/test_host_queue.py` SHA-256:
`564c86cadeca4640fa4e956a22cbe9e4586be04f1a357f5e0764737cc4c483f6`.

This supersedes the single freeze-boundary HOLD in
`20260925-host-serial-queue-independent-HOLD.md` (review commit `405355d7`).
The independent Verifier reviewed the complete successor diff and reran all
eight supplied tests. All passed, and `git diff --check 51ea5c4d 26303512`
passed. The new regression crosses the freeze during a real local fake
training subprocess: job 1 completes its fresh evaluation and durable
`evaluated_pending_review` receipt; job 2 never starts. The repair moves only
the freeze check to new-job/slot-acquisition/prelaunch guards. Source, STOP,
disk and quiescence checks remain active during completion.

The prior review's passing evidence carries forward unchanged: common
train-plus-eval lock and real parent-SIGKILL inheritance test; no consumed-label
relaunch; exact proposal/source/evaluator/argv pins; both real A0 train/eval log
parsers; all twelve actual unconditional proposal SHA/bindings and six-job
per-host schema checks; legal four-rank/LNC2, seed-42, 1,800-second command;
unchanged fresh public-20M/context-1024 evaluation; durable errors;
PAUSE_NEXT between jobs; no secrets, cleanup, host SSH, AWS or submission code.

No host was contacted and no runner or scientific code was edited by this
Verifier. This is operational source review, not a claim of host installation,
live job launch, scientific causality validation or BPB adoption. The Operator
fills actual host paths and existing warm cache roots, uses the exact common
`/home/ubuntu/.cache/trainium_freshstart/device0.lock`, stages pinned proposals
and source, and performs the documented deployment/machine preflight. The
runner itself then waits for the current A1/A2 train-and-eval holder to finish.
No further independent prelaunch review is required for the already-verified
file's twelve flag-only science arms under the controlling Final Push directive.
