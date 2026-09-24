# Independent verifier receipt: trn22 c5c A0 r2 pending final request

Reviewed 2026-09-24 23:33 UTC by a Verifier independent of the proposal
ratifying Supervisor, retry-package Implementer, and staging Operator. Scope
is the immutable *pending* top-level request, the completed-proposal addendum,
and archived gate evidence. I made no host connection or host mutation, and
did not launch native training/evaluation, reset a breaker, alter AWS, adopt a
model, or submit to the leaderboard.

## Verdict

**PASS exact request identity and completed-proposal linkage; DISPATCH HOLD.**
The pending request is a faithful top-level materialization of the previously
reviewed c5c A0 r2 stage. This review clears the separate final-request
*static* gate only. It is neither a live pre-dispatch receipt nor one-attempt
Supervisor authorization, and no new BPB/control result exists.

Operator source commit `8e6c58243e9e5cfe67dbec77b9d01b032c75d8db`
(integrated without content change as `bceae17e42061718b5c4588cc8d32c2090788cae`)
adds only two JSON evidence files; `git diff --check` passed. The request is
`research/v2/authorizations/fs-20260925-v2-c5c-a0-control-trn22-r2.request.pending.json`,
SHA-256 `7c4d1af77fe8c332170feebf444e0c292ef8d78ff3bf76d5ef3056533e2279cc`.
The separate run-log-parent observation is
`research/v2/authorizations/fs-20260925-v2-c5c-a0-control-trn22-r2.runlog-parent-readonly.json`,
SHA-256 `ebc2b6abc01330770d615b0573938a6a1b43de1dd5f275d045992be69ea438a6`.
Both parse as JSON and the only added top-level request keys beyond the
reviewed `launch_template` are `authorization` and
`external_boundary_observer`.

## Independent comparison

- I compared every key/value of the pending request's launch core against
  `launch_template` in the independently reviewed staged wrapper
  `fs-20260925-v2-c5c-a0-control-trn22-r2.request.staged.json`, SHA-256
  `8a22424d58afe1826ec6973f8ff9d294235ce8e6c499699d00b58dbd44528f0a`:
  **all fields equal**. The external observer object is also deeply equal to
  the staged observer. No source, command, effective environment, warm-cache,
  output, step-log, host, label, disk floor, or timeout field drifted.
- The command remains bounded by `timeout 3600s`, begins with `env -i`,
  contains LNC2/full-chip four-rank `torchrun`, compiler optlevel-1 flags,
  seed 42 and max-train-seconds 1800, and ends in the exact Track-A c5c A0
  argv. Its source is pinned to commit
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c`, root `train.py` SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and unchanged `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The prior staged-core review checked all six launcher/guard blob hashes,
  exact G2 argv/environment, and all six warm-cache paths; none changed here.
- The separate Supervisor ratification is
  `research/v2/authorizations/20260925-trn22-c5c-a0-r2-proposal-ratification.json`,
  commit `d0ed1997f281808da9146e729af66a9ceda6fcc8`, SHA-256
  `a5de803e0e138c56b10f6cbca39c3c1050f527e8956953ba6f8095649b8f1be1`.
  I parsed it, checked its one-file diff, verified it binds the unchanged
  draft-proposal SHA, this staged-wrapper SHA, the same source/argv/four science
  environment values/unique r2 label, and the independent static, G1,
  exact-environment four-rank CPU G2, disk-breaker, source-stage and staged-
  core receipts by exact SHA. Its explicit
  `proposal_complete_for_this_one_control=true`, historical-draft-gates
  explanation, rationale, success/abort bounds and
  `launch_authorized=false` make the hash-bound draft plus addendum a
  completed selected proposal **for final-request review only**. The pending
  request binds this addendum's exact SHA; it did not mutate the reviewed
  draft or grant dispatch.
- The pending request's authorization block also binds the exact stage-core
  independent-review SHA
  `6bfe09889c3022497a46711952211a5bb01560a1e5a052f92024d5453368d5b7`,
  Operator stage-preflight SHA
  `0ad7051c6e6eb543b95271f763eec8a4dbd8e580805b3266332364a71a7f9a06`,
  trn22 G1 calibration receipt SHA, CPU G2 r3 numerical-review SHA, human
  disk-breaker acknowledgement/review SHA, and the new parent observation
  SHA. The r1 native label and r2 CPU G2 label remain consumed; the science
  r2 label has no ledger result. The request contains no fresh public
  evaluation, checkpoint pruning, cache cleanup, adoption or submission
  authority.
- The read-only parent observation reports that the clean staged checkout's
  `research/run-logs` directory was absent at 23:30:28 UTC; the Operator did
  not create it. I independently inspected the exact pinned
  `fresh_start_launch.py`/`fresh_start_host.py` order: launcher preflight
  writes `<label>.preflight.json` via `write_json`, which creates the parent,
  **before** launcher manifest/job/log creation and guard/child spawn. Thus
  this normal launch path avoids the earlier direct-CPU-G2 missing-parent
  failure. If creation/write fails, the launcher cannot reach child spawn.
  This is a static proof, not a native launch result; no manual mkdir is
  approved by this receipt.

## Non-negotiable dispatch hold

The top-level file now **is executable by the known launcher if someone passes
its path**. I verified the pinned `fresh_start_launch.py` does not read
`dispatch_authorized` or `authorization`; both `false` values are procedural
metadata, **not** a mechanical interlock. Do not copy or pass this file to the
launcher merely because it passed static review. The workflow-v2 state still
reports `launch_authorized=false`, `hosts.trn22.launch_allowed=false`, and
ready scored jobs 0 at this review; it does not change as a consequence of
this receipt.

Before a run, an Operator must repeat the exact source/tool/prepare and
proposal/request hashes, clean checkout, retained checkpoint and six cache
device/inode identities, STOP/breaker checks, unique r2 path absence,
hostname, memory/inodes, no train/eval/guard/device/host-lock holder, and root
free space at least **10,737,418,240 bytes** immediately before dispatch.
The read-only external observer (script SHA-256
`a6f437fe8a9d0a14ec3025bc822f1afa12e3f355fca730e4ae0db7ae00263592`)
must start and be proven alive while the step file is absent **before** the
guard. A continuous root-free-space monitor must be bound to the authenticated
owned guard and stop it before the immutable **6,979,321,856-byte** operational
floor. A distinct Supervisor must then issue a written, exact-request-SHA,
this-review-SHA, fresh-preflight-SHA, one-label/one-attempt authorization;
only an Operator may execute it once. Any failed condition returns to HOLD.

After training, full logs, five sequential observed optimizer boundaries,
finite model-only checkpoint, budget compliance and clean quiescence must be
archived before a separately reviewed fresh public20M evaluation. A0 is a
control, not an improvement. This receipt makes no score or submission claim.
