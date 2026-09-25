# Independent offline assignment check — trn22 r10 r5 source stage

Checked 2026-09-25 11:46 UTC. Verdict: **PASS_ASSIGNMENT_ONLY** for
`80de62ef8779cb91e01c72a7e8620afc91396685` on
`codex/20260925-trn22-r10-remote-timeout-repair-r1-disabled`. This does not
approve a native training run, evaluation, cleanup, breaker reset, submission,
or a second stage attempt. I made no SSH or host contact and did not run the
stage entrypoint.

The subject adds only
`research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/SUPERVISOR_ONE_ATTEMPT_SOURCE_STAGE_ASSIGNMENT_R5.json`
relative to the independent r5 final-review commit `66de5da328c36c4597e4447a2eff0b1ddbe83a2d`.
Its SHA-256 is
`806c89ad2aa03cd706d5c8548c2862b0c8d0eb7e4389e5486b65f0e3859037bc`.
The request, review, and assignment are canonical sorted JSON. The assignment
binds request SHA-256 `25092856f0fef3c1e3cf85e40e45fa0d2d51971b7b8112f50dbc3761770971f2`
and independent r5 review SHA-256
`da237ac25e8547c7cc9b0cc6dbf1fec9e2875af6be94b7e407d17616181a0994`,
with the exact trn22 label, marker, one-attempt scope, `native_training_authorized=false`,
and `PRESERVE_EVEN_EMPTY_NO_CLEANUP_OR_RETRY` policy. The unchanged operator's
`_authority` equality gate accepts these exact bytes. Five offline mutations
(request hash, review hash, native escalation, repeat escalation, and scope
escalation) were each rejected.

The package manifest verified 14/14 entries. Exact retained hashes are
operator `6224763c0acdbba8bfadf4281498fe9801a8b8d4aaee3d37f8b58f2f97b412be`,
remote source `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`,
bundle `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`,
science `train.py` `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
and unchanged `prepare.py` `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
`git bundle verify` reports only e65 prerequisite and r10 target
`22bfbab5a77d6feb670d2d737504f8b70b08ddc1`. The 55 offline tests pass
under normal Python and optimized Python (one expected optimized skip).
`git diff --check` passes.

At this check's instant, the active Desktop controller has both relevant
breakers clear, global `launch_authorized=false`, and trn22
`launch_allowed=false`; active, former, and package STOP paths are absent.
The r10 one-attempt marker and four prior trn22 r9 marker paths are absent;
the marker parent is user-owned mode `0700`. These are **local point-in-time
observations**, not a substitute for fresh Operator preflight.

Remaining risks/gates: the live remote absolute GNU `timeout` identity remains
unknown; the previous read-only stage-timing model had only 19.642 ms worst
one-second margin, so an otherwise safe first attempt may fail closed and
consume the marker. The Operator must perform fresh local and host preflight,
then at most the exact one-shot source-stage command, preserve any provisional
bundle even if empty, retrieve the marker journal and distinct-session
readback/postflight, and obtain an independent result review. No native A0
training or BPB claim follows from this receipt.
