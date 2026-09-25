# Independent trn22 r10 timing r4 review — static PASS, dispatch HOLD

Reviewed 2026-09-25 by a Verifier separate from the Implementer. Exact
disabled subject: `cab7110373e0f09b97f83adf7688de951008127d` on
`origin/codex/20260925-trn22-r10-stage-timing-r4-disabled`. **PASS for the
disabled read-only source only**: this exact correction closes the late local
STOP bypass in the independent r3 HOLD `f5780f81`. This is not an enabled
timing request, live timing result, source-stage authority, native run or BPB
result. No SSH/host/EC2 contact, marker, output receipt, transfer, checkout,
cleanup, training, evaluation, AWS action or submission occurred.

The sole executable change from r3 is that `_run_ssh()` calls `_local_gates()`
before **every** new SSH phase, including `postflight=True`. The gate checks
STOP under active Desktop, former Documents and package roots, then the
active Desktop trn22/global breaker and host launch HOLD. A late refusal is
caught by `collect_once()`, leaves the retained local result failed closed,
and cannot be represented as a completed post-probe. I independently used
synthetic STOPs at each of the three roots, plus a tripped Desktop breaker,
while mocking SSH identity and subprocess: every postflight call raised
before identity or contact (**0 identity, 0 subprocess** in all four cases).
The package's separate mid-run regression also leaves exactly two synthetic
pre/timing contacts, no third post-probe, and terminal `FAILED_CLOSED`.

The exact package binds source-stage r3
`b3ec7406f0e7a3df6848698876e2106305dd283f`. Its included stage
operator, remote and HOLD request are byte-identical to that commit, with
SHA-256 values `a5fc2ebac3520f425d5ab0bdfb7ae0091a2e5db59b152dd8cd39eb67cff5beda`,
`da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`
and `48e3ad5d0bb1d077dbf8304e3ddfe7e3263892e522dc277f32812eea5c8e111a`.
The timing collector, unchanged remote diagnostic and disabled contract hash
to `90660ab0892ecb091f9dc852cea46f4d59f76a3365c46451381d9c78d6ba4ef8`,
`c43fbe1224de291c31a797079ed9708db4ca7fa0d7fe785b3ff67274c835f3ca`
and `09086904ae8ab1cba54e4c1fb35476a90c5654d7a8692a61eb167cd8bddd6bfc`.
The runbook hash table matches these bytes. The contract keeps dispatch,
remote writes, archive target open, process signals and training/evaluation/
AWS permission false. Remote code remains read-only; it neither invokes the
stage writer nor opens a target/bundle for creation. The only possible
`O_EXCL` write is a local raw receipt after separate authority.

Offline package tests passed **22/22** normal and **22/22** under `-O`;
`git diff --check` passed. Both direct entrypoints exited 1 at the absent
enabled request before SSH or output creation. The 20-trial clock-only model
has **not** run on trn22 and cannot measure actual `O_EXCL` delay, payload,
fsync or same-UID races. A future timing dispatch still requires a distinct
exact enabled request, independent final timing review, Supervisor assignment
and fresh live checks; source staging remains a separate gated operation.

Finally, the reviewed source worktree has `train.py` SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `prepare.py` SHA-256
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The active Desktop controller checkout has different `train.py` bytes
(`52ac9e6560a40592255e3499f4d42ac6e7073ea82da1b428e03dc984e6d4baa8`).
Any future execution must use an exact clean reviewed c5c package worktree
while reading the active Desktop controller state, not assume an archival
Desktop copy has the reviewed science identity. This PASS grants no host or
scored-run authority.
