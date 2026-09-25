# Independent trn22 r10 timing r3 review — HOLD before any enabled request

Reviewed 2026-09-25 by a Verifier separate from the Implementer. Exact
disabled subject: `f7fbaed75df2761f1c66f5858ce1a359c9da61eb` on
`origin/codex/20260925-trn22-r10-stage-timing-r3-disabled`. **HOLD for
final enablement**: an inherited late-STOP bypass remains in the timing
collector. The earlier source-only timing PASS at `e741c0bb` did not detect
this issue and must not be used as timing-dispatch authority. No SSH/host/
EC2 contact, marker, output receipt, transfer, checkout, cleanup, training,
evaluation, AWS action or submission occurred in this review.

## Verified exact binding and bounded source properties

The timing package pins corrected stage commit
`b3ec7406f0e7a3df6848698876e2106305dd283f`. Its locally included
stage files are byte-identical to that commit: operator SHA-256
`a5fc2ebac3520f425d5ab0bdfb7ae0091a2e5db59b152dd8cd39eb67cff5beda`,
remote SHA-256
`da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`,
and disabled request SHA-256
`48e3ad5d0bb1d077dbf8304e3ddfe7e3263892e522dc277f32812eea5c8e111a`.
The timing collector, remote diagnostic and HOLD contract hash respectively
to `f16b609508aefadd9228a8688b56b6e63bfcda99f5746d3a6ab687f7a3743f5a`,
`c43fbe1224de291c31a797079ed9708db4ca7fa0d7fe785b3ff67274c835f3ca`
and `4918f2cebc86c3934b57a020f9bc87035be6c983c6f4c222bb22d3e9461a2db8`.
The contract has dispatch, remote writes, target open, process signals and
training/evaluation/AWS permissions all false. The remote diagnostic's
functional changes from `f91f24d9` are stage-pin constants only; it uses
read-only source/lock opens, checks, and stdout JSONL, not the stage writer.
The collector checks active Desktop controller state and Desktop, Documents
and package STOP paths on ordinary preflight/timing contacts.

The offline suite passed **20/20 tests** normal and **20/20 tests** under
`python3 -O`; `git diff --check` passed. Direct normal and `-O` entrypoints
both exited 1 at the absent enabled request before SSH or local output.
No enabled request, independent final timing review or Supervisor assignment
is checked in. The twenty-trial timing model has **not** run on trn22. Its
clock-only would-open cannot measure actual `O_EXCL` scheduling/syscall delay,
payload/fsync, same-UID races or full source-stage feasibility.

## Blocking late-STOP witness

`operator_timing_readonly.py:331-334` calls `_local_gates()` only when
`postflight=False`. After timing starts, `collect_once()` unconditionally
calls `_run_ssh("probe", postflight=True)` at lines 574-579. Consequently a
new Desktop, Documents or package STOP can be bypassed before the post-probe
SSH connection, contradicting `RUNBOOK.md:32`'s claim that local STOP can
prevent that connection. A no-host mock witness made `_local_gates` raise
`synthetic STOP` and `_ssh_identity` raise `identity reached`: ordinary
`postflight=False` invoked the gate once and identity zero times; with
`postflight=True`, it invoked the gate **zero** times and identity once.
This is a safety-gate defect even though that post-probe is read-only and
any later failure would leave the result classified HOLD.

Smallest repair: run `_local_gates()` before **every** SSH phase, including
the post-probe; retain a late-STOP failure as an incomplete postflight and
terminal local HOLD. Add a regression that places a STOP after the timing
phase and proves post-probe reaches neither SSH identity nor subprocess.
Then rebind the changed collector/contract hashes and obtain a fresh
independent exact-source review. Separate enabled timing authority, live
checks and Supervisor assignment remain required.

The reviewed timing/source branches have `train.py` SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `prepare.py` SHA-256
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The active Desktop controller checkout instead has `train.py` SHA-256
`52ac9e6560a40592255e3499f4d42ac6e7073ea82da1b428e03dc984e6d4baa8`.
Any future source/timing execution must use an exact clean reviewed c5c
package worktree while reading the active Desktop controller state, not
assume an archival copy in the Desktop checkout has the reviewed science
identity. This HOLD grants no staging, native or BPB authority.
