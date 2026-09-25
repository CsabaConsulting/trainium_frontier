# Independent A2 tool-path read-only r2 review — static HOLD

Reviewed 2026-09-25 by a Verifier separate from the Implementer. Exact
disabled subject: `e4f523cc762dd384b5a2e4b2d1ef469324fb4677` on
`origin/codex/20260925-a2-toolpath-active-root-r2`, descending from r1
`ab978507`. **HOLD before any enabled capture** because post-capture
controller-state drift can leave a non-HOLD planning decision. No host/SSH/
EC2 contact, marker, local raw receipt, transfer, cleanup, training,
evaluation, AWS action or submission occurred in this review.

The r2 local gate does read breaker/launch state from the active Desktop
`research/v2/state.json` only; Documents state cannot enable contact. It
checks Desktop, Documents and package STOP paths before each SSH read, uses
no-follow regular-file metadata and stable before/after checks for the active
state, and rejects a tripped global/trn22 breaker or non-HOLD launch state.
The unchanged remote probe walks the two exact e65 tool paths with no-follow
directory descriptors and read-only opens; the decision requires fixed path,
UID, device, single link, exact SHA-256, clean e65 source and the 32-GiB/
10,000-inode/96-GiB planning floors. It does not create a remote file or
one-shot marker. A future authorized capture would write only an exclusive
**local** raw observation. The checked-in request keeps every authority flag
false; no enablement, independent review or Supervisor assignment is present.

The exact collector, unchanged remote probe and disabled request SHA-256 are
`0cc663ab1be7f7137b2715fab7aa4b2fb5c3f3327eb8bad6f33084dc5f75f93a`,
`51d5f6be8b313a51efc56609dd391610ee1b6528e9f118f95176679484ef83b7`
and `65e63ffa09213ea5831f5a2797378bc67a7b93acd215d7a519a9c976ec446dc1`.
The five pinned local request/transport/snapshot/Git-ACL/tool-source hashes
all passed `--check`. Offline tests passed **9/9** normal and **9/9** under
`-O`; `git diff --check` passed. Default `--capture` exited 3 with
`HOLD_NO_SUCCESS_CLAIM` in both modes before SSH or receipt creation.

## Blocking reproducible drift witness

`collect_readonly.py:350` saves `controller_before`, and lines 375-390 save
`controller_after`, but the code sets HOLD only when the second gate *fails*.
It never compares their `active_state_sha256` values. A no-host mock
simulated the two successful gates returning hashes `A` then `B` (both
breaker-clear/launch-false), with all three remote probes and the receipt
writer mocked. `capture()` preserved
`READ_ONLY_FEASIBLE_AT_OBSERVATION_TIME_ONLY` despite `A != B`. This
contradicts `RUNBOOK.md`'s statement that changed state after capture is
recorded as HOLD. A changed active controller state may represent new
authority/host facts even if its breaker bits remain clear.

Smallest repair: after a successful second gate, compare its state hash with
`controller_before`; any mismatch forces `planning_decision.status = HOLD`
with a distinct drift reason. Add a regression in which the active state
changes only a non-breaker field during capture and prove the final decision
is HOLD. Re-review the corrected exact collector and its enablement-byte
chain before any Operator contact. This receipt grants no A2 CPU-G2, native,
scored or BPB authority; the earlier A2 r4 HOLD remains controlling.
