# Independent trn21 r10 source-stage r3 active-root review — static PASS only

Reviewed 2026-09-25 by an independent Verifier. Exact disabled subject:
`7205abf5469edc616c4e4b58caee27bc1957bbd4` on
`origin/codex/20260925-trn21-r10-incremental-stage-r3-active-root`, a
descendant of r2 `3e84658933cf7f1c9758df894dd57fb265aba366`.
**PASS for the narrow active-controller-root source correction only.** This
does not convert the prior r2 static PASS into final stage authority. No
SSH/EC2/host contact, marker, transfer, checkout, cleanup, training,
evaluation, breaker reset, AWS action or submission occurred.

The r2 operator read breaker/launch state from the stale Documents checkout,
as recorded in the independent r2 HOLD addendum at
`research/v2/verification/20260925-r10-stage-controller-state-active-root-independent-HOLD.md`.
The exact r3 `operator_once.py` now reads that state only from
`/home/poojith-devan/Desktop/Trainium/research/v2/state.json`; its local
STOP scan covers Desktop, Documents and the package worktree. The Desktop
state read is in `_local_stop_and_breakers()`, called before `_ssh_identity()`
and before marker creation, and again before each SSH phase. The new
regression sets a stale clear Documents state and a tripped Desktop state;
the attempted preflight raises HOLD without identity inspection or a
subprocess. This resolves the specific stale-state risk for these r3 bytes.

The diff from r2 changes only `REQUEST.HOLD.json`, `RUNBOOK.md`,
`SHA256SUMS`, `operator_once.py` and `test_incremental_stage.py`. The remote
writer and bundle are byte-identical, as are the source, target, label and
one-attempt marker constants. The checked-in HOLD request has dispatch,
source-stage, native-training and evaluation authorization all false;
enabled request, final review and Supervisor assignment files are absent.
Direct normal and `-O` entrypoints each exit `HOLD: no enabled source-stage
request, final review or assignment` before SSH or marker.

| Exact r3 package item | SHA-256 |
| --- | --- |
| `operator_once.py` | `6ecad35400199b7c1cba41c78d5d4f14363d37180c40e8aab5e9b485846db74f` |
| `REQUEST.HOLD.json` | `4c48243b7a5bee3f5aa6333e8ec5e81e5d7d434d926cbce18af907bc5ddfdd7c` |
| `RUNBOOK.md` | `250ddb1a0c2d9da453fa27bd5301e2f6ed6a0d9c7eb0ebf7f9512313be1e41b3` |
| `SHA256SUMS` | `5a6551fd878a2e87f0b5ec4ac2421f96b0e72f0d121ad866808f54d670ea4e91` |
| `test_incremental_stage.py` | `4d23b06b984d1226d4fd33ee08dc47dafde3f0e3d7e458bfa0a18129f84bc304` |
| unchanged `remote_incremental.py` | `47e2c28bd6fd735904398a48d6c91ef95b93454cef9558b121c26b0ee5b66a0b` |
| unchanged `e65-to-r10.bundle` | `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e` |

`sha256sum -c SHA256SUMS` passed 7/7. The unchanged bundle verified with
e65 `e65a3097a5b00f6178188a935bb2acb9a5042c6c` as sole prerequisite
and r10 `22bfbab5a77d6feb670d2d737504f8b70b08ddc1` as sole HEAD.
Offline package tests passed **27/27** normal and **27/27** under `-O`;
`git diff --check 3e846589 7205abf5` passed.

This is not a fresh host preflight, a native validation, a BPB result or a
final enablement PASS. The r2 source-stage review's bounded noncooperating
race and live one-second feasibility limits remain. Any final stage attempt
requires a separate exact enabled request, independent final review,
Supervisor one-attempt assignment, fresh host checks and all protocol
conditions. No same-HEAD scored run is authorized by this receipt.
