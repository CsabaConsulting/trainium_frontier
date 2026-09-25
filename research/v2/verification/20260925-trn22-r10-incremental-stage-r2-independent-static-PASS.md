# Independent trn22 e65→r10 incremental source-stage r2 review — static PASS only

Reviewed 2026-09-25 10:01 UTC by a Verifier separate from the Implementer.
Exact disabled subject: `020bfa7ebd048f947d5207cbedaf028a29a4b159` on
`origin/codex/20260925-trn22-r10-stage-r2-pin-correction`, directly after
the held r1 subject `3d26e963df349f444cd560114400788bf572b062`.
The independent r1 HOLD is at `0f3f409927d371e63405f95de6c137f8adef5c1e`
(`research/v2/verification/20260925-trn22-r10-incremental-stage-r1-independent-static-HOLD.md`).

**Verdict: PASS for these exact disabled r2 source bytes only.** The r1
blocker—the malformed inherited-review digest—is corrected in
`operator_once.py`, `REQUEST.HOLD.json` and `RUNBOOK.md` to the actual
SHA-256 `5f1b18e45fda3e726ccde6063523ee2679f3a8dbc42480cc1e84936d1b78bb1d`.
I independently rehashed the r4 PASS receipt's bytes at claimed commit
`efd5f2396b8bb2c37865d3cb751f45ed2c07bb81`, confirmed equality with
the operator/request/runbook pins, and changed one final hex nibble in an
in-memory request copy: it no longer equals the actual receipt or the
operator's exact pin. The supplied regression asserts the corrected literal;
the independent actual-byte check closes the specific r1 review defect.

## Exact scope and checks

- The r1→r2 diff changes only the HOLD request, runbook, checksum list,
  operator's one pin constant, and its test. The `remote_incremental.py`
  Git blob (`ef1f9a6cce2ac86d138b79b10ac55560184d7152`) and checked-in
  bundle Git blob (`d462bbab0850e5a4ecd90c9ca9e013a72dda1b19`) are
  byte-identical to r1. The unique trn22 target, one-attempt label, local
  marker and four prior r9 marker checks are unchanged. Their **live**
  absence is not established by this static review.
- `sha256sum -c` passed for all eight package files. The direct 109,800-byte
  bundle retains SHA-256
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`;
  `git bundle verify` still reports sole prerequisite
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c` and HEAD
  `22bfbab5a77d6feb670d2d737504f8b70b08ddc1` (15 commits). The
  unchanged target retains `train.py` SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Supplied source-stage suites passed **36/36** in normal Python and
  **36/36** under `python3 -O`; `git diff --check` passed. The direct
  `operator_once.py` entrypoint under both interpreters exited 1 with
  `HOLD: no enabled source-stage request, final review or assignment`.
  No marker or SSH entry was reached. The pre-existing r4 boundary checks
  and no-unlink/no-retry behavior remain byte-identical: late persistent
  STOP, label, process/Neuron or disk failure is tested around the first
  exclusive open; even an empty failed provisional inode is retained.
  Failed postflight is recorded as HOLD rather than claimed as completed.
- No enabled request, independent final PASS or Supervisor assignment is
  checked in. `dispatch_authorized`, `source_stage_authorized`,
  `native_training_authorized` and `evaluation_authorized` remain false.

This PASS does **not** authorize host contact, a marker, source staging,
cleanup, native training, evaluation or submission. No such action occurred
in this review. The inherited non-atomic userspace-race limitations and the
unknown live ≤1-second timing feasibility remain. Before any host attempt,
require separate authorized read-only timing evidence and review, an exact
enabled request, independent final review, fresh host preflight and a
Supervisor one-attempt assignment. A successful stage then needs completed
cross-session readback/postflight and independent result/G2 review; native
A0 requires its own gates.
