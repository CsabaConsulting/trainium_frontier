# Independent trn22 r10 source-stage assignment check — local PASS only

**Verdict:** the exact Supervisor assignment is valid for **one trn22 r10
source-stage attempt only**, subject to a fresh Operator preflight. This
point-in-time local check is **not** a host preflight or a source-stage result.
Native training and evaluation remain HOLD. I did not call `stage_once`, the
operator entrypoint, SSH, any host action, or the marker writer.

## Subject and exact authority

- Subject: clean designated science worktree
  `/home/poojith-devan/Documents/ChatGPT/Trainium-trn22-r10-stage-r4-stop-recheck-r1`
  at `62e9cba75624fcb94c263086dc60d91fcafe4ffc`. Its sole change from
  independently reviewed `ac012821` is the assignment JSON below; no
  operator, remote source, bundle, science, or evaluator byte changed.
- Assignment:
  `research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/SUPERVISOR_ONE_ATTEMPT_SOURCE_STAGE_ASSIGNMENT.json`,
  exact 691-byte canonical sorted JSON plus newline, SHA-256
  `ed07f024e45b5a606c4c8ed3f0ad6bc335a6bcaead548bf89c1e0cbcd2ec5bb4`.
  It binds `trn22`, the fixed r10 stage-r1 label and marker, `one_attempt=true`,
  `native_training_authorized=false`, and the preserve-even-empty/no-cleanup-
  or-retry policy.
- Enabled source-stage request SHA-256
  `d19086a3f0c8aff9835f14f641d528980b12b21f45310070f6613ae261a33958`;
  independent final `PASS_SOURCE_STAGE_ONLY` review SHA-256
  `c9be026c2609fbbdcfdc85005630675959a2b6661befb91268dd4cf73d59d615`.
  The reviewed local `operator_once.py` is SHA-256
  `af888c89dbed4360895e5cf93299095efe3280893dafd1e504cd88b6a9c4408f`,
  unchanged remote source is
  `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`,
  and the 109,800-byte bundle is
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
- A local, non-dispatch call to `operator_once._authority(...)` accepted the
  actual request, review, assignment, remote and operator bytes. The exact
  assignment scope is
  `TRN22_R10_INCREMENTAL_SOURCE_STAGE_R1_ONLY_ONE_ATTEMPT`; the request also
  explicitly sets `evaluation_authorized=false`.

## Local gates observed at verification

- The designated worktree was clean at the pinned HEAD. Root `train.py`
  SHA-256 was
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  unchanged `prepare.py` SHA-256 was
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- The active Desktop controller's `research/v2/state.json` SHA-256 was
  `560daf07441133a14854bc108af484fb7a9c1f9f360379b830ca0bc85facb6d0`:
  `launch_authorized=false`, trn22 `launch_allowed=false`, and global/trn22
  breakers untripped. Active Desktop, former Documents controller, and this
  package worktree had no checked `research/agentic/STOP` or
  `research/v2/STOP`. The operator's read-only `_local_stop_and_breakers()`
  passed.
- The fixed r10 marker and all four prior r9 markers were absent by
  no-follow existence checks. Marker parent was owned by UID 1000 with mode
  `0700`. The operator's local `_ssh_identity()` verified the pinned
  known-hosts and host-key fingerprint without making an SSH connection.

These observations can become stale. The Operator must recheck local gates
and perform the complete fresh, read-only **host** preflight before consuming
the one-shot marker or attempting any source stage. The earlier trn22
clock-only timing test had just 19.642 ms of worst double-scan margin and did
not execute a real exclusive open; a safe timing refusal may consume the
one-shot marker. Preserve any provisional empty or nonempty artifact and do
not retry or clean it under this assignment. A successful stage still needs
independent result review and separate native authorization. The historical
`FINAL_PENDING_RUNBOOK.md` predates the review and assignment and does not
override these exact current authority files.
