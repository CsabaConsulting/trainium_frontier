# Independent final PASS — trn22 r10 source stage only

**Verdict: `PASS_SOURCE_STAGE_ONLY` for the exact clean Implementer subject
`1923a3a4e93d385a545fb40a6ddf0cf35aa3371c`** on
`origin/codex/20260925-trn22-r10-stage-r4-late-stop-recheck-r1`, reviewed
against prior independent final HOLDs `22b57441` and `24e00dfd`. The
collector-format canonical review is
`research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW.PASS.json`,
SHA-256 `c9be026c2609fbbdcfdc85005630675959a2b6661befb91268dd4cf73d59d615`.
This is **not** Supervisor assignment, Operator dispatch, host stage,
native-training/evaluation authority, AWS action, submission, or a BPB
result. I made no host/SSH contact, operational marker, transfer, target,
cleanup, training or evaluation.

## Exact identity and resolved HOLDs

- The `1a152d20`→`1923a3a4` diff changes only
  `FINAL_PENDING_RUNBOOK.md`, its `SHA256SUMS` entry, and a regression
  test. The single proposed Operator command at runbook line 103 now
  points to the designated clean r4 checkout
  `/home/poojith-devan/Documents/ChatGPT/Trainium-trn22-r10-stage-r4-stop-recheck-r1/research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/operator_once.py`.
  I checked that target's bytes independently; they match the request's
  repaired operator SHA-256, and the new test rejects any other command.
  This closes `24e00dfd`'s stale-r3 invocation HOLD.
- The repaired `operator_once.py` has the extra active Desktop
  `_local_stop_and_breakers()` check immediately after `_ssh_identity()`
  and before `_marker()`. I independently injected late STOP and breaker
  states at that seam under normal and optimized test bytecode: two local
  gate calls, **zero marker calls, zero SSH/remote calls** in each case.
  The existing per-SSH local STOP checks remain. This closes
  `22b57441`'s marker-consumption HOLD. The actual `-O` entrypoint
  refuses at its first gate.
- The enabled request is canonical and exact, SHA-256
  `d19086a3f0c8aff9835f14f641d528980b12b21f45310070f6613ae261a33958`.
  It pins operator
  `af888c89dbed4360895e5cf93299095efe3280893dafd1e504cd88b6a9c4408f`,
  unchanged remote source
  `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`,
  and unchanged 109,800-byte bundle
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
  The clean root `train.py` and `prepare.py` remain
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  Source e65, target 22bf, label, target path, remote bundle path,
  one-attempt marker, no-cleanup policy and native/eval false bits are
  unchanged. The active breaker root is Desktop; former Documents and
  package STOP roots remain checked.

## Offline checks and lifecycle boundary

- On exact `1923a3a4` **before adding this Verifier review artifact**:
  `sha256sum -c SHA256SUMS` passed 11/11; `git bundle verify` confirmed
  sole prerequisite `e65a3097a5b00f6178188a935bb2acb9a5042c6c` and
  exact HEAD `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`; the worktree
  was clean, `git diff --check` passed, and the complete suite ran 46
  normal tests with no failures and 46 under `-O` with no failures (one
  intentional skip). The repaired runbook SHA-256 is
  `19954183446238c270c23b59f184d581964f38b39eb2c665acd2ca5e96b3e2d4`;
  `SHA256SUMS` is
  `679f507417e9ec40d0b63a59eb68e4074e04ed29d44a88214a01786acc99fe38`.
- I parsed the new review as canonical JSON and called the exact
  `operator_once.py::_authority` equality gate offline with a hypothetical
  **in-memory** matching assignment: it accepted. Mutating the review's
  request hash, operator hash or `native_training_authorized` bit was
  rejected. **No assignment file was created.** With this PASS file
  present but the assignment still absent, direct normal and `-O`
  operator entrypoints return HOLD before marker or SSH.
- Adding the required final-review artifact changes the expected lifecycle
  state: the two pre-review tests
  `test_enabled_pending_request.EnabledPendingRequestTests.test_no_final_review_or_assignment_files_exist`
  and
  `test_incremental_stage.IncrementalStageTests.test_disabled_entry_and_missing_authority_precede_contact`
  now fail solely because they explicitly assert this review file is
  absent. I reran all **44 other tests** on the review branch: 44 normal
  passed, and 44 under `-O` passed with one skip. This is a documented
  test-lifecycle transition, not a passing claim for the unfiltered
  post-review suite. A separate Implementer may later re-scope those two
  historical assertions; I did not self-edit source or tests.
- At a local point in time, the named marker and active Desktop, former
  Documents, and package STOP paths were absent; active Desktop
  `launch_authorized` and trn22 `launch_allowed` were false and both
  relevant breakers clear. This is **not** a live host observation or
  future Operator preflight.

The independent timing review `02f02d7b` measured 20/20 **read-only
modeled** passes of the exact unchanged remote pre-open sequence: worst
double-label scan 980,357,752 ns, just **19.642 ms** under its one-second
bound; maximum modeled would-open age 70,135,175 ns. The changed local
operator check and runbook path do not alter that remote sequence, so
the observation carries forward **only as bounded feasibility evidence**,
not a measurement/hash attestation of this new local operator. A too-slow
scan fails before `O_EXCL`; a delayed actual open is checked again before
payload and any empty provisional inode is preserved. Thus the narrow
margin is a one-shot success risk, not an observed unsafe write; 20 trials
cannot guarantee actual `O_EXCL`, fsync, reconstruction or readback.

**Remaining authority:** integrate this exact review file into the
designated clean r4 execution worktree without changing operator, request,
remote, bundle or science bytes. A different Supervisor must independently
check the review hash, current Desktop STOP/breakers and absent one-attempt
marker, then issue the exact separate assignment. The Operator must perform
fresh local and host source/identity/lock/Neuron/process/disk/inode
preflight and obey fail-closed one-attempt semantics; any failure is HOLD,
not cleanup/retry authority. Retrieve the raw marker journal, distinct
session readback and attempted unconditional postflight for an independent
stage-result review. Matching r10 source/G2 and a separately reviewed
native A0 launch remain later gates. No BPB improvement is claimed.
