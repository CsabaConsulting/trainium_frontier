# Independent final HOLD — trn22 r10 source-stage enabled request r1

**Verdict: HOLD for dispatch of the exact source-stage request.** Reviewed the
clean Implementer subject `16e1c982d8e7ce5dcd53ca966cce65d5a57c7301`
(`origin/codex/20260925-trn22-r10-stage-r3-enabled-request-r1`) against
independently static-reviewed disabled r3 `b3ec7406f0e7a3df6848698876e2106305dd283f`.
This is not a native-training, evaluation, AWS or submission review. I made
no host/SSH contact, assignment, operational marker, transfer, target,
cleanup, training run or evaluation. The only marker created by my
adversarial test was inside a disposable local `TemporaryDirectory`.

## Blocking finding: a late local STOP can consume the one-attempt marker

`operator_once.py::stage_once` performs `_local_stop_and_breakers()` at line
318, then `_ssh_identity()` at line 319, then `_marker()` at line 320 with no
second local STOP/breaker read. `_ssh_identity()` invokes two separate local
`ssh-keygen` subprocesses, each with a 20-second timeout. A human STOP or
active Desktop breaker introduced during that interval can therefore exist
before the irreversible one-attempt marker is opened. The later `_ssh_phase`
rechecks local STOP before each SSH, so this path fails safely without host
contact, but it needlessly consumes the only approved stage attempt and
persists a marker after STOP. The previous disabled-source static PASS did
not establish this final enabled authority boundary.

Independent no-host witness: I mocked authority inputs, the local STOP
gate, and SSH identity to reach this seam; the identity callback introduced
a local STOP condition. I kept the real `_marker()` pointed at a private disposable
temporary directory, and made the remote callback refuse before contact.
Result: `marker_created=True`, three journal events, later STOP refusal,
zero host-contact calls. A separate marker-spy witness reached `_marker`
after exactly one prior STOP check. Neither test touched the real marker or
any host.

**Smallest repair:** after `_ssh_identity()` returns, call
`_local_stop_and_breakers()` again immediately before `_marker()`. Add a
normal and optimized-Python regression where STOP or the active Desktop
breaker trips during identity; assert no marker or SSH seam is reached.
Keep the existing per-SSH STOP checks and one-attempt semantics. Since the
operator bytes change, regenerate the enabled request's
`operator_source_sha256`, its canonical request SHA, `SHA256SUMS`, and all
downstream exact review/assignment pins. Submit the repaired exact source to
a different Verifier. **Do not add a final PASS JSON or dispatch this r1
subject.** No cleanup or retry is authorized by this finding.

## Other exact checks that passed, with bounded meaning

- The only `b3ec7406`→`16e1c982` changes are
  `FINAL_PENDING_RUNBOOK.md`, a canonical enabled request, `SHA256SUMS`,
  `test_enabled_pending_request.py`, and the historical test that formerly
  required the enabled request to be absent. The operational
  `operator_once.py`, `remote_incremental.py`, 109,800-byte bundle, root
  `train.py` and `prepare.py` are byte-identical to r3.
- Exact SHA-256: enabled request
  `b33a72fc3b5588f537c23ff4bb565344a92eabef58a33970db2bf2160a60ee91`;
  operator
  `a5fc2ebac3520f425d5ab0bdfb7ae0091a2e5db59b152dd8cd39eb67cff5beda`;
  remote
  `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`;
  bundle
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`;
  `SHA256SUMS`
  `6a96a4cba315be9589fbc91842c7c4f576a5e2bcb8b1c73abb33e43438f00a9f`.
  Clean execution-root `train.py` is
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  `prepare.py` is
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- `sha256sum -c SHA256SUMS` passed **11/11**. `git bundle verify`
  confirms sole prerequisite `e65a3097a5b00f6178188a935bb2acb9a5042c6c`
  and exact HEAD `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`.
  The source worktree was clean and `git diff --check` passed.
- Supplied offline tests ran **43** normal with no failures and **43** with
  `-O` with no failures (one intentional optimized-mode skip). Direct operator entrypoints in
  both modes HOLD without the missing independent final review and Supervisor
  assignment, before a real marker or SSH. The enabled request matches the
  operator's exact schema, source/target/label/marker, source hashes, and
  source-stage-only scope; it does not authorize native training/evaluation.
  The source has no host unlink/rename/cleanup path. Its readback and
  attempted unconditional postflight remain unchanged and still need a
  distinct result review after any actual stage.
- Point-in-time local observation: active Desktop controller state had
  `launch_authorized=false`, `hosts.trn22.launch_allowed=false` and both
  relevant breakers clear. Active Desktop, former Documents, and package
  STOP paths and the named r10 marker were absent. This is **not** a future
  preflight and says nothing about the remote host's present state.

The independent raw timing review `02f02d7b` validates **20/20 read-only
modeled trials**: maximum double-label scan 980,357,752 ns leaves only
19,642,248 ns (19.642 ms) below the one-second limit; maximum modeled
post-would-open age was 70,135,175 ns. This is a material *feasibility*
risk for one attempt, not this correctness HOLD: the r3 source checks a
too-slow scan before `O_EXCL` and a delayed actual open again before
payload, retaining even an empty provisional inode on post-open refusal.
The 20 trials do not measure real `O_EXCL`, payload, fsync, reconstruction
or readback and cannot guarantee one-shot success. After the STOP repair,
the Supervisor must explicitly weigh that bounded risk and perform fresh
host/local preflight; a separately proposed bounded pre-open retry would
require its own implementation and independent review.

One integration note: after a future independent final PASS file is added,
the current tests that assert its absence are pre-review tests, not a green
post-review integration suite. Adjust or re-scope those tests truthfully in
the eventual final package; never report this subject's pre-review 43/43 as
if it also covered a later added PASS/assignment.
