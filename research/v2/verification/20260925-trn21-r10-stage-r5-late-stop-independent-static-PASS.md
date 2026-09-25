# Independent static PASS — trn21 r10 r5 late-STOP source-stage repair

**Verdict: STATIC PASS for the exact disabled r5 source only.** I independently
reviewed `origin/codex/20260925-trn21-r10-stage-r5-stop-repair-disabled@2db60693ad4a27b0fb4a969f72c652d5becb84c6`
against the r4 final HOLD `06451f16d48026b9a167edf803ce57fdec787ce4`.
This receipt does not authorize SSH, source staging, a marker, native training,
evaluation, cleanup, AWS action, submission, or conversion of the r4 HOLD to
PASS. A new exact enabled request, independent **final** source-stage review,
Supervisor one-attempt assignment, fresh Operator preflight, and later
stage-result review remain separate gates.

## Exact offline subject and checks

- The changed paths are confined to the trn21 stage package. The r4 enabled
  request and its request-specific test were removed. The r5 `REQUEST.HOLD.json`
  keeps `dispatch_authorized`, `source_stage_authorized`,
  `native_training_authorized`, and `evaluation_authorized` false. No enabled
  request, final PASS JSON, or Supervisor assignment is present in the reviewed
  worktree. Both normal and optimized local entrypoints exit HOLD before SSH or
  marker creation. The local r2 one-attempt marker was absent at review time.
- `operator_once.py` SHA-256 is
  `f48a35c460498a81a56cc73f68f5130a4d91f319a88be7c0ad6a1668f80de70b`.
  The unchanged `remote_incremental.py` is
  `6f0da61a76595a6fea18662d07d7311c504ab3a4b641ac972c440438bda08bf6`;
  the unchanged 109,800-byte bundle is
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
  `git bundle verify` showed only e65
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c` as prerequisite and r10
  `22bfbab5a77d6feb670d2d737504f8b70b08ddc1` as HEAD. The reviewed
  `train.py` and `prepare.py` hashes remain `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- All ten `SHA256SUMS` entries passed. The package's 39 offline tests passed
  normally and 39 under `python3 -O`; `git diff --check` passed. No host or
  SSH call was made for this review.

## Safety judgment

The r4 final HOLD identified the interval between bounded local
`_ssh_identity()` and the read-only remote GNU-timeout probe. R5 adds
`_local_stop_and_breakers()` immediately after identity in **both**
`stage_once` and `_ssh_phase`. The phase command is now constructed before
identity, so a STOP arriving during that work is checked before the probe.
The existing post-probe STOP checks still precede the marker and each phase
SSH. The new test exercises active Desktop STOP and trn21 breaker insertion
during identity at both entrypoints and confirms no probe, marker, or phase
call. In my independent no-host mock witness, the old r4 source made one
remote-probe call after a synthetic late STOP at each entrypoint; r5 made
zero probe, marker, and phase calls at both entrypoints. The actual optimized
entry guard is also exercised, while synthetic normal flags permit testing
the seam under `-O` without bypassing that real guard in production.

This is a static judgment within the previously documented cooperative-lock
and ordinary-DAC model, not a claim of atomic global uniqueness or an atomic
STOP-to-SSH transition. The unchanged r4 remote one-final-scan path retains
the strict finite ≤1-second actual-open freshness gate and post-open scan
before payload. A persistent label arriving behind the final pre-open scan
can leave a retained **empty** provisional bundle and consume the future
one-shot marker; no cleanup or retry is authorized. A noncooperating
same-UID/privileged actor after the last post-open scan remains a documented
residual. Trn21 live timing of this one-scan path is **unmeasured**; the old
two-scan timing failed at 1.161466714 seconds and is not a pass for r5.

No source stage, native A0, table experiment, fresh evaluation, or new BPB
result follows from this static PASS.
