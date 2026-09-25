# Independent final HOLD — trn21 r10 r4 source-stage request

**Verdict: HOLD for the exact enabled source-stage request** at clean
`origin/codex/20260925-trn21-r10-stage-r4-enabled-final-pending@49f2941fb58e8d507dfe88ff1e7aea71eb1cd0f4`.
Its canonical `REQUEST.ENABLED_PENDING_FINAL_REVIEW.json` is SHA-256
`cc2bddbdda631c79f6fc8b6a7555504a4d994c3c3a5665d203c74e1fdfd72a76`.
Its exact schema is `trn21-r10-incremental-stage-enabled-request-v2`, bound
to host `trn21`, source e65, target r10, label
`fs-20260925-trn21-c5c-a0-r10-incremental-stage-r2`, target
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10`,
and marker `/home/poojith-devan/Desktop/Trainium-source-artifacts/trn21-c5c-a0-r10-incremental-stage-r2.one-attempt.jsonl`.
The frozen operator separately expects review schema
`trn21-r10-incremental-stage-independent-final-review-v2` and assignment
schema `trn21-r10-incremental-stage-supervisor-assignment-v2`; neither
file exists in the reviewed subject.
This receipt is **not** `INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW.PASS.json`;
no final PASS or Supervisor assignment was written. No host/SSH contact,
one-attempt marker, staging, native training, evaluation, cleanup, AWS action,
or submission occurred in this review.

## One blocking late-STOP path

In unchanged `operator_once.py` SHA-256
`b397a79246f2c854faf6f5a9bdbaa27f85f7fe3fd234958a01e8783c635996be`,
both `stage_once` and `_ssh_phase` call `_local_stop_and_breakers()`, then
`_ssh_identity()` (local `ssh-keygen` checks, each bounded up to 20 seconds),
then `_remote_timeout_identity()` (a **read-only SSH connection**) **without
another local STOP/breaker check between identity and the SSH probe**.
They recheck only *after* the probe. If the active Desktop STOP or trn21
breaker trips during `_ssh_identity()`, the remote probe contacts trn21
despite the new stop. An independent no-host mocked witness injected that
late STOP at the second gate: the remote-probe mock was called once before
the gate raised, in both stage entry and a phase; marker and phase SSH
remained absent. Thus the staging write is held, but the STOP contract of
no **new host contact** once a STOP is present is not met. The separately
reviewed trn22 r5 operator closes this same boundary; it is not authority
to bypass it here.

**Smallest repair:** add `_local_stop_and_breakers()` immediately after
`_ssh_identity()` and before `_remote_timeout_identity()` in both
`stage_once` and `_ssh_phase`. Add adversarial normal and `-O` no-host tests
that inject active Desktop STOP and trn21 breaker changes during identity
and assert **zero remote-probe calls, zero marker calls, and zero phase SSH**.
Under `-O`, exercise the seam with a synthetic nonoptimized `sys.flags`
object while also asserting the actual optimized entrypoint still refuses.
Keep the r4 one-final-scan `remote_incremental.py`, bundle, science bytes,
label, target, and marker unchanged. Recompute and bind the new operator
hash in a new canonical enabled request, runbook and manifest, then obtain
a new independent final review; this HOLD cannot be converted into PASS by
adding an assignment or retrying the current command.

## Bounded checks and residuals, not authority

- The earlier independently reviewed **disabled** r4 source was static-PASSed
  at `b8c9c36b4fa5b72a225d8d55aa223b8e2b341bec`; its receipt SHA-256 is
  `dd251ab06b59fe01136773d1c4f178869bf9f98771ffb58ee2cd81a9fe648d32`.
  The `b8c9c36b`→`49f2941f` diff adds only the enabled request and offline
  test and updates runbook/manifest. `train.py`, `prepare.py`, operator,
  remote and bundle bytes are unchanged. The clean science SHA-256 values
  remain `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`;
  remote SHA-256 is
  `6f0da61a76595a6fea18662d07d7311c504ab3a4b641ac972c440438bda08bf6`,
  and the 109,800-byte bundle SHA-256 is
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
  `git bundle verify` confirms sole prerequisite e65
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c` and r10 HEAD
  `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`.
- The ten-entry `SHA256SUMS` verified from its package directory; 39/39
  offline tests passed normally and 39/39 under `-O`. `git diff --check`
  passed. The actual normal entrypoint HOLDs for missing independent final
  PASS; `-O` HOLDs for forbidden optimization. The actual one-attempt marker
  is absent. I tested the exact canonical enabled request against the
  frozen `_authority` equality gate with hypothetical review/assignment
  values **in memory**: exact source/operator/bundle binding passed and a
  mutated native-authority bit was rejected. No authority file was created.
- The static PASS explicitly documents the accepted *bounded* residual of
  r4's **single** final pre-open broad label scan: a persistent label arriving
  behind that scan can lead to an **empty retained provisional bundle** and
  a consumed one-shot marker, caught by the mandatory post-open scan before
  payload/checkout. A noncooperating same-UID/privileged actor can race
  after the last post-open scan; the prior static review limits its judgment
  to the cooperative-lock/ordinary-DAC model and does not claim atomic
  uniqueness. This residual is disclosed, not the new HOLD cause and not a
  waiver of the STOP defect. The revised r4 ≤1-second live pre-open timing
  remains unmeasured; the prior two-scan read-only timing attempt failed at
  1.161466714 seconds and cannot prove r4 feasibility. A later authorized
  attempt may safely fail and consume its one-shot marker; no retry or
  cleanup follows from this review.

The request is source-stage-only (`native_training_authorized=false`,
`evaluation_authorized=false`), but its true dispatch bits do not authorize
operation while this HOLD controls. A repaired final review, separate
Supervisor assignment, fresh local/host preflight, retrieved raw stage
evidence and independent stage-result review remain required before any
native A0 or BPB claim.
