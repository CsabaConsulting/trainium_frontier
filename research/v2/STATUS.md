# Workflow installation and incremental launch status

Status: **both guest disks expanded; both host G1 calibrations PASS; scored training on both hosts still HOLD; no new BPB result.**

At 03:30 UTC, trn21's
[final exact checkout request](authorizations/trn21-c5c-a0-stage-r2-checkout-request/GATE_CONTRACT.FINAL_PENDING_REVIEW.json)
[passed independent offline review](authorizations/trn21-c5c-a0-stage-r2-checkout-request/INDEPENDENT_FINAL_REQUEST_REVIEW.PASS.json).
The [Supervisor's one-attempt checkout-only assignment](authorizations/trn21-c5c-a0-stage-r2-checkout-request/SUPERVISOR_CHECKOUT_ONLY_ASSIGNMENT.json)
passed local binding audit. The Operator's
[one-attempt terminal receipt](authorizations/trn21-c5c-a0-stage-r2-checkout-result/OPERATOR_CHECKOUT_ONLY_TERMINAL_RECEIPT.md)
reports exact clean source staged, preflight/stage/postflight all returning
zero, protected assets intact and the fixed marker consumed. A separate
Verifier [accepted the archived result for checkout only](authorizations/trn21-c5c-a0-stage-r2-checkout-result/INDEPENDENT_CHECKOUT_RESULT_REVIEW.PASS.json).
The [hardened CPU G2 successor](g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2-final-pending/GATE_CONTRACT.json)
[passed independent static review](g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2-final-pending/INDEPENDENT_STATIC_REVIEW.md)
for its disabled form. A new exact CPU-only request, review, preflight and
assignment remain necessary. CPU G2, native training and evaluation remain
HOLD; no scored control or new BPB exists.

At 03:12 UTC, trn22's
[transfer-only draft](authorizations/trn22-c5c-a0-r4-transfer-only/TRANSFER_REQUEST.HOLD.json)
received an [independent static HOLD](authorizations/trn22-c5c-a0-r4-transfer-only/INDEPENDENT_IMPLEMENTATION_REVIEW.md)
for unpinned SSH host identity, ignored local STOP, self-attested review
approval and an unproven root-device alias. An Implementer repaired the
disabled package at `e07781f2`, and it [passed independent static re-review](authorizations/trn22-c5c-a0-r4-transfer-only/INDEPENDENT_REPAIRED_STATIC_REVIEW.md).
A new enabled transfer-only request, separate review, fresh preflight and
Supervisor one-attempt assignment are still needed. No transfer or training
is authorized.

At 02:57 UTC, trn21's
[repaired disabled checkout draft](authorizations/trn21-c5c-a0-stage-r2-checkout-request/GATE_CONTRACT.json)
[passed independent offline review](verification/20260925-trn21-stage-r2-checkout-repair-static-review.md).
The enabled exact request, its separate review, fresh preflight and one-use
assignment remain HOLD. The [r2 CPU G2 draft](g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2/GATE_CONTRACT.json)
also [passed planning-only review](g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2/INDEPENDENT_STATIC_REVIEW.md),
with three safety pins to harden before host use. Neither PASS is a scored
run or new BPB.

At 02:48 UTC, trn22's
[read-only source-stage feasibility](authorizations/trn22-c5c-a0-r4-source-feasibility/OPERATOR_READONLY_FEASIBILITY_20260925T024228Z.md)
[passed independent planning review](authorizations/trn22-c5c-a0-r4-source-feasibility/INDEPENDENT_PLANNING_VERIFIER_RECEIPT.md).
It is a point-in-time snapshot only. The transfer-only package must still be
reviewed, then a fresh full preflight must archive raw transport details,
old-path inventory, mount alias reconciliation, STOP/breaker and disk checks
before any narrow assignment. No source transfer or new BPB.

At 02:44 UTC, trn22's frozen r4
[source capsule/plan](authorizations/trn22-c5c-a0-r4-source-stage/GATE_CONTRACT.json)
[passed independent offline static review](authorizations/trn22-c5c-a0-r4-source-stage/INDEPENDENT_STATIC_VERIFIER_RECEIPT.md).
The 6,368-member archive matches the frozen 446b source; the package commit
is not a source substitute. Read-only host feasibility and a separate
transfer-only package are in progress. No transfer, checkout, scored control
or BPB result follows from this review; native launch remains HOLD.

At 02:36 UTC, trn21's disabled stage-r2
[checkout request](authorizations/trn21-c5c-a0-stage-r2-checkout-request/GATE_CONTRACT.json)
received an [independent HOLD](authorizations/trn21-c5c-a0-stage-r2-checkout-request/INDEPENDENT_REQUEST_REVIEW.md):
the same assignment could retry with a different journal path after an early
failure, and a claimed review hash was not authenticated. Both were caught
offline before host use. A repaired request, separate review and one-shot
checkout-only assignment are required. trn22's exact-source capsule/plan is
under static review, not transferred. Both scored controls remain HOLD.

At 02:27 UTC, trn22's prior exact-env four-rank CPU G2 numeric PASS
[carried forward by independent source-identity review](disk-monitor/r4-hold/G2_R3_TO_R4_SOURCE_IDENTITY_DISPOSITION.md)
to the unchanged c5c training bytes and argv at frozen r4 source 446b6576.
It is not a new host r4 run; source stage, live safety/monitor and one-shot
native gates remain HOLD.

At 02:19:45 UTC, trn21's
[read-only feasibility refresh](authorizations/trn21-c5c-a0-stage-r2-extraction/OPERATOR_READONLY_FEASIBILITY_REFRESH_20260925T021945Z.md)
confirmed exact three-file persistence, absent checkout/output, intact assets,
idle device, closed STOP/breakers and 58,571,898,880 B free. It is not a
checkout preflight or authority; exact request review and new one-shot
assignment remain HOLD.

At 02:13 UTC, trn22 r4's
[disabled exact-hash draft](disk-monitor/r4-hold/GATE_CONTRACT.json)
[passed independent static review](disk-monitor/r4-hold/INDEPENDENT_VERIFIER_RECEIPT.md).
It is intentionally non-dispatchable; host source transfer/staging, exact
source G2 disposition, fresh preflight and one-shot native authority are not
done. No BPB.

At 02:11 UTC, the frozen trn21 checkout implementation
[passed independent local review](authorizations/trn21-c5c-a0-stage-r2-extraction/INDEPENDENT_IMPLEMENTATION_REVIEW.md).
The Verifier requires a separately reviewed exact request and failure-path
read-only full postflight before a checkout-only assignment. No host checkout,
CPU G2 or native run has occurred; no BPB changed.

At 02:06 UTC, trn22 r4 frozen source 446b6576
[passed independent local safety review](disk-monitor/R4_446B6576_INDEPENDENT_VERIFIER_RECEIPT.md).
This is not host or dispatch clearance: the old r3 request/bindings do not
match, and exact r4 package/static review, G2 disposition, fresh host preflight
and one-shot native authorization remain. A HOLD-only binding package is in
preparation. No scored A0 or BPB result.

At 02:03 UTC, trn21's archived three-artifact persistent-ext4 transfer
[passed independent result review](authorizations/trn21-c5c-a0-stage-r2/INDEPENDENT_TRANSFER_RESULT_RECEIPT.md).
It proves transfer only, not checkout or fresh host readiness. The separate
checkout-only package is under implementation review; extraction, CPU G2,
native training and evaluation remain HOLD. trn22's next r4 safety source is
still under independent review, with native dispatch HOLD. No new BPB.

At 01:54 UTC, trn21's transfer-only Operator archived a
[terminal three-artifact receipt](authorizations/trn21-c5c-a0-stage-r2/operator-transfer-only-terminal-receipt.json)
claiming distinct-SSH persistence and clean postflight. Its raw evidence is
under separate independent result review; no checkout, CPU G2, native training,
evaluation, or new BPB followed. The non-dispatchable
[checkout-only package](authorizations/trn21-c5c-a0-stage-r2-extraction/GATE_CONTRACT.json)
also awaits independent implementation review and a new assignment. trn22's
next r4 safety source is frozen at `446b657695f086f9b484fa601c9a3b61c2495912`
with local tests passing, but independent review is pending; the last reviewed
revision failed and native dispatch remains HOLD.

At 01:23 UTC, trn21's persistent-ext4 source-transfer r2 package had an
[independent static planning PASS](authorizations/trn21-c5c-a0-stage-r2/INDEPENDENT_VERIFIER_RECEIPT.md)
and a new [transfer-only assignment](authorizations/fs-20260925-v2-c5c-a0-control-trn21-r1.source-stage-r2-transfer-assignment.json).
The Operator may write only the three exact new ext4 targets after fresh full
preflight and must prove each in later distinct SSH sessions. There is no
retrieved transfer result yet; checkout/extraction, CPU G2, training and eval
remain HOLD. trn22's frozen r4 [999b](disk-monitor/R4_999B_MARKER_SPOOF_INDEPENDENT_RECEIPT.md)
and [f8fe](disk-monitor/R4_F8FE_INDEPENDENT_VERIFIER_RECEIPT.md) reviews both
failed safety gates. The subsequent 11379083 source also
[failed independent review](disk-monitor/R4_11379083_INDEPENDENT_VERIFIER_RECEIPT.md)
despite 63/63 intended local tests: a detached markerless worker could be
certified clean, and an early launcher refusal lacked no-spawn proof. No repair
or native run is approved. No new BPB.

At 00:44:23 UTC, the assigned trn21 c5c A0 source transfer reported an exact
104,005,787-byte SHA-verified capsule inside its SSH process, but the exact
remote `/dev/shm` file was absent from a separate process about five seconds
later. The [Operator's HOLD receipt](authorizations/fs-20260925-v2-c5c-a0-control-trn21-r1.source-stage-hold.md)
records an unchanged host boot and unknown cause; no checkout, CPU G2, native
training or evaluation occurred. That transfer path cannot be retried under
the old assignment. A [read-only addendum](authorizations/fs-20260925-v2-c5c-a0-control-trn21-r1.source-stage-forensic-addendum.md)
found stable follow-up mount namespaces but could not recover the exact
transfer-writer source. trn22 r3 [failed independent disk-interlock safety review](disk-monitor/R3_INDEPENDENT_VERIFIER_RECEIPT.md):
post-bind source drift can veto an emergency stop, and a TERM-ignoring guard
can outlive both monitors. The table rank-hash diagnostic has [static planning
PASS](../agentic/verification/20260925-c5c-table-hash50-static-review.md) only.
Neither host has produced a scored c5c control or new BPB result.

At 00:12:54 UTC, trn21's [read-only c5c A0 feasibility audit](authorizations/fs-20260925-v2-c5c-a0-control-trn21-r1.readonly-feasibility.json)
found 58,676,293,632 B root free, intact retained checkpoints and warm caches,
no matching job/device/lock holder, and no collision for the proposed A0 label.
The exact e65 source commit was not found in the 144 inspected host repositories;
that search does not prove host-wide absence. An [independent review](../agentic/verification/20260925-trn21-c5c-a0-r1-readonly-feasibility-review.md)
accepted the audit **for source-stage planning only**. A hash-bound
[source-stage/CPU-G2 package](g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-r1/GATE_CONTRACT.json)
later passed independent static review and received a stage-only assignment;
the failed persistence check above supersedes that path. The feasibility audit
and package did not authorize CPU G2, native training or evaluation. trn22
remains held for a reviewed continuous disk monitor and fail-closed liveness
interlock.

Latest: trn21's [guarded G1 r2 Operator evidence](g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/operator-receipt.json)
passed [independent result review](../agentic/verification/20260925-g1-trn21-r2-calibration-result-review.md):
the old retained checkpoint reproduced 0.9918353324459732 at context 1024
over exactly 20,971,520 tokens, and postflight was clean. This is not a new
model score or A0 launch authority. The next trn21 c5c A0 control is
conditionally ratified but awaits exact source stage/G2 and all live gates.
trn22 A0 r2 remains held for a reviewed continuous disk monitor plus liveness
response. Neither host has completed a new scored control.

Historical: trn21's one-label guarded G1 r2 public20M calibration [was authorized](g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/request.guard-authorized.json) and launched at 23:41:16 UTC after fresh preflight; guard PID 2835264 remained alive with no disk stop at 23:44:31 UTC. Its output and independent result review were pending at that time. trn22's exact-env CPU G2 r3 [passed independent numerical comparison](../agentic/verification/20260925-trn22-c5c-a0-r2-g2-r3-numeric-review.md); the c5c A0 r2 [final request passed static review](../agentic/verification/20260925-trn22-c5c-a0-r2-final-request-review.md) and [read-only prelaunch checks](authorizations/fs-20260925-v2-c5c-a0-control-trn22-r2.prelaunch-readonly.json) passed. Native dispatch is [HOLD](authorizations/fs-20260925-v2-c5c-a0-control-trn22-r2.monitor-hold.json) for an unreviewed continuous disk monitor; no A0 step or BPB exists. The repaired A2 native-smoke package [passed static review for staging only](../agentic/verification/20260925-v2-c5c-a2-native30-r2-static-review.md); its native and 50-step proof remain HOLD.

The proposed A2 table native smoke [failed independent static review](../agentic/verification/20260925-v2-c5c-a2-native30-static-review.md) before host use: its step-log parent was not authenticated, reproducing the class of trn22's CPU G2 failure. Its 30-step plan also does not satisfy the older 50-step rank-hash criterion. An Implementer is repairing the package; A2 still has no native or BPB result.

At 22:48:11 UTC, the trn22 A0 r2 Operator [archived a source-staging PASS and CPU G2 infra HOLD](g2-probes/fs-20260925-v2-c5c-a0-control-trn22-r2/operator-stage-hold.json). Its clean checkout has exact c5c/prepare/launcher hashes, unchanged retained checkpoint and six warm-cache identities, 58,452,987,904 B free after G2, and no train/eval/Neuron holder. The four-rank exact-environment CPU probe failed at step zero because its step-log path pointed into a deliberately clean checkout whose `research/run-logs` parent was absent. The G2 label is consumed; the A0 r2 science label is not. A separately reviewed, new-label G2 retry is being prepared with an isolated step-log path. **No native training, checkpoint, public BPB, or launch authorization resulted.** trn21's G1 r1 stage remains HOLD on absent host commit; [its r2 exact-retained-source package](g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/RUNBOOK_PENDING.md) has [independent static PASS for Operator staging only](../agentic/verification/20260925-g1-trn21-r2-pending-package-review.md). A separate Operator is assigned to staging; evaluator dispatch remains HOLD. Ready scored jobs remain zero on both hosts.

At 22:26:14 UTC, a [fresh read-only guest and quiescence audit](../agentic/operations/TRN2_POST_RESIZE_READ_ONLY_AUDIT_2026-09-25_0356.json)
verified that both root disks, partitions, and ext4 filesystems have grown. Root
free space is 58,907,762,688 B on trn21 and 58,454,982,656 B on trn22, above
both the 6.5-GiB operational floor and provisional 10-GiB A0 prelaunch gate.
Neither host had a matched training/evaluation process, Neuron holder, or host-lock
holder at audit time. This clears the **storage-only** block. It does not
authenticate the pinned code, checkpoint, caches, exact environment, or a new
one-run authorization. trn21's G1 calibration remains pending those checks;
trn22's reviewed zero-step A0 incident still has `host_clean=false` and its
controller disk breaker was still tripped at audit time. The human then
[conditionally authorized re-arm](authorizations/20260925-trn22-disk-breaker-human-ack.md),
the [fresh independent review](../agentic/verification/20260925-trn22-disk-breaker-rearm-review.md)
passed, and only the workflow-v2 trn22 disk breaker was cleared. Both host
`launch_allowed` flags and top-level `launch_authorized` remain false. No
training or evaluator was launched by these actions. The pre-resize snapshots
below are historical, not live state.

trn21 G1 [staging stopped safely](g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r1/operator-stage-hold.json):
fresh disk, checkpoint, caches, tokenizer and idle checks passed, but the pinned
source commit is absent from the host and its Git fetch returned 128. No staged
checkout or eval process exists. An exact-source transport is being reviewed.
The repaired [A2 table package](../submission-validation/20260925-v2-c5c-a2-m64d128-trn22-r1/GATE_CONTRACT.json)
has an [independent static PASS / dispatch HOLD](../agentic/verification/20260925-v2-c5c-a2-m64d128-trn22-r1-static-r2-review.md),
not a G2, native or BPB result.

Historical pre-resize status at 22:21:36 UTC: the human's console screenshot shows both exact root EBS
volumes at 250 GiB, and a [read-only guest audit](../agentic/operations/TRN2_EBS_GUEST_PENDING_AUDIT_2026-09-25_0351.json)
confirms both `/dev/nvme0n1` disks now expose 250 GiB. However, each root
`/dev/nvme0n1p1` ext4 partition/filesystem is still the old size: trn21 free
6,901,698,560 B and trn22 free 6,448,898,048 B, both below the immutable
6,979,321,856-B floor. The human is doing guest `growpart` and `resize2fs`;
no agent has done those actions. Training remains HOLD, and trn22 still needs
an explicit human breaker acknowledgement/reset after the guest resize.

Read-only guest disk audit at 22:05:10 UTC: trn21 has 6,901,743,616 B free and
trn22 6,448,918,528 B free, both below the 6,979,321,856-B floor. Both root
disks still report 200 GiB. This check did not query AWS-side resize state or
repeat the earlier process-quiescence audit. [Receipt](../agentic/operations/TRN2_DISK_READ_ONLY_AUDIT_2026-09-25_0335.json).

The new-label [trn22 A0 retry package](../submission-validation/20260925-v2-c5c-a0-control-trn22-r2/GATE_CONTRACT.json)
has an [independent static PASS / dispatch HOLD](../agentic/verification/20260925-v2-c5c-a0-trn22-r2-static-review.md).
Its 10-GiB prelaunch projection is provisional, not a proved maximum. Human EBS
expansion and explicit trn22 breaker reset, then fresh exact-environment G2,
staging, and one-shot authorization are still required. The trn21 G1
[pending calibration request](g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r1/request.pending.json)
has [independent static PASS / dispatch HOLD](../agentic/verification/20260925-g1-trn21-pending-package-review.md):
it is still non-dispatchable until host/storage and checkpoint/cache/environment
checks, a finalized request, and separate one-shot authorization pass.
The matching [trn21 c5c A0 control package](../submission-validation/20260925-v2-c5c-a0-control-trn21-r1/GATE_CONTRACT.json)
also has [independent static PASS / dispatch HOLD](../agentic/verification/20260925-v2-c5c-a0-trn21-r1-static-review.md).
It requires the human storage fix, trn21 G1, exact-env G2, live staging and
one-shot authorization before training; there is no new control score.
S0's [independent source audit](../agentic/verification/20260925-s0-submission-static-recheck.md)
found no proven rule violation but left the submission HOLD until native checks
and two cold bare-command replays pass.

Read-only Operator inventory at 21:50:53 UTC: trn21 free 6,901,788,672 B,
trn22 free 6,448,939,008 B; both root disks still 200 GiB, with no training
processes or Neuron holders. The requested EBS expansion is not reflected in
either guest. AWS-side status was not checked.

An hourly Codex heartbeat now checks this thread for a meaningful change while the
hosts are blocked. It is **not** a verified unattended host/Bedrock scheduler and
does not authorize training, AWS changes or breaker reset. Unchanged checks stay quiet.

## Safety update (2026-09-24 21:37 UTC)

- trn22's first warm c5c A0 attempt crossed the immutable 6.5-GiB root-disk floor during
  compilation **before any optimizer boundary**. The Operator authenticated and signaled
  only that guard. Terminal postflight at 21:28:07 UTC was rc 1 and `host_clean=false`
  solely for disk; no owned processes or cleanup actions remained. The observer reported
  `COMPLETE_NON_SCORED` with zero boundaries. There is no checkpoint/BPB result or win.
  The label is consumed and must never be reused. The hash-verified
  [terminal receipt](incidents/fs-20260925-v2-c5c-a0-control-trn22-r1/operator-terminal.json)
  and [independent review](../agentic/verification/20260925-a0-trn22-disk-safety-review.md)
  are archived. A `status=failed` row in `ledger.jsonl` makes the consumed label
  discoverable while the unmodified `decide.py` excludes it from controls and drift.
  Compilation consumed 1,500,942,336 bytes of root free space.
- trn22's latest checked free space is 6,448,951,296 bytes, below the
  6,979,321,856-byte floor;
  trn21 was already below the floor. Both need the human root EBS expansion in the
  [updated disk decision packet](DISK_DECISION_PACKET_2026-09-25.md). trn22's disk
  breaker remains tripped; only the human can acknowledge/reset it after resize and
  reviewed failure diagnosis. No new host run may start until then.
- The R5/new-runner local synthetic suite passed earlier, but its one-host production
  pilot and bare-S0 compatibility remain unverified. The timebox does not justify
  switching launchers or bypassing any disk/containment gate.
- The exact c5c last-layer A1 m64/d128 table-gate package is archived with an
  [independent static PASS / dispatch HOLD](../submission-validation/20260925-v2-c5c-a1-m64d128-trn21-r1/verifier.json).
  It has no G2, native-smoke, full-run or BPB result. Its launch must authenticate
  effective seed/budget environment; a valid same-host control is required before
  comparing a full A1 run, not before the cheaper diagnostic G2.

This safety update supersedes the earlier active-A0 status below.

## Live update (2026-09-24 21:26 UTC)

- The first warm `c5c410af...` A0 control was guarded-dispatched **once** on trn22 at
  21:25:57 UTC, guard PID `2963679`, label `fs-20260925-v2-c5c-a0-control-trn22-r1`.
  The separate read-only step-boundary observer was alive before dispatch. The exact
  [one-run authorization](authorizations/fs-20260925-v2-c5c-a0-control-trn22-r1.authorization.md)
  binds G1, G2, stage/fingerprint, cache/disk and observer evidence. No checkpoint, fresh
  public20M result, or verdict exists yet. Training remains under Operator monitoring.
- The pending warm S0 native package is independently static-reviewed but **HOLD**:
  it requires at least 8,833,482,752 free bytes on trn22 before dispatch, more than the
  measured roughly 7.95 billion. It is not a cold replay or submission-ready.
- Both hosts already contain the five train shards that `prepare.py setup` downloads by
  default. This excludes a missing-default-shard discovery but not uncertainty about
  official private runtime data. First place is the user-supplied 0.9464 official;
  best confirmed local public20M remains 0.9918353324 single/0.9919537735 mean.

This live update supersedes older A0-staging and G1-running statements below.

## Live update (2026-09-24 21:14 UTC)

- trn22 G1 fresh-process public20M calibration passed and reproduced the known checkpoint score
  exactly: `0.992968300659387` at 20,971,520 tokens/context 1024. The independently reviewed
  [Operator receipt](g1-calibrations/fs-20260925-g1-public20m-calibration-trn22-r1/operator-receipt.json)
  has SHA-256 `d78803b0ed21f9f4470560840ae7f78bb2b8ccaf2c82d4d8d9a472c4a6445a57`.
  This is evaluator calibration, **not a new model result or valid advancement control**.
- The trn22 Operator is staging the one warm dense `c5c410af...` A0 control. It has **not
  launched training**. Final authorization still needs the stage receipt, unique-root/source
  checks, live disk projection, G2 binding and read-only boundary observer started before the
  first optimizer step. trn21 remains disk-blocked; no deletion or AWS action occurred.
- The best confirmed candidate remains `0.9918353324` single, `0.9919537735` two-run mean.
  New-runner production migration and S0 cold replay remain pending.

This live update supersedes the older pending G1 statements below.

## Live update (2026-09-24 20:55 UTC)

- All four CPU probes passed, with three exact comparisons, at `50e346bf`. Receipt:
  `g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/operator-retrieval-receipt.json`.
- trn22 started retained-checkpoint calibration at 20:53:19 UTC, guard PID 2950407, label
  `fs-20260925-g1-public20m-calibration-trn22-r1`. Corrected request SHA `2efe1b6404ab262e818ef607a4630e82d641800a03813bf0427c3ad82a4d35b6`
  preserves the historical compiler flags and six existing warm cache directories. This is
  evaluator verification, not a new trained candidate or win. Historical runtime was about 12 minutes.
- Corrected warm A0 package `6f63f598` passed independent static/fingerprint checks; dispatch remains
  conditional on G1 and complete live gates. It establishes a new c5c same-host control, not a win.
- New-runner repair `d7733fb` passed the independent local synthetic safety suite. It is not merged
  into production, not bare-S0-ready, and still needs a one-host pilot. The legacy launch stays separate.
- trn21 remains blocked awaiting the human disk decision. No deletion, AWS action, breaker reset,
  new adoption, or new BPB result occurred. Ready scored-job count is still zero pending G1.

The live update above supersedes older pending-gate statements below.

## Latest human brief (25 September, 02:06 IST)

- [Active brief](CONTROLLER_BRIEF_2026-09-25.md): keep proven launches while piloting the repaired
  new runner on one host; do not gate independent legacy dense work on a wholesale migration.
- [Disk decision packet](DISK_DECISION_PACKET_2026-09-25.md): recommend human expansion of both
  exact root EBS volumes from 200 to 250 GiB. The historical 3.89 GB cleanup opportunity is gone:
  the old controller directory is empty. No deletion/AWS action occurred; LR0.015 checkpoints stay.
- Latest measured root free space: trn21 6.43 GiB, trn22 7.46 GiB. trn21 and table/cold replay work
  remain disk-blocked. trn22's warm dense path still needs its independent runtime gates.
- Independent static model review is integrated at `7150b59b`; the exact four CPU/G2 checks have
  a reviewed contained shell wrapper at `445cc9d0`. CPU checks are not scored training or BPB gains.
- Ready jobs: **zero**, not two per host yet. Proposals must not be counted as checked jobs.
- Use 2M screens only to triage; fresh public20M/context1024 and unchanged decide.py govern results.
  The active adoption threshold is mean delta `<= -0.0010`, not historical `-0.0008`.
- The local autonomous-loop service is inactive; its watchdog timer is active. A running unattended
  Bedrock scheduler has not been verified. Ideation cadence is now a 3–4-hour policy, not a claim
  of deployed scheduling.

- `launch_authorized` is `false`; no controller or launcher is active.
- The seeded queue is an exact copy of `reports/2026-09-24-queue-v2.json` and remains proposal-only.
- `ledger.jsonl` is empty.
- G1 source and paired-host metadata are recorded in `g1.json`; fresh checkpoint calibration remains pending.
- Independent G2 review and all native/Neuron gates remain pending.
- The old `research/agentic/autoloop/selection.py` selection/control loop is historical and inactive.
  Workflow v2 must not be run concurrently with it.
- Workflow-v2 selection requires a fresh-process context-1024 evaluation over 20,971,520 tokens and
  the unmodified `workflow/tools/decide.py` rules.

## Installed identities

| Artifact | SHA-256 |
|---|---|
| `04-train-research-35c9.py` | `35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2` |
| `05-train-submission-clean-116188.py` | `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d` |
| `submissions/research_v2/train.py` | `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` |
| `submissions/candidates/S0/train.py` | `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb` |
| `workflow/tools/decide.py` | `a47bedd258697b0791f48dcac3098d9071b5aaf00a7664d9e39987b4052adc66` |
| `research/v2/queue.json` | `1e9dcc4fd5f38e62829dc19460d2337662184ecfbd4d21d8a20f15993be9dd00` |

The imported CPU receipts and end-to-end transcript are upstream evidence. This installation did not
independently verify them and makes no native-hardware claim. See `UPSTREAM_PROVENANCE.json`.

## Current handoff (2026-09-24 20:27 UTC)

- Phase 0 exact-copy installation is integrated at `8b9fc841`; G1 facts at `d6629b61`.
- Main reran the upstream synthetic E2E successfully, but independent negative tests found real runner bugs:
  repeat failed labels could launch, effective environment/untracked source were not checked, and evaluation
  population was copied from config instead of verified from evaluator JSON. Reviewed repairs are in progress.
- Both hosts have the default five shards, 415 row groups and 424,960 documents. This is not a token count.
- Both retained organizer files match unchanged `prepare.py` SHA `5807c46e...`; retained calibration checkpoints
  exist. Fresh calibration has not run. Historical 20M evaluations took about 12 minutes each.
- Last live inventory: trn21 idle at 6.43 GiB free (below floor); trn22 idle at 7.46 GiB.
- The legacy local service was found auto-restarting and explicitly stopped (inactive/dead, PID 0). Its old
  breaker record was not reset. No instance or running training job was stopped.
- S0 and research-v2 passed independent static review at `0412d21f`; CPU/G2 and native checks are still required.
  The assigned Operator is preparing four baseline/S0 CPU checks, not a scored experiment.
- Best confirmed public20M remains single 0.9918353324, two-run mean 0.9919537735 on the older clean 116188 file.
  Installed S0 `ad223cdc...` is unscored and not ready to upload.
