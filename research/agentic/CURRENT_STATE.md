# Current state

Last reconciled: 2026-09-25 08:02 UTC

The disabled trn22 e65→r9 small-bundle stage candidate `1bbe2a38` received
[independent HOLD](../v2/verification/20260925-trn22-r9-incremental-source-stage-independent-HOLD.md)
at `002fe8b9`: synthetic late STOP/label injection after its final slow scan
still permits exclusive bundle and target creation; the label case can return
success. No host write or attempt occurred. A distinct repair is required
before any stage assignment. The trn21 r10 stage and exact-size A2 CPU-only G2
packages are being prepared/reviewed in parallel. Both launch flags remain
false; there is no new control, table score, S0 replay or BPB gain.

The [07:33 UTC dual-host Operator snapshot](../v2/observations/20260925-dual-host-readonly-operator/OPERATOR_READONLY_DUAL_HOST_SNAPSHOT_20260925T0733Z.md)
is archived at `7929cc2a` and has a [bounded independent PASS](../v2/verification/20260925-dual-host-readonly-final-pair-independent-PASS.md)
at `9868d9a6` for the final host pair and ACL follow-up. At that
instant both hosts had about 54 GiB free, no matched train/eval, Neuron-device
or host-lock holder, and clean e65 c5c A0 source checkouts with matching model
and evaluator hashes. This is point-in-time observation, not launch preflight or
authority; both launch flags remain false. The r9 trn22-only launcher has
source-only independent PASS. A disabled r10 trn21 profile was implemented at
`22bfbab5` and received [independent source-only PASS](../v2/launcher-safety/INDEPENDENT_SOURCE_VERIFIER_PASS_R10_TRN21.md)
at `cbdfb9d0`; its host-stage and native gates remain HOLD. A separate
Implementer is preparing a small-bundle, disabled trn22 source stage from its
existing e65 checkout. There has still been no new scored control, S0 cold
replay, table result or BPB gain. The Sep 25 noon-IST S0 target was missed.

The disabled trn22 final-r3 read-only timing diagnostic `4fd59da6` received
[independent HOLD](../v2/verification/20260925-trn22-fifo-final-r3-timing-independent-HOLD.md)
at `791a93ff`: a supplied-test `SyntaxError` prevents all eight tests from
loading, and completed trial timings could be lost if the 600-second SSH
timeout fires before the remote JSON emission. No host contact or timing
attempt occurred. The A0 launcher-source r9 repair `e046de16` has
[independent source-only PASS](../v2/launcher-safety/INDEPENDENT_SOURCE_VERIFIER_PASS_R9.md)
at `a669d4eb`, but its exact A0 profile is trn22-only and refuses trn21. A
disabled trn21 e65-to-r9 incremental stage draft (~99-KiB Git bundle) was
parked uncommitted because it cannot enable native trn21 A0. An exact trn21
r10 launcher-profile repair is under implementation and needs independent
review before a corrected stage package. No host transfer occurred.
Neither host has a new scored control, S0 replay, table result, or BPB win.

The operational launcher-source r8 successor `d3f39dd8` received an
[independent HOLD](../v2/launcher-safety/INDEPENDENT_SOURCE_VERIFIER_HOLD_R8.md)
at `b197f86c`: a request-controlled `PATH` can select an unbounded fake
`timeout`, and a shell wrapper can hide an optimized Python child. A distinct
r9 repair is under implementation. The disabled trn21 r6 transfer repair
`71d6701f` has
[independent static PASS](../v2/verification/20260925-trn21-r6-post-label-disabled-independent-static-PASS.md)
at `8f9e4402`, but dispatch remains HOLD because live one-second timing
feasibility is unknown. Neither changes the scored-run HOLD.

The distinct trn22 final-r3 transfer request `d571ea61` received an
[independent HOLD](../v2/verification/20260925-trn22-fifo-final-r3-independent-HOLD.md)
at `cbc2c03e`: its repaired STOP/Neuron timing checks pass synthetic safety
tests, but there is no live evidence that the serial post-label checks fit
the required one-second label-to-open window. A miss would safely prevent the
remote write while consuming the one-shot local attempt. No fixed final PASS,
Supervisor assignment, marker, SSH, host transfer, native run or new score
exists. A separate read-only timing diagnostic is being prepared; it is not
dispatch authority. The launcher r8 outcome is recorded above.

The Sep 25 noon-IST S0 submission-package target was missed. Neither of the
required cold replays has begun, and no new file is ready for human upload.
Distinct trn22 final-r2 and trn21 r5 final transfer-only candidates now have
[independent](../v2/verification/20260925-trn22-fifo-final-r2-independent-HOLD.md)
[HOLDs](../v2/verification/20260925-trn21-r5-enabled-final-transfer-independent-HOLD.md)
for STOP introduced during a slow remote final label scan before archive
creation. Neither has assignment or host action. A narrow
operational launcher-source successor `e74f9995` received independent
HOLD at `339c3009`: it still accepts an unbounded `timeout 0` and permits a
non-source fingerprint to reach the local SSH seam. A distinct source repair
is being prepared; model and scoring files remain unchanged. No native
control or table run has started.

The distinct trn22 FIFO/nonblocking-lock transfer successor now has
[independent disabled-only static PASS](../v2/verification/20260925-trn22-fifo-nonblock-5740-independent-static-review.md),
but remains dispatch HOLD pending a separately reviewed enabled request,
fresh host checks and one-shot Supervisor assignment. No trn22 source transfer
or scored run occurred. The trn21 final-r4 private-lock successor received
[independent static HOLD](../v2/verification/20260925-trn21-r4-private-lock-independent-static-HOLD.md)
because a below-floor terminal inode observation could falsely PASS the
preflight/postflight validator. Its distinct r5 repair now has
[independent disabled-only static PASS](../v2/verification/20260925-trn21-r5-inode-floor-disabled-independent-static-PASS.md),
but no enabled request, final review or host transfer. A distinct trn22
[final transfer-only r1 candidate](../v2/authorizations/trn22-c5c-a0-r4-transfer-lock-final-r1/RUNBOOK.md)
received [independent HOLD](../v2/verification/20260925-trn22-fifo-final-r1-independent-HOLD.md):
STOP could arise during slow local authorization/source rehash and still reach
marker or write-spawn. A distinct r2 repair is being prepared. There is no
assignment, host transfer, or scored work.
The trn22 A0 r5 native package now has
[independent static HOLD](../v2/verification/20260925-trn22-c5c-a0-r5-native-package-HOLD.md):
optimized Python strips an assertion-based disabled dispatch check in the
frozen launcher. Its offline reproducer used a fake SSH seam, not the host.
A distinct fail-closed successor is being prepared; r5 cannot launch.
The [r6 diagnostic packet](../v2/authorizations/trn22-c5c-a0-r6-optimization-hold/RUNBOOK.md)
also identifies assertion-based safety checks in the frozen remote launcher;
it is disabled/HOLD and cannot launch. A separate narrow operational source
update is underway, with model, data, evaluator and scoring bytes protected.
Best confirmed public20M is unchanged at 0.9918353324 single /
0.9919537735 two-run mean.

## Latest handoff: both transfer writers HOLD; CPU evidence archived

The assigned trn21 synthetic four-rank, three-step CPU-only G2 attempt ended
with return code zero and clean pre/postflight checks. The complete
[Operator receipt](../v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-final/operator-evidence-20260925T042828Z/OPERATOR_TERMINAL_RECEIPT.md)
and redacted raw journal are archived. Its
[independent numerical review](../v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-final/INDEPENDENT_CPU_G2_RESULT_REVIEW.PASS.json)
is **PASS_CPU_NUMERIC_ONLY**: all four ranks completed three finite steps,
62 parameter digests and the rank-0 checkpoint check match the pinned c5c
anchor and trn22 reference. This is not native-training or evaluation
authority. The review records a source-directory link-count change without
an archived child listing, and no separate step-log byte archive.

The trn21 clean-r4 transfer's
[independent final review](../v2/authorizations/trn21-c5c-a0-r4-transfer-only/INDEPENDENT_FINAL_ENABLEMENT_HOLD.md)
reproduced a late science-label race and placed that enabled package on HOLD.
A separate [final-r2 repair](../v2/authorizations/trn21-c5c-a0-r4-transfer-only/TRANSFER_REQUEST.FINAL_R2_PENDING_REVIEW.json)
also received a [Supervisor HOLD](../v2/authorizations/trn21-c5c-a0-r4-transfer-only/SUPERVISOR_FINAL_R2_HOLD.md)
after an independent Verifier found the label-continuity condition could fail.
Neither r1 nor r2 may dispatch. The trn22 r3 enabled request's earlier
[independent final PASS](../v2/authorizations/trn22-c5c-a0-r4-transfer-only/INDEPENDENT_CONTINUITY_FINAL_R3_REVIEW.json)
was [superseded by an independent HOLD](../v2/authorizations/trn22-c5c-a0-r4-transfer-only/INDEPENDENT_CONTINUITY_FINAL_R3_LATE_LABEL_HOLD_ADDENDUM.md)
after source analysis found the same label-timing gap. Its 04:58 UTC
[read-only host snapshot](../v2/authorizations/trn22-c5c-a0-r4-source-feasibility/OPERATOR_READONLY_FINAL_R3_FEASIBILITY_20260925T045814Z.md)
was clean, but is only point-in-time feasibility; it does not override the
writer HOLD. No transfer assignment may be issued for trn22 r3.
The newer trn22 [disabled lock-and-final-scan repair](../v2/authorizations/trn22-c5c-a0-r4-transfer-lock-final-scan-hold/GATE_CONTRACT.HOLD.json)
has an [independent static HOLD](../v2/authorizations/trn22-c5c-a0-r4-transfer-lock-final-scan-hold/INDEPENDENT_LOCK_FINAL_SCAN_STATIC_HOLD.md):
its runbook misstates a transport hash, and its group-write directory policy
rejects the host's observed `0775` lock parent. A separate read-only
[lock feasibility check](../v2/authorizations/trn22-c5c-a0-r4-source-feasibility/OPERATOR_LOCK_READONLY_FEASIBILITY_20260925T051136Z.md)
did acquire and release the existing lock without a metadata change, but
does not clear that policy HOLD. On trn21, a separate
[disabled final-r3 repair](../v2/authorizations/trn21-c5c-a0-r4-transfer-only/TRANSFER_REQUEST.FINAL_R3_HOLD.json)
received [independent static PASS for its disabled form only](../v2/authorizations/trn21-c5c-a0-r4-transfer-only/INDEPENDENT_FINAL_R3_DISABLED_STATIC_REVIEW.md).
Its [read-only lock/group observation](../v2/authorizations/trn21-c5c-a0-r4-source-feasibility/OPERATOR_LOCK_READONLY_FEASIBILITY_20260925T053029Z.md)
found private `.cache` mode `0700`, but the lock parent `0775` and lock file
`0664` cause r3 to fail closed. Its separate
[read-only ACL observation](../v2/authorizations/trn21-c5c-a0-r4-source-feasibility/OPERATOR_GROUP_ACL_READONLY_20260925T053728Z.md)
found no POSIX ACL grants along the private ancestry and no other local
GID-1000 account; alternate privileged paths remain unproved.
trn22's separate [read-only ACL observation](../v2/authorizations/trn22-c5c-a0-r4-source-feasibility/OPERATOR_GROUP_ACL_READONLY_20260925T053454Z.md)
found no POSIX ACL grant on its private `.cache` ancestry and no other local
GID-1000 account; that is bounded evidence for a new repair, not authority.
Its [new disabled private-cache repair](../v2/authorizations/trn22-c5c-a0-r4-transfer-lock-private-cache-hold/GATE_CONTRACT.HOLD.json)
received another [independent static HOLD](../v2/authorizations/trn22-c5c-a0-r4-transfer-lock-private-cache-hold/INDEPENDENT_DISABLED_STATIC_REVIEW.HOLD.md):
a pre-existing FIFO at the lock pathname can block the read-only open before
the type check. A distinct disabled successor is being prepared. The prior
trn22 static HOLDs continue to control.
On trn21, a [distinct disabled final-r4 private-lock successor](../v2/authorizations/trn21-c5c-a0-r4-transfer-only/TRANSFER_REQUEST.FINAL_R4_HOLD.json)
is under independent static review; its authorizations remain false. A
[disabled trn22 A0 r5 control package](../v2/authorizations/trn22-c5c-a0-r5-native-hold/GATE_CONTRACT.json)
is also under separate independent review for use only after an accepted r4
source checkout. Neither package starts a scored run.
Neither review authorizes an enabled request, host transfer, or native launch.
Neither r4 transfer has touched a host. Both scored controls, both S0 cold replays, and
table runs remain unstarted; best confirmed public20M is unchanged at
0.9918353324 single / 0.9919537735 two-run mean. The Sep 25 noon-IST S0
package target is at risk and must not be represented as ready.

## New handoff: trn22 final transfer HOLD on two reproduced defects

An [independent final review](../v2/authorizations/trn22-c5c-a0-r4-transfer-only/INDEPENDENT_CONTINUITY_FINAL_R2_HOLD.md)
of the repaired trn22 r4 transfer request is HOLD. Its adversarial tests
reproduced two blockers: Neuron activity arising during the final identity
audit did not prevent target creation, and file-backed old `.git` paths caused
`NotADirectoryError` before the full read-only host probe could produce a
receipt. A separate Implementer is repairing these without overwriting the
HOLD evidence. No final PASS or Supervisor transfer assignment exists, and no
trn22 transfer occurred. The trn22 Operator's read-only feasibility probe
reached the host and found it otherwise idle with roughly 54 GiB free, but
that point-in-time observation is not a transfer preflight or authority.

## New handoff: trn21 CPU-only G2 assigned; scored training still HOLD

The exact trn21 stage-r2 [CPU-only G2 final request](../v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-final/RUNBOOK.md)
has [independent final static PASS](../v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-final/INDEPENDENT_FINAL_CPU_G2_REVIEW.PASS.json).
The Supervisor issued the [one-attempt CPU-only assignment](../v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-final/SUPERVISOR_CPU_ONLY_ASSIGNMENT.json)
and locally authenticated its exact review/assignment byte chain. An Operator
must still perform fresh live preflight, the contained four-rank synthetic
three-step probe and a separate full read-only postflight. No trn21 CPU probe,
native training, evaluation or new BPB result has occurred yet. The old e65
checkout remains forbidden for native launch; r4 source staging is separate.

The disabled trn21 r4 transfer implementation also received
[independent static PASS](../v2/authorizations/trn21-c5c-a0-r4-transfer-only/INDEPENDENT_STATIC_TRANSFER_REVIEW.md)
but remains DISPATCH HOLD pending a separate enabled request, final review,
fresh host checks and one-shot assignment. On trn22, a repaired exact enabled
transfer request is under separate final review; no transfer has occurred.

## Current handoff: repaired transfer static PASS; final authority still HOLD

The trn21 [CPU-only G2 request](../v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-request/RUNBOOK.md)
received [independent offline PASS for its disabled binding only](../v2/verification/20260925-trn21-c5c-a0-g2-request-independent-review.md).
An [enabled CPU-only final request candidate](../v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-final/RUNBOOK.md)
is now under a separate independent review; its actual PASS file and Supervisor
assignment do not exist. Fresh exact-root host checks and four-rank numeric
result remain pending.
The [clean 446b source-stage plan for trn21](../v2/authorizations/trn21-c5c-a0-r4-source-stage-hold/RUNBOOK.md)
and its [transfer-only implementation](../v2/authorizations/trn21-c5c-a0-r4-transfer-only/RUNBOOK.md)
are disabled and under independent review; they preserve the accepted e65
checkout. No native launch may use e65's older safety launcher.

The trn22 [r4 transfer enablement candidate](../v2/authorizations/trn22-c5c-a0-r4-transfer-only/FINAL_ENABLEMENT_RUNBOOK.md)
received an [independent final HOLD](../v2/authorizations/trn22-c5c-a0-r4-transfer-only/INDEPENDENT_FINAL_ENABLEMENT_HOLD.md).
Adversarial tests found that preflight boot/mount identity was not bound to
the write session and a host STOP inserted after the probe did not prevent
target creation. A separate Implementer repaired both in a disabled package;
that package now has [independent static PASS](../v2/authorizations/trn22-c5c-a0-r4-transfer-only/INDEPENDENT_CONTINUITY_REPAIR_STATIC_REVIEW.md)
with 36 passing offline tests. A new exact enabled request is being prepared.
There is still no final PASS authority file or Supervisor assignment, and no
trn22 host transfer occurred.
Both scored controls, S0 cold replays and table runs remain unstarted; no
new BPB improvement is confirmed. The Sep 25 noon-IST S0 package target is
at risk and must not be declared ready without its cold replays.

## Current safety correction: trn21 staged source is not the native-launch source

The accepted trn21 checkout at `e65a3097` proves a clean c5c training/eval
source, but it does **not** authorize native A0. A read-only source comparison
against the independently reviewed trn22 r4 source `446b6576` found identical
`train.py` and `prepare.py` blobs, but `e65a3097` predates the safety-monitor
revision: `research/fresh_start_launch.py`, `fresh_start_guard.py`, and
`fresh_start_host.py` differ, and `fresh_start_monitor_interlock.py` plus both
`research/v2/disk-monitor/*guard_monitor.py` files are absent. Running the old
launcher would bypass the reviewed paired disk/STOP interlock. The trn21
exact-environment CPU G2 may proceed only under a separately reviewed CPU-only
request; no native launch may use the e65 checkout. A distinct clean 446b
source-stage/checkout and source-matched native request, review, fresh host
preflight, live monitors/observer, and one-shot assignment are required.
Neither host has produced a new scored control or BPB improvement.

## Current handoff: trn21 checkout assigned once; no training authorization

The trn21 [final exact checkout package](../v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/GATE_CONTRACT.FINAL_PENDING_REVIEW.json)
[passed independent offline review](../v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/INDEPENDENT_FINAL_REQUEST_REVIEW.PASS.json),
and the Supervisor issued a [checkout-only one-attempt assignment](../v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/SUPERVISOR_CHECKOUT_ONLY_ASSIGNMENT.json).
The Verifier separately checked all assignment bindings locally; the actual
review receipt is owner-owned, mode `0644`. The Operator's
[one-attempt checkout receipt](../v2/authorizations/trn21-c5c-a0-stage-r2-checkout-result/OPERATOR_CHECKOUT_ONLY_TERMINAL_RECEIPT.md)
reports exact e65/c5c source staged in the new checkout, three SSH phases
with zero return codes and empty stderr, a distinct full read-only
postflight, and intact protected assets. The fixed exclusive marker is now
consumed. A different Verifier
[accepted the archived result for checkout only](../v2/authorizations/trn21-c5c-a0-stage-r2-checkout-result/INDEPENDENT_CHECKOUT_RESULT_REVIEW.PASS.json).
This clears source staging, **not** CPU G2, native training, evaluation or
adoption. The [safety-hardened CPU G2 successor](../v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2-final-pending/GATE_CONTRACT.json)
[passed independent static review](../v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2-final-pending/INDEPENDENT_STATIC_REVIEW.md)
for its disabled form. A new exact CPU-only request, separate review, fresh
host preflight and one-attempt assignment remain required. No new BPB was
measured.

The trn22 [r4 transfer-only draft](../v2/authorizations/trn22-c5c-a0-r4-transfer-only/TRANSFER_REQUEST.HOLD.json)
received an [independent static HOLD](../v2/authorizations/trn22-c5c-a0-r4-transfer-only/INDEPENDENT_IMPLEMENTATION_REVIEW.md):
SSH host identity was not explicitly pinned, local STOP was ignored, review
approval was self-attested, and the root-device alias needs read-only
reconciliation. A separate Implementer repaired the disabled package; it
[passed independent static re-review](../v2/authorizations/trn22-c5c-a0-r4-transfer-only/INDEPENDENT_REPAIRED_STATIC_REVIEW.md).
A new exact enabled transfer-only request, separate final review, fresh host
preflight and Supervisor assignment are still required. No transfer occurred.
No trn22 transfer, checkout or training is authorized. Both scored A0
controls remain pending; best confirmed public20M remains `0.9918353324`
single / `0.9919537735` two-run mean.

## Current handoff: trn21 repaired checkout draft PASS offline; enabled request HOLD

The trn21 stage-r2 [repaired checkout draft](../v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/GATE_CONTRACT.json)
[passed independent offline review](../v2/verification/20260925-trn21-stage-r2-checkout-repair-static-review.md)
for its disabled form: a fixed exclusive/fsynced attempt marker blocks
same-assignment replay, and the wrapper verifies the actual future review
receipt before host contact. This does **not** authorize checkout; a final
enabled exact request, separate independent final review, fresh full host
preflight and new one-shot Supervisor assignment are still required. That
enabled package is being prepared. No host checkout occurred.

The trn21 [stage-r2 CPU G2 draft](../v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2/GATE_CONTRACT.json)
also [passed static planning review](../v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2/INDEPENDENT_STATIC_REVIEW.md)
only. Its builder does not yet enforce three final safety pins (one attempt
per label, CPU tool mode, and preflight inode); the final runnable request
must fix them after a verified checkout. No CPU G2, native run or BPB result.

## Current handoff: trn22 read-only feasibility planning PASS; transfer HOLD

The trn22 Operator's [read-only feasibility snapshot](../v2/authorizations/trn22-c5c-a0-r4-source-feasibility/OPERATOR_READONLY_FEASIBILITY_20260925T024228Z.md)
and [independent planning review](../v2/authorizations/trn22-c5c-a0-r4-source-feasibility/INDEPENDENT_PLANNING_VERIFIER_RECEIPT.md)
PASS for the point-in-time host condition: 58,453,950,464 B root free,
planned targets absent, protected checkpoint/caches intact, and no matched
process, device holder, STOP or breaker. This is **not** a transfer preflight
or authority. The next transfer request must capture raw SSH argv/transport,
reconcile `/dev/nvme0n1p1` versus `/dev/root` mount aliases, inventory old
r1/r2/r3 paths and redo all live safety checks immediately before any write.
Transfer-only package implementation is in progress. No trn22 source transfer,
checkout, G2, native control or BPB result occurred.

## Current handoff: trn22 exact-source capsule static PASS; host stage HOLD

The trn22 r4 [exact-source capsule plan](../v2/authorizations/trn22-c5c-a0-r4-source-stage/GATE_CONTRACT.json)
has [independent offline static PASS](../v2/authorizations/trn22-c5c-a0-r4-source-stage/INDEPENDENT_STATIC_VERIFIER_RECEIPT.md)
for the frozen 446b source and 6,368-member capsule only. The plan's Git
commit is on a divergent branch, so a future transfer must authenticate the
capsule HEAD and pinned file hashes directly, not substitute the package
commit. A read-only trn22 feasibility check and a separate disabled
transfer-only implementation are in progress. There has been no host transfer,
checkout, G2, native control or new BPB. Native dispatch remains HOLD.

## Current handoff: trn21 checkout request independently HOLD; repair required

The exact trn21 stage-r2 [checkout-request draft](../v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/GATE_CONTRACT.json)
is disabled. Its [independent request review](../v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/INDEPENDENT_REQUEST_REVIEW.md)
found two checkout-ready blockers before host use: the same assignment can
reach stage SSH twice with different journal paths after a pre-creation
failure, and the wrapper accepts an unauthenticated review hash. The frozen
draft and negative tests are archived as a HOLD, not launch authority.
An Implementer is preparing a repaired request; it needs a new independent
review and one-shot Supervisor assignment before any checkout. The earlier
transfer-only PASS and read-only feasibility remain valid only for their
limited scope. No trn21 checkout, CPU G2, native run or new BPB occurred.

Separately, a trn22 r4 exact-source capsule and HOLD-only stage plan have
been prepared in an isolated branch for independent static review. Nothing
has been transferred to trn22; its native dispatch remains HOLD.

## Current handoff: trn22 CPU numeric G2 carries forward; host/native gates HOLD

A separate Verifier's
[source-identity disposition](../v2/disk-monitor/r4-hold/G2_R3_TO_R4_SOURCE_IDENTITY_DISPOSITION.md)
accepted trn22's earlier four-rank CPU G2 **numeric** result for frozen r4:
`train.py`, `prepare.py`, ordered Track-A training argv, effective seed/LNC/
compiler/budget values and CPU harness bytes match. This avoids a redundant
CPU-numeric rerun if the eventual r4 checkout/request matches those exact
bytes and values. It is not a host r4 execution or safety-monitor proof.
Exact r4 source transfer/checkout, fresh host checks, new dispatchable
request/bindings, live paired monitors/observer and one-shot native authority
remain HOLD. No new BPB was measured.

## Current handoff: trn21 read-only feasibility refreshed; exact checkout still HOLD

At 02:19:45 UTC, the trn21 Operator's
[read-only refresh](../v2/authorizations/trn21-c5c-a0-stage-r2-extraction/OPERATOR_READONLY_FEASIBILITY_REFRESH_20260925T021945Z.md)
found all three independently verified persistent files unchanged, the new
checkout and science output absent, protected checkpoints/caches intact,
no process/device/lock holder, and 58,571,898,880 B root free. This is a
point-in-time observation only; a later checkout still needs its own fresh
preflight, reviewed exact request and Supervisor one-attempt assignment.
No host write, G2, native run or BPB result occurred.

## Current handoff: trn22 r4 disabled package static PASS; source transfer is next

The trn22 r4
[disabled, hash-bound package](../v2/disk-monitor/r4-hold/GATE_CONTRACT.json)
has [independent static PASS](../v2/disk-monitor/r4-hold/INDEPENDENT_VERIFIER_RECEIPT.md)
for consistency only. The request remains `dispatch_authorized=false`; its
base and pair bindings are `STATIC_ONLY`. None can launch or arm a monitor.
The next trn22 stage is a separate exact-source artifact/manifest, read-only
host feasibility, independently reviewed transfer/extraction plan and narrow
Supervisor source-stage assignment. Source-matched G2, fresh host checks and
native one-shot authority remain later gates. No new scored control or BPB.

## Current handoff: trn21 checkout implementation PASS; exact request still HOLD

The frozen trn21 r2 checkout package
[passed independent local implementation review](../v2/authorizations/trn21-c5c-a0-stage-r2-extraction/INDEPENDENT_IMPLEMENTATION_REVIEW.md).
The earlier three-file transfer already passed its separate result review.
Neither PASS is host checkout authority. An Implementer is building a
non-dispatchable exact one-attempt request that must include a fresh complete
preflight and **read-only full postflight even on extraction failure**. That
request needs separate review and Supervisor checkout-only assignment before
any Operator action. A failed attempt must leave any partial checkout intact
without cleanup/retry. CPU G2, native training and eval remain HOLD; no BPB
changed.

## Current handoff: trn22 r4 source safety PASS; exact launch bindings still HOLD

The frozen trn22 r4 repair at `446b657695f086f9b484fa601c9a3b61c2495912`
[passed independent local source safety review](../v2/disk-monitor/R4_446B6576_INDEPENDENT_VERIFIER_RECEIPT.md):
72 implementation tests and four independent adversarial tests pass. The
review covers source behavior only. The checked-in r3 request still pins older
code and is not dispatchable; an exact r4 package, independent package review,
source-matched G2 disposition, fresh host preflight and one-shot authority are
still missing. A separate Implementer is preparing HOLD-only bindings. No
native run or BPB result follows from this PASS.

## Current handoff: trn21 persistent transfer independently PASS; checkout review pending

The trn21 persistent-ext4 r2 three-file transfer
[passed independent result review](../v2/authorizations/trn21-c5c-a0-stage-r2/INDEPENDENT_TRANSFER_RESULT_RECEIPT.md).
The Verifier rehashed the raw Operator records and confirmed exact SHA/byte,
distinct-process persistence and final postflight evidence for all three files.
This accepts **transfer only**, not a staged checkout or a live host preflight.
The [checkout-only package](../v2/authorizations/trn21-c5c-a0-stage-r2-extraction/GATE_CONTRACT.json)
is now under separate implementation review; extraction, CPU G2, native
training and evaluation remain HOLD. No science label was consumed or new BPB
measured. The old `/dev/shm` disappearance remains unexplained.

trn22's frozen r4 safety repair still awaits independent review. Neither host
has a scored c5c control; best confirmed public20M remains 0.9918353324
single / 0.9919537735 two-run mean.

## Current handoff: trn21 transfer complete at Operator level; independent result review pending

The trn21 persistent-ext4 r2 transfer-only Operator archived all three new
targets and reported distinct-SSH persistence checks for each plus a final
read-only postflight. Its [terminal receipt](../v2/authorizations/trn21-c5c-a0-stage-r2/operator-transfer-only-terminal-receipt.json)
is durable in the shared branch, but a separate Verifier is now auditing the
raw evidence. This is **not yet an independently accepted source stage**.
Checkout/extraction, CPU G2, native training, and evaluation remain HOLD; no
science label was consumed and there is no new BPB result. A separate
[checkout-only package](../v2/authorizations/trn21-c5c-a0-stage-r2-extraction/GATE_CONTRACT.json)
exists locally but is non-dispatchable pending independent implementation review,
transfer-result review, and a new exact Supervisor assignment.

For trn22, the next r4 disk-interlock repair is frozen at source commit
`446b657695f086f9b484fa601c9a3b61c2495912` with local synthetic tests
passing; its **independent** safety review is pending. The older 11379083
failure remains controlling until that review passes. No native dispatch is
authorized. Best confirmed public20M remains 0.9918353324 single /
0.9919537735 two-run mean; the table hypothesis remains untested on the dense
stack.

## Current handoff: trn21 transfer-only r2 assigned; trn22 r4 still unsafe

The trn21 [persistent-ext4 source-stage r2 package](../v2/authorizations/trn21-c5c-a0-stage-r2/GATE_CONTRACT.json)
passed [independent static review](../v2/authorizations/trn21-c5c-a0-stage-r2/INDEPENDENT_VERIFIER_RECEIPT.md)
for planning only. A new [Supervisor transfer-only assignment](../v2/authorizations/fs-20260925-v2-c5c-a0-control-trn21-r1.source-stage-r2-transfer-assignment.json)
now permits the Operator to write only three exact new ext4 targets after fresh
full preflight, then prove persistence in distinct SSH sessions. Checkout,
extraction, CPU G2, native training and evaluation remain HOLD; no transfer
result has been retrieved yet. The prior `/dev/shm` target is not retried.

On trn22, two frozen r4 interlock revisions received independent
[999b FAIL](../v2/disk-monitor/R4_999B_MARKER_SPOOF_INDEPENDENT_RECEIPT.md)
and [f8fe FAIL](../v2/disk-monitor/R4_F8FE_INDEPENDENT_VERIFIER_RECEIPT.md)
receipts. The next frozen 11379083 source passed 63 local tests but also
[failed independent review](../v2/disk-monitor/R4_11379083_INDEPENDENT_VERIFIER_RECEIPT.md):
a markerless changed-session worker can yield a false clean finish, and early
launcher refusal lacks a durable no-spawn receipt. An Implementer is repairing
these gaps. No native dispatch is authorized.
There is no new scored control or BPB result; best confirmed public20M remains
0.9918353324 single / 0.9919537735 two-run mean.

## Current handoff: trn21 source transfer did not persist; both scored controls HOLD

The narrowly assigned trn21 c5c A0 [source stage](../v2/authorizations/fs-20260925-v2-c5c-a0-control-trn21-r1.source-stage-assignment.json)
stopped on a [failed persistence check](../v2/authorizations/fs-20260925-v2-c5c-a0-control-trn21-r1.source-stage-hold.md).
The pinned 104,005,787-byte capsule hashed correctly inside its transfer SSH
process at 00:44:23 UTC, but the exact `/dev/shm` target was absent in a fresh
SSH process about five seconds later. The host did not reboot; the transfer
session's mount namespace was not captured, so the cause remains unknown. Three
more read-only sessions [confirmed stable follow-up namespace and absence](../v2/authorizations/fs-20260925-v2-c5c-a0-control-trn21-r1.source-stage-forensic-addendum.md);
the remote writer source bytes were not retained and cannot be audited.
There was no checkout, CPU G2, native training, evaluation, or new score. The
stage path may not be retried under its prior assignment; a revised transfer
plan needs independent review and fresh authority. trn22 A0 r3 also remains
HOLD: its [independent interlock review](../v2/disk-monitor/R3_INDEPENDENT_VERIFIER_RECEIPT.md)
reproduced two emergency-stop failures, so a separate Implementer must repair
it before another review and exact-source CPU G2. The diagnostic-only table
rank-hash package has [static planning PASS](verification/20260925-c5c-table-hash50-static-review.md)
but no host or quality result. Best confirmed public20M remains 0.9918353324
single / 0.9919537735 two-run mean.

## Current handoff: trn21 source-stage feasibility independently reviewed

The [trn21 c5c A0 read-only feasibility receipt](../v2/authorizations/fs-20260925-v2-c5c-a0-control-trn21-r1.readonly-feasibility.json)
at 00:12:54 UTC reports 58,676,293,632 B root free, intact protected
checkpoints/caches, no matched job or Neuron/lock holder, and an unused A0
label. Its limited scan found no exact e65 source commit in 144 inspected
repositories under `/home/ubuntu`; this is not proof of host-wide absence.
The [independent review](verification/20260925-trn21-c5c-a0-r1-readonly-feasibility-review.md)
passed **only source-stage planning**. The separately prepared
[exact-source/CPU-G2 package](../v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-r1/GATE_CONTRACT.json)
is still HOLD pending independent package review and a separate Operator
assignment. No source transfer, CPU G2, native A0, new score or launch
authority follows. trn22 remains held for the reviewed fail-closed disk
monitor/interlock. Best public20M remains 0.9918353324 single / 0.9919537735
two-run mean.

## Current handoff: trn21 G1 independently PASS; both scored controls pending

The guarded trn21 G1 r2 retained-checkpoint calibration [archived its full
Operator evidence](../v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/operator-receipt.json)
and [passed independent result review](verification/20260925-g1-trn21-r2-calibration-result-review.md).
Its fresh public20M/context-1024 BPB exactly reproduced the old checkpoint's
0.9918353324459732; the guard returned 0, monitored disk with no stop, and
two stable postflight checks found the host clean. This proves that trn21's
scoring setup works; it is **not a new model result** or A0 training authority.
Both hosts now have host-specific G1 calibration PASS. trn21's c5c A0 control
is [conditionally ratified](../v2/authorizations/20260925-trn21-c5c-a0-r1-proposal-ratification.json)
but still needs an exact clean source stage, four-rank G2, independent review,
fresh preflight, safety monitor/observer, and one-shot authorization. An
Implementer is preparing a non-dispatchable stage/G2 package; no trn21 native
training has started. trn22's c5c A0 r2 remains held while the external disk
monitor gains a fail-closed liveness mechanism and independent review. Neither
host has a new scored control or BPB improvement; best public20M remains
0.9918353324 single / 0.9919537735 mean.

## Historical handoff: trn21 evaluator running; trn22 A0 monitor hold

The one-label trn21 G1 r2 guarded public20M retained-checkpoint evaluator
launched at 23:41:16 UTC after a fresh PASS preflight and the corrected
[exact authorization](../v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/request.guard-authorized.json).
The Operator reports guard PID 2835264 alive with 1-second disk samples and no
stop as of 23:44:31 UTC. There is no evaluator result or G1 PASS until the
archived postflight passes independent review; this is not a new model run.

On trn22, the c5c A0 r2 [final request passed independent static review](verification/20260925-trn22-c5c-a0-r2-final-request-review.md)
and a [fresh read-only prelaunch audit](../v2/authorizations/fs-20260925-v2-c5c-a0-control-trn22-r2.prelaunch-readonly.json)
passed with 58,452,860,928 B root free. Native dispatch is explicitly
[HOLD](../v2/authorizations/fs-20260925-v2-c5c-a0-control-trn22-r2.monitor-hold.json):
the proven launcher has no reviewed continuous disk monitor, and the earlier
attempt crossed the 6.5-GiB floor before step zero. An Implementer is preparing
an external <=1-second, authenticated owned-guard monitor with a 2-GiB reserve;
it will need independent review before one-shot authorization. The r2 science
label is unused. No native run or BPB improvement exists; best public20M remains
0.9918353324 single / 0.9919537735 mean.

## Historical handoff: trn22 CPU gate archived; trn21 evaluation stage verified

The corrected exact-environment, four-rank CPU G2 on trn22 [passed its Operator
probe](../v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn22-r3/operator-receipt.json)
and [independent numerical comparison](verification/20260925-trn22-c5c-a0-r2-g2-r3-numeric-review.md)
against the prior c5c anchor, with clean postflight and unchanged protected
artifacts. This is not native training or a BPB measurement. Its A0 final
request and fresh host preflight are being staged, with dispatch still HOLD.
The trn21 retained-checkpoint G1 r2 clean source stage
[passed independent review](verification/20260925-g1-trn21-r2-staged-request-review.md),
but its evaluator has not launched. A separately guarded, one-label authorization
and fresh preflight are next. Both hosts remain barred from scored training,
and the best measured public20M BPB remains 0.9918353324 single / 0.9919537735
two-run mean. The older sections below remain historical context.
The repaired A2 30-step native-smoke package [passed independent static review
for staging only](verification/20260925-v2-c5c-a2-native30-r2-static-review.md);
native dispatch and the separate 50-step table/moment hash proof remain HOLD.

## Current handoff: post-resize staging and CPU preflight, no new score

The trn22 c5c A0 r2 source-staging check passed on a new clean checkout, with
58,452,987,904 B free after its four-rank CPU probe. That exact-environment G2
[failed for infrastructure only](../v2/g2-probes/fs-20260925-v2-c5c-a0-control-trn22-r2/operator-stage-hold.json):
rank 0 could not append to the future science step log after synthetic step 0
because its parent did not exist in the clean checkout. There were no native
training steps. The G2 label is consumed; the A0 r2 science label remains unused.
The separate r3 CPU-only retry [passed independent safety review](verification/20260925-trn22-c5c-a0-r2-g2-r3-static-safety-review.md)
and a distinct Operator is assigned the fresh host checks and CPU probe.
No Neuron run,
checkpoint, BPB or new model verdict exists. On trn21, the G1 r1 stage remains
HOLD after the absent pinned commit/fetch failure; an r2 package to clone the
exact retained trained commit locally has independent static PASS and
[Operator staging passed](../v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/operator-stage-receipt.json).
Evaluator dispatch remains HOLD pending independent stage review, fresh live
checks and one-shot authority. Both hosts
still have launch authorization false and ready scored-job counts zero.
The best confirmed public20M stays 0.9918353324 single/0.9919537735 mean.
An independent review of the conditional A2 native-smoke package
[failed static gating](verification/20260925-v2-c5c-a2-native30-static-review.md)
because it did not authenticate the step-log parent and had not reconciled the
older 50-step per-rank-hash criterion. The Implementer is repairing the
package; there is no A2 host run or BPB result.
The storage-fixed section below supplies context and remains authoritative for
the disk expansion and trn22 breaker re-arm.

## Current handoff: storage fixed; trn22 breaker re-armed; no launch yet

The human completed both guest root-partition/ext4 expansions. The
[independent read-only host audit](operations/TRN2_POST_RESIZE_READ_ONLY_AUDIT_2026-09-25_0356.json)
at 22:26:14 UTC found 250-GiB root disks, grown p1 and ext4 root filesystems,
and 58,907,762,688 B free on trn21 / 58,454,982,656 B on trn22. Both exceed
the immutable 6.5-GiB floor and the provisional 10-GiB A0 prelaunch gate.
No matched train/eval process, Neuron holder, or host-lock holder was found.
This is a storage/quiescence finding, **not** a full preflight or launch approval.

trn21's next gate is the independently reviewed retained-checkpoint G1
public20M calibration. Fresh disk, checkpoint, six warm caches, tokenizer and
idle checks passed, but staging [stopped safely](../v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r1/operator-stage-hold.json)
because the pinned commit is missing on the host and Git fetch returned 128.
An exact-source transfer is being prepared; no evaluator was launched. The
static-reviewed trn21 A0 control then needs G1, exact-environment G2 and live
gates. The human conditionally [approved](../v2/authorizations/20260925-trn22-disk-breaker-human-ack.md)
re-arming trn22's named disk breaker; a fresh [independent safety review](verification/20260925-trn22-disk-breaker-rearm-review.md)
passed, and that workflow-v2 breaker has been cleared. The previous A0 r1
remains a terminal zero-step disk incident with no BPB and unchanged failed
postflight. trn22 launch remains disabled pending A0 r2's own G2, staging,
preflight, observer and one-shot authority. The repaired A2 table package now
has an independent [static PASS / dispatch HOLD](verification/20260925-v2-c5c-a2-m64d128-trn22-r1-static-r2-review.md);
it has no native or BPB result. Neither host has a new model result or a
submission-ready S0 file.
Best confirmed public20M remains 0.9918353324 single/0.9919537735 mean.
This handoff supersedes the pre-resize sections below.

## Historical handoff: both hosts disk-blocked after A0 compile safety stop

The human's Melbourne AWS console screenshot shows both exact root EBS volumes
at 250 GiB. A [read-only guest audit](operations/TRN2_EBS_GUEST_PENDING_AUDIT_2026-09-25_0351.json)
at 22:21:36 UTC confirms the 250-GiB parent disks are visible on both hosts,
but their root p1 ext4 partition/filesystem remains 200-GiB-sized. Root free
space is still below the immutable floor: trn21 6,901,698,560 B; trn22
6,448,898,048 B. The human is being guided through guest `growpart` and
`resize2fs`. No agent has resized the guest filesystem. trn22's breaker and
all launch holds remain in place pending independent post-resize checks.

At 22:05:10 UTC a new [read-only guest disk audit](operations/TRN2_DISK_READ_ONLY_AUDIT_2026-09-25_0335.json)
still found 200-GiB root disks, with 6,901,743,616 B free on trn21 and
6,448,918,528 B on trn22—both below the 6.5-GiB operational floor. The audit
did not check AWS-side volume state or repeat process quiescence. No new
training or BPB result occurred. The trn22 c5c A0 r2 retry now has an
[independent static PASS / dispatch HOLD](verification/20260925-v2-c5c-a0-trn22-r2-static-review.md),
not launch authorization. Its exact-env G2 and live gates must wait for human
EBS expansion and an explicit trn22 breaker reset. The trn21 G1
[pending request](../v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r1/request.pending.json)
has [independent static PASS / dispatch HOLD](verification/20260925-g1-trn21-pending-package-review.md)
but is not dispatchable until host/storage, checkpoint/cache/environment and
one-shot authority gates pass. An independent [S0 static audit](verification/20260925-s0-submission-static-recheck.md)
found no proven rule violation, but native checks and two cold bare-command
replays remain before the human can safely upload it.
The trn21 c5c A0 [pending control package](../submission-validation/20260925-v2-c5c-a0-control-trn21-r1/GATE_CONTRACT.json)
also passed [independent static review](verification/20260925-v2-c5c-a0-trn21-r1-static-review.md)
but is still dispatch HOLD pending storage, trn21 G1, exact-env G2, live
staging and one-shot authority. It has no BPB result.
The newer human brief schedules A1/trn21 and A2/trn22 in parallel once each has
a valid same-host control and exact gates; this supersedes the older A2-after-A1
schedule but not the need for same-host A1 to attribute the incremental layer-1
effect. See the [table gate brief](../v2/TABLE_GATE_BRIEF_2026-09-25.md).

The one authorized warm c5c A0 trn22 run was stopped after root free space fell
below the 6.5-GiB operational floor **before any optimizer step**. The authenticated
guard ended at 21:28:07 UTC with rc 1; no owned processes or cleanup actions remain.
The independent timing observer saw zero boundaries and reported NON_SCORED. No
checkpoint/public20M result or quality verdict exists; this label is consumed.
The hash-verified [terminal receipt](../v2/incidents/fs-20260925-v2-c5c-a0-control-trn22-r1/operator-terminal.json)
and [independent safety review](verification/20260925-a0-trn22-disk-safety-review.md)
are now archived. Compile-time free-space consumption was 1,500,942,336 bytes;
trn22's immediate post-incident free space was 6,448,951,296 bytes. A later
read-only Operator inventory at 21:50:53 UTC found 6,448,939,008 bytes free
on trn22 and 6,901,788,672 on trn21; both guests still expose 200-GiB root
disks, with no train process or Neuron holder. AWS-side resize status was not
queried.

trn21 and trn22 are both below the disk floor. The exact-size c5c A1 table-gate
package now has an [independent static PASS / dispatch HOLD](../submission-validation/20260925-v2-c5c-a1-m64d128-trn21-r1/verifier.json);
its four-rank/native checks and BPB run have not happened. The [disk packet](../v2/DISK_DECISION_PACKET_2026-09-25.md)
asks the human to expand both existing root volumes by 50 GiB, then acknowledge the
trn22 disk breaker after the reviewed diagnosis. No AWS action, deletion, new launch,
relaunch or breaker reset has occurred. The best confirmed public20M remains
0.9918353324 single/0.9919537735 mean. The current handoff supersedes the earlier
active-A0 text below.

## Current handoff: one warm A0 control running on trn22

The exact `c5c410af...` A0 dense baseline was guarded-dispatched once on trn22 at
21:25:57 UTC, label `fs-20260925-v2-c5c-a0-control-trn22-r1`, guard PID `2963679`.
Its one-run authorization and staged evidence are in `research/v2/authorizations/` and
`research/v2/fs-20260925-v2-c5c-a0-control-trn22-r1.operator-stage.json`. The read-only
boundary observer was started before launch. The Operator is monitoring disk, step
boundaries, loss and guard; **no scored result exists yet**. Do not dispatch a second trn22
job until terminal postflight and quiescence. This run is a baseline, not a quality win.

Warm S0 has a separate static-reviewed package on
`experiment/20260925-s0-warm-native-trn22-r1`, but its 8,833,482,752-byte prelaunch disk
requirement blocks it now; see `research/agentic/verification/20260925-s0-warm-native-package-review.md`.
trn21 remains disk-blocked. Both hosts have all five default training shards; the
newer first-place brief is `workflow/CONTROLLER_PROMPT.md`. Best confirmed public20M is
still 0.9918353324 single (0.9919537735 two-run mean). The present handoff supersedes
older pending-G1/A0-staging statements below.

## Current handoff: G1 passed on trn22; warm A0 not yet launched

The trn22 fresh-process public20M evaluator calibration reproduced its retained checkpoint's
historical `0.992968300659387` BPB exactly at context 1024 and 20,971,520 tokens. The archived
Operator receipt is under `research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn22-r1/`
and the independent review is `research/agentic/verification/20260925-g1-trn22-calibration-review.md`.
This is **not** a new candidate score or a valid advancement control. trn21 still needs separate
G1 calibration after human disk expansion. `research/v2/g1.json` now records trn22-specific PASS.

The first warm c5c A0 control is being staged on trn22 with a unique label and retained warm caches.
Training has **not** started. Its launch still requires a completed stage receipt, fresh disk/host
checks and a fail-closed external step-boundary observer started before the first step. The
new-runner `run_job.py` remains pilot-only. Best confirmed candidate is unchanged at
`0.9918353324` public20M (single run; two-run mean `0.9919537735`).

The current handoff above supersedes older pending-G1 statements below.

## Latest controller brief and live gate work

The human's newest brief is summarized in `research/v2/CONTROLLER_BRIEF_2026-09-25.md`.
It requires incremental migration using proven launches, not a whole-runner replacement;
2M screens are permitted again but only fresh public20M/context1024 is a confirmed result.
The stricter adoption mean delta `<= -0.0010` applies, with human acknowledgement.

The verified disk packet is `research/v2/DISK_DECISION_PACKET_2026-09-25.md`:
the historical 3.89 GB old-controller cleanup opportunity is gone; that directory is empty.
Both LR0.015 checkpoints remain intact. Human EBS expansion 200 -> 250 GiB per host is requested.
No deletion, volume action or breaker reset was performed. Cold S0 and table runs remain blocked.

trn22 completed all four contained CPU/G2 probes and all three exact comparisons PASS.
The compact, hash-verified evidence is committed at `50e346bf` under
`research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/`.
Fresh-process retained-checkpoint G1 calibration started at 20:53:19 UTC, guard PID 2950407,
label `fs-20260925-g1-public20m-calibration-trn22-r1`; the corrected request preserves the
historical compiler flags and warm cache paths. No new candidate is being trained in that job.
trn21 remains idle/disk-blocked. Best confirmed public20M is unchanged: single 0.9918353324,
two-run mean 0.9919537735. Ready full experiments remain zero until required gates are complete.

The local legacy autoloop service is inactive and its watchdog timer active; do not claim an
unattended research scheduler is running. The repaired new runner at `d7733fba` passed a root-run
synthetic suite, but one-host pilot and bare-S0 compatibility remain pending; production migration
is not authorized. This does not block the separately prepared legacy warm A0 launch.

## Requested workflow v2 installed; runtime gates still pending

This section supersedes the older 2M selection and legacy native5 critical-path statements below.
The user explicitly requires the workflow on `claude/trainium-frontier-research-nkp5zd`; exact upstream
`4bd08d0f` was installed at `8b9fc841`, with no changes to `decide.py` or its thresholds. The deployed
research-v2 source is `c5c410af...`; S0 is `ad223cdc...`. Neither has a new native score.

Current state and next actions are in `research/v2/STATUS.md`, `research/v2/state.json`, and
`research/v2/g1.json`. The new ledger is empty; historical 2M scores are not relabeled as 20M controls.
G1 source/host metadata are verified, but fresh calibration, independent CPU/G2 execution and native gates
remain pending. Independent synthetic tests found runner correctness defects; fixes must pass separate
review before any scored launch. The older custom tmpfs R5 work is preserved as unverified WIP at
`4b8cb008`, not a prerequisite for the upstream workflow or launch authority.

Both hosts were idle at the last inventory; trn21 had 6.43 GiB free, below its 6.5-GiB floor. The local
legacy `trainium-autoloop.service` was auto-restarting and has been stopped to prevent dual dispatch.
No EC2 action or training interruption occurred, and no historical breaker was reset.

The best confirmed public20M score is unchanged: 0.9918353324 single run, 0.9919537735 two-run mean.
Bedrock panel source reconciliation at `0b1d9dcf`/`a47af039` rejected two already-implemented suggestions;
it did not produce an experiment-ready new improvement. See
`research/agentic/BEDROCK_PANEL_SOURCE_RECONCILIATION_2026-09-25.md`.

## Track-A controls valid; A1 blocked before step 1 by ownership adapter

This section supersedes earlier Track-A readiness and host-control statements below.

- Exact `35c9ef79...` plus MATRIX_LR 0.015 controls are valid on both hosts. The `trn21`
  control completed 3,636 steps and measured fresh-process public2M BPB
  `0.9866973734526626`; its evidence is pushed at `b0b23cbc`. The `trn22` control completed
  3,634 steps and measured `0.9873495157460374`; its evidence is pushed at `499f753c`.
  Their descriptive difference is `+0.0006521422933748` for trn22, so candidates must use the
  control from the same host. The screen nomination thresholds are `<=0.9856973734526626` on
  trn21 and `<=0.9863495157460374` on trn22. This screen cannot establish a KEEP.
- A1 (last-layer bigram+trigram tables, multiplier 64, dimension 128, RMSProp) has no BPB result.
  Retry r2 passed staging and source-integrity checks, then failed on all four ranks before step 1
  at the immutable ownership adapter with
  `ValueError: ownership probe is restricted to AdamW and Muon`. There was no checkpoint or
  evaluation. The exact failure, traceback, empty step ledger, and clean postflight are pushed on
  `evidence/20260924-v2-track-a-a1-trn21-r2` at `9d92ff82`; trn21 ended idle with
  9,867,505,664 free bytes. Do not interpret this as evidence for or against tables.
- Read-only source analysis found that the adapter wraps all optimizer groups, although its plan
  supports only FP32 AdamW and Muon. A1 adds a BF16 RMSProp group. The selected repair is to retain
  owner partitioning for FP32 AdamW/Muon while excluding the known RMSProp group from owner
  planning and stepping that original full group identically on every rank. Unknown kinds still
  fail. This is an infrastructure repair proposal, not a model change or run authorization; it is
  now in separate implementation and independent-verification gates.
- Hybrid-repair implementation `bf46f25c` passed its initial six CPU/fake-world contracts but
  failed independent adversarial verification at `0569cf3e`: checkpoint group-kind tampering and
  post-plan live group-kind drift were not authenticated before state mutation/collectives. That
  could silently stop RMS updates or create rank-asymmetric failure. The FAIL receipt and tests are
  pushed on `experiment/20260924-v2-a1-hybrid-rmsprop-ownership-repair`; native packaging and
  hardware remain forbidden while an Implementer adds exact ordered group-schema validation and a
  fresh Verifier repeats the gate.
- Repair iteration 2 (`241ba406`) fixed group-kind/order drift and passed its supplied 10 tests,
  but independent verifier commit `a350e2d2` found that malformed/missing/swapped optimizer state
  and drifted ownership-summary fields could still reach raw loading or a collective. Native work
  remains forbidden. Iteration 3 must authenticate the exact per-kind optimizer-state schema and
  complete ownership summary before mutation/collectives, then receive another fresh verifier pass.
- Repair iteration 3 (`257b84bc`) passed 15 supplied tests but verifier `fc931eb4` still found
  unauthenticated counter/state relations, nonfinite state acceptance, over-rejection of legitimate
  rank-sparse empty state, and non-transactional post-load failures. Source/caller audit then
  selected a safer simplification: exact A1 production never serializes optimizer state; final.pt,
  warm-start, eval and scoring are model-only. Iteration 4 will therefore fail closed immediately
  on hybrid RMS optimizer `state_dict/load_state_dict`, preserve no-RMS ownership serialization,
  and document interruption as abort/restart-from-zero. This removes unneeded resume complexity
  without changing any scored forward/backward/update/checkpoint/eval behavior.
- Iteration 4 simplified hybrid RMS optimizer resume to an immediate fail-closed error; verifier
  `8431f38a` found one mutable cached-list bypass. The four-line iteration-5 repair `ec93e156`
  bound the gate to the authenticated plan. Independent verifier receipt `c35504bc` now PASSes all
  13 supplied tests plus a two-RMS mutation matrix. The accepted candidate SHA-256 is
  `9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae`; `prepare.py` remains
  `5807c46e...`. A distinct native-five-step equivalence package may be prepared, but no hardware
  or full A1 run is authorized by this receipt.
- A separate source audit confirms clean-step replicated RMSProp is semantically equivalent to the
  ordinary optimizer only if tables are absent from ownership/all-gather, post-AVG joint table
  clipping is unchanged, and each rank executes exactly one RMS step. Source cannot prove Neuron
  byte identity. The native five-step gate must record post-AVG/preclip and postclip table-gradient
  hashes, clip scalar, parameter/moment/step hashes on every rank, ownership inventory, one RMS call
  per rank/step, equality to an ordinary non-owned control, and adapter-state save/load continuation.
  `final.pt` is model-only and is not an optimizer-resume artifact. Static loss scaling also has no
  pre-update nonfinite guard, so clean finite assertions and injected failure checks are mandatory.
- A flags-only AdamW substitution is rejected as a repair: it changes the table optimizer, still
  conflicts with the adapter's FP32 requirement for BF16 tables, and historical paired evidence
  found worse per-step quality and throughput. Disabling ownership globally is also not selected
  because ownership previously improved dense whole-step time by about 5.06%.
- Track-A A4 (`--mlp-sandwich-norm`) was not scheduled because it is a prior dense duplicate.
  The full 1,800-second `t39-sandwich-1800s` run regressed context-1024 BPB by `+0.006100`
  (`1.040276` versus `1.034176`) and was already discarded. No new branch, edit, or run was made.
- Track-A A5 (`--relu2-tau 0.5`) was also stopped before packaging. The same shifted-ReLU
  mechanism already lost about `+0.002903` observed BPB in paired 900-second dense runs, and exact
  `35c9...` cannot run it as a flags-only arm because its fused training MLP hard-requires
  `RELU2_TAU == 0.0`. A valid retest would be a separately researched kernel implementation, not
  the requested one-factor flag experiment.
- Track-A A6 (`--head-gate`, whose default includes per-head normalization) was stopped as another
  prior dense negative. `t39-hgate-900s` ran about 25% slower per step, completed about 19.8% fewer
  steps, and lost roughly `+0.0135` BPB against paired controls. Exact `35c9...` has no fused NKI
  implementation to change that wall-clock tradeoff, and the flag bundles two explanatory factors.
- Track-A A8's embedding-LR 0.45 sub-arm was stopped as a closed axis. The exact prior
  `t04-embedlr-045-900s` change from 0.30 to 0.45 measured only `-0.000057` BPB, below noise; the
  durable study concluded that 0.30 is a flat optimum. No full run was repeated.
- A8 embedding LR 0.60 was also stopped on final-week value-of-information. It has no isolated
  positive evidence, and width scaling makes dense1024 at 0.60 only about 5.4% more aggressive
  than the old dense640 0.45 null point in effective optimizer LR. Matrix LR 0.015 provides no
  mechanism for moving this closed embedding optimum upward.
- A8 warmdown ratio 0.90 was stopped as an exact prior duplicate. The one-factor
  `t04-warmdown-90-900s` run was `+0.000254` BPB worse than its same-code control mean; ratio 0.60
  was also weakly negative, bracketing the 0.75 default. The durable schedule study closed this
  axis, so no 1,800-second repeat was launched.
- Track-A A9's x0-gate sub-arm was stopped as a decisive prior wall-clock negative. Its paired
  dense run was about 13% slower, completed 11.2% fewer steps, and lost `+0.008694` BPB; the
  estimated per-step effect was within noise. The same extra per-block reduction remains outside
  exact `35c9...` optimized adapters, so no repeat was packaged.
- A9 out-pool was likewise stopped as a prior dense negative/null. The exact one-factor
  `t39-outpool-900s` run lost about `+0.0013` BPB at both 1024 and 2048 context while completing
  1.4% fewer steps; its throughput-adjusted residual was also non-improving. No repeat was made.
- A10 (depth 8 x width 768 plus winning tables) is ineligible: there is no adopted table winner,
  and the proposed arm bundles depth, width, and table changes. Prior depth-8 evidence is also
  strongly wall-clock negative. The duplicate-cleared Track-A queue therefore has no immediately
  runnable arm: A1 awaits the hybrid-ownership repair gates; A2 shares that blocker; A3/A7 flags
  are absent; A4/A5/A6/A8/A9 are evidence-stopped; A10 is dependency-blocked. Unblocking and
  verifying A1 is the next useful Track-A action.
- The next researched code-change lane after A1 is dense local-conv ordering. Exact `35c9...`
  still snapshots Q/K/V before applying the trained causal-convolution residual, so same-block
  serial attention does not see local-conv output. Moving the existing update before those
  snapshots on the serial/non-sandwich path is one parameter-neutral, operator-count-neutral
  causal change with related positive evidence, but it has never run on exact dense35c9+LR0.015.
  It requires a new proposal, independent implementation/verification, equivalence and causality
  gates, native NKI/eager equality, and same-host throughput/BPB comparison; it is not currently
  hardware-authorized.
- Track B1 (`LNC1`, world size 8) is blocked and is not a one-factor topology arm on exact 35c9.
  Pack32 makes one world microbatch 262,144 tokens, exceeding and failing divisibility against the
  131,072-token total batch, while the ownership adapter explicitly requires world size 4.
  Changing pack or total batch would confound topology; all exact LNC1 kernel, ownership,
  collective, HBM, checkpoint, and fresh-cache gates also remain unauthenticated. Historical LNC1
  evidence is unfavorable. Do not propose B1 until A1 works and a separately matched pack16
  topology contract clears the documented component gates.
- A2 (bigram+trigram tables at layers `1,-1`, mult64, dim128, BF16 RMSProp) is
  `GO-CONDITIONAL` only after A1 passes and produces a valid archived same-host result. Historical
  evidence found adding layer 1 improved about `-0.0118` BPB at fixed steps, with a roughly 5.86%
  throughput cost; the current-stack prediction versus A1 is about `-0.010` fresh2M BPB with a
  wide `[-0.025,+0.005]` interval. The exact profile creates four 128-MiB tables plus roughly
  512 MiB each of replicated RMSProp moments and dense gradients, so ownership correctness, HBM,
  collectives, checkpoint/load, causality, throughput, and same-host A1 comparison must gate it.
  No executable A2 proposal may exist before A1 clears those dependencies.
- A duplicate-clear audit found one lower-upside survivor after A1/A2/A11:
  `--qkv-norm-cse`. On exact no-QK-shift serial 35c9, Q/K/V inputs are identical, so the flag
  reuses one normalization instead of computing three in each of six blocks, removing 12 redundant
  norm operations with no parameter/state change. There is CPU coverage but no durable hardware
  result. Predicted throughput gain is 0–2% (central about 1%) and central wall-clock BPB delta
  about `-0.0007`; require native identity/causality and reproducible >=1% same-host whole-step
  gain before any full run. This is queued behind the higher-upside A1/A2 and A11 work.
- The paused A2 scaffold is unauthorized and must not run: it has the same RMSProp/ownership
  blocker. Track B2 sparse touched-row work also remains rejected before hardware. The best
  confirmed submission candidate therefore remains the review-clean MATRIX_LR 0.015 dense file,
  best single public20M BPB `0.9918353324459732`, two-run mean `0.99195377346572045`.

## Standing mission v2 — table-first first-place campaign

At 2026-09-24 15:30 IST the human replaced the small-tuning critical path with the table-first
campaign in `STANDING_MISSION_V2_2026-09-24.md`. The exact Recursive primary source and the exact
`35c9...` flag surface have been audited. Track A now starts with per-host `35c9...` LR-0.015
controls followed by last-layer and layers-`1,-1` hashed-table arms. Track B develops a
touched-row-only synchronized table optimizer. Track C keeps the review-clean candidate behind
adopted research changes by at most 12 hours. Window-pattern and RoPE-base experiments are not
launchable as flag-only arms because those flags do not exist in `35c9...`.

Track B2's CPU feasibility contract and blocked native-probe harness independently passed on branch
`experiment/20260924-track-b2-ngram-touched-row` at verifier receipts `b58ce84f` and `a2a44065`.
This is not hardware authorization: the probe remains non-dispatchable until it is bound to an
authenticated successful `35c9...` Track-A receipt, retained checkpoint, independently reviewed
candidate, host and immutable request. One separately human-approved component probe must then
prove key/table/logit identity and native distributed-update semantics; even a passing probe does
not authorize sparse-update training.

The Track-A launcher repair independently passed 21/21 tests and hostile descendant-process
fixtures. Its exact dual-host controls are verifier-finalized and pushed on
`experiment/20260924-v2-track-a-35c9-controls-a1` at `65155c66`; A1 itself remains locked until an
authenticated archived trn21 control is bound and re-reviewed. The exact `35c9...` source and
evaluation files remain unchanged.

The completed review-clean trn22 control produced an authentic observational 20M measurement of
`0.992968300659387` at 3,631 steps, but it is `INVALID_CONTROL_FOR_ADVANCEMENT`: postflight free
space was 5,928,337,408 bytes, below the 6.5-GiB operational floor. Its evidence is pushed at
`68ecfeb1`, with Analyst verdict `59f4d8ab`. No trn22 candidate may compare against that run. A new
unique control is required after separately authorized cleanup, and the observed footprint implies
a conservative pre-staging floor of 8,833,482,752 bytes if the host must finish above 6.5 GiB.

The previously launched Trn22 `116188cf...` current-base control remains valid submission-safety
work and is allowed to finish; it is not mislabeled as a `35c9...` Track-A control. The old LR0125
and embedding-only harness work is parked unlaunched. The legacy autonomous fixed queue remains
inactive until the v2 proposals, independent receipts, and exact host controls replace it.

## Final-week override — LR 0.015 adopted, canonicalization in progress

This section supersedes the older score, base, queue, and host statements below.

- The best confirmed public-development candidate is the review-clean dense6x1024 file with
  `MATRIX_LR = 0.015`, SHA-256
  `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.
  Two seed-42, full-budget, fresh-process 20,971,520-token evaluations on `trn21` measured
  `0.9920722144854677` at 3,635 steps and `0.9918353324459732` at 3,654 steps. Their arithmetic
  mean is `0.99195377346572045`; both beat the prior public best `0.9928513116311253`.
- The evidence for both runs is now imported into this canonical controller checkout under
  `research/submission-validation/20260924-c595-matrixlr015-trn21/`. The confirmation Operator
  receipt records a clean seed-42 run, 1,800-second budget compliance, 20M token/83,462,186-byte
  evaluation, finite checkpoint, and exact candidate hash.
- A canonical root candidate was implemented from reviewed parent `4ed89f36` as commit
  `e12d0deedea806128c1999f4b0a6340b967a7e64`. It changes only root `train.py` and has the exact
  candidate hash above. Independent verification is still required before the controller may pin
  or run it.
- `TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS` is 900. The two LR 0.015 runs observed approximately
  839 and 831 seconds of startup, so cold-compile margin is only 61–69 seconds. Compile-cache
  reuse may accelerate research, but official-cold behavior must remain a separate gate.
- The old fixed queue based on `9173f626…` is retired for science decisions. It remains inert
  until the new base, controls, proposals, and independent receipts are installed. No old 640-stack
  proposal may be launched as a fallback.
- At 09:07 UTC both hosts remained idle and unheld. The Operator reclaimed 614,690,816 allocated
  bytes on `trn22` by deleting three exact, archived controller worktrees; it now has 7.4873 GiB
  free. `trn21` remains untouched at 6.8574 GiB because its LR0.015 checkpoint explicitly forbids
  pruning. The final-week operational gate is 6.5 GiB, still 1.5 GiB above the unchanged hard
  approximately-5 GiB protocol floor. See
  `operations/TRN2_FINAL_WEEK_GATE_CLEANUP_2026-09-24T090757Z.json`.
- Final-week selection uses context-1024 public BPB. A 2,097,152-token fresh-process evaluation may
  screen an arm, but cannot establish a KEEP. Promotion requires a fresh 20,971,520-token evaluation;
  adoption requires two negative 20M deltas with mean at most `-0.0008` against current-base,
  same-host controls. Submission and instance lifecycle actions remain human-only.
- The next host work is deliberately asymmetric: `trn22` needs a fresh current-base 2M+20M control;
  `trn21` first receives evaluator-only 2M calibration of its two retained exact-base checkpoints,
  then MATRIX_LR 0.0125. This replaces a redundant third `trn21` control training run. The next
  bracket is MATRIX_LR 0.0175; embedding 0.45 ranks ahead of 0.20. The 1080-second diagnostic,
  unembedding, scalar LR, and softcap are outside the immediate critical path.

## Human-authorized fixed-queue override — pending independent verification

The current implementation branch represents, but does not activate, the 2026-09-23 replacement
plan in `frontier-full-only-queue.json`. It parks
`20260923-local-conv-pre-attn-serial-only-r3`, pins the exact dense6x1024 lean train bytes
(`9173f626…`), restores `trn22`, and disables general proposals and smoke runs. The only eligible
order is exact cold control on `trn22`, MATRIX_LR 0.025 on `trn21`, then LR 0.015, warmdown 0.60,
warmdown 0.85, total batch 262144, and MAX_TRAIN_SECONDS 1080. Public BPB from a fresh context-1024
evaluation over exactly 20,971,520 tokens / 83,462,186 bytes is the only selection metric.

This is not launch approval. Every queue entry remains pending an independent Verifier receipt.
The persisted disk breaker is preserved and the fixed controller additionally requires the exact
pushed R8 cleanup receipt plus an explicit Operator state transition. No breaker was cleared and no
remote or hardware action was performed while implementing this override.

## Operational override — 2026-09-23 16:35 UTC

This section supersedes older dynamic controller, proposal, and host statements below.

- The controller branch is synced to origin at `1493ff2b080a5eb3b2a45fd72742906e43906736`.
  The systemd service exited fail-closed with status 3 and is waiting on its configured five-minute
  restart because the persisted breaker is open; an active service unit is not evidence of active
  research while this breaker remains set.
- Proposal `20260923-local-conv-pre-attn-serial-only-r3` passed proposal verification and independent
  implementation verification. Its isolated candidate is
  `79b0cecb2f655441f4e0898f6140308eea6072f0`. No smoke launched: two inventories measured idle
  `trn21` at 7.3 GiB free, below the immutable 7.4 GiB gate, and opened the breaker.
- Read-only live inventory at 16:34 UTC confirmed that both boxes are idle with no `torchrun`,
  `train.py`, or `run_experiment.py` process. Available bytes were 7,835,312,128 on `trn21` and
  8,628,547,584 on `trn22`. The controller still schedules only `trn21`; `trn22` remains excluded
  pending the separate hash-preserving cleanup/re-enable decision required by
  `operations/TRN22_DISK_GATE_RECOVERY_2026-09-23.md`.
- Read-only attribution found 2,254,217,216 bytes of immutable controller run directories on
  `trn21` and 4,371,390,464 bytes on `trn22`. No remote file was removed. The exact inventory and
  recovery boundary are recorded in
  `operations/TRN2_AUTOLOOP_DISK_BLOCK_2026-09-23_R1.md`.
- The latest full result remains the n-gram-table cooldown DISCARD, `1.007831` / 2072 steps versus
  the same-host control `1.005734` / 2071. No autonomous result beats the preserved public record
  `0.9928513116311253`; there is no new submission candidate.

## Primary-source research override — 2026-09-23 15:26 UTC

- The primary-source and source-history review is durable in
  `research/agentic/PRIMARY_SOURCE_RESEARCH_MAP_2026-09-23.md`. It is a ranked queue, not run
  authorization.
- `20260923-ngram-ve-cooldown-lr-flag-only-default-shape` completed its smoke process successfully
  at `1.168643` / 334 steps versus the same-host smoke control `1.165855` / 334. It preserved
  throughput but was `+0.002788` BPB worse at the short horizon. The proposal defines smoke as a
  correctness/throughput gate, so collection and the independent Analyst verdict still control
  whether a full run is eligible. This is not an improvement.
- After the active proposal closes, the strongest next architecture lead is a source-history
  repair: commit `7518d91e` introduced the Q/K/V snapshot before `local_conv` for a rejected MUDD
  experiment, and that ordering survived. The current trained local convolution therefore does not
  feed same-block default attention, contrary to its comments and its earlier pre-attention wiring.
  It has prior GPU and Trn2 positive evidence through related paths and matches the Canon-A placement
  studied in arXiv:2512.17351. The Researcher must still produce one bounded proposal and an
  independent verifier must approve it before implementation or hardware use.
- Secondary zero-hardware lanes are modern Muon feasibility (RowUpdateFloor/radial control,
  SOAP-Muon, then Newton-Muon) and completing the saved ROW-FP8 backward. Training-time random token
  replacement is parked until a measured stack actually enters a repeated-data regime.
- No new full-run score beats `0.9928513116311253`, and submission validation remains
  STOP/ABANDON_CURRENT_PROPOSAL.

## Reconciliation override — 2026-09-23 15:13 UTC

This section supersedes the stale dynamic run/box statements later in this handoff.

- The autonomous controller is healthy on `agent/control-plane-20260921` at `b933e2ca`, synced
  to origin, with breaker closed after cycle 287 and 37 retrieved runs.
- Latest measured experiment: `20260923-ngram-ve-double-hash-r2` smoke was a clean DISCARD at
  `1.171049` / 325 steps versus the same-host smoke control `1.165855` / 334 steps. It was 2.66%
  slower and `+0.005194` BPB worse. Verdict and checkpoint-prune evidence are pushed in
  `2ed3c130` and `b2cea2e7`; do not promote it to a full run.
- The preceding attention-scale 0.12 full run was also a clean DISCARD: `1.009764` / 2071 steps
  versus `1.005734` / 2071. The next value-residual proposal was rejected by the independent
  verifier before hardware use (`b933e2ca`).
- Both Trn2 boxes were idle with no Neuron holder at 15:04 UTC. Approved exact-target cleanup of
  three already archived/reconstructable checkouts restored trn21 to 8,371,343,360 free bytes;
  see `research/agentic/operations/TRN21_ARCHIVED_WORKTREE_CLEANUP_2026-09-23_R7.md`.
- The best valid autonomous full KEEP remains BF16 SDPA input casting at `1.002028` versus
  `1.005967`; it still does not beat the preserved public-development result `0.9928513116311253`
  (cold replay `0.9938556748843932`). There is no new submit-ready winner.
- A new external review was reconciled in
  `research/agentic/EXTERNAL_REVIEW_RECONCILIATION_2026-09-23_R2.md`. Its strongest duplicate-clear
  control-stack lead is the existing `--no-ngram-ve-flat-lr` flag (late table cooldown). Search of
  `DECISION_LOG.md`, `CURRENT_STATE.md`, `research/experiment-log.jsonl`, and `train.py` found the
  flag but no measured `flat_lr=False` run. Treat that as a hypothesis requiring the normal
  proposal and independent-verification gates, not as a result.
- Submission validation remains STOP/ABANDON_CURRENT_PROPOSAL at `beba1595` in the separate
  `Trainium-submission-validation` checkout. Do not submit the current lean-review file: the
  small compiled-Muon raw-byte discriminator is not a valid gate, and no replacement gate plus
  complete dress rehearsal has passed.

## Verified repository state

- Active controller checkout: `agent/control-plane-20260921`, clean and synced to
  `origin/agent/control-plane-20260921`; deployed controller code includes `3b26fee1`
- Authoritative research branch: `origin/research/fresh-start-20260918`
- Authoritative research commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Recovery-only snapshot: `origin/backup/fresh-start-20260918` at `c7f151a7`; it does not
  supersede or replace the authoritative branch
- Existing runner: `research/run_experiment.py`
- Existing ledger: `research/experiment-log.jsonl`
- Mandatory machine protocol: `research/experiment-protocol.md`
- Autonomous state: 30 retrieved result records (31 launch attempts including the failed initial
  `trn22` control); proposal `20260923-unet-skip-zero-init-r2` is smoke-verified and awaiting its
  full run; automatic Git push is enabled, but the controller breaker is open on the `trn21` disk
  gate described below
- Evidence durability: commit `691faccb` retrieved and SHA-verified the previously omitted
  per-run launch receipt for all 29 durable results. Future collection retrieves that receipt,
  enumerates every untracked path, and fails the correctness gate on any unexplained post-run dirt.
- Research grounding: after two source-contradicted proposals were independently rejected before
  hardware use, commit `3b26fee1` added a bounded authoritative `train.py` measured/closed evidence
  index to the Researcher prompt. The complete-source Astra verification gate remains unchanged.
- Phase 1 deadline: 2026-09-30 23:59 PT

## Known score state

- Leaderboard screenshot shows best accepted score: `1.0246`.
- Best preserved comparable public evaluation is `0.9928513116311253` BPB; its independent cold
  replay is `0.9938556748843932`. These are public development measurements, not the private
  leaderboard score.
- Best independently verified autonomous-loop full result is
  `20260922-bf16-sdpa-input-only-default-shape`: `1.002028` versus its same-host `trn22` control
  `1.005967`, at 2233 versus 2076 steps (7.56% apart, inside the 10% comparability gate). This is a
  valid KEEP relative to the pinned control, but it does not beat the preserved `0.992851` public
  development record.
- Latest completed full result is `20260922-bf16-norm-output-except-mlp-input`: `1.020309` versus
  its `trn21` control `1.005734`, at 2352 versus 2071 steps. The 13.57% step difference exceeds the
  comparability gate, so its durable verdict is CONFOUNDED, not a win.
- Latest completed smoke is `20260923-unet-skip-zero-init-r2`: `1.167766` at 330 steps on `trn21`,
  versus the same-host control smoke `1.165855` at 334 steps. All correctness gates passed and the
  independent Analyst returned KEEP/eligible-for-full. A smoke is not a comparable full-result win.
- The frozen one-file candidate is
  `submissions/dense6x1024_public0992851/train.py` on the authoritative branch.
- The exact commit/configuration that produced official `1.0246` is not yet recorded here.
- A later submission was rejected by review; its reviewer explanation is not yet recorded here.

## Blocking reconciliation tasks

1. Record the submission ID, commit, launch command, and config for official `1.0246`.
2. Record the complete reviewer reason for the rejected submission.
3. Record the current leaderboard rank and tenth-place cutoff.
4. `trn21` is idle but blocked at 7.26 GiB free, below the configured 7.40 GiB gate; the second
   failed full-run preflight opened the breaker at 2026-09-23 03:11 UTC. `trn22` remains excluded
   at about 7.37 GiB. Read-only attribution found 3.89 GB in old controller-created isolated run
   worktrees, but current cleanup authority covers scratch checkpoints only. See
   `research/agentic/operations/TRN21_UNET_FULL_DISK_BLOCK_2026-09-23.md`; do not lower the gate or
   remove worktrees/caches without explicit human coordination.
5. Confirm official team membership and replace the emailed shared PEM with per-user access.
6. The valid Bedrock route is `us-west-2` with `us.anthropic.claude-fable-5-1` and
   `us.openai.gpt-6-astra`; the old `eu-north-1` plus `global.*` failures are irrelevant. Fable's
   long streamed Researcher calls previously suffered silent read timeouts and premature
   termination. Commit `500615a9` raised the validated socket read timeout to 1,800 seconds while
   retaining the single fresh retry and fail-closed parsing. Multiple production calls have since
   completed beyond the former 600-second boundary, so the transport repair is now live-verified.

## Active technical direction

The active proposal is `20260923-unet-skip-zero-init-r2`, candidate commit `7e500778`. Its
independently verified smoke passed every correctness gate and earned KEEP, but its full promotion
has not started because the `trn21` disk breaker is open. It cannot be called an improvement until
a retrieved, comparable full run beats its same-host control. The strongest completed autonomous
full result remains BF16 SDPA input casting, which has not beaten the preserved `0.992851` public
development record. Recent norm-boundary variants increased throughput while worsening BPB or
became step-count-confounded, so throughput alone is not evidence of progress.

Source197 remains useful historical systems evidence: moving the same single library MLP from
layer 0 to layer 5 changed the compiled forward graph from 135.871 ms to 329.259 ms and added
16.344 GB of spill reloads. Any renewed caller-boundary/layout proposal must preserve ordinary MLP
behavior and define a same-host complete-forward/step gate; it must not assume `contiguous()` is a
fix without an evidence-linked boundary argument. The complete saved-ROW-FP8 MLP backward remains
on hold.

The saved ROW-FP8 forward is a real component lead (about 9.4% at one rank and 19.1% under four-rank
contention in source199), but it has no backward, whole-step, or BPB proof. It must achieve a
repeatable complete-step improvement of at least 3% and pass numerical/cache/evaluation gates before
any 30-minute quality run is proposed.

Do not reopen the parked attention redesign, generic full-model autocast FP8, Vector-engine square,
compiler-SBUF placement, width1152, or previously tested library-MLP compositions without a new
profile-backed causal hypothesis.

## Human strategic research priority — 2026-09-22

The human operator wants the unattended Researcher to search for mechanisms capable of closing the
leaderboard gap, not merely accumulate small step-rate changes. Within the standing authorization,
rank new hypotheses by plausible fixed-budget BPB upside and information gain across these lanes:

1. **Targeted FP8 or mixed-precision systems work.** Pursue precision changes only at a measured
   expensive boundary, with numerical, backward, whole-step, checkpoint, causality, and native-eval
   gates. The saved ROW-FP8 lead remains relevant; generic full-model autocast FP8 remains closed.
2. **Profile-directed NKI fusion.** Target a dominant end-to-end cost such as boundary/layout
   conversion, attention/MLP materialization, optimizer work, or cache refresh. An isolated kernel
   microbenchmark is not a win unless the complete step improves and BPB is preserved.
3. **Learning-efficiency re-optimization after a systems gain.** Once a measured throughput or
   memory win changes the feasible frontier, use bounded one-factor tests of capacity allocation,
   depth/width/context, batch or accumulation, learning-rate/warmdown, optimizer, and auxiliary
   memory capacity. Do not substitute an unstructured broad sweep or reopen a closed regime without
   new causal evidence.
4. **Other competition-legal architecture or training ideas.** Prefer mechanisms that improve
   loss per token or useful tokens processed within 1,800 seconds, and reconcile them against the
   full ledger before proposing a run.

Isolate causal factors first. A stack-confirmation experiment may combine only independently
validated KEEP changes and must declare the interaction hypothesis. Public development BPB is the
selection proxy; official private-shard BPB remains unknown until a human-approved submission.

## Human approvals required

- Development Trn2 launches are covered by `STANDING_AUTONOMOUS_AUTHORIZATION.md` after all
  proposal, independent-verification, preflight, and evidence gates pass.
- Any AWS resource mutation
- Any merge to the submission candidate
- Any leaderboard submission

## Agent-loop implementation

- Architecture and safety plan: `research/agentic/AUTONOMOUS_LOOP_PLAN.md`
- Current stage: unattended controls and flag experiments are live. The code-change path uses
  Researcher → proposal Verifier → isolated Implementer → deterministic diff/static gate →
  implementation Verifier → typed Operator → Analyst. Code candidates are limited to `train.py`,
  committed on isolated `autoloop/<proposal-id>` branches, and never merged automatically.
