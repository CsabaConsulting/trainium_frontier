# Manual next jobs — verified 26 September 18:28 IST

Run commands from /home/poojith-devan/Desktop/Trainium. Root is sole Operator.
Current consumed jobs: trn21-finalcold (18:27:15 IST), trn22-muon150 (18:12:10).
Both have four verified workers and their original lifetime locks held.
NEVER rerun either label. Warmdown .60 and .85 are complete and archived.
K2 native rescue FAILED with0steps; failure already collected and authenticated.
K2 and tables are CLOSED forPhase1; no further retry or table release.

## Current ordered work

| Host | Running | Next | Following |
|---|---|---|---|
| trn21 | trn21-finalcold | trn21-lr0168 after archiving cold/audit handoff | trn21-control |
| trn22 | momentum150 | trn22-finalseed43 | trn22-lr0168, then trn22-control |

Tables' K2 prerequisite failed. All table packets remain disabled.
Remaining LR/control packets are now in the active queue with explicit updated
predecessors. Seed43 may overlap the already-started cold replay: both use exact
cd0c bytes, and neither result depends on the other computationally. Actual
cold and second-seed results remain required before upload, as does Claude audit.
If momentum becomes a qualifying winner, revise unstarted confirmation work
deliberately instead of silently changing frozen packets.
No schedule stack: it lost. Final candidate is batch262144/LR.021/.75/momentum300
until a measured confirmed winner supersedes it.

## Read-only observation

```bash
cd /home/poojith-devan/Desktop/Trainium
python3 research/v2/operator/20260926-operator-recovery/manual_job.py status trn21
python3 research/v2/operator/20260926-operator-recovery/manual_job.py status trn22
```

Preserve active train AND attached eval. Routine checks every20–30minutes.
An observation timeout is not a failed run. Never unlink/replace a lock.

## Final cold completion

Collect ONCE only after terminal, all descendants absent, idle device and free
original lock. Existing bundles must be authenticated, never overwritten.

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn21-finalcold
python3 research/v2/operator/20260926-operator-recovery/analyze_current.py trn21-finalcold
```

The failed-K2 handoff is already consumed. Do not dispatch finalcold again.
Its immutable attempt records FAILED_K2_ARCHIVED_COLD_FALLBACK_ONLY.
Actual cold result and fresh20M must be authenticated before Downloads packaging.

## Momentum terminal and seed43 successor

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn22-muon150
python3 research/v2/operator/20260926-operator-recovery/analyze_current.py trn22-muon150
```

Seed43 requires completed same-host momentum and fresh original-lock/machine
checks; it does NOT wait for the other host's cold result:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py check trn22-finalseed43
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-finalseed43 research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-muon150/results-bundle.json
```

Seed43 uses identical clean cd0c source, warm research only. It measures seed
variation, not a matched causal gain versus seed42. Submission retains seed42.
If momentum wins, confirm/rebind deliberately rather than silently change packets.

## Remaining LR bracket and genuine controls

After cold replay is collected/authenticated and the audit handoff prepared:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn21-lr0168 research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-finalcold/results-bundle.json
```

After seed43 completes and is collected/authenticated:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn22-finalseed43
python3 research/v2/operator/20260926-operator-recovery/analyze_current.py trn22-finalseed43
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-lr0168 research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-finalseed43/results-bundle.json
```

After the corresponding LR arm finishes (one block per free host):

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn21-lr0168
python3 research/v2/operator/20260926-operator-recovery/analyze_current.py trn21-lr0168
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn21-control research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-lr0168/results-bundle.json
```

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn22-lr0168
python3 research/v2/operator/20260926-operator-recovery/analyze_current.py trn22-lr0168
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-control research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-lr0168/results-bundle.json
```

LR.0168 is standalone on batch262144/.75/momentum300, not a guessed stack with
momentum150. Controls useLR.021. All1800/fresh20M/seed42; cached research runs
are not submission qualification. Stop command chains on any error; no retries.

## Guards and handoff

The launch command authenticates frozen source, argv, proposal and packet; hashes
the staged host files; checks ps AND neuron-ls, fuser, STOP, disk/RAM, original
lock identity, prepare/harness and unused label. It repeats checks inside the
original exclusive flock at dispatch and retains that lock through all train,
eval and owned descendants. Detached host jobs survive laptop shutdown.
Lock: /home/ubuntu/.cache/trainium_freshstart/device0.lock
Identities: trn21=66305:596823; trn22=66305:832589.
Busy/unknown/mismatch means refuse, never force or duplicate.

After actual cold full run/fresh20M, prepare a unique Downloads audit folder:
exact train.py, unchanged accepted launch, Claude brief and measured receipts.
The tested exporter creates exactly five attachments; the exact launch command
is in AUDIT_BRIEF.md rather than a sixth attachment:

```bash
python3 workflow/candidates/final-batch262144-lr021/export_cold_audit.py --check
python3 workflow/candidates/final-batch262144-lr021/export_cold_audit.py
```

Do not overwrite old Downloads or claim upload-ready before Claude audit.
Poojith alone uploads. Warm startup never qualifies a submission.
