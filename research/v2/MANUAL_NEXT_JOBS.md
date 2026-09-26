# Manual next jobs — verified 26 September 18:13 IST

Run commands from /home/poojith-devan/Desktop/Trainium. Root is sole Operator.
Current consumed jobs: trn21-k2rescue (18:06:47 IST), trn22-muon150 (18:12:10).
Both have four verified workers and their original lifetime locks held.
NEVER rerun either label. Warmdown .60 and .85 are complete and archived.

## Current ordered work

| Host | Running | Next | Following |
|---|---|---|---|
| trn21 | K2 native201 | Qualified tables timing, otherwise trn21-finalcold | Remaining scalar arm after final-candidate decision |
| trn22 | momentum150 | trn22-finalseed43 after authenticated final cold | Remaining scalar arm after final-candidate decision |

Tables are conditional on net tokens >=+3% and acceptable native loss/correctness.
They have no released packet yet. Do not launch an old table source.
Remaining LR/control packets are prepared but removed from the active queue:
do not use old command chains. Rebind the queue and exact predecessor before use.
There are not three unconditional ready successors per host; do not claim otherwise.
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

## K2 terminal and safe cold successor

Collect ONCE only after terminal, all descendants absent, idle device and free
original lock. Existing bundles must be authenticated, never overwritten.

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn21-k2rescue
python3 research/v2/lanes/k2-outside-compile-rescue/analyze_native.py research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-k2rescue/results-bundle.json
```

If K2 qualifies, tables preempt this slot and require a newly bound packet/queue.
Otherwise, after timing completed successfully but failed the net screen:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py check trn21-finalcold
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn21-finalcold research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-k2rescue/results-bundle.json
```

A crashed timing predecessor currently fails this launcher's terminal-success
guard. Preserve the failure and independently prove idle, then log and implement
a narrowly bound failed-predecessor handoff; never fake a successful receipt.
Source review is NOT a gate for the already launched research timing. Its later
source-scope PASS does not substitute for actual native evidence or Claude audit.

## Momentum terminal and seed43 successor

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn22-muon150
python3 research/v2/operator/20260926-operator-recovery/analyze_current.py trn22-muon150
```

Seed43 requires BOTH completed momentum and actual clean candidate cold result:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn21-finalcold
python3 research/v2/operator/20260926-operator-recovery/analyze_current.py trn21-finalcold
python3 research/v2/operator/20260926-operator-recovery/manual_job.py check trn22-finalseed43
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-finalseed43 research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-muon150/results-bundle.json
```

Seed43 uses identical clean cd0c source, warm research only. It measures seed
variation, not a matched causal gain versus seed42. Submission retains seed42.
If momentum wins, confirm/rebind deliberately rather than silently change packets.

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
Do not overwrite old Downloads or claim upload-ready before Claude audit.
Poojith alone uploads. Warm startup never qualifies a submission.
