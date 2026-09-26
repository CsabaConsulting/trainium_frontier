# Manual next jobs —26 September latest two-arm order

Work in /home/poojith-devan/Desktop/Trainium. Root is sole Operator.
Preserve live trn21-finalcold and trn22-muon150 plus attached evaluations.
K2/custom-kernel tables CLOSED. No new agents/lanes. Checks25minutes.

| Host | Running | Next3 unstarted |
|---|---|---|
| trn21 | finalcold | a1framework; lr0168; control |
| trn22 | muon150 | compilerdefault5; finalseed43; lr0168 |

Compiler full is CONDITIONAL: cold5 startup<=950s or measured compensating
steady gain. Five steps alone cannot prove steady speed. Prepare a distinct
full packet before release. model-type=transformer already appears in archived
effective argv; verify current before a duplicate no-op test.
Cold -> seed43 -> five-file Downloads audit export remains the package order.
Confirmed base remains batch262144/LR.021/.75/momentum300; no Recursive.

## Observe

```bash
cd /home/poojith-devan/Desktop/Trainium
python3 research/v2/operator/20260926-operator-recovery/manual_job.py status trn21
python3 research/v2/operator/20260926-operator-recovery/manual_job.py status trn22
```

Collect ONCE only after terminal/outer exit and original lock/device idle.
Stop each sequence on error. Never overwrite an archive or rerun consumed labels.

## trn22 compiler cold5 after momentum

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn22-muon150
python3 research/v2/operator/20260926-operator-recovery/analyze_current.py trn22-muon150
python3 research/v2/operator/20260926-operator-recovery/manual_job.py check trn22-compilerdefault5
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-compilerdefault5 research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-muon150/results-bundle.json
```

Same cd0c source; removes only TRAIN_ENV optlevel1, retains BF16 autocast.
Five steps, private cold caches, no evaluation/BPB. Verify effective compiler argv.

## trn21 standard-framework A1 after cold final replay

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn21-finalcold
python3 research/v2/operator/20260926-operator-recovery/analyze_current.py trn21-finalcold
python3 research/v2/operator/20260926-operator-recovery/manual_job.py check trn21-a1framework
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn21-a1framework research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-finalcold/results-bundle.json
```

One full1800s b7/t7 last-layer lottery, ordinary F.embedding backward, FP32
masters/gradients/RMS. K2 flag rejected. Report startup, charge, tok/s, final20M.
No speed/step-count gate. No automatic adoption or submission.

## Seed43 after completed compiler smoke AND authenticated final cold

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn22-compilerdefault5
python3 research/v2/operator/20260926-operator-recovery/manual_job.py check trn22-finalseed43
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-finalseed43 research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-compilerdefault5/results-bundle.json
```

Completed smoke need not be a speed win. A failed/unknown terminal needs diagnosis,
not false PASS. Same cd0c seed43 is warm research, not cold proof or seed shopping.

## Remaining scalar arms

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn21-a1framework
python3 research/v2/operator/20260926-operator-recovery/analyze_current.py trn21-a1framework
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn21-lr0168 research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-a1framework/results-bundle.json
```

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn22-finalseed43
python3 research/v2/operator/20260926-operator-recovery/analyze_current.py trn22-finalseed43
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-lr0168 research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-finalseed43/results-bundle.json
```

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn21-lr0168
python3 research/v2/operator/20260926-operator-recovery/analyze_current.py trn21-lr0168
python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn21-control research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-lr0168/results-bundle.json
```

A measured lottery winner requires deliberate rebind of unstarted follow-ups,
not blind continuation of old-base LR arms or a fabricated confirmation.

## Automatic guards inside every launch above

Exact source/argv/proposal/packet hashes, remote copies, prepare/harness identity,
STOP, disk/RAM, unused label, ps AND neuron-ls and fuser. Acquire SAME exclusive
host flock, repeat checks inside it, retain through train/eval/all descendants.
Busy/unknown/mismatch refuses launch. Never unlink or replace the lock:
/home/ubuntu/.cache/trainium_freshstart/device0.lock
trn21=66305:596823; trn22=66305:832589.

After cold AND seed43 are authenticated, export five audit attachments:

```bash
python3 workflow/candidates/final-batch262144-lr021/export_cold_audit.py --check
python3 workflow/candidates/final-batch262144-lr021/export_cold_audit.py
```

Unique Downloads folder, accepted launch in AUDIT_BRIEF.md. No overwrite or
automatic upload. Claude audit and Poojith alone authorize submission.
