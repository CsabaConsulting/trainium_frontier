# Manual next jobs — current human momentum150 order

Running: trn21 warmdown0.60, launched17:05:44IST; trn22 warmdown0.85,
launched17:13:51IST on26 September. Do NOT rerun either label.
The latest human order replaces opposite-warmdown fallbacks with momentum150.
Confirmed batch262144/LR.021/ordinary.75 remains the final-package candidate;
Recursive stack and LR.02625 did not improve the matched public20M results.

## Next three jobs per host

| Host | First | Second fallback | Third fallback |
|---|---|---|---|
| trn21 | trn21-muon150 | trn21-lr0168 | trn21-control |
| trn22 | trn22-muon150 | trn22-lr0168 | trn22-control |

muon150 changes ONLY MUON_MOMENTUM_WARMUP_STEPS300 ->150 on the confirmed base;
source4b0641db9f009af1601ff48e6944a0c6d851164f42011f026d06ed20761c3020.
CPU scalar/reference and byte/AST checks PASS; this is the human-selected
scalar in unchanged reviewed logic, not a claimed new independent review.
The later LR fallback uses original momentum300, LR.0168, ordinary.75; it
does NOT silently stack an unscored momentum change. Final controls are genuine
prospectively labelled baseline runs. All seed42, full1800, fresh20M/context1024.
Later measured winners or qualified K2 may replace fallbacks; update this file
and queue/predecessors before doing so. Original ledger roles remain intact.

## Read-only check

Run from the laptop; no AI subscription/API call is needed by these commands:

```bash
cd /home/poojith-devan/Desktop/Trainium
python3 research/v2/operator/20260926-operator-recovery/manual_job.py status trn21
python3 research/v2/operator/20260926-operator-recovery/manual_job.py status trn22
```

If training OR attached evaluation is active, leave it alone. Check again in
20–30minutes. Never delete a lock, restart a consumed label or use python -O.
The collect step below succeeds only after terminal evidence and actual
device/process/lock quiescence; on failure preserve the error and stop.
If a results-bundle already exists, verify it rather than recollecting it.

## trn21: run ONE applicable block after its predecessor finishes

Current warmdown.60 -> momentum150:

```bash
cd /home/poojith-devan/Desktop/Trainium
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn21-warmdown060 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py check trn21-muon150 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn21-muon150 research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-warmdown060/results-bundle.json
```

Momentum150 -> standalone LR.0168 on the original confirmed batch/LR base:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn21-muon150 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn21-lr0168 research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-muon150/results-bundle.json
```

LR.0168 -> genuine baseline control:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn21-lr0168 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn21-control research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-lr0168/results-bundle.json
```

## trn22: run ONE applicable block after its predecessor finishes

Current warmdown.85 -> momentum150:

```bash
cd /home/poojith-devan/Desktop/Trainium
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn22-warmdown085 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py check trn22-muon150 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-muon150 research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-warmdown085/results-bundle.json
```

Momentum150 -> standalone LR.0168:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn22-muon150 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-lr0168 research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-muon150/results-bundle.json
```

LR.0168 -> genuine baseline control:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn22-lr0168 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-control research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-lr0168/results-bundle.json
```

After a control finishes, use collect trn21-control or collect trn22-control.
These commands archive, but do NOT adopt a change or submit. Root authenticates
metrics, appends one actual ledger row and runs unchanged decide.py.

## What launch does: lock, checks, launch

The command authenticates exact source/argv/proposal/packet, copies over the
pinned SSH connection, hashes on the host, then uses the existing dispatcher.
Inside original exclusive nonblocking flock it checks ps AND neuron-ls,
fuser, STOP, disk/RAM, source/prepare/harness, unused label and old queue-runner
absence. It reacquires the SAME lock and repeats checks at actual launch,
retaining it through training, attached evaluation and all owned descendants.
Host execution is detached; laptop shutdown does not kill the running job.
It is not a bare torchrun or an autonomous unverified scheduler.

Lock: /home/ubuntu/.cache/trainium_freshstart/device0.lock
Original identities: trn21=66305:596823, trn22=66305:832589.
Busy/unknown/error -> refuse, never force. Never replace/unlink the lock.
trn21 currently uses private cold training caches; trn22 existing research
caches. A warm result is not submission-cold evidence. Final package needs
genuine cold replay, fresh20M, Claude audit and Poojith's upload.
