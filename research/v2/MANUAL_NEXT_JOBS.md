# Manual next jobs — FINAL SPRINT, 26 September takeover

Run these from a laptop terminal. No AI/session quota is needed for these
Python/SSH commands. Keep the existing SSH configuration; do not paste keys.
The current host job AND attached evaluation must finish first.

Latest human order: first warmdown0.60 ontrn21 and0.85 ontrn22, launched directly
by this session, not dependent on the scheduler. Do not restart them if already
running. Subsequent rows are prepared fallback brackets on the current confirmed
batch262144/LR.021 base; a newly confirmed better stack or qualifying replica
supersedes those fallbacks. Update this guide before changing that order.

## Three jobs per host

| Host | Order | Exact job key | One change vs confirmed base | Readiness |
|---|---:|---|---|---|
| trn21 | 1 | trn21-warmdown060 | warmdown .75 -> .60 | Source/CPU/review and local transport PASS; fresh machine checks pending |
| trn21 | 2 | trn21-warmdown085 | warmdown .75 -> .85 | Prepared fallback, same shared source/review; local transport PASS |
| trn21 | 3 | trn21-lr0168 | matrix LR .021 -> .0168; warmdown .75 | Prepared fallback, same shared source/review; local transport PASS |
| trn22 | 1 | trn22-warmdown085 | warmdown .75 -> .85 | Source/CPU/review and local transport PASS; fresh machine checks pending |
| trn22 | 2 | trn22-warmdown060 | warmdown .75 -> .60 | Prepared fallback, same shared source/review; local transport PASS |
| trn22 | 3 | trn22-lr0168 | matrix LR .021 -> .0168; warmdown .75 | Prepared fallback, same shared source/review; local transport PASS |

All use source SHA256
`4ec8b1b46562612cdcbfe3d7f66b0c8d25666bdbac3516b30bd9149c2dc08d2d`,
seed42, eight layers x1024, full-chip LNC2/four ranks,1800-second official
accounting, fresh public20M/context1024. Exact source, argv, proposal, shared
review and CPU receipt are carried in each packet. No new BPB is claimed.
trn21 packets use private cold training caches; trn22 uses existing research
caches (cache hit not proven). Only measured cold runs qualify a submission.

## Setup and read-only check

```bash
cd /home/poojith-devan/Desktop/Trainium
python3 research/v2/operator/20260926-operator-recovery/manual_job.py status trn21
python3 research/v2/operator/20260926-operator-recovery/manual_job.py status trn22
```

These commands print `ps` job rows, `neuron-ls`, original lock state, free disk
and latest train/eval/terminal tails. If busy, leave it alone. Check again in
20–30 minutes. Never delete a lock, kill a job, lower a resource gate, or use
`python -O`. A check failure is not proof the host is idle.

## trn21 — run ONE block only, after its predecessor finishes

First, retrieve the currently running batch/LR+schedule result, then launch.60:

```bash
cd /home/poojith-devan/Desktop/Trainium
test -f research/v2/operator/20260926-operator-recovery/trn21-current-completed/results-bundle.json || python3 research/v2/operator/20260926-operator-recovery/retrieve_current.py trn21 --execute
python3 research/v2/operator/20260926-operator-recovery/manual_job.py check trn21-warmdown060 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn21-warmdown060
```

After .60 and its20M eval finish, archive it and launch.85 if still selected:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn21-warmdown060 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn21-warmdown085 research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-warmdown060/results-bundle.json
```

After .85 and its20M eval finish, archive it and launch LR.0168 if still selected:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn21-warmdown085 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn21-lr0168 research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-warmdown085/results-bundle.json
```

## trn22 — run ONE block only, after its predecessor finishes

First, retrieve the currently running LR.02625 result, then launch.85:

```bash
cd /home/poojith-devan/Desktop/Trainium
test -f research/v2/operator/20260926-operator-recovery/trn22-current-completed/results-bundle.json || python3 research/v2/operator/20260926-operator-recovery/retrieve_current.py trn22 --execute
python3 research/v2/operator/20260926-operator-recovery/manual_job.py check trn22-warmdown085 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-warmdown085
```

After .85 and its20M eval finish, archive it and launch.60 if still selected:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn22-warmdown085 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-warmdown060 research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-warmdown085/results-bundle.json
```

After .60 and its20M eval finish, archive it and launch LR.0168 if still selected:

```bash
python3 research/v2/operator/20260926-operator-recovery/manual_job.py collect trn22-warmdown060 && python3 research/v2/operator/20260926-operator-recovery/manual_job.py launch trn22-lr0168 research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-warmdown060/results-bundle.json
```

After the third job finishes, use `manual_job.py collect trn21-lr0168` or
`manual_job.py collect trn22-lr0168`. Retain all results/checkpoints; do not
claim adoption or submit from this fallback procedure. The main session
authenticates results, updates the ledger and runs the unchanged selector.

## What the launch command does — lock, checks, launch

`manual_job.py launch` authenticates its exact packet/proposal/source and prior
completed result, copies the bytes over the pinned SSH connection, verifies
all hashes on the host, and invokes the existing checked dispatcher. Inside
the original exclusive nonblocking `flock`, it checks `ps`, `neuron-ls`,
`fuser`, STOP files, disk/RAM, exact source/prepare/harness, unused label and
old queue-runner absence. The dispatcher reacquires the SAME lock and repeats
the idle checks; it retains the lock through training, evaluation and owned
descendant cleanup. It detaches on the host, so closing the laptop terminal
does not stop training. This is not a bare unsafe `torchrun` shortcut.

Locks: `/home/ubuntu/.cache/trainium_freshstart/device0.lock`;
trn21 device/inode66305:596823, trn22 66305:832589. Never replace/unlink either.
Busy/unknown/error => no launch. A consumed attempt is not retried under the
same label. Preserve the exact error and let the Operator diagnose it.

Successful `DETACHED_DISPATCH_STARTED` is not proof that training began:
run the read-only status command once to confirm actual workers/lock/logs.
If collection was already archived, reuse its exact bundle rather than deleting
it; launch alone is safe only after verifying it is the intended predecessor.

## Controller and outage behavior

Saved controller schedule is25minutes; old04:00/07:45 tasks are PAUSED.
Scheduler pickup/next tick has not yet been observed. No dependence on it for
the two immediate launches. A stale scheduled session cannot bypass the same
host lock. No extra agents, host-resident queue loop, AWS action or upload is
authorized by this document. All guards apply equally to manual launches.
