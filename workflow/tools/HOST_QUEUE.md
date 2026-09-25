# Host serial queue (Final Push, 25 September)

This small runner extends the proven trn21 manual A1 driver at `f54aad76` with
the same `device0.lock` used by both current detached A1/A2 drivers. It retains
the exact full-chip LNC2/four-rank training command, warm cache environment,
and fresh-process public-20M/context-1024 `prepare.py eval-public` command.
It does not use the inactive autoloop or the newer `run_job.py` staging path.

Only an Operator installs/starts it after independent code review. It supports
the already-verified c5c source and the exact f9b6a9cb schedule candidate only
when its independent review, exact-argv production CPU G2 and flag-off identity
receipts pass and are pinned in the source registry. Model/evaluator bytes are never edited;
the model is copied to a new per-label directory and hash-checked. All proposal
files must be staged as regular files and each binds the exact host/label/argv.

Manifest fields (JSON):

```
schema: 1
queue_id: unique descriptive label
host: trn21 or trn22
expected_hostname: actual hostname checked by the Operator
armed: false until Operator/Supervisor finishes the ready manifest
freeze_utc: 2026-09-29T12:30:00+00:00
source: {path: absolute, sha256: c5c410af...full SHA}
prepare: {path: absolute, sha256: 5807c46e...full SHA}
python, torchrun: absolute native-venv executable paths
train_cache, eval_cache: existing warm cache roots, with all six subdirectories
run_root: dedicated absolute directory; consumed labels live here permanently
device_lock: /home/ubuntu/.cache/trainium_freshstart/device0.lock
stop_paths: all applicable absolute STOP marker paths
jobs: [{label, proposal: {path, sha256}, argv, kind, arm_id, base_sha256,
        armed: false, readiness: READY_PENDING_HOST_PREFLIGHT, base_science_argv,
        same_host_control: {run_label, public_val_bpb, steps, review}}]
```

Each proposal must have `status: READY_PENDING_HOST_PREFLIGHT` and
`queue_binding: {host,label,source_sha256,argv,kind,arm_id,base_sha256}` exactly
matching the manifest. `argv` is an array passed directly to `torchrun`; it is
never shell-evaluated. Use explicit `--seed 42 --max-train-seconds 1800
--num-steps 1000000 --out-dir out --seq-len 1024 --no-eval-public`. The runner
does not append or alter flags. No arbitrary environment/command override is
accepted. The supplied manifest SHA pins all remaining flag values and paths.

Each job may specify `source_sha256` independently of `base_sha256`. The latter
is the comparison/control identity, which may be a composite SHA over the
source plus baseline argv when a winning table profile becomes the base.
If `same_host_control.base_sha256` is supplied, it must match that base ID.
This prevents comparing a table-base follow-up to unrelated dense controls.
Table records may carry the authorized `throughput_exempt: true` and
`training_budget_seconds: 1800` metadata; the host never computes a verdict.

Operator validates then starts detached, with the actual manifest SHA:

```
python3 -I -B host_queue.py --manifest /absolute/manifest.json --manifest-sha256 SHA
nohup python3 -I -B host_queue.py --manifest /absolute/manifest.json --manifest-sha256 SHA --run > /absolute/queue.log 2>&1 < /dev/null &
```

The default command validates schema only and launches nothing. `--run` requires
`armed: true` both on the queue and every job, correct machine hostname, exact source/evaluator hashes, intact
STOP conditions, free disk >=6.5 GiB, an idle process/device check and the common
device lock. The runner waits for the current detached run/eval to release that
lock. It retains the lock through training, evaluation and recording; the phase
child inherits it so a killed queue parent cannot cause another run to overlap.

Every phase logs its argv/environment, PID and exit status. `terminal.json`
records success or the real failure; `results.jsonl` is the host delivery spool
for the controller to retrieve and merge, not another scientific decision
ledger. Successful records remain `evaluated_pending_review`; no causality
check, adoption, score improvement or canonical verdict is invented. Failed
train/eval, invalid budget/shape/loss/token count, source drift, STOP, low disk
stops the queue. Freeze blocks new training starts, while a job already in
progress finishes its mandatory fresh evaluation and recording. Logs and
checkpoints are preserved, with no cleanup.

Restarting the same queue skips completed labels and repairs a missing spool
append from their terminal receipts. A consumed label without a terminal
receipt is observed until the host is idle, marked `interrupted`, and stops the
queue. It is never restarted, and no result from it is accepted. A surviving
host queue after a laptop disconnect needs observation only, not another start.

Create `run_root/PAUSE_NEXT` to stop **between** jobs. Do not kill an active job
to reorder the queue. When the queue reports `paused_between_jobs` and releases
the device lock, the Supervisor may supply a newly hashed manifest with a
different unstarted tail, preserving all consumed label directories. The
Operator removes only its PAUSE_NEXT marker and restarts with the new hash.
Thus these flag-bank fallbacks do not outrank new winners, replicates or stacks.

Local tests: `python3 -m unittest discover -s workflow/tools -p test_host_queue.py`.
They exercise fake sequential training/eval, durable duplicate prevention,
failed training, incomplete evaluation, changed bytes/STOP, proposal mismatch,
pause-before-launch, and a real SIGKILL of the parent with a surviving phase
that holds the lock. They require no host or Neuron access.

## Install before the table decision, then resolve from the controller

Add `resolution_inbox: /absolute/next-resolution.json` to the bootstrap
manifest, use `jobs: []`, and set `armed: true`. This starts an idle detached
waiter which holds no device lock while waiting for the controller's choice.
It cannot silently fall through to the old flag bank. The controller retrieves
both table results, resolves the user-approved branch, then stages a new
immutable complete manifest with the same host configuration and selected
ready jobs. Publish its pointer by writing a temporary file then atomically
renaming it to the inbox:

```json
{"manifest_path":"/absolute/resolved-v1.json","manifest_sha256":"FULL_SHA256"}
```

The digest is a content pin, not a cryptographic signature. Only the authorized
controller/Operator writes the inbox. It can publish `{"hold":true}` or leave
the inbox absent to stop new starts while another decision is pending. Each
resolved manifest keeps the bootstrap's source, evaluator, host, paths, cache,
STOP and freeze configuration unchanged. Jobs and the source registry may
change. Replacement tails are checked before every new job and again after
acquiring the shared device lock, so a table win can cancel old flag jobs even
while the waiter is blocked behind A1/A2. A running job always completes its
train/eval/receipt sequence before a replacement is considered. Completed or
started labels remain consumed across every replacement.

For schedule-source jobs, the resolved manifest needs this explicit registry:

```
approved_sources:
  FULL_SOURCE_SHA:
    path: /absolute/exact-schedule-train.py
    gates:
      diff_review: {path, sha256, source_sha256, passed: true}
      flag_off_identity: {path, sha256, source_sha256, passed: true}
      cpu_g2: {path, sha256, source_sha256, passed: true, argv: [exact job argv]}
```

`cpu_g2` may instead be a list of these descriptors when several exact argv
profiles have separate PASS receipts. Every receipt file is hash-checked;
every declaration binds the selected source SHA. At least one CPU receipt
must match the job's exact argv. The Supervisor supplies `passed: true` only
after reviewing the real evidence; the runner does not generate that approval.
A new SHA outside c5c/f9b6a9cb is rejected until separately integrated/reviewed.

The regression suite also tests empty-inbox waiting without holding the device,
replacement/cancellation before a lock becomes available, unapproved and
changed source rejection, and exact source gates with a distinct scientific
comparison base.
