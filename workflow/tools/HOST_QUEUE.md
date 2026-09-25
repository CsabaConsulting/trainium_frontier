# Host serial queue (Final Push, 25 September)

This small runner extends the proven trn21 manual A1 driver at `f54aad76` with
the same `device0.lock` used by both current detached A1/A2 drivers. It retains
the exact full-chip LNC2/four-rank training command, warm cache environment,
and fresh-process public-20M/context-1024 `prepare.py eval-public` command.
It does not use the inactive autoloop or the newer `run_job.py` staging path.

Only an Operator installs/starts it after independent code review. It supports
the already-verified c5c source only. Model/evaluator bytes are never edited;
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
or freeze stops the queue. Logs and checkpoints are preserved, with no cleanup.

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
