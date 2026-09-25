# Root-free guard monitor: static package, no dispatch authority

This package is an external safety mechanism for the **unused** trn22 c5c A0
science label `fs-20260925-v2-c5c-a0-control-trn22-r2`. It does not change
`train.py`, `prepare.py`, the evaluator, the pinned launcher, or the reviewed
science request. The monitor script is host-neutral; the accompanying binding
pins one host, label, source and request. **No monitor or native launch has been
authorized by this package.**

## Exact binding and what the script does

- Reviewed pending science request:
  `research/v2/authorizations/fs-20260925-v2-c5c-a0-control-trn22-r2.request.pending.json`,
  SHA-256 `7c4d1af77fe8c332170feebf444e0c292ef8d78ff3bf76d5ef3056533e2279cc`.
- Prior no-launch monitor hold:
  `research/v2/authorizations/fs-20260925-v2-c5c-a0-control-trn22-r2.monitor-hold.json`,
  SHA-256 `6a740101fbc5e6db52bb031330a6c65d85cba3ff08b05e929d8de9bc2d75c044`.
- Source commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c`, c5c train
  SHA-256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
- `root_free_guard_monitor.py` SHA-256 is recorded in
  `trn22-a0-r2.binding.STATIC-ONLY.json`; verify both before staging. The
  future one-shot monitor authorization must bind the binding file's **exact
  SHA-256**, the request SHA, and this one label. No authorization file is
  supplied here, so the script refuses to arm.
- It must be started **before** the pinned launcher. Before writing its
  fsynced `READY_BEFORE_LAUNCH` receipt, it checks the hostname, UID, pinned
  request/source/guard hashes and commit, unique absent job/process/launch/
  postflight paths, and at least `10,737,418,240` bytes root free.
- It polls `statvfs("/")` every 0.25 s; any observed gap over 1 s is a
  monitor failure. Every root-free sample and event is appended and fsynced
  to a unique external JSONL file; no existing evidence is overwritten.
- It waits for the pinned launcher-created `job.json` and `process.json`,
  matches **every original request field** in the job, checks the launch
  train-source hash, exact label/command in the process receipt, and authenticates
  the live guard and direct child from `/proc`: expected UID and Python
  executable, exact guard-script/job argv, exact child command, session and
  process-group ownership, and process birth ticks. A pidfd pins the guard
  against numeric PID reuse.
- At root free **≤ `9,126,805,504` bytes** (the immutable
  `6,979,321,856`-byte floor plus 2 GiB), or on a monitor fault after binding,
  it rechecks source, job/process hashes and both live identities, fsyncs a
  `TERM_INTENT`, then sends **SIGTERM only to the owned guard pidfd**. The
  pinned guard handles its own scoped child cleanup and postflight. The monitor
  never signals a process group, unknown process or reused PID, and never
  deletes/checkpoint-prunes anything.
- Missing or changed identity means `SIGNAL_REFUSED` / failed gate and urgent
  Operator intervention, **not** permission for a blind kill. A natural
  completion requires the guard's same-label `host_clean=true` postflight.
  Any disk stop, monitor fault, absent postflight or changed identity makes
  the science run invalid regardless of any model result.

## Independent gate before any native dispatch

1. A separate Verifier reviews the script, binding, exact request/hold,
   `fresh_start_launch.py`/guard behavior and test output. It must confirm
   that the external monitor is not mistaken for an optimizer-boundary
   observer. A static PASS is **not** a host/launch PASS.
2. An Operator may then stage only the reviewed script, binding and exact
   request into the binding's external `/home/ubuntu/trainium_a0_disk_monitor_20260925_r1`
   directory, owned by the expected UID and not group/world writable. Verify
   SHA-256 after transfer, `/usr/bin/python3` resolves to the interpreter
   actually used by the pinned launcher, and `id -u` is `1000`. Any mismatch
   requires a new binding and independent review; do not weaken checks.
3. Re-run the complete fresh host preflight: trn22 hostname, clean exact
   source commit/hashes, exact source tools and pending-request hash, all six
   warm-cache identities, G1/G2 receipts, no STOP/breaker, no process/device/
   host-lock holder, unique label/output paths, root free at least
   `10,737,418,240` bytes, adequate inodes/RAM, and the boundary observer
   ready before first step. The historical r1 failed label is not reusable.
4. The Supervisor issues a **separate one-label, one-attempt** monitor
   authorization JSON with schema
   `one-shot-root-free-monitor-authorization-v1`, exact binding SHA in
   `binding_sha256`, exact pending-request SHA in `request_sha256`, exact
   `label`, `one_attempt_only=true`, and `authorized=true`. It also binds the
   independent review and fresh live preflight in its separate native
   authorization. An Operator alone may execute a single reviewed attempt.
   Never pass the pending request to `fresh_start_launch.py` merely because
   its own `dispatch_authorized=false` metadata exists: that old launcher
   does not enforce this field.
5. Start this monitor first with `--binding` and `--authorization`, retaining
   a supervised live process and external stderr capture. Parse the fsynced
   ready receipt, verify its PID/UID/birth ticks/script/request/binding hashes,
   and independently verify the process and fresh root free. Its job/process
   bind timeout is 120 s, so dispatch must follow promptly. If readiness is
   absent or the monitor is not alive, **do not launch**.
6. An independent Operator-side liveness watcher must continue checking the
   monitor's PID/birth ticks and latest fsynced sample at ≤1 s intervals while
   the guard runs. A dead/stale monitor invalidates the run; stop only through
   an authenticated owned-guard procedure, never by PID guess. If such a
   watcher and failure response are not bound in the final one-shot run plan,
   **dispatch remains HOLD**. The monitor's SIGTERM/SIGINT handler itself
   invokes its authenticated fail-closed stop, but SIGKILL or total host
   failure requires the separate watcher/Operator response.
7. Preserve external monitor ready/events/stderr alongside the pinned launch,
   boundary, train, postflight and quiescence evidence. Verify every SHA and
   have a different Verifier/Analyst adjudicate. The monitor's exit `0` means
   only clean guard completion; `2` means stopped or failed job; `3` means
   monitor failure. None is a BPB result or authorizes the next job.

Synthetic-only tests (no EC2, no Neuron, no real signal):

```bash
python3 -m unittest discover -s research/v2/disk-monitor -p 'test_*.py' -v
```
