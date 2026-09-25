# r3 paired-monitor interlock — Implementer self-test receipt

This is **not** an independent Verifier approval and **not** an Operator
authorization. The r3 request has `dispatch_authorized: false`; no one-shot
authorization file exists. No EC2, AWS, SSH, native training, or leaderboard
action was performed in this implementation pass.

Source commit: `e4272b4dbd8e7b676694afec14b248fef588f8a8`.
New label: `fs-20260925-v2-c5c-a0-control-trn22-r3`.

Local command:

```text
python3 -m unittest research/test_fresh_start_monitor_interlock.py research/v2/disk-monitor/test_paired_root_free_guard_monitor.py research/v2/disk-monitor/test_root_free_guard_monitor.py -v
```

Result: **36 synthetic tests passed**. Coverage includes exact two-monitor
success; stale/dead/both-unavailable monitor refusal; PID reuse and foreign
argv refusal; missing mutual handshake; monitor failure event; changed script;
10 GiB launch gate; absent authorization; non-overwriting no-spawn receipt;
remote launch string syntax/order; peer death after guard binding; peer death
during preflight with a late process receipt; no-spawn receipt response; a
missing process receipt; foreign guard refusal; the original disk threshold,
PIDFD, source, and postflight checks. No test sends a real signal.

Static checks: `python3 -m py_compile` passed for the changed scripts;
`git diff --check` passed. The r3 command array equals the r2 command after
only r2-to-r3 label/output path substitution; environment maps match exactly.
`train.py` and `prepare.py` SHA-256s remain c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586
and 5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb.

The safety claim is limited to one monitor failing at a time and an exact
guard receipt. Simultaneous monitor/host/kernel failure, persistent disk I/O
failure, or unprovable guard identity (including a guard crash before its
process receipt is fsynced) are residual HOLD conditions. The earlier
exact-environment CPU G2 r3 pins source commit `e65a3097...`; the new
launcher-only source is `e4272b4d...`. I did not run a new G2 or rebind the
old receipt. The proposal records the source-identity gate as false pending
independent disposition; see `R3_INTERLOCK_RUNBOOK.md`.
