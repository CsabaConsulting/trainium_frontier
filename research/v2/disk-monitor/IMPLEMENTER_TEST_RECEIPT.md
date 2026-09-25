# Implementer-only synthetic receipt — not an independent PASS

Recorded 2026-09-24 23:57 UTC. Scope: host-neutral external root-free
monitor and disabled binding for trn22 c5c A0 r2. No EC2 connection, native
training, evaluator, guard signal, deletion, or launcher invocation occurred.

- `root_free_guard_monitor.py`: SHA-256
  `ae7538e01c5bff1338bc3c469a7d1e0980e12ade30daff99ae29c5c56ee54511`.
- `trn22-a0-r2.binding.STATIC-ONLY.json`: SHA-256
  `049e299d184da4c9b61253deab1a2e2dd0b3c38ba1eb7a3d81913ef0e419e64e`.
- Exact pending science request: SHA-256
  `7c4d1af77fe8c332170feebf444e0c292ef8d78ff3bf76d5ef3056533e2279cc`.
- Local synthetic command:
  `python3 -m unittest discover -s research/v2/disk-monitor -p 'test_*.py' -q`.
  Result: 13 tests, all PASS. Cases include threshold equality, no false
  signal at safe free space, changed guard and child birth ticks, foreign UID,
  changed job manifest, missing process receipt, pre-binding low space,
  delayed poll, synthetic `statvfs` failure, static-package authorization
  refusal, append-only evidence, and real local `/proc/self` parsing. All signal calls are fake
  objects; no real process was signaled.

**Verdict: IMPLEMENTER COMPLETE / DISPATCH HOLD.** A separate Verifier must
review the code and rerun adversarial tests. Host-local staging, exact UID and
Python path checks, an independent liveness watcher, fresh live preflight,
one-label Supervisor authorization, and Operator execution remain distinct
gates. There is no new BPB result.
