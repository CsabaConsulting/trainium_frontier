# Ordinary-only FP32-subnormal control: completed with mismatches

Assignment: d90e98e2 / precise prose f9206857. Frozen packet 500da845.
Label: `fs-20260926-ordinary-fp32-subnormal-trn21-d1`.

Actual dispatch: 2026-09-26 02:21:19.577558 UTC. Four ranks were observed live at
02:21:38.893908 UTC. Child ended with exit 1 after 22.522854481008835 seconds;
terminal timestamp 02:21:42.218705 UTC, status `DIAGNOSTIC_CHILD_FAILED`.
No timeout, retry, training, evaluation, K2 import, or subsequent launch occurred.

All four rank receipts agree:

- Sentinel and normal-magnitude witnesses: exact forward and gradient results in
  ordinary eager and ordinary compiled routes.
- Positive and negated FP32-subnormal witnesses: forward exact, gradient mismatch
  in **both** ordinary eager and ordinary compiled routes. Actual gradient has
  zero nonzero elements; expected gradient has 1,408. Maximum absolute error is
  1.2611686178923354e-44.
- Every before-forward, before-backward and after-backward input check passed.
- No per-route exception; each rank reports `ValueError: ROUTE_FAILED` from the
  strict validator. The terminal torchrun failure follows these rank failures.

This observation is not exclusive to compiled execution or K2. It is still a
failed diagnostic, not K2 qualification, permission to relax the oracle, or a
source-adoption decision. Independent interpretation belongs to the Supervisor
and Verifier. Original K2 failures remain unchanged.

`results-bundle.json` preserves all 31 raw files (7,295,580 embedded bytes),
authenticated against their individual SHA-256 and size fields. Bundle SHA-256:
`b91f230736f522d20174749b2792da1e3b34c9c842c3cf82889c41ec22152c0f`.
It is byte-identical to the first completed observation, not a fresh rerun.
`ordinary-control-summary.json` retains per-case hashes, exact counts and errors.

Explicit read-only postflight at 02:23:53.884409 UTC found all 11 observed
dispatch/flock/driver/timeout/torchrun/rank/compiler PIDs absent. Original live
PID/starttime/PPID/PGID/SID identities are retained beside the full current process
census. No related workload/compiler processes, no device holders, and idle
`neuron-ls`. Original common lock device/inode 66305:596823 was unchanged and
available. Historical queue PID 2983496 was absent. Postflight SHA-256:
`ca4c80678f3a4fa049cbcbaffbddaaa0103317c2685253dd010b41033585c8a5`.

No artifacts were deleted or checkpoints altered. trn21 idle clock begins at
02:21:42.218705 UTC; its 15-minute threshold is 02:36:42.218705 UTC.
trn22 remains HOLD after the separately archived checkpoint-helper failure.

Local preparation notes: two overbroad template assertions rejected the own
worktree's `trn22` directory name, and a JavaScript string-method typo occurred
before any remote call. They were corrected locally before the sole successful
remote staging/dispatch; frozen packet/runtime/source bytes were not changed.
