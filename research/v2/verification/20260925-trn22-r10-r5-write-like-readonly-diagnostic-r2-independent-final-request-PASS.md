# Independent final request review: trn22 r5 read-only diagnostic r2

**Verdict: PASS_READONLY_DIAGNOSTIC_ONLY for the exact enabled-pending
request.** This is not a Supervisor assignment, permission to contact trn22,
permission to retry source staging, or authorization for training, evaluation,
AWS action, cleanup, or marker reset. No host contact was made during this
review. The historical r5 write-phase failure is still of **unknown** cause.

## Binding

- Enabled-pending Implementer commit:
  `0fe98ae29bc82b684756d36c677975a17b87328d` on
  `codex/20260925-trn22-r10-write-diag-r2-enabled-pending`. Its sole change
  relative to the disabled source commit
  `e9ef3115c61aaac6740db4e84069645533f6bfa9` is the new canonical
  `REQUEST.ENABLED_AFTER_REVIEW.json`, SHA-256
  `8d1393dcb577b3584f532656126216513517155896717adca34b0793c322b04c`.
- Independent disabled-source static PASS: commit
  `6fe2b2a1e1d9828fe66349ef4cb275982e0597f7`, receipt SHA-256
  `237dbd07e72fb8f2ffebf6d32244e1b4129a7bfb90ccce1a507bb3aff6929f22`.
  The exact operator SHA-256 is
  `c1c740c22d6127b032b10763c5e5c3202abb199d99be5c0cdc3c846d663e6e18`;
  read-only remote SHA-256 is
  `e9cd37be06d644d99b6be1cc4ee73541a66d028998fd172fe03813fb51ac4ada`.
- Unchanged r5 stage remote SHA-256 is
  `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`;
  exact 109,800-byte e65-to-r10 bundle SHA-256 is
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
  Source/target commits and root science hashes match the pinned r2 static
  receipt. `sha256sum -c SHA256SUMS` passed **13/13** entries, including the
  consumed r5 result receipt and archived journal. The existing local r5
  consumed marker remained owner-only mode 0600, single-linked, and SHA-256
  `229a331b1aa8b302c607933273a7b09d6b35df9d3892e6ddbad41fb894bb1da8`.
- This final JSON review is exact canonical JSON with schema
  `trn22-r10-r5-readonly-write-like-diagnostic-r2-independent-final-v1`,
  verdict `PASS_READONLY_DIAGNOSTIC_ONLY`, request/source/bundle hashes,
  r2 diagnostic label and scope, and `source_stage_authorized=false`.
  SHA-256: `eea65a2e1f9660cd1e71cc33f7d92bc4a65051ee39334ee5bdd75c0a3f2c8fb9`.

## Independent checks

- The enabled request exactly equals the operator's `_authority` expected
  dictionary. It changes only schema and `dispatch_authorized` from the
  disabled request; source-stage, native, eval, AWS, cleanup, and marker-reset
  authorities all remain false. It cannot reuse the r1 diagnostic or the
  consumed r5 source-stage assignment.
- Normal and optimized (`-O`) unit suites each passed **20/20** tests.
  Direct normal entrypoint without the final review/assignment returned
  `HOLD_PRE_DISPATCH`, exit 1, before SSH. After adding this final review,
  a direct `_authority` check accepted the exact request and review, then
  refused solely at the absent Supervisor assignment. No host connection
  was attempted.
- Source/remote hashes are unchanged from the disabled static PASS. The
  separate pinned GNU-timeout identity SSH precedes each timed phase, with
  local active-Desktop/former-Documents/package STOP and breaker checks
  before and after the probe and immediately before the phase. The
  `write_like` branch uses only read-only probe/flock/root-fd/freshness
  checks and does not invoke r5 `perform("write")`, `O_CREAT`, or `O_EXCL`.
  The one-second pre-open gate is unchanged. The consumed marker is only
  checked by hash. Operator output is typed, bounded, and excludes raw SSH
  streams, exception text, keys, and environment contents.

## Remaining dispatch gates and integration caveat

The distinct `SUPERVISOR_READONLY_ASSIGNMENT.json` is absent and must remain
separately authorized. The operator requires schema
`trn22-r10-r5-readonly-write-like-diagnostic-r2-supervisor-v1`, assignment ID
`trn22-r10-r5-readonly-write-like-diagnostic-r2`, the exact request SHA-256
above, this review JSON SHA-256, scope, host `trn22`, exact r2 diagnostic
label, and false source-stage/native authorities. Its full equality check
must pass; no older assignment is transferable.

The runbook's proposed command points to the original disabled-source
worktree, whereas this review JSON is committed only on the independent
reviewer branch. That stale checkout must **not** be treated as ready to
dispatch. Before any Operator contact, an authorized Supervisor must
assemble the exact reviewed request, review, and assignment in one clean
science-identical checkout, identify the exact command path, and run fresh
local STOP/breaker/identity and host-capacity checks. Even a completed
read-only diagnostic would not prove a safe source-stage retry or lower BPB.
