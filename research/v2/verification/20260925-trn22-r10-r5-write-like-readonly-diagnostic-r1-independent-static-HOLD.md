# Independent static review: trn22 r5 write-like read-only diagnostic r1

**Verdict: STATIC HOLD.** This reviews disabled source only. It does not authorize
SSH, source staging, training, evaluation, AWS changes, cleanup, or marker reset.
The historical r5 write-phase failure remains of **unknown** cause; no live host
diagnosis was performed.

## Exact reviewed identity

- Implementer commit: `5b60d9ec3590f20ea4d4cc29dfecd57062e24124` on
  `codex/20260925-trn22-r10-write-like-readonly-diagnostic-r1-disabled`.
- Package: `research/v2/authorizations/trn22-r10-r5-write-like-readonly-diagnostic-r1-hold/`.
- Operator `operator_readonly.py`: SHA-256
  `9cb15520264c24c0991bb2d8307cdec60db7986289753e26aedc166d8dab295c`.
- Remote diagnostic `remote_readonly.py`: SHA-256
  `e9cd37be06d644d99b6be1cc4ee73541a66d028998fd172fe03813fb51ac4ada`.
- Canonical disabled request: SHA-256
  `2ee9253639a22b37f58f4b191bb2b39018ab02d7d9ea6be0ccd8f21ca9d1b3f9`,
  with `dispatch_authorized=false` and all source-stage/native/eval/AWS/cleanup/
  marker-reset authorities false.
- Unchanged r5 stage remote: SHA-256
  `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`;
  exact e65-to-r10 bundle: 109,800 bytes, SHA-256
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
  The pinned source/target commits are
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c` and
  `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`. Science files remain
  `train.py` `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `prepare.py` `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- `sha256sum -c SHA256SUMS` verified all 12 package/evidence/science entries.
  The source worktree was clean at the reviewed commit.

## What passed offline

- Normal and optimized (`-O`) suites each passed **16/16** tests. Generated
  remote Python compiled. The direct entrypoint, with no enabled request,
  returned `HOLD_PRE_DISPATCH` (exit 1) before any SSH.
- The only newly reachable remote branch for `write_like` is
  `remote_readonly.perform_readonly`; it uses the exact r5 read-only probe,
  stability check, source/protected/label/activity/disk checks, existing
  read-only flock, bundle-stdin hash, `O_RDONLY|O_DIRECTORY|O_NOFOLLOW`
  root-directory open, and unchanged one-second freshness checks. It never
  calls r5 `_write_bundle` or `_reconstruct`, nor opens the bundle with
  `O_CREAT`/`O_EXCL`. The preflight/postflight paths call only r5
  `perform("preflight")`/`perform("postflight")`.
- The consumed r5 marker is checked by hash, not reopened for writing or
  reset. Local active-Desktop/former-Documents/package STOP and breaker
  checks, exact source/bundle/science pins, and separate absent diagnostic
  request/review/assignment remain fail-closed. Published results reduce
  remote streams to bounded byte counts, SHA-256, typed statuses and
  whitelisted finite fields; raw stdout/stderr and exception text are not
  emitted.

## Blocking defect: timeout identity is checked after its use

`operator_readonly.py:380-397` invokes the phase SSH with remote command
starting `/usr/bin/timeout --foreground ...`; its only GNU-timeout version
check is embedded inside the Python process that this very timeout starts
(`operator_readonly.py:270-278`). The previously reviewed r5 stage operator
performed a **separate read-only SSH** of
`/usr/bin/env -i LC_ALL=C /usr/bin/timeout --version` before every phase and
rechecked local STOP/breakers after that probe (`operator_once.py:232-270` at
the reviewed r5 source). The new runbook claims the same pinned timeout
identity gate, but dispatch currently bypasses its ordering.

An offline mocked call to `_ssh_phase("preflight", ...)` recorded exactly one
`subprocess.run` SSH call: the first remote command started the outer timeout,
and no preceding version-probe SSH occurred. This witness made **no** network
call. Current tests pass because they do not assert this call order.

**Minimal repair:** before each phase, perform the prior bounded, pinned,
read-only remote GNU-timeout version probe; check local STOP/breakers before
that SSH, again after it, and immediately before the phase SSH. Reject any
nonzero result, stderr, oversize output, or version mismatch before starting
the timed phase. Add adversarial tests showing a malformed probe or an
intervening STOP prevents the phase SSH. Preserve all source/bundle pins and
the disabled authority. Re-review exact repaired bytes independently before
creating an enabled request or contacting trn22.
