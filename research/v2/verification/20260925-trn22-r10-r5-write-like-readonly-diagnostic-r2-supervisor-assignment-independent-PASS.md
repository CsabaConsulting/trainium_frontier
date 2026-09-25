# Independent assignment review: trn22 r5 read-only diagnostic r2

**Verdict: PASS for one read-only diagnostic assignment only.** This does
not authorize source staging, native training, evaluation, AWS changes,
cleanup, marker reset, or a second call. This was an offline assignment
review: the Operator was not run, and no SSH or other host contact occurred.
The cause of the historical r5 write-phase failure remains **unknown**.

## Exact authority chain and source

- Supervisor assignment commit:
  `37ee50c66057888b105fbf164c8c91b181f57d04` on
  `codex/20260925-trn22-r10-diag-r2-supervisor-assignment`. Its diff from
  independent final review commit
  `d752a0db8b41d27a6ae57639e53944cf67940706` adds only
  `SUPERVISOR_READONLY_ASSIGNMENT.json` and
  `SUPERVISOR_READONLY_ASSIGNMENT_SCOPE.md`; no operational source or
  science file changed.
- Canonical assignment SHA-256:
  `5926a4f028ccbcf3560c1979ef65c300de4a3bfef965d973b768eb04e14b6f12`.
  It equals the operator's exact expected dictionary: r2-only schema and
  assignment ID, request SHA-256
  `8d1393dcb577b3584f532656126216513517155896717adca34b0793c322b04c`,
  independent final-review JSON SHA-256
  `eea65a2e1f9660cd1e71cc33f7d92bc4a65051ee39334ee5bdd75c0a3f2c8fb9`,
  host `trn22`, exact r2 label and read-only scope, and false source-stage
  and native-training authorities. The request and review files on this
  branch match those SHA-256s exactly.
- Independent final-request PASS receipt SHA-256:
  `e7a832d6ad626f0f4ab40609f702762a90e2a0c93c501d2841fee2b4112c9b30`.
  Earlier independent disabled-source static PASS receipt SHA-256:
  `237dbd07e72fb8f2ffebf6d32244e1b4129a7bfb90ccce1a507bb3aff6929f22`.
  The source package pins operator
  `c1c740c22d6127b032b10763c5e5c3202abb199d99be5c0cdc3c846d663e6e18`,
  read-only remote
  `e9cd37be06d644d99b6be1cc4ee73541a66d028998fd172fe03813fb51ac4ada`,
  unchanged r5 stage remote
  `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`,
  and exact bundle
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
- `sha256sum -c SHA256SUMS` passed **13/13** source/evidence/science
  entries. The consumed r5 result receipt matched
  `58461a36b0f6c0a9901be40139cf3da75d65f5c08727f55f16e2fd1ad20d9dc6`;
  the archived journal and retained owner-only, single-linked local marker
  both matched
  `229a331b1aa8b302c607933273a7b09d6b35df9d3892e6ddbad41fb894bb1da8`.
  The designated Desktop execution checkout was clean at the assignment
  commit, with root `train.py` SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.

## Scope and remaining live gates

The Supervisor scope note expressly limits execution to **at most one**
call, states that success or failure consumes this operational assignment,
requires a separate independent assignment check and fresh full local/host
preflight, and names the exact Desktop science checkout/Operator path. It
correctly rejects the older `RUNBOOK.md` command pointing at a disabled
Implementer checkout. The assignment is a JSON/Markdown data change only;
it introduces no executable side effect, credential, or raw host output.

The diagnostic source remains the separately reviewed no-write path: a
pinned read-only GNU-timeout identity probe precedes each phase; local STOP
and breaker checks occur before/after that probe and before each phase;
the `write_like` branch uses read-only checks and never reaches r5
`perform("write")`, `O_CREAT`, or `O_EXCL`. Results are bounded and typed.
This assignment review does **not** assert that the host is presently idle,
that disk/inode thresholds pass, or that STOP/breaker state is unchanged;
those must be checked fresh by the designated Operator before contact.

There is no automatic one-call consumption latch in this read-only
diagnostic. The one-call limit is an operational Supervisor/Operator
obligation: record the single result durably, do not rerun after either
success or failure, and issue no source-stage retry from this assignment.
