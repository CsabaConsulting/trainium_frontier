# trn21 r4/446b final-r5 inode-floor repair — DISABLED / HOLD

**No dispatch authority.** This is a distinct Implementer-prepared successor
to the exact disabled final-r4 private-lock package. The r4 candidate remains
under its [independent static HOLD](../../verification/20260925-trn21-r4-private-lock-independent-static-HOLD.md),
and the r1/r2 HOLDs and r3 disabled-only scope remain in force. The r5 request
has `dispatch_authorized: false` and `source_transfer_authorized: false`.
Its entry point rejects before reading future authority, creating the distinct
r5 one-attempt marker or journal, or starting SSH. No host, AWS, source-transfer,
checkout, training, evaluation, cleanup, breaker-reset or submission action
was performed while preparing r5.

## Exact disabled subject

| New file | SHA-256 |
| --- | --- |
| `TRANSFER_REQUEST.FINAL_R5_HOLD.json` | `6cdd7e966238e20eed16ab0a5abcf8ad89ec324318de539b230b7513ac01d65e` |
| `operator_transfer_final_r5_once.py` | `75b1ad77476d3963208674b5f285efc5e78cca91d895c071c3c8eec9c1f0bdb7` |
| `stage_r4_transport_final_r5.py` | `edcd84c6269b5e78dd1440ea6a565913e33d078350dab574e5a89e4265080db1` |
| `test_operator_transfer_final_r5_once.py` | `ce787cfc5cc933cc66dfc592e4dda84c7b423bc959aa3489633bcbb432aba716` |

Starting canonical commit: `30ff004e7ded31fe881982d3b0533c7f86f94407`.
The r4 candidate canonical commit is
`bca55b2cbc10a4d22184b7308e2c8b9c2624dc5c`. Its exact request,
wrapper, transport, remote, tests and runbook remain unchanged and are pinned
by the r5 request and wrapper. The independent r4 HOLD has SHA-256
`7a7029e30a7a9193bd3e5fd8d241ec28f277ff1bd1966c864d41019a98c196d6`;
its adversarial test file has SHA-256
`b9798cd4d805fa6a1103f61fbaf06fb1fe55e99bdb78c9e1a0735c1d5a8f63da`.
Those files are predecessor evidence, not r5 authorization. The r5 transport
reuses the **unchanged** r4 remote payload at SHA-256
`ad30d93b464175c6df1550654076bc868a8bd8cd652fedd08bd5bfe91f055e2b`.

## Bounded repair

The independent r4 reproducer changed only the terminal
`identity_after.root_free_inodes` of otherwise clean preflight and postflight
receipts from at least `10000` to `9999`. The inherited base validator checked
the starting count but omitted the terminal count; it accepted both full
probes and could record a passing cross-session/postflight result. The writer's
separate inode check did not repair those full-probe receipt gates.

Final-r5's new transport requires **both**
`identity_before.root_free_inodes` and `identity_after.root_free_inodes` to be
plain integers at least `MIN_INODES` (`10000`) for every full probe. It checks
before and after inherited validation, so an unknown/noninteger count or a
synthetic mutation during that call fails closed. The r5 wrapper uses this
validator for preflight and unconditional full read-only postflight; the same
transport binds it to write-argv construction, the writer's full prewrite
probe, preflight-to-write comparison, and cross-session comparison. A count
below the floor at any of those full-probe samples causes HOLD. Equality at
`10000` remains accepted; this repair does not raise or lower the established
floor. The original byte-floor and all other safety gates remain unchanged.

The r4 remote payload, including the exact private `.cache` lock exception,
opened-inode ACL absence, same-namespace mount proof, nonblocking no-follow
lock open, compatible exclusive `flock`, STOP/process/Neuron/target checks,
last broad science-label scan immediately before `O_EXCL`, held lock through
write/receipt, distinct readback, and full postflight, is byte-identical.
The r5 request has a separate marker and fixed future review/assignment paths;
it does not alter any r4 candidate or HOLD file. A later enabled request, if
considered, must be separately named and independently reviewed.

## Offline checks and residual gates

`python3 -B -m unittest -v` in this package passed **58/58** local tests,
including the 51 preserved tests and seven new r5 tests. The r5 negatives
invert the Verifier's `9999` reproducer, cover both identity samples in
preflight and postflight, malformed counts, a drop between preflight/writer/
postflight stages, a synthetic mutation during inherited validation, and
continued lock/ACL/mount/label/readback checks. These are Implementer-local
checks, not an independent r5 PASS or a live host probe.

The r4 private-parent threat model remains bounded: privileged or same-UID
actors, already-open descriptors, another mount namespace, and noncooperating
writers racing the final scan are not excluded by the read-only receipts.
Unknown ACL, changed pinned lock metadata or observed relevant alias remains
HOLD. A different Verifier must review these exact r5 bytes and rerun the
adversarial inode case before any Supervisor could consider a distinct enabled
transfer-only request. Fresh live trn21 identity, protected-asset, quiescence,
disk/inode, STOP/breaker, lock/ACL/mount and science-label preflight would
still be required. No transfer, checkout, CPU G2 or native run is authorized.
