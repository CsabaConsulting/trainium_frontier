# trn21 r4/446b final-r5 source-transfer candidate — HOLD pending final review and assignment

This is a **distinct Implementer-prepared, transfer-only enabled-request
candidate**, not an Operator instruction or a native-launch approval. Its two
request flags express source-transfer intent, but the wrapper cannot reach its
one-attempt marker, journal, or SSH unless a separately issued, exact
independent final `PASS` and Supervisor one-shot assignment both exist at the
fixed paths in the request. Neither authority file was created here. This
candidate has not contacted trn21 or AWS, transferred source, checked out a
tree, run CPU G2 or native training, evaluated, cleaned, reset a breaker, or
submitted a score. The r1–r4/r5 historical files and HOLDs are unchanged.

## Exact candidate and reviewed predecessor

The candidate starts from canonical `73be2973dda3a0c1ff2d62a37612aa95a08f0027`,
which carries disabled r5 origin `0bacf0d009b4894264a2f8097b7e5cad8286bf5a`
and its [independent disabled-only static PASS](../../verification/20260925-trn21-r5-inode-floor-disabled-independent-static-PASS.md).
All hashes below are SHA-256 of actual file bytes; paths are relative to this
package unless otherwise shown.

| Subject | SHA-256 |
| --- | --- |
| `TRANSFER_REQUEST.FINAL_R5_ENABLED_PENDING_REVIEW.json` | `b611a7546d1c73d338dea14d083432b16fbea5616bd3002c5dd57850d1f0710c` |
| `operator_transfer_final_r5_enabled_once.py` | `327e4d687d28a1412fd769b80ba47463e13f5c32454dc51d906b59fcf23b1590` |
| `test_operator_transfer_final_r5_enabled_once.py` | `7990ea88376f29b7f25ebe86a2beb294d628c2f5d41633d2e41e2f96e453054b` |
| Reused r5 `stage_r4_transport_final_r5.py` | `edcd84c6269b5e78dd1440ea6a565913e33d078350dab574e5a89e4265080db1` |
| Reused r4 `stage_r4_remote_final_r4.py` | `ad30d93b464175c6df1550654076bc868a8bd8cd652fedd08bd5bfe91f055e2b` |
| Disabled r5 request | `6cdd7e966238e20eed16ab0a5abcf8ad89ec324318de539b230b7513ac01d65e` |
| Disabled r5 wrapper | `75b1ad77476d3963208674b5f285efc5e78cca91d895c071c3c8eec9c1f0bdb7` |
| Disabled r5 tests | `ce787cfc5cc933cc66dfc592e4dda84c7b423bc959aa3489633bcbb432aba716` |
| Disabled r5 runbook | `9b8c6239fc40b2d19b2ad09c98fdd48b4013156e23ebe66eb021955d1c8397a5` |
| Independent r5 disabled-only PASS | `be212cbbd50ac525049602c87b584fb2e8d21cca6c3e03ed29fcc587e67a58b4` |
| Independent r5 negative tests | `22997ceb8b89bb93340a227da7f790c40079c587aba6b5854fc1818f3ba72515` |

The wrapper pins and re-reads the reviewed predecessor bytes and PASS,
requires the disabled predecessor's two flags to remain false, and binds the
new request hash. Its future final-review and assignment shapes bind the
actual bytes of this wrapper, candidate tests, and this runbook. The final
review's byte hash is then bound by the Supervisor assignment. The candidate
runbook's own hash is deliberately not self-referential; a future independent
review must record it exactly. The old r4 independent HOLD remains pinned at
`7a7029e30a7a9193bd3e5fd8d241ec28f277ff1bd1966c864d41019a98c196d6`.

## One-attempt authority boundary

The request uses a new transfer-attempt label
`trn21-c5c-a0-r4-source-transfer-r5-enabled-r1-20260925`, assignment ID
`trn21-c5c-a0-r4-transfer-final-r5-enabled-r1`, and distinct fixed marker
ending `trn21-c5c-a0-r4-transfer-final-r5-enabled-r1-one-attempt.marker`.
The frozen science label `fs-20260925-v2-c5c-a0-control-trn21-r4` is only a
future training-label absence gate; this request cannot launch that job.

The only acceptable independent final review is the exact structured `PASS`
at `INDEPENDENT_FINAL_R5_ENABLED_REQUEST_REVIEW.json`, issued by a different
Verifier after review of these exact bytes and local adversarial tests. The
only acceptable Supervisor assignment is the exact structured transfer-only
file at `SUPERVISOR_FINAL_R5_ENABLED_TRANSFER_ONLY_ASSIGNMENT.json`, binding
the actual final-review bytes and this request. Both must be actual regular,
owner-matched, one-link, no-follow fixed-path evidence. A missing,
malformed, duplicate-key, stale, mismatched, or non-PASS review/assignment
causes HOLD before marker, journal, or SSH. In-memory test fixtures are not
authority files. This runbook supplies **no dispatch command**; a separate
role-compliant decision would have to specify one after final review.

## Preserved transfer gates and limited threat model

The unchanged r5 transport rejects either `identity_before` or
`identity_after` full-probe free-inode count below 10,000 in preflight and
postflight, including the independent Verifier's 9,999 reproducer. The
unchanged r4 remote retains the fresh protected-asset, idle-process, Neuron,
target-absence, disk-byte, boot/mount/parent identity, STOP and science-label
checks. Its existing host lock is opened read-only, nonblocking and
no-follow, checked against exact named/opened inode and mode evidence, then
held by compatible exclusive `flock` through write and receipt. The private
`.cache` exception requires its pinned `0700` ancestry, exact inner `0775`
directory and `0664` one-link lock file, matching UID/GID/inodes, absent
POSIX ACL grants, and no relevant alias observed in the same mount namespace.
Unknown or changed evidence is HOLD. The last broad science-label scan is
immediately before target `O_EXCL`; distinct SSH readback and unconditional
full read-only postflight follow, even on a failed transfer. No overwrite,
deletion, rename, cleanup or retry is authorized.

This is a bounded ordinary-DAC and cooperative-flock argument, not proof
against root/same-UID actors, already-open descriptors, other mount
namespaces, or uncooperative writers racing the last scan. Fresh host
identity, ACL/mount/lock, protected-asset, quiescence, STOP/breaker, target,
label, disk/inode and path checks remain required at actual operation time.
The point-in-time trn21 read-only receipts do not themselves grant transfer.

Source staging is only an intermediate step. It authorizes no checkout,
CPU G2, monitor operation, native run, evaluation or submission. The same
frozen 446b launcher has a [separate independent native-package HOLD](../../verification/20260925-trn22-c5c-a0-r5-native-package-HOLD.md)
concerning optimized Python assertions; this wrapper does not invoke that
launcher and cannot cure its HOLD. trn21 native authority would require its
own accepted package, machine gates and role separation after source stage.

## Offline Implementer checks

`python3 -B -m unittest -v` passed 66/66 package tests, and the same suite
under `python3 -O -B -m unittest -q` passed 66/66. The archived independent
r5 inode-floor negatives passed 5/5 locally; that rerun is not a final
independent review of this candidate. The local suite covers the absent
review, review-only/assignment-absent,
mismatched review and assignment, duplicate-key authority, predecessor-PASS
drift, native-scope mutation, and preserved r5 preflight/readback/postflight
chain. These checks do not substitute for the separate final Verifier or a
fresh host receipt. Until both future authority files are independently
issued and exact, disposition is **DISPATCH HOLD**.
