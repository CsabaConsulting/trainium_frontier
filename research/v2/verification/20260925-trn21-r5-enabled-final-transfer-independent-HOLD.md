# Independent trn21 r5 enabled transfer final review — HOLD

**Verdict: HOLD. Do not dispatch the transfer.** The final remote writer can
create the new capsule target after a persistent STOP appears during its last
science-label scan. This review grants no source transfer, checkout, CPU G2,
native training, evaluation, cleanup, AWS action, breaker reset, or submission.
I did not edit the candidate, contact trn21 or AWS, create a transfer marker or
journal, run a transfer, or issue a fixed-path final PASS or Supervisor assignment.

## Exact subject and authority boundary

The reviewed branch `codex/20260925-trn21-r5-final-transfer-enabled-candidate`
is clean at `fb3f52af4f522146e41b45314f2a9b356107dece`, directly atop
canonical disabled-r5 static-PASS commit
`73be2973dda3a0c1ff2d62a37612aa95a08f0027`. The origin disabled-r5
independent review at `c6d45af0a6f8f74cdbc0f432411b691b20ff80a0` has the
same SHA-256 `be212cbbd50ac525049602c87b584fb2e8d21cca6c3e03ed29fcc587e67a58b4`
as the canonical predecessor. The candidate diff adds exactly four files;
`git diff --check 73be2973 fb3f52af` passes. The disabled r5 files and all
older r1/r2 HOLD, r3 disabled-only, and r4 HOLD subjects are unchanged.

| New candidate file under `research/v2/authorizations/trn21-c5c-a0-r4-transfer-only/` | SHA-256 |
| --- | --- |
| `FINAL_R5_ENABLED_TRANSFER_ONLY_PENDING_REVIEW_RUNBOOK.md` | `833d7725990d8ff0ae17497aa4bb352efd43aee0ec0acf769a49e3ddeeae4de3` |
| `TRANSFER_REQUEST.FINAL_R5_ENABLED_PENDING_REVIEW.json` | `b611a7546d1c73d338dea14d083432b16fbea5616bd3002c5dd57850d1f0710c` |
| `operator_transfer_final_r5_enabled_once.py` | `327e4d687d28a1412fd769b80ba47463e13f5c32454dc51d906b59fcf23b1590` |
| `test_operator_transfer_final_r5_enabled_once.py` | `7990ea88376f29b7f25ebe86a2beb294d628c2f5d41633d2e41e2f96e453054b` |

The actual fixed-path `INDEPENDENT_FINAL_R5_ENABLED_REQUEST_REVIEW.json` and
`SUPERVISOR_FINAL_R5_ENABLED_TRANSFER_ONLY_ASSIGNMENT.json` are absent in the
reviewed candidate. Its distinct r5-enabled marker and shared transfer journal
are absent locally. The request's two true intent flags do not bypass the
wrapper's requirement for both exact future authority files. No final PASS
JSON was published by this Verifier.

## Blocking STOP-during-label-scan counterexample

The enabled request reuses byte-identical
`stage_r4_remote_final_r4.py`, SHA-256
`ad30d93b464175c6df1550654076bc868a8bd8cd652fedd08bd5bfe91f055e2b`.
In `_final_creation_gate`, `_activity_gate_snapshot()` finishes its last
STOP/process/Neuron/target check before `_label_scan()`. The latter runs a
host-wide `find` with a timeout of up to 45 seconds. After that scan returns,
the writer records the label result and returns directly to `os.open(...,
O_EXCL, ...)` for the target. It does not recheck STOP between the label scan
and target creation. Holding the launcher lock does not prevent a STOP file
from being created.

The independent local test
`research/v2/verification/test_trn21_r5_enabled_final_stop_independent.py`
(SHA-256 `9a391ee9798aee778815be1c7dd6966958d9731927d741eded2f3a3b871f6135`)
uses the archived synthetic writer harness and a disposable `/dev/shm` root.
It creates a real persistent local STOP fixture during the final label scan,
while the clean label result remains empty. The fixture makes any subsequent
STOP lookup report present. Nevertheless the remote writer creates the target
and returns `HASH_VERIFIED_IN_THIS_SESSION_ONLY`; its final-gate receipt still
reports the earlier STOP samples as clear. The test passes 1/1 under both
normal and optimized Python. It contacts no host and streams no real capsule.
The wrapper's later full read-only postflight might detect the STOP and mark
the attempt HOLD, but that cannot undo creation of a target while STOP was
already present. The one-shot marker would be consumed and any target retained.

## Other offline checks and limits

The supplied package suite passed **66/66** normally and **66/66** with `-O`.
Archived independent r5 inode-floor and r4 private-lock checks passed
**13/13** normally and **13/13** with `-O`. Seven additional local checks of
the enabled authority chain, actual missing files, unsafe fixed-path evidence,
first and late local breaker checks, write-spawn breaker check, final label
receipt, and terminal inode floor passed. None exercises a real SSH process.
The pinned local 104,154,875-byte capsule rehashed to
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`
and remained regular, owner-1000, mode `0600`, one link. The local known-hosts
file rehashed to `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`;
no private-key bytes were read.

The reused r5 transport requires both full-probe inode samples at least
10,000; the unchanged r4 writer has the bounded private `.cache` lock
exception with exact mode/inode/link/ACL and same-namespace mount checks,
read-only no-follow nonblocking open, exclusive `flock`, full live prewrite
probe, distinct readback, and unconditional postflight. These checks did not
remove the STOP gap. The private-lock inference remains bounded to ordinary
DAC and cooperative locking in the observed namespace; same-UID or privileged
actors, already-open descriptors, other mount namespaces, and uncooperative
writers are outside that proof. No host-state claim is made from this offline
review.

A separate Implementer must close the final-scan STOP gap and bind the new
after-scan evidence in the writer receipt and transport validator, then update
all affected hashes and exact request bindings. A different independent
Verifier must review those new bytes. Even a later transfer-only PASS would
still require fresh full trn21 host preflight, a Supervisor one-shot
transfer-only assignment, and independent review of the retrieved result.
Native launch remains under its own HOLD and later gates.
