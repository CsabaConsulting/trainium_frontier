# Independent trn22 FIFO transfer final-r1 review — HOLD

Reviewed 2026-09-25 UTC by a Verifier separate from the Implementer. Exact
pushed subject: `06ecfd8bd752f44cfb3f45e0221a166662bdffb1` on
`origin/codex/20260925-trn22-r4-transfer-lock-final-request-r1`.
The repository contract was read in its required order; those five instruction
files are unchanged from the preceding disabled-static review. This review
used only local source, local artifacts and intercepted synthetic SSH spawns.
The Implementer checkout remained clean. No host or AWS contact occurred.

**Verdict: FINAL ENABLEMENT HOLD for these exact bytes.** Do not create
`INDEPENDENT_FIFO_FINAL_R1_REVIEW.json`, issue a Supervisor assignment, create
the fixed marker/journal, or dispatch this request. Its
`dispatch_authorized: true` is request intent only. The disabled 5740 source
retains its separate static-only PASS; this HOLD concerns the new enabled
wrapper's local STOP continuity before host action.

| Exact file | SHA-256 |
| --- | --- |
| `RUNBOOK.md` | `b59284f85065401df1bbefe5b7956cc3b99da4d147745d32bcc23f7de79c7c81` |
| `TRANSFER_REQUEST.FIFO_FINAL_R1.json` | `647cbda44586f4da0514630bffd0868cbed3207875780093b29f538cea4001e3` |
| `operator_transfer_lock_core.py` | `6c3580db82f379f98108beebf5009952040c81aed3db75a0272c415fa92f849f` |
| `operator_transfer_lock_final_once.py` | `298a33d802d7583a473a3901c942cb2184c49e00d319b1738d5be29dcf717224` |
| `stage_r4_transport_lock_final.py` | `fde41bbe3ab5367840d1b225316b2d961a562a2dee4e6ca9509120e83b060d84` |
| Supplied `test_final_fifo_request.py` | `49b50dd22fca6d0636295a1f6c15ebb042146122aa1462ccf15dfc24a5ab68c9` |
| Verifier `test_trn22_fifo_final_r1_independent.py` | `6cfaa74050853874fa6fd080fa5bff2d349b05b1bb12c7ce1d0322f43f33449b` |

## Blocking timing gap

`operator_transfer_lock_final_once.py:253` checks local STOP/breakers, then
repeats `authorize_final()` at line 254. That reauthorization hashes the
request, review, subject files and source-stage evidence and verifies local
SSH identity, including subprocess calls with 20-second timeouts. A persistent
local STOP created during this interval is not checked before the exclusive
one-shot marker call at line 256. An independent synthetic STOP injected during
the second authorization reached the intercepted marker call; no real marker
was written. Thus a stopped request could consume its one attempt and start
read-only host preflight.

More seriously, line 273 is the **last** local STOP/breaker check before the
write. `core._run_ssh()` then builds the SSH command (including local host-key
verification) at `operator_transfer_lock_core.py:246`, rehashes the
104,154,875-byte source FD at lines 249–279, and reaches `subprocess.Popen`
at line 287 without another local STOP check. A synthetic persistent STOP
injected during that preparation still reached an intercepted `Popen` for the
write operation. The fake spawn raised before network contact; the wrapper
recorded failure and ran its read-only postflight. With a real spawn, the
remote writer checks **remote** STOP paths but cannot see this newly created
**local** STOP, so it could create the target despite the request's local
STOP-before-write gate. The postflight cannot undo that creation.

This is an avoidable interval containing host-key subprocesses and a full
capsule hash, distinct from the inevitable small race after a final check.
The repair must recheck local STOP/breakers after the slow reauthorization and
before marker creation, and after source-FD rehash immediately before the
write SSH spawn. Preserve the unconditional read-only postflight on any
write failure. A different Verifier must review repaired exact bytes; this
HOLD cannot be cleared by a point-in-time host preflight.

## Other checks and limits

The subject's request/core/transport/wrapper hashes match its runbook and
runtime pins. The source-stage contract/review hashes are
`79878b16f459d4c12b32cb0ebb85d205dad86d584ae09654e0e8161920deab69`
and `d49e8c03567210283074c44a2f2937594ada39f43e399f97d176c373c1a32c1e`.
The disabled-static review SHA is
`67c3e430bd14623b4f11a305f9a445ad6d3276e6e63833a693d004aa94988f6d`,
the unchanged remote SHA is
`cdf06f700722e144dcb304e517a7e000fe0c6b2bcb210e4d2bb38f92757d5e27`,
and the local capsule rehash is
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
The canonical disabled subject and static-review commits exist; the wrapper
authenticates their actual file bytes. The new transport embeds that byte-exact
remote under a non-main namespace, accepts only probe/write/verify, and pins
strict SSH host-key options. No `assert` gates control this final wrapper;
`python -O` still stopped on the absent fixed-path final review before marker
or SSH. The frozen 446b launcher's native gates remain outside transfer
authority.

The supplied **37/37** tests passed (8 new; 14 disabled successor; 4 prior
independent; 7 archived r3 final; 4 archived r3 static), as did **5/5** new
Verifier tests. Those new tests also reject tampered review/assignment fields,
boot/mount/private-ACL/disk, lock/label/STOP/process/Neuron/timing receipts,
and check optimized-Python refusal. `git diff --check` passed. The final
review, assignment, marker and journal were all absent when checked. No
transfer, checkout, G2, native run, evaluation, cleanup or submission occurred.
