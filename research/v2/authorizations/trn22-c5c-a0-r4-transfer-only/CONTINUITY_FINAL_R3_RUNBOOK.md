# trn22 r4 continuity-r3 exact transfer request — pending final review

This is a **new request**, not a transfer dispatch. It starts from canonical
control-plane `fab8860a1efe67351d103d41013656db130e9c38` and the
independently reviewed disabled r3 repair. The actual
`INDEPENDENT_CONTINUITY_R3_STATIC_REVIEW.md` has SHA-256
`f71e81cb1ac079593ccc5f0e2b06a3436ce0a1c9d45d6850ac46487a3eed3fed`;
its STATIC PASS covers only the disabled repair, not this final wrapper.
Historical final-r2 HOLD and prior PASS files remain unchanged and confer no
authority on this request.

| Exact artifact | SHA-256 |
| --- | --- |
| `TRANSFER_REQUEST.CONTINUITY_FINAL_R3.json` | `a2b1ad44023f32c010c8d169e25986830fa60a26d0162d281a4c0154b447e7a0` |
| `operator_transfer_continuity_final_r3.py` | `c44c4888a0babcc492a75c5c719d1c53def4e72d9f40182dc3dac7185c01bcf2` |
| Reviewed r3 `stage_r4_remote.py` | `e93efd99c0a58aa32537d945ddc8fc29f9d7ce0dabb89ed93526da30421f4086` |
| Reviewed r3 `stage_r4_transport.py` | `1fa3afb41dfbf8288e53c4aa43dd16b348013416e708d8432e97364b31556b51` |
| Frozen outside-Git 446b capsule, 104,154,875 bytes | `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a` |

The sole target, marker and journal are the fixed final-r1 paths in the
request. This request authorizes no alternate target, cleanup, overwrite,
retry, checkout, G2, native run, evaluation, monitor operation or submission.

The request's `dispatch_authorized: true` records **intent only**. The new
wrapper fails before marker, journal or SSH unless a separate Verifier creates
the exact fixed-path `INDEPENDENT_CONTINUITY_FINAL_R3_REVIEW.json` with a PASS
binding the actual request and wrapper SHA-256, and the Supervisor separately
creates `SUPERVISOR_CONTINUITY_TRANSFER_R3_ASSIGNMENT.json` binding the actual
review bytes and one attempt. Neither file exists in this package. A role
label alone is not provenance; the Supervisor must verify the review's true
independent origin. The wrapper rechecks both files before the marker and
again before write, and checks local STOP/breakers at both points.

After a future narrow assignment, the authorized sequence would be one fresh
read-only preflight, one preflight-bound exclusive write, a distinct SSH
readback, and a final read-only postflight even after failure. The repaired
remote writer performs a second full Neuron/lock/label/STOP/process gate at
the creation boundary. Partial targets, marker and raw journal are retained;
there is no deletion or retry path.

All work so far was local/offline. The new final-request tests plus the r3
repair and independent static tests pass **21/21**; the two preserved
independent tests that exposed the r2 blockers also pass, **23/23** total.
Tests use temporary files
and mocked SSH; they are not a final review or a current host feasibility
finding. The next step belongs to a **different independent Verifier**, who
must review these new actual bytes and issue PASS or HOLD. Only afterward
may the Supervisor consider a distinct one-shot transfer-only assignment and
fresh host preflight. No host contact is authorized by this runbook.
