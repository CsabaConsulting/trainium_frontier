# trn22 FIFO-lock r4 transfer-only final-r3 request — REVIEW HOLD

This is a **distinct exact request candidate**, not an Operator assignment or
host authorization. It repairs the independent final-r2 HOLD's reproduced
host-side gap: a persistent STOP appearing during the up-to-45-second final
label scan could reach target `O_EXCL`. The historical final-r2 HOLD has SHA
`b2c68151d8dc9ace6d08d8a0bc4d1828133bd2342eb240cbfe8ba25f5fbeed6b`.
The original disabled FIFO source `5740a0e0`, final-r1 and final-r2 packages,
and both final HOLD receipts are unchanged. The disabled source's static PASS
`248730a7`/`3c3d9c15` (receipt SHA
`67c3e430bd14623b4f11a305f9a445ad6d3276e6e63833a693d004aa94988f6d`)
is **historical only**. The new remote and transport bytes require a new
independent final review; no prior PASS authorizes them.

| Exact final-r3 file | SHA-256 |
| --- | --- |
| `TRANSFER_REQUEST.FIFO_FINAL_R3.json` | `92c0adc015da75e3e6825c8003b1dbaaf1d3a8997a7ccee833507fcbbce7648f` |
| `operator_transfer_lock_core_r3.py` | `a0d38c7e4c6e2ea93b369c469558ed156b0a9187b4040af8ca567eda3453914a` |
| `operator_transfer_lock_final_r3.py` | `ae8eb66fabafa33f8732a8f29f14a86e9d3e713abc57ea6316c0c0d2d7f24fae` |
| `stage_r4_transport_lock_final_r3.py` | `a12b8b202e18a8e809355c1fe0499ac7b2b68a94c5766ddb383f331f07b0c6c1` |
| `stage_r4_remote_lock_final_r3.py` | `fcf6f59e19e168a986de155338457d7b209a7bc2131674eea0841f7112ea0d5a` |
| `test_final_fifo_request_r3.py` | `eee0aa3111b7737dbfdb1767d27e5e3344ae71b0ba26fea016b5f369025f6c1e` |

The sole source is frozen commit
`446b657695f086f9b484fa601c9a3b61c2495912`, capsule 104,154,875 bytes,
SHA `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
The sole remote archive target is unchanged:
`/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz`.
The new assignment label is `trn22-c5c-a0-r4-fifo-lock-transfer-final-r3`.
Its unique local one-attempt marker and raw journal are under
`/home/poojith-devan/Desktop/Trainium-source-artifacts/`, named respectively
`trn22-c5c-a0-r4-fifo-final-r3-one-attempt.marker` and
`trn22-c5c-a0-r4-fifo-final-r3-one-attempt.jsonl`. Both are absent. Any
partial target, marker or journal is retained; no overwrite, cleanup or retry.

## Exact authority boundary

The request's `dispatch_authorized: true` is **intent only**. Before any
marker, journal or SSH, the final wrapper authenticates actual request,
historic PASS/HOLD, unchanged disabled source, new core, transport, remote,
and source-stage bytes. It requires two currently absent fixed-path files:
`INDEPENDENT_FIFO_FINAL_R3_REVIEW.json`, an exact structured PASS by a
separate Verifier binding the actual wrapper SHA, and
`SUPERVISOR_FIFO_FINAL_R3_ASSIGNMENT.json`, a separate one-shot assignment
binding the actual review SHA. This package creates neither file. Missing,
malformed or changed evidence stops before host contact. Even after those
files exist, a fresh Operator host preflight is required. Review and
assignment cover **transfer only**, never checkout, CPU G2, native run,
evaluation, scoring, submission or AWS lifecycle action.

## Bounded repair

The held existing `device0.lock` excludes the cooperating frozen launcher.
The remote writer retains all ancestry, private-cache 0700/no-ACL, lock,
host/boot/ext4, root-space/inode, protected checkpoint/cache, old-path,
STOP/process/Neuron, source-integrity and readback gates. After the broad
final label scan completes, the new remote code rechecks checkpoint holder,
Neuron holder/processes, cooperative lock identity, all fixed STOP paths and
the process inventory. A **second final STOP-path pass** follows the process
scan. The complete post-scan checks and target `O_EXCL` must fit inside the
unchanged one-second label-scan-to-open bound; slow or uncertain checks fail
closed without creating the target. The separate-session comparator requires
exact post-scan receipts and monotonic ordering/freshness. This closes the
demonstrated persistent STOP-during-scan interval; it cannot atomically
exclude same-UID/root or non-cooperating manual activity in the final tiny
race, so no absolute exclusion is claimed.

The local r2 STOP repairs are retained: recheck immediately before the
exclusive local marker after slow authority hashing, and after same-FD
capsule rehash/host-key setup immediately before the only write SSH spawn.
SSH still pins ED25519 host identity and isolated config, records raw
transport, verifies the archived bytes in a distinct SSH session, and runs a
fresh read-only postflight even on write failure. No private-key contents are
read into the receipt. The remote module's direct CLI remains DISPATCH HOLD;
the exact reviewed wrapper would embed only probe/write/verify functions.

Local-only evidence: **15/15** r3 tests pass in normal and optimized
(`python -O`) Python. Synthetic negative tests inject persistent host STOP
and Neuron activity during the final scan and make post-scan safety too slow;
all block before synthetic target creation. The comparator rejects missing,
stale, busy, or out-of-order post-scan receipts. Existing local STOP, one-shot,
source-FD and postflight tests still pass. No host, SSH, AWS, training or
transfer action occurred. Next: a different Verifier reviews these exact r3
bytes, then only a Supervisor may consider a one-shot transfer assignment.
Until then: **REVIEW HOLD; no dispatch**.
