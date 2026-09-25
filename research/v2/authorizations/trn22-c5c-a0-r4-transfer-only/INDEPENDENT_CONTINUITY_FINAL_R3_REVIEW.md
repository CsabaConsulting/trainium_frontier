# Independent final-r3 transfer-only review — PASS for request, dispatch HOLD

Reviewed 2026-09-25 UTC by a Verifier separate from the Implementer and
Operator. The subject is the four-file enabled-request commit
`fd8b0c74ed2c90905411b225471840c608cad17b` on
`codex/20260925-trn22-r4-final-r3-enabled-request`, based on canonical
disabled-repair/static-review commit `fab8860a1efe67351d103d41013656db130e9c38`.
I read the repository contract in order and inspected the actual request,
wrapper, runbook, reviewed disabled r3 bytes, earlier final-r2 HOLD, source
stage and independent static PASS. The candidate commit changes only its
four named final-request files; its diff whitespace check passed.

**Verdict: PASS for the exact final-r3 transfer-only request and wrapper.**
The companion actual fixed-path structured PASS is
`INDEPENDENT_CONTINUITY_FINAL_R3_REVIEW.json`, SHA-256
`1069bba48756a5977498fb214094413add9a67c8d06633c9171d45f10fd5210f`,
owner UID 1000, one link, mode `0644`. It matches the wrapper's complete
expected independent-review object and binds the actual request and wrapper
bytes. This is **not** dispatch by itself: the separate exact fixed-path
`SUPERVISOR_CONTINUITY_TRANSFER_R3_ASSIGNMENT.json` does not exist. An actual
`authorize_final()` check with this PASS in place still raises `SUPERVISOR R3
ASSIGNMENT HOLD` before marker, journal or SSH. The fixed marker and journal
also remain absent. The Supervisor must establish the review's independent
origin and separately issue a one-attempt assignment before any Operator use.

## Exact reviewed bindings

| Artifact | SHA-256 |
| --- | --- |
| `CONTINUITY_FINAL_R3_RUNBOOK.md` | `9fbb503d2a77fbf6d6eab14504695eae2978a973b0b4e9c575171e4844496e48` |
| `TRANSFER_REQUEST.CONTINUITY_FINAL_R3.json` | `a2b1ad44023f32c010c8d169e25986830fa60a26d0162d281a4c0154b447e7a0` |
| `operator_transfer_continuity_final_r3.py` | `c44c4888a0babcc492a75c5c719d1c53def4e72d9f40182dc3dac7185c01bcf2` |
| Reviewed r3 STATIC PASS | `f71e81cb1ac079593ccc5f0e2b06a3436ce0a1c9d45d6850ac46487a3eed3fed` |
| `stage_r4_remote.py` / `stage_r4_transport.py` | `e93efd99c0a58aa32537d945ddc8fc29f9d7ce0dabb89ed93526da30421f4086` / `1fa3afb41dfbf8288e53c4aa43dd16b348013416e708d8432e97364b31556b51` |
| Unchanged disabled core / marker helper | `7590e275f94f79ffac9098bfd3c0345d4c9cb5ce10a26624065b0e04d6189736` / `b3ad1290f5e9cc6655b57380b269b9c2e4082391ea27d4f81545388b892beb1d` |
| New supplied final-r3 tests / independent final-r3 tests | `d28b3e14a4f933543177fd933a089db8a2f1818a7135b53358769f90145cbe40` / `8214fac9b8d1f81c2583f5e18707e6bc8be72a9465d51410490aa1ac2970d5c2` |

The r3 Implementer code commit is `76485b9623d84351433cc3d2cd350f830c624d1e`;
its canonical byte-identical cherry-pick is
`e21135cc84fa941784e6a52fa142dadd0973de3f`. The earlier final-r2 HOLD
remains unchanged at `873bf131898e014adbb79a9cb6dd5cefd5f653897f8c80ae63b13e296f1e38f1`
and remains the verdict for that old request. Its two concrete defects were
retested against this exact r3 code. The frozen outside-Git
capsule rehashed to `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`
at 104,154,875 bytes, regular mode `0600`, one link. Pinned `known_hosts`
rehashes to `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`.
The private key was checked by metadata only (regular, owner UID 1000, one
link, `0600`); its contents were never read.

## Independent adversarial result

The targeted local suite passed **32/32**: supplied final-r3 tests **10/10**,
repaired disabled-r3 tests **7/7**, prior independent r3 tests **4/4**, new
independent final-r3 tests **7/7**, and four selected preserved independent
final-r2 checks **4/4**. The last group includes both tests that exposed the
late-Neuron and file-backed `.git` blockers, plus fixed marker/journal and
literal pinned SSH identity checks. The new tests did not simply trust the
Implementer's 23 passing cases: they used actual temporary review and
assignment files to reject missing, role-label-only, mutated and unbound
authority before marker/journal/SSH; exercised the actual private,
descriptor-relative `O_EXCL`/`fsync` marker and journal with same-path replay
refusal; and checked ordered preflight/write/distinct readback/final postflight
and raw SSH transport journaling with SSH fully mocked. A synthetic failed
write kept its marker/journal and still ran full read-only postflight.

Direct local remote-writer fixtures independently changed boot, mount and
parent identities during the second identity check, or introduced STOP,
science label and shared-lock activity after the first live check. Every
case stopped before the exact target's `O_EXCL` creation. A partial local
target was retained and a same-path retry refused. The original independent
late-Neuron test now blocks creation; a file-backed old `.git` no longer
aborts inventory, while the r3 static review's full 41-path fixture and
no-follow negative tests remain passing. The remote/transport source review
confirmed preflight-to-write boot/mount/ancestor/source binding, the second
complete checkpoint/lock/Neuron/label/STOP/process gate, protected old paths,
separate persistence readback, and postflight comparison. These are
point-in-time gates, not a guarantee against activity starting after the
last check.

No SSH, trn22/AWS contact, transfer, host write, checkout, G2, native run,
evaluation, cleanup, breaker reset, submission, or Supervisor assignment
occurred in this review. The archived 04:18–04:21 UTC read-only host snapshot
is historical evidence, **not** the fresh preflight required for an assigned
attempt. This PASS covers only one exact source-capsule transfer to the fixed
new ext4 archive path; independent transfer-result review and separate future
authority remain mandatory before any checkout or later stage.
