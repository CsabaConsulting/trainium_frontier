# Independent trn21 r4 transfer final enablement review — HOLD

Reviewed 2026-09-25 04:35 UTC by a Verifier separate from the Implementer.
Implementer source commit: `2cda547669059f49b173a08e273bff7f81bd734c`.
Canonical cherry-pick: `0a5fc72cb77161f902feeee2620d69cb93740c05`.
The six candidate package paths have identical bytes at those commits.

**Verdict: HOLD. Do not dispatch.** I did not create the fixed
`INDEPENDENT_FINAL_ENABLEMENT_REVIEW.json` PASS, a Supervisor assignment,
the one-attempt marker, or the raw journal. I did not contact trn21 or AWS,
transfer a capsule, check out source on a host, run CPU G2/native training,
evaluate, clean up, or submit. The current wrapper therefore still rejects
before its marker or SSH when the actual final PASS file is absent.

## Blocking late-label race

The writer's full live probe scans for the r4 science label. After that
probe, `stage_r4_remote_final.py` runs `_immediate_activity_gate()` twice,
including immediately before its descriptor-relative `O_EXCL` target create.
Neither late gate calls `_label_scan()`. A science-label path introduced after
the full writer probe is therefore invisible to both gates. The final
transport validator also accepts `final_creation_gate` without any late
science-label scan result, so the receipt cannot prove that gate.

The independent offline regression introduced a real file named with the r4
label after a clean full writer-probe fixture. Both late gates ran. A direct
local label scan found one match, yet the synthetic target was created and
the writer returned `HASH_VERIFIED_IN_THIS_SESSION_ONLY`. The target was
inside a disposable `/dev/shm` directory; no host path was touched. A second
negative test showed that `compare_preflight_to_write()` accepted a final
creation-gate receipt with no label-scan field.

The exact failing test file is
`independent_final_enablement_negative_tests.py`, SHA-256
`ec73e4a510d5b4010d92804b98285a128f967a01623063395e3c00f26cff59d4`.
`python3 -B -m unittest independent_final_enablement_negative_tests.py`
ran three tests: one PASS for actual fixed-path review and assignment bytes,
two intentional FAILs:

- `test_late_science_label_must_block_before_exclusive_target_create`:
  target created with the late label present.
- `test_final_receipt_must_prove_late_label_scan`:
  no error on missing final-gate label evidence.

The passing authority test used owner-checked, no-follow temporary files to
exercise the actual read and hash chain. Missing review or assignment,
changed review bytes, changed assignment review hash, and a review symlink
all stopped before a marker or SSH. A matching review plus assignment was
accepted only by `authorize_final()`; no dispatch path was entered.

## Other static evidence inspected

| Exact artifact | SHA-256 |
| --- | --- |
| Final transfer request | `04a6b215ac796927677a29aebdf414ffe0f361495cca8a45d2dec070bc60f136` |
| Final operator wrapper | `4c02dd1891009effe00d22b01966943ae85041a8ccbc0006792f3b76859feec1` |
| Final SSH transport/validators | `8c1e362fd586a00137016d7e0fad6f6ed758124cf7f66716a143895c91b512ba` |
| Final remote probe/writer/reader | `9b73cd84a46af46e9ce7e8729e328878a7b29c16ac1556e9adeb93785fd9b4b8` |
| Earlier independent disabled static PASS, not final authority | `3803dc589c7ebf7ffd870a3d2eaddeecec77496fd24fc3a1cc92ccba5a482ab3` |

I read the repository contract, current handoff, experiment protocol and
decision log in order, then the final runbook/request/wrapper/transport/remote,
disabled package and independent static review, source-stage HOLD plan, and
local capsule review. The pinned local capsule was a regular owner-1000,
single-link mode-0600 file of 104,154,875 bytes with SHA-256
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
The local no-SSH host-identity check matched the pinned ED25519 known-hosts
SHA-256 `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`
and fingerprint `SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc`;
no private-key bytes were read. The source and target constants, protected
e65/checkpoint/cache/old-path pins, boot/mount/source/STOP continuity,
exclusive fsynced marker, raw journal, distinct readback and unconditional
full postflight were inspected. The existing final test suite passed 11/11;
the disabled transfer and independent static suites passed 24/24.
`git diff --check 2cda5476^ 2cda5476` passed, and the six package paths
matched the canonical cherry-pick byte for byte.

The late-label race is sufficient to fail this final enablement gate.
An Implementer must repair the late writer gate and make the cross-session
receipt validator require its label evidence, then update every affected
hash/request binding. A separate Verifier must review those new exact bytes.
Fresh read-only host feasibility and a distinct Supervisor one-attempt
transfer-only assignment remain later prerequisites; this HOLD grants none.
