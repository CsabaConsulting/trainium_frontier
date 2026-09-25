# Independent trn22 r4 continuity-repair review — static PASS, dispatch HOLD

Reviewed 2026-09-25 UTC by a Verifier separate from the Implementer. The
reviewed Implementer commit is `d682a90b1e75bdac128938feda884484ea87a8d5`
on `codex/20260925-trn22-r4-continuity-repair`. I inspected its full diff and
actual local bytes, including `CONTINUITY_REPAIR_HOLD.md`, the two disabled
requests, the final/core wrappers, SSH transport, remote writer and supplied
tests. The earlier independent static PASS predates these changed bytes and is
not authority. The b80c69c4 final-candidate HOLD identified the continuity
and post-probe live-safety gaps; this review is a new, narrower review of the
repaired **disabled** package.

**Verdict: PASS for offline static repair correctness only. DISPATCH HOLD.**
The exact request currently says `dispatch_authorized: false`, and
`authorize_final()` rejects it before the one-shot marker, journal or SSH.
This is not an `INDEPENDENT_FINAL_ENABLEMENT_REVIEW.json`, a Supervisor
assignment, a host-readiness finding, or permission for a transfer. Neither
fixed-path authority JSON exists in the reviewed tree. I did not SSH, touch
the host, dispatch, extract, checkout, run G2/native/evaluation, clean up,
submit, or inspect private-key contents.

## Exact reviewed bindings

| Artifact | SHA-256 |
| --- | --- |
| Disabled final request (`TRANSFER_REQUEST.ENABLEMENT_CANDIDATE.json`; historical filename) | `833737bbe76a7b7acc287c247616527b199b5f515a46fcf8de0190595600ee57` |
| Disabled core request (`TRANSFER_REQUEST.HOLD.json`) | `2b7249002ec301962153efdfc3b003454dff9c2ba6fe80a09b780e10adb8ab0c` |
| Final wrapper | `b3ad1290f5e9cc6655b57380b269b9c2e4082391ea27d4f81545388b892beb1d` |
| Core wrapper | `7590e275f94f79ffac9098bfd3c0345d4c9cb5ce10a26624065b0e04d6189736` |
| SSH transport | `ea14fd19f6992400ad56e59c8dcbec620c784f616c4ff60f6d0568aa721a4ec0` |
| Remote probe/writer/reader | `023e578693c3fb826b81cfc645190d79e409757dba0bd8df29adad6e3f90a69d` |
| Implementer repair/HOLD note | `1a42750f7a0ed0a1ae2b3a9f38cdd8b5543e3fb8179e23bb0669e5374ed0afd7` |
| Independent adversarial regressions accompanying this receipt | `8709c2ea1b08cf8cadb6277cb1d2a178316e43f7106792c45f9153b6f4d581f3` |

The source stage remains frozen at commit
`446b657695f086f9b484fa601c9a3b61c2495912`. The sole outside-Git
capsule is 104,154,875 bytes with SHA-256
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`;
the only proposed destination is
`/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz`.
The existing source-stage contract and independent review retain SHA-256
`79878b16f459d4c12b32cb0ebb85d205dad86d584ae09654e0e8161920deab69`
and `d49e8c03567210283074c44a2f2937594ada39f43e399f97d176c373c1a32c1e`.
The local capsule checked as a private, regular, single-link mode-0600 file;
its bytes/hash matched the frozen source. Pinned `known_hosts` SHA-256 is
`e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`,
with the one expected ED25519 fingerprint
`SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U`; the private key was
checked by metadata only. Strict known-hosts SSH options remain fixed and
there was no network connection during review.

## Independent checks and result

The remote writer now requires a canonical, fresh, digest-bound preflight
token on its sole write operation. Its stable identity pin includes hostname,
UID, boot ID, mount namespace, ext4 mount, kernel block-device alias proof and
ancestor inode identities. The local comparator joins preflight to write,
write to a distinct read-only verify process, and preflight to the final
read-only postflight. A synthetic pre/post boot A with write/verify boot B is
rejected rather than becoming a false persistence PASS.

The writer checks live STOP paths and matched/unreadable processes after the
probe, alongside checkpoint fuser, lock, Neuron and label gates. It checks
identity, STOP and processes again just before the descriptor-relative
`O_CREAT|O_EXCL|O_NOFOLLOW` create. A STOP introduced during the intervening
live checks is caught at that last gate. The only archive create is the exact
new ext4 path, mode 0600; existing or partial targets are retained and a
same-path retry fails. The fixed marker and raw journal are independently
exclusive one-attempt gates. A failed write still attempts a distinct final
read-only postflight, records HOLD and closes the journal. The reviewed code
has no extraction, checkout, G2, native training, evaluation or cleanup path.

The maintained supplied suites passed **29/29** tests. My updated independent
adversarial suite passed **7/7** tests: boot A/write B/post A, changed remote
boot, tampered binding/receipt evidence, host STOP or busy process after
probe, a STOP appearing during live checks, exact target exclusivity and
partial preservation, disabled dispatch, and failure postflight. Combined
command: `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v
research/v2/authorizations/trn22-c5c-a0-r4-transfer-only/test_stage_r4_transfer.py
research/v2/authorizations/trn22-c5c-a0-r4-transfer-only/test_operator_transfer_final_once.py
research/v2/authorizations/trn22-c5c-a0-r4-transfer-only/independent_final_negative_tests.py`
— **36/36 PASS**. `git diff --check d682a90b^ d682a90b` also passed.
Historical independent negative tests that intentionally assert the old bug,
and older API-specific static fixtures, are not current pass-suite evidence.

This synthetic/static result does not prove current host quiescence, free
space, old-path inventory, mount alias state, or future SSH success. Any
successor enablement request must bind newly reviewed exact bytes; an
independent final enablement review and distinct Supervisor one-shot
transfer-only assignment remain separate gates. Until then, **no dispatch**.
