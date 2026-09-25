# Independent trn21 final-r3 disabled transfer static review

**Verdict: PASS_DISABLED_STATIC_ONLY.** This accepts the frozen final-r3
implementation as a fail-closed, non-dispatchable transfer-only candidate. It
is **not** a final enablement review, Supervisor assignment, host preflight,
source transfer, checkout, native-training or evaluation authority. The r1 and
r2 enablement HOLDs remain in force. I did not contact trn21 or create a
marker, assignment, SSH process, transfer target, or host artifact.

## Exact reviewed bytes

Implementer commit `a7f85570ea0074838029ba0fbde9570d89df9a57` and
canonical cherry-pick `b94101fe7b15a5650b2bb97055dd63ac03f08580` have
identical Git blobs for all six final-r3 files:

| File | SHA-256 |
| --- | --- |
| `FINAL_R3_DISABLED_RUNBOOK.md` | `46f4b5c39e1ab3db200455f730bdb6bde9371ef4b2b8a186edf7cc40b0b9dcb2` |
| `TRANSFER_REQUEST.FINAL_R3_HOLD.json` | `3763e3c6c469745a92547d27e46aa01b05874d9b7911cd7ce8f6dcdfb49dd21e` |
| `operator_transfer_final_r3_once.py` | `6bf2aea104e9dbdefc69ee48a5bfe095e5b8f46c3d99bdd9ba9ba9aecf913a1a` |
| `stage_r4_transport_final_r3.py` | `0b676d2105010a6d5e6770dc507133ca958dc677ae21079cf673713a9894a083` |
| `stage_r4_remote_final_r3.py` | `3a738cff02b26824ea0be1fb50ee16b6cacce96941a9e2162506c7cf608ead32` |
| `test_operator_transfer_final_r3_once.py` | `d404c81111c38a298f897af1026b08b22dd96163400df60fde0723175969225d` |

The local 104,154,875-byte capsule rehashed to the request's
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
Frozen source commit `446b657695f086f9b484fa601c9a3b61c2495912`
contains launcher SHA-256 `9488efa918119439a33cbe68cb5638bd1af3b15f1f8f61ec790a216e2bc1e3f4`
and host-lock module SHA-256
`b34c368c12175615514e63daebc296ec09332ef801e176e0d19cc3edc045fed3`.
The prior r2 Supervisor HOLD SHA-256 is
`1f82c8d13c4abd8099a4a13a778628398088ba19dd6f8e7718ff0837b50e5639`;
the r1 independent HOLD is
`368c39aabd5437a0b2937773de3ad7aee4548e93a7255aa1a5012f792bbc49d0`.
The older disabled static PASS
`3803dc589c7ebf7ffd870a3d2eaddeecec77496fd24fc3a1cc92ccba5a482ab3`
is not r3 authority.

## Static disposition

- The exact, pinned r3 request requires both `dispatch_authorized: false` and
  `source_transfer_authorized: false`. The entry point rejects before reading
  future review/assignment files and before any fixed marker or SSH path.
  Changing a JSON flag fails the request hash and exact-structure checks.
- The writer opens only the existing host-wide launcher lock, read-only and
  no-follow, and takes nonblocking exclusive `flock`. The path and flock mode
  match the frozen 446b launcher. It rejects a missing, replaced, held or
  untrusted lock without creating or modifying one. Its lock descriptor stays
  open across the last activity and label observations, exclusive target
  creation, streaming, fsync, same-descriptor readback and receipt construction.
  Receipt emission and later SSH sessions are outside the lock lifetime.
- The transport binds preflight/writer lock identity and full host identity,
  then checks final self-only lock-holder and clean-label evidence and order.
  Distinct readback and full postflight have their own lock/label checks.
  The wrapper consumes a fixed exclusive marker before SSH, keeps failed
  targets and evidence, has no same-path retry, and attempts a distinct full
  read-only postflight on a failed transfer phase.
- The accepted e65 checkout's pinned `.git/HEAD` remains a protected source
  requirement. The other old checkouts are inventoried at top level, without
  descending into a possible file-backed `.git`.

**Enablement limits:** trn21's actual `.cache`, `trainium_freshstart` and
`device0.lock` ownership/modes have not been established by a host observation.
The r3 trust rule rejects group/world-writable lock parents or file and
requires an owner-readable/writable regular file. The frozen launcher can
create usable paths without imposing those stricter modes; r3 may therefore
HOLD safely on the live host. A separate read-only host check and independent
review are required before any enabled request. The lock excludes the
cooperative frozen launcher, not arbitrary uncooperative activity after its
last observation; this review makes no atomic STOP/process/Neuron-continuity
claim for a future dispatch.

## Offline checks

- `python3 -B -m unittest -v` in the transfer package: **41/41 PASS** across
  the historical and r3 package modules.
- `python3 -B -m unittest -v independent_final_r3_disabled_static_tests`:
  **6/6 PASS**. The separate independent test file SHA-256 is
  `8f1b9297d1674831a7209aa8198d8382780a5ab5580edf70b3173b20b30994a7`.
  It checks disabled entry-point order, local existing-lock behavior and
  unsafe-mode rejection, no-follow old-Git classification, receipt drift
  rejection, and failed-phase postflight sequencing. Fixtures are local and
  disposable; no host or SSH was contacted.

No frozen package code, source model, evaluator, protected asset or controller
state was changed by this review.
