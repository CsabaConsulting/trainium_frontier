# Independent final review: trn21 c5c A0 stage-r2 checkout

**Verdict: PASS_EXACT_CHECKOUT_REQUEST_ONLY** for the enabled Implementer
successor at `5059c4385afabc33a4b22380b7ad990038c87d4c`. The actual
machine-readable Verifier receipt is
`research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/INDEPENDENT_FINAL_REQUEST_REVIEW.PASS.json`,
SHA-256 `acfc9036cf0864a2abf019e961e7680de976aba99b5ffcd656d2d4acf39f04eb`.
This accepts the exact **checkout-only request**, not host readiness or a
checkout result. There is still no Supervisor one-attempt assignment, live
preflight, host checkout, CPU G2, native control, eval or new BPB result.

## Exact enabled identity

| File under `research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/` | SHA-256 |
|---|---|
| `GATE_CONTRACT.FINAL_PENDING_REVIEW.json` | `3b0a5f66ef1e5fc71abe3da0591d849128679d3e7a80cb48b30b49eb6b783b30` |
| `RUNBOOK_FINAL_PENDING_REVIEW.md` | `ac685217be7ac1205a2816d73b988b1a2cfea73f697d8f4bd31bfba4fe6ff461` |
| `CHECKOUT_REQUEST.FINAL_PENDING_REVIEW.json` | `be97460e0fa4cd20c00d29c2eefbe0923e511b9327998629f7962fee07c29010` |
| `INVOCATION_BINDING.FINAL_PENDING_REVIEW.json` | `e023ce60824ed988e93c44bc186b1969a2d026a63d5d952f0e60fa1c8a746906` |
| `checkout_invocation_final.py` | `6d15b15b85abff9fa3ea2fa6c368f266833517dd066f8cc13f80831570c1ccd8` |
| `test_checkout_invocation_final.py` | `9e6482854e9e65cca9add8865c6495344e547291022b174be46a3372750b357f` |

The six files are additive. The earlier disabled request, binding, wrapper,
adapter and tests remain byte-for-byte unchanged. The enabled request differs
from the disabled draft only in its status, `checkout_authorized=true` and
the new wrapper SHA. The enabled binding differs in status,
`dispatch_authorized=true`, final request/wrapper identities and the recomputed
stage argv/stdin hashes. The final wrapper additionally requires literal
source, target, capsule, train and prepare pins and a Supervisor assignment
that names those, all code digests, both argv digests, stdin digest, fixed
marker and the actual final review digest, while forbidding G2/native/eval
authority. The frozen remote adapter remains
`e3faee8fe1274015b500134bcb767b041e37e6b7eef2e375f23d93a3ba8de3eb`;
the extraction source remains
`e66afb7a1e3368ed7854eaa679e2bad9a6b8529e3b793e0f8fff2c69c496785d`.

I independently rehashed e65 `train.py` at
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and unchanged `prepare.py` at
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The only source capsule remains the 104,005,787-byte transfer to
`/home/ubuntu/trainium-c5c-e65-one-commit-20260925.trn21-stage-r2.tar.gz`,
SHA-256 `32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264`,
and the sole checkout target is
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`.
The builder independently rebuilt the exact audit argv SHA-256
`cc1997adb5b815a739a12cbb09319f3869e08f7d8318899fea9496bee40f66a8`,
stage argv SHA-256
`48a58ffb289ffc82b057eb30ba66702433bc9e15d19cca4abd77acd9d2e1bddc`,
and 69,981-byte stage stdin SHA-256
`96058bc2d2ce90de150cac14fe3d5553f303abe5ead5d867c09d42f247f7428f`.

## Independent checks and limits

I reran **11/11 final-wrapper**, **10/10 disabled-wrapper**, **8/8 prior
independent repair**, and **9/9 frozen extraction** tests. My new independent
suite, `research/v2/verification/test_trn21_checkout_final_review.py` at
SHA-256 `1fbff42a579288ea8a933b10a3bcd74e0e7aed6cf5f705d920e3e1f7f3706ecb`,
passed **7/7**, for **45/45** offline tests overall. It specifically checked:

- Actual enabled builder pins, exact contract hashes and no review-hash cycle;
  direct wrapper execution and a missing assignment refuse without host contact.
- Missing, nonhex, wrong-path, wrong-digest, HOLD, wrong-binding, wrong-source,
  wrong-target, symlink and group/world-writable review files; wrong assignment
  argv or G2 authority. All stop before marker creation and synthetic transport.
- A synthetic failure before checkout creation consumes the fixed exclusive
  marker. The same assignment cannot reach stage again through the same or an
  alternate caller path. A distinct full read-only postflight follows the
  failed stage; failed preflight also gets postflight but no stage. Journal
  records forbid retry and preserve any partial checkout.
- Changed host, root-free floor, capsule inode, protected checkpoint digest,
  warm-cache inode, STOP path and already-present target all fail the fresh
  pre-stage snapshot gate. The previous repair review additionally covered
  marker/parent fsync ordering and failure, 0700 parent and symlink rejection.
- The **actual** JSON PASS file at the named path was read, SHA-256 matched and
  accepted by `verify_independent_request_review()` for these exact final
  bytes. No Operator transport was supplied for that check.

The tested transports were local synthetic callables, not SSH. No cleanup,
deletion, overwrite, rename, G2, training or evaluation was done. This review
does not make the archived point-in-time host snapshot a fresh preflight.
`git diff --check` was clean.

**Local file-mode caveat:** `checkout_invocation_final.py` intentionally rejects
a review receipt that is group/world-writable. This workspace's default umask
created the JSON as `0664`, so I changed only this exact local receipt to
`0644` before the actual-file authentication test; its content SHA-256 did
not change. Git tracks the executable bit, not `0644` versus `0664`. After
integrating this commit, the Supervisor/Operator must check that exact local
receipt's owner, regular-file/nlink-1 status, ancestry and non-writable mode,
set mode `0644` or `0600` if needed, then rehash it. A mode failure is HOLD,
not permission to weaken the checker. No other file is to be changed.

## Next authority boundary

The Supervisor may now prepare an **exact checkout-only, one-attempt**
assignment naming the JSON receipt digest above and every final request,
binding, source, target, code, audit/stage argv, stdin and marker pin. This
receipt alone does not permit calling the wrapper. The Operator must check
that fixed marker is still absent, its parent private, the local breaker/STOP
state clear and every fresh live full host preflight condition. Only then may
the one assigned checkout call occur. It must preserve any partial target on
failure and run the separate full read-only postflight even on preflight or
stage exception. Independent checkout-result review still gates CPU G2;
native training and evaluation have separate later gates. Any changed final
source/request/binding byte invalidates this PASS and requires a new review.
