# Independent Verifier — trn21 c5c A0 stage-r2 checkout request

**Verdict: HOLD for any checkout-ready successor.** The frozen draft at
`5fc762f851534a61fd29b8ce2ec5af07ac9ca5cb` is correctly disabled and
has consistent evidence, source, argv and stdin pins. It is suitable as a
non-dispatchable review artifact only. Its future Operator wrapper does not
mechanically enforce one attempt per Supervisor assignment and does not
authenticate the claimed independent request-review receipt. Do not enable,
assign, SSH, extract, run CPU G2, train or evaluate from this package.

## Exact reviewed bytes and positive checks

The seven new files at the commit above have these SHA-256 digests:

| File | SHA-256 |
|---|---|
| `CHECKOUT_REQUEST.HOLD.json` | `69e570f91ec337d680e721a8dc6ee2f178da09b969809ae89be6b7b1eb7ffdeb` |
| `GATE_CONTRACT.json` | `e8801ab3aeb0a6fbd31e1c48ad350983d9c3c50ca3759b91ba936a2d5c186bc1` |
| `INVOCATION_BINDING.HOLD.json` | `16ce854c76031838ee9c2e251bf46557aa8deb662b54115e6b5347463507b231` |
| `RUNBOOK.md` | `2e5873fa2c2c3ae0c81707387bdb62a3bb596a1ad8cf3c147f40e01004bc0fe7` |
| `checkout_invocation.py` | `e2b5d7b4b5d748f6716940e8ea912d2d793e0fbfdba4f396b370b37ad296d56a` |
| `remote_checkout_once.py` | `e3faee8fe1274015b500134bcb767b041e37e6b7eef2e375f23d93a3ba8de3eb` |
| `test_checkout_invocation.py` | `409205adc2cf7150b9590616d7070e996f45f5dddc5fc4fd65c15198d707e703` |

I independently rehashed the referenced archived transfer terminal receipt
(`9ab90252e5e21812f5aad2af7fd0bfd1e71205e2f8f740f6f41faf6606fe6a2e`),
its independent result review
(`bfafcb56ff449708dc31a1ebbe9023448018ea505e37a15847ae78bc3a3a2916`),
the frozen extraction contract
(`cb034ffea544c560104a4d558a8a18119f50a4db5088045e0c5d711635bfbd65`),
the extraction source
(`e66afb7a1e3368ed7854eaa679e2bad9a6b8529e3b793e0f8fff2c69c496785d`),
and its independent implementation review
(`1a05c6dab4cbae6f49bd7e58760155d26562ba39df6a85524fe4a50289679df4`).
The builder reconstructs and validates all nine transfer receipts with the
canonical bundle digest
`13db20bb2887eb6ff1c9feda31ea8453c9252ace18ff571021f866780e7cfe86`.
It binds the one capsule, 104,005,787 bytes at
`/home/ubuntu/trainium-c5c-e65-one-commit-20260925.trn21-stage-r2.tar.gz`,
SHA-256 `32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264`,
to the one absent checkout
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`
and e65 commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c`.
`train.py` and unchanged `prepare.py` are pinned to
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.

`build_plan()` recomputed the exact read-only pre/postflight argv SHA-256
`cc1997adb5b815a739a12cbb09319f3869e08f7d8318899fea9496bee40f66a8`,
single-stage argv SHA-256
`0700cf456560b75069bb4c18aa51db4b67f6657022e80e96bcd03f9c8fc3c187`,
and 69,541-byte stdin SHA-256
`a37231be7549c80d11861b9d6793c985b96780ac05ef8c751171c28d82178b7c`.
The command is one pinned `ssh -T` subprocess with strict host-key options,
not a shell chain. The archived full read-only probe captures host/boot/mount,
artifact hashes/inodes, protected checkpoints/caches, STOP/label/process/lock/
Neuron state and root free space. The wrapper validates a fresh snapshot before
stage and uses a separate call to that same full probe in `finally` after
preflight failure, stage error or success. The stage adapter rehashes its
request/source/bundle payload before calling the frozen extraction function.
The current request has `checkout_authorized=false`, the binding has
`dispatch_authorized=false`, and both the direct wrapper and adapter refuse it.
I found no cleanup, deletion, rename, host signal or training/evaluation call
in the new wrapper/adapter. `git diff --check` was clean.

The supplied wrapper suite passed **7/7**, the frozen extraction suite passed
**9/9** locally, and my separate negative suite
`INDEPENDENT_NEGATIVE_TESTS.py` (SHA-256
`58a9646e68e1b007266aef412d7f1621eb920db12f1f67de7ea006345d9677e1`)
passed **2/2**. All used synthetic local transport; none contacted trn21.

## Blocking adversarial findings

1. **Same assignment can cause two stage attempts.** `operator_once()` accepts
   any caller-supplied `journal_path` (`checkout_invocation.py:335-356`).
   `Journal` uses `O_EXCL` only for that path (`:225-238`), so a second path
   bypasses the claimed one-attempt journal interlock. The independent local
   test injected an otherwise enabled in-memory plan, returned a synthetic
   pre-creation stage failure, then invoked the **same** assignment with two
   distinct journal paths. Both invocations passed fresh synthetic preflight
   and issued the stage SSH argv once: **two stage attempts total**. Both
   reported HOLD; there was no real SSH or extraction. A real stage failure
   before `mkdir` leaves the checkout absent, so the next invocation would
   not be stopped by the absent-path check either. The `one_attempt_only`
   assignment field is not a single-use interlock.
2. **The independent review is not authenticated at dispatch.** The assignment
   check accepts any 64-character string as
   `independent_request_review_sha256` (`checkout_invocation.py:352-354`),
   without checking hex or rehashing an actual receipt. An independent local
   test used a nonexistent, non-hex `"z" * 64` value and reached the
   synthetic stage transport. This does not make the currently disabled draft
   dispatchable; it means an enabled successor would not enforce the named
   review gate.

## Repair and limits

The Implementer should bind **one exact local journal/attempt-marker path**
to the final request and Supervisor assignment, create it exclusively and
fsync it before any host interaction, and refuse a second invocation even if
the first fails before checkout creation. The final enabled package should
bind and rehash the actual independent review receipt named by its later
Supervisor assignment, not merely check its digest length. Add adversarial
tests for both. Changing the disabled authorization fields alters the request,
stage argv and stdin hashes, so the enabled successor needs a new frozen
package and **new independent review**; this receipt cannot be re-used as
authorization for it.

No SSH, live host observation, checkout, CPU G2, native run, evaluation,
cleanup, AWS action, breaker reset, adoption or submission was performed.
The prior transfer/result and extraction-code reviews remain valid only
within their stated scopes. A separate final request review, fresh complete
host preflight, narrow Supervisor checkout-only assignment, and independent
checkout-result review still gate every later stage.
