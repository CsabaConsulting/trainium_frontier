# Independent Verifier: repaired trn21 stage-r2 checkout request

**Verdict: PASS for the disabled draft's offline safety mechanics only.** Reviewed
the isolated Implementer commit `8ad634414321d02158e340ba16861e83ec8345b5`.
This receipt is **not** the `INDEPENDENT_FINAL_REQUEST_REVIEW.PASS.json` required
by an enabled request. It authorizes no SSH, host checkout, CPU G2, native
training, evaluation, cleanup, retry, AWS action, adoption or submission.

## Exact reviewed identity

| File under `research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/` | SHA-256 |
|---|---|
| `GATE_CONTRACT.json` | `375bd94cc7490ce772cea736e78c74eb14710db9d592a796af7be046d7840516` |
| `RUNBOOK.md` | `fd45017bca157ffca944bb3906166989c8d888b43adda2d15af6b436df6f0b88` |
| `CHECKOUT_REQUEST.HOLD.json` | `fb0ea06f2f3b21f92e06a75e305c54afa3ad05bd14f948cf93e97c3900569629` |
| `INVOCATION_BINDING.HOLD.json` | `aabf1ae54eb6a5984448b2db224235730107e4aa2aedba9828139410eda21c9c` |
| `checkout_invocation.py` | `ae06183a5f394c87e81ce5ea81718fab4c720c5ffcd49e9b54fe9901bc8efd11` |
| `remote_checkout_once.py` | `e3faee8fe1274015b500134bcb767b041e37e6b7eef2e375f23d93a3ba8de3eb` |
| `test_checkout_invocation.py` | `32019be67be19b770c18acf7d821274e40d543280abc924b7776c2cf5f652cfa` |

The previous independent HOLD receipt remains unchanged at
`8656f59af4ca658d558525b69ed4c2a3b8d54a382ccc88d8603f8c20a0bb1678`.
I rehashed the frozen transfer terminal/result and extraction source/review at
`9ab90252e5e21812f5aad2af7fd0bfd1e71205e2f8f740f6f41faf6606fe6a2e`,
`bfafcb56ff449708dc31a1ebbe9023448018ea505e37a15847ae78bc3a3a2916`,
`e66afb7a1e3368ed7854eaa679e2bad9a6b8529e3b793e0f8fff2c69c496785d`
and `1a05c6dab4cbae6f49bd7e58760155d26562ba39df4` respectively.
The builder reconstructed the nine-receipt persistence bundle and the exact
104,005,787-byte e65 capsule identity. Frozen `train.py` and `prepare.py`
remain `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.

The frozen request/binding point to the sole checkout
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`
and source commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c`.
Recomputed audit argv SHA-256:
`cc1997adb5b815a739a12cbb09319f3869e08f7d8318899fea9496bee40f66a8`.
Recomputed stage argv SHA-256:
`621078d7a2563c1249500532070b921712f63a7356c50d43991e57f96eb876fe`.
Recomputed 69,949-byte stage stdin SHA-256:
`98f91c7be055660d194ca148957ed0885c49bc8e7525c129cb23421703e8a8a6`.
The frozen extraction code additionally hardcodes and checks that same
source/checkout/train/prepare identity; the remote adapter rehashes request,
extraction source and persistence bundle before calling it. The local builder
rejects altered request target or binding argv; the later assignment must
match exact request/binding/source/argv hashes before any transport.

## Independent negative checks

I reran all **10/10 wrapper** and **9/9 extraction** supplied tests. My separate
offline suite, `research/v2/verification/test_trn21_checkout_repair.py` at
SHA-256 `d4fe8e18ce61a0437484008ca87d960f9c244512049c8e9b6f92dcf77db68bb0`,
passed **8/8 tests**. The latter used only synthetic local transports and
covered:

- A synthetic stage failure before checkout creation consumes the one fixed
  marker. Replaying the same assignment via the same path fails `O_EXCL`;
  using another caller path fails before marker or SSH. Exactly one synthetic
  stage transport was reached, followed by a distinct full read-only postflight.
- Marker and parent-directory fsync, and the bound assignment journal record,
  precede the first synthetic transport. A simulated fsync failure leaves the
  marker in place and makes no host call. A non-0700 or symlink final parent is
  rejected before creating the marker or contacting a host.
- Missing, nonhex, wrong-path, wrong-hash, HOLD and wrong-binding review
  receipts cannot create a marker or reach transport. The actual regular-file
  review is SHA-256 checked and its structured PASS fields bind the final
  request, binding, wrapper, adapter, both argv hashes, stdin, host, source,
  checkout and marker. An altered assignment source/request/binding/argv/marker
  is likewise refused before transport.
- The future review digest is intentionally absent from the frozen request
  and must be supplied and rehashed by a later Supervisor assignment; this
  avoids a request/review hash cycle. The draft request and binding retain
  `checkout_authorized=false` and `dispatch_authorized=false`. Direct wrapper
  execution exits `DISPATCH HOLD`; the remote adapter rejects its disabled
  request. `git diff --check` was clean.

No current test is a real host result. The point-in-time local marker parent
was observed owner-private mode 0700 on an ext filesystem, but a future
Operator must check it again. The previous transfer-result PASS and checkout
implementation PASS cover their own narrow stages only.

## Still required before any checkout

An Implementer must freeze an **enabled successor**. Flipping either false
field changes the request bytes, binding and stage argv/stdin hashes, so this
static receipt cannot be carried forward as its final review. A different
Verifier must inspect the exact enabled request and produce the structured
`INDEPENDENT_FINAL_REQUEST_REVIEW.PASS.json` for those exact bytes. Only then
may the Supervisor issue a new checkout-only, one-attempt assignment naming
that actual receipt's SHA-256. The Operator must rerun fresh local/host safety
checks, and any stage result requires independent review before CPU G2. A
failed attempt preserves any partial checkout; no same-path retry or cleanup.
