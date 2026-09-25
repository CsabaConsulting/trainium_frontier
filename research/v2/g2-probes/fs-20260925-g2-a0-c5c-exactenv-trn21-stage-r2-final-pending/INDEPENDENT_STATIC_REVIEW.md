# Independent static review: trn21 c5c A0 stage-r2 CPU G2 final-pending package

**Verdict: PASS for the disabled offline package only; HOLD for CPU G2, native
training, evaluation, and submission.** This is an independent Verifier review
of Implementer commit `74851cfc56030b36ba3933f6db3682d7b2c2a3c9`. It is
not a checkout-result review, CPU-only request, Supervisor assignment, host
preflight, or permission to run. I made no SSH connection and performed no
host, AWS, CPU G2, Neuron, training, or evaluation action.

## Exact reviewed bytes and lineage

The four target files at
`research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2-final-pending/`
were unchanged from the reviewed commit and rehashed locally:

| File | SHA-256 |
| --- | --- |
| `GATE_CONTRACT.json` | `743d75e489831d21cd682a3a7a2b84becd8f0dc8f8cfc914dc4aa27123c9d905` |
| `RUNBOOK.md` | `a02bb89ce4a3a1e7a5363faee998a0f15824405df8dc06721e8b5e6dd57c4309` |
| `build_g2_argv.py` | `f308730af1e98811d609c00e0bf348bc78c04354661e53674379184c71a96bdf` |
| `test_g2_argv.py` | `2347a5e9052eee0d5727a39eadf0eceb3bbf7c0e2adfc73df359d1d7cc2434df` |

The package independently rehashes the immutable earlier G2 draft contract,
builder, and static review (`9867b33c…`, `74b1d49c…`, `4712413b…`); the
accepted three-file transfer Operator receipt and separate result review
(`9ab90252…`, `bfafcb56…`); and the exact checkout request, independent
request-review PASS, and checkout-only Supervisor assignment (`be97460e…`,
`acfc9036…`, `4c1d1774…`). I independently recomputed those eight full
digests and found exact matches. The source remains the same single planned
trn21 checkout at
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`,
commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c`, `train.py`
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
and unmodified `prepare.py`
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The frozen 64-word launch and wait/stop/quiesce command digests still match
`1810cdc…`, `3a41612…`, `9216b16…`, and `c959acb…`; no science argv or
effective environment changed in this successor.

## Independent checks

- `git diff 74851cfc^ 74851cfc --check`: PASS.
- Supplied offline suite: **9/9 test methods PASS**, including direct
  execution refusing with `DISPATCH HOLD`.
- An independent, in-memory mutation matrix: **15/15 rejected**. It changed
  source checkout, commit, train/prepare hashes, CPU label/root, disk floor,
  tool UID/device/link count, contained-tool inode to a boolean, attempted
  checkout-result and review-hash spoofing, native authorization, and a frozen
  command digest. This matrix ran locally without writing or contacting a host.
- The earlier review's three specific bypasses are now rejected in both new
  and old contract inputs: `one_attempt_per_label=false`, tool mode `0777`,
  and `preflight_inode=0`. The new checker also rejects zero/boolean/wrong
  contained-tool inode and mismatches against the retained cross-session
  transfer result, final postflight, and reviewed checkout request. Required
  tool policy remains UID 1000, ext4 device 66305, mode `0600`, one link,
  preflight inode 294946 and contained inode 296647.

The package is fail-closed for its limited scope: all dispatch fields are
`false`; the checkout Operator-result and independent result-review bindings
are `null`; `build_plan()` refuses a forged positive checkout result or
dispatch bit; and executing the builder directly never runs its constructed
commands. These properties support static archiving, **not** host use.

## Remaining gates

A different Verifier must accept the actual checkout Operator evidence and
full postflight. Then an Implementer must create a **new** exact CPU-only
request binding that result and its review; a separate Verifier must review
that request. A fresh live host/source/tool/asset/label/STOP/breaker/disk
preflight and a narrow one-attempt Supervisor CPU-only assignment are needed
before an Operator may run four-rank synthetic CPU G2. The Operator must
retrieve the raw result, prove quiescence and perform an unconditional
read-only postflight; a different Verifier then checks numerical equality to
the pinned c5c anchor. Do not flip this disabled object's fields in place,
reuse an old label, substitute a tool or source checkout, clean protected
artifacts, or infer native/evaluation authority from a CPU result.
