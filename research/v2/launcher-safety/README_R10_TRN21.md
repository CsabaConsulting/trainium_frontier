# r10 exact trn21 A0 launcher profile — DISABLED / HOLD

This Implementer-only source change is a direct child of the independently
source-PASSed disabled r9 commit
`e046de16826bfc3b954136e3c60804aea0554ecc`. Its separate Verifier receipt
is `research/v2/launcher-safety/INDEPENDENT_SOURCE_VERIFIER_PASS_R9.md` on
`a669d4ebf8a8a94a31a0f8117b3ab39abe30e7c4` (SHA-256
`5ba9550b2bc2cb3ebf16792f0f1a9be75582899be9106f61c70271014895472d`).
That PASS covers r9 source only. This r10 diff needs a new, different
independent source review. No request is enabled or created here; no host,
SSH, AWS, transfer, checkout, monitor, guard, training, evaluation, cleanup,
breaker, assignment or submission action occurred.

## Exact review surface

Only the operational launcher and local synthetic tests change. The table is
SHA-256 of actual files; this runbook has no self-hash.

| File | SHA-256 |
| --- | --- |
| `research/fresh_start_launch.py` | `1be19bc31d2bdcde8abe9fe90cc1ac884d6efd5f7be197d7bdbfe64c257dcff7` |
| `research/test_fresh_start_launch_nonassert.py` | `7a2ac968ed4174426ff5bdc1381315b7d8466c016b0da96dd897c700644fec32` |
| `research/test_fresh_start_launch_exact_r9.py` | `03b031b4e8fb448281337baf5c2ca257acdfc8bc3d71cbd23381179be427c041` |
| `research/test_fresh_start_launch_trn21_r10.py` | `f7e1efc9bd2f8ba4a36e08793abccd4db4204eaa8f6e05464cec78188ed3d932` |

`train.py` remains
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
`prepare.py` remains
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The guard, host preflight, sync, interlock, and both disk-monitor production
files are unchanged from r9. The r9 trn22 profile retains its cache root,
absolute executable/argv and exact ten-key child environment. Its older
disabled r3 request still uses relative `timeout`/`env` and is refused.

## Trn21 pins and limits

The selected, content-complete trn21 A0 proposal is
`research/submission-validation/20260925-v2-c5c-a0-control-trn21-r1/EXPERIMENT_PROPOSAL.json`
(SHA-256 `55c6d500fbc535a45ad3afb6ef32882e30726b8ec90ac412e2ba987b5c71c1b9`),
ratified conditionally in
`research/v2/authorizations/20260925-trn21-c5c-a0-r1-proposal-ratification.json`
(SHA-256 `c483e2b00653ede82226a73cd9d9b7f3b01fb2988828c1f749f66e416641551d`).
The independent trn21 G2 result
`research/v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-final/INDEPENDENT_CPU_G2_RESULT_REVIEW.PASS.json`
(SHA-256 `57c5bf42682feac334a2c75d6f5ce003190764b8260914f2a6e8d4df8a4e757d`)
is **PASS_CPU_NUMERIC_ONLY** on the older e65 checkout. Its observed hostname
is `ip-172-31-34-85`; it does not authorize native training or establish
source identity for r10. The disabled trn21 native skeleton
`research/v2/authorizations/trn21-c5c-a0-native-request-hold/NATIVE_REQUEST.HOLD.json`
(SHA-256 `abe50c7b4ab64e1eadbfd12f997349c584adadf7813632895f55d0b89d336f2d`)
supplies the selected science label, cache root, and training settings.

The local and embedded remote launcher now permit **one** trn21 profile:

| Pin | Exact value |
| --- | --- |
| Host / hostname | `trn21` / `ip-172-31-34-85` |
| Science label | `fs-20260925-v2-c5c-a0-control-trn21-r1` |
| New private checkout target | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10` |
| Output directory | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10/out/fs-20260925-v2-c5c-a0-control-trn21-r1` |
| Step log | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10/research/run-logs/fs-20260925-v2-c5c-a0-control-trn21-r1.steps.jsonl` |
| Warm-cache root | `/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train` |

The `r10` checkout name is a new deterministic target of this disabled source
proposal. Its absence, ownership, clean source identity, eventual commit,
protected assets, and disk state are **not** proven here. It is distinct from
the accepted e65 `...trn21_stage_r2` checkout and the 446b-reserved
`...trn21_r4` target; neither is repointed or changed. Any future source stage
requires its own capsule, target review, fresh read-only preflight, assignment,
one-attempt Operator evidence and independent result review.

The trn21 child environment is exactly ten keys: LNC `2`, compiler flags
`--optlevel=1 --auto-cast matmult --auto-cast-type bf16`, seed `42`, budget
`1800`, and the six reviewed cache variables under the root above (`neff`,
`local`, `hlo`, `inductor`, `compiler`, `debug`). The ordered child command is
`/usr/bin/timeout --foreground --signal=TERM --kill-after=60s 3600s`
then `/usr/bin/env -i`, fixed `HOME`, `PATH`, `NKI_VENV_PATH` and
`PYTHONUNBUFFERED` assignments, those ten environment assignments, exact
output/step/run-ID assignments, and absolute native-venv `torchrun
--standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve
--no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output
--pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
--matrix-lr 0.015`. No request-supplied extra command token or environment
key is accepted. The embedded remote checks the c5c `train.py` and unchanged
`prepare.py` hashes before guard startup, in addition to the clean Git commit
and exact eleven-path fingerprint list.

The r9 absolute GNU-timeout version probe, five-second probe deadline,
optimized-Python refusal, full-chip topology, disk floors, STOP handling,
four live paired-monitor checks, interlock/no-spawn receipts and guard
containment stay in place. The future final request must independently bind
the exact r10 source commit and file hashes; this source change is not that
request. No trn21 paired-monitor target or native authority is implied.

## Local synthetic checks and remaining gates

The four launcher suites pass **31/31** in ordinary Python, `-O`, and inherited
`PYTHONOPTIMIZE=1`. The new suite tests the positive profile only through fake
SSH/Git/preflight/interlock/guard contacts, plus disabled request, wrong host,
hostname, label, old/reserved checkout, output, step log, argv, compiler,
cache, environment extras, fake `PATH`/timeout, non-GNU timeout, optimized
Python, fingerprint, dirty source, wrong commit and wrong science bytes.
The five retained r4 root-free, paired-monitor, host-cleanup,
early-refusal and interlock suites pass **59/59** in ordinary Python. These
are Implementer checks, not an independent Verifier verdict.

Disposition is **DISPATCH HOLD**. A separate Verifier must review this exact
diff and issue a source verdict. A source-PASS would still leave a clean r10
checkout, independent G2 source-identity decision or new numeric G2, full
host preflight, paired monitors and observer, exact disabled-to-enabled native
request review, one-shot Supervisor assignment, unconditional postflight and
separate scoring authorization pending. There is no new BPB result.
