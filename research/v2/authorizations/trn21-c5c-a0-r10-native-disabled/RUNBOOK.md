# trn21 c5c A0 r10 native control — disabled draft

**DISABLED.** This Implementer-only offline package is for an independent
Verifier. It authorizes no SSH, source transfer, checkout, monitor arm, guard,
training, evaluation, cleanup, protected-artifact deletion, breaker reset,
AWS action, adoption, or submission. Do not infer a launch from the selected
proposal. The unique science label is
`fs-20260925-v2-c5c-a0-control-trn21-r1`; the proposal identity is
`20260925-v2-c5c-a0-control-trn21-r10`. The frozen r10 launcher accepts
that r1 label, not an invented r10 science label. Its live-host absence and
one-attempt ledger absence remain to be proven immediately before dispatch.

## Frozen source, proposal, and draft chain

The scored source commit is
`22bfbab5a77d6feb670d2d737504f8b70b08ddc1`, not this documentation
commit. Its model SHA-256 is
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
unchanged `prepare.py` is
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`;
the exact r10 launcher is
`1be19bc31d2bdcde8abe9fe90cc1ac884d6efd5f7be197d7bdbfe64c257dcff7`.
The eleven source hashes in `GATE_CONTRACT.json` are checked against that
frozen commit by local offline tests. The independent source-only r10 launcher
review is
`research/v2/launcher-safety/INDEPENDENT_SOURCE_VERIFIER_PASS_R10_TRN21.md`
(SHA-256 `5e6fbfe6fd46608d0b220b99402185c78c809523026df7153882d3637de20c96`);
it is not host-stage or native authority.

The completed trn21 proposal at commit
`96fcec98a538673772febed826a653a4478cd9b0` hashes to
`1f616cde93b726553684b13c4c0ac2081e51bc289d90fdaff2d7ac18920b7e9d`.
An independent static review at
`dd9891c6d4dedfaa9d5762d400fc379c18670dcf` hashes to
`0b425319baac572fd0ee52e4c2cc9a60b1a2d071122d659707466dc9434a1aba`.
The Supervisor selected **this proposal only** in
`research/v2/authorizations/20260925-trn21-c5c-a0-r10-proposal-ratification.json`
(SHA-256 `1c5180d49e1122a8fdf992ac8735aefcbb71b92dfa924fffba335e33905b235b`).
That record explicitly has `launch_authorized=false`. The old e65
ratification does not cover this checkout.

| Draft artifact | SHA-256 |
| --- | --- |
| `NATIVE_REQUEST.DISABLED.json` | `f4612c71fd4cd1a429026b7a01d23c95de0652323a7dca34130b25d8a42454ce` |
| `BASE_BINDING.STATIC_ONLY.json` | `eb37d356edfc6e846a342e47f4e067c0d49dc30503199279bf320ae2bd651630` |
| `PAIR_BINDING.STATIC_ONLY.json` | `6613a520b85d219ec48c32be2368b039953d4b20fefa18f334dd273f8cedc766` |
| `PUBLIC20M_EVAL.DISABLED.json` | `444bb99406793ec4050a7d24dc0900adf1f21cc7da5d9d1f51e5e46608a715ca` |

The JSON chain is request SHA → base binding SHA → pair binding. The request
has `dispatch_authorized=false`; both monitor bindings are `STATIC_ONLY`;
the public-20M evaluation has `evaluation_authorized=false`. No one-shot
authorization file exists. Do not flip these drafts in place: a later enabled
request and binding chain requires separate independent review, fresh machine
preflight, and a one-label Supervisor assignment.

## Exact intended run

The r10 launcher accepts trn21 `ip-172-31-34-85`, UID 1000, with the
proposed private checkout
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10`.
That checkout is not yet accepted. The request's ordered child command is
`/usr/bin/timeout --foreground --signal=TERM --kill-after=60s 3600s`,
then `/usr/bin/env -i` and fixed assignments, then the native-venv
`torchrun --standalone --nproc_per_node=4 train.py` with exactly:

```text
--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015
```

The proposal's exact ten-key effective environment and 43-token ordered
child command are copied without science changes. LNC=2, seed=42, training
budget=1800 seconds, and
`NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16`.
The six train-cache paths are the **existing trn21 35c9 LR 0.015 control**
paths under
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train`.
Their ancestry and identities must be reauthenticated; no cold fallback,
copy, creation, pruning, or deletion is authorized. The trn21 G1 evaluator
uses its separate six established eval-cache paths, not these train caches.

The request pins the 10-GiB prelaunch root-free gate (10,737,418,240 B),
the 6.5-GiB immutable operational floor (6,979,321,856 B), and paired
monitor stop at 9,126,805,504 B. Paired monitors poll at 0.25 s, below
the one-second maximum. Their private directory is
`/home/ubuntu/trainium_a0_disk_monitor_20260925_r10`. The external
optimizer-boundary observer must be live **before** guard spawn; its script
hash is `a6f437fe8a9d0a14ec3025bc822f1afa12e3f355fca730e4ae0db7ae00263592`,
and the host copy must be reauthenticated. One run per machine; no second
active launcher. The unpiloted workflow-v2 `run_job.py` is not this run.

## Remaining gates before any native attempt

1. Independently accept an exact trn21 r10 source-stage request and matching
   trn22 r10 source stage. The current trn21
   `research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-r2-hold/REQUEST.HOLD.json`
   is a HOLD draft, not an accepted checkout. Source-stage operator receipts,
   clean HEAD, cross-session readback, postflight, and independent result
   acceptance are still required. Local `/usr/bin/timeout` compatibility
   discovered in source staging also requires independent resolution; this
   package does not repair a launcher or bypass a gate.
2. A different Verifier must issue a trn21 r10 G2 source-identity
   carry-forward disposition for actual accepted source, ordered argv,
   seed, LNC, compiler flags, time budget, CPU harness, and six caches
   against the old e65 four-rank numeric PASS. On mismatch, run a new
   separately reviewed and assigned r10 G2. The e65 receipt alone is not
   native authority.
3. Independently review this disabled package. Later, independently review
   the final enabled request and exact request→binding hashes. The ratified
   proposal is not that final review. The final Operator must prove the r1
   science label and all its output/log/attempt paths are unused.
4. Immediately before launch, perform `research/experiment-protocol.md`
   §1 checks on **both** hosts and local scored source: no training/eval/
   reconcile process, Neuron or shared-lock holder; root free space and
   mount/boot identities; same clean `22bfbab5` HEAD; exact eleven hashes,
   protected checkpoints, six cache identities, no STOP or breaker. The
   protected trn21 checkpoint hashes are
   `aaa4b7c0ff58c56df564954c21f8abdd786930bfaa1cc447f6fb8e5b0da83535`
   and `5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9`.
5. Authenticate two independently live paired disk monitors, STOP and owned
   guard stop path, and the read-only boundary observer before guard spawn.
   Any missing monitor/observer, floor failure, label marker, or holder is
   HOLD. Only then may a distinct one-label, one-attempt Supervisor
   assignment authorize an Operator run.

## Terminal, artifact, and evaluation gates

Retrieve exact command and environment, full log, step boundaries, paired
monitor samples, guard/interlock/observer receipts, checkpoint SHA, budget
fields, and an unconditional read-only postflight. Require finite losses,
sequential optimizer steps, full-chip use, no timeout/NaN/crash, a finite
model-only checkpoint, clean quiescence, and no disk-floor breach. Preserve
failures. An attempted label is consumed; retry requires a new label and
diagnosis.

The disabled public-20M plan is **not** a training subprocess or an enabled
eval request. After independently accepted training, bind the actual finite
checkpoint SHA in a separately reviewed and assigned fresh-process eval.
The calibrated trn21 G1 shape is `prepare.py eval-public`, Neuron,
batch size 1, context 1024, exactly 20,971,520 public tokens, 1,200-second
evaluator timeout, and 83,462,186 expected public bytes. Recheck host,
device, disk, lock, cache, source and checkpoint before evaluation; contain
it, prove quiescence, and retrieve trusted finite BPB/tokens/bytes. No 2M
screen or disabled draft is a valid scored control or BPB gain.

Local synthetic checks in `test_disabled_package.py` are Implementer
checks only. A different Verifier must independently inspect this package,
the frozen source and failure seams. This package has contacted no host.
