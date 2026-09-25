# trn22 c5c A0 r10 native control — disabled draft

**DISABLED.** This is an Implementer-only offline package for independent
review. It authorizes no SSH, source transfer, checkout, monitor, guard,
training, evaluation, deletion, breaker reset, AWS action, adoption, or
submission. The existing one-shot r1 disk failure remains failed; this draft
does not relabel it or clear a breaker.

The intended control is the existing A0 dense research-v2 recipe, not a new
quality arm. The label
`fs-20260925-v2-c5c-a0-control-trn22-r10` is new in the repository and must
still be proven absent on trn22 immediately before any eventual attempt.
The proposed checkout
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r10_stage_r1`
is the distinct r10-stage target, not the accepted e65 checkout or the
historical r4 target. Its creation and acceptance are pending. The proposed
private monitor directory is
`/home/ubuntu/trainium_a0_disk_monitor_20260925_r10`.

## Frozen source and draft chain

The scored source commit is
`22bfbab5a77d6feb670d2d737504f8b70b08ddc1`, not this package commit.
The pinned model is
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
the unchanged evaluator is
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The operational launcher is
`1be19bc31d2bdcde8abe9fe90cc1ac884d6efd5f7be197d7bdbfe64c257dcff7`.
The eleven source hashes in `GATE_CONTRACT.json` are recomputed from the
frozen Git commit in the offline tests. The independent r10 *source-only*
PASS is
`research/v2/launcher-safety/INDEPENDENT_SOURCE_VERIFIER_PASS_R10_TRN21.md`
(SHA-256 `5e6fbfe6fd46608d0b220b99402185c78c809523026df7153882d3637de20c96`);
it is not a native request or host-stage PASS.

| Draft artifact | SHA-256 |
| --- | --- |
| `NATIVE_REQUEST.DISABLED.json` | `3885415c561649bf4e8fe09e948e2793aec718b483566a4575ed0591d4b0977e` |
| `BASE_BINDING.STATIC_ONLY.json` | `4cc37ad06f23b0a17cf1225418e842870aeb3de6ed4086454ee23dfb6c6a0d51` |
| `PAIR_BINDING.STATIC_ONLY.json` | `9c067b8e97179f4ae3b6d4e410816a5e8826b94c8bda42ebb96bad05321b4700` |
| `PUBLIC20M_EVAL.DISABLED.json` | `45822174f7bd3842f89e2056cff821826de3e8ec7c81f56c3109e21c44da2ec2` |

The JSON chain is request SHA → base binding SHA → pair binding. Both
bindings are `STATIC_ONLY`, the native request has
`dispatch_authorized=false`, and the evaluation plan has
`evaluation_authorized=false`; no one-shot authorization file exists.
Do not flip these drafts in place. A later enabled request and binding
chain must be generated after the host-stage result, proposal, G2 disposition
and fresh preflight, then independently reviewed and assigned once.

## Exact intended run

The r10 launcher accepts trn22 host `ip-172-31-32-253` only with its
fixed full-chip profile. The child command is the ordered absolute
`/usr/bin/timeout --foreground --signal=TERM --kill-after=60s 3600s`,
then `/usr/bin/env -i` and the fixed assignments, then absolute native
`torchrun --standalone --nproc_per_node=4 train.py` with:

```text
--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015
```

The ten-key environment is exact: LNC=2; seed=42; budget=1800;
`NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16`;
and the six existing warm-cache paths under the prior trn22 control. No
request-controlled extra key, PATH override, alternate binary, flag, or
training duration is accepted. The request pins a 10-GiB prelaunch root-free
gate (10,737,418,240 B), 6.5-GiB immutable floor (6,979,321,856 B), and
paired monitor stop at 9,126,805,504 B. The monitors poll at 0.25 s,
below the 1-s maximum; the boundary observer must be live before guard spawn.
The observer is a separately staged external script, not a path in the frozen
r10 science tree; its controller-side bytes hash to
`a6f437fe8a9d0a14ec3025bc822f1afa12e3f355fca730e4ae0db7ae00263592`
and its host copy must be reauthenticated before use.
There is one run per host and no second active launcher; the new
`workflow/tools/run_job.py` remains an unpiloted alternative, not this run.

## Still required before one native attempt

1. Obtain an independent final PASS on an exact r10 source-stage request,
   fresh read-only trn22 preflight, and a narrow one-attempt Supervisor
   stage assignment. The archived r10 stage draft is HOLD, not authority.
   The Operator must stage once, prove stable fresh-session bytes and a clean
   r10 HEAD, preserve protected assets, and archive a terminal receipt. A
   different Verifier must accept the result.
2. Rebind the old trn22 four-rank, three-step G2 numeric PASS to the *actual*
   staged r10 train/prepare bytes, ordered argv, seed, LNC, compiler flags,
   budget, CPU harness and caches through a new independent source-identity
   disposition. The old e65 receipt alone is insufficient. If identity
   fails, complete a new independently reviewed r10 G2.
3. Have a Researcher complete the r10-label proposal based on
   `research/agentic/EXPERIMENT_PROPOSAL.json` and a Supervisor ratify it.
   The old r2 proposal/ratification proves the recipe but not this label.
   An independent Verifier reviews the final enabled request, exact
   request→binding hashes, source checkout and safety wiring. This disabled
   package's eventual static review does not replace that final review.
4. Immediately before any launch, the Operator performs all
   `research/experiment-protocol.md` §1 checks on **both** hosts: idle
   train/eval/reconcile processes, root disk, Neuron holders, and the same
   clean Git HEAD on trn21, trn22 and the local scored-source checkout. Record
   the full compiler flags and host in the environment receipt; seed=42.
   On trn22 also verify exact hostname/UID/boot/mount, private checkout,
   clean tracked and untracked status, all eleven source hashes, no STOP or
   breaker, no prior science label/output/log, shared lock and device
   quiescence, protected checkpoints and all six cache identities.
5. Arm two independently live authenticated paired root-free monitors with
   STOP and an owned guard stop path; start and authenticate the read-only
   optimizer-boundary observer *before* the guard. A missing monitor,
   observer, interlock, below-gate disk value or holder is HOLD. Only then
   may a separate one-label, one-attempt Supervisor assignment authorize an
   Operator to use the reviewed launcher.

## Terminal and evaluation gates

The Operator retrieves the exact command, environment, full log, step
boundaries, live monitor samples, guard/interlock/observer receipts,
checkpoint hash, budget fields and unconditional read-only postflight.
Require finite losses, sequential optimizer steps, no timeout/NaN/crash,
full-chip use, a finite model-only checkpoint, clean quiescence and no
disk-floor breach. Preserve every failure; a native attempt consumes its
label and a retry needs a new label and diagnosis.

Only after an independently accepted training terminal result can a
separately reviewed and assigned fresh-process public evaluation bind the
checkpoint SHA. The disabled eval plan is the calibrated G1 path:
`prepare.py eval-public`, Neuron, batch size 1, context 1024, exactly
20,971,520 public tokens, 1,200-s evaluator timeout and 83,462,186 expected
public bytes. Recheck host/device/disk/lock/cache/source before evaluation,
contain it and prove quiescence, then retrieve trusted finite BPB/tokens/
bytes. A 2M screen or this draft is not a valid control or a BPB gain.

Local synthetic checks in `test_disabled_package.py` are Implementer
checks only. A different Verifier must inspect the frozen source, this
package, and failure seams; no host has been contacted by this package.
