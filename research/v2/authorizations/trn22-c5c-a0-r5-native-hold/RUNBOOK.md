# trn22 c5c A0 r5: offline native-control package

**DISABLED. This Implementer package authorizes no host contact, transfer,
checkout, monitor arm, training, or evaluation.** The trn22 r4 source transfer
still has a controlling independent HOLD. No clean `446b6576` r4 host checkout or
new native result has been accepted. Do not stage these static files on trn22.

The new science label is
`fs-20260925-v2-c5c-a0-control-trn22-r5`. It is distinct from the consumed
zero-step r1 and the older r2/r3/r4 draft labels. The expected future source
checkout retains the *r4 checkout path*
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r4`;
only the r5 output, run-log, observer receipt, and new external-monitor paths
use the r5 science label. The checkout must be independently accepted as
clean at commit `446b657695f086f9b484fa601c9a3b61c2495912` before a native
request can be finalized. The package commit is not the scored source commit.

## Frozen bytes and science

`GATE_CONTRACT.json` binds these four exact draft files. The
`NATIVE_REQUEST.DISABLED.json` request has `dispatch_authorized: false`; the
frozen r4 launcher checks that flag before it can create a local dispatch
receipt or contact SSH. The monitor bindings are `STATIC_ONLY` and have no
one-shot authorization file. `PUBLIC20M_EVAL.DISABLED.json` also has
`evaluation_authorized: false` and no checkpoint hash.

| File in this directory | SHA-256 |
| --- | --- |
| `NATIVE_REQUEST.DISABLED.json` | `086b78e315050d3a0a44014e03c2ab28d4279eb5f9c0f2f4882447ed2ab461be` |
| `BASE_BINDING.STATIC_ONLY.json` | `4f3af43cae356f58802232bb47c8c3a12b323e7ffaa140b40025216f13dc3b9f` |
| `PAIR_BINDING.STATIC_ONLY.json` | `d45e2c680a15c8ba58858d2ddb9b285c4868cb739cf4d394172e70b3d7bbfce1` |
| `PUBLIC20M_EVAL.DISABLED.json` | `9714c20f8040d3bd9bee0879280db2353c06f9060a11e35ddcf78a3cf4652888` |

The request and both bindings pin the eleven production-file SHA-256 values
from frozen `446b6576`. In particular, `train.py` is
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and unchanged `prepare.py` is
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The r4 exact-source capsule is pinned as
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`
in the upstream stage contract; it has not been transferred under this
package. The old r4 request/binding chain remains historical and is not
renamed or overwritten.

The ordered training argv is the queue's dense Track-A A0 baseline:
`--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct
--nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce
--seq-len 1024 --no-eval-public --matrix-lr 0.015`. The child command is
bounded by `timeout --foreground --signal=TERM --kill-after=60s 3600s`, then
`env -i` with explicit `NEURON_LOGICAL_NC_CONFIG=2`, seed `42`,
`NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800`, and
`NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16`.
It uses `torchrun --standalone --nproc_per_node=4` on the full chip. The six
existing warm training-cache paths are exact strings in the request and must
be reauthenticated on the host. This is an unchanged control recipe, not a
quality arm or a submission command.

The older r2 A0 proposal and Supervisor ratification are evidence for the
same recipe and reason for a new control. Their one-attempt label is **r2**,
so they do not select or authorize **r5**. A completed r5-specific control
proposal based on `research/agentic/EXPERIMENT_PROPOSAL.json`, separate
Supervisor selection/ratification, and a new exact one-shot assignment remain
required. `research/v2/queue.json` lists A0 as a proposed control, not an
individual launch authorization.

## Gates before finalization or any native attempt

1. Repair the separately held r4 transfer writer and receive a new
   independent final PASS, fresh full read-only host preflight and a narrow
   Supervisor transfer-only assignment. A distinct Operator stages the exact
   capsule once, proves fresh-session byte readback, and archives a terminal
   receipt. A separate Verifier accepts that result. The earlier r3 PASS was
   superseded by a late-label HOLD, and the later lock/final-scan draft has
   its own independent static HOLD. Neither is transfer authority.
2. A separately reviewed and assigned checkout step reconstructs the exact
   `446b6576` source at the r4 path. It proves clean tracked and untracked
   status, exact HEAD and eleven source hashes, private path identity, and
   unchanged protected assets. A separate Verifier accepts its raw result.
   The r4 source-safety PASS and disabled-package static PASS cover only
   local code, not this host stage.
3. Rebind trn22's accepted G1 evaluator calibration and the independently
   reviewed r3-to-r4 four-rank G2 **numeric-only** carry-forward to the
   actual staged bytes, ordered argv, seed, LNC, compiler flags, budget and
   six caches. Any mismatch voids carry-forward and needs a new reviewed
   exact-source CPU probe. G2 does not prove Neuron compile or native safety.
4. Complete and ratify the r5 proposal. Materialize a **new** enabled native
   request after the checkout receipt, with exact source, host and science
   paths. Regenerate its request → base-binding → pair-binding SHA chain and
   review those final bytes independently. Never flip this disabled draft in
   place or reuse the old r4 binding hashes. The final request must retain
   the proven r4 safety launcher and paired monitor interlock; the new
   `run_job.py` is still a separate one-host pilot, not a substitute here.
5. Immediately before any launch, perform all of
   `research/experiment-protocol.md` §1 on **both boxes**: idle process,
   `df -h /`, Neuron holder, same-code sync across the local frozen source
   and both host checkouts, clean Git, full environment capture including
   box and compiler flags, and seed 42. The exact-code-sync result must be
   recorded; a different HEAD on either box is not silently waived. Also
   check trn22 hostname/UID/boot/mount, `/dev/neuron*`, shared lock,
   STOP/breakers, protected checkpoint and six cache path/device/inode
   identities, r5 label/output/log absence, clean r4 checkout and all
   request/source hashes. Root free must be at least `10,737,418,240` bytes
   before launch. The immutable operating floor is `6,979,321,856` bytes.
6. Stage and hash both r4 monitor scripts in a private new r5 directory;
   prove two independently live authenticated monitor roles, poll no slower
   than one second, with preemptive stop at `9,126,805,504` free bytes.
   Reauthenticate and start the read-only optimizer-boundary observer before
   the guard. A monitor error, below-floor sample, unavailable owned stop,
   or missing observer is HOLD. One run per host; no second active launcher.
   Only after a fresh full preflight and independent final review may the
   Supervisor bind a one-label, one-attempt assignment to the final bytes.
   An Operator alone may attempt the reviewed command once.

## Terminal evidence and fresh score

The Operator must archive the exact launch command, environment, host,
guard/monitor/observer receipts, continuous root-free samples, sequential
optimizer boundaries, complete log, checkpoint hash, budget fields and a
separate full read-only postflight. Require at least five sequential
boundaries, finite loss and model-only checkpoint, no NaN/crash/timeout or
over-budget, no disk-floor breach, and clean quiescence. Preserve artifacts
on failure; the r5 label is consumed by any attempted dispatch. A failure
needs diagnosis and a distinct later label.

After a valid training terminal review, a **separate** exact evaluation
request may bind the actual checkpoint SHA and reauthenticate source,
evaluator, environment and six warm eval caches. The disabled eval plan
spells the fresh-process `prepare.py eval-public` command: device `neuron`,
batch size 1, context 1024, exactly 20,971,520 public tokens, 1200-second
evaluator timeout. Recheck host idle/device/disk/lock and protected assets
before it. Retrieve the trusted JSON with `val_bpb`, token and byte counts,
finite nats and no eval timeout; require 20,971,520 tokens and the calibrated
83,462,186 public bytes. A public-2M screen alone cannot establish the A0
control. Compare the full result to the predeclared numerical reference band
and valid same-host controls, then record a control verdict without calling
it a quality win. `workflow/tools/decide.py` remains unchanged; any adoption
still needs human acknowledgement. No submission follows from this package.

Local Implementer check (not an independent Verifier receipt):

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/v2/authorizations/trn22-c5c-a0-r5-native-hold/test_disabled_package.py -v
```
