# A1 native five-step equivalence gate

## Current comparator-correction state

G0 ran once under the exact historical request-bearing head
`03bfeabf0f4293e30dc86eecf3aa9cce49e9735b`. Its immutable evidence is commit
`37aa3644e63e20a1427cfa69b44fd6de7e25cd81`; independent evidence verifier commit
`01d3cc04b3d55ed4d453fc1aff0e0eddad581682` correctly recorded FAIL because the accepted
comparator froze the dense exchange plan. Analysis commit
`2ee8c5c0283c760683aff3de48f243845a803d0c` established that the live A1 plan contains 66
owned FP32 entries, including two `1024 x 128` projections and two `8 x 32` n-gram gates.

The correction binds all 66 named entries, their ordered optimizer groups, shape, element count,
dtype, estimated cost, owner and offset; the two named BF16 replicated RMSProp tables; exact owner
counts `[27405059,30540035,30408963,29362179]`; stride `30540160`; padded/output elements
`122160640`; padded bytes `488642560`; and five FP32 gathers per rank. The exact manifest is
`a1_ownership_manifest.json`, SHA-256
`8119df41c8a9ecbfc3f9e69067c7e6ac162c2d5d8ba0a15635265304dcf865a2`. It also pins the
production/gate source hashes and the four immutable G0 report hashes.

Independent verifier commit `0f64bb06b7d1e9a8afd572b3c72c92acb8a360ee` then found the
first correction could hash one pathname instance and parse a later replacement, and could return
after a short result write. This second Implementer correction opens each authenticated manifest,
source, and rank report exactly once with `O_RDONLY|O_CLOEXEC|O_NOFOLLOW`, constrains its regular
single-link owner/mode/size identity, hashes and parses the same captured bytes while that descriptor
and its parent chain remain open, and revalidates the inode/path after both operations. Result output
uses an O_EXCL inode, a complete EINTR-safe write loop, file and directory fsync, same-inode exact
JSON readback, and removes its own partial inode on failure. These changes are still pending fresh
independent verification and do not unlock either later arm.

Independent verifier commit `3fb4deb222152564d7bf5dc4028246edcfd30c7a` accepted the
descriptor-bound input repair but found the result could still be overwritten through its writable
descriptor, detached by a parent rename, redirected through an ancestor symlink, or left at the
authorized name after hardlink drift. The r3 publication repair writes only a private O_EXCL 0600
temporary inode under a pinned parent dirfd, closes the writable descriptor, repeatedly reopens and
hashes exact bytes read-only, publishes with `renameat2(RENAME_NOREPLACE)`, fsyncs the pinned
directory, revalidates every lexical ancestor from `/`, and performs final read-only inode/content
seals. Failure cleanup removes only the created inode at the authorized temp/final names despite
link/mode/size drift and unlinks substituted symlink names without following their targets. This
local correction is also non-executable and pending independent verification.

This Implementer correction is not self-verified. The non-executable request
`g0-ownership-revalidation.pending.json` requires a fresh independent Verifier to reconstruct the
manifest from the pinned source and flags, run the positive archive check and mutation suite, and
issue a superseding G0 receipt. G1/G2, host access, training, cleanup, full A1 and submission all
remain locked.

The local-only immutable G0 check writes a new O_EXCL result and performs no host action:

```text
python research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/compare_gate_reports.py \
  --validate-g0-only \
  research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/runtime/research/run-logs/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/native-gate \
  --output <new-local-verifier-owned-output>
```

The production candidate remains byte-for-byte
`9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae`; only authenticated
gate copies import `research/a1_native_gate_observer.py`.

## Historical G0-only external finalization (consumed; do not reuse)

`g0_finalization.py` is a fail-closed materializer and runtime, not authority. An independent
Verifier must create a later `g0-authorization.json` that pins the current template, driver,
proposal, runbook, tests, all source hashes, reviewed implementation, and verifier receipt. The
authorization must enable G0 only; G1/G2, checkpoint pruning, and cache cleanup stay false. The
Verifier then supplies three literals out of band: the exact clean request-bearing HEAD, the
SHA-256 of the committed `g0-request.template.json` (the immutable request trust root), and the
SHA-256 of `g0-authorization.json`. Neither hash nor the HEAD is derived from the checkout by the
Operator, and the authorization never embeds its own hash.

Only after that separate verification may the Operator substitute the three supplied literals in
this non-executable template and materialize one concrete standard-launcher request:

```text
python research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g0_finalization.py materialize \
  --authorized-head EXTERNAL_40_HEX_HEAD \
  --request-sha256 EXTERNAL_64_HEX_REQUEST_TEMPLATE_SHA256 \
  --authorization-sha256 EXTERNAL_64_HEX_AUTHORIZATION_SHA256 \
  --request-template research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g0-request.template.json \
  --authorization research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g0-authorization.json \
  --output /tmp/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1.request.json
```

The sentinel spelling above is intentionally not valid CLI input. This repository does not supply
the literals. The materializer requires the exact symbolic branch, exact external HEAD, clean
protected files, exact source hashes, executable independently finalized authorization, and
O_EXCL output. It never launches. After inspecting its emitted SHA, the later authorized Operator
would pass that concrete file unchanged to the existing launcher:

```text
python research/fresh_start_launch.py \
  /tmp/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1.request.json
```

The commands in this historical section were consumed by the archived G0 run and are not reusable.
The corrected comparator changes protected package bytes, so the old authorization cannot
authorize a descendant or any rerun. No rerun is requested by this correction.

## Fixed order and lifecycle

The sole eligible host is `trn21` (`ip-172-31-34-85`). Only G0 can be finalized in this package.
G1 stays locked until the exact G0 archive and clean postflight are retrieved, hashed, committed,
and externally bound; G2 stays locked until the corresponding authenticated G1 evidence. Before G0,
the standard fresh-start launcher/guard must prove: the exact symbolic branch at the independently
approved HEAD, clean tree and origin alignment, exact regular no-follow source files and hashes,
no live training or descendants, and absent unique remote/output/report/cache paths. It must sample
the same filesystem holding the arm artifacts and require at least 6,979,321,856 free bytes. Below
that threshold, STOP; below 5,368,709,120 bytes, hard abort. This package grants no deletion or
cleanup authority.

Each arm must run in its own private process/session group. A normal exit is not sufficient while
any descendant lives. On timeout/error, apply bounded TERM and then KILL to the whole group, reap,
prove process-group disappearance, then fsync/hash immutable complete stdout/stderr, manifest,
checkpoint and rank reports. Revalidate their sizes/hashes before the archived completion record.
Archive a failure with all available process/preflight/postflight evidence; never advance to the
next arm after a failure or survivor. These are the verified A1-r2 lifecycle protections, not an
invitation to reuse its request.

For each arm set exactly:

```text
NEURON_LOGICAL_NC_CONFIG=2
NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16
NEURON_COMPETITION_R1_SEED=42
PYTHONUNBUFFERED=1
NEURON_COMPILE_CACHE_DIR=<the exact unique initially absent cache in requests.pending.json>
```

The finalized launcher must construct the argv array below from the committed G0 template without shell
evaluation. `<SOURCE>`, `<ARM>`, `<OBSERVATION>`, `<LABEL>`, `<OUT>` and `<REPORT>` are the literal
fields of the corresponding request entry, not Operator choices:

```text
/home/ubuntu/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=4 <SOURCE> \
  --num-steps 5 --max-train-seconds 1800 \
  --n-embd 1024 --ngram-ve --ngram-ve-layers=-1 --ngram-ve-table-mult 64 \
  --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr \
  --ngram-ve-bf16 --ngram-ve-clip-exclude --no-qk-shift --compile-sdpa-direct \
  --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce \
  --seq-len 1024 --no-eval-public --matrix-lr 0.015 \
  --out-dir <OUT> --native-gate-arm <ARM> <OBSERVATION> \
  --native-gate-label <LABEL> --native-gate-host trn21 --native-gate-report-dir <REPORT>
```

G0 uses `train_gate_hybrid.py` and passes
`--no-native-gate-observation`; G1/G2 pass `--native-gate-observation`. Those and the ordinary
source's omission of ownership installation are the enumerated gate-only differences. G0 still
emits losses, pre/post persistent model and logical optimizer hashes, final state and ordered
collectives, which makes observation ON/OFF non-perturbation testable. G1/G2 additionally emit only
compact SHA256/shape/dtype/norm/finite metadata for the b5/t5 gradients and clip scalars; tensors
are never printed or stored.

All three runs start from seed 42, the identical data cursor and fresh model/optimizer state. Five
means exactly five completed optimizer updates, not five microbatches. A skipped update, missing or
nonfinite table gradient, table in owner planning, missed/duplicate RMS call, rank disagreement,
config drift or extra/missing record is a failed gate. Wall time and throughput are diagnostics and
never equality criteria.

## Exact comparison and checkpoint gate

After all three arm archives are retrieved without mutation, run the comparator once into an
initially absent output:

```text
python research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/compare_gate_reports.py \
  --g0 <retrieved-G0-native-gate-dir> --g1 <retrieved-G1-native-gate-dir> \
  --g2 <retrieved-G2-native-gate-dir> --output <new-comparison.json>
```

There is no numeric tolerance. G0/G1 shared records must be exact, proving the observer is
non-perturbing. The RMS verdict requires same-rank G1/G2 equality for every local pre-AVG,
post-AVG/preclip and postclip table gradient; clip scalar bytes; table pre/post parameter,
`exp_avg_sq`, step and bias; cross-hybrid-rank equality after AVG; exclusion from ownership; one RMS
call per update; and no RMS-specific collective. The integrated verdict separately requires exact
loss/full persistent model and named logical AdamW/Muon state equality, normalizing hybrid owner
shards by their named union. If RMS is exact but a non-table integrated record differs, the only
valid outcome is `RMS_PASS_INTEGRATED_BLOCKED`; it does not authorize full A1.

Then run the model-only gate into another initially absent output:

```text
python research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/verify_model_checkpoints.py \
  --hybrid-source research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_hybrid.py \
  --ordinary-source research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_ordinary.py \
  --g0-checkpoint <G0-final.pt> --g1-checkpoint <G1-final.pt> --g2-checkpoint <G2-final.pt> \
  --output <new-model-check.json>
```

It permits exactly the model-only checkpoint schema, uses `weights_only=True`, strict-loads the
unchanged eval model, checks all persistent tensors finite, and requires exact checkpoint state,
fixed eval-mode logits and causal-prefix identity. No optimizer serialization/resume is attempted;
hybrid interruption means abort and restart with a new label.

A native gate is complete only after all arm evidence, comparison, model check, source manifests,
pre/postflight and process-quiescence receipts are committed and independently reviewed. Only
`PASS` for both verdicts can support a later, separate full-A1 authorization.

## Local package verification

The Implementer may run only CPU/static tests:

```text
python -m unittest -v \
  research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/test_gate_package.py
python -m py_compile research/a1_native_gate_observer.py \
  research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/*.py
```
