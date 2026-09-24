# A1 native five-step equivalence gate

This is a prepared, non-executable package. `requests.pending.json` deliberately contains
`PENDING_EXTERNAL_VERIFIER` and `executable=false`. It authorizes no SSH, host mutation, cache
cleanup, native execution, training, full A1, or submission. An independent Verifier must first
pin the clean origin-aligned final HEAD, every package/source hash, and three distinct runtime
requests. The production candidate remains byte-for-byte
`9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae`; only authenticated
gate copies import `research/a1_native_gate_observer.py`.

## Fixed order and lifecycle

The sole eligible host is `trn21`. Run G0, then G1, then G2, never concurrently. Before every arm,
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

The finalized launcher must construct the argv arrays below from the pending request without shell
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

G0 and G1 use `train_gate_hybrid.py`; G2 uses `train_gate_ordinary.py`. G0 passes
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
