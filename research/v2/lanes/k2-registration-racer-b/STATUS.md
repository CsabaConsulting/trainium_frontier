# K2 registration racer B — own CPU implementation complete

Updated 2026-09-26 01:14 UTC. Implementer B only; independent approval pending.

Selected proposal: `20260926-k2-registered-forward-autograd`, proposal SHA256
`6d9c2f46b1d25cc3507b8fcc7c068f5dbbd3dc392707058b9d7b4de7dbc3c107`.
Isolated branch `codex/20260926-k2-registration-racer-b`, based on `87a34f13`.
No other racing implementation, worktree, notes or results were inspected.

## Implemented

`adapter.py` imports neither Torch nor NKI until explicit `enabled=True`.
The factory constructs one stable six-Tensor forward op per requested dtype,
using `nki_op(mutates_args=())`, and registers its derivative. Its forward
is exactly `F.embedding(ids, weight).to(output_dtype)`. The context saves
only counts, starts, positions and maximums. Backward calls the prebuilt
unchanged consumer on `gradient.contiguous()` plus the four saved tensors,
then returns that FP32 result and five `None` values.

The default native dependency path loads the immutable sibling
`consumer_nki.py` and calls `build_kernel(enabled=True)`; the protected
factory still returns `resident_consumer[2]`. There is no explicit bridge
wrapper, second consumer op, global tensor cache, production import or host
binding. The original `DiagnosticEmbedding` class is a separately named
failure control; its complete class AST is identical to the protected
original checker. Existing native failures remain failures.

## Own checks and raw receipts

Exact command from the worktree root:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python research/v2/lanes/k2-registration-racer-b/test_cpu.py
```

`results/cpu_final.json`: **307 checks PASS, 382 raw records, 4.344648 seconds**.
`results/cpu_final.stderr.log` is empty. Torch is `2.13.0+cpu`; one intra-op
and one inter-op thread; seed42. Shapes are V8/C4/T8, with largest semantic
tensor9x4 (144bytes). The explicit test tensor bound is under64MiB; that bound
does not describe PyTorch library/process RSS. No TorchNeuron/NKI imports.

The installed SDK `nki_op` function is compiled directly from its unchanged,
hash-pinned AST, with the local real `torch.library` dependencies. Tests check
fake registration and actual FakeTensor execution, FunctionalTensorMode
decomposition registration, autograd registration, six-input nonmutating
schema, and protected source hashes. The CPU consumer substitute uses a
per-row tensor recurrence; the oracle independently follows token order in
Python and rounds every addition with IEEE FP32 pack/unpack.

There are108 ordinary/original/candidate route executions over eager and
`aot_eager(fullgraph=True,dynamic=False)`, both dtypes, starting with FP32
unique nonzero upstream. Fixtures include duplicates, a grouped-rounding
discriminator, unused rows, logical-empty metadata, signed zero, changed
valid metadata, and two distinct tangents including noncontiguous inputs.
Each reused compiled candidate sees changing metadata/tangents. Every route
records strict forward/dW byte hashes, finite checks, nonzero count/error,
and all seven input hashes/shapes/dtypes/strides before and after.

Eight additional actual-consumer negative executions (zero/wrong consumer,
each dtype, eager/compiled) produce different dW and are rejected. In the
compiled FP32 sentinel, the zero consumer gives0 nonzeros/max_abs1.125;
the wrong consumer gives max_abs1.0. This establishes local registered
backward participation. It is stronger than the separate16 result-validator
negatives for missing/zero/wrong/nonfinite/dtype/shape/stale results.

Development receipts remain intact: `cpu.json` is the first297-check pass;
`cpu_v2.json` records a failed added discriminator test because its initial
cancellation fixture did not actually separate grouped and sequential
rounding. Only that fixture changed before the final307-check pass; the
adapter stayed byte-identical. Earlier test-source hashes are historical
development versions; `cpu_final.json` binds the delivered test source.

Final SHA256:

- Adapter: `cabdb4e05d8b3beac78d6b8c60fd862a1d257afd2e3ab5809d45f70851cc4349`
- Test: `12d8c1276f192f89fe80b5d9eff40106066fd5955f813cc7c7e62f6589f0307d`
- Final raw JSON: `465bbad22080d0c94515fee6d0b60975fc38b59353b56384016af300795289de`

## Limits and next

This is own CPU evidence, not independent G2 approval. The actual native
environment uses Torch2.11/NKI0.5; native kernel execution, automatic HOP
selection, all20/all4 qualification, cold startup, whole-step speed, full-model
three-step identity, higher-order gradients and BPB are untested/unmeasured.
The CPU original control passes here; this does not reproduce or erase its
recorded native zero-gradient failure. No host/SSH/AWS/network/package install,
training, launch, production edit or submission occurred.

Next: supervisor compares racing evidence, then a different fresh reviewer
receives only the selected diff/spec/receipts and executes an independent
exactness test. Adapter remains disabled. No native assignment is granted.
Undo: leave this directory unused; all existing paths remain unchanged.
