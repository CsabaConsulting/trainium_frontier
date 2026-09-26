# K2 BF16 production implementation scoping

Status: **INCOMPLETE / HOLD**. Implementer only. No independent source review,
native component, G2, timing, training, quality, adoption or submission result.
The requested real-shape producer + consumer has **not been implemented**.
Frozen `workflow/proposals/research-v2/train.py` is byte-unchanged from
7419927c7417b5666a984ec9c2b849feadabaaa9, SHA256
36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb.
No patch should be described as ready for fresh diff review or a host component.

## Done

- Isolated sparse worktree on root filesystem:
  `/home/poojith-devan/Desktop/Trainium-k2-bf16-production-impl`, branch
  `codex/k2-bf16-production-impl`, based on the selected frozen source.
- Read selected proposal and Supervisor selection plus fresh contract review.
  That review supports the mathematical domain only, not implementation.
- Implemented a bounded, synthetic CPU model of stable count/prefix routing,
  immutable per-call ID snapshots, exact sequential FP32 consumer, and explicit
  semantic metadata validation. This is **not a device implementation** and
  cannot be used as a production CPU side channel or fallback.
- Own local tests cover both subnormal signs, normal-to-subnormal cancellation,
  both order witnesses, signed zero inputs, overflow/NaN/Inf diagnostics,
  interleaved IDs, all-one-row skew, unused rows, loader-buffer reuse, separate
  WTE/VE7 outputs, microstep boundary witness, and corrupt metadata rejection.
  Nonfinite results remain FAIL_NONFINITE_DIAGNOSTIC, never equality passes.
- Integer-only full-shape cost estimate included; largest tensor has514 elements.
- `build_native()` default raises EXPERIMENT_DISABLED; even explicit enabled=True
  raises PRODUCTION_METADATA_VALIDATION_AND_NKI_COMPOSITION_UNIMPLEMENTED.
  No new training flag or scientific-source edit was made.

## Blocked / limits

The missing implementation is the actual device producer with fail-closed
semantic metadata validation, followed by legal real-shape ordered consumption.
In-range corrupt positions can silently select the wrong tangent while every
DMA address remains valid. Bounds checking alone is insufficient. The tiny
CPU validator cannot be transferred to production Python conditionals or ID
copies. I have not established the corresponding legal NKI composition.

Exact local capability check with the existing Python environment returned:
`{'nki': False, 'neuronxcc': False, 'torch_neuronx': False}` via
`importlib.util.find_spec`. No install or host access was performed.
This is a **local tooling/evidence limitation**, NOT evidence that NKI or Trn2
cannot implement the design. No compiler error or native unsupported-operation
claim exists. Public documentation supports the ingredients described below;
their composition still needs code and qualification. This is a partial
implementation handoff, not fulfillment of the requested production prototype.

No tolerance, value-based fallback, tangent narrowing, grouped gradient sums,
NaN equivalence, hidden CPU-ID route, or microstep concatenation was introduced.
Original unrestricted FP32 failures and protected kernel2c829/adapterd761 remain
untouched and failed as applicable. There is no meaningful native compile-time
or speed estimate; those fields are null, not zero.

## Next

Implement the device producer/semantic validation and runtime loops before
touching the selected scientific source. The docs support flattening prefix
addresses `id*256+tile` to avoid simultaneous scalar/vector DMA offsets; all
such integers are below2^24. Device indices must be validated before indirect
reads, without silently truncating invalid int64 IDs to int32. Runtime errors,
validation output dependencies, metadata lifetime, and both symmetric512-channel
owners need actual source treatment; a fake Python simulator is not proof.

Only after a complete default-OFF train.py candidate and its own local tests:
fresh independent diff/spec/raw-receipts review, then a separately proposed
real-shape component-first native gate. No assignment, label, launcher, host,
cache, training, data, evaluator or controller-state action is included here.

## Reproduction

From this worktree:

```
/home/poojith-devan/venv/bin/python -B research/v2/lanes/k2-bf16-production/test_model.py
/home/poojith-devan/venv/bin/python -O -B research/v2/lanes/k2-bf16-production/test_model.py
```

Raw receipts are raw-normal.json and raw-optimized.json. These are supplied
Implementer tests only, never an independent verifier receipt.
