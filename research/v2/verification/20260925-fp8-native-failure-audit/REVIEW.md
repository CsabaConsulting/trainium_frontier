# Independent FP8 native-failure audit

Date: 2026-09-25. **HOLD_NATIVE_BACKWARD_UNRESOLVED.** The archived native
component run failed correctly under its declared gate. No specific production
source defect or bound-`apply` checker defect is established by the available
evidence. In particular, changing the compile entry point is a diagnostic
contrast, not a proven fix. No source edits, native retries, threshold changes,
host access, or timing/quality runs were performed in this review.

## Exact evidence

- Raw archive: `c1ac162c`,
  `research/v2/operator/20260925-trn21-l1-fp8-native-r1/component/`.
- Candidate source: commit `50a901374ae6f76b35b8556d32fd72b1ece28fc6`, SHA-256
  `921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858`.
- Checker: `research/v2/lanes/fp8-projections/check_native.py`, SHA-256
  `7ba967b06834071f33cf9158c1f2a76555e9850d0abc988421b4b870ce1e3bc0`.
- Independently verified all eight retrieved artifact hashes against the
  archive's retrieval manifest. All four rank traces report the same first
  `[256,256,256]` failure. The process exited nonzero; later shapes and prefix
  checks were not reached, and no component PASS receipt was produced.

Observed first-case relative RMS: forward `0.0050534564`, saved-input direct
output `0.0026584163`, saved-weight direct output `0.0038507837`; both dX and dW
are `1.0`, exceeding the unchanged `0.01` ceiling. Compiled/direct forward
error is exactly zero. This proves a backward correctness failure in the tested
native path, not acceptable numerical drift and not a throughput result.

## Trace and representativeness

The checker uses BF16 input/upstream gradient and FP32 master weights requiring
gradients. Its `y.backward(gy)` oracle is `gy @ effective_w` and
`(gy.T @ effective_x).float()`. That matches the source's explicit BF16 STE
backward (source lines 1610–1619), including per-token effective operands inside
the dW reduction. There is no demonstrated wrong scale, transpose, normalization,
or loss-scaling factor in that oracle. The separate direct forward calls are not
backpropagated and therefore should not add to the leaf gradients.

There are two important coverage limits:

1. Checker line 70 compiles the bound `_FP8Projection.apply` with
   `fullgraph=True`. Production calls `apply` inside `Linear.forward` (line 1654)
   and compiles the enclosing model with the default graph-break policy
   (lines 4032–4035). The checker tests the real custom function under the
   Neuron backend, but not the identical production capture boundary. This
   difference alone does **not** demonstrate that its backward is invalid.
2. Checker lines 60–61 obtain `xe/we` from separate direct kernel calls. The
   `saved_x/saved_w` fields at line 76 compare those direct outputs, not tensors
   inspected from the compiled autograd context. `compiled_vs_direct` is only
   a forward-output comparison. Thus those small errors do not prove that the
   compiled backward received nonzero/correct saved operands or upstream G.

The raw log reports neither actual gradient norm/nonzero counts nor compiled
saved tensors/backward graph. All-zero gradients would produce exactly the
observed relative RMS and max errors, but these summary metrics do not uniquely
prove zero gradients or distinguish omitted/custom backward, saved-output
capture, or native BF16 execution failure.

## Independent bounded CPU proof

`probe.py` extracts the exact source definitions by AST without editing them or
loading a model. On Torch `2.13.0+cpu`, one CPU thread, seed42, BF16 8x16 input
and FP32 16x16 weight, the following all return bit-identical forward, dX and dW
against the exact STE oracle, with nonzero gradients:

- eager bound `apply`;
- `aot_eager` compiled bound `apply`, fullgraph;
- `aot_eager` compiled enclosing function calling `apply`, fullgraph;
- `aot_eager` compiled actual production `Linear`, default graph-break policy.

Every dX has128 and every dW256 nonzero entries; each relative RMS is0.
This independently agrees with the Implementer's separate CPU investigation.
It refutes a backend-independent claim that compiling bound `apply` necessarily
drops the custom backward. It does **not** exonerate native capture: the archived
host uses a different Torch/Neuron stack and no Neuron execution occurred here.

The probe also independently regenerates the original seed42 256-square CPU
oracle, without a model: max absolute expected dX is `4.875`, dW `326.0`.
These exactly match the archived native max errors, strengthening the zero-
gradient hypothesis without upgrading it to an observation. Results and eight
archive hash checks are durable in `probe.json`.

Reproduction:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/verification/20260925-fp8-native-failure-audit/probe.py
```

## Minimum useful next validation (not launch authorization)

Keep the failed run as FAIL and preserve all declared tolerances. For an
independently reviewed diagnostic on the same installed native stack, reuse the
first seed42/256-square case and exact source, but exercise the actual
`Linear(..., bias=False, fp8_projection=True)` under production-style enclosing
compilation. Record finite flags, norms/nonzero counts and existing dX/dW errors
before any fail-fast. Contrast with eager `apply` on fresh leaf inputs/weights,
and with the retained bound-`apply` path if needed to identify a capture-boundary
effect. These are small component paths, not full models or timing processes.

If the production-style path also fails, compare its gradients with explicit
BF16 VJPs from native direct effective operands (`G @ we`, `G.T @ xe`) to
separate operand-production from backward-capture/execution issues. Inspect
compiled saved outputs/backward graph only as needed to resolve that contrast;
graph availability is not a new gate. A production-style PASS on this one case
would localize the original failure but would not complete the component gate:
retain the existing remaining layouts, edge cases and prefix/padding checks.

Do not infer a source patch, NKI API migration, threshold relaxation or timing
eligibility from the present evidence. The native component requirement, matched
whole-step >=3% requirement and later fixed-budget quality requirements stand.
