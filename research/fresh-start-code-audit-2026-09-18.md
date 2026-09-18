# Fresh-start code audit — 2026-09-18

Scope: current `train.py`, the fixed `prepare.py` evaluation/data contract, selected
scratch diagnostics, and read-only inspection of the installed native PyTorch
backend. No device probes were run by this audit while the two training jobs were
active. Line references below refer to engineering commit `ca63567` unless stated.

## Highest-priority finding: the default attention path now forces graph breaks

`train.py:2790` unconditionally decorates `nki_attention` with
`@torch._dynamo.disable`. `CausalSelfAttention.forward`, line 2937, calls this
wrapper whenever the mask is purely causal, including when `USE_NKI_ATTN=False`.
Consequently the default path leaves the compiled model around every attention
call. The flag's fallback preserves the eager SDPA formula; it does **not** preserve
the pre-existing compiled execution graph.

This matters beyond Python overhead. On the installed native backend,
`torch_neuronx/neuron_dynamo_backend/decompositions.py:532` checks eligibility for
its own NKI flash attention. The default is enabled by
`TORCH_NEURONX_ENABLE_NKI_SDPA`, and the baseline satisfies the stated conditions:
sequence dimensions divisible by 512, head dimension at most 128, no additive
bias, no dropout, and batch times head count at most 512. The decomposition calls
`wrapped_flash_fwd = wrap_nki(attention_cte)` and
`wrapped_flash_bwd = wrap_nki(attention_bwd)` (lines 59–60), with a grid matching
logical Neuron cores.

Follow-up source inspection corrects an initially plausible interpretation: the
native eager SDPA path also has an NKI-first implementation. In
`csrc/aten/SdpBackendSelection.cpp:44`, 4-D Neuron SDPA selects the overrideable
backend. `python_ops/scaled_dot_product_fused_attention.py:364` orders its NKI
implementation before MLIR, reuses the same eligibility helpers, and its lower
function calls the traceable NKI forward/backward wrappers (`:92`, `:141`). Thus
the graph break is proven, but **loss of all NKI attention is not proven**. Both
paths may execute NKI with different graph boundaries, layout work, scheduling and
surrounding fusion. Actual kernel/profile evidence is still needed.

The backend catches errors from the NKI attention call and falls back to its
manual decomposition (forward lines 675–688, backward lines 1027–1057). The manual
backward explicitly converts Q/K/V/dO to FP32 (`:842`), and the caller supplies
`mixed_precision=True` (`:1052`). Its score, dV, dP, dQ and dK matrix products are
therefore plausible targets for compiler BF16 autocasting **if that fallback is
the path actually taken**. This is a source-supported hypothesis, not a measured
explanation of the current autocast results.

Smallest experiment: call `nki_attention` only when `USE_NKI_ATTN=True`; otherwise
call `F.scaled_dot_product_attention(..., is_causal=True, scale=...)` directly from
the attention module, as before the custom attention integration. Keep the disabled
wrapper for the custom-kernel arm. Compare Dynamo graph breaks first, then the
four-rank forward/backward profile and a short training/causality run. Cross the
direct-vs-wrapper comparison with compiler autocast none vs BF16. This can distinguish
an execution regression from a useful lower-precision algorithm.

Implemented this comparison as opt-in `--compile-sdpa-direct`, default **False**,
with a snapshot/environment override and startup flag print. With
`USE_NKI_ATTN=True`, its existing disabled wrapper retains precedence; explicit
attention masks keep their original direct SDPA path.

`research/scratch/fresh_start_attention_graph_check.py` passed on trn22 using CPU
only (`TORCH_DEVICE_BACKEND_AUTOLOAD=0`, one CPU thread). Sixteen real-attention
module cases covering FP32/BF16, default/custom scale, causal/explicit mask and
custom-wrapper precedence had bit-exact outputs and input/parameter gradients.
Dynamo reported historical **2 graphs / 1 break**, direct **1 graph / 0 breaks**,
and custom-wrapper **2 graphs / 1 break**. Direct fullgraph capture with
`backend='eager'` also passed. This validates routing and graph capture, not Neuron
compilation, kernel selection or a speedup.

Historical evidence should be reinterpreted: `trn2-experiment-results.md:6377–6425`
established that the existing RoPE/CE NKI calls were already traceable with one
graph and zero graph breaks. Those measurements predate the disabled attention
wrapper. A historical claim of zero graph breaks is not evidence about today's
model.

## Current dtype and kernel map

| Component | Source behavior | Implication |
| --- | --- | --- |
| Model residuals and linear inputs | `compute_dtype_for` selects BF16 on Neuron (`:1400`); embedding result converts to that dtype (`:3785`) | Model compute was already substantially BF16 before compiler autocast was enabled. |
| Linear weights | FP32 parameters are cached/cast to each input dtype (`:2588–2619`) | Dense forward GEMMs are ordinarily BF16 at the PyTorch boundary; FP32 master weights remain. |
| Final logits | BF16 linear output is converted to FP32 (`:3842`) | Softcap/CE sees FP32 logits even though the head matrix multiply consumed BF16. |
| RoPE + Q/K RMSNorm | NKI pair enabled, subject to shape/device/contiguity guards (`:2428`, `:2774`) | This is an existing successful fusion, not missing work. |
| Softcap + cross entropy | NKI pair enabled for contiguous FP32 logits with row count divisible by 128 (`:2514`) | Replacing generic CE alone repeats an implemented optimization. |
| Muon | `USE_COMPILE_MUON=True` (`:3928`); it supersedes the NKI wrapper fusion in `main` | The default optimizer uses a compiled math body, not the advertised `USE_NKI_MUON=True` path. |
| Muon Newton–Schulz | Explicit `g.bfloat16()` before the iterations (`:3999`, also eager `:4645`), FP32 state/update scaffolding | Baseline Newton–Schulz GEMMs are already BF16. Do not describe global BF16 autocast as a proven FP32→BF16 NS transition. |
| Hashed n-gram tables | BF16 parameters and `zeros_like(p)` second moments (`:4448`) | Table optimizer state precision deserves direct numerical examination. |

There is a single broad `try/except Exception` around all NKI definitions
(`:2413–2415`). A failure sets `_HAS_NKI=False` and silently falls back. Future
profiles should record `_HAS_NKI` and the actual selected branches once, alongside
effective compiler flags and graph counts. Enabling a flag is not proof that its
kernel executed.

## The installed backend has no per-region compiler-flags option

Source root inspected read-only on trn22:
`/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx`.

* `neuron_dynamo_backend/backend.py:180` reads `options['model_name']`;
  `:259` reads `options['use_tensorizer_backend']`. There is no compiler-flags,
  auto-cast, or precision option routed to compilation.
* `neuron_dynamo_backend/compile.py:148` passes only content key, MLIR bytes,
  collective presence and tensorizer choice to `_C.compile_graph`.
* `csrc/core/NeuronBindings.cpp:1894` exposes that same four-argument interface.
* `csrc/core/KernelExecution.cpp:113` builds compiler arguments as
  `--model-type transformer --auto-cast=none` followed by process-wide
  `NEURON_CC_FLAGS`. They are captured by the compiled-kernel constructor and
  included in its cache key.
* `neuron_dynamo_backend/settings.py:18` explicitly leaves torch.compile option
  configuration as a TODO.

Therefore `torch.compile(..., options={'auto_cast': 'none'})` does **not** implement
a safe per-Muon override here. Swapping the environment around lazy compilation
is not a supported per-region API and is unsafe to assume under asynchronous
compilation. Preserve the installed package. If scoped precision proves necessary,
the practical code-local route is explicit math/kernel choices for the identified
matrix products, validated against traced HLO and hardware. First locate the actual
FP32 matrix products; do not pre-emptively rewrite Muon.

## Optimization and objective checks

### Local clipping is a distinct algorithm; an opt-in comparison is implemented

Before this audit, the loop clipped each rank's local gradients before averaging
them. In general,

`mean(clip(g_rank)) != clip(mean(g_rank))`.

The difference can change the update direction, not just its magnitude. Clipping
the averaged gradient is a reasonable untested quality experiment. It is not a
proven explanation for the LNC1/LNC2 quality discrepancy: the old no-clipping
comparison retained almost all of that gap (`trn2-experiment-results.md:2213–2234`).

Implemented `--clip-after-reduce` / `CLIP_AFTER_REDUCE` with default **False**, a
checkpoint toggle snapshot, environment override, startup logging and a shared
`clip_and_sync_gradients` helper (`train.py:4835`). Both compile-only and real
training paths use it. Loss unscaling and accumulation finish before it; the
separate n-gram clipping group is preserved. Flat-buffer clipping remains supported
when there is one group. Deferred Adam updates (`--adam-every-n > 1`) are rejected
with this option because unreduced gradients would otherwise enter an averaged
gradient norm. Flat buffer plus separate n-gram clipping also fails explicitly.

`research/scratch/fresh_start_clip_order_check.py` performs real two-rank CPU Gloo
collectives, implementing AVG via SUM/division for Gloo compatibility. Twenty cases
per rank cover backward accumulation, FP32/BF16 table gradients, separate clipping,
flat buffers, both orders and clipping disabled. It compares actual SGD updates
against independently averaged/clipped gradients, checks the historical path
bit-exactly, and rejects unsupported combinations. Root reports it passed on both
instances. `py_compile` and `git diff --check` also passed locally.

Recommended comparison: matched compiler/runtime, same seed and full 1800-second
budget, off vs on, with no other quality changes. If promising, repeat another seed
and inspect clipping frequency/norm distributions before retuning the threshold.

### Byte-weighted token loss is not the BPB objective

The fixed evaluator computes

`BPB = sum(NLL_i for valid positive-byte targets) / (ln(2) * sum(bytes_i))`

in `prepare.py:628–649`. For a fixed evaluation token stream, the denominator is
constant with respect to model parameters. Thus ordinary summed token NLL is the
correct objective, apart from zero-byte target masking. Multiplying each token
NLL by that token's byte count (`train.py:2514–2565`) optimizes a different
objective. Mean-normalizing the weights does not fix that mismatch, and normalizing
by mean vocabulary byte length does not ensure constant expected loss scale under
the empirical token distribution.

Keep `LOSS_BYTE_WEIGHT=False`. If testing alignment with scoring, exclude zero-byte
targets from the training numerator while retaining their tokens as model inputs.
This is likely a small effect and must be evaluated as a modeling change; do not
modify `prepare.py` or claim that per-byte weighting is required for BPB.

### BF16 RMSProp second moments can fail to decay

The default tables are BF16 and their second moments are allocated with
`zeros_like(p)` (`:4448`). With `NGRAM_VE_BETA2=0.999`, the update is
`v.lerp_(g*g, 0.001)` (`:4478` and `:4491`). For an untouched row, the mathematical
update is `v *= 0.999`, but a 0.1% decrement is below half an ordinary BF16 ULP for
normal positive values. Round-to-nearest can leave `v` exactly unchanged on every
zero-gradient step. Frequent rows can also accumulate small squared-gradient changes
poorly. The implementation is therefore not automatically equivalent to a
0.999-decay RMSProp process merely because the input scalar has that value.

This is an audit hypothesis until verified on the installed CPU/device kernels and
rounding mode. Root is pursuing it independently. The smallest useful experiment
separates **table storage** from **optimizer-state precision**: keep table parameters
BF16 and make second moments FP32. Avoid using an all-FP32-table arm as the sole
test, since that confounds memory traffic, gradient dtype and parameter rounding.
Record time as well as quality; four 1,048,576×128 FP32 moments have a material
memory-bandwidth cost.

Follow-up implemented `--ngram-rms-fp32`, default **False**, explicitly changing
**second-moment state and optimizer arithmetic precision** while retaining the
table/gradient dtype. Gradients cast before squaring; moments, denominators and
updates stay FP32, and the final subtraction writes the existing BF16 parameters.
Foreach/scalar and tensor/float LR paths are supported. The CLI rejects AdamW table
mode, and a populated/restored moment with mismatched dtype fails before the update
instead of silently losing precision. This matters because the installed
`Optimizer.load_state_dict` casts the FP32 moment back to BF16 parameter dtype.
Scored training starts with fresh optimizer state.

`fresh_start_rms_fp32_check.py` passed CPU-only on trn22 across 16 configurations
and 32 sparse/outlier/zero/None-gradient steps each. Default OFF remained bit-exact
to the historical update; FP32 state matched the independent recurrence within
4.77e-7 absolute error on the tested outlier values. Gradients retained their
original dtype and contents, and AdamW was unchanged. After 256 zero-gradient
steps, the old moment was **1.0×** its initial value; the FP32 moment was
**0.77404308×**, versus mathematical **0.77404282×**. CPU evidence is saved in
`research/run-logs/fs-rms-fp32-cpu-trn22.summary.json`. Native mixed-dtype foreach
behavior, throughput and full-budget quality still require hardware validation.

## Data exposure is not exactly the reported epoch counter

`prepare.make_dataloader` best-fit packs documents. When none fits the remaining
space, it pops the shortest document, consumes its prefix, and discards the rest
(`prepare.py:509–512`). This behavior is also copied into
`make_dataloader_requeue`. The training loader packs `PACK_FACTOR * SEQ_LEN` and
then reshapes to shorter independent rows (`train.py:6338` onward), so the pack
boundary and model context boundary are different objects. The evaluator uses a
finite continuous stream that covers selected documents without this truncation
(`prepare.py:521` onward).

Consequences: the public raw corpus token count is not the number of unique
training tokens actually presented per epoch; the buffer can read into the next
epoch before earlier documents have all been consumed; changing pack factor can
change order/truncation and context boundaries as well as throughput. Shuffling row
groups alone leaves those differences intact. This does not establish that the
packer is the plateau's cause.

Smallest data audit: count per-rank document bytes/tokens read, consumed, dropped,
and replayed, at the actual pack geometry. If discarded tails are meaningful,
compare a train.py-local continuous/remainder-preserving packer against baseline
at matched tokens/steps and at full wall time. Treat it as a changed data sampler;
the fixed evaluator stays untouched. Full-corpus token counting alone does not
answer this question.

## Measurement gaps and NKI priority

`research/scratch/phase_timer.py` defaults to n-grams disabled, mult=8 and dim=0
(`:92–108`) and explicitly overwrites train.py globals (`:143–148`). The current
default is n-grams enabled, mult=128, dim=128. Always provide the real configuration
or use a harness that inherits it; a bare historical phase timer is not a current
submission profile. Its aggregate output should also include every effective flag
that affects geometry and graph partitioning.

The first high-value NKI task is proving that the native optimized attention path
actually executes, fixing avoidable graph breaks, and profiling what remains. A
new hand-written attention kernel is not automatically faster: the current custom
backward recomputes probabilities/output and repeatedly reads/writes FP32 dK/dV HBM
scratch inside query/KV loops (`train.py:2184–2368`). It is a correctness prototype,
not evidence against optimized tiled attention in general. Its wrapper also
materializes Q/K/V/gradient layouts and allocates five zero-filled outputs/scratch
tensors (`:2401–2410`).

After a current end-to-end profile:

1. Prioritize any dense FP32 attention backward GEMMs exposed by fallback; direct
   use of the backend's existing NKI kernel is the smallest candidate.
2. For sparse n-gram work, separate backward scatter, dense optimizer traffic and
   communication. A true sparse/lazy RMSProp update with last-touched timestamps
   could avoid scanning all rows, but exact handling of duplicate hashes,
   all-rank aggregation and intervening moment decay is essential. This is a
   substantial algorithm/kernel task after the state-precision check.
3. Consider fused linear-plus-loss only if the current profile shows meaningful
   head/logit traffic. Softcap+CE alone already exists; a head+loss kernel must also
   provide dX and dW while preserving the trusted logits-only eval interface.
4. Treat a scaled FP8 MLP/GEMM island as a separate moonshot, with per-operand
   scaling, representative activation/gradient distributions and BF16/FP32
   accumulation where required. Compiler-wide FP8 and torch._scaled_mm
   negatives do not settle a correctly scaled NKI implementation. Start with a
   measured heavy matrix product, not an all-model conversion.

Old `research/scratch/fp8_full_model.py` is not a reusable current-model benchmark:
it constructs vocabulary 50,257 and width `DEPTH * ASPECT_RATIO` without the current
round-up, and sets a historical `USE_FP8_LINEAR` global absent from today's model.
Historical recorded results must be tied to their original source revision; running
the old script against the present `train.py` would not test the advertised arm.

No single source finding proves a sub-1.0 result. The concrete fresh-start sequence
is: restore/verify the intended attention graph, localize the compiler BF16 change,
test the already-evidenced wider model under recovered throughput, then isolate
gradient clipping and table-state precision. These are falsifiable fundamentals
with much stronger grounding than another broad architecture sweep.
