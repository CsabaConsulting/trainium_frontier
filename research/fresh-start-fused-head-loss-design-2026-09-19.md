# Vocabulary projection and loss: eliminate intermediate traffic

Proposed on 2026-09-19, after the exact head-gradient-only change improved a
whole step by only 0.53%. This is a bounded follow-up to the user's broader
fast-function ideas: avoid computing, storing or rereading an intermediate
instead of merely replacing an individual function. It is not a measured gain.
The immediate priority remains explaining the 17.74% whole-model FP8 regression
with the paired wave89 caller traces.

## Evidence and compatibility

[Apple's Cut Cross-Entropy paper](https://machinelearning.apple.com/research/cut-your-losses)
fuses classifier matrix multiplication and loss reduction without materializing
the complete logits matrix in global memory. Its GPU measurements do not predict
Trainium speed. The implementation also has optional gradient filtering, which
must be separated from the memory/layout idea.

Reviewed source commit: `3de376c106a1916bc5e1b619f9c77c87a461ee1c`.
The [API at that commit](https://github.com/apple/ml-cross-entropy/blob/3de376c106a1916bc5e1b619f9c77c87a461ee1c/cut_cross_entropy/linear_cross_entropy.py)
supports softcapping and explicit filtering/accumulation controls. The
[README](https://github.com/apple/ml-cross-entropy/blob/3de376c106a1916bc5e1b619f9c77c87a461ee1c/README.md)
distinguishes variants with improved accumulation and disabled filtering.
Start our investigation with all vocabulary contributions retained, FP32
accumulation and the existing softcap15 objective. Do not attribute a filtered
gradient's speed to exact work reuse.

AWS also documents an experimental
[NKI cross-entropy kernel](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/cross-entropy.html),
with [source in nki-library 2.32](https://github.com/aws-neuron/nki-library/blob/2.32/src/nkilib_src/nkilib/experimental/loss/cross_entropy.py).
It takes already-materialized logits, supports chunked online reduction and
LNC2, and defaults to BF16 internal arithmetic and in-place backward. It is not
a fused vocabulary projection, and its documented backward is ordinary CE,
without our tanh-softcap derivative or arbitrary upstream loss gradient.
Those defaults do not meet the current contract. Read its tiling as a reference;
do not install a new SDK or substitute it into training without new gates. Our
installed NKI is 0.5, so newer APIs require an explicit compatibility review.

## Current contract that a new kernel must preserve

For token rows `X`, head weights `W`, target index `y`, and softcap `s=15`:

```
z = X @ W.T                 # BF16 operands, FP32 accumulators
u = tanh(z / s)
a = s * u
loss[t] = log(sum_v exp(a[t,v] - a[t,y[t]]))
dz[t,v] = upstream[t] * (softmax(a)[t,v] - 1{v=y[t]}) * (1-u[t,v]^2)
```

Wave83 established that the compiled original head retains FP32 accumulator
values into CE. Explicit BF16 rounding of `z` changes the objective seen by the
loss and failed the exact gate. Retain FP32 logits internally. Preserve the
target-relative loss calculation, including confident-target cases; an
unguarded `logsumexp(a)-a_y` rewrite can lose small positive losses.

The head consumes BF16 gradient values at its existing backward cast boundary;
the separately gated gradient-only CE variant establishes that local boundary.
FP32 master gradients and repeated microbatch accumulation must still be routed
exactly once. Do not infer whole-head equivalence from a pre-rounded-logit test.

At our current local geometry, `16384 * 8192 * 4` is 536,870,912 bytes per FP32
logits tensor per rank. This is an allocation size, not an observed traffic count
or a predicted speedup. Recomputing a head projection adds substantial work, so
avoiding its allocation may still lose time for an 8192-word vocabulary.

## Experiments, in order

1. **Attribute current cost.** Use the new native caller trace and a complete
   head/CE component timer. Separate device execution from host gaps and retain
   missing-DMA-metadata warnings. Do not sum overlapping instruction durations
   into a fictitious wall-time share.
2. **Forward prototype with ordinary backward saves.** For 128 token rows,
   stream 512-wide vocabulary weight tiles through SBUF, retain FP32 projection
   values, then apply the existing target gather and 512-wide CE reduction.
   First retain the ordinary logits save so backward has the unchanged input.
   This isolates fusion from recomputation and new gradient layouts. Charge
   weight layouts/preparation; measure any compiler spills in the real caller.
3. **Recompute prototype.** Save input, weights and FP32 loss statistics, then
   recompute vocabulary tiles during backward. Form the existing BF16 `dz`
   boundary and accumulate `dX` and `dW` in FP32. Keep all vocabulary terms.
   Account for both cores' partial-weight-gradient reduction and final master
   writes. Charge the extra head GEMM; no claim that lower memory means faster.
4. **Optional approximation, only separately.** Gradient filtering, polynomial
   softcap/exp composition, or compressed saved statistics each changes another
   contract. They require an independent error analysis and quality experiment,
   and must not be hidden inside the first reuse/layout comparison.

First compare projection accumulators against an independent FP64 product and
a predeclared accumulation-error bound. Validate softcap, target-relative loss,
saved statistics, `dz`, `dX` and `dW` separately, including zeros, cancellation,
outliers, confident targets and arbitrary upstream gradients. A changed GEMM
reduction order is not presumed bitwise identical. Preserve exact diagnostics
against the compiled original even when mathematical-equivalence bounds pass.

Then require installed actual-head tests, two-microbatch FP32 accumulation,
parameter/cache refresh, serialization, causality, batch independence and
repeated predictions. Time complete head forward/backward and complete model
updates after clean pre/postflight. Only a complete-step improvement of at least
3%, with all required numerical gates satisfied, advances to a 30-minute
quality comparison. No quality run follows a forward-only component result. The caller source
aggregates ordinary Linear operations and does not isolate the head cost. The
earlier complete head/CE component14.99ms establishes a measurable target, not
a whole-model forecast. A bounded forward prototype now tests fusion and
transposed-weight preparation directly.


## First prototype, 2026-09-19 02:21 UTC

`head_loss_fused_nki.py` retains128 token rows of FP32 logits and softcapped
values in SBUF. It streams BF16 weights from a contiguous[C,V] transpose;
it does not retain a16MiB weight cache beside the vocabulary row tiles. The
complete FP32 logits tensor is saved for ordinary backward. Target gather
and512-wide exp reductions preserve the previously gated CE algorithm.
There is no claim that streamed weights will outperform native GEMM.

`head_loss_fused_probe.py` declares gamma(C)*sum(abs(products)) FP32 accumulation
bounds before execution, without BF16 output-rounding allowance. Exact loss/lse
on frozen logits, nonnegative finite losses, repetition, row independence and
input integrity are required. Fixtures include zero/tiny/random values,
cancellation, outliers and confident correct/wrong targets. Exposed native
output precision is audited; a rounded baseline cannot become speed evidence.
The timer charges one transpose plus two forwards including logits saves.
Backward, autograd and optimizer costs are absent; no quality promotion follows.


## Exact saved-intermediate follow-up

The existing CE backward needs `u=tanh(z/15)` and saved logsumexp, not raw logits
itself: `dz=g*(exp(15*u-lse)-one_hot(y))*(1-u*u)`. After the first FP32 logits
prototype passes, test saving the FP32 `u` tile in place of the equally sized
raw-logit save. Fused head/CE can write it directly. This avoids a backward
tanh evaluation without gradient filtering or a new approximation, provided
saved and recomputed hardware `u` values match exactly. The complete head's
input/weight gradients still need their independent gates and original cast
boundaries; this algebra does not establish whole-autograd correctness.

For the original nonfused head, writing a separate `u` tensor adds a full HBM
write and may lose. For a fused producer it replaces an existing save of the
same size. Time the complete forward/backward with all saves, not just the
removed Scalar instruction. Prototype status: proposed; not implemented.


## First hardware result: preserve the original failed sign gate

Source99 gate ends clean02:30:35 with a failed nonnegative-loss check on the
confident-correct fixture (vocabulary1024). Candidate logits/loss/lse match the
compiled native head and exact CE on frozen logits. All projection bounds pass.
The negative value is only about5.8e-8; the independent FP64 loss is tiny and
positive. Zero/tiny/random/cancellation/outlier fixtures preceding it pass.
The sign assertion remains failed, unmodified; no speed result exists yet.

Next diagnostic exposes target softcap, affine argument, exponential,
denominator and logarithm, plus runtime exp(0)/log(1) controls. It must match
the original CE loss/lse exactly. This distinguishes a fusion error from the
reference primitive/reduction behavior before any compatibility-contract change.
No production clamp or approximation change is proposed to hide the failure.
