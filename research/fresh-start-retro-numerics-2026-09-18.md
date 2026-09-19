# Retro numerical techniques on Trainium

Research started 2026-09-18; hardware evidence updated at23:39UTC.
The user's scope is **all useful functions and arithmetic transformations**, not
only square root or lookup tables. The objective remains better BPB within the
fixed training time. A faster primitive alone is not a training result.

## What transfers

The useful family includes range reduction, polynomial and rational
approximation, exponent-bit manipulation, Newton refinement, precomputation,
symmetry, a representation chosen to make indexing cheap, and eliminating
work through an invariant. Multiplication/addition may still use FP32; the
trade is an expensive operation or data movement for cheaper arithmetic.

The [original Doom tables header](https://github.com/id-Software/DOOM/blob/master/linuxdoom-1.10/tables.h)
declares the sine, tangent and angle tables and reuse of sine storage for
cosine. [Quake III's source](https://github.com/id-Software/Quake-III-Arena/blob/master/code/game/q_math.c)
contains the well-known inverse-square-root bit seed followed by Newton
refinement. These are distinct examples. For this accelerator, the hardware
estimate is also a candidate seed; the historical integer constant need not
be the fastest or most accurate starting point.

[David Given's Bogomandel implementation account](http://cowlark.com/2018-05-26-bogomandel/index.html)
is particularly relevant: algebra replaces a general product with squares,
then the number representation makes each value useful as a table address.
That suggests looking for identities and layout changes together. Replacing
Trainium matrix multiplies with square-table accesses would have a very
different cost balance and has no demonstrated advantage.

[David Aramant's Buddhabrot work](https://davidaramant.github.io/post/the-buddhabrot-part-3/)
uses domain tests and repeated-state detection to avoid iterations. The
transferable idea is proving that work can be omitted. Fractal escape and
periodicity tests themselves do not apply to ordinary dense transformer
activations. Likewise, the fixed-point Mandelbrot articles show that changing
representation has scale-dependent costs; they do not establish integer
arithmetic as universally faster than floating point.

Modern primary references:

- [Schraudolph's 1999 exponential approximation](https://nic.schraudolph.org/bib2html/b2hd-Schraudolph99.html)
  uses the IEEE754 representation to approximate exponentiation without a
  conventional transcendental routine. It is a lead for range/bit arithmetic,
  not a reason to accept its raw approximation error for training.
- [SLEEF](https://sleef.org/) develops vectorized elementary functions with
  explicit accuracy choices and high-precision reference checks. Its focus
  on reducing branches and gathers is relevant when comparing tables with
  arithmetic on this machine.
- [Sollya fpminimax](https://www.sollya.org/sollya-5.0/help.php?name=fpminimax)
  considers representable polynomial coefficients. Optimizing an FP64
  polynomial and then blindly rounding its coefficients is not the complete
  numerical design problem.

## Hardware changes the trade

AWS documents [piecewise polynomial Scalar Engine activations](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.api.shared.html).
It also documents different precision/throughput for Scalar versus Vector
reciprocal and Scalar versus GpSIMD inverse square root. Scalar functions have
limited ranges: sqrt[2^-116,2^118], rsqrt[2^-87,2^97], reciprocal
±[2^-42,2^42]. Inputs outside their supported ranges can produce invalid results.
This makes both engine selection and range handling part of the experiment.

My inference: a hand polynomial can still help if it avoids another operation,
reuses intermediates, shifts work away from a busy engine, or improves a fused
pipeline. A standalone polynomial with more instructions may lose to the
existing hardware approximation. We will measure this instead of assuming a
CPU speedup transfers.

The installed NKI0.5 API supports zero-copy tensor dtype views and bitwise
operators, useful for exponent/mantissa manipulation. Tensor-scalar arithmetic
converts integer inputs to FP32 internally; that distinction matters when
claiming an integer-only fast path. Full-tile Horner products also need explicit
tensor operations; they cannot all be counted as one scalar FMA instruction.

For tables, [nc_n_gather](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_n_gather.html)
selects within each partition. A naive65,536-entry FP32 table replicated over
128partitions occupies32MiB **per function**, before working buffers. This is
our storage calculation, not a measured allocation. Symmetry, restricting the
active interval, interpolation and polynomial tails could reduce it.
[local_gather](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.local_gather.html)
has different grouped-index semantics; it is not a transparent shared cache
that makes that replication cost disappear.

## Audit of existing code: avoid repeating completed work

- RoPE already precomputes sine/cosine in `GPT._precompute_rotary_embeddings`.
  Replacing runtime sine calls is therefore not a new training-loop saving.
- The existing NKI softcap/CE already omits row-max subtraction using the
  softcap bound. It shifts by the target logit, avoiding subtraction of two
  close numbers for confident predictions. It also chunks exponent sums to
  control FP32 accumulation error. Preserve those properties.
- Norm's Scalar rsqrt plus one Newton correction already passed numerical
  gates but lost in the complete normalization. A normalization has one
  inverse statistic per token; bulk reciprocal throughput is not its dominant
  issue. Packed statistic DMA helped, but the native operation still won.
- Causal attention, fused softcap/CE, compiled Muon and BF16 weight caches are
  already enabled in the current dense reference. A proposal to merely enable
  them would duplicate existing work.

## Prioritized experiments

| Target | Candidate | Why it may help | Required check |
| --- | --- | --- | --- |
| AdamW denominator | Scalar sqrt and Scalar reciprocal plus Newton correction | Replace two slower bulk operations without more SBUF buffers | FP64 moment/weight stages, zeros/tiny values, schedules, checkpoint/cache tests, complete optimizer/step timing |
| exp in bounded loss | Range-reduced degree5/7 polynomial on Vector Engine | Test arithmetic substitution and moving work off Scalar Engine | Boundary/positive/exp(0)/repeat checks, normalized probability and gradient sensitivity, then fused loss timing |
| Softcap + exp + derivative | Approximate the composition or compact BF16-domain tables with polynomial tails | Remove repeated transforms and intermediate traffic | Actual input dtype, all finite BF16 codes or certified interval, target-shift stability, backward and full-model tests |
| log near1 | log1p-style reduction or a small-interval polynomial | Avoid cancellation/approximation error for very small losses | Confident correct/wrong tokens and FP64 reference; row-level frequency may make speed impact small |
| tanh/sigmoid/SiLU | Rational/minimax alternatives; share exp/reciprocal intermediates | Useful where several requested outputs reuse the same value | Saturation, sign/symmetry, derivative error and denominator range; architecture changes labeled separately |
| Matrix/optimizer pipeline | Reuse data and eliminate repeated work | Potentially larger savings than a faster scalar function | Exact state/weight gates and whole-step measurement |

The first two probes are implemented in scratch. No proposed exp/table/tanh
replacement has been installed into training. A coarse hard-sigmoid or other
large activation change is an architecture experiment, not an equivalence
preserving kernel replacement, and needs its own quality comparison.

### AdamW arithmetic probe

`adamw_fused_fp32.py` now has explicit math modes:0 existing GpSIMD-rsqrt/Vector-
reciprocal;1 Scalar sqrt;2 Scalar reciprocal with one Newton step;3 both;
4 power-of-two-scaled Scalar rsqrt plus one Newton correction and mode2 reciprocal;
5 two rsqrt Newton corrections;6 quadratic inverse-root correction.
The step is `y <- y*(2-d*y)` for denominator d. A dead gradient tile is reused
as workspace; gradient HBM remains unchanged.

Scalar sqrt's minimum input is2^-116. Clamping smaller variances to that
endpoint changes the square root by at most2^-58, about3.47e-18, half an FP32
ULP of the model's1e-10 epsilon. This is a bounded AdamW-denominator approximation,
**not** a correct general square-root routine at zero. The unchanged numerical
gates decide whether the complete update is accurate enough. Clipped gradients,
zero-initialized variance and the current betas keep the training values far
below the upper hardware range; unsupported uses need separate range handling.

Wave64 mode3 fails one8M-table first-step weight at gradient
-7.873063978536265e-9. Its error1.53044e-7 exceeds the unchanged1.50674e-7
bound. This reproduces the previously measured native Scalar-sqrt error,
far above the lower clamp; moments and finite checks pass. The tiny-gradient
failure is not explained by zero handling or NaNs.

Mode4 fixes that small-input path but fails12 weights in the large-gradient
case. The worst recorded update has gradient-1194.599609375 and initial
variance1034481.4375. One Newton correction is not sufficient there. The
documented activation input range does not promise uniformly small relative
error. `rsqrt_refinement_probe.py` now exposes seed, square/product and two
successive corrections across180 exponents and4096 mantissas per exponent,
plus the recorded counterexamples. Wave66 now measures a maximum relative seed
error0.0031982446; after one correction1.5274123e-5, after two1.3826867e-7.
All786432 values are finite/positive and repeat exactly. FP64 recurrence using
the actual seed predicts the failure; arithmetic differs from that recurrence
by at most1.5031e-7. The seed's accuracy explains why one correction fails.
The probe stores intermediate stages and establishes no production timing.

Modes5/6 now pass all34 original hardware numerical gates. Mode5 loses
timing1.363799vs1.262366ms; mode6 loses1.255049vs1.239520ms. Neither advances. Mode6 uses q=x*r*r and
`r *= 1.875 - 1.25*q + 0.375*q*q`, the quadratic inverse-root expansion about
q=1. It needs fewer elementwise instructions than two Newton corrections and
reuses dead variance workspace. It is not Halley's rational iteration. A
million-point local FP32 seed-error sweep[-.004,.004] at x=1 stays below2.981e-7
relative error; this is sampled arithmetic evidence, not a full-range proof.
Both retain their original bounds and failed speed records; mode2 is preferred.

Reciprocal-only mode2 passes all34 unchanged numerical gates. The wave65
paired/interleaved test measures **1.085410ms versus1.304101ms** for mode0
functional NKI AdamW, **16.7695%** faster for this8M-element component, with
scalar staging and weight copyback charged. The three mode2 samples are
1.0854/1.0933/1.0732ms; controls1.2274/1.3252/1.3041ms. Both use identical
scheduled scalar updates. This is not a whole-optimizer or training gain.
Wave66's explicit mode2 integration test passes repeated scheduled updates,
None/zero/scalar fallback, Parameter identity, real BF16 Linear cache refresh,
CPU checkpoint serialization/restore and exact continuation. It records17
numerical checks,13 trajectory checks and7 fallback checks. Training still
uses mode0; there is no full-step gain claim for mode2 yet.

Evidence: `fs-w64-adamw-scalar-sqrt-reciprocal-newton-gate-r1`,
`fs-w65-adamw-scaled-rsqrt-newton-gate-r1`, and
`fs-w65-adamw-refined-reciprocal-paired-gate-r1`,
`fs-w66-rsqrt-seed-refinement-diagnostic-r1` and
`fs-w66-adamw-refined-reciprocal-integration-gate-r1`. All postflights are clean;
failures and original bounds are preserved.

The executable/design queue is now in
`fresh-start-kernel-queue-2026-09-18.md`. New `fp8_weight_cache_nki.py` implements
online power-of-two scaling through exponent bits, avoiding log2/floor/exp2
while retaining the necessary amax reduction. Its storage-only MLP control
requires exact output/gradient agreement for identical quantized BF16 weights,
before charging calibration, master casts and two forward/backward calls.
Wave69 passes calibration, exact paired NKI outputs/gradients and explicit
FP64-propagated backward bounds at real shape. It loses component timing:
17.898402vs14.989582ms. This is storage-only evidence, not an FP8 GEMM result;
actual FP8 arithmetic islands remain open.

### Polynomial exponential probe

Use `x = n*ln(2) + r`, then `exp(x) ≈ 2^n P(r)` for x∈[-30,30]. Construct
2^n through exponent bits; use split ln2 constants to preserve the small
remainder. Both nearest and directed integer rounding leave r in the fitted
[-.694,.694] interval. Degree5/7 coefficients were generated by Chebyshev
interpolation; the constant is set to1 so exp(0) stays exact. These are initial
coefficients, not claimed minimax optima.

Local FP32 arithmetic over2,000,001 points, checking truncation/floor/nearest,
found maximum relative errors1.85e-5 for degree5 and2.22e-7 for degree7, with
no decreasing steps on that grid. This sampled result is not a proof or a
Trainium result. The device probe separately declares2.5e-5/2e-6 relative
budgets, covers exponent boundaries and confident/uniform loss patterns, and
includes HBM traffic in the timing. Probability/gradient diagnostics normalize
on the CPU in FP64 to isolate the exp error; they do not replace a device
softmax/backward integration gate.

Files: `exp_polynomial_nki.py`, `exp_polynomial_probe.py`. Wave65 passes all9
device records. Real8M-element maximum relative errors are1.36600e-5/1.18306e-7
for degree5/7, versus1.13351e-5 for Scalar exp. The higher degree is about96times
more accurate on this sampled range, but **slower**:0.446690/0.524650ms versus
0.125238ms, including HBM transfers, over3×30 interleaved calls. These are
3.57×/4.19× the original latency. Close this standalone performance path;
retain the precise approximation as a diagnostic. No training integration.

### Composed loss function and BF16 indexing

With the current BF16 branch-norm option and no output bigram bias, the final
Linear receives BF16 and produces BF16 before the explicit FP32 upcast.
Verify this in the compiled path before specializing a replacement. The
finite input set offers a different table opportunity from an arbitrary FP32
function. However, target-relative exponentiation depends on a row value, so
it cannot simply become a unary table without reorganizing the loss.

A possible design tables a bounded composition, folds sign/saturation, uses
small-argument polynomial tails, and preserves a cancellation-resistant loss
and gradient. Keep the current target-shifted implementation as the reference.
First measure the present head+loss and its dtype/traffic; then compare total
costs, including table loading, index conversion, gathers and backward.

The next loss probe is deliberately exact: `softcap_ce_parallel_nki.py` and
`softcap_ce_parallel_probe.py` retain all primitive approximations and reduction
widths. They compare the original kernel with an explicit one-core control,
two-core row partitioning, and a one-element-per-row SBUF target gather that
removes the full-vocabulary target mask/multiply/reduction. AWS documents the
[LNC launch grid](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/lnc.html)
and the [per-partition gather contract](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_n_gather.html).
The installed original kernel's LNC metadata and actual native timings will be
recorded; no assumed twofold speedup. Require exact loss, saved logsumexp and
backward gradients, repetition and row independence at the real16384×8192
shape. One gather output per partition avoids the large-output grouping used
by the earlier failing embedding prototype; it still needs its own device gate.

## Experimental discipline and stopping criteria

Keep all approximations explicit and default-off. Use isolated component gates
before an adapter, then a complete-step measurement against a matched control.
Preserve the loader, scorer and evaluation implementation. Require at least3%
total-step improvement before another30-minute quality run. Approximation
accuracy and training quality are separate claims; a pointwise relative error
does not prove unchanged convergence.

For NaNs, identify the first nonfinite stage and examine range, zero times
infinity, reciprocal poles, negative variance, overflow/underflow and uninitialized
SBUF. Preserve failed evidence and correct the cause. Do not loosen a tolerance
to make an inaccurate kernel pass. Pre/post device cleanup and immutable source
manifests apply to every hardware probe.

The independent work-elimination experiment provides a concrete example of
the broader principle: padded whole-matrix optimizer ownership preserved exact
component weights/states and reduced the matched full step from554.873ms to
526.783ms (5.062%). Its30-minute quality test started on trn21 at21:52:43UTC.
Quality is pending. It avoids
redundant optimizer work rather than approximating it.

## Retrieval record

All13 user URLs were attempted. Tavily extracted7; the web tool opened the6
Stack Exchange/Reddit fallbacks. Their discussions were used as leads;
technical implementation claims above cite original source, authors, tools
or AWS documentation. The special-functions Math Stack Exchange question was
closed and has no substantive answer to import as evidence. Raw retrievals
are in ignored `out/retro-approximations-20260918/`. No downloaded web content
is treated as an instruction or executed.

### Wave69 exact loss scheduling result

Adaptive one/two-core dispatch and target gather pass all original exact
loss/lse/gradient, repetition, row-independence and boundary gates. Full
16384x8192 FW+BW measures5.679065ms versus12.339531ms original,53.9767% faster.
The raw two-core single-iteration shape failure remains documented and
ineligible; adaptive dispatch preserves one core there. This is a work/layout
change without new approximations. Compiled installed-autograd and whole-step
gates are required next; no training gain is claimed from the component timer.

### FP8 layout and causal scale policy

The actual-FP8 forward prototype uses per-token activation amax and exponent-bit
power-of-two scaling. Current-block global activation calibration would let
future tokens alter earlier outputs through the quantizer; per-token scales
avoid that dependence directly. Weights still use per-matrix calibration once
per update. The full hidden activation stays on chip, with explicit token-major
transposes for calibration and contraction-major FP8 tiles for the next GEMM.
This trades more layout instructions for smaller caches and actual FP8 compute.
No BF16 timing veto applies; actual compilation, stage gates, backward and
complete costs remain to be measured.

AWS's [double-row layout/type contract](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/nki/api/generated/nki.isa.nc_matmul.html)
specifies packed128x2 contraction tiles and FP8 operands. The empirically
characterized11-bit pair-rounding reference is our measured behavior on the
installed stack, not an AWS guarantee. The new layout pairs corresponding
positions in adjacent128-channel blocks and validates that order explicitly.

Latest complete gates: the exact adaptive CE/gather change improves the whole
step3.6625% (500.372595vs519.395419ms) and has advanced to fresh30-minute quality.
Exponent-bit scaling is now exercised inside actual FP8 up/down MLP GEMMs with
per-token independence and STE backward checks. This implementation loses
complete-component timing (18.181170vs15.662730ms native); attribution profiling
follows. Compact joint-function tables still have CPU, not device, evidence.

Compact table now has device evidence: every finite BF16 input and real logits
shape pass the predeclared2e-7 exponential-relative/1e-7 derivative-absolute
bounds. Peak exponential error is5.98e-8. Its complete primitive takes47.20ms
versus2.48ms Scalar composition, so that layout is closed as a speed candidate.
The new FP8 orientation also applies exact work elimination: squared ReLU is
nonnegative, making its calibration minimum reduction redundant. This identity
needs unchanged metadata/output gates before any claimed performance benefit.

### 2026-09-19, 00:08 UTC — completed gates and invariant queue

Wave77 token-row FP8 passes the unchanged stagewise and FP64 STE backward
contracts, including real dimensions and accumulated FP32 master gradients.
Its complete component is 17.979725 ms versus 18.161799 ms original FP8,
15.673194 ms native BF16 and 14.643596 ms NKI BF16 cache. The 1.0025% layout
gain is insufficient; profile attribution follows, with no quality promotion.
RMS-normalized input bounds now motivate a restricted fixed-scale FP8 input
experiment that can eliminate max/min reductions. Its overflow/subnormal/
causality/quality contract is separate and documented in the kernel queue.

The exact-CE/ownership/T1024-pack16 quality run completed within budget at
3,548 updates and small public 0.988605 BPB. Checkpoint locally hash-verified;
matched larger public evaluation pending. No private leaderboard claim.

### 2026-09-19, 00:33 UTC — dtype and caller layout gains

BF16 logits/gradient memory boundary passes72 exact loss/gradient comparisons
and saved-statistic checks. Complete loss-plus-cast component6.233179ms versus
10.071585ms adaptive FP32. Real head/caller integration is the next gate; no
whole-model speed claim. Token-row FP8 backward passes original FP64 bounds
and accumulation, improving16.515157vs17.920212ms (same forward) while still
trailing native15.675626ms. Profiling will test the compiler-transpose hypothesis.
The new FP8 backward design note records scale-factor algebra, packed-cache
and producer/statistic candidates, state constraints and primary references.

### Vocabulary projection and loss fusion

The next work-elimination design is in
`fresh-start-fused-head-loss-design-2026-09-19.md`. It reviews Apple CCE and AWS
experimental NKI CE, retains FP32 head accumulator values and softcap15, and
separates forward fusion from recomputation and optional gradient filtering.
The current caller trace determines whether its complete cost warrants a
prototype. Lower memory use alone is not evidence of faster training.
