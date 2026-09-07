# YAT-kernel / SLAY attention — parallel research track

Status: exploratory, offline-first. Does not touch the locked `train.py` config or the
live Bucket A-E hardware work tracked in `trn2-experiment-results.md` /
`experiment-rollout-plan.md`. Raised by the user, explicitly to run alongside that work
without interfering with it.

## Source material

- `articles/No More DeLuLu Physics-Inspired Kernel Networks for Geometrically-Grounded Neural Computation.pdf`
- `articles/SLAY Geometry-Aware Spherical Linearized Attention with Yat-Kernel.pdf`
- `articles/what-a-finite-kernel-buys-an-mlp-jax-flax-nnx.ipynb` (not yet read in depth — likely illustrates the same E-product/MLP concept as paper 1, lower priority)
- `research/gemini_ai_mode_yat_kernel.md` — an AI-summarized conversation the user themselves flagged "take it with a grain of salt." **Not used as a primary source below** — both papers were read directly and independently, and the arXiv IDs verified live against arxiv.org.
- `research/trainium_nanogpt_yat.py` — an unverified implementation sketch (from the same AI conversation) of a naive quadratic YAT-attention NanoGPT block. Contains real bugs as written (see "Sketch review" below) — not usable as-is, but useful as a starting shape.

## Primary-source verification (both papers are real, not fabricated)

This session has repeatedly caught fabricated technical claims in AI-generated docs
(fake NKI APIs, a fake compiler flag). Checked the same way here: fetched both PDFs
directly, read the real content, and confirmed both arXiv IDs resolve to real, live
listings with real authors:

- **`arXiv:2603.12276`** — "No More DeLuLu: Physics-Inspired Kernel Networks for
  Geometrically-Grounded Neural Computation," Taha Bouhsine (Azetta AI). Confirmed live
  on arxiv.org/abs/2603.12276.
- **`arXiv:2602.04915`** — "SLAY: Geometry-Aware Spherical Linearized Attention with
  Yat-Kernel," Jose Miguel Luna (Columbia), Taha Bouhsine (Azetta.ai), **Krzysztof
  Choromanski (Google DeepMind)**. Confirmed live on arxiv.org/abs/2602.04915.
  Choromanski is a real, credentialed researcher — co-inventor of Performer/FAVOR+
  (positive random features for linear attention), which this paper directly builds on.
  His involvement is real signal that the linearization math is credible, not a
  fabrication.

## What each paper actually claims

### Paper 1 — "No More DeLuLu" (the E-product / Yat-kernel)

Defines a single kernel operator:

```
E(w, x) := (w^T x)^2 / (||w - x||^2 + eps)
```

— squared alignment (numerator) divided by inverse-square proximity (denominator),
explicitly inspired by physical inverse-square laws. Proven a valid Mercer kernel with a
unique RKHS embedding, Lipschitz-continuous, self-regularizing (bounded response as
`||x|| -> infinity`).

Two uses in the paper:
1. **Neural Matter Network (NMN) layers** — replaces `Linear -> activation -> LayerNorm`
   entirely with a single E-product-based layer (no activation function, no
   normalization layer — normalization is baked into the kernel's denominator).
2. **YAT-attention** — replaces the *similarity function* inside standard softmax
   attention: `S_ij = K_E(q_i, k_j)` instead of `q_i^T k_j / sqrt(d)`, but still
   `softmax_j(S_ij) @ V` row-wise over the full `L x L` score matrix. **This is still
   `O(L^2)` — no compute or memory saving over standard attention**, just a different
   score function.

Real result, language modeling: Aether-GPT2 (124M params, both attention *and* MLP
blocks replaced, trained 2.5B tokens FineWeb) vs. GPT-2 baseline, identical
hyperparameters:

| | GPT-2 | Aether-GPT2 |
|---|---|---|
| Activation | GeLU | E (none) |
| LayerNorm | Yes | No |
| Val loss (final) | 4.6417 | **4.5747** |
| Val improvement | — | **1.45%** |
| Memory | — | **-15-25%** (no LayerNorm/activation storage) |
| Throughput | comparable | comparable |

A real, modest, measured win — but note **throughput is "comparable," not faster**, and
the memory saving comes from dropping LayerNorm/activation *storage for backward*, not
from the attention mechanism being cheaper.

### Paper 2 — "SLAY" (the linear-time version)

Genuinely different, more sophisticated mechanism. Starts from the same E-product but
constrains queries/keys to the unit sphere (L2-normalize), then uses **Bernstein's
theorem** to express `1/(C - 2x)` (the spherical E-product's denominator, after
expansion) as a Laplace integral of exponentials, approximates that integral via
**Gauss-Laguerre quadrature** (R quadrature nodes), and approximates the resulting
exponential-times-polynomial kernel via **positive random features** (the same class of
technique as Performer/FAVOR+ — unbiased, strictly positive, avoids the
numerical-instability problems of *signed* random-feature approximations like
TensorSketch/Random Maclaurin, which the paper's own Table 1/2 shows going numerically
unstable). This yields a real, unbiased, **`O(L)`** linear-attention estimator — the
`L x L` score matrix is never materialized.

Real results, language modeling: SLAYformer (125M params, GPT-2-small architecture,
2.5B tokens, Chinchilla-scaled), compared against full quadratic softmax and other
linear-attention baselines under matched training conditions:

| Method | Complexity | Val Loss | PPL |
|---|---|---|---|
| Yat (Exact) | O(n^2) | 4.5747 | 97.03 |
| Standard Softmax | O(n^2) | 4.6417 | 103.73 |
| Yat (Spherical) | O(n^2) | 4.7180 | 112.00 |
| **SLAYformer** | **O(n)** | **4.6760** | **107.35** |
| Linear (ELU+1) | O(n) | 5.0884 | 161.99 |
| Cosformer | O(n) | 5.1983 | 180.97 |
| FAVOR+ (Performer) | O(n) | 5.4524 | 233.32 |

**Real, honest reading of this table**: SLAY is closer to full softmax attention than
any other linear-attention baseline by a wide margin — a genuine advance in the
linear-attention literature. But SLAY is still **measurably worse than standard softmax**
(`4.6760` vs `4.6417`, ~0.7% relative) and **worse than exact quadratic Yat**
(`4.5747`) — the linearization costs real quality, it isn't free. The paper's own
framing ("within a narrow margin," "closely approaches") is accurate but the gap is real,
not zero.

Scaling claim (Figure 2/3): SLAY scales near-linearly in latency/memory up to ~128K
tokens, where quadratic methods OOM beyond ~16K; SLAY "sustains high throughput
comparable to other linear baselines," and the paper claims SLAYformers "process
sequences up to 30x longer than standard softmax attention." This is the part that
actually matches the user's stated interest (long context, avoiding quadratic blowup) —
but it's a claim about *scaling to very long sequences*, evaluated at full,
generously-trained model scales, not about our specific tiny/short-budget regime.

## Correcting the AI-summarized doc's framing

`gemini_ai_mode_yat_kernel.md` (and the user's own initial framing) described this as
something that would "eliminate KV cache bloat" broadly. That's only true of **SLAY**
specifically — paper 1's plain YAT-attention is still `O(L^2)`, no memory/compute win
over standard attention at all. Also worth noting: "KV cache" is strictly an
*inference-time* concept (caching past K/V during autoregressive generation) — not
directly relevant to Phase 1 *training* (no generation happening), though the underlying
compute-complexity argument (quadratic vs. linear attention cost as a function of
sequence length) does transfer to training cost too, which is the actually-relevant
angle for us.

## Why this is a real headwind for *our* specific regime (not a weakness of the papers)

1. **Attention is not currently our bottleneck.** The Bucket C profile (fresh, at the
   current locked shape: `SEQ_LEN=2048, USE_CAUSAL_FASTPATH=True`, kernel ON) found
   `_scaled_dot_product_fused_attention_overrideable` at only **4.7%** of the profiled
   window — already fused, no dispatch overhead, not a target. SLAY's whole value
   proposition (saving quadratic attention cost) targets a cost we don't currently pay
   much of, at our current context length.
2. **We already have real, measured, contradicting evidence that "longer context
   helps" here.** This round's `SEQ_LEN` sweep found `SEQ_LEN=4096` genuinely *worse*
   than `2048` in quality (`public_val_bpb` 1.209781 vs 1.198086 at the 900s proxy) —
   and step count kept *rising* through 4096 (396 vs 387), meaning the downturn is a
   real quality effect, not a compute-cost/throughput effect. SLAY fixes compute cost at
   long context; it does not obviously fix whatever caused that quality downturn. There
   is no evidence yet that swapping to SLAY would let us push past `SEQ_LEN=2048`
   *and* keep winning on quality — that's a real, open, unverified empirical question,
   not something to assume either way.
3. **SLAY's own best published number is measurably worse than full softmax**, even at
   generous GPT-2-scale training (2.5B tokens, full convergence) — a real cost, not
   free. Our regime (55.7M params, ~500-750 steps, ~100M tokens) is far more
   under-trained/data-starved than the paper's setting; there's no basis yet for
   assuming the approximation gap shrinks (it could easily be worse, with less training
   to average out the estimator's bias/variance).
4. **Implementation is genuinely high-effort and high-risk.** SLAY's real mechanism
   needs: spherical normalization, Gauss-Laguerre quadrature nodes, positive random
   features (a` la Performer), sketched tensor-product feature fusion — novel machinery
   with no existing NKI/Neuron kernel or PyTorch library support to lean on. The paper's
   own appendix separately flags causal vs. non-causal implementation as needing
   different handling — **causal linear attention typically requires a cumulative-sum
   over the sequence dimension**, which is a sequential, position-dependent
   recurrence. That is a plausible *new* throughput risk on Trainium specifically:
   sequential recurrence patterns are notoriously unfriendly to accelerators without a
   dedicated fused kernel (cumsum-style ops don't parallelize the way matmuls do) —
   exactly the class of problem this whole competition has been fighting all session
   (many small/sequential ops = dispatch/serialization overhead).
5. **Every published result is at a scale far beyond ours.** 124-125M params, 2.5B+
   tokens, full convergence — vs. our 55.7M params, ~500-750 steps. No evidence this
   transfers to an ultra-short, wall-clock-constrained budget; it could go either way.

None of this means "don't pursue it" — it means: treat this as a genuine, uncertain-payoff
research bet, not an assumed win, and gate hardware time accordingly.

## Sketch review: `research/trainium_nanogpt_yat.py`

This file (from the AI conversation, unverified) implements the naive quadratic
YAT-attention only (not SLAY's linearization) as a NanoGPT block. Real issues found on
inspection, before this is used as a starting point for anything:

- `q_norm`/`k_norm`/`distance_sq` expansion is algebraically correct
  (`||q-k||^2 = ||q||^2 - 2<q,k> + ||k||^2`), matching both papers' own approach.
- `torch.clamp(distance_sq, min=0.0)` before adding `epsilon` is a reasonable numerical
  safety net (squared-distance expansion can go slightly negative from floating-point
  cancellation) — sound.
- **Step 5 (SLAY normalization) is confused/mislabeled**: the code applies causal
  masking, then does `A / (sum(A, dim=-1) + eps)` as an ad-hoc "normalization,"
  *separate from* the softmax step already implicit in neither paper's exact
  formulation. Neither paper's exact (non-linearized) YAT-attention uses this — paper
  1's YAT-attention explicitly still uses `softmax_j(K_E(q_i,k_j))`, a real softmax, not
  a sum-and-divide. This sketch is not a faithful implementation of either paper's exact
  mechanism as written — it's a third, ad-hoc variant. Do not treat it as a validated
  reference; it needs to be rebuilt against the papers' actual formulas.
- The line `torch.matmul(q, k.transpose(-2, -1))` is computed **twice** (once for
  `alignment`, once inside `distance_sq`) — redundant, would want to compute once and
  reuse for both, real but minor inefficiency, not a correctness bug.
- The unrelated `NanoGPTYatModel.forward`'s cross_entropy line has a real Python syntax
  bug (`logits.view(-index1, ...)` — `-index1` is not a valid literal, this file as
  written does not execute as-is). Cosmetic/copy-paste artifact, not a design issue, but
  confirms this sketch was never actually run.

**Conclusion**: don't use this file directly. Rebuild both the exact-YAT reference and,
if pursued, the SLAY linearization from the papers' actual formulas, with the same
build-and-verify discipline as everything else this session.

## Plan

Phased, offline-first, no interference with the live Bucket A-E hardware track.

### Phase 1 (now) — exact quadratic YAT-attention as SLAY's own verification oracle

Per SLAY's own evaluation methodology (their Table 2/5 compare "Exact/Spherical YAT" vs
"SLAY Approx" as the ground truth) — build exact YAT-attention *and* SLAY's
linearization together, not as a separately-gated "maybe skip" step, with exact YAT
serving as the numerical correctness reference for SLAY, exactly like this session's
kernel-vs-eager equivalence harnesses (rope+QK-norm vs. eager `F.rms_norm`/
`apply_rotary_emb`).

Concretely, in a new scratch module (e.g. `research/scratch/yat_slay_prototype.py`,
standalone, does not touch `train.py`):
1. Exact YAT-attention: `E(q,k) = (q^T k)^2 / (||q-k||^2 + eps)`, real softmax over the
   full score matrix (matching paper 1's actual formula, not the sketch's ad-hoc
   sum-normalize), causal-masked. CPU-only, tiny toy shapes.
2. SLAY linearization: spherical normalization, Bernstein/Laplace integral via
   Gauss-Laguerre quadrature, positive random features (Performer-style), sketched
   tensor-product fusion, linear-time reordered computation (Algorithm 1 in the SLAY
   paper). Verify against (1) — relative L2 error / cosine similarity, matching the
   metrics the paper itself reports (their best config, "Anchor" polynomial features,
   achieved `Rel. l2 = 0.527`, `Cos = 0.850` against exact spherical YAT — a real,
   nontrivial baseline to beat or match, not a rounding-error-level match).
3. Gradient-flow sanity on both (no NaNs/instability across a range of input scales,
   matching this session's equivalence-harness discipline for the NKI kernels).

### Phase 2 (only if Phase 1 is clean) — small, clearly-separated real-hardware smoke test

A throwaway experimental copy or an off-by-default toggle, distinct label, tiny step
count, single device. Purpose: check whether the new op sequence (multiple full-tensor
elementwise passes per head for exact YAT; quadrature/random-feature/tensor-sketch ops
for SLAY, plus the causal cumulative-sum concern above) is even competitive with the
already-fused, near-peak SDPA kernel on this hardware — not assumed either way. Does not
touch the locked default config; does not run in the same wall-clock slot as an in-flight
Bucket A-E run.

### Phase 3 (later, low priority given the headwinds above) — full integration attempt

Only if Phase 1-2 are both genuinely promising and long-context is still the goal by
then. This is a large, uncertain-payoff undertaking given current evidence — not a
near-term commitment.

### Engine-mapping / overlap design notes (for whenever a real NKI kernel is attempted)

Raised by the user before Phase 2 hardware work: is SLAY's extra computation
distributable/overlappable across Trainium2's Tensor/Vector/Scalar/GpSimd engines?

**Within-kernel overlap — realistic, with precedent.** SLAY's ops map cleanly:
matmuls (anchor/polynomial features, PRF projections, the `Ψ(Q)·(Ψ(K)^T·V)` linear-time
core) → Tensor Engine, same 128×128 array our linear layers already hit at 89.6% of peak
FLOPs (Bucket C profile). L2-normalize, `exp()` for PRF, division, elementwise fusion →
Vector/Scalar Engine. The Bucket C profile already shows real concurrent multi-engine
utilization on the *current* model (TensorE 47.1%, DMA 39.8%, VectorE 23.8%, GpSimd
21.6%, ScalarE 16.8% — summing past 100%, i.e. several engines already run
simultaneously under the compiler's own scheduling). A well-tiled NKI kernel pipelines
this the same way the existing rope+QK-norm/relu² kernels do. Per-head independence
(PRF/polynomial features for head *i* don't depend on head *i+1*) is real, exploitable
parallelism if the kernel is written to expose it.

**Overlap with the rest of the model — real but limited, unmeasured.** Attention feeds
directly into the same layer's MLP — genuinely sequential, can't overlap with itself.
DMA weight-prefetch for the next layer plausibly overlaps regardless. With
`DEVICE_BATCH_SIZE=1`, there isn't much cross-batch-element parallelism to exploit
today. This part is speculative until an actual implementation is profiled — no real
number to back it yet, only the current model's baseline concurrency.

**The real risk is the causal cumulative-sum, not general overlap.** Phase 1's
`torch.cumsum` is explicitly correctness-only, not a real hardware design — it's the one
genuinely sequential piece (each position's running sum depends on the previous one),
the least overlap-friendly part of the whole mechanism. Established fix, with real
precedent (FlashLinearAttention/RWKV-style kernels): **chunked/blocked causal linear
attention** — within a chunk, compute the causal contribution via a masked matmul
(parallel, Tensor-Engine-friendly, the same trick flash attention's own causal masking
uses); carry only a small running state *between* chunks (sequential, but `L/chunk_size`
steps instead of `L`). This is the right target design if Phase 2/3 gets far enough to
need a real kernel, not a literal cumsum.

## Log

### Phase 1 — done, real success, numbers beat the papers' own reported accuracy

`research/scratch/yat_slay_prototype.py` (CPU-only, offline, run on the remote box's
existing `native_venv` since it already has torch/numpy — negligible resource use
alongside the concurrent Bucket A-E GPU-bound training job, no meaningful interference).

Implemented and verified against each other:
- `exact_yat_attention_softmax` — paper 1's plain E-product + real softmax.
- `exact_spherical_yat_attention` — SLAY's own ground-truth reference (spherical
  E-product, exact `O(L^2)`, normalized-kernel-weighted average).
- `slay_attention` — the `O(L)` linearization (Bernstein/Laplace via Gauss-Laguerre
  quadrature, exact degree-2 polynomial features, Performer-style positive random
  features, causal via cumulative sum).

**Result**: both exact variants ran cleanly, gradients flow with no NaN. SLAY's
approximation, at tiny toy shapes (`B=2,H=2,L=16,D=8`, random Gaussian Q/K/V):

| Config | rel_l2_err | cos_sim |
|---|---|---|
| R=3, prf_dim=32 | 0.272 | 0.965 |
| R=6, prf_dim=64 | 0.163 | 0.987 |
| R=10, prf_dim=128 | 0.146 | 0.990 |

The paper's own best published config ("Anchor" low-rank features): `rel_l2=0.527,
cos=0.850`. **Every tested config here beats that** — most likely because this
prototype uses the *exact* degree-2 polynomial feature map (`vec(uu^T)`, `O(d^2)`, no
approximation) rather than the paper's low-rank anchor-feature approximation of that
same factor; only the PRF/exponential term is randomized here, same as the paper. Error
improves monotonically with more quadrature nodes/PRF dimension, exactly matching the
theory's prediction — a real, self-consistent check that the math is implemented
correctly, not just "it ran."

**Numerical-stability sweep** (input scale `0.001` to `40`): error is *exactly* identical
across all scales. Not a bug — a correct, direct consequence of the spherical
normalization (Q/K are L2-normalized before use in both the exact and SLAY paths, so raw
input magnitude cancels out entirely) — matches the papers' own claimed
self-regularization/dimensional-self-normalization property (Corollary 2 in paper 1),
now independently confirmed rather than just cited.

**Conclusion: Phase 1 passed cleanly.** The core SLAY math is implementable correctly on
our stack and behaves as the papers claim. This resolves the main *correctness* risk
flagged in the plan — it does not resolve the *value* questions (does this help our
regime specifically, is a causal-cumsum implementation fast enough on Trainium) — those
are Phase 2's job, gated on a real hardware smoke test, not yet run.

### Phase 2 — done, real hardware, decisive negative

`research/scratch/slay_trn2_smoke.py` (real Trn2 device, `~/workspace/native_venv`,
`torch.device("neuron")`, native eager — no XLA). Throughput-only: not a training run,
does not import `train.py`, does not touch the locked default config, ran with the
device otherwise idle (verified via `neuron-ls`/`ps` before launch, no in-flight Bucket
A-E job).

**Back-of-envelope FLOP estimate done first (before spending any hardware time)**: causal
SDPA ≈ `L²·d ≈ 5.37e8` flops/head at the real shape (`L=2048, d=128`). SLAY's
chunked-linear cost is `O(L·C·m)` (`C`=chunk size, `m=R·P·prf_dim` the feature dim).
Beating SDPA on FLOPs alone needs `C·m < L·d = 262144` — e.g. `C<64` at a modest
`m=4096` (`R=4,P=32,prf_dim=32`). Low-rank poly feature swapped in for Phase 1's exact
`vec(uu^T)` (unusable at real `head_dim=128`, `d²=16384`): **Random Maclaurin features**
(Kar & Karnick 2012) — `phi(u)_j=(1/√P)(a_j^Tu)(b_j^Tu)`, unbiased estimator of
`(u^Tv)²`, pure matmul+elementwise, no FFT/hashing. A separate, established technique
from Phase 1's oracle — not re-verified for accuracy here, this script's only job is
throughput.

**Real measured result at the exact locked shape (`B=1,H=5,L=2048,D=128`, fwd+bwd)**:

| Config | m | ms/iter | vs SDPA (`0.227ms`) |
|---|---|---|---|
| SDPA (fused kernel) | — | 0.227 | 1.0x |
| SLAY R=4,P=32,prf_dim=32,chunk=64 | 4096 | 304.4 | **1338x slower** |
| SLAY R=4,P=32,prf_dim=32,chunk=256 | 4096 | 106.3 | **467x slower** |
| SLAY R=3,P=16,prf_dim=16,chunk=64 | 768 | 172.0 | **756x slower** |

Worse than the FLOP-only estimate predicted, and for a diagnosable reason: bigger chunk
(fewer sequential Python-loop iterations, fewer NEFF dispatches) is *faster* despite
identical FLOPs — confirms per-op dispatch-gap serialization dominates, not raw compute.
Same root cause already documented in Bucket E (native TorchNeuron eager backend, one
NEFF per aten op, no cross-op fusion) — SLAY's op sequence has far more small ops per
attention call than SDPA's one fused kernel, so it eats that overhead far harder.

**Verdict: real, decisive negative. Do not pursue Phase 3 (full integration) this
round.** Even a hypothetical NKI-fused SLAY kernel would need to close a ~500x-1300x gap
through fusion alone — and attention is only 4.7% of the current profiled step, so even
a 10x kernel win over this measurement would still leave SLAY ~50-130x slower than the
existing fused SDPA on the one op it would replace. Not a close call, not scale-transfer
ambiguity like XSA/Mamba2 — this is a direct, same-scale, real-hardware measurement.
Deprioritizing, not discarding, per the standing preference: a from-scratch NKI kernel
(chunked causal linear attention, engine-mapped per the design notes above) is the only
path that could plausibly change this verdict, and it would need to close three orders
of magnitude, not one — flag as very low priority, worth revisiting only if consulting
Taha Bouhsine directly surfaces a materially different implementation strategy (e.g. a
production SLAY kernel with real measured overhead, not this correctness-first
prototype's naive chunking).

**Track closed for this round.** Both YAT/SLAY and Mamba2/hybrid (separate doc) now
have real, hardware/research-confirmed "skip this round" verdicts. Returning focus to
Bucket A-E work (torch.compile Bucket E, remaining Bucket C/D items) per the plan's own
execution order.

### Phase 3 — real NKI-kernel attempt, gated at the measurement step. Verdict: do not build.

Re-opened at the user's explicit request: the Phase 2 negative measured a *naive PyTorch*
implementation, whose 467-1338x gap was dominated by per-op dispatch overhead — not a
verdict on a genuinely fused kernel. This phase asks the fair question: **could a real,
well-optimized, fully-fused NKI kernel close that gap?** Per the instruction to
"measure before you build anything elaborate," the gate ran first.

`research/scratch/slay_nki_feasibility.py` — offline, CPU-only, numpy-only.
**Zero hardware time used** (no device contention, nothing to schedule around).

Three independent gates. Real Trn2 datasheet numbers throughout (NeuronCore-v3, from the
NKI arch guide, not estimated): TensorE 79 BF16 dense TFLOPS = 3.95e13 MAC/s; VectorE 512
BF16 elem/cycle @ 0.96 GHz = 4.92e11 elem/s; ScalarE 128 elem/cycle @ 1.2 GHz = 1.54e11
elem/s. **The engine ratios are the headline hardware fact: TensorE:VectorE = 80:1,
TensorE:ScalarE = 257:1.**

**Gate A — cost model (engine-aware, perfect 100%-of-peak kernel, perfect overlap).**
Modelled the *best case* SLAY, deliberately stronger than either prior prototype. Phase 1
used exact `vec(uu^T)` (`d²`=16384 at d=128, unusable); Phase 2 used the tensor product
`m = R·P·prf_dim` (4096 at modest settings). Both are more expensive than necessary: since
`x²e^{ax} = d²/da² e^{ax}`, differentiating the PRF estimator twice in the quadrature
parameter gives features for the *combined* `x²e^{ax}` kernel directly, so the polynomial
and exponential factors need not be tensored — `m = c·R·prf_dim` with `c≈4` instead of
`m = R·P·prf_dim`, roughly `P/c ≈ 8x` smaller, with the random projection shared across
all R nodes. (Honest caveat: derivatives of a positive estimator are not positive, so this
form sacrifices the strict-positivity that motivates SLAY's use of PRF. If the gate had
passed we would owe it a real variance/stability study.)

Results — and this is where the honest answer diverges from the expected one:

- The model reproduces the measured baseline well. Causal flash SDPA at the locked shape:
  best-overlap 0.238 ms fwd+bwd, worst-case 0.635 ms; **real measured 0.227 ms**. The
  fused SDPA is running essentially at this hardware's ceiling — there is no
  implementation slack to exploit. The model also independently reproduces
  `trainium2_arch.md`'s own remark that self-attention on Trn2 is "ScalarE/VectorE
  bound": ScalarE 0.068 ms ≈ TensorE 0.068 ms ≈ VectorE 0.064 ms per forward, i.e. the
  op is balanced-to-elementwise-bound, not TensorE-bound.
- **Gate A does NOT fail.** There is a real window: a perfect kernel at `m=256, C=128`
  models at **0.50x SDPA** (2x faster); `m=512, C=128` at 0.84x. Above that it degrades
  fast (`m=1024` → 1.62x slower; `m=4096`, Phase 2's setting → 6.3x slower, which is
  what a perfectly-fused version of the Phase 2 measurement would have given, confirming
  the 467x was indeed ~99% dispatch overhead).
- Hard, formulation-independent ceiling: the two state matmuls cost `2·L·m·D` MACs and no
  feature map can avoid them, vs causal SDPA's `L²·D`, so beating SDPA needs `m < L/2 =
  1024`; adding the unavoidable intra-chunk `L·C·m` term tightens it to **m < 683 at
  C=128** (512 at C=256, 341 at C=512).

**Gate B — is the kernel representable at m ≤ 683? No.** This is what kills it.
`E_sph(q,k) = f(q̂·k̂)` is a dot-product kernel on `S^{d-1}`, so its Mercer eigenspaces
*are* the spherical harmonics, with multiplicity `N(d,n)`. At d=128: `N=1, 128, 8255,
349504` for degrees 0,1,2,3. A feature map must spend `N(d,n)` dims to represent degree n.
The `m ≤ 683` budget buys degrees 0-1 exactly (129 dims) and then only a ~554-dim slice of
the **8255-dim degree-2 space (6.7%)** — and degree 2 is precisely the `x²` numerator that
*is* the Yat kernel. Measured residuals of least-squares polynomial fits to `f` over the
real `x` distribution at d=128:

| poly degree | feature dims needed | residual rel err |
|---|---|---|
| 1 | 129 | 0.916 |
| 2 | 8,384 | 0.445 |
| 3 | 357,888 | 0.239 |

Corroborating measurements:
- **The initially-encouraging small Gram ranks are a sub-sampling artifact.** Numerical
  rank at 95% energy climbs monotonically with sample size and never saturates: 96, 157,
  243, 345, 448 at L=256…4096. The relevant object is the kernel *operator*, not one
  `L×L` slice.
- Even the **SVD-optimal, data-dependent oracle** — a ceiling no fixed feature map can
  reach — has rel err **0.267 at m=256** and **0.166 at m=512** at the real L=2048, and
  degrades further as L grows (0.294 / 0.207 at L=4096).
- Real implementable maps in that window are far worse: Random Maclaurin (Phase 2's map)
  approximating just the `(q̂·k̂)²` factor at d=128 has rel err **2.85 at P=700** and
  **9.42 at P=64**; it needs P≈16384 to reach 0.589.
- **The framework predicts the paper's own number**, which is real validation rather than
  assertion: SLAY's best published config ("Anchor" features) reports `rel_l2 = 0.527`,
  landing exactly between the degree-1 (0.916) and degree-2 (0.445) fits — where a map
  capturing degrees 0-1 plus a partial degree-2 slice must land. **0.527 is not a tuning
  failure by the authors; it is the harmonic budget at d=128.**
- **This also retracts Phase 1's headline claim.** Phase 1's "every config beats the
  paper's own accuracy (`rel_l2` 0.146 vs 0.527)" was an artifact of running at `d=8`,
  where the exact degree-2 map costs `d(d+1)/2 = 36` dims and is therefore affordable.
  That advantage does not exist at the real `d=128`. Phase 1 was not wrong about the math
  being implementable — it was measuring a regime where the central obstruction is ~230x
  weaker and thus invisible.

**Conclusion: SLAY's affordable-m window and its accurate-m window do not overlap at
d=128.** `m ≤ 683` to beat SDPA; `m ≥ 8384` to represent the kernel it is approximating.

**Gate C — what a win would even be worth.** Independent of A and B, and on its own
sufficient. Attention is 4.7% of the profiled step (Bucket C). Using the real
step-count→bpb sensitivity from the DEPTH=5/6 A/B at the 900s proxy (+31% steps bought
0.0045 bpb):

| attention speedup | step time | extra steps | bpb gained |
|---|---|---|---|
| 1.5x | -1.57% | +1.59% | ~0.00023 |
| 2.0x | -2.35% | +2.41% | ~0.00035 |
| infinite | -4.70% | +4.93% | ~0.00072 |

Against SLAY's own published quality cost vs standard softmax (val loss 4.6760 vs 4.6417
= 0.74% relative → **~0.00926 bpb lost** at our 1.2527) — at 125M params / 2.5B tokens /
full convergence, i.e. its most favourable published setting. **Even an infinitely fast
attention gains ~13x less bpb than SLAY's best case costs**, and the best *modelled*
kernel (2x) gains ~26x less. Our regime is far more under-trained than the paper's, with
no reason the approximation gap would shrink.

**Verdict: do not build the kernel.** Not because the fusion idea was wrong — Gate A
confirms the user's instinct that the Phase 2 gap was almost entirely dispatch overhead,
and that a perfect kernel really could be ~2x faster than SDPA in a narrow window. It
fails because (B) that window is mathematically too narrow to hold the kernel SLAY is
approximating at `head_dim=128`, and (C) even winning it would be worth ~26x less than
the approximation's own quality cost. Both are first-principles results at the real shape,
not scale-transfer guesses — a stronger and more specific basis for the negative than
Phase 2's dispatch-overhead measurement.

**What would change this verdict** (recorded so the track is deprioritized, not lost):
- A much longer context. The obstruction is `m < L/2` versus `m ≥ 8384`, so the windows
  only overlap at `L ≳ 17k`. That matches the SLAY paper's own scaling story (its wins
  are demonstrated at 16K-128K tokens) — and directly explains why it does not transfer
  here. Our own measured `SEQ_LEN` sweep found 2048 an interior optimum with 4096
  *worse*, so this repo is moving away from that regime, not toward it.
- A much smaller `head_dim`. The degree-2 cost is `d(d+1)/2`, so `d=32` needs 528 dims —
  inside the budget. But `HEAD_DIM=128` is a separately-tuned locked choice, and shrinking
  d also shrinks SDPA's cost proportionally, so this is a different experiment, not a fix.
- Not fixable by a better kernel, a better random-feature scheme, or more quadrature
  nodes. Gate B's bound holds for the paper's anchor features, Random Maclaurin, the
  moment form derived here, and any map not yet invented.
