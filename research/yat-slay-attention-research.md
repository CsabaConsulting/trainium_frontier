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

**Next**: Phase 2 (small, clearly-separated real-hardware smoke test) is a real Trn2
time spend, however small — checking in with the user before launching it, per the
plan's own gate for this phase.
