# Diffusion language models at NanoGPT scale — would one help here?

Research pass on 2026-09-12 (local box, no hardware). Prompted by the question behind the
shared Google AI Mode link: *"is there a nanoGPT-like diffusion text model? something like
downsizing DiffusionGemma?"* The AI Mode answer itself is rendered client-side and could
not be retrieved; everything below comes from primary sources (papers, model cards, repos)
and from this repo's own scoring code and run logs.

## 0. Short answer

1. **Yes, nanoGPT-style diffusion LMs exist**, several of them, from 10M-parameter
   character-level toys to a 20M–780M LLaDA clone that trains on the same ClimbMix data
   this competition uses (§1).
2. **"Downsizing DiffusionGemma" is not a meaningful path.** DiffusionGemma is a 26B MoE
   hybrid whose *encoder* is a causal Gemma 4 and whose *denoiser* is the same weights run
   bidirectionally over 256-token canvases. Its pretraining recipe is unpublished; only a
   LoRA fine-tuning adapter is public. The small open re-implementations already *are* the
   downsized version of the idea (§2).
3. **At NanoGPT scale and compute, diffusion LMs lose to autoregressive (AR) models by a
   wide, well-measured margin.** Matched 10M models: 2x the validation NLL. 110M models on
   OpenWebText: perplexity 23 vs 17.5. Scaling-law studies from 14M to 2B parameters put the
   gap at **14–16x more compute** for masked diffusion to match AR likelihood (23x for
   uniform-state, 32x for Eso-LM), with the compute-optimal diffusion model about half the
   size of the AR one. Diffusion only overtakes AR after tens to hundreds of epochs over a
   small dataset; this competition trains for 0.9 epochs (§3–§4).
4. **For Phase 1 the question is moot regardless of quality**: the scorer computes
   cross-entropy from a single causal forward pass `model(x)` and runs a causality check
   that flips future tokens. A diffusion model has no such forward, its likelihood is an
   ELBO or a multi-pass exact likelihood, and a bidirectional model fails the causality
   check outright (§5).
5. Translated to this repo's exchange rate, a hypothetical masked-diffusion train.py at
   the same 30-minute budget would sit around **1.15–1.30 bpb-equivalent** (ELBO), versus
   the AR 1.034 local / 1.0548 board, and could not be scored anyway (§4).

Verdict: no Phase 1 value. Nothing in the diffusion toolbox transfers to a causal AR
model under a <1-epoch budget either (§6).

## 1. What exists at "nanoGPT scale"

| repo | size | notes |
|---|---|---|
| `nathan-barry/tiny-diffusion` (927★) | 10.7M, char-level, Tiny Shakespeare | 365 lines; a matched `gpt.py` (313 lines) shares ~80% of the code. The diffusion model "trains for twice as long because half as many tokens count towards the loss". |
| `SoumilBhandari/nanoDLM` | 10M, 6 layers, 384-dim, char-level | Fork of the above with bug fixes (loss normalised by B·T, not `mask.sum()`), RoPE, self-conditioning, and an **honest matched AR baseline** (same arch/params/steps/data): MDM val char NLL ≤1.097 (ELBO) vs AR **0.527**; sample PPL under an AR scorer 3.38 vs 1.21. The only thing the MDM wins is infilling, which AR cannot do. |
| `Lukas-Xue/nanoLLaDA` (110★) | 20M–780M, BPE, **ClimbMix-400B** | ~500 lines, nanochat-style. Depth 4 = 20M, depth 12 = 135M, depth 20 = 477M. Documents `loss = CE(logits[masked], targets[masked]) / mask_ratio` and `is_causal=False` as the entire diff from GPT. d20 (477M) trained 9.2B tokens on 4×L4 reaches val diffusion loss 3.02 (its own tokenizer; not comparable to this repo's numbers). CORE eval is done by ELBO scoring, not by a forward pass. |
| `ash80/diffusion-gpt` (262★) | 7.23M, char-level | "From baby GPT to diffusion GPT": an annotated notebook that converts Karpathy's baby GPT into a discrete diffusion model. Named in the AI Mode answer. |
| `lapp0/nano-block-diffusion` (15★) | small, PyTorch | Block-diffusion trainer, i.e. the fixed-block canvas style DiffusionGemma uses. Named in the AI Mode answer. |
| "Micro Diffusion" (~150 lines NumPy) | toy | Named in the AI Mode answer; not found by a GitHub search under that name, treat as unverified. |
| `jaisidhsingh/nano-dlm` | small, JAX | "nanoGPT for diffusion LMs", MDLM objective on OpenWebText. |
| `kuleshov-group/mdlm` (713★) | 110M–170M reference | The NeurIPS 2024 MDLM code with AR/SEDD/D3PM baselines; DiT or AR-transformer backbone. |
| `ZHZisZZ/dllm` (paper 2602.22661) | framework | "minimal, reproducible recipes for building small DLMs from scratch" and for converting any BERT or AR LM into a DLM. |
| `s-sahoo/scaling-dllms` (ICML 2026) | 14M–1.7B sweep | Scaling-law code for AR/MDLM/Duo/Eso-LM; released checkpoints are 1.7B only. |
| `HKUNLP/DiffuLLaMA` | 124M GPT-2 → DiffuGPT | AR-to-diffusion adaptation, the recipe Dream 7B and (very likely) DiffusionGemma use. |

The architectural recipe is identical across all of them and is five lines on top of
nanoGPT: drop the causal mask, add a `[MASK]` token, corrupt a random fraction
`t ~ U(0,1)` of positions, predict the originals, weight the masked-position
cross-entropy by `1/t` (this makes it an ELBO; dropping the weight gives the ~12%
cheaper plain-CE variant of §3.3). Sampling is iterative confidence-based unmasking.

## 2. What DiffusionGemma actually is

Sources: `ai.google.dev/gemma/docs/diffusiongemma`, HF card
`google/diffusiongemma-26B-A4B-it`, vLLM blog 2026-06-10, `google-deepmind/gemma`
`gemma/diffusion/`.

- One size only: 25.2B total / 3.8B active, 128 experts + 1 shared, 8 active; 30 layers;
  262K vocab; sliding window 1024; context to 256K. Apache 2.0. Docs last updated
  2026-06-10.
- **Uniform-state, not masked, diffusion.** The Gemma docs page "Diffusion in Text
  Generation Explained" (`ai.google.dev/gemma/docs/diffusiongemma/explained`) states that
  DiffusionGemma "implements Uniform State Diffusion": the canvas starts as random
  vocabulary tokens, and unaccepted positions are re-noised by resampling random tokens
  rather than being replaced with a `[MASK]`. That puts it in the Duo family of Sahoo et
  al. (ICML 2026), whose likelihood scaling gap to AR is the *larger* one (≈23x vs ≈16x for
  masked) but which sits on a better speed-quality Pareto frontier, consistent with a model
  sold on throughput. The nano repos in §1 are all masked/absorbing, so none of them is a
  literal miniature of DiffusionGemma's corruption process.
- **Hybrid, not pure diffusion.** An *autoregressive encoder* (causal attention) prefills
  the prompt and writes the KV cache; a *decoder* mode of the same weights runs
  bidirectional attention over a 256-token "canvas" and reads that cache. A finished canvas
  is re-encoded causally and appended. vLLM notes the committed KV "is identical to
  autoregressive KV", so prefix caching works unchanged. Generation is therefore
  block-autoregressive: parallel inside a block, left-to-right across blocks.
- Sampler: ≤48 denoising steps per canvas, typically 12–16 with adaptive stopping;
  Gumbel-max candidates at every position; accept positions from most to least confident
  until an accumulated-entropy budget (0.1) is spent; fully renoise the rest; stop when
  the argmax canvas is stable for two steps and mean entropy < 0.005. Self-conditioning
  feeds the previous step's softmax back in through a gated MLP.
- Quality is **below** the AR Gemma 4 26B-A4B on every published benchmark but one
  (MMLU Pro 77.6 vs 82.6, GPQA Diamond 73.2 vs 82.3, LiveCodeBench 69.1 vs 77.1,
  BigBench Extra Hard 47.6 vs 64.8; HLE-no-tools 11.0 vs 8.7 is the exception).
  The selling point is throughput: ~15–20 tokens per forward, >1100 tok/s at batch 1 on
  H100 FP8, ~5–6x an AR baseline.
- **Training is not published.** Neither the docs, the model card, nor the vLLM post says
  whether it was trained from scratch or adapted from Gemma 4 AR weights, nor for how many
  tokens. The shared-weight causal-encoder design, the "identical to AR KV" property, and
  the fact that the public `hackable_diffusion_adapter` only does LoRA/full SFT on 128- or
  256-token canvases all point to AR-initialised adaptation (the Dream 7B / DiffuLLaMA
  pattern), but that is an inference, not a documented fact.

So "downsizing DiffusionGemma" would mean: take a small AR model, add a mask token, and
fine-tune it into a block denoiser. That is exactly DiffuGPT (124M) and the nano repos
above. There is no small DiffusionGemma checkpoint and no from-scratch recipe to shrink.

## 3. Measured performance vs AR, smallest scales first

### 3.1 Matched 10M models (nanoDLM, character level)

Same architecture, params, steps, data. MDM ELBO NLL ≤1.097 vs AR 0.527 nats/char. Even
allowing for ELBO looseness (DUEL, §5, finds the true gap shrinks by up to a third), the
diffusion model is far behind at this scale and budget.

### 3.2 110M–170M models, 33B–262B tokens (MDLM, SEDD, BD3-LM)

| benchmark | AR | SEDD | MDLM | BD3-LM |
|---|---|---|---|---|
| LM1B, 33B tokens, 110M | 22.32 | ≤32.79 | ≤27.04 | |
| LM1B, 327B tokens, 110M | 20.86 | | ≤23.00 | |
| OWT, 262B tokens | 17.54 | ≤24.10 | ≤22.98–23.21 | L'=16 ≤22.27, L'=8 ≤21.68, L'=4 ≤20.73 |

The gap narrows with tokens (27.04→23.00 on LM1B is 10x the data) but does not close.
Block diffusion interpolates toward AR as the block shrinks; at block size 1 it is AR "in
expectation" yet still shows "a two point perplexity gap" on LM1B from the variance of the
diffusion objective.

### 3.3 Scaling laws, 14M–2.1B parameters, 6e18–1e20 FLOPs

- Nie et al. 2024 (19M–2121M on SlimPajama): MDM loss follows a power law "at a rate
  similar to ARMs" but "MDMs still require approximately 16 times more computational
  resources than ARMs to achieve comparable validation losses"; optimal MDM size ≈ half
  the AR size at a given budget.
- Sahoo et al., ICML 2026 (14M–1708M, exact FLOP counting): MDLM ≈16x (≈14x with the
  plain-CE loss, the "12% more FLOPs-efficient" claim), Duo (uniform-state) ≈23x, Eso-LM
  ≈32x; compute-optimal diffusion models ≈2x smaller than AR. IsoFLOP curves are
  "well-behaved" down to their smallest 14M configuration; nothing suggests the gap
  shrinks at small scale.
- Ni et al. 2025 ("Super Data Learners", 1B–8B): DLMs "require at least about two orders of
  magnitude (>100x) more FLOPs than AR models to reach optimal performance"; with a
  linear mask schedule AR tokens receive "approximately twice the expected signals".

### 3.4 When diffusion does win: repeated data

Prabhudesai et al., NeurIPS 2025 (7M–2.5B, 25–100M unique tokens, up to 800 epochs):
diffusion's effective data-repetition half-life R_D* ≈ 500 vs ≈ 30 for AR; fresh-data
equivalence holds "up to ≈4 epochs" for AR and "≈100 epochs" for diffusion; the critical
compute where diffusion overtakes AR scales as C_crit ∝ U^2.174 in unique tokens U. At the
single-epoch compute-optimal point diffusion is far worse (they print 10.65 vs 7.07 for
AR); only after hundreds of epochs does it flip (3.55 vs 3.71). Their summary:
"if you are compute-constrained, use autoregressive models; if you are data-constrained,
use diffusion models." Ni et al. see the same crossover but only when unique data is
≤1.5B tokens repeated for tens of epochs, and note AR now tolerates ~10 epochs with mild
degradation.

## 4. Translating to this competition's budget

From the current locked run (`t04-stack-fullbudget`, 2026-09-10): 56M-parameter model,
0.81 s/step, 131,072 tokens/step, ~2200 steps, ~288M tokens, 0.91 epochs of the 316M
unique public train tokens, ~8% of bf16 peak on one Trn2 chip.

| quantity | value |
|---|---|
| training compute C = 6·N·D | 6 × 56e6 × 288e6 ≈ **1e17 FLOPs** |
| smallest budget in any diffusion scaling study | 6e18 (60x larger) |
| epochs | 0.91 (AR-favoured regime; crossover needs ≫10) |
| 100 epochs of the public shards | 31.6B tokens ≈ 110 full budgets |

Two independent ways to price a masked-diffusion train.py at this budget:

- **Compute-multiplier route.** A 16x gap means an MDLM at 1e17 FLOPs matches an AR model
  trained on 1/16 of the run, ≈138 steps. This notebook's exchange rate is
  bpb ≈ a − 0.077·ln(steps), so the penalty is 0.077·ln(16) ≈ **+0.21 bpb**, i.e. ~1.25
  ELBO-bpb against the AR 1.034. (Exchange rate is fitted on AR runs; treat as order of
  magnitude.)
- **Perplexity-ratio route.** The asymptotic OWT gap ln(23.21/17.54) = 0.28 nats/token.
  This tokenizer averages ≈4.0 bytes/token (loss 2.87 nats ↔ 1.034 bpb), so 0.28 nats ≈
  **+0.10 bpb** even with 262B tokens of training; the small-data LM1B gap
  ln(27.04/22.32)=0.19 and the 10M-model 2x gap say the penalty grows as the budget
  shrinks.

Either way the projection is 1.15–1.30 bpb-equivalent, worse than every AR submission
this team has made, before the scoring problem in §5.

Hardware side, for completeness: on Trn2 a bidirectional model saves the causal mask
(SDPA already runs the fused fast path, so little to gain), needs no KV cache (irrelevant
for training), and supervises only ~50% of positions per step at the same forward/backward
cost, so tokens-of-signal per second halve. The `1/t` weighting also makes the loss scale
vary per step, which interacts badly with the fused softcap-CE NKI kernel and with the
recompile-on-new-scalar trap documented in the notebook.

## 5. The blocker: the scorer cannot score a diffusion model

From `prepare.py` and README (unchanged upstream, md5-verified):

- `evaluate_bpb` does `logits = model(x)` once per batch and computes
  `F.cross_entropy(logits, y)` on the *clean* input. The contract for `load_for_eval` is a
  "pure, stateless, causal forward — logits must be a function of the input tokens only".
- `causality_check` flips the second half of the sequence and requires the logits of the
  first half to move by ≤1e-2. Any model with bidirectional attention fails this by
  construction.
- A masked diffusion model has no meaningful output for a clean, unmasked input: it was
  trained to predict masked positions from visible ones. Its likelihood is either the
  ELBO (many noised forwards, Monte Carlo over `t`) or DUEL's exact likelihood
  (arXiv 2603.01367, March 2026), which needs a deterministic unmasking *sequence* of
  forwards. Neither is one causal pass, and neither would be computed by the organizers'
  fixed `evaluate_bpb`.
- Block diffusion with block size 1 reduces to AR (with a variance penalty). Eso-LM with
  α₀=1 is AR. In other words, the only diffusion models that *can* be scored here are the
  ones that have stopped being diffusion models.

Anything non-causal in the forward is also an explicit rules violation for Phase 1
("pure, stateless, causal forward").

**On Feng et al., "Theoretical Benefit and Limitation of Diffusion Language Model"
(arXiv 2502.09622, NeurIPS 2025).** The paper proves that a well-trained MDM reaches
near-optimal *perplexity of its samples* in a number of sampling steps that does not grow
with sequence length, but that the *sequence error rate* needs steps linear in length,
which erases the speed advantage over AR for tasks needing fully correct outputs. Both
halves are about **sampling cost at inference**. They say nothing about how much training
compute an MDM needs to reach a given held-out likelihood (§3.3), and the "perplexity" in
the favourable half is generative perplexity of samples under an external scorer, not the
model's own bits-per-byte on held-out text. The result is therefore orthogonal to this
competition: our scorer never samples, and the training-compute gap and the causal-forward
contract stand as before. Its practical lesson is for Phase 2-style generation tasks, where
it says MDMs are not a free lunch either.

## 6. Does anything transfer to the AR model we can submit?

What the rules constrain is only the **evaluation forward**: one causal pass, logits from
tokens. The **training objective is free**. So a diffusion-flavoured *auxiliary loss* on
an otherwise causal model is legal; the question is only whether it lowers next-token loss
per unit of the 30-minute budget. Checked against the notebook and the frontier plan:

- **Any-order / permutation training then left-to-right eval**: the model would spend
  capacity on orders the scorer never uses; DUEL's own oracle-order result (36.5 vs 52.1
  PPL) shows how much the order matters. Negative expected value.
- **Masked-token or multi-token auxiliary losses (MTP)**: the closest AR-compatible cousin
  of the diffusion objective. A masked diffusion model is trained to predict hidden tokens
  from bidirectional context; the causal analogue is predicting several *future* tokens
  from the left context, which is exactly multi-token prediction. That is the only reason
  MTP appears in this document. Multi-token prediction was already tested and removed by the
  best-known solution on this exact task (Recursive, plan §11.1, "hurts step count in
  5-min budget"); this repo has it queued last as T3.7 and has not run it. Diffusion adds
  no new argument for moving it up.
- **Data-repetition advantage**: irrelevant at 0.91 epochs. It would only matter if the
  rules shrank the training set or the budget grew ~100x.
- **Self-conditioning / entropy-bounded acceptance**: inference-time tricks; the scorer
  never samples.
- **Bidirectional context**: exactly what the causality check forbids.

Conclusion: no Phase 1 lever. Phase 2's CORE evaluation is also run through the same
`load_for_eval` forward, so the same constraint holds there.

## 7. If you still want the experiment

The cheapest honest test is `nanoLLaDA` because it already reads ClimbMix-400B: train its
d4 (20M) or d12 (135M) config and this repo's `train.py` on the same public shards for the
same wall-clock, then compare the MDLM ELBO (its `--eval val`) with this repo's
`evaluate_bpb` in nats/token, remembering the ELBO is an *upper bound* that flatters
nothing. Expected outcome from §3–§4: the diffusion loss lands 0.3–0.6 nats/token above
the AR loss at this budget. It would settle the curiosity, not move the leaderboard.

## 8. Sources

- DiffusionGemma docs: https://ai.google.dev/gemma/docs/diffusiongemma ; model card
  https://huggingface.co/google/diffusiongemma-26B-A4B-it ; vLLM blog
  https://vllm.ai/blog/2026-06-10-diffusion-gemma ; adapter
  https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion ;
  https://github.com/google/hackable_diffusion
- Gemini Diffusion (May 2025): https://deepmind.google/models/gemini-diffusion
- MDLM, Sahoo et al., NeurIPS 2024: arXiv 2406.07524 ; code kuleshov-group/mdlm
- SEDD, Lou et al.: arXiv 2310.16834 · MD4, Shi et al.: arXiv 2406.04329
- Block Diffusion (BD3-LM), Arriola et al., ICLR 2025: arXiv 2503.09573 ; https://m-arriola.com/bd3lms
- Scaling up Masked Diffusion Models on Text, Nie et al.: arXiv 2410.18514
- Scaling Beyond Masked Diffusion Language Models, Sahoo et al., ICML 2026: arXiv 2602.15014 ; s-sahoo/scaling-dllms
- Diffusion Beats Autoregressive in Data-Constrained Settings, Prabhudesai et al., NeurIPS 2025: arXiv 2507.15857
- Diffusion Language Models are Super Data Learners, Ni et al.: arXiv 2511.03276
- LLaDA, Nie et al.: arXiv 2502.09992 · Dream 7B: arXiv 2508.15487 · DiffuGPT/DiffuLLaMA: arXiv 2410.17891
- Eso-LM: arXiv 2506.01928 · Duo: arXiv 2506.10892
- DUEL exact likelihood for masked diffusion: arXiv 2603.01367
- Generative Frontiers (evaluation of DLMs at GPT-2-small scale): arXiv 2604.02718
- dLLM framework: arXiv 2602.22661
- Nano repos: nathan-barry/tiny-diffusion, SoumilBhandari/nanoDLM, Lukas-Xue/nanoLLaDA,
  jaisidhsingh/nano-dlm
- This repo: `prepare.py` (`evaluate_bpb`, `causality_check`), README "Submission Format",
  `research/run-logs/t04-stack-fullbudget.log`, `research/trn2-experiment-results.md`
  (C2 token count, exchange rate), `research/frontier-research-plan-2026-09-09.md` §11.
