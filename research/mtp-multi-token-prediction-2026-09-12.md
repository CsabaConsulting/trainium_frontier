# Multi-token prediction (MTP) for the Phase 1 val_bpb push — evidence review

Research pass on 2026-09-12 (local box, no hardware). Starting points were three Google AI
Mode answers the user pasted (how to bolt MTP heads onto nanoGPT; DeepSeek's DSpark; whether
MTP speeds training). This document checks those claims against primary sources and against
the one metric that matters here: next-token bits per byte from a single causal forward, at
a 56M-parameter model trained for ~2200 steps / ~288M tokens in 30 minutes on one Trn2 chip.

## 0. Short answer

1. **MTP is legal here** and, unlike diffusion, needs no eval-side change: the extra
   heads or extra loss terms are training-only, `load_for_eval` returns the same causal
   model, and the scorer never sees them (§2).
2. **The AI Mode answers are right about mechanics and wrong about the payoff.** They are
   right that MTP slows every step and that heads can be trained jointly or post-hoc. Their
   claims of "richer gradient signal", "better sample efficiency", and "same loss in fewer
   tokens" are not supported for next-token likelihood at small scale. Every controlled
   measurement on NTP loss or bpb that I could find at ≤3B parameters is neutral or
   negative (§3).
3. **Direct bpb evidence.** DeepSeek-V3's own ablation: Pile-test BPB 0.729→0.729 (15.7B)
   and 0.658→0.657 (228.7B) with MTP, while benchmarks improve. Aynetdinov & Akbik (ACL 2025):
   at 1.3B, a 2-head MTP model is 3.7% worse on MiniPile BPB and 7.5% worse on LAMBADA BPB
   than the NTP baseline; 4 heads are ~10–20% worse; curricula reduce the damage but "do not
   lead to performance improvements over the NTP baseline". Karpathy's nanochat d12 port:
   per-step "same/slightly worse", wall-clock "noticeably worse". Parameter Golf controlled
   tests: +0.0018 and +0.028 bpb worse; one un-replicated −0.0037 claim (§3.2–3.4).
4. **The one positive** is the modded-nanogpt speedrun (record #53, still in the current
   record): a *parameter-free* MTP that reuses the same logits, weights `[1, 0.5, 0.25→0]`
   on the next three tokens during the first third of training, then anneals to plain NTP.
   Its isolated effect was never ablated (the PR bundled four changes and says "TODO:
   Include ablations"), it needed a fused CE kernel to be cheap (record #60), and the
   whole record was worth 80 steps of ~1900 (≈3 s of 120 s). That is the only variant worth
   considering here (§3.5, §5).
5. **Cost model for this train.py.** The unembedding is ~12% of forward FLOPs, so each
   extra *linear* head costs ~12% step time ≈ +0.009 bpb at the notebook's exchange rate,
   and a DeepSeek-style extra block ~17% ≈ +0.012. The parameter-free variant costs only
   extra work inside the softcap-CE kernel, which is unmeasured but small, provided the NKI
   kernel is extended to take k targets; calling the existing kernel k times would cost
   k× the kernel's share (§4).
6. **Verdict: keep T3.7 last.** Expected effect on val_bpb is between −0.004 and +0.03 with
   a median near zero, against a floor cost of ≥ +0.0015 and a kernel-engineering day. If it
   is ever tried, only the parameter-free, unnormalized, early-annealed variant, and only
   as a paired 900 s A/B with two seeds because the plausible effect is at the noise floor
   (§6). Recursive reached the same conclusion on this exact task: "JEPA MTP removed:
   multi-token prediction hurts step count in 5-min budget" (§3.6).

## 1. What "MTP" means — the variants matter more than the name

| variant | extra parameters | extra compute per step | who uses it | purpose |
|---|---|---|---|---|
| **Parameter-free shared-logits** (modded-nanogpt #53) | none | k−1 extra gathers + weighted CE on the *same* `[T, V]` logits; near-zero FLOPs if fused into the CE kernel | modded-nanogpt records #53 → current; nanochat port (negative) | training signal only |
| **Independent linear heads** on the trunk state | (k−1)·d·V, discarded at export | one extra `d×V` matmul + CE per head, fwd and bwd | Parameter Golf #88/#212/#1031/#1094; ACL 2025 "LL heads" | training signal; self-speculative decoding |
| **Heads as transformer layers** (Gloeckle et al. 2024) | one block per head; they remove trunk layers to match params | 1.07–1.22× training time (Table S5) | Meta 0.3B–13B study | quality at scale; 2–3× decoding |
| **Sequential module** (DeepSeek-V3, "MTP-1") | one block + `d×2d` projection per depth; shares embed/unembed | ≈ one extra layer per depth | DeepSeek-V3, Qwen3-Next, Step 3.5, Nemotron 3 | auxiliary loss + speculative decoding |
| **Register tokens** (MuToR, NeurIPS 2025) | negligible | longer sequences | fine-tuning studies | aligns with NTP objective |
| **Medusa / EAGLE / DFlash / DSpark** | draft heads or a small draft model | post-hoc, base frozen | serving stacks | inference speed only |

DSpark (arXiv 2607.05147, DeepSeek + PKU, July 2026) is a speculative-decoding framework: a
parallel drafter plus a lightweight sequential "Markov" module to fix suffix decay, and
confidence-scheduled verification lengths. It claims 60–85% faster per-user generation in
DeepSeek-V4 serving at matched throughput. It makes no claim about the base model's
next-token quality, and its abstract does not describe joint pretraining. For a competition
that never samples, DSpark, DFlash, Medusa and EAGLE are out of scope entirely.

## 2. Rules and scorer: MTP is allowed, and free at eval

- README "Submission Format": `load_for_eval` must return a model with a "pure, stateless,
  causal forward — logits must be a function of the input tokens only". Extra training
  heads that are dropped (or simply not called) satisfy this; the DeepSeek report itself
  notes the modules "can be directly discarded" at inference.
- `prepare.py evaluate_bpb` calls `model(x)` once and computes CE from the returned logits.
  Only the main head's logits are scored. `causality_check` is unaffected because the trunk
  stays causal.
- Parameter Golf's rules are the same in this respect and its record submissions ship MTP
  heads "discarded at export" (#88, #1031, #1094). Recursive's `optimized_from_karpathy.py`
  has a comment at the `lm_head` line: `# JEPA MTP removed: multi-token prediction hurts
  step count in 5-min budget`, i.e. removed for cost, not legality.

So the whole question is empirical: does an MTP loss lower *next-token* loss per unit of
wall-clock at this scale?

## 3. Evidence, ordered by relevance to next-token bpb at small scale

### 3.1 Gloeckle et al. 2024 (Meta), arXiv 2404.19737 — the paper the AI Mode answer leans on

- Fig. 3 / Table S7 caption: "Multi-token prediction models are worse than the baseline for
  small model sizes, but outperform the baseline at scale." At 0.3B, n=4 drops MBPP@1 from
  1.8 to 1.0 and MBPP@100 from 29.9 to 20.1; at 0.6B still worse; 1.3B roughly break-even;
  gains start at 3B and grow to 13B.
- Natural language (7B, 200B tokens): "the 2-future token prediction model performs on par
  with the next-token prediction baseline"; "The 4-future token prediction model suffers a
  performance degradation." "Larger model sizes might be necessary to see improvements."
- The paper reports no next-token perplexity comparison at all; its benefits are
  generative/code benchmarks and 2–3× self-speculative decoding.
- Cost: n=4 training time 1.22× at 0.3B, 1.13× at 0.6B, 1.07–1.09× at ≥3B (Table S5).
- Tokenization: gains are largest at byte level (n=8 bytes, 7B); the authors say "optimal
  vocabulary sizes for multi-token prediction are likely different from those for next-token
  prediction". This repo's 8192-token vocabulary averages ≈4 bytes/token, between the two
  regimes.

Their own summary sentence: "We believe this usefulness only at scale to be a likely reason
why multi-token prediction has so far been largely overlooked." Our model is 56M parameters,
five times smaller than their smallest and worst-affected size.

### 3.2 DeepSeek-V3 (arXiv 2412.19437) — the only large ablation that reports BPB

MTP-1 (one sequential module, λ=0.3 for 10T tokens then 0.1) vs no MTP, same data:

| | small MoE (15.7B, 1.33T tok) | large MoE (228.7B, 540B tok) |
|---|---|---|
| Pile-test BPB | 0.729 → **0.729** | 0.658 → **0.657** |
| MMLU 5-shot | 50.0 → 53.3 | 67.5 → 66.6 |
| HumanEval | 20.7 → 26.8 | 44.5 → 53.7 |
| GSM8K | 25.4 → 31.4 | 72.3 → 74.0 |

The benchmark gains are real; the held-out likelihood does not move even at 15.7B–228.7B
parameters and trillion-token budgets. This is the cleanest statement that MTP's benefit
is not a lower next-token loss.

### 3.3 Aynetdinov & Akbik, ACL 2025 — small subword models, NTP metrics, 1 epoch

1.3B and 3B models, FineWeb-Edu, one epoch, evaluated on NTP tasks with the main head only
(Table 1, static MTP without curriculum vs NTP baseline "1 LL"):

| model | heads | MiniPile BPB | LAMBADA BPB |
|---|---|---|---|
| 1.3B | 1 (NTP) | **1.08** | **1.34** |
| 1.3B | 2 linear | 1.12 (+3.7%) | 1.44 (+7.5%) |
| 1.3B | 2 transformer-layer | 1.11 | 1.41 |
| 1.3B | 4 linear | 1.19 (+10%) | 1.61 (+20%) |
| 3B | 1 (NTP) | **1.03** | **1.17** |
| 3B | 2 linear | 1.06 | 1.27 |
| 3B | 4 linear | 1.12 | 1.44 |
| byte-level 1.3B | 1 (NTP) | 1.14 | 1.06 |
| byte-level 1.3B | 4 linear | 1.16 | 1.09 |
| byte-level 1.3B | 8 linear | 1.23 | 1.19 |

Their forward and reverse curricula claw back part of the loss (reverse: −6% MiniPile BPB
relative to static 4-head MTP) but the paper states the curricula "do not lead to performance
improvements over the NTP baseline when applied to subword-level models of both sizes". This
is the closest published analogue to our metric and regime (small model, single epoch), and
it is negative even at 1.3B–3B.

### 3.4 Practitioner reports at nanoGPT-to-nanochat scale

- **nanochat, Karpathy, `dev/LOG.md` 2026-01-12** (d12 ≈ 135M): ported the modded-nanogpt
  schedule; "Weights normalized to sum to 1". GPU memory 34→47 GB, MFU 41→40%,
  val/bpb per step "same/slightly worse", val/bpb by wall clock "noticeably worse".
  "Negative result for nanochat ... pure overhead at the moment." (Discussion #481 lists it
  under "What didn't work": "+13GB memory, no improvement".)
- **Parameter Golf** (16 MB models, ~25M params, 600 s on 8×H100, bpb metric):
  - #212 controlled test: auxiliary t+2 head at 0.5× weight → **+0.0018 bpb**, 3% slower.
  - #375 ($500 systematic study, 13 techniques): MTP **+0.028 bpb**, "throughput penalty
    kills it".
  - Issue #140 summary: "MTP (#212, #236: 86ms)" vs a 67 ms baseline step, filed under
    "throughput cost exceeds quality gain".
  - #1031: 2 heads at weight 0.1, discarded at export, **−0.0037 bpb** claimed, 1 seed,
    flagged "unvalidated" by the maintainers; carried into the current record #1094.
  - The Parameter Golf retrospective paper (arXiv 2607.01517, 1,430 submissions) does not
    list MTP among its 84 tracked technique flags at all.
- **Recursive** (best-known result on this repo's exact task, plan §11): removed MTP from
  Karpathy's baseline for step-count reasons (§2 quote).

### 3.5 modded-nanogpt record #53 (Dec 2025) — the positive case, examined

- Implementation (PR #178, still in `train_gpt.py` today): no new parameters. Compute
  `logsumexp` once per position, gather the logits of the next `n_predict` targets, and take
  `loss = (cross_entropy * mtp_weights).sum()` with weights **not normalized**: the NTP term
  keeps weight 1 and the extra offsets add on top. Schedule tied to the batch-size stages:
  `[1, 0.5, 0.25→0]` in the first third, `[1, 0.5→0]` in the second, `[1]` after.
- Result: 80 fewer steps of ~1900 at "slightly higher step time", ≈3.06 s of a 122.8 s run,
  **bundled with** untying embed/lm_head at 2/3 of training, an Adam CWD change, halved
  attention init, and an LR-schedule tweak. The record README ends with "TODO: Include
  ablations for the above"; none were published, and no issue or PR in the repo since
  contains an MTP ablation.
- Record #60 (Jan 2026) had to write a fused Triton softcap+MTP cross-entropy kernel to
  remove the `unfold`/`gather` materialization cost; record #75/#79 tuned it further. The
  parameter-free MTP is cheap only once the CE kernel handles k targets natively.
- Author's stated intuition: early training "is learning the n-gram distribution"; MTP
  "increases the signal, and can be considered a cheap proxy of increasing the batch size".
- Karpathy's port that failed differs in exactly one visible way: it normalized the weights
  to sum to 1, which dilutes the NTP term to 1/1.75 early on. Whether that explains the
  disagreement is untested; it is the first thing any replication should control.

### 3.6 Summary table (effect on next-token loss / bpb, main head only)

| source | scale | variant | effect | cost |
|---|---|---|---|---|
| DeepSeek-V3 ablation | 15.7B / 228.7B | sequential MTP-1 | 0.000 / −0.001 BPB | ~1 extra layer |
| ACL 2025 | 1.3B / 3B, 1 epoch | 2 linear heads | +3.7% / +2.9% MiniPile BPB | 1 extra head |
| ACL 2025 | 1.3B | 4 linear heads | +10% | 3 extra heads |
| Gloeckle 2024 | 0.3B–1.3B | 4 layer-heads | worse on code; no PPL reported | 1.12–1.22× time |
| Gloeckle 2024 | 7B NL | n=2 / n=4 | "on par" / "degradation" | 1.02–1.07× |
| nanochat d12 | ~135M | param-free, normalized | same/slightly worse per step; worse per wall-clock | +38% memory, −1pt MFU |
| Parameter Golf #212 | ~25M | t+2 head, w=0.5 | +0.0018 bpb | +3% step |
| Parameter Golf #375 | ~25M | heads | +0.028 bpb | large |
| Parameter Golf #1031 | ~25M | 2 heads, w=0.1 | −0.0037 bpb (1 seed) | unreported |
| modded-nanogpt #53 | 124M-class speedrun | param-free, unnormalized, annealed | ≤ −80 steps of 1900, confounded | needs fused kernel |
| Recursive (this task) | ~50M, 5 min | (Karpathy baseline's MTP) | removed: "hurts step count" | — |

## 4. Cost model for this repo's train.py

Locked shape: n_embd 640, 6 layers, MLP 4×, vocab 8192, SEQ_LEN 2048, 55.7M params,
0.81 s/step, 131,072 tokens/step, ~2200 steps. Exchange rate from the notebook:
Δbpb ≈ 0.077·ln(steps ratio), so a step-time overhead x costs ≈ 0.077·ln(1+x).

Forward FLOPs per token (analytic): trunk matmuls 2×29.5M ≈ 59M; causal attention
≈ 2·T·d per layer ≈ 15.7M; unembedding 2·640·8192 ≈ 10.5M. **The lm_head is ≈12% of forward
matmul FLOPs**, and the same share of backward.

| variant | step-time overhead (estimate) | bpb cost at 0.077·ln(1+x) | needed gain to break even |
|---|---|---|---|
| parameter-free, k targets fused into the NKI softcap-CE kernel | ~1–3% (extra gathers/reductions over `[N, 8192]`; kernel work required) | 0.0008–0.0023 | > 0.002 |
| parameter-free, calling the existing kernel k times | k × (CE kernel share, unmeasured; likely 2–5%) → 4–15% for k=3 | 0.003–0.011 | > 0.01 |
| one extra linear head (Parameter Golf style) | ~12% fwd+bwd + one more CE | ≈ 0.009 | > 0.01 |
| two extra linear heads | ~24% | ≈ 0.017 | > 0.02 |
| DeepSeek-style block (1 of 6 layers) + projection | ~17–20% | ≈ 0.012–0.014 | > 0.014 |
| Gloeckle-style layer heads, parameter-matched | trunk shrinks 6→5 layers | DEPTH=5 was already measured worse in the Tier-0 sweep | — |

Additional Trainium-specific costs, all documented in the notebook: every new per-step
Python float reaching the device recompiles the graph, so the annealed weights must live in
a device tensor updated in place (same trap as the LR multiplier, `USE_TENSOR_LR_SCALARS`);
the fused softcap-CE NKI kernel is an autograd `Function` island outside `torch.compile`,
so any extension is hand-written NKI with its own equivalence harness
(`research/scratch/softcap_ce_equiv.py`); and `USE_DOC_MASK` packing means the t+2 target
can belong to the next document, which the loss must either mask (needs `ignore_index`
support the kernel currently lacks) or tolerate as label noise.

Against those costs, the measured next-token gains in §3 range from −0.004 (one
unreplicated seed) through 0 (DeepSeek, nanochat per step) to +0.002…+0.03 (Parameter
Golf, ACL). At this repo's paired-900 s noise floor of ~0.001–0.002 bpb, even the optimistic
case is barely resolvable.

## 5. What is different about our regime, honestly

Arguments *for* a small effect here that the literature does not cover:

- We are heavily undertrained: ~5 tokens per parameter versus 20+ in every study above.
  The modded-nanogpt author's "n-gram signal early" intuition applies most in the first few
  hundred steps, which are a larger fraction of a 2200-step run than of a 1T-token run.
- Our vocabulary is small (8192, ≈4 bytes/token). Gloeckle found MTP most useful at byte
  level; the ACL byte-level static MTP was still worse on NTP bpb, but by less than subword.
- The hashed n-gram value-embedding experiment (T3.1) showed a real −0.031 bpb quality gain
  from injecting n-gram signal, lost only to its throughput cost. Parameter-free MTP is a
  cheaper way to push n-gram statistics into the same weights, so the idea is not absurd.

Arguments *against*, which dominate:

- Every NTP-loss measurement at ≤3B is neutral or negative, including a single-epoch study
  built specifically for small models (§3.3), and the one large ablation with BPB shows zero.
- The only positive report is confounded, unablated, and from a 124M-class model with a
  fused kernel we do not have.
- Recursive, on this exact task and budget, removed it.
- The schedule already anneals LR from 25% of the run; MTP's "early boost" competes for the
  same early steps where the LR is highest and the loss is already dropping fastest.

## 6. Recommendation

Keep T3.7 where the plan put it: last in Tier 3. The Tier 1 throughput items (each 1% of
step time is worth 0.0008 bpb for certain) and the eval-context finding (scorer at 1024
context, worth ~0.016) have higher expected value per hour of box time.

If it is tried anyway, the only defensible experiment is:

1. **Variant:** parameter-free, shared logits, **unnormalized** weights, k=3 →2 →1 annealed
   over the first two thirds, exactly the modded-nanogpt schedule, implemented by extending
   `_softcap_ce_fwd/_bwd` to accept `tgt [N, k]` and weights `[k]`
   (forward: `loss_i = Σ_k w_k (lse_i − z_i[t_{i+k}])`; backward:
   `dz_i = (Σ_k w_k)·softmax(z_i) − Σ_k w_k·onehot(t_{i+k})`), bit-exact to the current
   kernel when k=1 and w=[1]. Verify with the existing equivalence harness and a 15-step
   causality smoke.
2. **Weights as a device tensor** updated in place per step, never a Python float.
3. **Measurement:** two paired 900 s A/B runs each arm, same day, per the notebook
   protocol; the decision bar is the usual non-overlapping step counts and bpb. Expect the
   step count to drop by the kernel overhead and the bpb effect to sit at ±0.002.
4. **Do not** test linear heads, DeepSeek blocks, or normalized weights: the first two are
   priced out by §4 before any quality effect, and the third is the configuration that
   already failed in nanochat.

Also worth noting for the write-up of the AI Mode answers: "self-speculative decoding
2–3× faster" and the DSpark material are inference-side and irrelevant to val_bpb; "better
representation learning … richer gradient signal" is the claim that the DeepSeek BPB
numbers and the ACL table contradict for next-token likelihood; "slower per step" is correct
and is the part that decides the matter under a fixed wall-clock budget.

## 7. Note on the two AI Mode "draft module sizing" answers (added same day)

The user also shared two AI Mode answers that size a DSpark companion drafter for a
nanoGPT-scale target (claims: "0 extra weights for the MTP modules"; a companion module of
"~4.8M weights" for a 768-wide GPT-2 or "~265k" for a 256-wide toy; formulas
`layers × 4 d²` for a parallel backbone, `2 × rank × d` for a rank-256 Markov head,
`3 × d` for a confidence head; frozen copies of `embed_tokens` and `lm_head` bundled into
the draft checkpoint). Assessment:

- **Not applicable.** Every component described (parallel drafter, Markov head, confidence
  head, frozen embedding/unembedding clones, vLLM/SGLang integration) exists to accept or
  reject draft tokens at *generation* time. `evaluate_bpb` never generates; it scores the
  base model's logits on given text. A perfect drafter changes val_bpb by exactly zero, and
  the rules would not even let it into the scored forward.
- **Internally inconsistent on "zero extra weights".** MTP as used by DeepSeek-V3 adds a
  transformer block plus a `d×2d` projection per depth (§1); only the modded-nanogpt
  shared-logits variant is genuinely parameter-free. The answers conflate the two.
- **For the record, on this repo's shape** (d=640, vocab 8192): a 1-layer drafter would be
  ≈4·640² ≈ 1.6M weights, a rank-256 Markov head ≈ 0.3M, and the frozen embed/unembed clones
  5.2M each. None of it is trained or stored in a Phase 1 submission. Phase 2's CORE eval
  also runs through `load_for_eval`'s forward, so it is equally irrelevant there.
- The "DeepSpec" toolkit named in the first answer could not be verified from the DSpark
  abstract or repo listings and should be treated as unconfirmed.

So the user's instinct is correct: like the other exotic lines (SSM/RWKV hybrids for
inference efficiency, diffusion for parallel sampling), the inference-speed half of the
MTP story is worth zero here, and the training-signal half is what §3 evaluates and
finds wanting.

## 8. Sources

- Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction",
  arXiv 2404.19737 (Fig. 3, Tables S5/S7, §3.1).
- DeepSeek-V3 Technical Report, arXiv 2412.19437 (§2.2, §4.2, §4.5.1 Table).
- Aynetdinov & Akbik, "Pre-Training Curriculum for Multi-Token Prediction in Language
  Models", ACL 2025 long, pp. 25573–25588 (Table 1; code github.com/aynetdia/mtp_curriculum).
- Gerontopoulos et al., "Multi-Token Prediction Needs Registers" (MuToR), arXiv 2505.10518.
- Ahn, Lamb, Langford, "Efficient Joint Prediction of Multiple Future Tokens", arXiv 2503.21801.
- Mahajan et al., "Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries",
  arXiv 2510.14751 (ICLR 2026; 3B/8B only).
- DSpark, arXiv 2607.05147; DFlash, arXiv 2602.06036 (ICML 2026).
- modded-nanogpt: PR #178 / record #53 README (2025-12-22), PR #199 / record #60, PR #151
  (Smear-MTP, medium track), current `train_gpt.py` schedule stages.
- karpathy/nanochat `dev/LOG.md` 2026-01-12 "Multi-Token Prediction (MTP)"; Discussion #481.
- openai/parameter-golf: PRs #88, #212, #236, #375, #1031, #1094; issue #140 ("What Doesn't
  Work"); arXiv 2607.01517 "Parameter Golf: What Really Works?".
- recursive-org/first-steps-toward-automated-ai-research,
  `nanochat_autoresearch/solutions/optimized_from_karpathy.py` line 251.
- Raschka, LLM Architecture Gallery: MTP (variant taxonomy; Qwen3-Next, Step 3.5, Nemotron 3).
- This repo: `prepare.py` (`evaluate_bpb`, `causality_check`), `train.py`
  (`_softcap_ce_fwd/_bwd`, `USE_TENSOR_LR_SCALARS`, `USE_DOC_MASK`),
  `research/trn2-experiment-results.md` (phase timing table, T3.1 n-gram VE results,
  exchange rate), `research/frontier-research-plan-2026-09-09.md` §5 T3.7, §11.
