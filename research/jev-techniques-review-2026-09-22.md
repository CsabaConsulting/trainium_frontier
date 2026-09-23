# Jev techniques review (2026-09-22)

Question asked: can any technique from Jev, the TypeSafe AI "System One" decision model, be
used in our Phase 1 recipe? Inputs: the two Gemini deep-research reports
`research/NanoGPT vs Jev.md` and `research/NanoGPT vs Jev vs Trainium NKI.md` (the PDFs carry
the same text). No experiments were run; the Trn2 boxes were occupied. Every claim below cites
either a repo measurement or a source checked today.

## 0. Short answer

No. None of the four proposed techniques improves our scorer target, and two of them are
measured regressions on this exact stack. The reports' premise (padding waste, learned absolute
positions, an SFT/DPO training loop) describes vanilla nanoGPT, not our train.py.

| Gemini item | Claimed gain | Status here | Verdict |
|---|---|---|---|
| 1. Block-causal (intra-document) masking | 1.2–1.5× from eliminating padding | Loader pads nothing; doc masking measured **+0.0063 bpb worse and slower**; any additive mask drops the compiler's NKI SDPA path (**265 → 240 steps at 900 s**) | closed, stays closed |
| 2. Trunk-and-branch prefix sharing | 1.5–2.5× on multi-completion SFT/DPO | Phase 1 is single-pass next-token pretraining; there are no shared prompts with several completions | not applicable |
| 3. Sparse MoE backbone | 2–3× faster convergence | Closed in plan T4.4 on evidence (data-starved experts under the ~288M-token epoch, op-count-bound step at ~8% MFU); the reports add no new evidence | stays closed |
| 4. RoPE with per-document position resets | stability, arbitrary packing | We already run RoPE through the fused NKI rope+QK-norm kernel; resets need per-token cos/sin gathers and are only coherent together with item 1 | no |
| NKI mapping (attention_cte tile skipping, MoE subkernels, RoPE fusion) | up to 50% TensorE cycles | The compiler already lowers our SDPA to the attention_cte/attention_bwd pair; its packing arguments are unreachable from `F.scaled_dot_product_attention`; our own NKI attention integration was correct but 1.9× slower | no |

## 1. What Jev is, and what the reports actually describe

Verified today:

- Jev is a proprietary model from TypeSafe AI, released 2026-09-15 (current version jev-1.13).
  It takes a block of "state" plus typed questions and returns calibrated probabilities over
  declared options in one forward pass; it generates no text. TypeSafe describes a new
  architecture, a "parallel sampler" and a training method called Reinforcement Learning for
  Calibrated Decisions (RLCD). TypeSafe has **not published the architecture, weights or a
  technical report** (Wikipedia "Jev (AI model)"; TypeSafe blog "Introducing System One Models
  & Jev"; MarkTechPost 2026-09-19).
- arXiv carries at least eleven papers from 19–22 September 2026 that use Jev through its API
  (for example 2609.26550 "JEV-as-a-Judge", 2609.23886 "this-that-model-1.0", 2609.26758
  "Type-Safe Is Not Error-Free"). None describes Jev's internals; the open re-implementations
  (JevLite, this-that-model) are LoRA-tuned open-weight LLMs with a logit readout at a
  designated position.
- "Kev" (cited in the Gemini report as the source of "branch position resets") does not appear
  on arXiv, on Wikipedia or in TypeSafe's public material. Treat it as unverifiable.
- The seven "principles" the report numbers ("share the state, isolate the questions",
  "schedule branches as a batch", "sparse capacity") are paraphrases of TypeSafe's API
  documentation: the state is ingested once and every question is scored in parallel and in
  isolation against it. They describe the serving interface, not a training recipe.

So the four techniques are Gemini's reconstruction of how such an interface could be built,
using well-known mechanisms that predate Jev: FlexAttention document masking (modded-nanogpt),
prefix caching/tree attention, sparse MoE and RoPE position ids. They are assessed below on
their own merits against our measurements. The one thing Jev really is, a pointer/option
readout instead of a vocabulary head, is unusable here because the scorer needs full
8192-vocabulary logits from a causal forward (`prepare.evaluate_bpb`).

## 2. Technique by technique

### 2.1 Block-causal (intra-document) masking

**The premise does not hold.** The competition loader (`prepare.make_dataloader`, "BOS-aligned
best-fit packed") fills every row of `seq_len + 1` tokens with whole documents by best fit and
truncates a document to fill the remainder. There is no padding token anywhere in training, so
the 20–50% "padding elimination" gain is exactly zero for us. Our launch uses `--pack-factor 32
--seq-len 1024`: the loader builds 32768-token rows and train.py reshapes them into 32 rows of
1024, so every training row starts wherever the reshape cuts, usually mid-document. Only the
question of cross-document attention leakage remains, and it was answered:

- GPU proxy (2026-09-04, `research/gpu-experiment-results.md` §"Follow-up: document-boundary attention masking"): `USE_DOC_MASK=True`
  built the mask from BOS positions inside `forward` (`doc_id = cumsum(idx == BOS)`), passed
  causality checks, and scored **1.171596 vs 1.165273 (+0.0063 bpb), 863 s vs 680 s**.
  Discarded as a real regression. Karpathy's note on the same question: attending across BOS
  "does not seem to make things much worse".
- Trn2 (`research/gpu-closed-techniques-trn2-revisit-2026-09-12.md` §1, §2.3): the only fused
  fast path is `is_causal=True`. `USE_CAUSAL_FASTPATH` was worth **265 vs 240 steps at the 900 s
  proxy (+10%)**; any additive `[T, T]` mask gives that back. Attention is **~10.6% of the
  step** (fwd 2.6% + NKI flash backward ~8%, `research/trn2-experiment-results.md` "Post-compile
  re-profile"), so even a mask that halved attention work could recover at most ~5% against
  the 10% loss.
- Why the loss happens (`research/fresh-start-code-audit-2026-09-18.md`): torch_neuronx lowers
  `F.scaled_dot_product_attention` to the nki-library `attention_cte` / `attention_bwd` pair
  only when there is **no additive bias**, sequence divisible by 512 and head dim ≤ 128.
  A document mask is an additive bias, so it falls back to the manual decomposition (FP32
  score matrix, full T×T materialisation). FlexAttention is a Triton kernel and does not exist
  on Neuron.
- Eval consistency: the scorer's `make_eval_dataloader` concatenates the validation documents
  into one stream (BOS per document) and slices fixed 1024-token rows, so most eval rows begin
  mid-document and contain several BOS tokens. A train-time document mask would have to be
  reproduced inside `forward` from BOS positions at eval time (as the old `USE_DOC_MASK` did).
  That is legal under the contract (pure function of `idx`), but the leading partial document
  of every eval row has no BOS and would attend to nothing outside itself either way. The
  current unmasked training rows, cut by the pack-32 reshape, already match the scorer's row
  distribution; masking would create a mismatch, not remove one.

Verdict: closed on measurement, and the Trn2 kernel path makes it strictly worse than the
GPU number. Nothing in the Jev report changes the evidence.

### 2.2 Trunk-and-branch prefix sharing

The 1.75× FLOP example assumes a shared 1500-token prompt with two 250-token completions
(SFT, DPO, multi-sample). Phase 1 trains one causal language model on shards 0–4 with plain
next-token cross-entropy; there is no prompt/completion structure, no pairwise preference data
and no rejection sampling. Evaluation is one forward per row. There is no place in the
pipeline where a prefix is evaluated twice, so there is nothing to share.

Verdict: not applicable to this competition phase.

### 2.3 Sparse MoE backbone

Plan `research/frontier-research-plan-2026-09-09.md` T4.4 closed MoE on 2026-09-13 and the
Jev report adds no data point:

- Quality at our scale: Parameter Golf #480 (2-expert MoE) was **−0.06 to −0.08 bpb worse** than
  dense on 8×H100; the Apple MoE scaling laws (ICML 2025) put optimal sparsity at 0 below
  ~500M parameters. The one positive (modded-nanogpt-moe, −1.8% loss) was at 124M dense
  equivalent on 2.4B tokens, about 8× our data.
- Data: the epoch counter wraps near step 2195 at 131,072 tokens per step (~288M tokens;
  `count_train_tokens.py` counts ~316M in the five shards). With k experts each sees ~288M/k
  tokens, which is the data-starved regime where MoE loses.
- Hardware: the step is op-count-bound at ~8% MFU (plan §"single biggest fact"). Routing, top-k,
  permute/gather/scatter and per-expert small matmuls add launches and worse TensorE shapes;
  dynamic per-expert token counts need capacity-factor padding for static shapes. Neuron's MoE
  support (NxD `RouterTopK` / `ExpertMLPs`) lives in the torch-xla NeuronX-Distributed stack,
  not the native path we run, and the scorer needs one consolidated module.
- The report's metric, "2–3× faster convergence per wall-clock hour", is the wrong one for a
  30-minute budget where each optimizer step already costs more launches than FLOPs.
- The nki-library MoE subkernels the report names (Permute Routed Tokens, Blockwise MM) were not
  in the kernels we vendored or audited (`research/fresh-start-nki-library-audit-2026-09-20.md`);
  that audit found the library's TKG/GPT-OSS material to be decode-oriented. A training MoE also
  needs the router backward and the load-balance auxiliary loss, neither of which a forward
  dispatch kernel provides.

Verdict: stays closed.

### 2.4 RoPE with per-document position resets

The report's starting point is nanoGPT's learned absolute `wpe`. Ours is already RoPE
(`ROPE_BASE=100000`) with QK-norm, fused into one NKI kernel (`rope_norm`, cos/sin tables of
shape `[1, T, 1, D/2]` shared by every row). Position resets would mean:

- per-token position ids, so cos/sin become `[B, T, D/2]` gathers per row instead of one shared
  `[T, D/2]` slice: a kernel signature change plus extra HBM traffic on the Vector/Scalar
  engines that the profiles show are already loaded;
- coherence only together with the document mask (§2.1): without it, tokens of different
  documents in the same row would share positions and attend to each other at fabricated
  relative distances, which is worse than the current consistent monotone positions;
- the same eval-consistency problem as §2.1, because scorer rows start mid-document and
  positions would have to be derived from BOS inside `forward`.

Verdict: no. It is a sub-item of §2.1 and inherits its closure.

### 2.5 The NKI mapping claims

- **attention_cte tile skipping and sequence packing.** The kernels do support it: both
  `attention_cte` and `attention_bwd` in the vendored copy
  (`research/vendor/nki_library_92d11f63/core/attention/`) take `bound_min` / `bound_max`
  per-query KV ranges ("Sequence Packing ... Compatible with causal masking"), and the
  2.31_release and 2.32_release files on GitHub carry the same arguments (checked with `gh api`).
  But our stack already runs this exact kernel pair through the compiler's SDPA lowering, and
  `F.scaled_dot_product_attention` has no argument to pass the bounds. Using them means our own
  NKI attention integration. That was done: branch `nki-attn-parked` (commits `5e9043eb`,
  `934f3cbd`) ported a fused causal fwd/bwd SDPA, passed causality and produced an identical
  loss trajectory, and was **1.9× slower per step**. Adding bounds to a slower path cannot win.
- **Prefix caching (`k_prior` / `v_prior`).** Inference semantics: the prior K/V have no gradient
  path, and the kernel doc states packing is "not compatible with prefix caching". Irrelevant
  for §2.2 anyway.
- **RoPE fused into the QKV epilogue.** Already done in substance by the rope+QK-norm kernel;
  the reset variant needs per-token tables (§2.4).
- **Hardware rules** (128-partition, 24 MB SBUF, no divergence): our shapes (seq 1024, head dim
  128, B·T multiples of 128) already satisfy them; they are constraints, not gains.

## 3. Recommendation

Do not spend Trn2 time on any of the four items. If a falsification is ever wanted when the
boxes are free, the cheapest is the existing `USE_DOC_MASK` toggle at the 900 s proxy, with
the expected outcome of ≥10% fewer steps and a worse bpb, consistent with the two measurements
above. The only element of Jev that is genuinely new, the typed-option readout trained with
RLCD, is incompatible with a scorer that consumes full-vocabulary logits.

## 4. Sources checked today

- TypeSafe AI blog, "Introducing System One Models & Jev": architecture undisclosed; RLCD;
  parallel sampler; cardinality up to 255 with a two-stage scoring for large option sets.
- Wikipedia, "Jev (AI model)": released 2026-09-15, jev-1.13.0, "has not published Jev's exact
  architecture, weights, or a technical paper".
- arXiv export API: 2609.22753, 2609.23136, 2609.23886, 2609.23959, 2609.23986, 2609.24052,
  2609.24395, 2609.24965, 2609.26532, 2609.26550, 2609.26758 (all API users or open
  re-implementations; no primary Jev paper; no "Kev").
- aws-neuron/nki-library `attention_cte.py` / `attention_bwd.py` at 92d11f63 (vendored) and at
  2.31_release / 2.32_release (GitHub): `bound_min`/`bound_max` packing, `k_prior` prefix caching.
- Repo: `prepare.py` (`make_dataloader`, `make_eval_dataloader`, `evaluate_bpb`, eval-public
  defaults batch 1 / seq 1024); `research/gpu-experiment-results.md` (document-boundary masking follow-up);
  `research/gpu-closed-techniques-trn2-revisit-2026-09-12.md` (§1, §2.2, §2.3);
  `research/trn2-experiment-results.md` (post-compile re-profile);
  `research/fresh-start-code-audit-2026-09-18.md` (SDPA lowering conditions);
  `research/frontier-research-plan-2026-09-09.md` (T4.4 MoE, ~8% MFU);
  `research/fresh-start-nki-library-audit-2026-09-20.md`; `research/fresh-start-evidence-2026-09-18.md`
  (NKI attention smoke 1.9× slower); branch `nki-attn-parked`.
