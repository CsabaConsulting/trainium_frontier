# DeepSeek-V4.1-Flash's architecture list, checked against a 56M model scored by a single causal forward

Written 2026-09-13 (local box, no hardware). Source: the model card
(`huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash`), the mHC paper (arXiv 2512.24880), YOCO
(arXiv 2405.05254), modded-nanogpt records #73/#77, and this repo's own MUDD, Engram-lite and
T3.1 results. The model card gives **no perplexity or training-efficiency figures** for any
component; its claims are benchmarks for a 552B-parameter MoE trained on 45T tokens.

## 0. Short answer

Of the nine named components, **seven are serving-cost optimisations** that the competition's
scorer never exercises (no KV cache, no prefill/decode split, no long context, no generation),
and the **two training-side ideas map onto things this project has already measured and
discarded** (Engram → Engram-lite, T3.2; mHC → the MUDD / hyper-connection family, negative at
DEPTH=6). The only fragment with a plausible training-time analog here is CED's shared KV,
whose small-model cousin is YOCO; it is worth at most an ablation arm inside the second box's
plain-block probe, not a project. Nothing in the list changes the plan.

| component | what it is | training or serving? | analog here | verdict |
|---|---|---|---|---|
| **CED** (20-layer causal encoder → 20-layer decoder; decoder KV projected from final encoder states) | cross-layer KV sharing so prefill runs half the model | architectural, motivated by serving (8B active prefill / 16B decode) | YOCO (self-decoder + cross-decoder sharing one KV); for us: layers 4–6 attend to K/V projected from layer 3's output | **ablation arm only** (§1) |
| **CSA2** (Full / Reindex / Reuse modes; share main KV and indexer K; reuse top-k indices) | sparse-attention index sharing across layers | serving + long-context training at 64K–1M | none: our attention is dense at SEQ_LEN 2048 and ~10.6% of the step behind a fused flash kernel | out of scope |
| **Hierarchical Sparse Indexer** | bounds indexer cost independent of context length | serving/long context | none | out of scope |
| **FP4 main KV cache** (E2M1, one E4M3 scale per 16 channels; 890 B/token) | KV compression | serving | no KV cache exists in a single-forward score | out of scope |
| **SWA Bounded Replay** | rebuild sliding-window KV by replaying n_win tokens instead of persisting | serving | no KV persistence | out of scope |
| **Single-Pass mHC + Mega-mHC kernel** | widened residual stream with a learned mixing matrix projected onto a doubly-stochastic manifold (Sinkhorn), fused into one kernel | training-time architecture, with a kernel to make it affordable | hyper-connections (modded-nanogpt #73 "partitioned hyperconnections", simplified in #77); this repo's MUDD (dynamic dense residual mixing) — **negative at DEPTH=6**; x0-gate — 13% slower, per-step neutral | **closed** (§2) |
| **Engram conditional memory** (196B params, token-based sparse lookup) | hashed n-gram memory gated by context | training-time architecture | T3.1 hashed n-gram VE (**adopted**, −0.015) and T3.2 Engram-lite (**discarded**: per-step +0.0025 regardless of capacity) | already done both ways |
| **DSpark** | semi-autoregressive drafter + confidence-scheduled verification | serving | lossless speculative decoding; scorer never samples (`mtp-multi-token-prediction-2026-09-12.md` §1, §7) | out of scope |
| **MoE** (1 shared + 384 routed experts, 6 active) | sparse FFN capacity | training-time architecture | plan T4.4 (2026-09-13): PG #480 −0.06…−0.08 at small scale; experts data-starved under the 288M-token wall | **closed** |

## 1. CED / shared KV — the one fragment worth a line in the track document

What CED does at training time: decoder layers do not compute their own K and V; they attend
to K/V projected once from the encoder's top hidden state. YOCO (Microsoft, 2024) is the same
idea at 160M–13B ("a cross-decoder stacked upon a self-decoder", "competitive results versus
Transformers across model sizes"); its abstract gives no small-scale perplexity numbers and the
secondary sources that quote parity to within a few hundredths of a perplexity point could not be
verified, so treat quality as "roughly neutral, unverified at our scale".

Why it is even on the list here: at DEPTH=6 a 3+3 split removes the K and V projections from
three layers (≈2.5M params, ≈5% of forward matmul FLOPs) and three launches per micro-step, on a
step that is op-count-bound. Under the data wall that throughput is convertible only into width
at a fixed step count, so the realistic upside is a few thousandths of bpb, and the risk is that
the decoder half loses the per-layer key/value refinement that a 6-layer model has little of to
spare. The rope+QK-norm NKI kernel still runs on Q per layer and once on the shared K.

Recommendation: **an ablation arm in Probe 3 of the throughput-to-capacity track**
(`--shared-kv-from N`: layers > N reuse K/V projected from layer N's output), measured for
s/step first and 900 s paired second, priced with the decomposition rule. Not a main-box item.

## 2. mHC — why it is closed without a run

- The mechanism is a widened residual stream (n parallel streams) with a learned n×n mixing
  matrix per layer, constrained to be doubly-stochastic so the identity mapping survives. Its
  cost is elementwise: n× the residual traffic plus the mixing, which is why DeepSeek needed a
  fused "Mega-mHC" kernel. On this stack every elementwise addition to the block body has been
  repriced by the compiler's scheduling, in both directions (frozen qk-shift −10% step time,
  x0-gate +13%), and there is no fused kernel; an NKI mHC kernel would be a project on the scale
  of the Muon kernel for an effect the paper itself frames as a 3B–27B stability/scalability win.
- This repo's nearest relative, MUDD (dynamic dense connections across all earlier hidden
  states), was a clean negative at DEPTH=6, and the simpler x0-gate was per-step neutral and
  slower. Hyper-connections helped the 124M-class GPU speedrun by ~1% (record #73) and were then
  simplified away to a single saved activation (record #77) — a launch-count story, not a
  quality one.
- The identity-mapping argument mHC fixes is a deep-network problem (their motivating failures
  are at 27B); a 6-layer residual stream does not have it.

## 3. What the model card does say about training, for the record

Pre-trained from scratch on 45T multimodal tokens; sparse attention trained at 64K sequence
length, context extended to 1M at 34T tokens; post-training SFT → RL → on-policy distillation
"without algorithmic modifications", with gains attributed to the agent-task data pipeline.
Base-model benchmark deltas vs V4-Flash-Base (MMLU-Pro 74.1 vs 68.3, HumanEval 79.4 vs 69.5)
come with 2× the backbone parameters (552B vs 284B) and are not attributable to any single
component. Nothing here is a training-efficiency claim.

## 4. Where this lands in the repo

- Plan T4.4: one closing line for the V4.1 list (mHC/hyper-connections closed with MUDD; the
  rest serving-side or already covered).
- Track document Probe 3: the `--shared-kv-from` ablation arm.
- No change to the main box's queue.
