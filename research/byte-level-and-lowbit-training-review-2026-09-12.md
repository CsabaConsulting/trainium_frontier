# Byte-level hierarchies (BLT / MBLM), MatMul-free / ternary training, and Hyena — do they fit this competition?

Written 2026-09-12 (local box, no hardware) in response to the extended Gemini Deep Research
prompt (`research/frontier-research-prompt-2026-09-12.md`). Neither byte-level patching nor
low-bit training had been researched here before; Hyena had not either, though its cousins
(linear attention, SSMs) were measured and closed. This report checks the three claims in the
prompt against the papers, the competition's scoring interface, and this stack's measured cost
structure, and ends with an audit of the Gemini roadmap document itself.

## 0. Short answer

| technique | scorable here? | quality at ~50M params / ~1 GB of text | Trn2 fit | verdict |
|---|---|---|---|---|
| **BLT / MBLM byte patching** | **No** — the scorer feeds fixed 8192-BPE token ids and reads a distribution over the same 8192 tokens; the tokenizer may not be modified | BPE wins until ~150B bytes of training for the *smallest* class BLT studied (450–760M params); we train on ~1.2 GB, 100× below the crossover | dynamic patch lengths are the opposite of the static shapes this stack needs; the entropy patcher is a 100M-param byte LM that must run on the fly | **not viable**; its one transferable piece is byte-aware token embeddings (§1.5) |
| **BitNet b1.58 / MatMul-free ternary** | yes (weights are still weights) | worse at small scale: 700M ternary PPL 12.87 vs 12.33 FP16; parity only at 3B; Parameter Golf ternary 1.177 vs ~1.11 stack | zero speedup: TensorE has no ternary/1-bit path, so ternary weights are multiplied as bf16 anyway; adds absmean/round/STE elementwise work to the busy engines; SR is bf16 rounding, not a quantiser, and fails the scorer's causality check | **not viable** |
| **Hyena long convolutions** | yes | parity with Transformers at 125–355M in the paper, at seq 2K, "20% less compute" | `torch.fft` is unsupported on Neuron (CPU fallback); attention is 10.6% of our step behind a fused flash kernel, so the ceiling is ≤ ~5% steps; same class of result as linear attention/SSMs measured here | **not viable** |
| Compatible cousin: **Mixture-of-Depths** (static top-k capacity routing) | yes | paper: matches dense at equal FLOPs and wall-clock, fewer FLOPs per forward | static shapes preserved by design; but gathers/scatters land on GpSimd (the `embedding_dense_backward` pathology) and our step is launch-bound, not FLOP-bound | low priority; only if Tier 1 makes the step FLOP-bound |

The three "Trainium advantage" arguments in the prompt do not hold: tokenization is not a
bottleneck here (host tokenises the corpus 40× faster than the device consumes it), stochastic
rounding is a floating-point rounding mode that has nothing to do with ternary quantisation and
is already ruled out by the scorer's causality tolerance, and FFT is not a Neuron operator.

## 1. Byte Latent Transformer, Multiscale Byte LMs, SpaceByte

### 1.1 What BLT actually is (Pagnoni et al., arXiv 2412.09871, ACL 2025)

- Bytes are grouped into variable-length patches by **next-byte entropy**; the entropy comes
  from a **separately trained byte-level LM** ("a transformer with 100M parameters, 14 layers,
  hidden 512, sliding window 512 bytes") trained on the same data. Patching runs "on the fly
  during training" as a dataloading step. Learning it end-to-end is listed as future work.
- Architecture: a lightweight local encoder and decoder (1–3 layers) around a large latent
  transformer that "consumes the bulk of the flops"; byte embeddings plus **hash n-gram byte
  embeddings for n=3–8** (500k hashes per order) — the byte-level analogue of this repo's
  T3.1 hashed n-gram tables, which BLT credits with 0.850 → 0.826 bpb (Table 8).
- Dynamic shapes are handled by packing patches per batch, padding/truncating byte sequences
  to 12k–24k, and FlexAttention for the dynamic cross-attention masks.
- The paper's own caveat (§9): "our implementations may yet not be at parity with
  tokenizer-based models in terms of wall-clock time." All comparisons are theoretical
  FLOP-matched.

### 1.2 The scale at which it wins

BLT's smallest class (3.1e8 inference FLOPs/byte): Llama 3 BPE at 450M vs BLT at 610–760M
params. BPE is compute-optimal at 50B training bytes; **BLT crosses over at ~150B bytes** —
"BPE wins at small budgets", and the 8-byte-patch model "starts at a significantly worse point
compared to bpe Llama 2 at 1B but ends up better than bpe at 7B scale." This competition trains
a 56M-parameter model on ~288M tokens ≈ 1.15 GB of text. That is ~100× below the crossover of a
model class ten times larger than ours. The 8B/4T-byte headline (BLT-Entropy 61.1 vs Llama 3
60.0 average) is irrelevant to the regime.

**MBLM** (Egli et al., arXiv 2502.14553) is a three-stage hierarchy (global Mamba, two local
transformer decoders) built for **5-million-byte contexts** on one A100; it is a long-context
architecture, not a training-efficiency one, and its byte-level quality at our budget is not
competitive with BPE for the same reason as BLT. **SpaceByte** (Slagle, arXiv 2404.14408)
reaches compute parity with subword tokenizers at ~1B scale by inserting global blocks at
space characters — a static patcher, i.e. a tokenizer by another name.

### 1.3 The scorer makes it moot

- README: "Modifying `prepare.py`, the evaluation harness, the data pipeline, **the
  tokenizer**, or the scoring functions" is not permitted; "The tokenizer is a fixed BPE
  tokenizer (vocab size 8,192), identical for all participants."
- `evaluate_bpb` tokenises the validation shard with that tokenizer, calls `model(x)` on
  **token ids**, requires `logits.size(-1) ≥ 8192`, and computes token-level cross-entropy that
  it then divides by byte counts. A byte-in/byte-out model has no place to plug in: the
  positions are tokens, the output must be a distribution over 8192 tokens per token position.
- `val_bpb` is already tokenizer-independent by construction (nats per token ÷ bytes per
  token), so the prompt's "subword tokenization creates artificial cross-entropy boundaries"
  is not a bpb argument; BLT's gain, where it exists, is compute allocation, not metric
  accounting.
- Our BPE at ~4.0 bytes/token *is* a static patcher with patch size 4, close to BLT's own
  4.4–4.5 bytes/patch for its entropy models. What BLT adds on top is variable patch length,
  which requires the 100M-param entropy model per byte and dynamic shapes.

### 1.4 The "Trainium advantage" claim, checked

- "Tokenizers are painful outside the host CPU": the host tokenises all five training shards
  (316M tokens) in 50.3 s on 12 threads (`count_train_tokens.py`), ≈ 6.3M tokens/s, against a
  device consumption of ~160k tokens/s. Tokenisation is 40× faster than needed and already
  overlapped; the measured dataloader exposure (20–30 ms/step) is packing and dispatch, not
  string handling.
- "Pushing raw bytes to the Tensor engine": a byte model processes ~4× more positions for the
  same text, so at equal FLOPs it must be ~4× narrower or shallower, and the local
  encoder/decoder add launches. On a launch-bound step this is the wrong direction.
- Dynamic patch lengths mean dynamic tensor shapes, and the notebook's most expensive lesson
  is that any per-step shape or scalar change recompiles the graph (the LR-scalar recompile
  cost 2.4× steps until fixed). BLT is the antithesis of a static-shape workload.

### 1.5 What does transfer: byte-aware *token* embeddings

The one BLT component that fits a token interface is enriching each token's input embedding
with the bytes it is made of: hash n-gram embeddings over byte substrings (BLT Table 8),
"byte-feature embedding init" (Recursive's from-vanilla run, plan §11.1), and Over-Encoding
(hierarchical 1–n-gram input embeddings, Parameter Golf, arXiv 2501.16975: a 400M model
matching a 1B baseline's loss at the same cost). With an 8192 vocabulary and ~4 bytes/token the
token table already encodes most of this, so the expected effect is small, but it is legal, is a
one-time table lookup at the input, and belongs in the same family as the T3.1 tables that are now
the default (−0.0152 bpb at 1800 s, dress-rehearsed at 2140 steps, board ~1.039). Recommendation: fold "initialise `wte` from byte features" into the T3.x
queue as a zero-cost one-liner; do not build a byte-level model.

## 2. MatMul-free / ternary / 1-bit training

### 2.1 What the papers measure

- **BitNet b1.58** (Ma et al., arXiv 2402.17764), Table 1, RedPajama 100B tokens, vs FP16
  LLaMA: 700M **12.87 vs 12.33** PPL, 1.3B 11.29 vs 11.25, 3B 9.91 vs 10.04. Ternary is worse
  at 700M and reaches parity only at 3B. The paper claims no training speedup; all gains are
  inference memory (2.6–3.6×), latency (1.2–2.7×) and energy, on a custom 2-bit kernel.
- **MatMul-free LM** (Zhu et al., arXiv 2406.02528): ternary weights via absmean + STE with
  full-precision latent weights, 8-bit activations, GLU and a linear-attention token mixer.
  The scaling curve is projected to cross Transformer++ "at approximately 10^23 FLOPs"
  (Llama-3-8B-class compute); at 370M–2.7B it is behind. The 25.6% faster iteration / 61% less
  memory comes from a fused RMSNorm+quantise Triton kernel on an A100 at batch 256, and the
  "addition-only" arithmetic is realised on an FPGA / Loihi 2, not on any GPU or Trainium.
  Ternary training also needs a 5–13× larger learning rate because "standard LR gives
  excessively small updates that have no impact on the clipping operation."
- **Parameter Golf** (same 16 MB-scale regime as ours): BitNet b1.58 at 68M params scored
  1.177 vs ~1.11–1.12 for the standard bf16 stack; "standard stack breaks on ternary"; the
  retrospective paper ranks Ternary at −0.013 bpb, among the harmful flags.

### 2.2 Why it cannot speed up training here

- **Training a ternary model is not ternary arithmetic.** The forward multiplies quantised
  weights that live in bf16/fp32 latent tensors; the backward is full precision through the
  STE. On GPUs the matmul runs on the same tensor cores as bf16. On Trn2 the TensorE supports
  bf16/fp16/fp8/int8 matmul paths; there is no ternary or 1-bit path, so a ternary matrix is
  simply a bf16 matrix with three distinct values. Zero FLOP saving, zero DMA saving (the
  operands are still bf16 unless int8-packed, which reintroduces the missing fast-path problem
  documented for fp8 in the engineering-stance doc).
- **It adds elementwise work.** Absmean scale, round, clip and the activation quantiser are
  per-element passes over every weight and activation each step, landing on the
  Vector/Scalar/GpSimd engines that already sit within a 2.4× band of TensorE on a
  dispatch-bound step. This is the same failure mode as SLAY, KAN and DCMHA here.
- **Quality cost at 56M params.** Every small-scale measurement (BitNet 700M, Parameter Golf
  68–74M) is worse than bf16, and the LR regime that ternary needs (3–4e-3) is far from the
  Muon/AdamW schedule this repo has spent a week tuning.
- **Memory is not our constraint.** The 61% memory saving is irrelevant on a chip with 96 GB
  HBM running a 56M-parameter model.

### 2.3 Stochastic rounding is not the "secret sauce"

Neuron's stochastic rounding (`NEURON_RT_STOCHASTIC_ROUNDING_EN=1`, NeuronCore-v2+) is an
alternative to round-nearest-even when a floating-point result is rounded to a lower-precision
*float* format (the doc's example: repeatedly adding 0.5 to an fp16 1024.0). It exists so that
small bf16 accumulations do not get squashed. It has nothing to do with integer or ternary
quantisation, which is a deterministic absmean/round/clip with a straight-through gradient; the
"tiny fractional updates" in BitNet are carried by the full-precision latent weights, not by
the rounding mode. And this stack has already measured SR: it fails `causality_check` because
its own forward-to-forward noise (0.0113) exceeds the harness's 1e-2 tolerance
(`USE_STOCHASTIC_ROUNDING = False`, notebook). Any submission relying on SR would be rejected
at pre-flight regardless of what it did for training.

## 3. Hyena / long convolutions (from the prompt's cursory search)

- **Claim** (Poli et al., arXiv 2302.10866): implicit long convolutions via FFT plus
  data-controlled gating; Transformer-quality perplexity on WikiText-103 and The Pile at
  125M–355M with "a 20% reduction in training compute required at sequence length 2K";
  2× faster than optimised attention at 8K, 100× at 64K.
- **Neuron reality:** `torch.fft` (`aten::fft_fft`) is listed as unsupported by torch-neuronx;
  the documented remedy is partitioning the FFT to the CPU. A per-layer host round trip per
  micro-step would be catastrophic on a step that already loses 20–30 ms to host exposure. A
  direct-space long convolution instead is O(N²) — no better than attention — and its
  implementation as a chunked scan is the Mamba2/SSD path this repo priced at +0.11–0.17 s per
  mixer with an NKI kernel.
- **Ceiling:** attention is ~10.6% of the step behind a fused NKI flash kernel at seq 2048; even
  a free replacement is worth ≤ ~5% steps ≈ 0.004 bpb. Hyena's 20% compute saving is a
  statement about FLOPs at 2K on GPUs where attention is a larger share; here it is not.
- **Evidence class:** this is the fourth token-mixer alternative examined (after SLAY, Mamba2,
  RWKV); every one lost on this hardware for the same structural reasons. Closed without a run.

## 4. The compatible cousin worth naming: Mixture-of-Depths

The one part of BLT's pitch that a token-interface model *can* copy is "spend less compute on
predictable positions." Mixture-of-Depths (Raposo et al., arXiv 2404.02258) does this with a
per-block top-k router that lets only k tokens through the attention+MLP of that block; because
k is fixed, "the computation graph stays static with known tensor sizes," which is exactly the
constraint this stack imposes. The paper reports matching dense baselines "for equivalent FLOPS
and wall-clock times to train" while using a fraction of the FLOPs per forward pass.

Why it is still low priority here: the saving is FLOPs, and our step is launch- and DMA-bound
at ~8% MFU, so fewer FLOPs per block translate weakly into time (wider matmuls cost 0.65× per
FLOP; the routing gathers/scatters cost launches and land on GpSimd, the engine behind the
`embedding_dense_backward` pathology T3.1 had to work around). It also needs an auxiliary
router loss and a causal top-k at eval (the paper's "predictor" router) to keep the forward
stateless. Revisit only if Tier 1 work makes the step FLOP-bound; file next to MTP.

## 5. Audit of the Gemini Deep Research 2 roadmap against this repo's measurements

The document (`Trainium 2 Training Optimization Gemini Deep Research 2.md/.pdf`) is a
restatement of the GPU-proxy results and HLAT with a roadmap; several of its Phase 1A/1B
prescriptions are things this repo has since measured the other way. Anyone reading it as a
plan should apply these corrections:

| Gemini recommendation | this repo's measurement | status |
|---|---|---|
| LNC=1, 8 ranks, pure DP | LNC=2 / 4 ranks won decisively (639 vs 500 steps); LNC=1 now fails to compile and shows a +0.24 bpb cliff | **refuted**; launch command is LNC=2 |
| Stochastic rounding on, β2=0.95 | SR fails the scorer's causality check (false positive but disqualifying); β2=0.95 adopted on its own merits | SR refuted; β2 kept |
| `--distribution-strategy=llm-training --model-type=transformer --lnc=1 --internal-hlo2tensorizer-options='--fast-relayout=true'`, `NEURON_FUSE_SOFTMAX=1`, async inflight 3 | `--fast-relayout` does not exist in this SDK's `neuronx-cc --help`; the flag bundle wins alone but loses combined with the causal fast path; `--optlevel=1` is the measured winner | **refuted** |
| ZeRO-1 optimizer sharding | memory is not a constraint (96 GB HBM, 56M params); adds collectives | unnecessary |
| PACK_FACTOR = 2, T = 1024 | PACK_FACTOR 4 and SEQ_LEN 2048 won on Trn2 | superseded |
| INIT_SCALE 0.68 | 0.50 adopted in the T0.4 sweep | superseded |
| `neuron_parallel_compile` AOT before the timed run | compile time is uncharged up to 900 s by the runner; cold compile measured ~30–60 s | unnecessary for scoring; fine for dev |
| NKI fused RMSNorm as the first kernel | measured **NO-GO**: torch.compile already fused it (exposed time 3.8% → 0.6%) | refuted |
| NKI fused RoPE, NKI softcap+CE | both built, verified to bf16-ULP, default on | done |
| NKI FlashAttention | the SDK's NKI flash kernel is already what SDPA dispatches to | done |
| Warmup 0, warmdown 0.75 / floor 0.05, WD decay to zero, VE alternating, local conv k=4, weight-cast cache | all adopted (local conv now k=2, trained) | done |

Net: the Gemini roadmap is correct about the GPU-era schedule items (which are already in) and
wrong about every Trn2-specific runtime prescription that this repo has since tested. Its
strategic claim that "SwiGLU gated MLPs inflate step latency" matches the GPU measurement but,
per the GPU-closed audit, the parameter-matched coalesced form is untested here.

## 6. Recommendations

1. Do not build byte-level, ternary, or Hyena models for Phase 1. Record all three in plan
   T4.4 (done alongside this document).
2. Take the one legal fragment of the byte-level idea — byte-feature initialisation or a small
   hashed byte-n-gram input table — as a zero-cost item in the T3 queue, expected ≤ 0.002 bpb.
3. Keep Mixture-of-Depths on the "if the step ever becomes FLOP-bound" list with MTP.
4. When the next Gemini/AI deep-research document arrives, run its runtime prescriptions
   through the engineering-stance and GPU-closed audits before acting; the failure mode is
   confident restatement of HLAT-era settings that have since been measured false here.

## 7. Sources

- BLT: Pagnoni et al., arXiv 2412.09871 (ACL 2025) — §§ on entropy patcher, Table 2 crossover,
  Table 8 hash n-grams, §9 wall-clock caveat. MBLM: Egli et al., arXiv 2502.14553. SpaceByte:
  Slagle, arXiv 2404.14408. Over-Tokenized Transformer / Over-Encoding: arXiv 2501.16975.
- BitNet b1.58: Ma et al., arXiv 2402.17764, Table 1. MatMul-free LM: Zhu et al., arXiv
  2406.02528, §5.1, §6.4. Parameter Golf issue #140 (BitNet #126/#367, ternary #216/#641);
  *Parameter Golf: What Really Works?*, arXiv 2607.01517 (Ternary −0.013).
- Neuron rounding modes: awsdocs-neuron "Neuron Rounding Modes"; torch-neuronx supported
  operators / analyze API (`aten::fft_fft` unsupported, CPU partitioning).
- Hyena: Poli et al., arXiv 2302.10866. Mixture-of-Depths: Raposo et al., arXiv 2404.02258.
- This repo: README (tokenizer fixed; what may not be modified), `prepare.py` (`evaluate_bpb`,
  `causality_check`), `research/trn2-experiment-results.md` (SR causality false positive,
  LNC A/Bs, compiler-flag crashes, RMSNorm NO-GO, count_train_tokens timing, T3.1),
  `research/engineering-stance-parallelism-precision-2026-09-12.md`,
  `research/gpu-closed-techniques-trn2-revisit-2026-09-12.md`.
