# Deep-and-thin (MobileLLM / SmolLM) versus our 6×1024 recipe (2026-09-25)

Question: Meta's MobileLLM (arXiv 2402.14905) and Hugging Face's SmolLM argue that sub-1B
models should spend parameters on depth (30 layers × 576) rather than width. Can that help us?
Desk research only; no runs were started and the Trn2 boxes were not touched. Literature facts
were collected by a research agent today (URLs in §5); repo facts cite the notebook.

## 0. Short answer

No, not under this competition's objective. The deep-and-thin result is real but was measured
under three conditions we do not have: downstream zero-shot accuracy (not language-modelling
loss), 250B–1T training tokens (we see ~480M), and a fixed parameter count with the wall clock
free (ours is fixed at 30 minutes on a launch-bound chip where every layer costs the same time
regardless of width). Every fixed-wall-clock depth increase we have measured on Trn2 lost, and
the one fixed-parameter perplexity study in the literature puts the 134M optimum at about 12
layers with a flat basin, not 30. A 30×576 model would run about 2.7× slower per step here and
finish roughly 1,400 updates instead of 3,660.

| Claim | Where it holds | Here | Verdict |
|---|---|---|---|
| 30 layers beat 12 at ~125M | MobileLLM: 8-task zero-shot accuracy, 0.25T tokens, params fixed | scorer is val BPB; ~0.5T tokens fewer; time fixed | does not transfer |
| Depth lowers perplexity at fixed params | Petty et al.: yes, but saturates at ~12 layers for 134M and degrades past ~21 | our 6 → 8 layers at fixed width lost under wall clock | already tested, negative |
| Deep-thin trains for the same cost | true on GPUs at equal FLOPs | Trn2 step is dispatch-bound: extra layer ≈ +1/6 step, width shrink recovers only 0.65× | premise fails |
| Tied embeddings free up parameters | 32k–49k vocab, 20% of params | 8192 vocab, untied head is 8.4M of 117M | little to gain; not tested, low priority (§4b) |
| GQA 3:1 | KV-cache memory at inference | K/V projections ≈ 2–4% of step; 5-head GQA impossible, closed in T4.4 | closed |
| Immediate block-wise layer sharing | +0.7–1.1 accuracy points for 2× compute per token | doubles step time | wrong trade for a time budget (§4b) |

## 1. What the sources actually measured

**MobileLLM (ICML 2024).** 19 models at ~125M and ~350M, depth and width varied jointly at head
dim 64, trained 120k iterations on 0.25T tokens for the ablation and 1T for the released models.
The metric is exclusively downstream: eight zero-shot common-sense tasks plus TriviaQA and RACE.
The paper reports no training loss, validation loss or perplexity anywhere. Table 11 for the
125M class (average accuracy): 4L/1280 43.3, **6L/1024 42.3**, 8L/896 43.0, 12L/768 43.9,
18L/640 43.9, 24L/576 44.8, 30L/512 44.8, 42L/448 44.5, 62L/384 44.7. The whole deep-thin gain
over the 12-layer baseline is 0.9 points; SwiGLU alone was worth 1.3 points in the same table.
Released 125M shape: 30 layers, 576 wide, 9 heads, 3 KV heads, FFN 1536, tied 32k embeddings,
124.6M parameters. The Google AI summary the user pasted is accurate on these numbers.

**SmolLM / SmolLM2.** SmolLM-135M and SmolLM2-135M are byte-identical in shape to MobileLLM-125M
(30/576/9/3/1536) with a 49,152 vocabulary, trained on 600B and 2T tokens. The SmolLM blog says
"we adopted a design similar to MobileLLM ... prioritizing depth over width"; neither the blog
nor the SmolLM2 paper (arXiv 2502.02737) contains its own depth-versus-width ablation. SmolLM
is an adoption of MobileLLM's result, not independent evidence.

**Petty et al. (NAACL 2024, arXiv 2310.19956)** is the one study that holds parameters fixed and
measures perplexity. At 134M (d_model 768, FFN shrunk to pay for depth) C4 validation perplexity
is 6L 18.7, 8L 18.3, **12L 18.1**, 16L 18.2, 21L 18.3, 26L 18.6, 32L 19.2. Six to twelve layers
buys 3.3% perplexity (~0.05 nats/token); beyond 16 it gets worse. They also measured that
"deeper models train more slowly than shallower ones when controlling for total parameter
count", with latency linear in depth, and recommend shallower models when minimising GPU-hours.

**Kaplan et al. 2020** (loss, not accuracy): "Aspect ratio in particular can vary by a factor of
40 while only slightly impacting performance"; a (6, 4288) model reaches a loss within 3% of the
(48, 1600) GPT-2. Tay et al. 2021 ("Scale Efficiently") find shape matters for downstream
fine-tuning but say Kaplan's shape-independence holds for the upstream loss.

**What the same labs chose later.** MobileLLM-R1 (Meta, September 2025) halved depth to
15 layers at 140M (576 wide, FFN 2048) and 360M (1024 wide). Qwen3-0.6B is 28 × 1024;
nanochat's default rule is `model_dim = depth × 64` (d20 = 20 × 1280); OpenAI Parameter Golf
records under a 10-minute budget sit at 11 layers. Practice has converged on aspect ratios
around 40–90, not 19 (576/30) and not 171 (our 1024/6).

## 2. What this repo has measured

All numbers are `public_val_bpb` on the fixed budget stated, from `research/trn2-experiment-results.md`.

**Depth at proportional width, 900 s** (aspect 64): 4L/256 529 steps 1.3308; **6L/384 399 steps
1.2972**; 8L/512 308 steps 1.3428; 10L/640 235 steps 1.3685; 12L/768 193 steps 1.3657.

**Width at fixed depth 6, 900 s:** 256 1.3093; 384 1.2972; 512 1.2708; **640 1.2696**; 768 1.2736.

**Depth at fixed width (T0.2, 900 s, 2026-09-11):** (6, 640) 1094 steps 1.081255;
(7, 640) 908 steps +0.0126; (8, 640) 839 steps +0.0137; (6, 768) 885 steps +0.0094;
(6, 640, head dim 64 / 10 heads) 999 steps +0.0179. Every deeper or narrower-head arm lost on
both quality and step count. Note the head-dim-64 arm: MobileLLM's "head dim close to 64"
recommendation was the worst of the four here.

**Full 1800 s confirmations:** 8L/768 1217 steps 1.0524 vs 6L/640 1995 steps 1.0435;
5L/512 1082 steps 1.1173 vs 6L/640 847 steps 1.1087 (the shallower model won the 900 s proxy and
lost the full budget, so proxies are escalated before any shape decision); width 512 2433 steps
1.0542 and width 768 1623 steps 1.0442 vs 640 1995 steps 1.0435.

**Current recipe:** 6 × 1024, 117.45M parameters (83.9M linear, 33.5M embeddings), 3,660
updates in 1,785 charged seconds (0.488 s/step), 480M tokens = 1.5 passes over the ~316M-token
training set, 0.9928513 public BPB. Width 1152 at the same budget: +20.6% step time,
0.99381 (worse). Marginal-cost rules from the notebook: an extra layer at fixed width costs
about one sixth of a 6-layer step (7th layer −17% steps, 8th −23%); shrinking width saves only
~0.65× of the proportional parameter reduction, because the step is dispatch-bound at ~8% MFU
and 94% of it is model microbatches.

## 3. Why the paradigm inverts under our constraints

1. **Time, not parameters, is the fixed resource.** MobileLLM compares shapes at equal parameter
   count with training time free. Applying the notebook's cost rules, a 30 × 576 model (about
   129M parameters with our untied 8192 vocabulary and ReLU² ratio-4 MLP) runs ≈2.7× slower per
   step: ≈1,370 updates and ≈180M tokens (0.57 epoch) in 1,800 s instead of 3,660 updates and
   1.5 epochs. Even 9 × 1024 (130M) drops to ≈2,500 updates. Every measured point on this chip
   says the extra parameters do not pay back that many lost updates.
2. **Depth is expensive here in a way it is not on an H100.** Each layer is a fixed set of
   launches and collectives (all-reduces run per layer), so layer count scales step time almost
   linearly, while narrowing the matmuls saves little because a 576-wide GEMM under-fills the
   128×128 systolic array. Petty et al. saw latency linear in depth even on GPUs; the effect is
   stronger on a dispatch-bound step.
3. **The metric is language-modelling loss.** The deep-thin advantage appears on reasoning
   benchmarks; on loss the literature says shape is a few-percent effect with a broad basin
   (Kaplan) and a 134M optimum near 12 layers (Petty). Our own loss measurements at fixed width
   put the optimum at 6 because of the step-count penalty, and 7 or 8 layers each lost by
   10–14× the noise floor.
4. **Data is short, not long.** MobileLLM's ablation used 0.25T tokens; we make 1.5 passes over
   316M. Deep models buy generalisation from long training; here a bigger or deeper model would
   see fewer tokens, not more.
5. **The auxiliary tricks address memory, not loss.** Tied embeddings and GQA exist to fit a
   phone's DRAM and KV cache. With an 8192 vocabulary the untied head is 8.4M parameters; GQA at
   8 heads of 128 was closed in T4.4 (K/V projections are 2–4% of the step). Layer sharing runs
   every block twice for +0.7–1.1 accuracy points; under a wall clock that halves the updates.

## 4. What could still be worth a run, and what not

- **Not worth it:** 30 × 576, any head dim 64 shape, layer sharing, GQA. All either measured
  negative here or contradict the fixed-time arithmetic by a wide margin.
- **Only conditional candidate:** 7 or 8 layers at width 1024 (aspect 128–146). It was tested at
  width 640 and lost 0.013, and 8 × 768 lost 0.009 at the full budget. It would become worth a
  single full-budget pair only if a throughput change ever moves the step by more than ~15%,
  which is the plan's standing re-test rule for T0.2. Nothing since has met it.
- **Cheap curiosity, low expected value:** tying `wte` and `lm_head` (8.4M parameters, one fewer
  AdamW group). MobileLLM lost 0.2 accuracy points from tying; the loss effect at our vocabulary
  is unknown and probably below noise. Not a priority.

## 4b. Follow-up (2026-09-25): tied embeddings and block-wise layer sharing

Asked separately after the review was shared. Neither helps, and for the same reason the
deep-and-thin shape does not: MobileLLM optimises under a *parameter* cap, this competition
caps *wall clock* on a dispatch-bound step. Parameters that live in matmuls cost time;
parameters that live in lookup tables are nearly free. Both tricks move in the wrong direction.

**Tied embeddings (share `wte` and `lm_head`).**

- The "15–30% of the budget" premise assumes a 32k–50k vocabulary. Ours is 8192: `wte` and
  `lm_head` are 8.4M parameters each, so tying saves 8.4M of 117.45M (≈7%).
- Saving them buys no time. The head matmul and the input gather run identically whether the two
  tensors share storage. The only saving is two AdamW moment buffers, which do not touch the step.
- There is nowhere to reinvest. MobileLLM reinvests freed parameters into layers; every such
  reinvestment measured here (7th/8th layer, width 1152) cost step time and lost under the budget
  (§2).
- Plain tying is known to break in this lineage: Karpathy's autoresearch log records it as
  catastrophic (val 3.2 bpb) because `wte` trains at ≈75× the head's LR with a very different
  init scale (0.8 vs 0.001). modded-nanogpt only makes it work by copying `lm_head.T` into `wte`
  at init and untying at 2/3 of training; that is already plan item T3.9, low priority.
- MobileLLM itself lost ≈0.2 accuracy points from tying and accepted it as a
  memory trade. There is no loss gain to import.
- The repo already exploits the real asymmetry in the opposite direction: 33.5M of the 117.45M
  parameters sit in the embedding family (`wte`, `lm_head`, value-embedding tables on alternating
  layers) because gathers are almost free on the step while matmul parameters are not.

**Block-wise layer sharing (MobileLLM-LS: run each block twice with shared weights).**

- Parameter count stays flat but compute doubles. The phone-side justification is that the
  repeated block's weights are already in cache/SRAM. On Trn2 that argument is void: the step is
  launch-bound at ≈8% MFU, not weight-bandwidth-bound.
- A repeated pass costs the same step time as a fresh layer (≈+1/6 per pass at width 1024), and a
  fresh layer has at least the capacity of a repeated one. The fresh 7th and 8th layers already
  lost at both 900 s and 1800 s (§2), so a shared repeat cannot win.
- The Parameter Golf frontier lists universal-transformer style weight sharing as negative
  (`research/frontier-research-plan-2026-09-09.md` §11.3), consistent with our closed lines.

**Verdict.** Layer sharing is closed by the existing depth measurements; no experiment. Tied
embeddings stay a cheap curiosity with expected effect at or below the ≈0.001 bpb noise floor,
worth trying only in the T3.9 form (tie with `lm_head.T` init, untie at 2/3) and only on a GPU
proxy first, never on Trn2 time.

## 5. Sources

MobileLLM arXiv 2402.14905v2 (Tables 2, 9, 10, 11, 12; Sec 2.2–2.3, 3.6); github.com/facebookresearch/MobileLLM;
MobileLLM-R1 and MobileLLM-Pro model cards (huggingface.co/facebook); SmolLM blog huggingface.co/blog/smollm;
SmolLM2 arXiv 2502.02737; HF config.json for SmolLM-135M, SmolLM2-135M/360M/1.7B, SmolLM3-3B, Qwen3-0.6B;
Petty et al. arXiv 2310.19956v2 (Table 3, Sec 3.1, 4, 8); Kaplan et al. arXiv 2001.08361 §3.1; Tay et al.
arXiv 2109.10686; Levine et al. arXiv 2006.12467; karpathy/nanochat `scripts/base_train.py`; openai/parameter-golf
README. The facebook/MobileLLM-125M config.json is gated and was not read directly; the HF smol-training-playbook is
JS-rendered and was not read. Repo: `research/trn2-experiment-results.md` (Solution 11 depth and width sweeps,
T0.2 2026-09-11, Tier 0.3/0.4), `research/engineering-stance-parallelism-precision-2026-09-12.md`,
`research/fresh-start-capacity-revisit-2026-09-19.md`, `submissions/dense6x1024_public0992851/README.md`.
