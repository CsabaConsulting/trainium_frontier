# Techniques closed on GPU evidence that might behave differently on Trainium 2 — audit and tally

Written 2026-09-12 (local box, no hardware). Question: which items did this project close or
skip on GPU-proxy measurements or GPU-based reading (Karpathy's nanochat/autoresearch logs,
modded-nanogpt), and could Trainium 2's different cost structure flip any of them? Sources:
`research/gpu-experiment-results.md` (RTX PRO 6000 Max-Q, eager, fixed 800 steps, 384-wide /
86M-param model, 2026-09-04), `research/enhancement-opportunities.md` (the day-one scan that
inherited Karpathy's verdicts), Karpathy's nanochat "What didn't work" list (Discussion #481,
8×H100, d24), `train.py` toggle comments, and the Trn2 notebook.

## 0. Short answer

1. **There is a real precedent for the question.** Three GPU-era decisions already flipped on
   Trn2: `PACK_FACTOR` (GPU said 2, Trn2 says 4, +2.6% steps), model width (Karpathy's "dim 640
   too big" on a 5-minute H100 vs 640 winning here), and bigram tables (Karpathy: "works, but
   not by too much" at d24; here −0.013 once placed on the last layer). All three flipped for
   *cost-structure or regime* reasons, not because quality physics changed.
2. **What Trn2 changes is cost, not quality.** Items closed on GPU because they were *slower*
   or *used memory* deserve a second look; items closed because they were *worse per step at
   equal steps* stay closed unless the regime moved (it did: 384→640 wide, 800→2200 steps, so
   the notebook's own flagged gap "individual-toggle re-ablation at DEPTH=6 remains undone"
   is the honest caveat on every quality-closed item).
3. **Four candidates survive, ranked (§3):** parallel attention+MLP block with a coalesced
   input projection (the one genuinely Trn2-favoured idea), parameter-matched coalesced
   SwiGLU, partial-rotation RoPE inside the NKI kernel, and Recursive's input-dependent x0 gate
   as the rehabilitated form of the GPU-discarded selective gate. Expected values are small
   (0.001–0.005 bpb each); all are 900 s paired tests behind the n-gram, eval-context and Tier 1
   work.
4. **Two GPU closures get *worse* on Trn2, not better:** sliding-window attention and
   document-boundary masking, because any additive mask forfeits the on-device causal fast
   path that won +10% steps here (§2.2–2.3). Leave them closed.
5. Everything closed *on Trn2 itself* (FP8, stochastic rounding, fused QKV, XSA, MUDD/DCMHA,
   Mamba2, RWKV, SLAY, value residual, requeue packer, LNC=1) is out of scope for this audit and
   stays closed (§4).

## 1. Trn2 cost-structure facts used to judge each item

All from the notebook and plan, current code:

- Dispatch/serialisation-bound: 67 NEFF launches per micro-step after compile; device busy
  97% but ~28% of the window is serialisation loss between engines; ~8% MFU overall.
- Engine shares within a 2.4× band (DMA 27.8%, TensorE 22.5%, GpSimd 20.1%, Vector 17.1%,
  Scalar 11.6% in the eager profile); TensorE-active efficiency 63% under compile.
- **Wider matmuls cost ~0.65× proportional time; extra launches/layers cost ~1.0×** (width and
  MLP-ratio sweeps vs depth-8, §… of the engineering-stance doc). Rule: fold work into existing
  launches; never add standalone ops.
- `USE_CAUSAL_FASTPATH` (`is_causal=True`, no additive mask) was worth +10% steps
  (265 vs 240 at the 900 s proxy). **Any technique that needs an additive attention mask
  gives that back.**
- Attention is ~10.6% of the step (fwd ~4.7% + NKI flash backward 9.4%/… measured
  separately); RoPE+QK-norm were 12.6 + 6.0 ms of a 91 ms window before their kernel fusion.
- Memory is not a constraint: 96 GB HBM per chip; training uses a fraction; the only overflow
  ever hit was an eval batch of 64 rows.
- Compile time is uncharged up to 900 s per run, so extra graphs are free; the
  Vector/Scalar/GpSimd engines are already loaded, so extra elementwise work is not.
- Regime shift since the GPU phase: 384-wide/86M → 640-wide/55.7M, 800 fixed steps → ~2200
  steps in 1800 s, eager → torch.compile, no NKI kernels → three fused kernels.

## 2. Tally of GPU-closed items

### 2.1 Closed on the GPU proxy by measurement (2026-09-04), never re-run on Trn2

| item | GPU result (800 steps, 384-wide) | why it was closed | Trn2 factor | verdict |
|---|---|---|---|---|
| **Gated MLP / SwiGLU** (`USE_GATED_MLP`, sol9b) | bpb +0.0039, params 85.9M→102.3M, time +19% | worse *and* slower *and* bigger | throughput half could shrink: coalesce gate+up into one `640×3584` matmul (0.65× elasticity, one launch); SiLU is native on ScalarE. Quality half is hardware-independent and the small-model consensus (modded-nanogpt, nanochat, Recursive, PG #340/#344) is ReLU² ≥ SwiGLU | **revisit, low priority** (~15–20%): parameter-matched (hidden 8/3·d), coalesced, compile on, one 900 s pair |
| **Sliding-window attention** (`WINDOW_PATTERN=SSSSL`, frac 0.125) | +0.0013 bundled; **+0.0328 alone** | large quality regression at seq 1024 | needs an additive window mask → loses the causal fast path (−10% steps); attention is only ~10.6% of the step so the FLOP saving is ≤ 3–4% even if free. Karpathy kept SWA at d24 only because FlexAttention/FA3 make windows cheap on H100 | **stays closed; Trn2 makes it worse** |
| **Doc-boundary masking** (`USE_DOC_MASK`) | +0.0063, time 863 s vs 680 s | worse and slower | same additive-mask penalty on Trn2; Karpathy: attending across BOS "does not seem to make things much worse" | **stays closed; Trn2 makes it worse** |
| **PACK_FACTOR** 8/16/1 vs 2 | 2 best on GPU | bin-size sweep | **already flipped on Trn2**: 4 wins (+2.6% steps, −0.0017); 8 past the optimum per microbench | done |
| **Selective gate** (`USE_SELECTIVE_GATE`, data-dependent resid/x0 lambdas) | +0.0092 | real regression | extra elementwise ops on loaded engines (Trn2-unfavourable); but Recursive's winning recipe uses an input-dependent x0 gate `2σ(s_i·mean(x))`, a cheaper cousin credited at 50M scale | **revisit only in Recursive's form**, bundled with the cheap one-liners (shifted ReLU², sandwich norm) |
| `ROPE_BASE=200000` | +0.0269 at seq 1024 | regression | hyperparameter, hardware-independent; Trn2 later found base 500000 vs 100000 indistinguishable at seq 4096. Recursive uses 1e6 at their shape | not Trn2-specific; fold into a T0.4-style sweep if ever |
| `VE_WD=0.003` | flat/marginal negative | — | hardware-independent | closed |
| `MUON_MOMENTUM_WARMUP_STEPS=200` | +0.0008 | regression | **re-tested on Trn2** (Tier-0): 300 confirmed | done |
| `INIT_SCALE` 0.68 | GPU optimum | — | **re-swept on Trn2** (T0.4): 0.50 adopted | done (regime, not hardware) |

### 2.2 Dismissed by reading GPU ablations, never run anywhere here

| item | source of the "no" | GPU reason | Trn2 factor | verdict |
|---|---|---|---|---|
| **Parallel attention + MLP block** (GPT-J/PaLM: `x + attn(n(x)) + mlp(n(x))`) | Karpathy `46cf5f2` "much worse" | quality at his depth/budget | **the one Trn2-favoured candidate**: removes a sequential dependency per layer so attention and MLP matmuls can overlap across engines; the Q/K/V and `c_fc` input projections share the same normed input and can be **one `640×4480` matmul** (fold-into-existing-launch rule); at 6 layers the effective-depth loss is the risk (PaLM found it neutral only at scale) | **revisit first**: 900 s pair; expect +3–6% steps vs an unknown quality loss; go/no-go on net bpb |
| **MQA / GQA** | Karpathy `b85567f` "too few KV heads" | quality | GQA impossible at 5 heads; MQA saves ~9% forward FLOPs on K/V projections only (~2–4% step); payoff is KV cache, unused | low; see T4.4 |
| **Half-truncated / partial RoPE** | nanochat: "no improvement" | rope is ~free under FA3 | rope was 12.6 ms of a 91 ms window here before fusion and still lives in a hand-written NKI kernel; rotating half the head dim halves that work. NeoX/Pythia Trainium blog reports partial rotation needs the zero-frequency-padding trick to be fast | **revisit, low-medium**: kernel edit + equivalence harness, ~1–3% step (0.001–0.002 bpb) at neutral quality |
| **Skip connections / backout** (records #11/#40/#45/#77) | nanochat: "no improvement, +2 GB" | memory + no gain | memory is free here, but "no improvement" is quality, and Recursive's version (pooling the last three layers' outputs, zero-init) is the form worth trying — already plan T3.5 | not Trn2-specific; keep in T3.5 |
| **Smear gate / attention gates** | nanochat: "negligible" at d24 | scale | Recursive: decisive at 50M (head gate). Scale, not hardware — plan T3.3/T3.6 | not Trn2-specific |
| **FP8 lm_head** | nanochat: +2 GB, 1% | memory | no fp8 GEMM path on this stack | closed on Trn2 |
| **Varlen attention** | nanochat: BOS dataloader covers it | — | no varlen kernel on Neuron | closed |
| **Batch-size schedule** | nanochat: "too complex" | — | **tested on Trn2** (T2.1): discard | done |
| **Bigram embeddings** | nanochat: "works, but not by too much" | scale | **flipped on Trn2**: last-layer tables −0.013/−0.014 at zero step cost | done (precedent) |
| **Depth 10–11 / dim 640 "too big"** | Karpathy 5-min H100 | step count | **flipped**: 640 wins here at DEPTH=6 | done (precedent) |
| **Asymmetric softcap** | nanochat: "slightly worse" | quality | elementwise, cheap on ScalarE; quality is the issue | closed |
| **FlashAttention-3** | GPU kernel | — | Neuron already runs an NKI flash kernel | n/a |

### 2.3 Why the two mask-based items get worse on Trn2, spelled out

On H100 with FlexAttention/FA3, a block-sparse or windowed mask *saves* FLOPs at near-zero
overhead, which is why Karpathy kept the SSSL pattern. On this stack the only fused fast path
is `is_causal=True`; a window or document mask must be passed as an additive `[T, T]` tensor,
which the notebook measured as the difference between 240 and 265 steps at 900 s. Attention
is ~10.6% of the step, so even a window that halved attention work would recover ≤ 5% against
a 10% loss, and the GPU quality regression (+0.033 for windowing alone) is far above anything
the step count could buy back. Same logic for doc masking, with a smaller and also-negative
quality effect.

## 3. Recommended revisits, in order (all behind current Tier 1 / n-gram / eval-context work)

1. **Parallel block with coalesced input projection.** Implement as a toggle: per layer,
   `h = norm(x)`; one matmul `h @ W_in` producing `[q,k,v,fc]`; attention and ReLU² MLP from the
   split; `x = x + λ_a·attn_out + λ_m·mlp_out`. Keeps the rope+QK-norm kernel (Q/K slices
   unchanged) and the causal fast path. Measure with `phase_timer` first (expect fewer launches
   and a larger, better-utilised matmul), then one 900 s pair. Decision on net bpb; the risk is
   quality at DEPTH=6.
2. **Parameter-matched, coalesced SwiGLU.** `hidden = 1664` (≈8/3·640 rounded to 128), gate and
   up as one `640×3328` matmul, SiLU·up in compile. Same protocol. Prior: 15–20%.
3. **Partial RoPE in `_rope_norm_fwd/_bwd`.** Rotate the first 64 of 128 dims, pad frequencies
   with zeros per the Trainium trick rather than slicing/concatenating; verify bit-exactness of
   the unrotated half in the existing harness; phase_timer then 900 s pair. Prior for a net win:
   ~30%, magnitude ≤ 0.002.
4. **Recursive's cheap one-liners as a bundle** (already in plan §11.2 item 3): shifted ReLU²
   `relu(h−0.5)²`, sandwich norm on the MLP, input-dependent x0 gate. Zero throughput cost by
   construction; this is where the GPU-discarded selective gate gets its fair Trn2 hearing.

Do not revisit: sliding window, doc masking, `ROPE_BASE` as a standalone item, `VE_WD`, MQA
(unless the width changes), anything memory-motivated.

## 4. Closed on Trn2 for Trn2-specific reasons (out of scope, listed so nobody confuses them)

FP8 via `torch._scaled_mm` (no GEMM path; 46 s/call), stochastic rounding (scorer's causality
tolerance), fused QKV (breaks the rope+QK-norm kernel path), XSA (worse both axes), MUDD and
DCMHA (compile limits / negative at DEPTH=6), Mamba2/SSD hybrids, RWKV x051a, SLAY/YAT,
value residual (negative at the locked shape), requeue packer, LNC=1/8 ranks (compile bug +
0.24 bpb cliff), `--nki-relu2` (ambiguous). Each has a hardware measurement in the notebook.

## 5. General lesson for the remaining weeks

The GPU→Trn2 flips all had the same shape: a technique's *cost* changed because launches,
masks, fusion or memory are priced differently on Neuron, while its *quality* effect did not.
So when triaging any future idea from GPU literature, ask two separate questions: (a) does the
quality claim hold at ~50M parameters and ~2200 steps (independent of hardware), and (b) can
the work be folded into an existing launch, avoid additive masks, and stay off the elementwise
engines. Only ideas that pass (a) are worth testing (b) on the box.
