# Trn2-Specific Optimization Research

Evidence base behind `experiment-rollout-plan.md`'s Solution 13 (config/env wins) and the
rewritten Solution 12 (NKI kernel work). Sources: 6 PDFs in `trainium_frontier/articles/`
and 4 cloned sibling repos (`nki-samples`, `nki-llama`, `nanochat-contribution-fork`,
`neuron-workshops`), read by three parallel research agents. This doc exists so nothing
gets re-researched — it records what each source *did* and *did not* contain.

## Source 1 — NKI kernel-writing PDFs (all read in full)

`NKI Performance Optimizations.pdf` (30pp), `Release Notes for Neuron Component NKI
Library.pdf` (9pp), `Deep Dive into FlashAttention on AWS Neuron NKI.pdf` (6pp).

**Profiling methodology.** `neuron-profile` GUI Summary tab: `mbu_estimated_percent` /
`mfu_estimated_percent`, per-engine `*_active_time_percent`, `dma_active_time_percent`.
Compare engine-active vs. DMA-active % to tell compute-bound from data-movement-bound.
For idle gaps: enable "Show unsampled data," read the semaphore wait condition on the
first instruction after the gap — `S[n] (Scalar)` = waiting on another engine
(pipelining issue), `q*` prefix = waiting on a `nl.load` DMA, `qSyncSpillReload0` =
blocked on spill traffic. Hovering a DMA gives `nki_source_location` back to the source
line.

**Fusion.** "Operator fusion is exactly done through explicit loop fusion" — one
`for tile in kernel_in_hbm: load → op0 → op1 → store` instead of separate loops with
spill_save/spill_reload between them. Watch `spill_save_bytes`/`spill_reload_bytes` — if
either is >~30% of `sbuf_read_bytes`/`sbuf_write_bytes`, the compiler is round-tripping
through HBM between ops. **Gotcha**: declare SBUF buffers inside the innermost loop, not
hoisted above it, or you get unexpected spilling.

**Instruction combining.** `nl.multiply`+`nl.add`+`nl.exp` (3 instructions) → single
`nisa.activation(nl.exp, data, bias, scale)` = 3x lower latency, one pass over data. Also
`nisa.tensor_scalar(op0=..., op1=...)` for two-op fusion, `nisa.tensor_tensor_scan` for
scans. Directly relevant to our softcap+cross-entropy fusion (Solution 12 step 4).

**RMSNorm layout.** Unlike LayerNorm's `nisa.bn_stats` (forces free-dim reduction),
RMSNorm's square+sum has no layout requirement — reduce along the free dim with VectorE
`nisa.tensor_reduce`, or along the partition dim with TensorE `nisa.nc_matmul`. In a
`linear→RMSNorm→linear` chain (our `norm()` sandwiched between Linears throughout
`Block`/`CausalSelfAttention`), map the first linear's weights as the *stationary*
tensor and use TensorE cross-partition summation to skip transposes.

**Tile/partition sizing.** Engines want ≥128 elements/partition in the free dim, else
pay ~189ns initiation interval for ~1 cycle of useful work. Use all 128 partitions —
`channels=64` costs the same as `channels=128` (our `n_embd=640`, `HEAD_DIM=128` — the
128 constant already matches).

**DMA sizing.** Target >32 KiB/transfer. Avoid `nl.load_transpose2d` (low bandwidth);
prefer `nl.load` + `nisa.nc_transpose`. **NKI currently runs on a single NeuronCore** —
our submission shape is `nproc_per_node=8`; DDP replicates the whole model per core so
each core's kernel invocation should be independent, but this has never been exercised
on any hardware and needs explicit verification, not an assumption.

**Attention engine mapping.** matmuls→TensorE, exp→ScalarE, sum→VectorE,
reciprocal-scale→ScalarE, causal triangular masking→GpSimdE.

**Prebuilt `nkilib` kernels (v2.29.0), check before hand-rolling:** RoPE Kernel
(contiguous + interleaved layouts), `rmsnorm_tkg`/`layernorm_tkg` subkernels, Cross
Entropy Kernel (online log-sum-exp, memory-efficient for large vocab), Attention CTE/BWD
(`use_causal_mask`, `softmax_scale` params), and an "Attention Block TKG" fusing
RMSNorm+QKV+RoPE+attention+output-proj with a `rmsnorm_QK_pre_rope_W_Q`/`_W_K` param —
our exact QK-norm-before-RoPE architecture. **Caveat: TKG = token generation
(inference), not training** — likely not directly usable for training-time kernels;
worth a short check on the real SDK, no more.

**Version-pinning trap.** Namespace moved `neuronxcc.nki.*` → `nki.*` in SDK 2.28;
`nl.par_dim` was removed from library internals in 2.29 even though the Perf-Opts doc
still shows it. Check the installed NKI version on the real Trn2 DLAMI before copying
any API call verbatim from these PDFs or from the recovered git history below (which is
from SDK ≤2.32).

**FlashAttention reference structure** (forward-only in this source, no backward
shown): SBUF ~2MB forces tiling (a 512×512 score matrix at 128 lanes = 64MB, way over
budget). Outer loop over Batch×Heads → 128 P-dim lanes; inner loop tiles K/V in
128-column chunks; TensorE does 128×128 matmul per burst; VectorE updates the
online-softmax rescale `exp(m_old - m_new)` and accumulator in-place in SBUF; async DMA
prefetches the next K/V block during the current matmul. All in **one `@nki.jit`
function** to avoid multi-launch overhead. Write-back is async — fire `nl.store` as soon
as a row-block is fully normalized.

## Source 2 — existing kernel implementations in cloned repos

### `nki-samples` — the good stuff is in git history, not HEAD

Current HEAD (`ea7dfed`, SDK 2.32) has only `src/nki_samples/tutorials/` (attention
perf, mamba, matmul, pools, transpose, mxfp) plus `contributed/`. The reference kernel
library and legacy Neuron-team kernels were **deleted** in `8370e11` ("Remove legacy
kernels", #115) and `c4b806e` ("Remove reference kernels", #116) — still retrievable via
`git show <commit>^:<path>`.

**RMSNorm:**
- `git show 8370e11^:src/nki_samples/tutorials/rmsnorm/rmsnorm_nki_kernels.py` —
  `@nki.jit`, `nl.affine_range(ceil(rows/128))`, `nl.load(mask=...)`, `nl.square` →
  `nl.sum(axis=[1])` → `nl.rsqrt` → `nl.multiply` with `broadcast_to`. Forward-only.
  JAX variant also present (`rmsnorm_jax.py`).
- `git show 8370e11^:contributed/neuron-team-kernels/rmsnorm_quant.py` (+
  `src/nki_samples/reference/rmsnorm_quant/rmsnorm_quant.py` at `c4b806e^`) —
  production-grade: `nl.par_dim(GROUP_S_SZ)` 3D SBUF tiles,
  `nisa.activation_reduce(op=nl.square, ...)` fusing square+reduce in one instruction,
  `nisa.activation(op=nl.rsqrt)`, gamma broadcast via `nisa.nc_matmul(ones, w)` on
  TensorE. **This is the pattern to copy.**
- LayerNorm (for context, not our target):
  `git show 8370e11^:src/nki_samples/tutorials/layernorm/layernorm_nki_kernel.py` — v1
  naive (`nl.mean`, `nl.rsqrt`), v2 using `nl.tile_size.bn_stats_fmax` /
  `bn_stats`+`bn_aggr`. Both forward-only.

**RoPE:**
- `git show 8370e11^:contributed/neuron-team-kernels/RoPE.py` —
  `RoPE_sbuf(x_in_sb, cos_sb, sin_sb, x_out_sb)` on `[d_head, S]` layout (head dim on
  partitions, `d_head ≤ 128`), cos/sin `[d_head//2, S]`, four `nisa.tensor_tensor`
  multiplies + one add/subtract for the rotate-half form. Forward-only. **Layout is
  transposed vs. our `[B,S,H,D]`.**
- `git show 8370e11^:contributed/neuron-team-kernels/llama3_transformer.py:44`
  `transpose_and_RoPE(...)` — fused transpose+RoPE from a QKV projection output.
  `:135` `sp_tile_fused_add_rmsnorm_all_gather` / `:194` `tiled_rmsnorm_qkv` use the
  builtin `nl.rms_norm(x, w, axis=-1, n=D, epsilon=...)` — worth knowing this exists.

**Attention:**
- `git show c4b806e^:src/nki_samples/reference/attention.py` — full flash attention:
  `flash_fwd` (line 284, supports `logit_bias`, causal mask, dropout, online max/sum in
  `_flash_attention_core` line 85) **and** `flash_attn_bwd` (line 521) — the only
  fwd+bwd kernel in the repo. Also `fused_self_attn_for_SD_small_head_size` (line 1081).
- `git show c4b806e^:src/nki_samples/reference/allocated_attention.py` — same math with
  explicit SBUF/PSUM allocation.
- Live in HEAD:
  `nki-samples/src/nki_samples/tutorials/attention_fwd_performance/attention_kernels.py`
  — `attn_fwd_v1`…`v10`, a step-by-step optimization ladder (v10 uses
  `SBLinearAllocator`/`PsumAllocator` `nl.NKIObject` classes, lines 937–1155). Best
  learning resource for a flash-style kernel. **Forward-only.**
- `contributed/pipelined_attention.py`, `contributed/decode_attention.py` (online
  softmax, GQA, `nisa.tensor_scalar`/`tensor_reduce`).

**Softmax / cross-entropy / softcap: no standalone kernel anywhere, HEAD or history.**
Softmax only exists inline inside the attention kernels. Fusing our softcap into
cross-entropy is genuinely from-scratch work unless the `nkilib` Cross Entropy Kernel
(Source 1) is importable on the real SDK.

### `nki-llama` — NxD inference wrapper, one NKI kernel

`nki-llama/llama.py` is a `neuronx_distributed_inference` model (imports
`NeuronAttentionBase`, `ColumnParallelLinear`, HF `LlamaRotaryEmbedding`). Only one
hand-written NKI kernel: `nki_rmsnorm_kernel` (`llama.py:89-147`) — a 3D `[B, S, H]`
copy of the tutorial RMSNorm but using plain Python `range` instead of
`nl.affine_range` (a correctness/perf gap to note if copying it), gated by
`--enable-nki` via `CustomRMSNorm` (`llama.py:150-172`). Everything else uses prebuilt
`neuronxcc.nki._private_kernels` ISA kernels (`mlp_isa_kernel`,
`rmsnorm_quant_isa_kernel`) wrapped in `torch_neuronx.xla_impl.ops.nki_jit`.
Attention/RoPE are NxD/HF torch, not NKI. **Inference-only, TP-based** — not our launch
shape.

### `nanochat-contribution-fork` — not Trainium related

Zero Trainium/Neuron code. Only hits for "neuron" are "per-neuron"
(`nanochat/optim.py:84-85`) and a NorMuon mention in `dev/LOG.md:1029`. CUDA-targeted:
`nanochat/flash_attention.py:25` gates on `torch.cuda.is_available()` and loads
FlashAttention-3. Value is architectural only — `nanochat/gpt.py` is the same lineage as
our model (rotary `:57`, `relu().square()` `:139`, parameter-free `norm()` `:42`). **No
kernel content to extract.**

## Source 3 — Trn2 training-practice PDFs + `neuron-workshops`

### Llama2-on-EKS (16 pages read) — mostly infra, three transferable bits

Llama2-**7B** via AWS Neuron Reference for NeMo Megatron, 4× trn1.32xlarge = 64
Trainium1 chips / 128 NeuronCores, MPI Operator on EKS, FSx Lustre, EFA, TP+PP+SP+DP,
ZeRO-1, selective activation checkpointing. ~20,101 steps at ~28–29 s/it. ~80% of the doc
is Terraform/kubectl/Tensorboard and does not transfer to a single-chip 86M-param run.

Transferable:
- **Ahead-of-time graph compilation is a separate job** producing NEFFs into a
  **persistent Neuron cache** the training job then reads (~10 min for 7B). For a
  30-minute budget, the equivalent — warm the Neuron persistent cache before the timed
  run — is the single most valuable idea here. Our `--compile-only` flag already exists
  for this.
- **Offline tokenization is expensive** (~60 min on trn1.32xl for a 1B-token sample) —
  pre-tokenize once, never inside the timed loop. `prepare.py` already does this; no
  action needed, but recording it closes the question.
- `neuron-top` for per-NeuronCore utilization (92–100% in their run) — the "are we
  actually using all 8 cores" check, relevant to the correctness gate, not just speed.

### HLAT (10 pages read) — 7B/70B on up to 4096 chips, several real nuggets despite scale

Scale is irrelevant (256× trn1.32xlarge, 1.8T tokens, TP=8/PP=1/DP=256 for 7B).
Trainium-motivated choices that plausibly transfer to an 86M single-chip run:

- **Online dataloader**: tokenization + packing during training, on CPU, parallel with
  device fwd/bwd — "no impact on training throughput," contrasted with NeMo's offline
  154h-to-pretokenize-2T-tokens path. Not applicable — `prepare.py` already
  pre-tokenizes + packs, and our run is far too short for online tokenization to matter.
- **Layer coalescing**: fuse Q/K/V into one `qkv_proj`, same for SwiGLU's two
  projections — fewer, larger matmuls (Listing 1). **Confirmed applicable to us**:
  `train.py:356-358` has three separate `Linear`s for Q/K/V. → Solution 13c.
- **Constant attention mask built on device** via `torch.triu` on `device='xla'` instead
  of passing a mask tensor — compiler then skips masked-out compute. → Solution 13d
  (`is_causal=True`, drop the additive mask when purely causal).
- **BF16 + stochastic rounding** gave higher throughput and lower memory with equal
  convergence at 7B (diverged only at 70B). → Solution 13b.
- Compiler/runtime flags: `--distribution-strategy=llm-training`, `--model-type=transformer`,
  `NEURON_FUSE_SOFTMAX=1`, `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3`. → Solution
  13a. **Caveat we're carrying forward honestly**: these were used in a TP/PP-sharded
  context; verify individually that they aren't no-ops or regressions under plain DDP.
- **Store the Neuron persistent cache on local disk, not shared/network FS** — shared
  FSx caused communication bottlenecks/timeouts.
- Hyperparameters: AdamW **β2=0.95** (β2=0.99 caused instability at 7B/70B) — our
  embedding/VE groups use `betas=(0.8, 0.995)` (`train.py:505,510`), a mismatch worth a
  Trn2-side test (deferred, not GPU-retested, per explicit decision). Grad-clip 1.0,
  cosine LR to 10% of max, linear warmup, WD 0.1 on all params including norms (no
  measured difference vs. excluding). Scaled init: attention/MLP output layers scaled by
  1/sqrt(2·n_layer) — also deferred to Trn2.
- Their own advice: "develop and test on a smaller model and scale up afterwards" —
  recompilation is the dev-loop tax. Already our practice this whole session.

### Inference-silicon doc (4 pages) — no hard numbers

Purely qualitative (SBUF is a manually-managed scratchpad; overflow stalls; HBM
round-trips defeat fusion; use NKI for forced fusion; `neuron-profile` for pipeline
stalls). **No SBUF capacity, PE array dimensions, or Trainium2 HBM bandwidth figure
anywhere in it.** The only hard numbers available anywhere in the cloned material are
from the NKI lab (Source 1): 128 partition lanes, ~20x HBM bandwidth vs. SBUF, and
Trainium1 = 2 NeuronCore-v2 / 32 GiB HBM @ 840 GiB/s (Trainium1, not Trn2 — no Trn2-specific
figure found in any source).

### `neuron-workshops` — no true match for our setup

Repo is inference-heavy (NxD, NKI, vLLM, Chess). Only training code:
`labs/FineTuning/HuggingFaceExample/01_finetuning/assets/finetune_model.py` (+
`finetune_chess_model.py`, notebooks for Qwen3-1.7B/TinyLlama-1.1B) — LoRA SFT via
`optimum.neuron` `NeuronSFTTrainer`/`NeuronTrainingArguments` on trn1.2xlarge with
`torchrun --nproc_per_node 2`, but **`tensor_parallel_size=2`, not DDP** — structurally
close only in launch shape. Reusable habits, not code: bf16 on, `packing=True` +
`max_seq_length` in the SFT config, `gradient_accumulation_steps` as a first-class knob,
`is_master_ordinal()` gating rank-0-only checkpoint/consolidation work
(`torch_xla.core.xla_model`), `NEURON_RT_VISIBLE_CORES` to restrict visible cores.
Checkpointing is sharded per-rank with a separate consolidation script — reminder that on
Neuron, saving is a collective/multi-file affair; keep it out of the 30-minute hot path.
**No training code applicable to plain single-chip DDP found.**

## Source 4 — Gemini Deep Research cross-check (`Trainium 2 Training Optimization Gemini Deep Research.pdf`, 18pp, read in full)

**Caveat first**: this report's "GPU Proxy Ablation Dynamics" section (pages 1-6) reproduces
our own `gpu-experiment-results.md` numbers to the decimal (baseline `1.273797`, Solution 1
`1.215996`, final `1.106820`, etc., including the exact `PACK_FACTOR` sweep and the SwiGLU/
selective-gating discards). This is not an independent replication — Gemini was evidently
given our own repo/results as input. Treat that section as a restatement, not new evidence.
The hardware/SDK sections (pages 6-18) draw on public AWS Neuron docs (see its Works Cited)
and are genuinely new material, cross-checked below against what Source 1's PDFs already
covered.

**Genuinely new, worth acting on:**

- **`--internal-hlo2tensorizer-options='--fast-relayout=true'`** — **disproven on real Trn2 hardware, do not use.** Claimed a `neuronx-cc` flag implementing tensor transpositions via TensorE identity-matrix matmuls instead of software-emulated permutation loops, ~2x speedup. When actually tried (`trn2-experiment-results.md`, Solution 13a), it broke `neuronx-cc` compilation outright — `neuronx-cc compile --help` on the installed SDK (2.26.6360.0) confirms this flag and `--fast-relayout` **do not exist** on this version. Exactly the "verify on real hardware before trusting the claim" caveat this doc already carried, borne out. `--distribution-strategy=llm-training --model-type=transformer --lnc=1` are all confirmed present in `--help` and were verified crash-free and numerically clean in isolation and combined.
- **`NEURON_COMPILE_CACHE_DIR=/tmp/neuron_cache`** — the exact env var for the "persistent
  cache on local NVMe, not network FS" advice already in the first-hour checklist (item 6)
  and in the EKS/HLAT sources; we had the *advice* but not the *variable name* until now.
- **`neuron_parallel_compile`** — the official SDK CLI that traces the model graph across
  all ranks simultaneously and multi-threads the compiler lowering across host CPUs, run
  once before the timed window starts. This is the AOT-precompile mechanism the SDK
  actually ships, as opposed to our own `--compile-only` flag (which forces synthetic steps
  through `train.py` itself). Check on the real box whether `neuron_parallel_compile` works
  against our custom `train.py` directly, or whether `--compile-only` is the right tool for
  our case and `neuron_parallel_compile` is redundant/inapplicable. Open question, not yet
  resolved.
- **Small-DMA-Trap threshold, exact formula**: `F ≥ 32768 bytes / (128 partitions × bytes/element)`
  — 64 elements for FP32, 128 elements for BF16, to clear the 32 KiB/transfer floor.
  Sharpens Source 1's general ">32 KiB" guidance with the actual per-dtype number. Directly
  relevant to any kernel touching our 32-channel gates (`VE_GATE_CHANNELS`,
  `SELECTIVE_GATE_CHANNELS`) — those are well under 128 partitions, so the **partition
  vectorization** technique applies: pack multiple logical 32-wide vectors into disjoint
  slices of one 128-wide SBUF tile rather than running a 32/128-occupancy instruction.
- **PF-transpose elimination via layout co-design**: naming a mechanism our RMSNorm-kernel
  note already implied but didn't name — mapping a `Linear`'s weight as the *stationary*
  TensorE matrix and the activation as the *moving* matrix swaps the row/column emission
  layout in PSUM, pre-aligning activations with VectorE's reduction requirement and
  eliminating the compiler-inserted PF-transpose instruction entirely. Same trick as Source
  1's "map the first linear's weights as stationary" note, more precisely named.
- **Hard Trn2 numbers, previously flagged as absent from every source** (see "found empty"
  below, now superseded): 8× NeuronCore-v3 per chip, 667 dense BF16 TFLOPS / 1,299 FP8
  TFLOPS / 181 FP32 TFLOPS, 96 GiB HBM3 @ **2.9 TB/s**, DMA aggregate **3.5 TB/s**, 224 MiB
  SBUF total (28 MiB/core, ~20x HBM bandwidth — matches Source 1's qualitative "~20x" claim
  with a real number now), 16 MiB PSUM (2 MiB/core), NeuronLink-v3 1.28 TB/s in a 2D-torus
  (irrelevant to Phase 1's single-chip scope). TensorE natively supports FP8 (E4M3/E5M2) —
  not recommended to pursue given the precision sensitivity already observed (β2/stochastic-
  rounding interaction below); flagging as known-available, not a plan item.
- **Quantified recompile cost**: 60-180s per JIT recompile event; 2-3 unintended events can
  burn 15-20% of the 30-minute budget. Hard number for what the first-hour checklist already
  treats qualitatively as a reason to keep shapes static and precompile ahead of time.
- **Stochastic rounding × AdamW β2 interaction, sharper mechanism**: stochastic rounding
  injects high-frequency zero-mean variance into every gradient; β2=0.99 (HLAT's own value,
  and close to our current embedding groups' 0.995) averages over too long a window to track
  it, causing instability — β2=0.95 lets the second-moment estimate adapt fast enough. This
  is the same conclusion HLAT already gave (already in Solution 13/checklist as "deferred to
  Trn2"), but now with a mechanism specifically tied to stochastic rounding rather than just
  scale — worth noting if we enable stochastic rounding (13b) and the β2 toggle together,
  since the report implies they're coupled: turning on SR *without* also dropping β2 may
  reproduce the instability HLAT saw.
- **ZeRO-1 optimizer-state sharding across the 8 DP ranks** — genuinely new lever, not in
  any of our other sources. AdamW+Muon optimizer state for an 85.9M-param model is ~1.0-1.4
  GB/rank; with 12 GiB/rank available under LNC=1 there's no memory pressure forcing this.
  **Assessed as low priority**: it's a memory optimization we don't need at this model size,
  and adopting it means a `neuronx_distributed` dependency and sharding/gather logic we
  don't currently have — a real complexity cost for zero measured benefit at our scale. Note
  for the record, don't implement unless the depth/width resweep (Solution 11) pushes the
  model meaningfully larger than 85.9M and memory actually becomes tight.
- **LNC=1 vs LNC=2 decision, quantified**: report's own table puts the crossover at ~100M
  params (LNC=1/pure-DP=8 optimal below that, LNC=2/hybrid optimal for memory-bound models
  above ~500M). Confirms — doesn't change — our existing default of `--lnc 1`. Re-check this
  cutoff if Solution 11's resweep pushes depth/width meaningfully upward.
- **VectorE/GPSIMD shared-bus "high-performance mode"**: claimed 2-4x speedup on select
  BF16/FP16 vector transforms by sharing a wide physical memory bus between VectorE and
  GPSIMD. No activation mechanism given in the report (unclear if it's automatic, a compiler
  flag, or requires specific instruction selection). Open question — check Neuron SDK release
  notes/docs on the real box; don't assume it's on by default.

**Checked and consistent with what we already had (no new information):** RMSNorm/RoPE/
softcap-into-CE kernel priority order and rationale, `NEURON_FUSE_SOFTMAX=1` /
`NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3` / `--distribution-strategy=llm-training` /
`--model-type=transformer` (all already in Solution 13a), `record_step_boundary` submission
requirement, static-shape/AOT-compile discipline, NKI FlashAttention online-softmax
structure, hardware transpose acceleration as the mechanism the report claimed
`--fast-relayout` used (moot — the flag doesn't exist on the installed SDK, see above).

## What was checked and found empty (don't re-research)

- No standalone softmax or cross-entropy NKI kernel anywhere in `nki-samples`, HEAD or
  history.
- No Trainium2-specific hard numbers (SBUF size, PE dims, HBM bandwidth) in any of the
  original 6 PDFs — everything concrete there is Trainium1 or generic-NKI. **Superseded**:
  Source 4 (Gemini Deep Research report, added later) supplies these from public AWS docs —
  see its section above for the actual figures.
- `nanochat-contribution-fork`: zero Neuron/Trainium code, CUDA/FA3-only.
- `neuron-workshops`: no DDP training code, only TP-based LoRA SFT.
- `nki-llama`: inference-only (NxD), one hand-written kernel (RMSNorm, using plain
  `range` not `nl.affine_range` — a gap, not a template to copy verbatim).
