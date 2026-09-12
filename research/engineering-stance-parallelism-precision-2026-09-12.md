# Engineering stance: parallelism, precision, and the Trainium training references we used

Written 2026-09-12 to consolidate decisions that were previously scattered across
`train.py` toggle comments, `research/trn2-experiment-results.md` (FP8 sections, LNC A/Bs,
T1.2), `research/trn2-optimization-research.md` (HLAT digest) and the frontier plan (T1.11,
T4.4). Nothing here is new evidence; every number cites the place it was measured. Purpose:
stop future agents from re-opening tensor/pipeline parallelism, FP8, or stochastic rounding
without new facts.

## 1. Parallelism: pure data parallelism over 4 logical cores, and why

**What runs.** `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`.
The Trainium2 chip has 8 physical NeuronCores; LNC=2 fuses pairs into 4 logical cores, one per
rank. Each rank holds a full replica of the ~56M-parameter model. `TOTAL_BATCH_SIZE`
131,072 tokens is split across the 4 ranks; with `DEVICE_BATCH_SIZE=1` and `PACK_FACTOR=4`
each rank runs 4 micro-steps of 8,192 tokens (4 packed rows of 2048), then gradients are
averaged with `dist.all_reduce(AVG)` (`sync_gradients`, 62 calls/step, 1.2–3.5% of step time
per the T1.1 phase table; a bit-exact bucketed `sync_gradients_flat` exists, T1.2). Optimizer
state is replicated (no ZeRO; it would save memory we do not need).

**The only "tensor parallelism" we use is LNC=2 itself.** Inside a logical core the compiler
splits each operator across the two physical cores. It is intra-op parallelism handled by the
toolchain: no collectives in our code, no sharded weights, nothing to consolidate.

**Why not tensor parallelism (TP).** TP earns its cost when weights or activations do not fit
on one device, or when matmuls are large enough that their shards stay efficient. Neither
holds: the entire model and optimizer fit in a small fraction of one core's HBM, and the
matmuls are already tiny (`n_embd=640`, MLP hidden 2560). A 4-way split gives 160-wide
shards and adds two all-reduces per layer in forward and two in backward, on a graph that is
already dispatch-bound at ~8% of peak (plan §0). Strictly worse.

**Why not pipeline parallelism (PP).** Six layers over four stages is 1.5 layers per stage;
with 4 micro-steps the bubble is ≈ (stages−1)/micro-steps = 3/4 of a stage-time idle per step.
That is far larger than the 1–3% we pay for all-reduce. PP also needs many micro-batches to
amortise, which pushes toward smaller micro-steps, the opposite of what Trn2 wants
(T1.5: bigger packed micro-steps were a real win).

**Rules constraints that fix the shape.** Submissions "must run on the full chip", so the only
legal shapes are 4 ranks at LNC=2 or 8 ranks at LNC=1. The scorer requires a "consolidated,
single-device, replicated full model" with "no collectives in forward", so any TP/PP/FSDP
scheme would have to be merged before `final.pt` is written regardless.

**Measured, at the locked shape.** LNC=2/4 ranks beat LNC=1/8 ranks decisively
(`sol13-lnc2`: 639 vs 500 steps at full budget; notebook "LNC=2 A/B ... changes the launch
command") and became the launch command on 2026-09-06. LNC=1/8 ranks currently fails to
compile under the newest code (SDK `l2_norm_kernel_lnc0` bug) and, when forced to run, shows a
+0.24 bpb quality cliff at identical throughput whose mechanism is still open (T0.6 follow-ups
ruled out clip placement and the NKI kernels). None of that changes the decision.

**Where the parallelism-adjacent slack actually is.** (a) All-reduce: 62 launches/step; the
bit-exact flat-buffer path (T1.2) cuts them to one and is available. (b) Micro-step size:
PACK_FACTOR 2→4 was a paired win (+2.6% steps); memory allows more if the compiler handles the
shape. (c) The dataloader's exposed host time (20–30 ms/step, T1.1). Everything else that
moved throughput this competition was per-core work: torch.compile (452→67 launches per
micro-step), tensor-valued LR scalars (2.4× steps), foreach optimizer, NKI Muon, optlevel=1.

## 2. Precision: fp32 masters, bf16 compute, fp32 loss — and FP8 is closed

**What runs.** Parameters are fp32; `COMPUTE_DTYPE = torch.bfloat16` for activations and
matmuls, with `USE_WEIGHT_CACHE` casting each Linear's weight to bf16 once per optimizer step
instead of every forward; the fused softcap cross-entropy NKI kernel accumulates its
log-sum-exp and loss in fp32 (verified to 2.44 bf16 ULP forward / 0.22 ULP backward against
fp64). Muon's Newton–Schulz runs in the dtype the notebook verified bit-exact against the
eager reference when the NKI kernels were added.

**FP8: implemented, measured on hardware, closed as a double negative** (notebook "FP8
training-path precision — Phase 1/2", `USE_FP8_LINEAR`, commit `d5aa87c`). Design followed
modded-nanogpt records #84/#89: current-scaling, forward-only fp8 for the MLP matmuls,
bf16/fp32 backward. Results:
- *Speed, decisive:* there is no fp8 tensor-engine GEMM exposed through this PyTorch stack.
  The only numerically correct route, `torch._scaled_mm`, measured **46,366 ms per call**
  against 0.340 ms for bf16 `F.linear` at the real `[2048,640]×[640,2560]` shape, on every
  call, not a compile artefact. `torch.matmul` on fp8 operands does not raise but returns an
  all-NaN fp8 result at this shape. The bf16→fp8 cast alone (3.1 ms) and fp8→bf16→bf16-matmul
  (5.2 ms) are sane, so fp8 as a dtype is fine; the GEMM path is simply absent.
- *Quality:* fp8 forward relative L2 error 0.0374 vs bf16's 0.00287 against an fp64 reference,
  i.e. **13× the error**, and the hardware number reproduced the CPU harness
  (`research/scratch/fp8_linear_equiv.py`) to four decimals.
- Kept in-tree, default off, `--fp8-linear` for A/B, per the honest-negatives convention.

**The real FP8 path is a hand-written NKI kernel** using `nki.isa.nc_matmul(...,
perf_mode=double_row)` with both operands fp8, which is the only way Trainium2 runs fp8 at
2× bf16 tensor-engine throughput (plan T1.11). It is deferred for a structural reason: the
step is dispatch/launch-bound at ~8% MFU with TensorE-active efficiency ~63%, so doubling
TensorE peak buys little until Tier 1 work makes the graph TensorE-bound (the plan's gate is
">~60% of the step on TensorE"). It is also a large kernel project with its own equivalence
harness, and the 13× per-matmul error would need a loss-trajectory study at 2000 steps.

**Stochastic rounding (the Neuron "dynamic" precision feature): tested, discarded.**
`NEURON_RT_STOCHASTIC_ROUNDING_EN=1` (HLAT's recipe) fails `causality_check` with
max_abs_diff 0.021 at the locked shape. The notebook proved this a false positive — SR's own
forward-to-forward noise on identical input is 0.0113, already above the harness's fixed 1e-2
tolerance — but the official scorer runs the same check with the same tolerance, so it is
unusable regardless of any throughput effect (`USE_STOCHASTIC_ROUNDING = False`).

**Things that sound related and do not apply.**
- Parameter Golf's int6/int5/GPTQ work exists to fit a 16 MB artifact. Phase 1 has no size
  cap; weight quantisation would add error at eval and buy nothing.
- modded-nanogpt's FP8 records are ~1% wins on GPU fp8 tensor cores through a mature
  `torch.compile` path; the mechanism does not exist on this stack (above).
- bf16 master weights (Recursive runs everything in bf16 with mantissa tracking, plan §11.2
  item 7): untested here; it would cut the per-step weight-cast (already cached, ~0.3% of the
  step) and optimizer memory traffic, at some quality risk. Low priority, not closed.

## 3. External Trainium training references: what was read and what came of it

### 3.1 HLAT (arXiv 2404.10630v2) — read on 2026-09-03/04, ten pages, PDF in `articles/`

7B/70B Llama-style pretraining on up to 4096 Trainium1 chips (TP=8, PP=1, DP=256 for 7B).
Every transferable item was turned into a Solution-13 toggle and A/B'd at the locked shape:

| HLAT item | our toggle | outcome |
|---|---|---|
| Q/K/V coalesced into one projection | `USE_FUSED_QKV` | **negative**: 206 vs 240 steps and worse bpb; it disables the rope+QK-norm NKI kernel fusion. Off. |
| Causal mask built on device so the compiler skips masked compute | `USE_CAUSAL_FASTPATH` | **adopted**: 265 vs 240 steps, −0.0166 bpb at the 900 s proxy; "the best of all four Solution-13 toggles". |
| `--model-type=transformer --distribution-strategy=llm-training`, `NEURON_FUSE_SOFTMAX=1`, `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3` | `USE_TRN2_COMPILER_FLAGS` | small win alone (246 vs 240 steps) but net negative combined with the causal fast path. Off; superseded by `--optlevel=1` (T0.6). |
| BF16 + stochastic rounding | `USE_STOCHASTIC_ROUNDING` | **discarded** (causality-check false positive, §2). |
| AdamW β2 = 0.95 (0.99 unstable at 7B/70B) | `EMBED_ADAMW_BETA2` 0.995 → 0.95 | **adopted**; re-confirmed in the Tier-0 resweep (995/1.094327 vs 977/1.097342). |
| Neuron persistent cache on local disk, not network FS | `NEURON_COMPILE_CACHE_DIR=/tmp/neuron_cache` | done from the first Trn2 session; cold-cache cost measured at ~30 s. |
| Online tokenisation/packing overlapped with the device | — | already how `prepare.py`/the requeue packer work; not a change. |
| Scaled output init 1/√(2·n_layer) | `ATTN_CPROJ_INIT_SCALE`, `INIT_SCALE` | covered by the T0.4 sweep (attn.c_proj init 0.10, INIT_SCALE 0.50 adopted). |
| "Develop on a small model, scale later" | — | is the whole methodology. |

HLAT was fully mined; nothing in it remains untested except its parallelism recipe, which is
inapplicable (§1).

### 3.2 AWS blog "Frugality meets accuracy: cost-efficient training of GPT NeoX and Pythia models with AWS Trainium" — not previously consulted; read 2026-09-12

Content: Pythia 6.9B (TP=8, PP=1) and GPT-NeoX 20B (TP=8, PP=4) trained with
`neuronx-nemo-megatron` on SLURM clusters of 1–64 `trn1.32xl` nodes, "full BF16 with
stochastic rounding", 10k pretraining iterations on a Wikipedia subset, ahead-of-time compile
job then run job, ~3.2M tokens/$ for 20B with near-linear scaling 4→64 nodes, and loss /
grad-norm curves that track an A100 P4d baseline. Compiler flags, ZeRO, sequence parallelism
and activation checkpointing are not stated; they live in the linked SLURM scripts.

Relevance to a 56M single-chip run: **none beyond what HLAT already gave.** Specifically:
- The parallelism recipe (TP=8, PP=4) is the large-model case §1 rules out.
- "Full BF16 + stochastic rounding" is the precision recipe §2 already tested and discarded
  for scorer reasons.
- The one Neuron-specific trick — implementing *partial-rotation* RoPE by zero-padding the
  frequencies and rotating the whole head dimension, which raised seq/s on Trainium — does
  not apply: we rotate the full head dimension inside a hand-written NKI rope+QK-norm kernel.
- The ahead-of-time compile-then-run pattern is what the runner's compile-time exclusion and
  our `--compile-only` flag already implement.
- The convergence-parity result is reassuring but was never in doubt for us; the numerics
  that matter here were verified kernel by kernel to bf16-ULP level.

No action items. Recorded so it is not re-read.

### 3.3 Other Trainium sources already in the record

`articles/` holds the NKI performance-optimisation guide, the FlashAttention-on-NKI deep
dive, the Llama2-on-EKS training guide, and the NKI library release notes; their digests are
in `research/trn2-optimization-research.md`. The Gemini "AI Mode" notes in
`research/gemini_ai_mode_trainium.md` proposed 8-way data parallelism and "pure FP8"
execution; both were overtaken by measurement (LNC=2/4 ranks won; FP8 has no GEMM path).

## 4. One-paragraph version for the plan

Data parallelism over 4 logical cores at LNC=2 is the only sensible layout for a 56M model
on one chip: TP shards 640-wide matmuls into inefficiency and adds collectives, PP has a
75%-of-a-stage bubble at 4 micro-steps, and the scorer needs a consolidated single-device
model anyway. Precision is fp32 masters / bf16 compute / fp32 loss; FP8 through PyTorch has
no tensor-engine path on this stack (46 s per GEMM) and costs 13× the bf16 error, the NKI
double-row route waits until the graph is TensorE-bound, and stochastic rounding fails the
causality check. HLAT was mined to completion (causal fast path and β2=0.95 adopted, fused
QKV / compiler flags / SR rejected); the NeoX-Pythia Trainium blog adds nothing for this
scale.
