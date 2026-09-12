# Frontier Research Plan — Phase 1 val_bpb push (written 2026-09-09)

Deliverable of the deep-research pass requested in `research/frontier-research-prompt.md`.
This file is an **instruction set for the agentic experiment loop**. It assumes the loop has
read `README.md`, `trn2-experiment-results.md` (the lab notebook) and `train.py`, and that it
follows the measurement discipline already established there (paired same-day A/Bs, escalate
900s proxies to 1800s before locking capacity changes, log sync after every run).

Sources consulted for this plan (all verified live, not from memory): the upstream competition
repo via `gh api repos/aws-neuron/trainium-frontier` (PRs 1–6, README/prepare.py identical to
local), the devpost forum rulings (topic 45042 custom packing, 45077 compile exclusion), the
modded-nanogpt record table and record write-ups #26–#89, the current `train_gpt.py` of
modded-nanogpt, "Parameter Golf: What Really Works?" (arXiv 2607.01517, EMNLP-BabyLM 2026) plus
its dataset repo `PMP56/pmgolf-analysis` and the OpenAI `parameter-golf` leaderboard/PRs,
DeepSeek "Conditional Memory via Scalable Lookup" (Engram, arXiv 2601.07372v2), "Fast Catch-Up,
Late Switching: Optimal Batch Size Scheduling" (arXiv 2602.14208), "Seesaw" (arXiv 2510.14717),
"Time is Not Compute" (arXiv 2603.28823), the AWS Neuron docs for Trainium2, `nki.isa.nc_matmul`,
the TorchNeuron-native overview, and the `neuronx-cc` CLI reference (`--optlevel`, `--model-type`,
`--auto-cast`, `--enable-mixed-precision-accumulation`, `--logical-nc-config`).

---

## 0. Where we stand, and the thesis of this plan

**Locked baseline (best confirmed, full 1800s):** `2000 steps / public_val_bpb 1.042278`
(`gradclip1-fullbudget`). Config: DEPTH=6, n_embd=640, n_head=5 (HEAD_DIM=128), SEQ_LEN=2048,
TOTAL_BATCH_SIZE=131072, PACK_FACTOR=2, DEVICE_BATCH_SIZE=1, LNC=2 / 4 ranks, `torch.compile`
ON, NKI rope+QK-norm ON, NKI softcap+CE ON, NKI Muon elementwise ON, foreach optimizer ON,
tensor-lr scalars ON, EMBED_ADAMW_BETA2=0.95, GRAD_CLIP=1.0, WARMDOWN_RATIO=0.75,
FINAL_LR_FRAC=0.05, INIT_SCALE=0.68, VE on alternating layers, `dt ≈ 0.90 s/step`,
`≈145k tok/s`, 261.5M tokens seen, epoch 1 throughout.

**Leaderboard context:** top scores were reported at `1.00–1.06` around Sep 6–7. The last
submitted entry scored `1.1394` on the private shard for a local `1.15361` (local ≈ leaderboard
+0.014 at that time, SEQ_LEN=1024). Phase 1 closes **2026-09-30**; 5 submissions/week ⇒ roughly
15 submissions remain. A realistic target to be safely inside the top 10 is **≤ 1.00 local**,
i.e. ≈ 0.04–0.05 bpb below the current best.

**How much is a step worth?** The notebook's fit `bpb ≈ a − 0.077·ln(steps)` on current-code
runs is predictive (it predicted the NKI-Muon delta within 0.0002). Use it as the exchange rate:

| throughput gain | extra steps | bpb gain |
|---|---|---|
| +5% | 2000 → 2100 | −0.0038 |
| +15% | 2000 → 2300 | −0.0108 |
| +30% | 2000 → 2600 | −0.0202 |
| +50% | 2000 → 3000 | −0.0312 |
| +100% | 2000 → 4000 | −0.0534 |

**The single biggest fact this plan is built on: we run at ≈ 8% MFU.** Per step:
`6 × 55.7M × 131072 ≈ 4.4e13` matmul FLOPs + `12·h·d·T·tokens ≈ 2.1e12` attention FLOPs
≈ `4.6e13` FLOP in 0.90 s ⇒ **≈ 51 TFLOP/s against a 667 TFLOP/s bf16 chip peak (7.7%)**. The
post-compile profile confirms why: one micro-step of `[2, 2048]` tokens takes a 70.9 ms forward
NEFF + 77.1 ms backward NEFF on one physical core (= `1.43e12 FLOP / 0.148 s ≈ 9.7 TFLOP/s`,
12% of a core's 83 TFLOP/s), TensorE-active efficiency only 63%, +75% transpose instructions
vs eager, "compiled matmuls are smaller". The isolated eager `F.linear` at `[2048,640]×[640,2560]`
measured 0.340 ms = 19.7 TFLOP/s = 24% of one core. Everything architectural in this plan is worth
0.002–0.01 bpb per item; a 1.5× throughput fix is worth 0.03 on its own. **Throughput is Tier 1.**

**Three other systematic gaps found in the notebook:**
1. **Depth and width were never decoupled.** `build_config` derives `n_embd = round_up(DEPTH×ASPECT_RATIO, 128)`, so DEPTH=7 was only ever tested at n_embd=768 and DEPTH=8 at 768; width=768 at DEPTH=6 lost by only 0.0007. "DEPTH=7 or 8 at n_embd=640" has never run.
2. **Hyperparameters were re-swept after the 2.4× step-count jump, but only the capacity ones** (depth, width, batch, momentum warmup, β2). `MATRIX_LR=0.025`, `EMBEDDING_LR`, `UNEMBEDDING_LR`, `WEIGHT_DECAY`, `WARMDOWN_RATIO`, `FINAL_LR_FRAC`, `INIT_SCALE`, `LOGIT_SOFTCAP`, `HEAD_DIM`, the MLP ratio and the attention scale have never been swept on Trn2 at all (most were inherited from the GPU proxy at ~800–1450 steps, some from Karpathy's H100 run). modded-nanogpt's most recent records moved exactly these (min-LR 0.1→0.15, cooldown 0.5→0.6, attn scale 0.1, Muon LR 0.023 at 768-wide).
3. **`USE_LOCAL_CONV` is computed every step but its weight is in no optimizer group and stays zero** (proven in the notebook). The intended mechanism, a learned depthwise causal conv before attention, has therefore **never been tested on Trn2**, and the compute for it is already paid. Registering the weight with the optimizer is the cheapest capacity experiment available.

**Where to stand out from the crowd:** every strong team will have compile, fused kernels,
Muon and a tuned schedule. Two ideas are both well-evidenced elsewhere and unlikely to be widely
attempted here (**see §11: both best-known solutions to the nanochat autoresearch task, Recursive 0.9109 and Aster 0.9098, credit exactly this**): (a) **hashed n-gram (bigram/trigram) embeddings** — modded-nanogpt record #62
(the largest single 2026 record, −5.1 s of 104 s), Parameter Golf's frontier core, DeepSeek Engram
— which add capacity at near-zero FLOPs and fit Trainium (gathers, no new matmuls); (b) a
**neural + count-based n-gram hybrid at evaluation** — the *only* technique with a robust
within-frontier effect in the Parameter Golf analysis (+0.080 bpb, 95% CI [0.06, 0.10]) — which is
a stateless causal function of the input tokens and can be assessed **offline with zero device
time** before any rules question is asked (see §6).

---

## 1. Measurement protocol (unchanged rules, plus fixes to known blind spots)

1. **Noise floor:** three same-day repeats of the baseline spanned 0.00032 bpb / ±1 step at 900s.
   Any single-run delta < 0.001 is noise. Any keep/discard with delta < 0.003 needs a paired
   repeat. Add a `--seed` CLI flag (currently `torch.manual_seed(42)` is hardcoded) so noise can be
   estimated cheaply; keep seed 42 as the default for comparability.
2. **Pairing:** run A and B back-to-back on the same day at the same commit; quote only the paired
   delta (box speed drifts a few % day to day — documented in the foreach section).
3. **Escalation ladder:** 300s smoke (correctness only) → 900s proxy (direction) → 1800s (decision).
   Capacity-affecting changes (depth, width, embeddings, batch schedule) **must** be decided at
   1800s: DEPTH=5 won at 900s and lost at 1800s.
4. **Thermal cliff:** the cliff has not been seen since the tensor-lr fix (dt 0.90 through step
   1990), but keep an eye on the tail `dt` in every log and note idle gaps between runs.
5. **Config drift guard:** always read `grad_accum_steps × micro_tokens`, `num_params_M`,
   `seq_len`, `world_size`, `lnc` from the run header, never from the label (the 262144 mislabel).
6. **Log hygiene:** pull `research/experiment-log.jsonl` and `research/run-logs/*.log` after
   every run; record `git_commit` from the *local* repo in `--description`.
7. **Two calibration facts to establish in the first hour (they change how every number is read):**
   - **C1 — official eval context length.** `prepare.py eval-public` defaults to
     `--seq-len 1024 --batch-size 1 --eval-tokens 2097152`, while `train.py` evaluates at
     SEQ_LEN=2048. The official runner's `evaluate_bpb(...)` arguments are not published. Run
     `python prepare.py eval-public --train-py train.py --checkpoint <best> --seq-len 1024` and
     `--seq-len 2048` (and `--eval-tokens 0` for the full shard) on the best checkpoint and record
     the gap. **Official rules page (devpost, fetched 2026-09-09): scoring is "val_bpb on a pinned
     validation shard, computed over ~20M tokens"** — ten times the local default of 2M
     (`--eval-tokens 2097152`); run local confirmations with `--eval-tokens 20971520` so the
     noise characteristics match the leaderboard. The context length is still not stated anywhere
     public (all 17 forum threads were swept with `tvly extract`; none mention it). If the 1024 number is materially worse, (i) ask on the devpost forum which seq_len
     the scorer uses, (ii) compare the local-vs-leaderboard delta of the two past submissions
     (SEQ_LEN=1024 submission: local 1.15361 → board 1.1394; SEQ_LEN=2048 submission: local ? →
     board ?) — if the 2048 submission's board score sits *above* its local score while the 1024
     one sat below, the scorer is almost certainly evaluating at 1024. Also make `load_for_eval`
     robust to `t > config.sequence_len` (extend the rotary tables to e.g. 8192 at eval) so a
     longer scorer context can never crash the run.
   - **C2 — data volume.** Count tokens in the 5 train shards once (tokenize on the box, ~minutes):
     `epoch` never left 1 at 261.5M tokens; any throughput win ≥ 1.5× will cross into epoch 2, which
     is fine (≤ 4 epochs is known to be near-free) but must be known, and affects §6's n-gram counts.

---

## 2. Tier 0 — cheap, high-confidence gap closers (first 2–3 days of box time)

Ordered by expected value per hour of device time. Each item states what to change, why, the
protocol, and the decision rule.

### T0.1 Learned local conv (the dead weight)
- **Change:** add `block.local_conv.weight` to an optimizer group. Preferred: its own AdamW group
  (`lr=MATRIX_LR`, `betas=(0.9,0.95)`, `wd=0.0`) like `ssm_params`; alternative: Muon over the
  `[C, k]` matrix is meaningless (k=4), so AdamW. Keep zero init (identity at step 0).
- **Why:** the conv is already computed every step (5.4% of the compiled window — and removing it
  made the compiled graph 6.9× slower, so it stays anyway). Training it is free capacity; the
  RWKV/Mamba/Hyena/"smear" literature and modded-nanogpt record #34 all show a 1-token-lookback
  mechanism is worth ~0.3–0.7% loss in small models.
- **Protocol:** 900s paired A/B vs baseline; verify `num_params` unchanged and that the optimizer
  now covers 62/62 params (adapt `research/scratch/local_conv_deadparam_check.py`). Also try
  `LOCAL_CONV_KERNEL ∈ {2, 4, 8}` (shape change → new compile, fine).
- **Decision:** keep if ≥ 0.002 better at 900s, confirm at 1800s.

### T0.2 Decouple depth from width
- **Change:** add `N_EMBD` (explicit override) so `build_config` uses it when set. Sweep
  `(DEPTH, n_embd) ∈ {(7,640), (8,640), (6,768) [known: 1623/1.04415], (7,512)}`. Also
  `HEAD_DIM=64` at (6,640) → 10 heads (never tested; more heads usually help small models; check
  the rope+norm NKI kernel still runs — it is per-head on HEAD_DIM and was written for 128, may
  need a 64-partition variant or fallback).
- **Why:** width 768@6 was within noise of 640@6 at full budget while costing 19% of steps; a
  7th layer at 640 costs ~14% steps and adds depth where the model is shallow. `Time is Not
  Compute` predicts N* grows with throughput — if Tier 1 lands, re-run this sweep.
- **Protocol:** 900s for direction, **1800s for the decision** (capacity axis).

### T0.3 MLP ratio and attention scale
- **Change:** `MLP_RATIO ∈ {3, 4, 6}` (hidden = ratio×n_embd; keep multiples of 128); SDPA
  `scale=` sweep `{1/√128=0.088, 0.10, 0.12, 0.15}` with QK-norm (modded-nanogpt uses 0.1 and
  raises it with YaRN; Parameter Golf frontier used "QK-Gain 5–5.25" i.e. a learned per-head gain
  after QK-norm). Implement as a per-head learnable gain multiplying q after `rope_norm` (zero
  extra launches inside the compiled graph) or as a fixed `scale=` kwarg.
- **Decision:** 900s proxy, confirm winner at 1800s only if ≥ 0.003.

### T0.4 Schedule / optimizer resweep at the 2000-step regime (one axis at a time, 900s paired)
- `MATRIX_LR ∈ {0.02, 0.025, 0.03, 0.035}` (varunneal: optimal Muon LR ≈ 0.035→0.015 as token
  budget grows 134M→1B; we are at 262M; Polar Express raises the optimal LR).
- `FINAL_LR_FRAC ∈ {0.05, 0.10, 0.15}` (record #72: raising the floor 0.1→0.15 was a record).
- `WARMDOWN_RATIO ∈ {0.6, 0.75, 0.9}` (records use 0.55–0.6 at ~1300–1700 steps).
- `WEIGHT_DECAY ∈ {0.05, 0.1, 0.2}` and `WD_DECAY_TO_ZERO` on/off — cautious WD is already in.
- `EMBEDDING_LR ∈ {0.2, 0.3, 0.45}`, `UNEMBEDDING_LR ∈ {0.004, 0.006, 0.009}`.
- `INIT_SCALE ∈ {0.5, 0.68, 0.85}`; **nonzero `c_proj` init** (record #64 "tuned nonzero attn
  V/O init": zero-init output projections delay learning in short runs): try
  `uniform(-0.1·s, 0.1·s)` and `0.25·s` for `attn.c_proj` (keep `mlp.c_proj` zero first).
- `LOGIT_SOFTCAP ∈ {15, 30}` and the asymmetric rescale `23·sigmoid((z+5)/7.5)` (record #54,
  Parameter Golf "AsymLogit" +0.155 all-data). The softcap+CE NKI kernel fuses `tanh`; the
  sigmoid form needs a small kernel change (`nisa.activation(sigmoid)` exists) — do it only if the
  eager A/B (with `--no-nki-softcap-ce` on both arms) shows ≥ 0.003.
- Muon momentum **cooldown** 0.95→0.85 over the last 50 steps (modded-nanogpt `get_muon_momentum`).
- `MUON ns_steps ∈ {4, 5}` (5 is default; 4 saves 12 matmuls/step, quality check only).
- **Decision:** coordinate descent; lock each axis at 900s only if ≥ 0.002 and paired; then one
  1800s confirmation of the whole new hyperparameter set.

### T0.5 Value embeddings coverage
- **Change:** `has_ve` → all 6 layers (adds 3 tables × 5.24M params, near-zero FLOPs); and the
  modded-nanogpt layout (skip layer 0, cover layers 1,2 and the last 3). Also VE gate input
  channels `VE_GATE_CHANNELS ∈ {12, 32}` and `VE_WD ∈ {0, 0.003}` (Karpathy's tiny-WD sequence).
- **Why:** Karpathy's own ablation found full-coverage VE optimal, yet the port went alternating
  "to match Karpathy" (contradiction noted in `enhancement-opportunities.md` #4 vs. Solution 7).
- **Cost check:** `embedding_dense_backward` was 6.4% of the eager window; measure the compiled
  step time with 6 tables before judging quality.

### T0.6 Compiler flags *under compile* and LNC re-check
- `USE_TRN2_COMPILER_FLAGS` (`--model-type=transformer --distribution-strategy=llm-training
  --lnc=<n>`) was only A/B'd in **eager**. Under `torch.compile` the whole graph is compiled and
  `--model-type=transformer` / `--optlevel 3` can matter far more. Sweep `NEURON_CC_FLAGS`:
  `--optlevel {1,2,3}` × `--model-type {generic,transformer}` (both documented, both present in
  `neuronx-cc compile --help` on the box); also `--enable-fast-loading-neuron-binaries`
  (load-time only). Judge on steady-state `dt` at 100 fixed steps, then 900s.
- `NEURON_LOGICAL_NC_CONFIG=1 / --nproc_per_node 8` was chosen in the eager era (330 vs 265
  steps). Re-run one 900s at LNC=1 with the current code; the compiled graph and per-rank
  micro-batch interact with the 2-core sharding differently.

### T0.7 Eval-side insurance (zero training cost)
- Robust `load_for_eval` for any `seq_len` (see C1). Save both raw and EMA weights (§4.2) in
  `final.pt`; `load_for_eval` picks whichever the local eval showed better.

---

## 3. Tier 1 — Throughput: from 8% MFU toward 15–20%

The dispatch-gap model of this backend is now predictive (0.145 ms per avoided launch; ~1.5% of
step time per 13 ms saved). Attack in this order; each item is independent.

### T1.1 Re-profile the *current* step, end to end, and split host vs device
Before changing anything, produce one table for the locked config at the real shape (4 ranks,
LNC=2, compile on): `forward NEFF`, `backward NEFF`, `grad clip`, `all_reduce (62 calls)`,
`optimizer (Muon + AdamW)`, `weight-cache refresh (62 casts)`, `dataloader fetch()`, `control
all_reduce + .item()`, `synchronize` waits. Use `research/scratch/phase_timer.py` +
`neuron-nki-profile-analysis-agent`. The profile in the notebook is at nproc=1 and predates the
tensor-lr fix; the host half of the step has *never* been decomposed.
- **Direct test for dataloader exposure:** run 100 steps with `fetch()` replaced by a cached
  batch (synthetic constant `x,y`). `dt_synthetic` vs `dt_real` = exposed host data cost. The
  box has **12 vCPUs** for 4 ranks × (Python dispatch + `tokenizer_threads=4` + best-fit packing
  loop + runtime threads); contention is plausible and would also explain why LNC=2 (4
  processes) beat LNC=1 (8 processes). Watch `top`/`mpstat` during a run.
- If exposed cost ≥ 1%: **background prefetch thread** (double-buffered pinned CPU tensors, queue
  depth 2–4; tiktoken releases the GIL, the packing loop does not — measure). Also try
  `tokenizer_threads=2`. Rules-safe: pure `train.py` dataloader wrapper (forum 45042).

### T1.2 Gradient all-reduce bucketing (62 → 1–2 launches/step)
`sync_gradients` issues one `dist.all_reduce` per parameter (62). Allocate one flat fp32 buffer,
point every `p.grad` at a view of it before the first backward (autograd accumulates in place;
use `zero_grad(set_to_none=False)`), and all-reduce the flat buffer once. Expected ≈ 60 fewer
launches ≈ 9 ms ≈ 1%. Optional: bf16 all-reduce (halves 223 MB of traffic; quality check needed).
Verify bit-equality of the averaged grads vs the per-tensor path.

### T1.3 `clip_grad_norm_` launch count
On the `neuron` device `torch.nn.utils.clip_grad_norm_` may fall back to per-tensor norms (62
norm + 62 mul launches). Test `foreach=True` explicitly (`torch._foreach_norm` exists and eager
foreach works on this stack per the notebook); or compute the global norm on the flat buffer from
T1.2 (one norm, one mul). Expected 1–2%.

### T1.4 Adam every other step (record #39)
Update the AdamW groups (wte, lm_head, VE tables, scalars) only on odd steps, letting their grads
accumulate (do **not** zero them on even steps). Muon params update every step. Halves AdamW cost
(~15 ms → ~8 ms) and halves those tables' all-reduce volume. Quality-neutral in modded-nanogpt
(they retuned Adam β1 0.8→0.7 alongside). If the flat-buffer from T1.2 is used, keep Adam grads
in a separate buffer that is not zeroed on even steps.

### T1.5 Micro-batch size anomaly (potentially the biggest lever)
`PACK_FACTOR=4` (micro `[4,2048]`) ran **22% fewer steps** than `[2,2048]`; that contradicts
every expectation (fewer launches, larger matmuls). Resolve it:
1. Microbenchmark the compiled fwd+bwd of the real model at micro shapes `[1,2048], [2,2048],
   [4,2048], [8,2048]` with synthetic data on 1 rank and on 4 ranks; report ms/token.
2. Also `DEVICE_BATCH_SIZE=2, PACK_FACTOR=2` (same model shape as PACK_FACTOR=4, different
   loader) to separate loader from device effects.
3. If ms/token *rises* with batch: inspect the NEFF (`neuron-profile`) for SBUF spills
   (`spill_save_bytes`) and matmul instruction shapes; try `--optlevel 3`; try `-O1`; try
   eager for that shape. If ms/token *falls* with batch (as it should), then the PACK_FACTOR=4
   loss came from the loader or memory, and `grad_accum 8 → 4 or 2` is a 10–25% step-time win.
4. Recall `TOTAL_BATCH_SIZE` must stay divisible by `DEVICE_BATCH_SIZE×PACK_FACTOR×SEQ_LEN×world`.

### T1.6 Remove the transposes: store weights transposed
The compiled graph carries +75% transpose instructions vs eager and `F.linear` is 48% of the
window. `F.linear` computes `x @ W.T`; TensorE wants the contraction dim on partitions. Test a
`Linear` variant that stores `weight_t = [in, out]` and computes `x @ weight_t` (modded-nanogpt
record #27 "transposed MLP matrix" is the same idea on GPU). Zero quality effect; measure
`dt` at fixed steps and `matmul instruction count/utilization` in the profile. Keep `state_dict`
compatibility via `load_for_eval` transposing back if needed.

### T1.7 Compile the optimizer step (blocked before; unblock it)
`torch.compile(optimizer.step, backend="neuron")` failed on `_foreach_copy_` (no StableHLO
lowering) and would have recompiled on every Python-float change. Both are fixable: (a) write a
loop-form Muon/AdamW without any `_foreach_*` for the compiled variant; (b) carry `lr`,
`momentum`, `weight_decay`, `bias1/bias2` as device tensors (the `_device_scalar` pattern) so the
graph has no per-step constants. Prize: the optimizer's remaining ~370 launches (~54 ms, 6%)
collapse to a handful. Risk: medium (compile of a graph with 62 parameters of 4 shapes). Try
`torch.compile(fullgraph=False)`, and fall back to compiling only the AdamW part if Muon's
Newton–Schulz misbehaves.

### T1.8 Make the two NKI kernels traceable (`@nki_op` + `register_autograd`)
Under compile the rope+QK-norm and softcap+CE kernels are ~6% "islands" (100% exposed). The
TorchNeuron-native overview documents `@nki_op` with `torch.library.register_autograd` so NKI
kernels embed in the compiled graph as custom ops. A prototype exists
(`research/scratch/nki_custom_ops_train_py.patch`) and was judged "no win" **for the Mamba2
graph**; it was not evaluated as the transformer's default. Re-test on the transformer: expected
up to 6% of window if the compiler overlaps them.

### T1.9 Fewer host syncs per step
Two `synchronize(device)` calls, one `.item()` for the loss and one control all-reduce per step
each drain the queue. Keep one sync (needed for `record_step_boundary` timing) and read the loss
every 10 steps for logging (the NaN guard can use the control tensor every step without `.item()`
on the loss by computing `isnan` on device). Expected 0.5–1%.

### T1.10 Weight-cast cache
`refresh_weight_cache` casts 62 weights every step (62 launches). Fold into one `_foreach_copy_`
into a preallocated bf16 buffer set, or drop the cache and let the compiled graph fuse the cast
(measure; the cache was a no-op win in eager).

### T1.11 (Only after T1.1–T1.6) FP8 via NKI `double_row`
Trainium2 TensorE runs fp8 at 2× bf16 throughput only through `nki.isa.nc_matmul(...,
perf_mode=double_row)` (both operands fp8 e4m3/e5m2, contraction split across partition×2). The
torch path (`_scaled_mm`) is dead (136521× slower, notebook). A hand-written fp8 MLP kernel is a
large project and only pays once the graph is TensorE-bound (it is not today). Defer unless T1.5/6
push TensorE utilization above ~60% of the step.

---

## 4. Tier 2 — Training dynamics (cheap to implement, no recompiles)

### T2.1 Batch-size ramp (record #46; "Fast Catch-Up, Late Switching"; Seesaw)
- **Mechanism:** keep the micro shape fixed (no recompile) and vary `grad_accum_steps` with
  elapsed-time fraction: e.g. stage A `65536` (accum 4) for the first 35% of the charged budget,
  stage B `131072` (accum 8) to 75%, stage C `262144` (accum 16) for the cooldown tail.
  Scale Muon LR by ≈ √(B/B_ref) (modded-nanogpt used (16/8)^0.6 and (24/8)^0.5): with the
  0.025 tuned at 131072, use ≈ 0.018 / 0.025 / 0.035; Adam groups scale the same way.
- **Why it should beat every fixed batch:** we measured 65536 → 3544 steps/1.0488, 131072 → 2000/
  1.0423, 262144 → 1038/1.0510. Small batches are more token-efficient early (common patterns),
  large batches waste tokens early but are harmless late; the theory paper shows the loss
  "catches up" quickly after a late switch. Kimi K2 / GLM-4.5 do this at scale.
- **Protocol:** implement `BATCH_SCHEDULE = [(0.0, 65536, lr_mul_a), (0.35, 131072, 1.0),
  (0.75, 262144, lr_mul_c)]` keyed on the same `elapsed/max_train_seconds` progress the LR
  schedule uses; log the stage. Sweep the two boundaries `{(0.25,0.7),(0.35,0.75),(0.5,0.85)}` and
  the LR exponent `{0.5, 0.6}`. Decide at 1800s only (it changes how many steps the run gets).
- **Interaction:** `WARMDOWN_RATIO` and `MUON_MOMENTUM_WARMUP_STEPS` are step/time based; keep them.
  A **sequence-length curriculum** (1024 → 2048 at 1/3) is the companion trick (record #72) but
  needs a second compiled graph (~45–60 s charged unless compiled inside the excluded first 5
  steps) — try only after T2.1 works, and pre-compile the second shape during steps 0–4.

### T2.2 EMA / weight averaging of the final weights
- Maintain an EMA of all parameters (`torch._foreach_lerp_` on a shadow copy every step, or every
  k=4 steps with decay^k; ~1–3 launches). Decay 0.995–0.999 (Parameter Golf frontier used 0.9965
  at ~5000 steps; ablation there: 0.998 over-smoothed, 0.996 ≈ neutral). Save both raw and EMA in
  the checkpoint; evaluate both locally; `load_for_eval` returns the better one (fixed at
  submission time, not chosen at eval).
- Expected small (LR already decays to 5%), but free, and EMA was in 767 Parameter Golf
  submissions (+0.123 all-data) and modded-nanogpt Track-2 record #13.

### T2.3 Cautious weight decay on Adam groups (record #50) and higher WD on embeddings
`mask = update * p > 0` before applying WD in `_adamw_step_foreach`; modded uses `wd_mul 150`
on embed/lm_head and `5` on VE with a tiny base. Sweep `lm_head wd ∈ {0.01, 0.05, 0.1}`,
`wte wd ∈ {0.001, 0.01}`.

### T2.4 Smoothed scalars (record #52)
Give `scalar_params` (resid/x0 lambdas, gates) their own Adam betas `(0.9, 0.99)` and a lower
LR than `SCALAR_LR=0.2` (`{0.05, 0.1}`); the notebook already saw a NaN when a matrix landed in
this group, so the group's LR is aggressive. Cheap 900s A/B.

---

## 5. Tier 3 — Architecture additions that suit Trainium (gathers and elementwise, no new big matmuls)

All must be **static-shape**, add few launches inside the compiled graph, and be zero-init
no-ops at step 0 (the notebook's discipline). Test each alone at 900s paired, then stack.

### T3.1 Bigram hash embedding (+ sign trick) — top pick
- **modded-nanogpt #62/#68/#83 recipe (verified in `train_gpt.py`):**
  `h_t = ((36313·x_t) XOR (27191·x_{t-1})) mod (R−1)`, position 0 → reserved index `R−1`;
  `R = 7.5×vocab` there (use `R ∈ {8×8192=65536, 131072}` here; tokens 8192² = 67M pairs so
  collisions are frequent but "meaningful collisions are rare due to token distribution");
  table `E_b ∈ R^{R×n_embd}`, **zero-init**; sign trick: `s_t = sign_table[(x_t XOR x_{t-1}) mod 8192]`
  with `sign_table = randn(8192, n_embd).sign()` (bf16 buffer), `b_t = E_b[h_t] · s_t`.
  Inject into the residual stream at the input (`x = x0 + λ_0·b`) and at every block input
  (`x = x + λ_i·b`, `λ` learned scalars init 0.1; the record found "very simple direct addition
  to the residual stream outperformed" gated/trigram/multi-hash variants). Optimizer: AdamW like
  the embeddings (`lr = EMBEDDING_LR·dmodel_scale`, betas `(0.8, 0.95)`, small WD).
- **Trainium notes:** compute `h_t` and the sign index on the **host** inside `fetch()` (numpy
  int64 ops on the `[2,2048]` int32 batch; adds one small H2D copy) unless int bitwise ops prove
  fast on device under compile. Two extra `nn.Embedding` gathers per micro-step; dense scatter
  backward on `R×640` (65536×640×4 B = 168 MB zero+scatter per micro, ~0.1 ms of HBM time) and
  one more tensor in the all-reduce (T1.2 makes that one launch). Params +42M (tie-break only).
- **Expected:** record #62 removed 165 of 1765 steps (~9% of compute) at equal loss ⇒ ≈ 0.006–
  0.01 bpb here; Parameter Golf frontier adopted it in 635 submissions.
- **Variant to test second:** trigram hash `((a·x_t) XOR (b·x_{t-1}) XOR (c·x_{t-2})) mod R'`
  concatenated (Engram uses n=2,3; 4-grams were "slightly suboptimal").

### T3.2 Engram-lite conditional memory (the "architecture" version of n-grams)
DeepSeek's recipe, scaled down: K hash heads per order (2 and 3), tables of prime size, dim
`d_mem` (e.g. 256 per head), concatenated `e_t`; `k_t = W_K e_t, v_t = W_V e_t`,
`α_t = σ(RMSNorm(h_t)·RMSNorm(k_t)/√d)`, `ṽ_t = α_t v_t`, then `Y = SiLU(Conv1D_dw(RMSNorm(ṽ))) + ṽ`
(conv zero-init), `H ← H + Y` **at layer 2 of 6** (their ablation: layer 2 best of 12, two
modules at 2 and 6 slightly better). Their allocation law puts ~20–25% of "sparse" parameters
in memory; for us that means tables of ~15–30M params. Tables get 5× embedding LR, no WD.
Costs two small matmuls (`e_t→k,v`) — fine. Do this only if T3.1 shows a clear win (it is the
gated, context-aware superset of T3.1).

### T3.3 Smear (record #34)
`x = cat([x[:1], x[1:] + λ·σ(g(x[1:, :12]))·x[:-1]])` on the raw token embeddings before the
first norm; `g: Linear(12,1)` zero-init, `λ` scalar init 0. ~5 elementwise launches, ~0.3–0.5%
loss. Redundant with a *trained* local conv (T0.1) — test T0.1 first, then smear on top.

### T3.4 Partial key offset (record #49, "enables 1-layer induction")
On one or two layers: `k[:, 1:, :, 64:] = k[:, :-1, :, 64:]` (shift the non-rotating / second
half of the key head-dim by one position). One slice-copy per layer; causal by construction.
With the rope+norm NKI kernel, apply the shift **after** `rope_norm(k)` on the stationary half.

### T3.5 Skip / backout (records #11, #40, #45, #77)
For DEPTH=6: cache the residual after layer 1; add `σ(λ)·cache` into layer 4's input (λ init
−1.5 ⇒ 0.18); "backout": `x_final = x − β·x_after_layer_3` before the final norm (β init 0.5,
learned). A few elementwise ops. MUDD (the heavy dynamic version) already lost here; this is the
2-scalar static version. Also try the "single saved activation": layers 4–5 attention reads
`norm(cache_after_layer_3)` as its input (attention only, MLP unchanged).

### T3.6 Sparse attention gate / attention output gate (record #28; Parameter Golf +0.159, present
in "all four lowest verified" PG scores)
Per-head gate on the attention output from the first 12 residual channels:
`y = y · σ(Linear(12, n_head)(x[..., :12]))` (bias-init so σ≈0.5–0.75). Lets a head no-op by
context and removes the BOS-as-sink dependency. One small matmul + one multiply per layer.

### T3.7 Multi-token prediction with annealed weights (record #53) and prefix-token aux loss (#88)
- MTP without extra parameters: loss = `Σ_k w_k · CE(logits_t, y_{t+k})` with `w = [1, 0.5, 0.25→0]`
  in the first third of the budget, `[1, 0.5→0]` in the second, `[1]` after. Extra cost: one extra
  softcap-CE kernel call per extra offset (the kernel already takes arbitrary targets; shift `y`).
  Record #53 saved 80 steps of ~1900 (~4% compute-equivalent).
- Prefix-token loss: a static table `prefix[T] = id of the longest proper byte-prefix of token T
  that is itself a token, else −1`, derived from the fixed tokenizer at startup (reading the
  tokenizer is allowed; no `prepare.py` changes), weight 0.25 annealed to 0 by 2/3 of the run.
  Needs `ignore_index` support in the CE kernel (or mask multiply). Record #88 ≈ 1%.

### T3.8 Attention-free first layer / MLP-free layer (records #30, #35)
Dropping the first attention layer (or first MLP) saved ~1% time at equal loss in a 11-layer
model; at DEPTH=6 the relative cost is larger, so test only combined with DEPTH=7@640 (T0.2).

### T3.9 Tied embeddings early, untie at 2/3 (records #51/#53)
Karpathy's autoresearch found tying catastrophic (val 3.2), likely because of the 75× embedding
LR; modded-nanogpt makes it work by copying `lm_head.T` into `wte` at init and untying at 2/3.
Low priority; try last.

---

## 6. Tier 4 — Stand-out bets

### T4.1 Neural + count-based n-gram hybrid at evaluation (assess offline first, zero device time)
> **2026-09-12 update — see `research/ngram-drafter-and-count-mixer-2026-09-12.md`.** The Parameter Golf
> evidence cited below is eval-time caches that update on the scored text (illegal here) and the PG n-gram
> track was invalidated on 2026-03-27 for a normalisation bug (#978: a properly normalised n-gram alone is
> 1.51 bpb). Keep the item, but base it on kNN-LM / Infini-gram and on this repo's T3.1 measurement
> (learned bigram tables: −0.020 to −0.024 bpb per step, lost only to device cost). Design and offline
> protocol are in the new document; rules question still open.

- **Evidence:** Parameter Golf's within-frontier analysis: N-gram Backoff is the *only* technique
  with a robust positive effect (+0.080 bpb, 95% CI [0.06, 0.10], p<0.0001) among 429 frontier
  submissions; PPM byte mixture +0.03. Those are persistent n-gram models blended with the neural
  distribution, `P = λ·P_neural + (1−λ)·P_ngram`, λ context-dependent. kNN-LM/Infini-gram show the
  same for undertrained neural LMs — and a 55M model after one pass over 260M tokens is exactly
  that. (Note the *within-sequence* causal "token tilt" variant gave only −0.00014 in PG once made
  fully causal — do **not** bother with within-row caches; attention already copies.)
- **Rules reading (state it explicitly in the forum question):** the tables are produced inside
  the 30-minute training loop from the train shards (allowed: "assets produced within your own
  training loop on the provided data"; architecture is fully open), the forward is a pure,
  stateless, causal function of the input tokens with no collectives, and `evaluate_bpb`
  recomputes CE from our logits, so we output `log(mix)` (softmax-invariant). The tie-break on
  parameter count is the only cost. **Ask lping2024 on devpost before spending a submission on
  it**; phrase it as a hybrid neural/n-gram language model, cite kNN-LM/Engram.
- **Offline feasibility (do this first, on the box's CPU while the device runs other A/Bs):**
  1. Dump per-token neural log-probs (`[N, 8192]` fp16, or just the log-prob of the target plus
     top-64) for the 2M public-val tokens from the best checkpoint (`prepare.py`'s eval loader,
     `seq_len 2048`).
  2. Tokenize the 5 train shards once (tiktoken, 12 vCPUs); build hashed count tables for orders
     2–5 (numpy: `np.unique`/`bincount` on 64-bit hashes of contexts, keep top-K=16 continuations
     per bucket with counts; buckets 2^22–2^24). Implement stupid backoff and interpolated
     Kneser–Ney.
  3. Fit λ: global, per-order, and a small logistic model of `(max order matched, count, neural
     entropy)` → λ. Report bpb of `mix` vs neural alone on the same 2M tokens. Also report the
     *train-shard-held-out* version (build on shards 0–3, evaluate on shard 4 tokens through the
     model) to make sure nothing about the public val shard is being fit.
  4. **Go/no-go:** ≥ 0.005 bpb gain → build the device path; < 0.003 → drop.
- **Device implementation sketch (only after go):** tables as buffers in the module
  (`ctx_hash → [K] token ids (int32), [K] log-probs (fp16)`, plus per-bucket backoff mass); in
  `forward` compute context hashes for orders 2–5 with int32 multiply/XOR/mod on device (smoke-test
  op support under eager on `neuron`; eval runs eager), gather `[T, K]`, `scatter_add` into a
  `[T, 8192]` zero tensor, mix with `softmax(logits)` in fp32, return `log`. Eval cost is a few
  ms per row; 20-minute eval timeout is not a constraint. Counting during training: accumulate
  bigram/trigram counts on the **host** from the same `fetch()` batches (numpy `bincount` on the
  int32 rows; ~1 ms/row), all-gather/merge counts across the 4 ranks once at the end (or let rank 0
  count everything from a host-side pass in a background thread), save into `final.pt`.
- **Why nobody else will do it:** it is unusual for a "train a transformer" contest and requires
  the offline harness; if the organizers rule it out, nothing is lost but CPU time.

### T4.2 Hashed n-gram capacity inside the model (T3.1/T3.2) — the fully rules-safe cousin
Even if T4.1 is disallowed, hashed bigram/trigram embeddings are plain model parameters trained
by gradient descent (Engram, modded-nanogpt #62). Prioritize T3.1 above every other Tier-3 item.

### T4.3 Use of the excluded startup window (ask before relying on it)
The runner excludes launch→5th step (≤ 900 s). Legitimate uses: compile *all* graphs the run will
need (a second sequence-length shape for T2.1's curriculum; the eval shape), warm the kernel
cache, and — arguably — pre-tokenize/pre-pack the data or build T4.1's count tables. Anything
that looks like *training* in that window (giant early steps) would be read as exploitation of
the evaluation infrastructure and must not be done. Ask on the forum whether data preprocessing
in the startup window is acceptable; do not assume.

### T4.4 What not to spend time on (closed with real hardware evidence in the notebook)
Mamba2/SSD hybrids (best M=1 still +0.013), RWKV x051a (2.05× slower and worse per step), SLAY/
YAT (harmonic-budget argument), MUDD and DCMHA at DEPTH=6, XSA, value-residual (as implemented),
FP8 via torch (`_scaled_mm`), stochastic rounding (fails the causality tolerance), fused QKV
(disables the rope kernel), sliding-window attention (both sub-components lost), doc-boundary
masking, requeue packer, PACK_FACTOR 4 (until T1.5 explains it), `--nki-relu2`.

---

## 7. Sequencing for the remaining ~3 weeks (5 submissions/week)

**Week 1 (Sep 9–15): calibration + Tier 0 + Tier 1 quick wins.**
Day 1: C1/C2 calibration, `--seed`, robust `load_for_eval`, T0.1 (trained local conv), T1.1
profile + synthetic-batch test, T1.2/T1.3 (bucketing, clip). Days 2–4: T0.4 coordinate descent
(one axis per 900s pair), T0.6 flags under compile, T1.5 micro-batch anomaly, T1.4, T1.9/T1.10.
Days 5–7: T0.2/T0.3 (depth@640, HEAD_DIM 64, MLP ratio, attn scale) with 1800s confirmations;
**submit** the best 1800s-confirmed stack by Sep 14 (submission #1 of the week; keep 2 in reserve).
In parallel on CPU: T4.1 offline harness; post the T4.1 and T4.3 forum questions on day 1.

**Week 2 (Sep 16–22): Tier 2 + Tier 3.**
T2.1 batch ramp (decide at 1800s), T2.2 EMA, T2.3/T2.4; then T3.1 bigram hash (then trigram),
T3.4 key offset, T3.6 attention gate, T3.5 skip/backout, T3.7 MTP/prefix; T1.6/T1.7/T1.8 if
Week 1 showed TensorE/launch headroom. Re-run the depth/width point (T0.2) once throughput has
changed by > 15% (the optimum moves with throughput). Submit twice.

**Week 3 (Sep 23–30): consolidation.**
Freeze the stack by Sep 26; run 3 seeds at 1800s; evaluate at seq_len 1024 and 2048; final
submissions with the most robust config (not the single luckiest run). Keep one submission for a
last-day fallback identical to the previous best in case the platform rejects a new kernel.

**Submission checklist (every time):** `python prepare.py check --train-py train.py
--launch-command "..."`; `causality_passed: True` with `max_abs 0.0`; no `[SHAPE-CRITICAL]`
warnings; `startup_allowance_est` < 900; `over_budget: False`; `eval-public` at both 1024 and
2048 recorded; launch command uses only `NEURON_LOGICAL_NC_CONFIG=<1|2> torchrun --standalone
--nproc_per_node=<8|4> train.py` (no shell tokens); no imports beyond the environment.

---

## 8. Expected-value summary (honest ranges; the exchange rate is 0.077·ln(steps))

| Item | Kind | Cost to try | Plausible gain (bpb) | Confidence |
|---|---|---|---|---|
| T1.5 micro-batch fix (if per-token time drops with batch) | speed | 1 day | 0.01–0.03 | medium |
| T0.4 LR/schedule resweep at 2000 steps | quality | 2 days | 0.005–0.015 | high |
| T2.1 batch-size ramp | quality | 1 day | 0.005–0.012 | medium-high |
| T3.1 bigram hash embedding | capacity | 1 day | 0.005–0.012 | medium-high |
| T1.2+T1.3+T1.4+T1.9+T1.10 launch trimming | speed | 1 day | 0.004–0.008 | high |
| T0.1 trained local conv | capacity | 2 hours | 0.002–0.006 | medium |
| T0.2 depth 7–8 @ 640 (after speed wins) | capacity | 1 day | 0–0.008 | medium |
| T0.6 compiler flags under compile | speed | 3 hours | 0–0.01 | low-medium |
| T1.7 compiled optimizer | speed | 2 days | 0.003–0.006 | low-medium |
| T1.6 transposed weights | speed | 4 hours | 0–0.01 | low-medium |
| T0.5 VE on all layers | capacity | 3 hours | 0–0.005 | medium |
| T3.4/T3.5/T3.6 key offset, skip/backout, attn gate | quality | 2 hours each | 0.001–0.004 each | medium |
| T3.7 MTP / prefix aux loss | quality | 1 day | 0.002–0.006 | medium |
| T2.2 EMA | quality | 2 hours | 0–0.003 | high (small) |
| T4.1 neural+n-gram hybrid at eval | stand-out | 1 day CPU + 1 day device | 0.005–0.03 | medium, rules risk |

If half of the high/medium-high items land, the stack reaches ≈ 1.00–1.01; with T1.5 or T4.1
paying out it reaches the 0.97–0.99 range.

---

## 9. Implementation notes and traps (collected from the notebook, keep them in view)

- Any per-step Python float that reaches a device op as `alpha=`/constant recompiles the graph
  (the 2.1× tensor-lr finding). New scalars in T2.1 (lr multipliers), T2.2 (EMA decay), T3.7 (MTP
  weights) **must** be device tensors updated with `copy_`.
- `_foreach_*` is eager-only on this stack; don't put it inside anything you want to
  `torch.compile`.
- New `torch.autograd.Function` NKI kernels become compile islands; prefer `@nki_op` (T1.8).
- `causality_check` tolerance is 1e-2 and is a false-positive trap for anything nondeterministic
  in forward (stochastic rounding). Bigram/trigram hashes and n-gram lookups are deterministic —
  but verify `causality_max_abs == 0.0` after every new module.
- Shape-critical toggles change `state_dict` keys: register them in `SHAPE_CRITICAL_TOGGLES` and
  make `load_for_eval` construct the *same* architecture (read toggles from the checkpoint payload,
  as already done).
- Keep `USE_LOCAL_CONV=True` even if T0.1 shows no quality gain: removing the conv made the
  compiled graph 6.9× slower (unexplained compiler behaviour) — if T1.x ever re-profiles this,
  the mystery is worth 5.4% of the window.
- The box's git history differs from the local repo; provenance is `label + launch_command +
  notebook`, so write the notebook entry before starting the next run.
- Tavily's API key in this environment is invalid (401) and the `tvly` CLI install was blocked;
  `gh api` (authenticated) reaches the upstream `aws-neuron/trainium-frontier` repo and public
  repos, and arXiv's export API works with `curl -L`. Use those for further web checks.

## 10. Reference links

- Competition upstream: `gh api repos/aws-neuron/trainium-frontier` (PR #6 = compile-time exclusion; README/prepare.py identical to local as of 2026-09-09). Forum: https://trainium-frontier.devpost.com/forum_topics (45042 custom packing ruling; 45077 compile exclusion).
- modded-nanogpt records: https://github.com/KellerJordan/modded-nanogpt (records/track_1_short/2026-01-19_BigramHashEmbedding, 2025-11-29_BatchSizeSchedule, 2025-09-18_Smear, 2025-09-30_CustomBatching, 2025-10-04_Backout, 2025-12-22_MultiTokenPrediction, 2025-12-26_LogitRescale, 2026-02-10_ShortWindow, 2025-08-23_SparseAttnGate). Batch schedule essay: https://varunneal.github.io/essays/muon
- Parameter Golf: https://github.com/openai/parameter-golf ; analysis paper arXiv 2607.01517 and https://github.com/PMP56/pmgolf-analysis (results/SUMMARY.md).
- Engram: arXiv 2601.07372v2; follow-ups 2606.08347 (Tensorized Engram), 2605.24869 (Lngram).
- Batch-size scheduling theory: arXiv 2602.14208 (Fast Catch-Up, Late Switching), arXiv 2510.14717 (Seesaw). Wall-clock scaling: arXiv 2603.28823.
- Neuron: TorchNeuron-native overview (adaptive eager fusion, `@nki_op`, DDP support) https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html ; `nki.isa.nc_matmul` (fp8 `double_row` = 2× throughput, fp8 both operands) ; `neuronx-cc` CLI reference (`--optlevel {1,2,3}` default 2, `--model-type {generic,transformer}`, `--logical-nc-config`, `--enable-mixed-precision-accumulation` default on) ; Trainium2 arch page (667 TFLOP/s bf16, 1299 fp8, 2.9 TB/s HBM, 28 MiB SBUF/core). trn2.3xlarge: 12 vCPU, 128 GB RAM, 1 chip.

---

## 11. Addendum (same day, after the Tavily key was fixed) — new evidence that re-ranks the plan

The failed searches were rerun. Most results were confirmatory, but one find is of a different
class from everything else in this document: **the actual training scripts of the two best known
solutions to the nanochat autoresearch task** (same lineage as our baseline, same ClimbMix data,
same val_bpb metric; 5 minutes on one GPU instead of 30 minutes on one Trn2 chip):

| Solution (10-seed mean val_bpb, Recursive's harness, B200, 5 min) | val_bpb |
|---|---|
| Karpathy seed `train.py` | 0.9751 |
| Karpathy's overnight-tuned solution (commit 438a26e, the recipe our Solutions 1–10 mined) | 0.9646 |
| autoresearch@home community best | 0.9372 |
| Recursive `optimized_from_vanilla.py` (from a plain transformer) | 0.9344 |
| **Recursive `optimized_from_karpathy.py`** (code public) | **0.9109** |
| Aster (code not public; 3 changes described) | 0.9098 |

Source: `github.com/recursive-org/first-steps-toward-automated-ai-research/nanochat_autoresearch/`
(`solutions/optimized_from_karpathy.py`, 1267 lines, Apache-2.0; copied to `/tmp/research/` on
this box — re-download if missing) and https://www.asterlab.ai/research/announcing_preliminary_results_from_our_autonomous_research_system .
Caveats before transferring anything: their tokenizer is 32,768-vocab (ours 8,192), model is
DEPTH=8 × 768-wide with 6 heads, batch 147k tokens, ~1,000–1,500 steps, FlashAttention-4 with
sliding windows, everything in bf16, B200. Their per-seed spread is ±0.005; ours is ±0.0003.

### 11.1 What both SOTA solutions credit most: hashed n-gram tables (confirms and upgrades T3.1/T3.2/T4)

**Recursive (0.9109) — hashed bigram/trigram *value* embeddings, gated per head:**
- On every VE layer (alternating layers, same `has_ve` rule as ours): two half-`kv_dim` tables
  per order, K=2 independent hashes concatenated; bigram on all VE layers, trigram on the first
  and last two VE layers. Hash: `((prev*p1) ^ (cur*p2)) % table_size`, trigram
  `((prev2*a) ^ (prev*b) ^ (cur*c)) % size`, with **different prime pairs per layer** so
  collisions decorrelate across layers. `table_size = 64 × vocab` (they went as far as
  ~2M rows/table; "1–2B sparse parameters on a ~50M model").
- Injection exactly where our `USE_VE` already injects: `v = v + 2σ(gate_b(x[..., 32:64])) · bigram_ve
  + 2σ(gate_t(x[..., 64:96])) · trigram_ve`, gates zero-init (neutral = 1.0), each gate reading a
  **disjoint** 32-channel slice of the residual (unigram VE reads 0:32).
- Tables stored **bf16**, optimized with **RMSProp (no first moment)**, β2 0.999 → 0.9999 in the
  last 30% of warmdown, **no LR warmdown at all for the tables** ("sparse tables benefit from
  full-rate training"), LR = embedding LR (0.6 × dmodel-scale).
- Plus: per-head **RMSNorm on the attention output** followed by a per-head gate
  `y = norm(y) · 2σ(head_gate(x[..., :32]))` on all layers (their "head-level MoE", = the sparse
  attention gate that tops Parameter Golf's all-data table and is in all four lowest PG scores);
  **sandwich norm on the MLP** (`x = x + norm(mlp(norm(x)))`); **shifted ReLU²**
  `relu(h − 0.5)²` ("tau=0.5 confirmed optimal"); input-dependent x0 gate
  `2σ(s_i · mean(x))`; zero-init **pooling of the last three layers' outputs** into the final
  residual; softcap `16.5·tanh(z/15)`; rotary base 1e6; `x0_lambdas` init 0.1; `wte` init std 1.0;
  Adam **Demon** (β1 0.8 → 0.55 across its warmdown), Muon momentum quadratic warmdown
  0.95 → 0.79, NorMuon β2 ramp 0.95 → 0.97; **WARMDOWN_RATIO 0.95** (decay starts at 5% of the
  budget!) with FINAL_LR_FRAC 0.05 and a separate Adam warmdown 0.65; MATRIX_LR 0.04,
  EMBEDDING_LR 0.6, UNEMBEDDING_LR 0.004; WD 0.1 decaying to zero (their WD "pulses" look like
  overfit noise — ignore). They explicitly **removed MTP** ("hurts step count in 5-min budget").
- Their from-vanilla run adds: **causal token shift on Q/K inputs** (`x + β·x_prev`, per-dim β
  init 0.3; V unshifted), an EMA/SWA of weights started at 50% of the budget with decay 0.98
  per step and evaluated with the averaged weights, per-depth LR/momentum scaling
  (LR ×1.15 at the bottom → ×0.85 at the top, momentum 0.90 → 0.97), a batch-size increase
  131072 → 262144 partway (our T2.1), byte-feature embedding init, and K=3 input-token-routed
  low-rank output heads.

**Aster (0.9098) — three described changes:** (1) hashed bigram/trigram/fourgram *input*
embeddings added to the residual; (2) an **output bigram logit table** `nn.Embedding(V, V)`
indexed by the current token, added directly to the next-token logits, gated per token by
`σ(Linear(n_embd→1))` with bias −2, zero-init (exact no-op at init), stored bf16 and added before
the float upcast; (3) "free output n-gram backoff": the same hashed n-gram tensors re-injected as
a per-token-gated residual on the final hidden state before `lm_head` (`2σ(gate(x))`, zero-init
`out_lambdas`).

**Karpathy's nanochat discussion #472** notes bigram hash embeddings stopped helping at d25
scale and Engram-layer variants were "okay but not great" at d12; Recursive/Aster show they are
decisive at our scale. Both facts are consistent with the U-shaped allocation law in the Engram
paper: memory helps most when the backbone is small and undertrained — exactly our regime.

### 11.2 Revised priorities (supersedes §8 ordering for Tier 3/4)

1. **T3.1/T3.2 → implement as n-gram VALUE embeddings first** (Recursive's variant), because the
   code path exists (`USE_VE`, `ve_gate`, `has_ve`): add `bigram_ve`/`trigram_ve` tables and two
   extra gates per VE layer reading channels 32:64 and 64:96. Sizes for vocab 8,192 with
   `half_kv = 320`: `8×V = 65,536` rows → 21M params/table (126M for 2 tables × 3 layers, 0.25 GB
   bf16); `16×V` → 252M; `64×V` → 1.0B (2 GB bf16, plus RMSProp state). Start at 8×V, escalate.
   Hash on the **host** in `fetch()` (int64 numpy) unless int ops compile cleanly on device.
   Optimizer: RMSProp (β2 0.999, eps 1e-10, no WD), **no LR decay for the tables**, bf16 storage
   (their choice; try fp32 first if bf16 RMSProp misbehaves). Tie-break cost: parameters. Budget
   the extra all-reduce (tables are dense-gradient; consider Adam-every-other-step for them).
2. **Residual-side bigram hash (modded-nanogpt #62)** as the A/B alternative, then **Aster's output
   bigram logit table** (`V×V` bf16 = 67M params, one gather + one add per micro-step, zero-init):
   this is the **rules-safe, gradient-trained version of T4.1** and should be tried *before* any
   count-based n-gram mixing. If it wins, the count-based T4.1 becomes optional.
3. **Cheap one-liners from Recursive to A/B in the T0.4 sweep:** `relu(h−0.5)²`; sandwich norm on
   the MLP; per-head output norm + head gate (T3.6); last-3-layer output pooling (T3.5 variant);
   softcap 16.5/15; `x0_lambda` init 0.1; rotary base 1e6; causal Q/K token shift
   (`x + β·x_prev`, β init 0.3 — complements the trained local conv, T0.1); Demon β1 for Adam
   groups; Muon momentum warmdown; NorMuon β2 ramp 0.95→0.97; **WARMDOWN_RATIO up to 0.9–0.95**
   (three independent groups — Recursive, modded-nanogpt's 0.6, Karpathy's 0.75 — disagree, so
   sweep {0.6, 0.75, 0.9, 0.95} rather than trust any one).
4. **EMA with decay 0.98/step starting at 50%** (Recursive) is a concrete setting for T2.2.
5. **Batch ramp 131072 → 262144** (Recursive from-vanilla) is a concrete setting for T2.1.
6. **MTP (T3.7) demoted**: Recursive removed it for step-count reasons; keep it last.
7. Everything in Tier 1 (throughput) stands unchanged; Recursive runs *everything* in bf16
   (params included) — add "bf16 master weights for matrix params (with mantissa tracking)" as a
   T1 candidate; it halves optimizer traffic and all-reduce volume.

### 11.3 Other confirmations from the rerun searches
- Parameter Golf frontier (16 MB, 10 min): SmearGate, BigramHash, XSA, EMA, sliding-window eval
  in >50% of frontier stacks; N-gram backoff the only robust within-frontier win; SSMs and
  universal transformers negative. Consistent with our own closed lines.
- SkyPilot's autoresearch run (1.003 → 0.974): width AR=96 the biggest gain, `muon_beta2`
  0.95 → 0.98 worth ~0.001, `FINAL_LR_FRAC=0.05` over 0.03, deeper models lost on the 5-min budget.
- Horizon-free schedules + weight averaging (Kempner/arXiv 2602.03702) match, not beat, tuned
  cosine — EMA is a small, safe gain here, not a large one.
- Engram (arXiv 2601.07372v2): optimum ~20–25% of "sparse" parameters in memory; layer-2 injection
  best; tokenizer canonicalization and context-aware gating matter more than the conv.
- TorchNeuron-native docs: "Adaptive Eager Execution" fuses queued op sequences heuristically and
  is disabled by `torch.use_deterministic_algorithms()`; `@nki_op` + `register_autograd` is the
  documented way to make NKI kernels traceable under `torch.compile` (T1.8).
- The two Tavily deep-research reports produced nothing beyond the above (generic guidance; one
  wrongly claimed `--model-type` does not exist — it is in the CLI reference and in `--help` on the box).

### 11.4 Tavily CLI sweep (installed by the user after the addendum above)
- `tvly map` + `tvly extract` over all 17 devpost forum threads: no ruling on the scorer's
  sequence length or on n-gram/eval-time components; organizer confirmations only that custom
  packing in `train.py` is admissible, compile time is excluded, SageMaker is not allowed, the
  AMIs 0731/0824 are identical, and submissions go through the GitHub leaderboard. One
  participant's platform failure surfaced as `USER_TRAINING_CRASHED (exit=1, status=no_run_meta)`
  with no `train.log` — keep the first 5 steps cheap and deterministic so the runner always gets
  its metadata.
- Rules page: eval is ~20M tokens (see C1). Updates page is empty.
- The autoresearch@home community-best page is JavaScript-only (unextractable); Aster's per-step
  annotations live in an SVG chart (unextractable). The Neuron docs subtree map surfaced no
  native-backend performance guide beyond the overview already read.
- For the loop: use `tvly extract <urls> --json` for full-text page grabs, `tvly map/crawl` for
  sweeping a site, and the REST search helper (`/tmp/research/tsearch.py`) for queries.

---

## 12. STATUS as of 2026-09-10 ~07:30 UTC — end of the Days 2–4 block (read this first)

### 12.1 Current best config and the exact submission launch command

Everything below is a **hardcoded default in `train.py` at commit `bf1a3ae`** — no flags are
needed, which matters because the submission launch command is fixed:

```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py
```

Best measured full-budget result at that command: **2205 steps / `public_val_bpb` 1.034005**
(`t04-stack-fullbudget`, paired repeat 2199 / 1.034101). Previous locked best was
2100 / 1.038386, so the Days 2–4 block is worth **−0.004381 bpb**. (Reminder, already
established, do not re-derive: the standard 2M-token quick eval reads ~0.006 bpb optimistic
versus the 20M/full-shard eval; it does not affect paired A/B direction.)

What changed in `train.py` during Days 2–4, all of it default-on now:

| change | item | measured effect |
|---|---|---|
| `TRAIN_LOCAL_CONV = True`, `LOCAL_CONV_KERNEL = 2` | T0.1 | already locked Day 1; defaults flip verified a no-op |
| `PACK_FACTOR = 4` | T1.5 | 900s paired −0.001734, +2.6% steps |
| `NEURON_CC_FLAGS=--optlevel=1` (via `os.environ.setdefault`) | T0.6 | 900s paired −0.001399, +1.8% steps |
| prebuilt `cached_linears` list (no per-step `model.modules()` walk) | T1.10 | free, ~0 measurable |
| `MATRIX_LR 0.020`, `WEIGHT_DECAY 0.20`, `INIT_SCALE 0.50`, `ATTN_CPROJ_INIT_SCALE 0.10`, `MUON_MOMENTUM_COOLDOWN_FRAC 0.045` | T0.4 stack | 1800s paired −0.001392, non-overlapping |

New flags added for reproducibility (all default to the locked values, so they change nothing
unless passed): `--pack-factor`, `--grad-clip`, tri-state `--clip-foreach`, `--adam-every-n`,
`--adam-every-n-lr-comp`, and the full T0.4 knob set (`--matrix-lr`, `--embedding-lr`,
`--unembedding-lr`, `--scalar-lr`, `--weight-decay`, `--wd-decay-to-zero`, `--final-lr-frac`,
`--warmdown-ratio`, `--init-scale`, `--attn-cproj-init`, `--logit-softcap`, `--muon-ns-steps`,
`--muon-momentum-cooldown-frac`, `--muon-momentum-final`).

### 12.2 Days 2–4 list: done vs pending

**Done and closed (full write-ups in `research/trn2-experiment-results.md`, Day 2/Day 3 sections):**

1. **T0.1 defaults flip** — done, verified a no-op versus the flagged run.
2. **T0.4 resweep** — done. 27 runs. Six axes confirmed optimal (`FINAL_LR_FRAC` 0.05,
   `UNEMBEDDING_LR` 0.006, `LOGIT_SOFTCAP` 15.0, `MUON_NS_STEPS` 5, `EMBEDDING_LR` 0.30,
   `WARMDOWN_RATIO` 0.75); five sub-noise axes adopted as a combined stack confirmed at 1800s.
3. **T0.6 compiler flags + LNC re-check** — done. `--optlevel=1` adopted; `--optlevel={2,3}` and
   `--model-type={generic,transformer}` are all no-ops or worse. LNC=1/8-proc does not compile
   under the current code (SDK `l2_norm_kernel_lnc0` bug) and, forced to run, is a **+0.24 bpb**
   quality cliff at identical throughput — mechanism still unexplained (clip placement was tested
   and **refuted**).
4. **T1.5 PACK_FACTOR anomaly** — resolved. The old "PF=4 is 22% slower" row was a stale-commit
   artifact (its logged launch command has no `--pack-factor`). Synthetic microbenchmark says PF=4
   is 2.9% cheaper per token and PF=8 is past the optimum; PF=4 adopted.
5. **T1.4 Adam-every-other-step** — implemented, correct (causality clean, deferred all-reduce
   proven exact), +1.1% throughput, +0.000887 bpb → **discarded**. The lr-compensated variant is
   +0.028 bpb → discarded decisively.
6. **T1.9 / T1.10 host syncs and weight-cast cache** — measured, both are ~0.2% of step time;
   only the free half (hoisting the `model.modules()` walk) was taken. Nothing else shipped.

**Pending / open, in the order I would pick them up:**

- **The LNC=1 0.24 bpb cliff is unexplained.** Ruled out: throughput, `GRAD_CLIP` in either
  placement, the `--clip-foreach` flag. Prime suspects, in order: LNC=1 variants of the NKI
  kernels (`USE_NKI_ROPE_NORM`, `USE_NKI_SOFTCAP_CE`, `USE_NKI_MUON`) being numerically worse or
  buggy (the compile crash already proves those paths are separately generated and less exercised);
  something world-size-dependent in the data pipeline at 8 ranks; Muon's per-rank Newton–Schulz
  with a halved per-rank batch. Cheap first probe: LNC=1 with all three NKI toggles off.
- **`GRAD_CLIP` is now nearly free and nearly pointless** (`--grad-clip 0` at LNC=2 costs only
  +0.000857 at 900s, inside noise, and buys +7.5 steps). Not changed, but the historical "real win"
  no longer reproduces at this scale — worth one 1800s pair before anyone relies on it.
- **T0.2 / T0.3 (depth/width, MLP ratio)** — untouched by design, they are the next block.
- **Tier 2 / Tier 3 / Tier 4** (batch-size ramp, EMA, hashed n-gram tables, etc.) — untouched.
- Not attempted this block, deliberately: T1.6 (transposed weights), T1.7 (compile the optimizer
  step), T1.8 (`@nki_op` traceability), T1.11 (FP8 via NKI).

### 12.3 Traps re-confirmed this block (keep these in view)

- **Any new per-step Python float that reaches a device op recompiles the whole graph every step
  during cooldown.** The Muon momentum cooldown adopted above is a device tensor updated in place
  for exactly this reason; the evidence that it worked is that step count did not move.
- **`torch._foreach_*` is eager-only on this stack** — never inside anything `torch.compile`d.
- **Read `grad_accum × micro_tokens`, `num_params_M`, `seq_len`, `world_size`, `lnc` out of each
  run's own logged header**, never from the label. The T1.5 anomaly was a stale-commit artifact
  that only became visible by reading the *logged launch command* of the old row.
- **Verify edited `train.py` with `compile(src, 'train.py', 'exec')`, not `ast.parse`** —
  `ast.parse` does not catch symtable errors like "name used prior to global declaration".
- **The box's `.git` is stale and disposable.** `research/run_experiment.py` runs `git rev-parse`
  on the box, so its `git_commit` / `git_dirty` fields in `experiment-log.jsonl` are meaningless;
  trust `md5sum train.py` against local HEAD instead.
- **Sub-noise single-axis wins do not necessarily add.** `WEIGHT_DECAY 0.20` (−0.00055) plus
  `WD_DECAY_TO_ZERO` off (−0.00058) measured **+0.00183** together. Always test the stack.

### 12.4 OPEN AND URGENT (2026-09-10 ~09:40 UTC): the leaderboard scored 1.1495, we measure 1.034

> **RESOLVED (2026-09-10 evening, local-box research pass, no hardware): see
> `research/submission-gap-analysis-2026-09-10.md`.** The submitted file still has
> `NUM_STEPS = 1000` (`train.py:191`) and `--num-steps` defaults to it; the bare submission
> command passes no flags, so the official run stopped at step 1000 (~815 s) with the
> wall-clock LR schedule at `lrm ~0.74` and no cooldown. Every local full-budget run,
> including the cold-cache test below, was launched via `run_experiment.py` with
> `--num-steps 5000 --max-train-seconds 1800`, so "the exact submitted command" was never
> actually run. Expected score of a 1000-step no-cooldown run: ~1.12-1.16. The four
> hypotheses below are not needed; the optlevel/pack-factor hedge submissions would not
> fix it. Fix: make the time budget the only binding control, then resubmit once.


The submission of this exact `train.py` with the bare launch command scored **1.1495**. Four
independent full-1800s runs here give **1.034005 / 1.034101 / 1.034118 / 1.034459**. Gap ~0.115 bpb
— about 10x the known local-vs-board calibration bias, so it is a real behavioural difference on
the official runner, not an eval artifact.

**Ruled out (tested, see the notebook's "cold-compile-cache hypothesis is REFUTED" section):**
a cold NEFF/compile cache. With `/tmp/neuron_backend` moved aside and a fresh empty
`NEURON_COMPILE_CACHE_DIR`, the exact submitted command produced **2198 steps / 1.034459 /
`startup_allowance_est` 57.7s** — cold compile costs ~30s here, and the run is indistinguishable
from warm. So `train.py`'s 900s startup-allowance back-dating never fires and never cut the budget.

**Key quantitative clue:** at this notebook's ≈0.077·ln(steps) exchange rate, 1.1495 corresponds to
roughly **490–660 charged steps**, not 2200 — and that lands exactly on the historical
recompile-every-step cluster (639/1.15361, 847/1.108697, 427/1.180633). The official run almost
certainly trained ~4x fewer steps, rather than trained the same steps to a worse loss.

**Untested hypotheses, best first:**

1. **Per-step recompiles on the official SDK/compiler version** (the `USE_TENSOR_LR_SCALARS`
   failure mode returning because a different backend version handles one of the per-step scalars
   differently). Cheap hardening independent of diagnosis: quantise the lr multiplier so only a
   bounded number of distinct values ever exist.
2. **`NEURON_CC_FLAGS=--optlevel=1`** (set via `os.environ.setdefault` in `train.py`, so it applies
   on the runner too) behaving differently under a different `neuronx-cc`. A no-`--optlevel`
   fallback submission is a one-line, zero-risk hedge.
3. **`PACK_FACTOR = 4`**, the other Day-2 change, on a more contended official host (its win here
   was partly loader-side).
4. **A runner-side crash/retry** producing a truncated run (the forum documents
   `USER_TRAINING_CRASHED (exit=1, status=no_run_meta)` for others).

**Highest-value next action is not a local run: get the official `train.log` / `num_steps` for that
submission.** `num_steps` alone separates hypotheses 1–3 (throughput collapse) from 4 (truncated
budget). Until that gap is understood, do not spend budget on further hyperparameter tuning — a
0.115 environment gap dwarfs every win in §12.2 combined.
