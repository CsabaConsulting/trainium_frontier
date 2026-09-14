# Trainium Frontier — active plan

## Status (2026-09-14)
- **Locked config**: hashed n-gram value embeddings on **layers 1 AND the last layer**
  (`USE_NGRAM_VE=True`, `NGRAM_VE_LAYERS="1,-1"`, bigram+trigram, **mult=128**, dim=128) plus the two
  free throughput changes — T1.7 compiled Muon (`USE_COMPILE_MUON=True`) and the frozen Q/K-shift
  identity (`USE_QK_SHIFT=True`, `QK_SHIFT_BETA=0.0`, `QK_SHIFT_FREEZE=True`). 592.9M params.
  Local: **1.004363 bpb / 2135 steps at the full 1800s** (was 1.018985 / 2139 with the previous
  last-layer-only, mult-64, no-throughput-flags config: **−0.0146**). All of it is in `train.py`'s
  module-level defaults, so the launch command is still flagless:
  ```
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --no-eval-public --num-steps 1000000
  ```
  **Heads-up for anything else running experiments: those defaults CHANGED on 2026-09-14.** A bare
  `train.py` is no longer the 190.1M last-layer-only config, so any A/B that treats "no flags" as the
  old baseline must now pass `--ngram-ve-layers "-1" --ngram-ve-table-mult 64 --no-compile-muon
  --no-qk-shift` explicitly.
- **Real board score**: 1.0384 (sub-c23baa39, 9/12/26 1:05 PM) — predicted 1.039, off by 0.0006,
  tightest calibration match yet. Trust the calibration chain.
- **Leaderboard** (2026-09-12 snapshot): top 10 span 0.9683–1.0303; we'd sit just outside at
  1.0384. Live race, everyone still improving. Need multiple more real wins, not one silver
  bullet. Deadline 2026-09-30 (~17 days left), 5 submissions/week.
- **Eval-reading rule**: scorer very likely evaluates at seq_len=1024 (2M/1024 + ~0.006 window
  optimism matched the board score to 0.0009). Use `--eval-seq-len 1024` on any run whose number
  you want to compare to a real board score. This does NOT change what we train/submit —
  SEQ_LEN=2048 is strictly dominant at either grading length (T0.7 finding).
- **The data wall**: the corpus is 5 shards, one epoch ≈ 2195 steps ≈ 288M tokens, replayed in
  identical order forever (no reshuffling helps — tried, made it worse). Locked config runs 2139
  steps, 97.4% of one epoch. A throughput win is only convertible for its first ~2.6% of headroom;
  past that it's worth ~0 or negative (crosses into a repeated slice during LR cooldown). Every
  900s→1800s extrapolation is optimistic for throughput-positive changes as a result. Current
  strategy: spend verified-free speedups on **capacity per step held at a fixed step count**,
  not on letting steps grow.

## In progress
- Nothing running. Locked config unchanged (1.018985 bpb / 2139 steps at 1800s).

## Closed 2026-09-13 (evening): the capacity-per-step pivot — Engram-lite DISCARDED
Full write-up in `research/trn2-experiment-results.md` (T3.2 section). Implemented behind
`--engram`, default OFF; kept in the tree, `causality_max_abs == 0.0` on hardware, 8 CPU harness
arms green.

- **The trade closed arithmetically.** Frozen qk-shift re-confirmed at **+9.2%** at the locked
  config with the n-gram tables on (0.835 → 0.765 s/step). `engram + frozen` runs **2.4% FASTER
  than the locked default** while carrying **+35% parameters**. That is exactly the trade the pivot
  asked for.
- **And the capacity bought nothing.** Paired 900s, control repeated: best Engram config is
  **per-step +0.00086 bpb (neutral)**; the large-table one is **+0.0025 worse**. Its whole 900s
  advantage is a 4.8% speedup that the wall will not honour (2139 × 1.048 ≈ 2242 > 2195). No 1800s
  run spent.
- **The per-step penalty is capacity-INDEPENDENT** (+0.00254 vs +0.00255 across 8.4M vs 67.4M new
  parameters) — so it is not a warm-up effect, it is the mechanism.
- **Reusable: op count is the binding constraint, not FLOPs or parameters.** Share mode
  (+0.33M params, no gather, no all-reduce) still cost +23.4%. Buy capacity as *width on ops that
  already exist*, never as new mechanisms.
- **Reusable: the frozen qk-shift is superadditive with block-level work** — it saves 0.070 s/step
  on the base and **0.250 s/step** with Engram present. Attach it to any arm that adds block ops,
  and re-measure its budget with that arm rather than assuming it.
- **Reusable: T3.1's "only the last layer is affordable" rule does NOT generalize.** Mid-stack
  Engram cost 1 step out of 1129 (nothing) and was 0.0016 better. That rule is specific to
  injecting into attention's V inside a block. Measure the site; don't inherit it.

## 2026-09-14 — torch_xla graph mode: CLOSED NEGATIVE. T1.7 unblocked with a measured 8.18×
Probes 0–1 of `research/throughput-to-capacity-track-2026-09-13.md` are done (full detail in
`research/trn2-experiment-results.md`, dated section). `train.py` untouched; all work lives in
`research/scratch/xla_probe/`.
- **The "we run at 8% of peak so a 2–3× loop must exist" premise does not survive contact.** At the
  locked shape (depth 6, 640, seq 2048, dbs 4, 4 ranks, LNC=2) torch_xla graph mode is **0.997× at
  ga=4 (0.7975 vs 0.7952 s/step) and 1.114× at ga=1 (0.2283 vs 0.2544 s/step)**. Gate floor was 1.2×.
  **Nothing ported.** The 2.44× at the *upstream starter's* shape (dbs 1, seq 1024, ga 16: 0.6084 vs
  1.4816 s/step) is a small-micro-batch artefact — graph mode only wins while per-op work is too small
  to amortize dispatch, and our fat 8192-token-per-rank micro-step already amortizes it. **Rule: below
  ~8k tokens per micro-step, graph mode is worth revisiting; at or above it, it is not.**
- **XLA has a silent correctness bug we would have shipped into:** `index_add_` across a `mark_step()`
  boundary keeps only the **last** call, so gradient accumulation computes wrong embedding gradients
  (ours has three embedding-family tables). Exact without the per-micro `mark_step()`, but that single
  unrolled graph did not finish compiling in 17 minutes at a *toy* shape. Also: XLA compile cost
  235–638 s of the 900 s allowance versus native's 3.7 s warm.
- **T1.7 is unblocked and measured: `torch.compile(backend="neuron")` runs the Muon step 8.18× faster**
  (9.51 → 1.16 ms at `[6,640,640]` bf16) once `torch._foreach_copy_` is gone. With it present the
  neuron backend fails Torch-MLIR legalization — that op *is* the T1.7 blocker, now named. Sizing:
  native `0.2544 s` (ga=1) vs `0.7952 s` (ga=4) decomposes to one micro-step = 0.1803 s and
  **collective + optimizer = 0.0741 s = 9.3% of the step**, so ~6% throughput. Spend it as width at
  fixed steps, not as extra steps (2139 × 1.06 ≈ 2270 > 2195 wall).
- Two environment traps worth not re-paying: `islpy 2026.2.2` (allowed by neuronx-cc's `~=2026.1`)
  breaks **every** XLA compile with `[NCC_ISMP902] Simplifier error: is_subset()` — pin `islpy==2026.1`,
  and clear `/var/tmp/neuron-compile-cache` because the failure is cached. And torch_xla 2.11 defines
  but never calls `_aws_ec2_inf_trn_init()`, so 4-rank `torchrun` under PJRT NEURON must set
  `NEURON_RT_VISIBLE_CORES`/`NEURON_PJRT_*`/`NEURON_RT_ROOT_COMM_ID` itself or every rank requests all
  four logical cores and three of them die with `NRT_FAILURE`.
- Incidental, applies to the native path: the starter kit's `dt` halves at step 20 exactly when `lrm`
  hits 1.000 — **the per-step-Python-scalar recompile hazard is native-backend too**, worth ~2× for the
  first 20 steps. Our `USE_TENSOR_LR_SCALARS` discipline already covers it; do not relax it.
- Probes 2 and 3 of that track are **not started** (scope was 0–1).

## 2026-09-14 (afternoon) — T1.7 SHIPPED, and the budget bought n-gram VE on layer 1: **1.004363 (−0.0146)**
Full detail in the dated `research/trn2-experiment-results.md` section. New defaults are in `train.py`.
- **T1.7 `--compile-muon` KEEP: +3.38%**, not the ~6% the isolated 8.18× predicted — that estimate was
  priced against a non-NKI optimizer and the shipped path already had the NKI Muon fusion, which the
  compiled body *replaces* (a compiled region cannot hold an `nki.jit` call). The source rewrite is
  bit-exact on CPU; the *compiled* execution is not (3.25e-03 max|diff| over 3 steps, bf16-ULP scale in
  the Newton-Schulz chain, same order as the shipped rope+norm and softcap+CE kernels). Verified where
  it counts: the step-matched control with it ON reproduces 1.018985 to 3e-6.
- **Stacks slightly SUPER-additively with the frozen qk-shift: +13.67% combined** (13.19% predicted
  multiplicatively). Both now default ON.
- **The T3.1 "early n-gram layer costs +34%" penalty is a ONE-RANK ARTEFACT.** At 4 ranks layer 1 costs
  5.86%, layer 3 costs 3.04%. Generalise: any "this is unaffordable" verdict measured at 1 rank must be
  re-measured at 4 before it closes a door, because the cross-pcore reshard is much cheaper when the
  residual is already sharded four ways.
- **Layer 1 is the win: −0.0118 bpb at an exact step match**, −0.0130 including the leftover steps its
  budget buys. Layer 3 is *harmful* (worse than control on the proxy, twice) and `{1,3,5}` is worse than
  `{1,5}`. **The position space is closed**: `has_ve()` = `i % 2 == (n_layer-1) % 2`, so at DEPTH 6 only
  {1,3,5} exist and `"0,-1"`/`"2,-1"`/`"1,2,-1"` are silently degenerate no-ops (measured).
- **mult 64 → 128 is now worth −0.0016**, reversing the 2026-09-12 "capacity saturates at 64" verdict.
  Lesson: *a capacity axis measured as saturated at one injection site is not saturated at two.*
  dim 256 and dim256+mult128 stay discarded (dominated / +0.0177 step penalty).
- Dress-rehearsed with the literal bare command; 592.9M params, 2135 steps in the official 1800 s,
  60 steps below the data wall, `causality_max_abs 0.0`.

## Queued next, in order (real, not discarded — just sequenced behind the capacity work)
1. T1.6/T1.8 throughput (transposed weights, `@nki_op` traceable kernels). **T1.7 is DONE** (shipped
   2026-09-14, +3.38%). Any further win here should be spent the same way T1.7's was: on capacity at a
   fixed step count. Note there is now ~0 slack left — the locked config uses the full 1800 s and lands
   60 steps under the wall — so the *next* speedup has to be paired with a capacity spend to be worth
   anything at all.
2. ~~N-gram VE cost reduction on layers 1/3~~ — **partly obsolete**: layer 1 is affordable as-is at
   4 ranks and is now shipped; layer 3 is harmful on quality, so making it *cheaper* buys nothing. What
   remains is only "make layer 1 cheaper", worth ~5.9% of a step, and there is nothing left to spend it
   on that has been measured positive.
3. LNC=1 quality-cliff mystery (curiosity-tier, lowest priority of the three).

Note: any future throughput win from 1-3 should be spent the same way as the current pivot —
on capacity at fixed step count, not on more steps, until/unless the data-wall situation changes
(e.g. more train data ever becomes available, which it won't under the rules).

## Small/cheap, still unimplemented
- Byte-feature `wte` init (≤0.002 bpb, near-zero cost).
- Parameter-matched coalesced SwiGLU (~15-20% chance of a small win).
- Cautious weight decay on Adam groups / smoothed scalar-group betas (T2.3/T2.4).
- `lr_multiplier` defense-in-depth against step-capping (optional hardening on top of the
  NUM_STEPS fix; cheap, low priority since the actual cap is already gone).

## Blocked on user, do not act without explicit go-ahead
- Any Devpost forum post (external/visible action) — currently nothing queued to post; the one
  drafted question (frozen n-gram count-mixer rules check) is moot since that idea died.

## Scheduled
- Week 3 (Sep 23-30): freeze the stack, 3-seed confirm at 1800s, eval at seq_len 1024 and 2048,
  final submissions on the most robust config, one fallback slot held in reserve.

## Closed for good (real hardware/evidence verdicts — do not revisit without new evidence)
Mamba2/SSD, RWKV x051a, SLAY/YAT, MUDD, DCMHA, XSA, value-residual, FP8 (`_scaled_mm`),
`torch.library` custom-op registration, stochastic rounding, fused QKV, sliding-window attention,
doc-boundary masking, requeue packer, `--nki-relu2`, custom tokenizer (illegal), byte-level
models (BLT/MBLM/SpaceByte), ternary/BitNet, Hyena, KAN/KAT as FFN, diffusion LMs, MTP,
Mixture-of-Depths (parked, not viable while launch-bound) — T0.4 hyperparameter resweep, T0.6
compiler flags, T1.5 PACK_FACTOR anomaly (resolved, PF=4 real), T1.4 Adam-every-N, T1.9/T1.10
host-sync/weight-cache, T0.2 depth/width decoupling, T0.3 MLP-ratio/attn-scale, T0.7 eval-seq-len
robustness, T2.1 batch-size ramp, T2.2 EMA, T4.1 count-based n-gram output mixer, T4.2 Aster
output bigram table, T3.9 grab-bag (shifted ReLU², head-gate, out-pool — sandwich-norm/qk-shift
were real but only via throughput, now devalued by the data wall), x0-gate, parallel attn+MLP
block, partial RoPE (scoped, not worth the risk), `--epoch-shuffle` (made the data wall worse).
Full detail for all of the above: `research/trn2-experiment-results.md` (git-tracked, the real
source of truth) and git history.

## Discipline (standing rules, unchanged)
- Local repo is the only git source of truth; the Trn2 box is disposable — rsync out, run, pull
  `research/experiment-log.jsonl` + run-logs back, commit+push locally, every time.
- **One owner of `train.py` at a time.** No exceptions.
- Capacity-affecting changes (depth/width/batch/embeddings) decided at 1800s only.
- **Every 1800s result must now also be checked against the ~2195-step data wall** before trusting
  it, in addition to the noise-floor check.
- Any single-run delta <0.001 bpb is noise; <0.003 needs a paired same-day repeat.
- **Any 900s result must be decomposed into per-step vs throughput components** before trusting
  it (several false reads this week came from skipping this).
- New per-step Python scalars must be device tensors updated via `copy_` — never a bare Python
  float reaching a compiled op (the tensor-lr-scalars lesson).
- `torch._foreach_*` is eager-only — never inside anything `torch.compile`d. (Confirmed 2026-09-14:
  the neuron backend fails Torch-MLIR legalization on `_foreach_copy_`; `openxla` accepts it.)
- **Gradient accumulation + graph-mode boundaries is a correctness trap, not just a perf question**
  (2026-09-14: `index_add_` across `mark_step()` keeps only the last call). If any future work puts a
  graph boundary inside the accumulation loop on any backend, verify per-group grad L2 against the
  eager path before believing a loss curve.
- **Dress rehearsal before every submission recommendation**: run the literal bare launch command
  once, not through `research/run_experiment.py` (which forces its own `--num-steps`).
- **tmux for any box command that outlives a possible SSH disconnect** (every 900s/1800s run):
  `tmux new-session -d -s <name> '... > <logfile> 2>&1; echo DONE >> <logfile>'`, poll with
  `tail <logfile>` over a fresh connection.
- Never touch global git config. Never force-push.
