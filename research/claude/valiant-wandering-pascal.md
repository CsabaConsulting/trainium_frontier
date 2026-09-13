# Trainium Frontier — active plan

## Status (2026-09-13)
- **Locked config**: T3.1 last-layer hashed n-gram value embeddings (`USE_NGRAM_VE=True`,
  `NGRAM_VE_LAYERS="-1"`, bigram+trigram, mult=64, dim=128; 190.1M params). Local: 1.018985 bpb /
  2139 steps at 1800s. Launch command (no flags needed, all defaults):
  ```
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --no-eval-public --num-steps 1000000
  ```
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

## Queued next, in order (real, not discarded — just sequenced behind the capacity work)
1. T1.6/T1.7/T1.8 throughput (transposed weights, compiled optimizer, `@nki_op` traceable kernels)
   — deep device-profiling knowledge from the T3.1 root-cause work makes these more tractable now.
2. N-gram VE cost reduction on layers 1/3 (currently cheap only on the last layer) — ~0.007
   headroom if the gather can be moved out of the compiled graph.
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
- `torch._foreach_*` is eager-only — never inside anything `torch.compile`d.
- **Dress rehearsal before every submission recommendation**: run the literal bare launch command
  once, not through `research/run_experiment.py` (which forces its own `--num-steps`).
- **tmux for any box command that outlives a possible SSH disconnect** (every 900s/1800s run):
  `tmux new-session -d -s <name> '... > <logfile> 2>&1; echo DONE >> <logfile>'`, poll with
  `tail <logfile>` over a fresh connection.
- Never touch global git config. Never force-push.
