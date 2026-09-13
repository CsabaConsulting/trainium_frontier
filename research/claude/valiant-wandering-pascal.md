# Trainium Frontier — active plan (reprioritized per research/frontier-research-plan-2026-09-09.md)

## Merged 2026-09-12 (afternoon): GPU-closed-vs-Trn2 audit + byte-level/ternary/Hyena review
Two new research docs from the laptop session, both real, disciplined, evidence-based (not
hype-chasing) — `research/gpu-closed-techniques-trn2-revisit-2026-09-12.md` and
`research/byte-level-and-lowbit-training-review-2026-09-12.md`.

**Byte-level (BLT/MBLM), ternary (BitNet), and Hyena: all correctly ruled out, don't revisit.**
BLT crosses over BPE at ~150B bytes; we train on ~1.15GB, ~100x below that, and the scorer feeds
fixed 8192-BPE token ids anyway (tokenizer can't be modified) — byte models have no place to plug
in. Ternary gets zero speedup on Trn2 (TensorE has no ternary path, quantized weights are
multiplied as bf16 anyway) and adds elementwise work on already-loaded engines, plus is worse
quality at this scale regardless of hardware. Hyena needs `torch.fft`, unsupported on Neuron. One
transferable fragment: byte-feature `wte` init (legal, zero-cost, ≤0.002 bpb) — small, queue it.

**GPU-closed-item audit — real precedent for "GPU closed it, Trn2 might not": PACK_FACTOR,
model width, and bigram tables already flipped for cost-structure reasons, not quality-physics
reasons.** Confirms (again) that sliding-window attention and doc-boundary masking get *worse*,
not better, on Trn2 — any additive attention mask forfeits the causal-fastpath's +10% steps win,
so don't revisit either. Four ranked, still-untested candidates, small expected value each
(0.001-0.005) but real and cheap (900s pairs):
1. **Parallel attention+MLP block with a coalesced input projection** (GPT-J/PaLM style,
   `x + attn(n(x)) + mlp(n(x))`, one `640×4480` matmul for Q/K/V+fc) — the one genuinely
   Trn2-favored idea (removes a sequential dependency, folds into an existing launch).
2. Parameter-matched coalesced SwiGLU (hidden=1664, one matmul for gate+up).
3. Partial RoPE in the NKI kernel (rotate half the head dim, zero-pad frequencies) — kernel edit,
   needs its own equivalence harness.
4. Recursive's input-dependent x0 gate specifically (`2σ(s_i·mean(x))`) — NOT the same as the
   T3.9 grab-bag already tested (that covered shifted ReLU², sandwich-norm, qk-shift, head-gate,
   out-pool; x0 gate was never in that batch).

**Also confirms my running agent's own results, already merged**: Aster's output bigram table
(T4.2) discarded. T3.9 grab-bag: shifted ReLU² closed (+0.0029 worse), head-gate loses (+25%
step cost), out-pool neutral; **sandwich-norm (-0.0031) and qk-shift (-0.0061) both real**, but
~90% *throughput* wins from compiler scheduling, not per-step quality — stacked to -0.0067.
Queue candidates 1/3/4 above as the next architecture-track task once the current agent wraps up.

## MAJOR (2026-09-13): the data wall — throughput work is devalued, strategy needs to pivot
Real, well-diagnosed finding from the compiler-scheduling-anomaly harvest. The corpus is only 5
shards, replayed in **identical order every epoch, no reshuffling** (`prepare._document_batches`).
One epoch ≈ 2195 steps ≈ 288M tokens. The locked 1800s config runs 2139 steps — **97.4% of one
epoch**. Any throughput win that pushes past ~2195 steps crosses into epoch 2, and every arm that
crossed it landed 1.0207-1.0250 vs 1.0183-1.0192 below it, **including a mechanism-free bit-exact
identity arm** — so it's a pure data effect, not a mechanism cost. `--epoch-shuffle` was built and
tested as the fix and made it *worse* (+0.0025 vs the arm it was meant to rescue) — the corpus's
fixed replay order is already recency-maximizing (epoch 2 opens with the part of the corpus
forgotten longest), so shuffling in recently-seen row groups is actively worse. No reordering
variant is worth trying; this is now closed as a real, permanent constraint, not a bug to fix.

**Consequence, asymmetric and important**: a speedup is only convertible for its first ~2.6% of
headroom and worth ~0 or negative beyond that; a slowdown is still fully penalized at the normal
(steeper) below-the-wall rate. Every 900s→1800s extrapolation this whole competition has been
optimistic for throughput-positive changes (a 900s run only reaches half an epoch, can't see the
wall). **This devalues T1.6/T1.7/T1.8 throughput work, the n-gram-layers-1/3 cost reduction, and
the LNC=1 mystery** — all of which are throughput plays that would mostly just push further past
a wall that's already ~0-value past 2.6%. Not closing them outright (a large enough throughput
win could still matter if it also frees room to add real capacity without growing step count),
but they're no longer the obvious next move they looked like yesterday.

**Recommended pivot, from the agent, needs a go-ahead before starting**: stop buying throughput,
spend a verified-neutral speedup (the frozen qk-shift, +9-10% free, zero quality) on **capacity
per step held at fixed step count instead** — e.g. widen the MLP, add heads, or add another
n-gram layer at ~9% more FLOPs while deliberately holding step count near ~2150 (not the old
DEPTH=8/width-768 sweeps, which paid for capacity *with* steps and lost — this is different:
spend the already-banked free speedup on capacity, net step count unchanged). This is a real
strategic redirection, not a small item — decide before launching more architecture-track work.

**Also closed this batch**: x0 gate (per-step neutral, real cost, discarded — and falsifies the
"any block perturbation buys throughput" over-generalization from item 1, since it added ops and
got *slower*), parallel attn+MLP block (large real negative, +0.0147-0.0149 step-matched — the
coalesced-input-projection idea is a real, separate throughput lever worth ~0 under the wall but
noted as a keeper if the wall situation ever changes), partial RoPE (properly scoped, not
attempted — expected payoff 0.0004 bpb vs ~1 day of real kernel-engineering risk, correctly not
worth it).

## Leaderboard reality check (2026-09-12 ~13:00): calibration confirmed, real gap to close
Submission history (all real): 1.1811 → 1.1590 → 1.1394 → 1.1495 (NUM_STEPS bug) → 1.0548 (fix)
→ **1.0384** (T3.1 last-layer n-gram VE, sub-c23baa39, 9/12/26 1:05 PM). Predicted board was
~1.039 — **off by only 0.0006**, the tightest the calibration chain has ever landed. Trust it
going forward.

**Public leaderboard, same snapshot**: #1 0.9683, #2 0.9692, #3 0.9733, #4 0.9747, #5 0.9863,
#6 0.9898, #7 0.9943, #8 0.9945, #9 1.0069, #10 1.0303. We'd sit just outside the published top
10 at 1.0384. Everyone is actively improving (several updates same-day as this check) — this is
a live race, not a fixed target. Getting comfortably under 1.00 needs **multiple more real wins
stacked**, not one silver bullet; budget accordingly and keep pushing hard rather than declaring
victory on any single result. Deadline 2026-09-30, 18 days left.

**Priority order given this**, most to least promising:
1. Aster output bigram table + Recursive/Aster one-liners — **running now** (agent launched
   2026-09-12 ~13:10), independent of the submission, results pending.
2. Cheap n-gram on layers 1/3 too (not just last layer) — well-diagnosed ~0.007 headroom from
   the profiling work, concrete engineering path (gather out of the compiled graph / static
   one-hot / LNC=1), queue next after item 1 reports back.
3. T1.6/T1.7/T1.8 throughput work (transposed weights, compiled optimizer, `@nki_op` traceable
   kernels) — never seriously attempted, could be worth more than any single remaining
   architecture idea if real (throughput ≈ 0.03 bpb per 1.5x per the original research plan's
   own exchange-rate model), now more tractable given the deep device-profiling knowledge from
   the T3.1 root-cause work (cross-pcore DMA descriptor bottleneck under LNC=2).
4. T3.2 Engram-lite (the gated, context-aware superset of the n-gram VE idea) — revisit now that
   T3.1 proved the underlying mechanism works on this hardware; was previously deprioritized as
   "only if T3.1 shows a clear win," which it now has, decisively.
5. LNC=1 quality-cliff mystery — still open, still curiosity-tier, lower priority than the above.

## LIVE (2026-09-12): T3.1 reversed from discard to real win, new lead in flight
T3.1 (hashed bigram/trigram value embeddings) was closed as "exactly break-even" on 2026-09-11.
The profiling-focused follow-up found the real cause of its throughput cost: **it's a
layer-position effect, not the table/gather/modulus/gate.** A table on the *last* layer only
(`NGRAM_VE_LAYERS=-1`) makes the step *faster* than baseline; tables on early layers (1, 3) cost
+5 to +48ms. Paired 900s A/B, commits `1eede36`/`39c5e0a`:

| arm | bpb | steps | Δ |
|---|---|---|---|
| control | 1.081306 | 1094 | — |
| bigram, layer −1, 524288 rows×128 | 1.068101 | 1099 | **−0.0132** |
| bigram+trigram, layer −1, 262144 rows each | 1.066917 | 1084 | **−0.0144** |

Causality exact on all arms. This is a real win, bigger than the entire T0.4 stack, and it's now
the leading lever, not a closed item. A profiling-focused agent is currently running the 1800s
escalation + further overhead work — do not duplicate, check its status before touching T3.1.

**New complementary idea, from merged laptop research
(`research/ngram-drafter-and-count-mixer-2026-09-12.md`), queue after T3.1's 1800s result lands**:
a **frozen, host-counted n-gram output mixer** (bigram/trigram counted from train shards on CPU,
~0 device cost, stored in the checkpoint, blended into the output distribution in `forward`) —
different mechanism from T3.1 (output-side vs. attention-value-side), estimated +0.005-0.015 bpb,
falsifiable **entirely offline, zero device time** via the doc's §6 protocol (dump neural
logprobs, build count tables, fit a gate on held-out train-shard tokens, report bpb of mix vs.
neural alone — go/no-go at ≥0.005/<0.003) before touching `train.py` or spending any device time.
Decision rule from the doc: test on top of the *new* T3.1 checkpoint, not the old baseline — if
residual gain <0.003, drop; if ≥0.005, still worth pursuing (and worth the rules question below).
The doc also corrects the evidence base T4.1 leaned on: Parameter Golf's spectacular n-gram
numbers are contaminated (eval-time caches that update on scored text, illegal here, plus a real
normalization bug invalidating 33+ PRs) — the clean evidence is kNN-LM/Infini-gram (both real,
smaller, ~13-22% ppl gains) and this repo's own T3.1 numbers.

**Rules question drafted, not yet posted** (`research/ngram-drafter-and-count-mixer-2026-09-12.md`
§9) — asks whether a frozen train-shard-counted table is an acceptable "asset produced within
your own training loop," whether it counts toward the parameter tie-break, and whether host-side
counting in the excluded startup window is acceptable. **Do not post to the Devpost forum without
the user's explicit go-ahead** — standing rule, external/visible action.

**Also merged, no action needed**: diffusion LMs confirmed not applicable (unscoreable under the
single-causal-forward contract, and 14-16x compute penalty even if it were); MTP confirmed
neutral-to-negative at ≤3B per real evidence (Gloeckle et al., DeepSeek-V3 ablation), stays last
in Tier 3; a parallelism/precision stance doc consolidates existing decisions (data-parallel only,
bf16 not fp8) with no new action.

## RESOLVED (2026-09-10 evening): the 1.1495-vs-1.034 leaderboard gap
Root cause found and fixed, no hardware needed — `research/submission-gap-analysis-2026-09-10.md`
(full evidence chain) + commit `ba37f86`. The submitted `train.py` still had the starter kit's
`NUM_STEPS = 1000`; the bare official launch command passes no `--num-steps`, so the scored run
stopped at step 1000 (~815s) with the wall-clock LR schedule at `lrm ~0.74` and **no cooldown ever
ran**. Every local full-1800s run went through `research/run_experiment.py`, which always passes
`--num-steps 5000`, so the bare command was never actually exercised locally until this was found.
Expected score of a 1000-step, no-cooldown run: ~1.12-1.16 — matches the 1.1495 that came back.
The cold-cache/optlevel/pack-factor/LNC hypotheses chased on 2026-09-10 were real, careful, but
chasing a red herring; none of them was the cause and the optlevel=2/pack-factor=2 hedge would
not have fixed it. **Fix applied**: `NUM_STEPS = 1_000_000` (train.py:191, deliberately
unreachable — the wall-clock budget is now the only control that can stop a scored run), plus a
startup warning and a step-50 host-only projection warning if a passed `--num-steps` would bind
before the time budget. Verified: `python3 -m py_compile train.py` clean, pushed to `origin/main`.

**Action: resubmit this exact file now.** Submitting doesn't need our Trn2 box — only the file +
launch command go to the competition's own infrastructure. Expected board score ~1.03-1.06 (see
the gap-analysis doc §3.1 for why it's a range, not the exact local 1.034005 — 2M-token local eval
is optimistic vs. 20M/full-shard, and the real scorer's context length, 1024 vs 2048, is still not
confirmed). This is a strictly-dominant resubmission over 1.1495; do it before spending any more
slots on hypothesis-testing submissions.

**Still open, lower priority, no hardware needed to think about them:**
- Which historical submission actually scored 1.1394 is ambiguous (notebook says SEQ_LEN=1024/
  submission 2; a laptop session's timestamp math points at SEQ_LEN=2048/submission 3 instead) —
  resolving it would pin down the scorer's context length, worth ~0.016 bpb of eval-side work if
  it's 1024. Check the leaderboard's per-submission history if the UI ever exposes past scores.
- LNC=1's real, unexplained +0.24 bpb quality cliff (ruled out: throughput, GRAD_CLIP placement,
  `--clip-foreach`, NKI kernels). Not blocking — we submit at LNC=2, always have. Curiosity only.
- Recommendation from the gap-analysis doc not yet applied (optional, cheap): make `lr_multiplier`
  robust to being step-capped anyway (`progress = max(elapsed/max_train_seconds, (step+1)/num_steps)`)
  as defense-in-depth on top of the NUM_STEPS fix. Low priority now that the actual cap is gone.

**New standing rule — dress rehearsal before every future submission**: run the *exact* launch
string (no `run_experiment.py`, no extra flags) once at full budget before trusting a config
enough to submit it. Confirm the log's first line (`training target=`) and the final `num_steps`
match expectation. This gap existed for days because nobody did this once.

**New standing rule — screen/tmux for any run that must survive a disconnect.** The user will
periodically exit/resume Claude Code, and the Trn2 box itself gets reprovisioned/goes down
unpredictably (down right now, as of 2026-09-11 — do not attempt to reconnect until given a new
SSH address). Any command on the box that isn't already self-backgrounding and isn't guaranteed
to finish before a plausible disconnect (i.e. every 900s/1800s training run) must be launched
inside a named `tmux` session, not a bare `nohup ... &`:
```
ssh ... "tmux new-session -d -s <name> 'source ~/workspace/native_venv/bin/activate && cd ~/trainium_frontier && <full command> > <logfile> 2>&1'"
```
Reattach/poll with `tmux capture-pane -p -t <name>` or `tail -f <logfile>` over a fresh SSH
connection — the job survives the SSH session ending regardless of what killed it. Use one
session name per run, don't reuse names across runs (stale panes make it easy to misread which
run's output you're looking at).

## Status (superseded by the RESOLVED section above for anything about the 1.1495 submission)
Locked baseline, best confirmed full-1800s (commit `bf1a3ae`/`527e4fd`): **2205 steps /
1.034005 bpb** (paired repeat: 2199/1.034101). Every flag from Day 1 + T0.4's stack is now
a hardcoded default (`TRAIN_LOCAL_CONV`, `LOCAL_CONV_KERNEL=2`, `PACK_FACTOR=4`,
`NEURON_CC_FLAGS=--optlevel=1`, `MATRIX_LR=0.020`, `WEIGHT_DECAY=0.20`, `INIT_SCALE=0.50`,
`ATTN_CPROJ_INIT_SCALE=0.10`, `MUON_MOMENTUM_COOLDOWN_FRAC=0.045`), so the dev launch
command is now just the bare command. Last real submission: 1.1394 (board). Deadline
2026-09-30, 5 submissions/week, ~15 remain.

**Lean submission launch command (use this one for the real leaderboard submission, not
the bare dev command)**:
```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --no-eval-public
```
`--eval-public` defaults ON because our own experiment-log pipeline depends on it to
report `public_val_bpb` for every dev A/B run — do NOT flip that global default, it would
silently break our own measurement tooling. `--no-eval-public` is scoped to the
submission command only. Why it's safe: README confirms "a participant-reported loss is
never trusted" — the leaderboard independently reruns `evaluate_bpb()` against our saved
checkpoint on its own private shard, so our own post-training eval has zero effect on the
score either way; it only costs real wall-clock (causality_check + a 2M-token eager eval)
for a number nobody scores us on. Audited the rest of the CLI surface (all other
`BooleanOptionalAction` flags) — everything else is an architecture toggle already
correctly defaulted OFF, matching the locked config; `--eval-public` was the only
dev-only overhead running by default in the charged path.

**Read this before trusting any absolute bpb number, including the one above**: C1
calibration (Day 1) found the standard 2M-token quick eval (what every number in this
plan/notebook uses, including 1.038386 above) reads **~0.006 bpb optimistic** vs. a
20M-token or full-shard eval at the same seq_len — the public val shard's first ~2M
tokens are measurably easier than the rest. This is an absolute-number bias only; every
paired A/B *decision* this competition still holds (the bias cancels between arms of a
same-size comparison). Also confirmed: 20M tokens already matches the full shard
(Δ=0.0003) — safe as the standard size going forward — and seq_len matters far more than
token count (Δ=0.016 between 1024 and 2048 at fixed 20M, 2048 scoring better, weak
evidence the real scorer is closer to 2048 than to `prepare.py`'s CLI default of 1024).
**Superseded on the scorer-length question (2026-09-11)**: the evidence now points at **1024**
(board 1.0548 vs our 1.049682@1024 + ~0.006 window optimism ≈ 1.0557, a 0.0009 match; the
2048 reading of 1.034328 is 0.021 away and cannot be reconciled). The Δ is +0.0154-0.0162
across five checkpoints. Per T0.7 this changes **nothing** about what to submit — `SEQ_LEN=2048`
is strictly dominant at either grading length — it only means: **quote the @1024 number when
predicting a board score.** Use `--eval-seq-len 1024` on any run whose number you plan to
compare against the leaderboard.

## Day 1 — DONE (commits `12f7658`, `7ada0f2`)
All 8 items from the list below completed and verified. Real results: T0.1 is a
confirmed win (now the new baseline, see Status above). T1.2 (`--grad-bucket`) is a real
negative — **hangs on real Neuron hardware** in both eager and compiled mode, stays OFF,
not root-caused. T1.3 (`--clip-foreach`) is neutral, not adopted. T1.1 profile: dataloader
`fetch()` costs ~2.5-3.3% of step time exposed (real, actionable next), host contention
67.5% busy/not saturating at 4 ranks. `--seed` flag and robust `load_for_eval`
(`EVAL_MIN_SEQ_LEN=8192`) both landed and verified. C2: 315.9M train tokens confirmed,
epoch stays 1. Full detail in `research/trn2-experiment-results.md`'s "Day 1" section.

**Action for every subsequent run this week**: use `--train-local-conv
--local-conv-kernel 2` as part of the base config (T0.1's win) when comparing anything
else — it was deliberately left an opt-in flag rather than a flipped default, for commit
provenance, so it must be passed explicitly (or the globals flipped) or new sweeps will
be comparing against the *old*, now-beaten, baseline.

## Closed for good (real hardware verdicts, do not revisit without new evidence)
Mamba2/SSD hybrid, RWKV x051a, SLAY/YAT, MUDD, DCMHA, XSA, value-residual (as implemented),
FP8 via torch `_scaled_mm`, `torch.library` custom-op registration, stochastic rounding,
fused QKV, sliding-window attention, doc-boundary masking, requeue packer, PACK_FACTOR=4
(until T1.5 explains it), `--nki-relu2`. Full detail: `research/trn2-experiment-results.md`
and old plan history in git. Custom tokenizer: **not permitted** (README.md:176/178-179),
closed.

## Why reprioritize
Research plan's thesis, verified against our own numbers: **we run at ~8% MFU** (51 vs 667
TFLOP/s bf16 peak) — throughput is worth more than any single architecture idea (1.5x
throughput ≈ 0.03 bpb, bigger than everything in old Tier 2/3/4 combined). Plus a genuinely
new external data point: Recursive/Aster (nanochat autoresearch, same lineage/metric)
reached 0.9109/0.9098 vs Karpathy's tuned 0.9646 — a ~0.05 bpb gap closed almost entirely by
**hashed n-gram value embeddings** + a cluster of cheap schedule/norm one-liners. That reweights
Tier 3 (architecture) toward T3.1 specifically and demotes speculative items we already tried
(MTP, tied embeddings).

## New priority order (supersedes old Tier 1/2/3/4 ordering)

### Day 1 — calibration + free wins (today)
1. **C1**: confirm official eval token count (20M vs local 2M default) and settle the
   seq_len ambiguity (1024 vs 2048) via `prepare.py eval-public --seq-len {1024,2048}
   --eval-tokens 0` on current best checkpoint; compare local-vs-board delta of the two past
   submissions (1024 sub: local 1.15361→board 1.1394; need the 2048 sub's board score for
   the same comparison — pull it from the leaderboard if still visible).
2. **C2**: token-count the 5 train shards once; confirm epoch=1 still holds at current
   throughput.
3. Add `--seed` CLI flag (currently hardcoded `torch.manual_seed(42)`); keep 42 default.
4. Robust `load_for_eval` for any seq_len (extend rotary tables to 8192) — zero training
   cost, pure insurance.
5. **T0.1** — register `block.local_conv.weight` in an AdamW group (own group, lr=MATRIX_LR,
   betas=(0.9,0.95), wd=0). It's dead weight today; compute is already paid (and known to be
   6.9x slower to remove — keep the op, just stop discarding its gradient). 900s paired A/B,
   sweep `LOCAL_CONV_KERNEL ∈ {2,4,8}`. Keep if ≥0.002 at 900s, confirm 1800s.
6. **T1.1** — end-to-end host/device re-profile of the *current* code (post tensor-lr-scalars
   + NKI Muon); synthetic-batch test to measure dataloader host exposure (12 vCPU / 4 ranks
   contention is plausible).
7. **T1.2/T1.3** — flat-buffer gradient all-reduce (62→1-2 launches) + `foreach` clip_grad_norm.

Single owner of `train.py` for all of the above — do serially, not in parallel (collision
lesson from earlier this session stands).

### Days 2-4 — DONE (see `research/frontier-research-plan-2026-09-09.md` §12.2/§12.4 and the
### RESOLVED section at the top of this file). Next actionable block is Days 5-7 below.
- **T0.4** coordinate-descent hyperparam resweep at the 2000-step regime (MATRIX_LR,
  FINAL_LR_FRAC, WARMDOWN_RATIO, WEIGHT_DECAY, EMBEDDING_LR/UNEMBEDDING_LR, INIT_SCALE +
  nonzero c_proj init, LOGIT_SOFTCAP, Muon momentum cooldown, ns_steps) — one axis per 900s
  pair, lock ≥0.002 paired deltas only.
- **T0.6** compiler flags under compile (`--optlevel`, `--model-type=transformer`, LNC
  re-check under the compiled graph — old A/B was eager-only).
- **T1.5** resolve the PACK_FACTOR=4 anomaly (micro-batch shape microbenchmark 1/2/4/8) —
  potentially the single biggest throughput lever if it's a loader/memory artifact, not a
  real hardware wall.
- **T1.4** Adam-every-other-step, **T1.9/T1.10** host-sync + weight-cast trimming.

### Days 5-7
- **T0.2 — DONE 2026-09-11, all four arms discarded, locked shape survives.** `--depth`/`--n-embd`/
  `--head-dim` flags landed (`b434862`). 900s: d7@640 +0.0126, d8@640 +0.0137, d6@768 +0.0094,
  hd64/10heads +0.0179 (identical params — cleanest row, loses on quality *and* -8.7% throughput).
  1800s decision pair on the one close arm: d6@768 = 1771/1.037112 vs control 2187/1.034328, i.e.
  **+0.0028 at 2048 and +0.0026 at 1024** — loses at both candidate scorer lengths. Step deficit is
  a fixed -19% at both budgets while the gap shrinks 3.4x (0.0094→0.0028), so 768 is the better
  *asymptotic* shape but is not reached inside 1800s. Re-test 768 and only 768 if throughput ever
  improves ~1.4x. NKI rope+QK-norm kernel verified generic in D (rel_l2 ~3e-5 at D=64), no fallback.
- **T0.3 — DONE 2026-09-11, nothing adopted.** `MLP_RATIO=4` is a *bracketed* interior optimum
  (ratio 3: +0.0023 despite +67 steps; ratio 6: +0.0017 with -113 steps). SDPA `scale` sweep
  {0.0884 default, 0.10, 0.12, 0.15} is flat inside noise (+0.0004/+0.0003/+0.0009, non-monotone) —
  QK-norm removes the *magnitude* degeneracy but the model keeps an equivalent free knob in q/k
  angular concentration, so `scale` is a soft reparameterization, not a hyperparameter. **The whole
  softmax-temperature family (learnable per-head gain, learned global temp, entropy targets) has a
  measured ceiling of ~0.001 = the noise floor. Not built, do not revisit.**
- **T0.7 eval-side seq_len — DONE 2026-09-11.** `--eval-seq-len` flag added (`8973cff`, zero cost
  when unset, second in-process eval at a shorter context). Training at the eval length is worth
  **-0.000035** (exact tie, 30x below noise), and the missing 2x2 cell shows **`SEQ_LEN=2048` is
  strictly dominant**: tied at 1024, +0.0187 better at 2048, because the 1024-trained model gets
  *worse* (+0.0029) when given 2048 of context (untrained rotary positions). So the unresolved
  scorer-context question **cannot change which config we submit**, only what number to expect back.
  The length-curriculum / mixed-length / position-reweighting family is closed on both sides.
- Confirm best stack at 1800s, **submit** (keep 2 submissions in reserve this week). *Status: no new
  submission warranted from Days 5-7 — zero config changes were adopted, and the locked config
  reproduced a 3rd time (2187/1.034328, +0.00028 vs the lock). Calibration chain now predicts the
  board to 0.0009: 1.049682 (2M@1024) + ~0.006 window optimism ≈ 1.0557 vs actual 1.0548.*
- In parallel, zero device time: **T4.1 offline n-gram harness** on CPU (dump neural
  logprobs, build hashed count tables from train shards, fit λ, report bpb of mix vs neural
  alone on held-out train-shard tokens). Go/no-go: ≥0.005 → build device path; <0.003 → drop.
  **Do not post the T4.1/T4.3 forum questions without checking with the user first** — that's
  an external, visible action (devpost forum).

### Week 2
- **T3.1 — top architectural pick**: hashed bigram/trigram **value embeddings** (Recursive's
  recipe) riding the existing `USE_VE`/`ve_gate`/`has_ve` code path — two extra tables + two
  extra gates (channels 32:64, 64:96) per VE layer. Start table size 8×vocab, RMSProp (β2
  0.999, no WD), **no LR decay on these tables**, bf16 storage. This is the single idea with
  the strongest outside evidence (Recursive 0.9109, Aster 0.9098, Parameter Golf's only
  robust within-frontier win, modded-nanogpt record #62). Test alone at 900s paired, then
  stack.
- **T2.1** batch-size ramp (131072→262144 partway, LR scaled ≈√(B/B_ref)) — 1800s decision.
- **T2.2** EMA (decay 0.995-0.999, or Recursive's 0.98/step from 50% of budget) — cheap,
  small, safe.
- **T2.3/T2.4** cautious WD on Adam groups + smoothed scalar-group betas.
- Grab-bag one-liners from Recursive/Aster to fold into the T0.4 sweep: shifted ReLU²
  (`relu(h-0.5)²`), sandwich norm on MLP, per-head output norm+gate (T3.6), last-3-layer
  pooling, softcap 16.5/15, rotary base 1e6, causal Q/K token shift (complements T0.1),
  WARMDOWN_RATIO up to 0.9-0.95 (sweep, don't trust any one source).
- **Aster's output bigram logit table** (V×V bf16, zero-init gate) as the rules-safe
  gradient-trained stand-in for count-based n-gram mixing — try before spending effort on
  T4.1's device path.
- T1.6/T1.7/T1.8 (transposed weights, compiled optimizer, `@nki_op` traceable kernels) only
  if Week 1 showed real TensorE/launch headroom in the T1.1 profile.
- Re-run T0.2 depth/width once throughput has moved >15% (optimum shifts with throughput).

### Week 3 (Sep 23-30) — consolidation
Freeze stack by Sep 26, 3-seed confirm at 1800s, eval at both seq_len 1024 and 2048, final
submissions on the most robust config. Keep one submission identical to previous best as a
last-day fallback.

## Discipline (unchanged)
- Local repo is the only git source of truth; box is disposable — rsync out, run, pull
  `research/experiment-log.jsonl` + run-logs back, commit+push locally, every time.
- **One owner of `train.py` at a time.** No exceptions (real collision incident earlier).
- Capacity-affecting changes (depth/width/batch/embeddings) decided at 1800s only.
- Any single-run delta <0.001 is noise; <0.003 needs a paired repeat.
- New per-step Python scalars (EMA decay, batch-ramp LR multipliers, MTP weights) must be
  device tensors updated via `copy_` — never bake a Python float into the compiled graph
  (the tensor-lr-scalars lesson).
- Confirmations should move to `--eval-tokens 20971520` once C1 is settled, to match the
  real scorer's noise floor.

## Old Tier 2/3/4 kernel-engineering ambitions (SLAY-from-scratch fused kernel, Mamba2 NKI
SSD kernel) — superseded. Both architectures are closed on real hardware evidence per
`research/trn2-experiment-results.md` and the new research plan's own T4.4 closure list.
Not being revived unless new evidence appears.
