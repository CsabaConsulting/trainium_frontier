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
- Box1: 3-seed variance check on the locked config, full 1800s budget each (pulled forward from
  the freeze-period schedule). Still running.
- Box2: cross-box variance point (seed 7, different hardware) vs box1's 3-seed check below.

## Closed 2026-09-15: 3-seed variance check — real noise floor is bigger than assumed, WITH a key caveat
`--seed` (default 42, env `NEURON_COMPETITION_R1_SEED`) controls weight-init only (no dropout in
this model; data order is separately controlled by inert `EPOCH_SHUFFLE_SEED`). Full 1800s, 3
seeds: 42→1.004355/2139 steps, 43→1.008910/2136, 44→1.010491/2140. **Spread: max-min 0.006136,
std ≈0.0026-0.0032** — 3-6x bigger than this week's informal 0.001/0.003 convention.
**Important caveat/clarification (mine, not the agent's — read carefully before re-litigating any
closure)**: this spread is CROSS-SEED (different weight init each run). Nearly every closure this
week was a SAME-SEED paired A/B (control and arm both at the untouched default seed=42, varying
only the tested flag) — and this week's own same-seed control-pair reruns have consistently shown
a MUCH tighter spread (~0.0001-0.0009, matching the old convention almost exactly). That ~30x gap
between same-seed and cross-seed spread means most of the newly-measured variance comes from
weight-init itself, not generic thermal/compile jitter — so **same-seed paired closures this week
(layer-1 injection, table-mult 64→128, EMBED_ADAMW_BETA2, NGRAM_VE_BETA2, etc.) do NOT need to be
re-opened**; their own internal noise floor is the right one to judge them against, and it's the
tight one. What the cross-seed number DOES mean: **any single real-submission run carries ~0.003-
0.006 of pure weight-init luck**, which is bigger than several of the smaller "real" deltas found
this week and comparable to the gap to the next leaderboard rank. Commit `01182ff`.

### Box2 cross-box point (seed 7, full 1800s): 1.006322 — sits INSIDE box1's [1.004355, 1.010491]
No box-to-box hardware/thermal confound found; the spread is genuinely seed-driven, not a
per-machine effect. Commit `2575872`.

### Seed hunt: full batch complete (14 unique seeds, 16 full-1800s-budget runs), box2's confirming repeat of 42 still pending
Sorted best to worst (`public_val_bpb`, full 1800s each; 5/13 averaged across their two
measurements): **42=1.004355**, 3=1.005730, 2=1.006220, 7=1.006322, 1=1.006380, 8=1.006441,
55=1.007353, 144=1.008352, 43=1.008910, 5=1.010038(avg of 1.009551/1.010524), 44=1.010491,
34=1.010748, 89=1.010848, 13=1.010899(avg of 1.010303/1.011495).

**Seed 42 (the untouched current default) is still the best of all 14**, but its lead over 2nd
place (seed 3, gap 0.001375) is roughly the SAME SIZE as the same-seed noise band the 5/13
duplicate measurements revealed (~0.001-0.0012 between two runs of the identical seed). So on
current evidence, **42's lead over 3 is not yet clearly distinguishable from noise** — box2's
confirming repeat of 42 (still running) is exactly the data point needed to settle this.

**Real, useful side finding (box1's cheap-proxy-vs-full-budget check, seeds 5/13)**: the 300-step
proxy got the one PAIRWISE call right (5 beats 13 at both proxy and full budget) but the ABSOLUTE
ranking did not hold — at full budget neither 5 nor 13 was the best or worst of their batch; the
actual extremes (55 best, 89 worst) were seeds the proxy never even tested. **Conclusion: the
cheap 300-step proxy is not a reliable seed-picker on its own** — pairwise comparisons might
transfer, but "find the best/worst seed" does not. Any future seed screening should go straight
to full budget, not through this proxy.

All commits: box2 `de062fb`,`ec9e73a`,`39c27f4`,`de5c7fe`,`d269254`,`ddb0b7f` (batch) + `417457c`
(seed-42 repeat: 1.005049, avg 1.004702); box1 `26e4500`,`994124c`,`afae61d`,`a29f4b4`,`cb91730`,
`b4bdfde`,`e126d26` (batch, complete) + `98b81d4` (seed-3 repeat: 1.005414, avg 1.005572, delta
from first measurement -0.000316, tighter than the noise band). **42-vs-3 gap now down to 0.00087
on 2 points each — squarely inside the observed same-seed noise band.** Seed 42 still leads on
every individual measurement and both averages, so nothing argues for switching, but with only 4
total data points the gap can no longer be called "clearly outside noise" the way the 1-point
comparison suggested. **Working conclusion, still holding**: keep seed 42 (no downside — it's
already the default and still the best-measured average).

### Widened search, batch 2/3 (25 of ~26 seeds measured): still no clear winner, one near-miss
Batch 2 (box2, seeds 200/300/400/500/600/700, commits `fd51d9e`,`7bbb3bb`,`85680f5`,`9051660`,
`2541431`,`a77899a`): best was 300 (1.005664, competitive with seed 3's average but not better),
worst two (200, 700) ~1.0121, among the worst of the whole hunt. Batch 3 (box1, seeds
1000/1500/2000/2500/3000/3500, commits `a60daf7`,`79bc565`,`58edb94`,`e2e9c58`, +3000/3500
pending): 1000/1500/2000/2500 all clearly worse than both leaders. **Seed 3000 = 1.004900 — the
closest non-42 result of the entire hunt, only 0.0002 off seed 42's 1.004702 average, still
inside the noise band but worth a confirming repeat.** No seed has yet cleared 42 by the ~0.002
bar across 25 measurements. Box1's batch 3 now fully complete (commits `a60daf7`,`79bc565`,
`58edb94`,`e2e9c58`,`d9367f0`,`7881d38`): 1000/1500/2000/2500/3500 all clearly worse (+0.0034 to
+0.0101 vs seed 42's average); seed 3000 confirmed as the standout at +0.000198 vs seed 42's
average, inside the noise band, still not a win. Seed 3000's confirming repeat (box2, commit
`77f6b7d`) landed at 1.005650, avg 1.005275 — still behind seed 42's 1.004702, gap 0.000573, a
statistical tie at best, not a real win.

### CLOSED 2026-09-15: seed hunt concluded — KEEP SEED 42, NO CHANGE
26 unique seeds measured at full 1800s budget (32 total runs incl. repeats). Seed 42 (the
untouched current default) won every single comparison and every average across the whole hunt.
**Clarifying insight from the user, confirmed correct**: `SEED` only drives `torch.manual_seed()`
before weight init (no dropout in this model, data order is separately fixed) — so this whole
exercise was pure max-of-N luck-hunting over a noise distribution with std ~0.003, not an
algorithmic lever. With std this size, beating the incumbent by a real (non-noise) margin needs
either far more draws than 26 or genuine luck; returns are provably diminishing per additional
draw. **User's explicit call: stop here, keep seed 42.** No `train.py` change — the file was
never touched during the entire hunt, so there is nothing to revert. Both boxes free.
**Reusable finding for later, if anyone revisits this**: don't bother re-litigating the seed
question with fewer than ~20-30 draws; the noise floor here is real and substantial.

## 2026-09-15 — deep research pass for fresh ideas, one item closed same-day
A fork did a deeper literature scan + re-audit of "closed for good" items for stale assumptions.
Confirmed still-correctly-closed: T4.1/T4.2 n-gram output mixers, value-residual, coalesced
SwiGLU, epoch-shuffle — none were closed on assumptions the current 592.9M/2-injection-site regime
invalidates. Flagged then CLOSED same-day: whether our dev venv (custom `native_venv`, not the
README's named `neurips-torchneuronx-native-beta-2.11.3-20260824`) risks silently differing from
the real grading environment, given several real wins are compiler-scheduling quirks. Resolved via
web search: this is the user's OWN forum post from days ago
(`trainium-frontier.devpost.com/forum_topics/45051`) — the named venv doesn't exist even on the
officially-specified AMI, nobody (including the competition manager) flagged a fix, and it's the
same environment already used for every number in this notebook, already implicitly validated by
real board scores matching local predictions to ~0.0006 across multiple past submissions. No
action needed, no new instance spun up.

**Remaining candidate, not acted on, low priority**: model ensembling (split the 30-min budget
across N concurrent shorter/smaller runs) — a real technique in the Slowrun community, connects to
this session's own measured seed variance, but every throughput-vs-capacity result this week says
splitting compute this way loses badly (nothing clears breakeven above ~10% cost, let alone a 50%+
split). User approved a small deliberate 2-way-split check at 900s to settle it — now CLOSED:
equal-total-compute comparison (control world_size=4 vs 2x world_size=2 replicas, wall-clock
matched at 900s each since batch size auto-scales with world_size): control alone 1.051295,
replica A/B alone 1.108744/1.108619, ensemble(A,B) via averaged logits 1.088441. Ensemble beats
either solo replica (recovers ~35% of the standalone degradation) but loses to the control by
**+0.037146** — 10-35x the noise floor, decisive and unambiguous. Confirms the strong prior.
Commit `7a74a8f`, train.py untouched. Not worth pursuing further.

## Closed 2026-09-16: final wrap-up dress rehearsal — CLEAN
Literal bare launch command, box2, post seed-hunt conclusion (nothing in train.py changed).
2131 steps, `public_val_bpb=1.003045`, causality clean, over_budget False, 592.9M params — vs
last known-good (`fac0efc`: 2119 steps, 1.003408): +12 steps, -0.000363, both favorable, delta
well inside the established same-seed noise band. No regression. Commit `190412c`, train.py
untouched. Box2 now idle per the user's approved plan — no new work there without checking in.

### Preliminary 8-seed short-budget screen (box1, 300-step proxy, exploratory, no seed adopted)
Seeds 1,2,3,5,8,13,21,100 at fixed 300 steps (`lrm=1.0`, no cooldown). Range 0.007314, std
~0.0021-0.0023 — same order of magnitude as the full-budget 3-seed spread, no dramatic outlier.
Best: seed 5 (1.221123). Worst: seed 13 (1.228437). Doesn't include 42, so can't say if 42 still
looks good among these. Commit `21fb193`, train.py/SEED untouched.

### User decision (2026-09-15): YES, run the real seed hunt
Asked the user directly given this is a strategic call, not a technical one. **Approved**: screen
~10-20 seeds at full 1800s budget, lock in the best as the new `SEED` default, verify+document
like any other closure. 4 full-budget seeds already in hand (42/43/44 on box1, 7 on box2). Launching
the remainder across both boxes now.

### Live idea, needs a decision before spending real box-hours on it (SUPERSEDED by the above)
Seed 42 (our unmodified default) already beat 43 and 44 in this one-sample test. Since the actual
graded run is the literal bare launch command with NO `--seed` flag, whatever `SEED` module
default sits in train.py IS what the grader's own from-scratch training run will use — meaning
**we can pick a good default seed and it directly, deterministically carries into the real
submission** (same-seed reproducibility is tight, ~0.0001-0.001, so a locally-measured good seed
should reproduce close to itself at grading time regardless of whether its advantage is "real" or
one-off luck). This is a legitimate, low-risk, non-architectural lever worth potentially 0.003-
0.006 bpb — on the same order as several of this week's real wins — for the cost of running a
batch of candidate seeds. Flagged to the user before committing serious box-hours to it (is this
within the spirit of the rules the user wants to play by); a cheap preliminary screen is running
on box1 in the meantime, non-committal, changes nothing yet.

## Closed 2026-09-15: regression-check dress rehearsal — CLEAN
Full bare-launch run on box2, post this week's 5 train.py-touching commits (incl. the step-cap
guard, `dffa5ee`). 2119 steps, `public_val_bpb 1.003408`, causality clean, over_budget False,
592.9M params — vs the 2026-09-14 known-good (2122 steps, 1.004550): −3 steps, **−0.001142**
(slightly better, inside noise, not regression-shaped). No interaction found between this week's
changes. **Note for next dress rehearsal**: use the truly flagless launch command (no
`--no-eval-public`) — that flag suppresses the in-process causality/bpb print and forced a
post-hoc recovery via `prepare.py eval-public` this time (worked, cross-validated to ~2e-5, but
adds unnecessary steps). Commit `fac0efc`, docs-only, train.py untouched.

## 2026-09-15 — quick external-research pass, no action taken
Ran a short web-search scan for fresh techniques (Muon-descendant optimizer benchmarks, recent
sparse-embedding papers, nanogpt-speedrun record list) after the n-gram-VE-table hyperparameter
line closed out. Findings, for the record, not acted on:
- **Muon remains undefeated.** Keller Jordan's public optimizer benchmark (Shampoo, spectral
  descent, MARS, SWAN, AdaMuon, etc. vs Muon) still has "Muon with aux Adam" on top — which is
  exactly our current setup (Muon for hidden weights, AdamW for embeddings/scalars/lm_head). No
  optimizer swap to chase.
- **Multi-head hashing (2026 ACL "Deep Sparse Embedding" paper)** uses K parallel small hash
  heads per n-gram order instead of one big table, reducing collision correlation at fixed param
  count. Real, structurally distinct from anything tried here — but deprioritized: our own
  mult=256 result this week (quadrupling effective rows) bought only -0.0022 bpb, i.e. collision
  rate is already shown NOT to be the bottleneck at our table size, so a smarter hash likely
  wouldn't unlock much more. Also plausibly costs extra op-count (K small gathers vs 1 big one),
  which is the exact lever the Engram-lite closure already flagged as the real cost driver, not
  parameter count. Not pursued.
- **Attention-window warmup** (nanogpt speedrun record #13, growing the attention window over
  training) — deprioritized: static sliding-window attention is already in the "closed for good"
  list here as a real loss at our seq_len=2048, so restricting attention early is unlikely to help
  either, and full attention likely isn't even the bottleneck now (n-gram gather is).

## Closed 2026-09-15: `NGRAM_VE_OPT` rmsprop-vs-adamw — CLOSED NEGATIVE
First empirical test of Recursive's stated rmsprop rationale (rare hashed-row updates get damped
by Adam's m/sqrt(v)) on this stack at the current locked shape. adamw costs +1.52% s/step AND is
worse on quality (+0.00225/step, reproduced twice, well above noise). Break-even moot — both axes
point negative. Net at 1800s: +0.00343 worse. `NGRAM_VE_OPT` stays "rmsprop". Commit `4e740ee`,
train.py untouched. Together with the beta2/lr closures: **all three untested n-gram-VE-table
optimizer knobs are now closed** (lr neutral, beta2 lost, optimizer-family lost) — this whole line
of attack on the table group is exhausted, not just under-explored.

## Closed 2026-09-15: step-cap defense-in-depth guard — SHIPPED
Adds `STEP_CAP_GUARD_MIN_PROGRESS=0.5`/`MAX=0.98` (train.py:237-238) and a loud rank-0 WARNING
(non-fatal print, not an assert) when the loop exits via the step cap while
`0.5 <= elapsed/max_train_seconds < 0.98` — catches a reintroduced 2026-09-10-style bug (step cap
binds well before the real wall-clock budget) without spamming on this week's many intentional
short `--num-steps` probes (fixed at ~6.7%/~20% of budget, well under the 0.5 floor). Verified:
normal 300s run at locked defaults (345 steps, time-budget exit) — silent, no false positive.
Deliberate repro (`--num-steps 250 --max-train-seconds 300`, cap binds at 68.3%) — fired exactly
as designed. `py_compile` clean. Commit `dffa5ee`, md5 `24cce227d3bcf9750f6ea07f9d58a8d1` (train.py
genuinely changed — this is a real code change, not a flag-only run).

## Closed 2026-09-15: `NGRAM_VE_BETA2` sweep — CLOSED NEGATIVE
Tested 0.95 (mirrors EMBED_ADAMW_BETA2's winning value) and 0.99 (midpoint) against locked 0.999.
Throughput unaffected (pure hyperparameter, 0.19%/0.04% spread). 0.99: +0.0002, clear noise.
0.95: +0.0016 on first pass, escalated to a same-day repeat per the 0.001-0.003 rule → +0.0025;
both pairs agree in sign/magnitude (mean +0.0020, 7-18x the noise band) — a **real, reproducible
loss**, opposite sign from the EMBED_ADAMW_BETA2 precedent this was modeled on. `NGRAM_VE_BETA2`
stays 0.999. Plausible mechanism (speculative, not separately measured): RMSProp's no-first-moment
design for rarely-touched hashed rows means a lower β2 decays the lone second-moment estimate
faster between rare visits — opposite of the dense, always-visited two-moment case
EMBED_ADAMW_BETA2 governs. Commit `5e8a079`, train.py untouched. Together with the `NGRAM_VE_LR`
neutral result: **the "never-tuned inherited default" heuristic is now 1-for-3** on this table
group (won for EMBED_ADAMW_BETA2, neutral for lr, lost for beta2) — a real, useful calibration on
how much further to push this specific line of attack.

## Closed 2026-09-15: `NGRAM_VE_LR` sweep — CLOSED NEUTRAL
Tested whether the n-gram VE tables' own lr (sentinel 0.0 → inherits `EMBEDDING_LR=0.30`, never
tuned standalone) matters, same pattern that won for `EMBED_ADAMW_BETA2` this week. Bracketed
0.15/0.60 vs 0.30 control, paired 900s, throughput confirmed unaffected (pure hyperparameter,
0.034% spread across all 4 runs). Quality delta: lr=0.15 −0.00032, lr=0.60 +0.00032 (both well
under the 0.001 noise floor; control's own pair spread was 0.000088). No repeat/1800s triggered.
`NGRAM_VE_LR` stays at its 0.0 sentinel. Commit `512b43e`, train.py untouched. Unlike beta2 (still
running on box1) or the earlier EMBED_ADAMW_BETA2 win, this specific knob doesn't need tuning —
the "never-tuned inherited default" heuristic doesn't universally pay off, good to know before
trying it on any more knobs.

## Closed 2026-09-15: `--ngram-ve-table-mult 256` (dim unchanged) — CLOSE NEGATIVE
Isolates the other axis of the dim/mult grid from the dim=256 closure above. Control 592.9M/
0.8410 s-step; arm (mult=256) 1129.8M/1.0360 s-step, **+23.2% s/step**. Per-step quality −0.0022
(real, reproduced to 0.00002-0.00004 across 2 reps). Break-even bar +0.0161 at 1800s → net
**+0.0139 worse**, needs +19.7% more throughput than measured. `NGRAM_VE_TABLE_MULT` stays 128.
**Notable**: shortfall (+19.7% needed, net +0.0139) is nearly identical to dim=256's isolated
result (+19.4% needed, net +0.0137) — consistent with table cost ∝ rows×width, so doubling
EITHER factor alone costs/buys about the same; the pre-layer-1-era "dim256+mult128 together
nearly free" precedent does not transfer to either axis alone at the two-site config. (A combined
dim=256+mult=256 probe would likely also lose, by the same ∝rows×width logic, but wasn't run —
low priority given both marginal axes already miss break-even by ~6x.) Commit `7e62330`,
train.py untouched.

## Closed 2026-09-15: `--depth 8 --n-embd 640` (fixed-width depth increase) — CLOSED NEGATIVE
Decoupled the earlier confounded 768×8 result (which also bumped width) from a pure depth
increase. Confirmed `NGRAM_VE_LAYERS="1,-1"` still resolves correctly at depth=8 (`{1,7}`, both
odd, matches `has_ve`'s parity rule). Paired 900s, real val_bpb: control 592.9M/537 steps/
val_bpb 1.112144; arm 608.0M/548 steps/val_bpb 1.107724. **Throughput −48.70%** — the largest
single-technique cost measured on this whole track (the layer-1 n-gram gather's downstream-
resharding cost scales with depth below the injection site: span grows 4→6 layers). Per-step
quality −0.00286 to −0.00303 (real win, opposite sign from a misleading 120-step proxy). Breakeven
bar: forgone-steps penalty +0.05136 vs −0.00286 gain → net **+0.0485 worse**, needs +87.8% more
throughput than measured (biggest lever ever found on this stack is ~5.86%). `DEPTH`/
`ASPECT_RATIO` stay locked. Commit `98eed62`, train.py untouched (no code changes needed).

## Closed 2026-09-15: `--ngram-ve-dim 256` on the locked two-injection-site config — CLOSE NEGATIVE
Re-tested the pre-layer-1-era "dim256+mult128 nearly free" precedent at the real current config
(`layers="1,-1"`, mult=128 locked). Paired 900s, real val_bpb (not proxy loss): control 592.9M/
157724 tok/s/0.8308 s-step/val_bpb 1.069576; arm (+dim256) 1130.1M/127359 tok/s/1.0290 s-step/
val_bpb 1.066135. Throughput −19.25%, step-matched per-step quality −0.00281 (real, above noise).
Break-even bar: forgone-steps penalty +0.01649 at 1800s budget vs only −0.00281 gain → net
**+0.01368 worse**, would need ~+19.4% more throughput than measured to break even — no lever
this whole track has found buys anywhere near that (biggest win, layer-1 itself, cost only
−5.86%). `NGRAM_VE_DIM` stays 128. Confirms/supersedes the earlier 120-step-proxy 2026-09-14
closure on firmer (real val_bpb) footing. Commit `8810a71`, verified clean, py_compile OK.

## Closed 2026-09-15: `--muon-lr-fp32` (the bf16-lr-quantization side finding) — CLOSED NEGATIVE
Added `USE_MUON_LR_FP32`/`--muon-lr-fp32` (default False, unchanged behavior), threaded through
both `_muon_lr_g` call sites. CPU check confirmed the flag does what it claims (flag off is
bit-identical to before; flag on differs by up to 0.4566% relative, matching the expected bf16-ULP
envelope, touching only `w_out`) — mechanistically real, not a no-op. Paired 900s hardware A/B
(control-r1/arm/control-r2): throughput genuinely unchanged (884.7/884.5/884.7s, ~1055 steps all
three) confirming the zero-cost prediction, but per-step-quality residual after throughput
decomposition is -0.0000547 (@2048) / -0.0000672 (@1024) — ~1/15th of the 0.001 noise floor. No
repeat or 1800s run triggered (reserved for signals ≥ noise floor). `USE_MUON_LR_FP32` stays False.
Commit `df82d08`, verified: HEAD=origin/main, py_compile OK, md5 `ec7bc96b66904431d3717fdaef4856d8`.

## Closed 2026-09-15: Recursive's remaining schedule tricks (Demon β1, NorMuon β2 ramp, per-depth LR/momentum)
All three closed negative/neutral. Demon β1: neutral at warmdown window, worse over full run.
NorMuon β2 ramp: our `beta2`=0.90 already sits on the flat top of the curve, confirmed our
`MUON_BETA2` IS the NorMuon mechanism — nothing to ramp toward, all 3 variants neutral. Per-depth
LR/momentum: LR-only +0.000719 (inside noise floor), momentum-only worse, combined NOT additive
(worse than LR-only alone). Real catch: the equivalence gate caught a genuine bf16 `lr`-cast
type-promotion bug (0-dim fp32 `lr` vs `[K,1,1]` `lr` round differently against bf16 `g` —
different common-dtype rule). Fixed in `_muon_lr_g` (train.py:3546) so the per-depth path can't
smuggle in a precision confound. Side finding this surfaced: the locked path itself quantizes
every Muon group's effective lr to bf16 (~0.4% relative) every step — flagged as its own
candidate free win, now being tested standalone on box2 (see In progress above). Verified clean:
`b1eaf3b`, py_compile OK, md5 matches box2.

## Closed 2026-09-14: dead-code sweep of `train.py` — 7,275 → 5,191 lines (−2,084, −28.6%)
Motivated by the `REJECTED_BY_REVIEW` false positive (the judge's "unsafe `torch.load`" claim is
simply wrong — `weights_only=True` has always been there) plus `README.md`'s "simpler code"
tie-break. Nine closed families deleted outright, not defaulted to `False`: RWKV, Mamba2/SSD (incl.
its NKI kernel pair), DCMHA, MUDD, XSA, value-residual, `Fp8Linear`, stochastic rounding, requeue
packer — 20 toggles and 3 CLI flags gone. `make_dataloader_requeue()` is KEPT (`USE_EPOCH_SHUFFLE`
needs it) and its body is now byte-for-byte `prepare.make_dataloader()`'s. Nothing on the locked
path touched. One commit per family (`d4abc14`, `73a1700`, `39efdc9`, `b93426a`, `8fd47d9`,
`796a223`, `2c38d23`), each gated on a CPU fingerprint probe proving bit-exactness against the
pre-cleanup file (loss to 12 s.f., all 80 param md5s, all 74 grad md5s, optimizer group structure).
Hardware: `load_for_eval` on the real `out/final.pt` gives an identical `val_bpb`
(1.0035637044807473) under old and new, `causality_max_abs 0.0` under both, `prepare.py check`
passes, and a 110-step same-command run sits inside the old binary's own run-to-run band. Full
write-up + verification table: the 2026-09-14 section of `research/trn2-experiment-results.md`.

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

## Queue as of 2026-09-14 evening — all of the previous "queued next" is now resolved
T1.7 shipped (compiled Muon, real +3.38%). N-gram VE layer-1 addition shipped (real -0.0118).
T1.6 (transposed weights) and T1.8 (NKI kernel tracing) both closed negative this week, with
real numbers (see the dated 2026-09-14 sections in `research/trn2-experiment-results.md`).
**LNC=1: corrected same evening** — this morning's "runtime-level deadlock, nothing to fix"
verdict was measured against a stale compile cache. Clearing every Neuron cache dir turns the
silent hang into a real, narrow, informative compile error (`nkilib`'s `foreach_norm`'s
`l2_norm_kernel_lnc0`, invalid `CollectiveKind::SendRecv` on Trn2 — the same bug from months ago,
never actually fixed). `--no-clip-foreach` (or `--grad-clip 0`) routes around it and LNC=1 *runs*
real training steps again at world_size=1. **Full closure, same evening**: at the real 4-rank
shape, with both software bugs routed around (fresh cache, explicit `--lnc=1`, `--no-clip-foreach`),
LNC=1 gets further than ever (real `torch.compile`, real Muon specializations) then hits a **real
device OOM** (`ret=-12`/`ENOMEM`, `Failed to allocate DEVICE memory`) — LNC=1's HBM isn't pooled
across cores the way LNC=2's is, and the current 592.9M-param model (537M of it n-gram tables, 10x
the 55.7M-param model that produced the old cliff number) plausibly doesn't fit one core's share.
**All four causes from the user's own research (`research/gemini_ai_mode_lnc_1_failure.md`) are
now accounted for** — compile/runtime mismatch (fixed), NKI-kernel bug (fixed), device OOM
(confirmed, real, not fixable by a flag), old SDK version (confirmed real, not chased, out of
scope). LNC=1 is closed for good with a complete answer, not an assumption. Still curiosity-tier,
still submitting at LNC=2. **New standing rule**: clear caches before trusting any "X doesn't
work at all" conclusion on this stack, not just before a cold dress rehearsal.

**Currently running**: small/cheap items (byte-feature `wte` init, coalesced SwiGLU, T2.3/T2.4
cautious WD/scalar betas) on box 1.

Note: any future throughput win should be spent the same way as the T1.7/qk-shift pivot — on
capacity at fixed step count, not on more steps, until/unless the data-wall situation changes
(e.g. more train data ever becomes available, which it won't under the rules).

## Closed 2026-09-15: small/cheap batch — all four DISCARD/neutral
Byte-feature `wte` init: DISCARD, monotone dose-response the wrong way (+0.0022 worse at mix 1.0,
still +0.0012 worse at mix 0.25). Coalesced SwiGLU: DISCARD, worse on both per-step (+0.0042) AND
throughput. T2.3 cautious WD: exactly neutral (−0.00009, noise), stays OFF; refactor proven
bitwise-free via `research/scratch/cautious_wd_equiv.py`. T2.4 smoothed scalar betas: neutral/
negative, re-confirms `SCALAR_LR=0.20` is already the top of the curve. All four kept in tree
behind their flags, default OFF, zero cost at default. Full write-up:
`research/trn2-experiment-results.md` 2026-09-15 "three small items" section.

## Small/cheap, still unimplemented
- `lr_multiplier` defense-in-depth against step-capping (optional hardening on top of the
  NUM_STEPS fix; cheap, low priority since the actual cap is already gone).

## Blocked on user, do not act without explicit go-ahead
- Any Devpost forum post (external/visible action) — currently nothing queued to post; the one
  drafted question (frozen n-gram count-mixer rules check) is moot since that idea died.

## RULED, permanent hard constraint (organizers answered directly, 2026-09-14)
The excluded launch→5th-step (≤900s) startup window may **not** be used to (a) pre-fill n-gram
tables or any count-based asset ahead of the charged loop, or (b) via uneven step construction —
engineering the first 5 steps to run long and carry disproportionate real learning, pacing faster
once the clock resets. Both confirmed prohibited, not merely risky. Verified `train.py` is clean
against this (only reads the harness's own exclusion constants to re-base internal logging, never
inflates steps 0-4's compute). Legitimate uses remain: compiling extra graph shapes, cache warmup.
Permanent — do not revisit either variant. Full detail: `research/frontier-research-plan-2026-09-09.md` T4.3.

## Scheduled
- Keep actively researching/experimenting through ~Sep 27. Only the **last 2-3 days** (~Sep
  28-30) are a real freeze — one final dress-rehearsed submission of whatever's best-known-good
  by then, plus a quick 3-seed variance check (~1.5h box time, not a big undertaking). The
  current board score stays on the leaderboard regardless of what happens after it, so there's
  no need for a long freeze — just enough runway to not risk a broken last-minute submission.
  (Revised down from the original "freeze all of Week 3" text, which was inherited from the
  Sep 9 plan and was more conservative than the actual risk warrants.)

## Distance to the leaderboard (2026-09-12 snapshot, will have moved — live race)
- To 10th (1.0303): **0.0081** — plausibly one more real win (the n-gram-layers-1/3 fix alone
  is estimated ~0.007). Realistic near-term target.
- To 1st (0.9683): **0.0701** — needs several more structural wins, a much bigger ask.

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
- **Keep both boxes occupied whenever possible.** They cost the same whether idle or busy —
  whenever one frees up, launch the next queued item immediately rather than waiting.
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
