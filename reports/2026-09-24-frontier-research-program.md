# Trainium Frontier Phase 1: evidence audit and research program to the freeze

Written 2026-09-24 11:35 UTC (17:05 IST). Research freeze: 2026-09-29 18:00 IST (12:30 UTC). Competition
deadline: 2026-09-30 23:59 PDT, which is **2026-10-01 12:29 IST**.

**Inputs audited (SHA-256):**

| File in this repo | SHA-256 | Notes |
|---|---|---|
| `01-governing-rules-protocol-and-mission.md` | `12d00de390389183bbc33832842d627d51a3523e5557c4f7c44ec8114dfef11a` | README, experiment protocol, standing mission v2 |
| `02-current-state-and-decision-log.md` | `b38027c6d2d9473be8dbcf5041998841730f66f96ad7d44576d4677d764b4409` | CURRENT_STATE, DECISION_LOG |
| `03-experiment-log.jsonl` | `e7d9554d9e89b49b40fea356e82c5361851f0d5a1290c04cc2109202de0d47bc` | 401 rows, no NUL corruption, ends 2026-09-18T02:57Z |
| `04-train-research-35c9.py` | `35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2` | matches the mission's research-file hash |
| `05-train-submission-clean-116188.py` | `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d` | matches the mission's clean-candidate hash |

External primary sources were read at pinned revisions (Section 10). The Recursive file
`nanochat_autoresearch/solutions/optimized_from_karpathy.py` at commit `a962ec43e2e3d7c018e59a2ece623fe6e232fdfb`
was downloaded and hashed: **`f92ecc814039ff177b3bfd23c7967918bc44e88765572923277bfa40b7fb597d`**, which matches the mission.

Conventions: **"public-20M"** means a fresh-process `prepare.py eval-public` at context 1024 over 20,971,520 tokens.
**"public-2M"** means the 2,097,152-token run of the same evaluator. As far as I can tell it is a prefix of the 20M
stream; verify this in `prepare.py`. Every Δ is candidate minus same-host control, so negative is better. Anything
labelled *estimate* is my projection, not a measurement.

---

## 0. One-screen summary

1. **A1/A2 as written will not start on the exact 35c9 bytes.** When run as `__main__`, 35c9 installs
   `_install_submission_training()`. That wraps `GPT.setup_optimizer` in `OwnedOptimizer`, and its
   `make_plan()` raises `ValueError` for any group whose kind is not `adamw`/`muon` and for any non-FP32 parameter
   (`04-train-research-35c9.py:4455-4466`, `:4771-4772`). The A1/A2 flags create a `kind='rmsprop'` group of BF16
   tables (`:2153-2155`). The same adapter also asserts `RELU2_TAU == 0.0` (`:4404`), so the "ReLU² tau 0.5" arm
   crashes too. Tables need a small, verified code change first (IMPL-1 in §4). This is the first critical-path task.
2. **The +0.005 public→official offset holds only for the 2M prefix.** The one exact pair in the ledger
   (`dress-rehearsal-newbox`, official 1.0548) gives **official − public-20M = −0.0009** and official − public-2M = +0.0053.
   The current best (public-20M 0.99195) therefore projects to **about 0.991 official**, not about 0.997. First
   place (0.9508) corresponds to roughly public-20M 0.952, and the top-10 cutoff (0.9832) to roughly 0.984. This is n = 1,
   so the uncertainty is ±0.002.
3. **Repeated data is probably the central scientific issue.** An epoch is about 2,150 steps × 131,072 ≈ 282M tokens.
   Dense runs about 3,640 steps, or **1.7 epochs**, and the whole second epoch falls inside the LR cooldown. On the 640
   table stack, every change that pushed a run past about 2,150 steps got worse (qkfreeze +0.0023 from about 200
   epoch-2 steps; epoch-shuffle +0.0048; the autoloop's "CONFOUNDED" throughput wins at 2,346–2,716 steps were
   +0.010 to +0.015). All historical table gains (−0.015 last layer, −0.030 total) were measured **within one
   epoch**. So tables on dense are still the highest-value bet, but they are not a safe one.
4. **Recommended critical path:** IMPL-1 (table optimizer bypass, about 3–5 h including verification). Meanwhile, run
   per-host 35c9 controls, an epoch-value diagnostic, and flag-only arms that the adapter actually accepts. Then run
   A1 on trn21 and A2 on trn22 in parallel, and cross-host replicates. Keep a fresh-tail data order and an epoch-2
   table freeze as pre-registered follow-ups.
5. **Submit a cleaned dense file early** (human decision). It banks roughly 0.991 official against today's 1.0246, and,
   more importantly, it tests whether a review-clean file passes review. That is the largest binary risk left.
6. **Probabilities (public-20M; official ≈ public − 0.001):** ≤0.980 by Sep 25 23:00 IST: 6%. ≤0.975 by Sep 26
   23:00: 6%. ≤0.960 by Sep 28 12:00: 2%. ≤0.945 by the freeze: <1%. **By the freeze:** ≤0.984 (about today's top-10
   line) 45–50%; ≤0.980 about 28%; ≤0.975 about 15%. Median final: about 0.985, 80% interval 0.975–0.992.
   First place: <1%.

---

## 1. Executive judgment

**Strongest current path.** Treat hashed n-gram value tables as the one high-upside mechanism, but gate them on
the repeated-data question.

1. Unblock tables in 35c9 with IMPL-1: route table groups around optimizer ownership, keep them replicated, and
   all-reduce their gradients densely, which is how the historical table runs worked. Acceptance is proof that
   `--no-ngram-ve` behaviour is unchanged.
2. Screen A1 (layer −1) and A2 (layers 1,−1) in parallel, one per host, against same-host controls with 20M
   evals. Then replicate cross-host.
3. In parallel, measure how much dense gains from its second epoch. D1 is a run compressed to one epoch; FT is a
   fresh-tail data order. Both are cheap and decide whether tables should be frozen in epoch 2, and whether
   throughput is worth anything at all.
4. Only if tables are confirmed, scale them with B2 (touched-row updates) to Recursive-like placement and size, then
   re-bracket matrix LR and warmdown.
5. Keep the review-clean file within 12 h of the research best. Submit a cleaned dense file now to de-risk review.

**Realistic final range.** These are estimates.

| Scenario | P | Best confirmed public-20M at freeze |
|---|---|---|
| Tables fail on dense (epoch-2 interaction, overhead) | 0.35 | 0.9895–0.9920 (flag-bank/schedule gains only) |
| Tables transfer partially (−0.004 to −0.010) | 0.40 | 0.980–0.987 |
| Tables transfer strongly and B2 scales them | 0.25 | 0.968–0.980 |

**Milestones, answered honestly.** The mission's milestones are campaign targets. As forecasts they are not
supported by the evidence.

| Milestone (public-20M) | Deadline (IST) | Hours from now | P(by deadline) | P(by freeze) | Why |
|---|---|---|---|---|---|
| ≤0.980 | Sep 25 23:00 | ~30 | **6%** | 28% | Needs about −0.012 from the first table screen *and* a same-day replicate. A2's single-epoch analogue was about −0.012 to −0.015 on 640, before any epoch-2 penalty. |
| ≤0.975 | Sep 26 23:00 | ~54 | **6%** | 15% | Needs tables plus B2 scaling plus retune, all landing within 2 days. |
| ≤0.960 | Sep 28 12:00 | ~91 | **2%** | 3% | Requires near-full transfer of the 640 stack's −0.030 *plus* Recursive-scale tables *plus* no epoch-2 penalty. |
| ≤0.945 | Sep 29 18:00 | ~121 | **<1%** | <1% | No mechanism in the evidence accounts for −0.047. |
| ≤0.984 (≈ official top-10 today) | freeze | ~121 | n/a | **45–50%** | The cutoff will probably tighten before the deadline. Treat about 0.981 as the real top-10 target, which is about 35%. |

Do not plan around first place. The leader's 0.9508 is about 0.041 better than our best, and no measured or
literature mechanism here closes that gap in five days.

---

## 2. Evidence correction

### 2.1 What was supplied and what is missing

| Required item | Status | Consequence |
|---|---|---|
| README, protocol, standing mission v2 | Supplied, embedded in file 01 | Controlling rules |
| CURRENT_STATE, DECISION_LOG | Supplied, file 02 | Only source for the 2026-09-19 to 24 dense lineage and the autoloop results |
| `research/experiment-log.jsonl` | **Supplied but stale**: 401 rows ending **2026-09-18T02:57Z** | It holds **no** dense-6×1024 run, no 0.9928513 record, no LR-0.015 runs, none of the 09-22/23 autoloop runs, and nothing for the official 1.0246 submission. Every dense-era claim below is supported only by CURRENT_STATE/DECISION_LOG text and is **provisional**. |
| 35c9 research `train.py` | Supplied, hash verified | Audited |
| 116188 clean `train.py` | Supplied, hash verified | Audited, §7.4 |
| `prepare.py` (evaluator, loader, `TRAIN_STARTUP_*` constants, `--num-train-shards` default) | **Missing** | Eval context, token prefix semantics, startup-cap enforcement and dataset size are **inferred**, not read. |
| `research/trn2-experiment-results.md` ("data wall" section), run logs, verifier receipts, Track-B feasibility/probe docs, proposals | **Missing** | B2 status is taken from the mission text only. Epoch-boundary details come from ledger numbers plus the 35c9 docstring at `:88-100`. |
| Official submission IDs and configs (1.1495, 1.0548, 1.0246), full reviewer texts | **Missing** | Only one public/official pair can be reconstructed (§2.4). |
| Signed-in leaderboard read | Missing (human-reported) | 0.9508 and 0.9832 are unverified. |

### 2.2 Measured facts reconstructed from the supplied files

All numbers are single runs unless noted. "2048" and "1024" denote eval context. Old-ledger evals are
**2M** unless stated.

| Claim in brief | What the evidence says | Verdict |
|---|---|---|
| Best dense: 0.99207 / 0.99184, 3,635 / 3,654 steps, mean 0.99195 | DECISION_LOG rows dated 2026-09-24; not in the supplied ledger | Accept provisionally (evidence dir not supplied) |
| Previous dense 0.9928513 | 35c9 docstring (`:1-8`) and CURRENT_STATE; cold replay 0.99386 | Provisional |
| Last-layer bigram+trigram ≈ −0.0152 on 640 | `t31L-l5tri-m64-1800s` 1.018985 (2,139 steps) vs `t31L-base-1800s` 1.034199 (2,191) → **−0.0152 @2048**; @1024-2M 1.03405 vs 1.049507 → **−0.0155** | **Verified**, time-matched at 1,795 s, single epoch, step cost −2.4% |
| Layers 1,−1 "large step-matched gain" | `t17-l1-2150` 1.007376 vs `t17-ctrl-2150` 1.019126 → −0.0118 step-matched, but **+6.2% wall time** (1,681.5 vs 1,583.6 s). Time-matched full budget: `t17-l1-m128` 1.004363 (2,135 steps, 1,794 s, 592.9M params) vs `t31L-l5tri-m64-1800s` 1.018985 → **−0.0146** | Verified. The fair basis is time-matched. Note that `t17-l1-m128` both adds layer-1 tables *and* doubles both layers to m128 (4 tables, 537M params), so the extra −0.015 over layer −1 m64 is the combination of the two, not layer 1 alone. Still within one epoch |
| Table size | `t31M-l5tri-m128` 1.019171 (2,067 steps) ≈ m64 (1.018985); d256 ≈ d128 (−0.0005) | Size beyond m64 at one layer did not help under dense sync, because it cost steps |
| Dense width 1152 ≈ 1024; dense depth 8 worse | **Not in supplied evidence.** Also, 35c9's MLP adapter asserts an SBUF budget that **width 1152 fails** (4·1152·4608 + 2·5760·512 + 2 MiB = 27.9 MiB > 24 MiB, `:4432`), so any 1152 result came from different code | **Unverifiable** |
| Table-LR cooldown worse | `20260923-ngram-ve-cooldown-lr-flag-only-default-shape`: 1.007831 vs 1.005734 (2,072/2,071 steps) → **+0.0021** | Verified (one run, 640 stack) |
| Attention scale 0.12 worse | 1.009764 vs 1.005734 → **+0.0040** | Verified (one run) |
| Double-hash tables worse/slower | 300-step smoke only: 1.171049 / 325 steps vs 1.165855 / 334 → +0.0052, −2.7% steps | **Not a valid negative.** The team's own rule says smokes mislead for tables. Status: untested at full budget |
| BF16/kernel opts helped 640 but did not beat dense | `bf16-sdpa-input` KEEP −0.0039 at +7.6% steps; `bf16-norm-output*` "CONFOUNDED" +0.010 to +0.015 at 2,346–2,716 steps | Verified numbers. **Mechanism correction:** those runs crossed the ~2,150-step epoch boundary with tables (§2.3 item 4), so they are not evidence that BF16 norm outputs harm quality. Dense uses `--bf16-norm-output` today |
| Short smokes mislead | `t31-ngram-fixed300` −0.031 step-matched vs `t31-bi3-m32d128-900s` −0.0016 time-matched; `weight-ema` smoke DISCARD while `t22-ema` full is −0.0001 | Verified |
| LNC1/world 8 | `t06-lnc1-8proc-noclipforeach-900s` **1.324435** at 1,088 steps vs LNC2 1.083798 at 1,080 steps; reproduced in two more runs (1.322584; 1.33769 with NKI off) | **A 0.24-BPB quality failure at matched steps that was never root-caused.** It matters for B1 (§8) |
| FP8 | Every `sub1-p2b-autocast-*` FP8 smoke scored 3.18–3.23 BPB (≈ no learning) or NaN; several crashed; `autocast-bf16-fullv2` +30% steps but +0.010 worse (2,719 steps, also past the epoch boundary, so confounded) | Full-model FP8: **closed** |

### 2.3 Corrections to the brief

1. **A1/A2 are not flag-only** (see §0.1). Static reading gives a certain `ValueError` at `setup_optimizer`,
   before compile. Confirm it in minutes with the G2 dry-run (§4.1) instead of spending a host slot. The standing
   mission's "exact flag audit" lists flags that the file *parses*. Parsing is not the same as running under
   `_install_submission_training`:

   | Flag arm | Runs on exact 35c9? | Reason |
   |---|---|---|
   | `--ngram-ve … --ngram-ve-opt rmsprop --ngram-ve-bf16` (A1, A2) | **No** | `make_plan` rejects `kind='rmsprop'` and BF16 params |
   | `--ngram-ve --ngram-ve-opt adamw --no-ngram-ve-bf16` | Yes, but a different arm | FP32 AdamW tables become *owned*; about a 1 GiB FP32 all-gather per step. Historically worse optimizer (`ngmveopt-arm` +0.002). Not recommended |
   | `--relu2-tau 0.5` | **No** | `assert train_module.RELU2_TAU == 0.0` (`:4404`) |
   | `--n-embd 1152`, `--mlp-ratio 5` | **No** | MLP adapter SBUF assert (`:4432`) |
   | `--mlp-sandwich-norm`, `--head-gate[-norm]`, `--x0-gate`, `--out-pool`, LR/WD/warmdown/final-LR flags, `--depth 8 --n-embd 768`, `--epoch-shuffle` | Yes by static reading | Still require the G2 dry-run |
   | `--adam-every-n >1` | No with `--clip-after-reduce` | Explicit `parser.error` (`:3555`) |

2. **Offset.** "Official ≈ public-1024 + 0.005" is correct for **public-2M** only (§2.4). Current selection uses 20M,
   where the best estimate is **−0.001 ± 0.002**. The practical effect: the team has been about 0.006 too pessimistic
   about where 0.99195 would land.
3. **Adoption threshold is inconsistent between governing docs.** CURRENT_STATE says a mean of ≤ −0.0008;
   STANDING_MISSION_V2 (later, human-approved 15:30 IST) says ≤ −0.001. Use −0.001 and record the supersession.
4. **The "data wall" explains most of the autoloop's "CONFOUNDED" verdicts.** On the 640 table stack:
   `qkfreeze-frozen-1800s` at 2,354 steps is **1.021278**, while the identical config capped at 1,648 s (2,150 steps)
   is **1.018967**, so +0.0023 from about 200 epoch-2 steps taken at the lowest LR. `epochshuf-frozen-1800s` at
   2,341 steps is 1.023799 (+0.0048). The autoloop's no-ngram-ve (2,548 steps), bf16-norm-output (2,716) and
   batch-ramp (2,397) runs all crossed the boundary and all got worse. The only KEEP, bf16-sdpa-input, stopped at
   2,233 steps. This mechanism was never written down, and it changes the value of throughput on dense (item 6).
5. **"TTTL" in the Recursive file is the attention window pattern** (`WINDOW_PATTERN = "TTTL"`, three windows of
   `seq/4` plus one long window, `recursive.py:867`), not test-time training. There is no TTT in that file. Any
   eval-time adaptation would also break the `load_for_eval` contract.
6. **The throughput→BPB exchange rate is unknown for dense.** On 640 within one epoch, 900 s → 1,800 s moved BPB
   1.0813 → 1.0343. That is −0.047 per doubling, or about **−0.0006 per +1% steps**. The two dense LR-0.015 runs agree
   (+0.5% steps, −0.00024), but that is inside noise. Dense spends its last 41% of steps on repeated data, where the
   640 evidence says marginal steps can be **harmful**. Until D1/FT (§4) measure it, do not rank kernel or FP8 work
   by the single-epoch rate.
7. **Recursive's 0.9109 is not comparable,** even directionally, to more than about ±0.02. Its eval context is 2048,
   not 1024; the 640 ledger shows 2048-context evals about 0.015 lower. It trains about 1 epoch in 5 min on a B200.
   Its **own** 10-seed SD is **0.0067** (`results/val_bpb.csv`: mean 0.91087, min 0.903891, max 0.922095), so its
   mean has an SE of about 0.002. Converted to a 1024-context estimate: about 0.926 ± 0.01.
8. **B1 (LNC1/world 8) has an unresolved correctness history,** not only an assert (§2.2, LNC1 row).
9. **Startup margin.** The budget clock charges `max(0, startup + 30 s − 900 s)` (`:3946-3950`). Dense cold startup of
   831–839 s leaves **31–39 s** before any charge, not the 61–69 s stated in CURRENT_STATE. Any graph growth
   (tables, B2) must be measured **cold**. Whether the official harness also enforces a hard wall-clock kill is
   unknown without `prepare.py`.

### 2.4 Public → official calibration (every recoverable pair)

| Pair | Official | Public-1024-2M | Public-1024-20M | Public-2048-2M | official − 20M | official − 2M |
|---|---|---|---|---|---|---|
| `dress-rehearsal-newbox` (code `0d8d856`, 2,192 steps) vs the 2026-09-10 23:59 PT submission of the same fix | **1.0548** | 1.049535 | **1.0557118** | 1.033421 | **−0.0009** | **+0.0053** |
| 1.1495 (capped 1,000-step submission) | 1.1495 | n/a | n/a | n/a | not recoverable | not recoverable |
| 1.0246 (config unrecorded) | 1.0246 | ? | ? | ? | not recoverable | not recoverable |

Caveats: the official and local numbers come from **different training runs** of identical code, so the pair includes
run-to-run noise (SD about 0.0003–0.0005). The 2048-context eval is 0.021 away from official, which strongly suggests
the scorer uses **context 1024** (the ledger note says the same). The 2M prefix scored 0.0062 *easier* than 20M on
this checkpoint.

A consistency check that stays a hypothesis: if 1.0246 came from the 592.9M-param table stack (public-1024-2M
1.0192–1.0204), then official − 2M ≈ +0.004 to +0.005, which matches the pair above. **Action:** the human records
the 1.0246 submission's commit. Every future submission becomes a calibration pair.

**Working rule:** official ≈ public-20M − 0.001 (±0.002). Compare only 20M to 20M, and 2M to 2M.

### 2.5 Noise model (measured)

| Setting | Replicates | SD |
|---|---|---|
| 640 table stack, 900 s, same commit/day | `ngm256-ctrl` ×4, `ngmbeta2-ctrl` ×3, `ngmveopt-ctrl` ×4 | 0.00025, 0.00012, 0.00034 |
| 640, 1,800 s | `t04-stack` ×3; `t04-defaults` ×2; `sub1` record pair | 0.00006; 0.00053; 0.00024 |
| Dense LR 0.015, 1,800 s, trn21 | 2 runs | \|Δ\| = 0.00024 |

Treat single-run SD as about 0.0003, so a candidate-minus-control Δ from single runs has SD about 0.0004. The rule
"both hosts negative, mean ≤ −0.001" is about a 3.5σ bar on the mean, which is appropriate given roughly 100 decisions
before the freeze. Eval-sampling noise on a *paired* Δ is unknown for 2M versus 20M. Measure it with the evaluator-only
calibration in §4.5.

---

## 3. Bottleneck diagnosis (ranked)

| Rank | Factor | Evidence | Size of lever |
|---|---|---|---|
| 1 | **Workflow latency and host utilization** | Across 2026-09-22/23 the DECISION_LOG shows about 23 results (about 14 full runs, about 7 smokes, 1 failed launch) over about 96 host-hours, roughly **15–20% utilization**. Disk-gate breakers (7.26–7.4 GiB) blocked hosts repeatedly; a stale queue drove the autoloop. | Largest controllable loss. The host budget to the freeze is about 2 × 121 h ≈ 240 host-hours, **≈200+ full-run slots**. Implementation and verification throughput is the real constraint. |
| 2 | **Repeated-data behaviour** | Dense runs about 1.7 epochs and the whole second epoch is inside cooldown. The 640 evidence shows epoch-2 low-LR steps hurting (§2.3.4). The dense epoch-2 value has never been measured. | Unknown sign. It sets the value of throughput and table size. Cheapest high-information measurements available: D1 and FT. |
| 3 | **Architecture capacity for n-gram memory (tables)** | −0.015 (layer −1) and −0.030 (layers 1,−1, m128) on 640 within one epoch. Recursive uses about 2.8B table params. | Largest known effect size, but it interacts with rank 2. |
| 4 | **Table optimization and system cost** | Dense gradient all-reduce plus dense RMSProp scale with table size: +2.4% time at 134M params, +6.2% at 268M, +13% at 537M on 640. On dense's 0.49 s/step the same absolute cost is about twice as large in relative terms. | Blocks Recursive-scale tables. B2 removes it, *if* rank 2 says steps are worth buying. |
| 5 | **Learning efficiency (optimizer, schedule)** | LR 0.015 gave −0.0009. Most 640 schedule sweeps were within ±0.001. Muon already has NorMuon-style row normalization, Polar Express, cautious WD and aspect-ratio LR scaling. | Small increments (≤0.001 each), cheap flag arms. |
| 6 | **Raw throughput** | MFU is about **22%** (estimate): about 541M FLOP/token × about 268k tok/s ≈ 145 TFLOP/s against 632–655 dense BF16 TFLOPS per chip. | Worth −0.0005 per 1% *only* if epoch-2 steps help. Pending D1. |
| 7 | **Compilation** | Cold startup 831–839 s against a 900 s cap minus 30 s margin. | A submission-safety risk, not a BPB lever. Every graph change needs a cold measurement. |
| 8 | **Evaluation noise** | Run SD about 0.0003; the 2M prefix is biased about −0.006 against 20M. | Controlled by replication and 20M-only selection. |

**Diagnostics that separate the hypotheses.** None needs a new mechanism. Q13 is answered here.

| Question | Cheap diagnostic | Reading |
|---|---|---|
| Undertrained or not? | D1: the control config with `--max-train-seconds T_e`, where T_e is the charged time at which the control log's `epoch` field turns 2. The schedule compresses to one epoch. | If BPB(1 epoch) − BPB(1.7 epochs) is < 0.004, epoch 2 is nearly worthless: favour capacity, tables frozen in epoch 2, fresh-tail order. If it is > 0.010, steps are valuable: favour throughput and B2. |
| Overfitting or memorization? | Evaluator-only, on existing checkpoints: BPB on training documents seen twice (the opening ~41% of epoch 1), documents seen once, and public val. Research script only; `prepare.py` untouched. Also look for a step-change in train loss at the `epoch 1→2` log line. | A gap between seen-twice and seen-once documents greater than about 0.02 means memorization in the cooldown. Tables will make it worse. |
| Optimizer mismatch? | Log per-group update-RMS/param-RMS every 50 steps, and gradient-noise scale from the four per-rank gradients before all-reduce (\|mean\|² against mean\|g_i\|²). Research build only. | B_noise ≫ 131k suggests larger batches are safe; ≪ 131k points to LR/batch. Update/param ratios by group show which LR is off. |
| Throughput bottleneck? | One `neuron-profile` capture of 20 steady-state steps (D2), plus host-side timing of `fetch()` and the per-step `.item()` syncs (`:3944`, `:3963`). | Rank ops by critical-path share. Host stall ≥3% makes an async loader a cheap, exactness-gated win (modded-nanogpt record #33 is precedent). |

---

## 4. Ranked experiment portfolio

### 4.1 Gates before any hardware (no host time)

| ID | Task | Owner | Time | Output |
|---|---|---|---|---|
| **G1** | Read `prepare.py` for: the `--num-train-shards` default; `evaluate_bpb` seq-len and token-prefix semantics; `TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS` and `TRAIN_STARTUP_STEPS_EXCLUDED` enforcement. On both hosts, count train shards and row groups and confirm they equal the default. | Operator (read-only) + Verifier | 15 min | Receipt. **If dev hosts hold fewer shards than an official `prepare.py setup`, every epoch-boundary conclusion here is void.** |
| **G2** | Preflight dry-run harness: 4-rank CPU/gloo, exact launch argv, real shapes. Import 35c9, run `_install_submission_training`, build the model, call `setup_optimizer`, run one synthetic fwd/bwd/step on a tiny batch. It must catch the A1/A2/relu2/SBUF failures listed in §2.3. | Implementer → Verifier | 1–2 h | Make it mandatory for every arm. It turns a 15–60 min wasted host slot into a 2-minute CPU check. |
| **IMPL-1** | **Table optimizer bypass** in 35c9 (≈30–50 lines). In the `setup_optimizer` wrapper, build `OwnedOptimizer` only over groups where `kind ∈ {adamw, muon}` and every param is FP32. Keep the remaining groups (`rmsprop` tables) replicated and step them after the owned step: every rank holds identical all-reduced grads, so every rank applies the same update. Leave `split_clip_params`/clip-exclude untouched. | Implementer → Verifier | 3–5 h incl. verification | Acceptance: (a) with `--no-ngram-ve`, the ownership plan JSON (`SUBMISSION_OPTIMIZER_INTEGRATION`) is byte-identical to 35c9's, and the first 50 logged losses match a 35c9 control to printed precision (steps 0–1 exactly); (b) gloo 4-rank: per-rank table SHA-256 identical after 5 steps and equal to a single-process reference that uses the AVG gradient; (c) a 50-step Neuron smoke is finite with per-rank table hashes equal; (d) `load_for_eval` strict-loads and gives deterministic logits. |
| IMPL-2 | Epoch-2 table freeze (`--ngram-ve-freeze-epoch 2`): from the step where `loader_state['epoch'] ≥ 2`, skip the table update, and skip the table all-reduce to recover time. | Implementer | 1 h | Prepared speculatively, used only if A1/A2 show a memorization signal |
| IMPL-3 | **Fresh-tail data order** (`--fresh-tail-frac f`, default 0.25). Per rank, split the row-group list into A (first 1−f) and B (last f), and iterate A, A, B. The lowest-LR tail then trains on never-seen documents. Same data, reordered; a data-sampling strategy is explicitly permitted. | Implementer | 2 h | Must include an identity-order control, because the requeue loader's `buffer_min=512` may differ from `prepare.make_dataloader`. Verify first-N-batch hashes. |
| B2 | Touched-row table update (in progress). Recommended **static-shape** design, below. | Implementer (Track B) | 1–2 days | Exactness against IMPL-1's dense path, then a speed run |

**B2 design recommendation.** modded-nanogpt's sparse comms (record #71) use variable-size `all_to_all`, which is a
poor fit for Neuron's static-shape compile. On Trn2, prefer a fixed-shape scheme:

1. `all_gather` each rank's `(idx[R], grad_rows[R,d])`, with R = 32×1024 tokens per rank per table.
2. `index_add_` into a persistent FP32 scratch buffer, then `g = scratch[idx] / world`. Duplicates see identical
   summed rows.
3. Compute the n-gram clip norm from the scratch buffer, so it matches dense exactly.
4. **Lazy RMSProp with catch-up decay:** `v[idx] ← β₂^(t − last[idx]) · v[idx]`, then the usual update, then
   `last[idx] = t` and `scratch[idx] = 0`. Duplicate writes carry identical values, so the result is deterministic.

With catch-up decay, lazy RMSProp is mathematically **identical** to dense RMSProp: untouched rows get zero update and
only decay v. Validation therefore becomes an equivalence test, not a new optimizer, and it is tolerance-based only
because of BF16 rounding. Traffic per table at m64 d128 is about 25 MB all-gather per rank, against about 200 MB for
a ring all-reduce.

### 4.2 Portfolio

Label convention: `20260924-<ID>-<slug>-<host>-r<n>`. The base is `35c9+IMPL-1` for table arms and exact 35c9 for
flag arms. Base command:

`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015`

Every scored run gets a fresh-process public-20M eval. Δ estimates are vs the same-host control, in public-20M units.

| Pri | ID | Exact causal change | Class | Base | Host | Control | Eval | Exp. Δ (est.) | 80% interval | P(Δ≤−0.001) | Effort | Host-h | Depends on | Abort | If positive | If negative |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A0 | none (controls) | control | 35c9 exact | both | n/a | 20M | 0 | ±0.0005 | n/a | none | 2 (+2 pooled later) | G1 | correctness | n/a | n/a |
| 2 | **A2** | `--ngram-ve --ngram-ve-layers 1,-1 --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude` replacing `--no-ngram-ve` | memory capacity | 35c9+IMPL-1 | trn22, then trn21 | A0 same host | 20M | **−0.008** | −0.017 … +0.003 | 0.60 | IMPL-1 | 2 | IMPL-1, G2 | NaN, over-budget, startup >870 s | Adopt after two-host rule; → A3/A4 via B2; retune LR | Check the epoch-boundary train-loss drop → A2f (IMPL-2) once; if also ≥0, stop tables |
| 3 | **A1** | same flags with `--ngram-ve-layers=-1` | memory capacity | 35c9+IMPL-1 | trn21, then trn22 | A0 same host | 20M | **−0.005** | −0.013 … +0.003 | 0.60 | IMPL-1 | 2 | IMPL-1, G2 | as A2 | Superseded by A2 if A2 wins by more | A1f once, then stop |
| 4 | **D1** | control config with `--max-train-seconds T_e` (≈ time of first `epoch 2` log line) | diagnostic (repeated data) | 35c9 exact | either | A0 same host | 20M | +0.004 … +0.015 (a measurement) | n/a | n/a | none | 0.7 | A0 log | correctness | Small gap: epoch 2 low-value → prioritise capacity, IMPL-2, FT; demote throughput | Large gap: steps valuable → B2 and throughput move up |
| 5 | **FT** | IMPL-3 `--fresh-tail-frac 0.25` | data order / repeated data | 35c9+IMPL-3 | trn21, then trn22 | identity-order control on the same loader | 20M | **−0.002** | −0.006 … +0.002 | 0.45 | 2 h | 2 (+1 control) | IMPL-3, G2 | correctness | Adopt; also applies under tables | Drop; record that tail freshness does not matter |
| 6 | A2f / A1f | the better table arm + IMPL-2 epoch-2 freeze | repeated data × memory | 35c9+IMPL-1+2 | cross | the unfrozen arm and A0 | 20M | −0.002 vs the unfrozen arm | −0.006 … +0.003 | 0.45 | 1 h | 2 | A1/A2 result | as A2 | Adopt freeze policy | Tables are not memorization-limited |
| 7 | B2-X / B2-F | touched-row update, exactness then full run at the adopted table layout | systems (table sync) | adopted table base | alternate | IMPL-1 dense path, same host | 20M | −0.002 … −0.003 (steps only) | −0.005 … +0.001 | 0.55 | ongoing | 1.5 | A2 adopted, native probe | exactness fail | Enables A3/A4 | Cap tables at A2 size |
| 8 | A3 | Recursive placement: tables on every VE layer (1,3,5) | memory placement | B2 base | cross | adopted base | 20M | −0.003 | −0.008 … +0.003 | 0.50 | flags | 2 | B2 | as A2 | → A4 | Keep 1,−1 |
| 9 | A4 | `--ngram-ve-table-mult 128` (then 256) at the best placement | memory size | B2 base | cross | adopted base | 20M | −0.002 | −0.006 … +0.002 | 0.45 | flags | 2 | B2 | cold startup >870 s | Next size | Stop size scaling |
| 10 | F1 / F2 | `--matrix-lr 0.0175` / `0.0125` | optimizer | 35c9 exact (re-bracket after adoption) | one each | A0 | 20M | −0.0003 | −0.0015 … +0.001 | 0.30–0.35 | none | 1 each | none | correctness | Replicate cross-host | Close bracket |
| 11 | F3 | `--warmdown-ratio 0.85` (0.9 only if 0.85 wins) | schedule | 35c9 | any | A0 | 20M | −0.0005 | −0.002 … +0.001 | 0.40 | none | 1 | none | correctness | Try 0.9 | Stop |
| 12 | F5 | `--embedding-lr 0.45` (0.6 only if 0.45 wins) | optimizer | 35c9 | any | A0 | 20M | −0.0004 | −0.0015 … +0.001 | 0.40 | none | 1 | none | correctness | Try 0.6 | Stop |
| 13 | F6 | `--mlp-sandwich-norm` | norm placement (Recursive `:235`) | 35c9 | any | A0 | 20M | −0.001 | −0.004 … +0.002 | 0.40 | none | 1 | G2 | correctness | Replicate | Stop |
| 14 | F7 | `--weight-decay 0.35` | regularization for multi-epoch ([2509.14786]) | 35c9 | any | A0 | 20M | −0.0005 | −0.002 … +0.001 | 0.35 | none | 1 | none | correctness | Try 0.5 | Stop |
| 15 | F8 | `--final-lr-frac 0.0` | schedule, decay-to-zero ([2502.15938]) | 35c9 | any | A0 | 20M | −0.0003 | −0.0015 … +0.001 | 0.35 | none | 1 | none | correctness | Replicate | Stop |
| 16 | D2 | `neuron-profile` of 20 steady-state steps, plus host-side `fetch()`/sync timing | diagnostic (throughput) | 35c9 exact | either, 30 min | n/a | none | n/a | n/a | n/a | none | 0.5 | none | n/a | If host stall ≥3%: async loader (exactness-gated) | No kernel work |
| 17 | F9 | `--epoch-shuffle` (after verifying epoch-1 batch identity) | data order | 35c9 | any | A0 | 20M | 0.000 | −0.002 … +0.002 | 0.35 | none | 1 | G2, identity check | correctness | Compare against FT | Stop |
| 18 | F10–F12 | `--head-gate` / `--x0-gate` / `--out-pool` (separately) | gating/pooling (Recursive `:203,:597,:619`) | 35c9 | idle-fill only | A0 | 20M | +0.0005 | −0.002 … +0.004 | 0.20–0.25 | none | 1 each | G2 | step time <−8% | Replicate | Stop (head gate was −20% steps on 640) |
| 19 | SC-1 | declared stack confirmation: `--depth 8 --n-embd 768` + adopted tables + adopted flags | capacity allocation | adopted base | cross | adopted base | 20M | −0.001 | −0.006 … +0.004 | 0.35 | none | 2 | ≥1 table adoption | correctness | Adopt | Keep 6×1024 |

**Expected gain per host-hour (rough EV = P × |Δ| / host-h):** A2 ≈ 0.0024/h, A1 ≈ 0.0015/h, FT ≈ 0.0005/h,
A3/A4 ≈ 0.0006/h (conditional), flag bank ≈ 0.0001–0.0004/h. Diagnostics D1/D2 are worth more than their EV
suggests because they decide the ordering of everything else. **Recommended order:** G1 → G2 + IMPL-1 → A0 → (D1, F-bank
while IMPL-1 is verified) → A1 ∥ A2 → replicates → FT and A1f/A2f → B2 → A3/A4 → retune.

### 4.3 Recursive recipe: source-level transfer matrix (Q2)

Recursive: `optimized_from_karpathy.py`, Apache-2.0 (derived from MIT nanochat), commit `a962ec43`.
Harness `solutions/lib.py`: `MAX_SEQ_LEN=2048`, `TIME_BUDGET=300`, `EVAL_TOKENS=40*524288`, `VAL_SHARD=6542`.
This matches Karpathy's `autoresearch/prepare.py`, whose default is **10 training shards**.

| Technique | Recursive source | 35c9 / dense status | Transfer verdict |
|---|---|---|---|
| Hashed bigram value tables on **all** VE layers; trigram on first, second-to-last and last VE layers; `64×vocab` rows; **K=2 factored half-width tables with independent hashes**, concatenated to full `kv_dim` (768) | `:156`, `:272-313`, `:594-614` | Team: dim-128 tables plus a Muon-trained up-projection, single hash, layers chosen by flag. Recursive's total is about **2.8B BF16 table params** (7 layer-tables × 2 × 524,288 × 384) against a team maximum of 537M | **Most likely to transfer (mechanism).** Its scale does not transfer without B2: dense sync costs about +3% time per 134M params. The K=2 multi-hash idea was only smoke-tested here, so it is untested |
| RMSProp for tables, β₂=0.999, flat LR (`NGRAM_WARMDOWN_RATIO=0`), late β₂ ramp to 0.9999 | `:672-679`, `:878`, `:883`, `:1009` | Present (`--ngram-ve-opt rmsprop`, `--ngram-ve-flat-lr`). On 640: β₂ 0.95 worse, 0.99 ≈ same; table LR 0.15–0.6 flat; cooldown worse | Already transferred. The β₂ ramp has low expected value |
| Per-table gates, zero-init (2σ(0)=1), reading disjoint 32-channel slices | `:142-172`, `:176-192` | Present (`ve_gate_bi`, `ve_gate_tri`, `:1628-1629`) | Transferred |
| MLP output sandwich norm `x + norm(mlp(norm(x)))` | `:235` | Flag `--mlp-sandwich-norm`; runs under the adapter | Cheap test (F6); 640 evidence confounded by the data wall |
| ReLU²(h − 0.5) | `:217` | Flag exists; **crashes under the 35c9 adapter** | Needs a custom-backward change; low EV; park |
| Per-head output RMSNorm plus 2σ head gate | `:200-204` | `--head-gate --head-gate-norm`; −20% steps on 640 | Low; idle-fill only |
| x0 gate 2σ(s·mean(x)) | `:597` | `--x0-gate`; −11% steps on 640 | Low |
| Output pooling over the last 3 layers | `:592-625` | `--out-pool`; +0.0014 on 640 at 900 s | Low |
| Window pattern **TTTL** (3 × seq/4 plus 1 long), last layer long | `:867`, `_compute_window_sizes` | Not flag-exposed (`WINDOW_PATTERN='L'` constant) | Mainly an FA4 speed trick at 2048 context. At T=1024 with compiler SDPA there is no speed benefit unless the kernel skips blocks. **Do not port** |
| RoPE base 1e6; softcap `16.5·tanh(x/15)`; `wte` init std 1.0 | `:374`, `:631`, `:326` | Not flag-exposed (1e5 / 15 / 0.8) | Tiny at T=1024; park |
| Muon: Polar Express, NorMuon row normalization, cautious WD mask, √(rows/cols) LR scaling | `:682-728`, `:838` | Present in 35c9/116188 (`_muon_math`, `:2865`) | Already transferred |
| Muon momentum quadratic decay to 0.79 and β₂ ramp 0.95→0.97 in warmdown; Adam Demon β₁ 0.8→0.55; separate Adam warmdown 0.65 | `:1000-1050`, `:1115-1170` | Flags: `--muon-beta2(-final)`, `--demon-beta1` (DISCARD on 640), momentum cooldown | Low; harness-tuned |
| Muon warmdown **0.95**, final LR 0.05 | `:881`, `:884` | `--warmdown-ratio` (0.75) | F3 tests 0.85, then 0.9 |
| WD "pulses" (5× rectangular at 1.5–4% progress, triangular at 80%) | `:1052-1095` | Absent | **Do not port.** Overfit to a 5-minute, 10-seed harness |
| Whole model in BF16, no FP32 master weights | `:932` | FP32 params plus BF16 compute | Risky with Muon on long runs; skip |
| Depth 8 × 768; batch 147,456 tokens/step at 2048 context | `:867-887` | 6×1024, 131,072 tokens/step at 1024 context | SC-1 tests 8×768 only once tables are adopted. Batch is already comparable |
| Multi-token prediction removed | `:251` | Absent | Agrees with the team's step-budget view |

**Hardware assumptions that do not transfer:** FA4 sliding-window kernels (`:30-91`); `mode="max-autotune"` compile;
a single-GPU dense-gradient embedding backward on 192 GB (Trn2 per LNC2 core has 24 GiB, and dense grad plus state
for 2.8B BF16 params would need ~17 GB per rank *replicated*); and a 5-minute, single-epoch data regime.

### 4.4 External ideas register (Q2, Q8–Q10)

| Idea | Primary source (license) | Mechanism | Why it might transfer | Why it might fail | Trn2 difficulty | Correctness risk | Throughput | BPB (est.) | Host-h | Cheapest falsifier |
|---|---|---|---|---|---|---|---|---|---|---|
| Hashed n-gram value tables | Recursive (Apache-2.0); modded-nanogpt #62 bigram hash (MIT); Engram arXiv:2601.07372; Over-Tokenized Transformer arXiv:2501.16975 | O(1) lookup memory for local n-gram statistics | −0.015 to −0.030 measured on 640 | Epoch-2 memorization; step cost | Low with IMPL-1 | Low (causal by construction, `:1998-2043`) | −2% to −8% | −0.004 to −0.010 | 4 | A1/A2 single-host screen |
| Sparse table gradient communication | modded-nanogpt record #71 (MIT), `train_gpt.py` `sparse_comms_*` | Send only touched rows | Removes the size-linear cost | Neuron static shapes rule out variable all-to-all; use the fixed all-gather | Medium | Medium (duplicates, AVG, clipping) | +2% to +8% at A2 | −0.002 (steps) plus scaling | 2–3 | B2-X exactness, 200 steps |
| Data-constrained regularization (higher WD) | Kim et al. arXiv:2509.14786 | Suppress overfitting across epochs | Dense is at 1.7 epochs | 640 WD sweep was flat within one epoch | None | None | 0 | −0.0005 | 1 | F7 |
| Repeated-data scaling | Muennighoff et al. arXiv:2305.16264; Hernandez et al. arXiv:2205.10487 | Up to ~4 epochs is nearly as good as fresh data, but repeated subsets can hurt disproportionately | Frames D1/FT | The 640 data shows harm when the repeat lands in the low-LR tail | None | None | 0 | via FT | 2 | D1 |
| Decay-to-zero LR | Bergsma et al. arXiv:2502.15938 | Linear D2Z beats a 10× decay floor | FINAL_LR_FRAC is 0.05 | 640 `finallrfrac-002` was flat | None | None | 0 | −0.0003 | 1 | F8 |
| NorMuon / Polar Express | arXiv:2510.05491 / arXiv:2505.16932 | Row-normalized orthogonalized updates / optimal polar iterations | Already in both files | n/a | done | n/a | n/a | 0 | 0 | n/a |
| FP8 matmul (double-FP8 TensorE) | Neuron Trn2 arch guide (`nki/guides/architecture/trainium2_arch.rst`: 158 FP8 vs 79 BF16 dense TFLOPS per core, `perf_mode=double_row`); DeepSeek-V3 fine-grained FP8 arXiv:2412.19437; modded-nanogpt FP8 head | 2× TensorE throughput on eligible matmuls | 22% MFU; 1024-wide matmuls | TensorE is probably not the bottleneck; full-model FP8 learned nothing (§2.2); no backward exists; the value of steps is unknown | High | High | +0–10% (est.) | 0 to −0.003 | 6+ | **Park** (§8) |
| Async data loading and fewer host syncs | modded-nanogpt record #33 (MIT) | Overlap host packing with device work | The loader (prepare's `make_dataloader`; 35c9's copy at `:37-86`) packs best-fit rows in Python on the training thread, and the loop forces `.item()` each step | Gain is 0 if the device is the bottleneck | Low | Low if batch identity is proven | +0–5% | 0 to −0.002 | 1 | D2 host timing |
| Multi-hash K=2 factored tables | Recursive `:272-313`; BLT-style hashing | Reduce collision damage | Mechanism is plausible | 300-step smoke negative; −2.7% steps | Low | Low | −3% | ±0.002 | 2 | Only after B2 |
| Window attention / RoPE base / TTT | Recursive; Parameter Golf TTT entries | Speed (FA4) or eval-time adaptation | n/a | T=1024 already; **TTT violates the pure-forward `load_for_eval` contract** | n/a | **Rules** | n/a | n/a | 0 | Do not run |

### 4.5 Decision rules and when to use 2M (Q5)

- **Always evaluate full runs at public-20M.** Evaluation is forward-only and costs a few minutes against a
  45–55 min run slot. The 2M prefix saves little and was 0.006 easier on the one measured checkpoint. So the
  "2M screen" buys almost no host time while adding bias and variance. **Challenge to the workflow:** the expensive
  screening decision is whether to *train*, not how much to *evaluate*.
- **2M is actively misleading** for any change whose effect depends on document mix or rare contexts: tables, data
  order (FT, epoch-shuffle), context and positional changes. It is also misleading for any Δ smaller than about 0.001,
  and for any absolute comparison against a 20M number.
- **One evaluator-only calibration job now:** evaluate the two retained LR-0.015 checkpoints and one control at both
  2M and 20M. This measures the prefix bias and paired-Δ agreement for dense at zero training cost.
- **Nominate** for a cross-host replicate when a single-host 20M Δ ≤ −0.0006, steps are within 10% (or it is a
  declared throughput arm), and every correctness gate passes.
- **Adopt** when both hosts are negative, the mean is ≤ −0.0010, and |Δ21 − Δ22| ≤ 0.0015 (otherwise run a third
  replicate). The controls must be current, same host, same base bytes.
- **Discard** when a single-host Δ ≥ +0.0005, or the two-host mean is > −0.0003. The ambiguous band gets one
  replicate only for high-prior arms (tables, FT).
- **Throughput-only arms** (B2, async loader) need an exactness gate (steps 0–1 identical; first 200 losses within
  1e-4 relative; parameter hashes identical when numerics are unchanged) plus ≥2% back-to-back same-host step-time
  gain. A BPB run is then required because of the epoch-2 question.
- **Control refresh:** a new control that differs from the pooled host mean by more than 0.0010 triggers an alarm
  and a rerun. Pool at least 2 controls per host per base.
- **Big-win hold:** Δ ≤ −0.010, or any public-20M < 0.975, puts the candidate on hold for an independent cold replay
  and audit before anyone reports it.

---

## 5. Two-host schedule (Q4, Q6, Q7)

### 5.1 Rules

1. One run per host. Comparisons are same-host. **Screen on one host, replicate on the other.** Two different
   high-prior arms run in parallel (A1 on trn21, A2 on trn22), then swap hosts for the replicates.
2. Each host keeps at least 2 pooled controls per base, refreshed after 8 runs or 12 h, whichever comes first,
   and interleaved as ABBA around high-prior arms.
3. Research runs use a persistent warm compile cache (§6.5), which cuts startup from about 14 min to minutes and
   gains about 25% more slots. Every submission candidate additionally gets a **cold** compile measurement.
4. Hosts never idle more than 15 minutes. The idle-queue policy is §5.4.
5. Before every launch: G2 dry-run pass, disk gate, holder check, clean git, exact bytes hash.

### 5.2 Next 48 hours (IST; one slot ≈ 50–55 min cold, 40–45 min warm)

Implementation lanes run in parallel with host work. **L1:** IMPL-1 → verify (target ready by Sep 24 23:00).
**L2:** G2 dry-run harness (by 20:00). **L3:** R1 cleanup of 116188 plus the equivalence harness (§7) (by Sep 25
09:00). **L4:** IMPL-2 and IMPL-3 (by Sep 25 06:00). **L5:** B2 native probe, then the static-shape exactness test
(ongoing).

| Time (IST) | trn21 | trn22 | Decision point |
|---|---|---|---|
| Sep 24 17:00–18:00 | G1 read-only checks; finish whatever is running | Finish the 116188 clean control → 20M eval | G1 receipt |
| 18:00–19:00 | **A0-C21a** 35c9 control | **A0-C22a** 35c9 control | |
| 19:00–20:00 | **D1** epoch-value run (T_e from C21a's log; shorter) | **F1** `--matrix-lr 0.0175` | |
| 20:00–21:00 | **F6** `--mlp-sandwich-norm` (after G2) | **F3** `--warmdown-ratio 0.85` | D1 result: re-rank FT and throughput |
| 21:00–22:00 | **F5** `--embedding-lr 0.45` | **F7** `--weight-decay 0.35` | |
| 22:00–23:00 | **A0-C21b** control (pooled; also on IMPL-1 bytes with `--no-ngram-ve` if IMPL-1 is ready; this is the equivalence evidence) | **F2** `--matrix-lr 0.0125` | IMPL-1 verifier receipt |
| Sep 24 23:00–Sep 25 00:00 | **A1** screen | **A2** screen | |
| 00:00–01:00 | replicate of the best nominated F-arm from trn22 | replicate of the best nominated F-arm from trn21 | A1/A2 single-host Δ |
| 01:00–02:00 | **A2** replicate (cross-host) | **A1** replicate (cross-host) | |
| 02:00–03:00 | **A0-C21c** control (ABBA) | **A0-C22b** control | **Table decision #1 (≈03:00)** |
| 03:00–05:00 | If a table arm is adopted: **A2f** (epoch-2 freeze) screen, then a new-base control. If not: **A1f** once, then **FT** | If adopted: new-base control ×1, then **FT** on the new base. If not: **FT** | |
| 05:00–07:00 | FT replicate / D2 profile (30 min) + F8 | cross replicate / F9 (if identity is verified) | FT decision |
| 07:00–10:00 | **R1 lockstep** (100-step equivalence, ~20 min), then **R1 cold replay #1** | New-base F-bank re-screens (the matrix LR bracket on the table base) | **S0 go/no-go (≈10:00)** |
| 10:00–12:00 | **R1 cold replay #2** | re-screens | S0 package to the human |
| 12:00–18:00 | B2-X exactness (200 steps) → **B2-F** full run at the adopted layout | Re-bracket winners; second new-base control | B2 exactness receipt |
| 18:00–24:00 | **A3** (all VE layers via B2) screen → replicate on trn22 | **A4** (mult 128) screen → replicate on trn21 | Table-size decision (≈Sep 26 02:00) |
| Sep 26 00:00–06:00 | Replicates, then controls on the updated base | Replicates, then controls | |
| 06:00–12:00 | **Clean port of the adopted table stack**: lockstep, then cold replay #1 | Warmdown/LR retune on the new base | |
| 12:00–18:00 | Clean cold replay #2 | SC-1 (8×768 + tables) screen | **S1 package (≈18:00)** if the clean table stack is ≥0.004 better than S0 |

If IMPL-1 slips, trn21 and trn22 continue the F-bank and FT in that order. Nobody waits.

### 5.3 Day-by-day to the freeze

| Day (IST) | Science | Systems | Submission track |
|---|---|---|---|
| **Sep 24** | Controls, D1, F-bank start | G2, IMPL-1, IMPL-2/3 | R1 cleanup begins |
| **Sep 25** | A1/A2 decision (~03:00); FT; epoch-2 freeze test; re-bracket on the new base | B2 exactness | **S0** cold-replayed; human decides (recommended: submit) |
| **Sep 26** | B2 scaling (A3/A4); SC-1 | B2-F | Clean port of the table stack; **S1** decision in the evening |
| **Sep 27** | Final retune (warmdown, LR, table size); two-host confirmations only for the top two changes | Cold-compile measurement of the best stack | Clean port within 12 h of the research best |
| **Sep 28** | **Stack confirmation by 12:00** (two hosts × 2 replicates) | none new | S2 package by 12:00; clean cold replays ×2 |
| **Sep 29 until 18:00** | No new mechanisms after 06:00. Replicates and cold replays of the final candidate only | none | Final package frozen at 18:00 |
| **Sep 29 18:00 → Oct 1 12:29** | none | none | Human submits. **Recommended by Sep 30 12:00 IST** to leave time for one retry if review rejects |

### 5.4 When the main queue is empty (Q7)

Priority order, applied per host:

1. A nominated replicate waiting for this host.
2. A control refresh, if the pooled-control rule is due.
3. The next F-bank arm (G2-passed and pre-approved) in §4.2 order.
4. A cold replay of the current submission candidate (evidence for readiness).
5. A diagnostic: D2 profile, memorization eval, the 2M/20M calibration job.
6. A base replicate to tighten the control mean.

Never seed-hunt. Never re-run an old-640 arm.

---

## 6. Autonomous workflow design (controller v2) (Q15)

### 6.1 State machine

```
PROPOSED ─verify→ PROPOSAL_VERIFIED ─(flag-only)→ PREFLIGHT_PENDING
                              └─(code)→ IMPLEMENTING → IMPL_SUBMITTED ─verify→ IMPL_VERIFIED → PREFLIGHT_PENDING
PREFLIGHT_PENDING ─G2 dry-run + static gates→ READY ─host lease→ LAUNCHING → RUNNING
RUNNING ─exit→ DRAINING ─group empty + no /dev/neuron holders→ QUIESCED ─hash→ COLLECTED
COLLECTED ─eval job→ EVALUATING → EVALUATED ─archive+verify→ ARCHIVED ─rules→ COMPARED
COMPARED → {NOMINATED → (replicate child) | ADOPT_PENDING_HUMAN_ACK | DISCARDED | CONFOUNDED | FAILED}
ADOPTED → CLEAN_PORTING → CLEAN_PORT_VERIFIED → CLEAN_REPLAYED(x2) → SUBMISSION_READY (human acts)
Any state ─correctness failure→ QUARANTINED (breaker; human reset only)
```

Transitions are idempotent and keyed by `(label, transition, input_hash)`. A crashed label is never relaunched;
a retry is a new label `…-r2` with a diagnosis attached.

### 6.2 Durable state schema

Storage: an append-only, hash-chained `events.jsonl` (each event carries `prev_sha256`), plus a materialized
`state.json`, both on the controller workstation and committed to an evidence branch every ≤15 min. The queue schema
is in `reports/2026-09-24-queue-v2.json`.

```json
{
  "label": "20260924-A2-ngram-l1m1-m64d128-trn22-r1",
  "state": "READY",
  "kind": "table_arm | flag_arm | control | diagnostic | clean_replay | throughput_arm",
  "base": {"file": "train.py", "sha256": "35c9…+IMPL-1 sha", "commit": "…"},
  "argv": ["--n-embd","1024", "..."],
  "env": {"NEURON_LOGICAL_NC_CONFIG":"2", "NEURON_CC_FLAGS":"--optlevel=1 --auto-cast matmult --auto-cast-type bf16", "NEURON_COMPILE_CACHE_DIR":"/var/neuron-cache/research"},
  "host": "trn22", "control_labels": ["…C22a","…C22b"], "eval": {"tokens": 20971520, "seq_len": 1024},
  "gates": {"proposal_receipt":"sha", "impl_receipt":"sha|null", "g2_dryrun":"sha", "disk_free_bytes": 0, "holders": []},
  "run": {"pgid": 0, "sid": 0, "cgroup": "exp-…", "started_utc": "", "exit_code": null},
  "artifacts": {"log_sha256":"", "ckpt_sha256":"", "ckpt_archive_uri":"", "eval_json_sha256":""},
  "metrics": {"steps": null, "training_seconds": null, "startup_s": null, "over_budget": null, "public_20m": null, "causality_max_abs": null},
  "compare": {"delta": null, "controls_mean": null, "rule": "v2-2026-09-24", "verdict": null},
  "parents": [], "children": [], "human_ack": null
}
```

### 6.3 Locking and crash recovery

- **Controller singleton:** `flock` on `/var/lock/trn-controller.lock` plus a lease record (`holder`, `expires_utc`,
  renewed every 60 s) in `state.json`. A second controller refuses to start while the lease is live.
- **Host lease:** atomically create `/var/lib/trn-exp/lease` with `O_CREAT|O_EXCL`, containing
  `{label, controller_id, pid, expires}`. It is acquired only after `pgrep -f 'torchrun|train.py|run_experiment.py'`
  returns nothing and `fuser /dev/neuron*` is silent. It is released only from QUIESCED.
- **Recovery on restart:** for each label in LAUNCHING, RUNNING or DRAINING, look up its cgroup and pgid on the host.
  If alive, resume monitoring. If dead, run the quiescence proof, then classify from log markers and `final.pt`
  (TRAINING_DONE or FAILED). Never guess. Unknown means QUARANTINED.

### 6.4 Process containment and quiescence (keeps the repaired-launcher gates; never weaken them)

1. Launch in a **dedicated cgroup and session**:
   `systemd-run --unit=exp-<label> --slice=trn-exp.slice --property=KillMode=control-group --collect -- setsid torchrun …`.
   Descendants can escape a process group with `setsid`, but an unprivileged process cannot leave its cgroup. The
   process group stays as a second layer.
2. Stop sequence: `SIGTERM` to the cgroup (and `-PGID`), wait 60 s, `SIGKILL` to the cgroup, then poll until
   `cgroup.procs` is empty, no process has the session SID, `fuser /dev/neuron*` is empty, and no process has CWD or
   open files under the run directory (`lsof +D`). If anything survives 120 s: **QUARANTINED + breaker**.
3. Quiescence receipt, taken *before* hashing: timestamps, the empty listings above, and `stat` mtimes of evidence
   files taken twice 10 s apart and required to be unchanged. Hash `final.pt`, the log and the eval JSON only after
   the receipt exists.

### 6.5 Checkpoint, disk and compile-cache policy

- **Checkpoint:** rank-0 `final.pt` (≈0.5 GB dense, ≈1.0 GB at A2) → SHA-256 → copy to a content-addressed archive
  (`archive/sha256/<hash>` on the controller workstation; an existing S3 bucket is fine if a human has already
  provisioned one) → verify the hash remotely → delete on the host, **except** the current pooled controls' latest
  checkpoint and the current submission candidate (at most 2 per host).
- **Disk gates (unchanged):** 6.5 GiB launch floor; about 5 GiB hard floor. Add a predictive gate: free ≥ expected run
  footprint (checkpoint + logs + cache growth, per `kind`) + 5 GiB. **Human action strongly recommended:** grow each
  host's EBS volume by 50–100 GiB. It is the single cheapest fix for repeated breaker trips, and it is AWS lifecycle,
  so human-only.
- **Compile cache:** research uses a persistent `NEURON_COMPILE_CACHE_DIR=/var/neuron-cache/research`, recorded in the
  receipt, with LRU pruning to ≤X GiB performed only while the host is QUIESCED. Submission candidates additionally
  run with a fresh empty cache dir (as the official environment has no cache), and startup must be ≤ 840 s (30 s
  headroom under 870). Never set `NEURON_CC_FLAGS` externally without `--optlevel=1` (protocol §1.5).

### 6.6 Git and artifact store

- Branches: `research/35c9-v2` (IMPL commits, one per implementation, verifier receipt SHA in the message);
  `exp/<label>` for code arms; `evidence/v2` (append-only: proposal, receipts, launch, log.gz, eval JSON, compare
  JSON, verdict); `submission/candidates` (only the clean `train.py` plus `launch-command.txt`; each candidate
  tagged `sub-cand/<date>-<sha8>`).
- The controller commits evidence after each transition batch and pushes with 4× exponential backoff. **No force
  pushes, no history rewrites, no checkpoints in Git.** A merge to `submission/candidates` needs a human-approved PR.

### 6.7 Summaries, alerts and circuit breakers

- **Six-hour summary** (committed and sent): utilization per host; runs by verdict; best confirmed public-20M; pending
  human decisions; disk; breaker state; controls drift; top-3 next queue items.
- **Alerts** (immediate): milestone reached; ADOPT pending human ack; submission-ready; any correctness failure
  (NaN, `over_budget`, causality ≠ 0, dirty git, hash mismatch, surviving process); disk below 6.5 GiB; host idle
  >15 min; control drift >0.0010; big-win hold; cold startup >840 s.
- **Breakers:** 2 consecutive launch failures on one host → block that host. Any correctness failure → pause all
  code arms (flag arms may continue on the other host only if the failure is host-local and diagnosed). Disk floor →
  block the host. Three FAILED verdicts in 6 h → global pause.

### 6.8 What still requires a human

Leaderboard submission; any AWS mutation (EBS resize, instance stop/start, new buckets); breaker reset after a
correctness failure; acknowledging an ADOPT that changes the base; merges to the submission branch; changes to
decision thresholds; anything touching credentials, `.env` or PEM files.

### 6.9 Controller loop (pseudocode)

```python
while lease.renew():
    for host in hosts:
        h = probe(host)                                   # procs, holders, disk, lease, cgroup
        if h.busy: monitor(h.label); continue
        if h.draining: prove_quiescence_or_quarantine(h); continue
        if not gates_ok(h): alert_if_idle(host, 15*60); continue
        job = next_job(host)                              # §5.4 priority; same-host control constraint
        if job is None: alert_if_idle(host, 15*60); continue
        assert job.g2_receipt and bytes_hash(job) == job.base.sha256
        launch_in_cgroup(host, job)                       # records pgid/sid/cgroup before returning
    for label in collected_not_evaluated(): launch_eval(label)   # fresh process, 20M, same host
    for label in evaluated(): archive_verify_delete(label); compare(label); decide(label)
    commit_and_push_evidence(); maybe_six_hour_summary()
    sleep(30)
```

---

## 7. Submission pipeline (Q14)

### 7.1 Build the clean file

Port by **specialization**, not by rewriting:

1. Start from the exact research bytes plus the adopted flags.
2. Replace each parsed flag with its constant.
3. Delete branches that are unreachable under those constants.
4. Delete unused functions, flags and validations.
5. Put the training-only mechanisms (custom MLP backward, CE kernel, optimizer ownership, table update) **inside the
   model and optimizer classes** that `load_for_eval` also uses, so there is one forward.

Tables port as ordinary `nn.Embedding` weights plus their gates and projections. The table update code is
optimizer-side and never runs in eval.

### 7.2 Prove equivalence (catches arithmetic drift)

| Level | Test | Pass condition |
|---|---|---|
| Static | Dump model config, `state_dict` keys/shapes/dtypes, optimizer group structure (kind, lr, betas, eps, wd, order of param shapes), and the ownership plan from both files at seed 42 on CPU | Byte-identical JSON |
| Init | SHA-256 of all initial parameters after `init_weights()` (CPU, seed 42) | Identical |
| Lockstep | Both files for 100 steps on the same host, same seed and data, `--no-eval-public` | Steps 0–1 losses bit-identical. Steps 2–99 within 1e-4 relative (wall-clock WD/LR schedules add timing jitter). Per-step `dt` within 3% |
| Full | Two **cold** replays of the clean file: fresh cache dir, the bare launch command exactly as submitted, no extra env except `NEURON_LOGICAL_NC_CONFIG` | Both public-20M within ±0.0008 of the research mean; startup ≤ 840 s; `over_budget=False`; finite checkpoint; `load_for_eval` strict load |
| Timing | Clean candidate within 12 h of the research best (mission rule) | Tracked in state |

### 7.3 Reviewer-style audit checklist (run by an independent Verifier)

1. Exactly one forward path used by training and `load_for_eval`, with no `self.training`-dependent math.
2. No reads of `public_val` or any val split in the training script. Move the in-process eval and causality check
   into research tooling.
3. No monkeypatching (`types.MethodType`, reassigning class methods at import or `__main__`).
4. No names or strings that suggest evaluation special-casing ("adapter", "probe", "Eval", "submission").
5. No flags that advertise absent features. The launch command is as bare as possible.
6. No environment-dependent behaviour between training and eval, beyond documented compiler precision.
7. `torch.load(weights_only=True)`, `strict=True`.
8. No network, no file writes outside `out_dir` and the compile cache.
9. The docstring states the full chip (LNC2 × 4), the 1,800 s budget logic, and that tables are causal n-gram lookups
   of past tokens only.

### 7.4 Audit of the current clean candidate `116188` (findings)

Numerics-neutral cleanup. Call it "R1". Fix it, prove it with §7.2, then submit (S0).

| Lines | Finding | Why a reviewer could flag it | Fix |
|---|---|---|---|
| `22-25` | `NEURON_CC_FLAGS` gets `--auto-cast matmult --auto-cast-type bf16` only under `__main__`; the import path used by the scorer gets `--optlevel=1` | Training and eval are compiled with different precision flags | Keep, but document in the docstring that eval runs at ≥ training precision and parameters are identical. Or set flags in `main()` only |
| `32`, `36`, `69` | `SEQ_LEN=2048`, `N_EMBD=0`, `PACK_FACTOR=4`, while the model is trained with `--seq-len 1024 --n-embd 1024 --pack-factor 32` | Constants contradict the trained model | Bake 1024/1024/32 in as constants; drop those flags |
| `656-668`, `684-686`, `761-765`, `802-808` | Dead GQA and "coalesced" (`all`/`vfc`) branches | Unused complexity | Delete |
| `746` | `MLP.forward` switches implementation on `self.training` (custom-backward path in training, plain path in eval) | Precisely a "training-only adapter" / train-eval fork | Always call the same function. The `_FusedMLP` forward is `reference_forward`, so it is identical and legal under `no_grad` |
| `912` | `depth = None; depth = DEPTH if depth is None else depth` | Dead code | Delete |
| `1256-1267` | `EVAL_MIN_SEQ_LEN = 8192` enlarges the rotary tables in `load_for_eval` | Looks like eval special-casing | Keep only with a comment and a test that logits at positions < 1024 are unchanged. Or size the tables to the scorer context |
| `1311`, `1316`, `1322-1347` | `--ngram-ve` / `--qk-shift` flags that "must be passed as --no-…" | Advertises absent features | Delete the flags and checks; simplify the launch command |
| `1334-1343` | Constant-folded validations for byte-wte-init, batch-ramp, weight-EMA | Code referencing features that do not exist | Delete |
| `1470-1472` | In-process `causality_check` and `evaluate_bpb(split='public_val')` | The training script reads validation data | Remove from the submission file |
| `1501-1600` | `make_plan` message says "ownership **probe**"; hard-coded measured-ms constants; unused optimizer `state_dict`/`load_state_dict` resume path; `selected_lanes` "NKI CE **adapter**" wording at the end of the file | Probe and adapter vocabulary; unused resume machinery | Rename descriptively, co-locate with the CE kernel, comment the constants as load-balancing weights, delete resume code |
| `1387` | `assert world_size == 4` | Fine, but undocumented | State the full-chip LNC2 × 4 requirement in the docstring |

### 7.5 Submission timing and quota

The quota rule is 5 per team per week. Unknown: whether rejections count, and whether the week is calendar-based.
**Human: confirm with the organizers today.** If the week is Monday to Sunday and rejections count, the rejections on
Sep 22 and 23 leave 3 uses through Sep 27 and 5 more from Sep 28.

| Submission | Content | Gate | Earliest |
|---|---|---|---|
| **S0** (first point to consider) | R1 = cleaned 116188 dense, numerics unchanged | §7.2 all levels plus the §7.3 audit by an independent Verifier | **Sep 25 ~10:00–12:00 IST** |
| S1 | First adopted table stack, clean-ported | Two-host adoption; 2 clean cold replays both ≤ S0 public − 0.004; audit | Sep 26 evening |
| S2 / final | Best confirmed stack at the freeze | Same, plus cold startup ≤ 840 s | Sep 28 12:00 package; submit Sep 29 18:00 → **Sep 30 12:00 IST** |

Each submission is also a public→official calibration pair; record its commit and 20M value.

---

## 8. Stop list (Q16)

| Stop | Reason |
|---|---|
| Launching A1/A2 on exact 35c9 bytes | They crash at `make_plan` (§2.3). Use IMPL-1 |
| `--relu2-tau` arms on 35c9 | Assertion at `:4404` |
| Width 1152 / MLP ratio 5 on 35c9 | SBUF assert at `:4432`; the "1152 ≈ 1024" claim is unverifiable |
| Using "+0.005" with 20M numbers | §2.4 |
| 2M screening | §4.5; always 20M |
| 300-step or 900-s smokes as quality screens for tables, schedule, capacity, data order | Ledger shows sign flips |
| **B1 LNC1/world 8** | A 0.24-BPB failure at matched steps (Sep 9–10) was never root-caused; the assert is not the only blocker; B2 covers the table-sync motive |
| FP8 (any form) before the freeze | Full autocast learned nothing; the ROW-FP8 backward does not exist; TensorE is not shown to be the bottleneck; the value of steps in epoch 2 is unknown |
| New NKI kernels without a profile | No critical-path evidence; keep existing kernels (rope+norm, softcap-CE, local conv) and the compiled Muon |
| Porting Recursive's WD pulses, Demon β₁, momentum/β₂ ramps, TTTL windows, whole-model BF16 | Harness-overfit or non-transferable (§4.3) |
| Engram, output bigram, MUDD, DCMHA, RWKV/Mamba hybrids, U-Net skips, weight EMA, batch ramp, byte-loss weighting, byte-WTE init, SwiGLU, cautious WD on Adam, Adam-every-N | Measured negative, neutral or incompatible in the ledger or DECISION_LOG |
| Optimizing the old 640 stack | Superseded by dense by about 0.03 |
| Seed hunting; relaunching crashed labels without diagnosis; exporting `NEURON_CC_FLAGS` without `--optlevel=1` | Protocol |
| Leaving hosts idle while waiting for implementation | §5.4 |

---

## 9. Immediate commands to the team

**Next three jobs, trn21**
1. `20260924-A0-ctrl35c9-trn21-r1`: exact 35c9 base command, seed 42, 1,800 s, fresh 20M eval.
2. `20260924-D1-epochvalue-trn21-r1`: the same command plus `--max-train-seconds T_e` (from job 1's first
   `epoch 2` log line), fresh 20M eval.
3. `20260924-A1-ngram-lm1-m64d128-trn21-r1` on 35c9+IMPL-1 once the receipt exists. Until then,
   `20260924-F6-sandwich-trn21-r1` (`--mlp-sandwich-norm`, after G2).

**Next three jobs, trn22**
1. Finish the 116188 clean control and eval it (submission-safety evidence), then
   `20260924-A0-ctrl35c9-trn22-r1`.
2. `20260924-F1-mlr0175-trn22-r1` (`--matrix-lr 0.0175`).
3. `20260924-A2-ngram-l1m1-m64d128-trn22-r1` on 35c9+IMPL-1 once verified. Until then,
   `20260924-F3-wd085-trn22-r1` (`--warmdown-ratio 0.85`).

**Next two implementation tasks**
1. **IMPL-1** table optimizer bypass (§4.1) with its four acceptance tests.
2. **G2** preflight dry-run harness, mandatory for every arm. Then IMPL-3 (fresh tail) and IMPL-2 (epoch-2 freeze),
   and R1 cleanup of 116188 in the submission lane.

**Next Verifier tasks**
1. Confirm §2.3's claim that A1/A2 and `--relu2-tau` fail on exact 35c9 (G2 on CPU). Sign off G1 (shard count,
   scorer context and prefix, startup enforcement).
2. Verify IMPL-1: plan-JSON identity, the `--no-ngram-ve` lockstep, per-rank table hash equality, `load_for_eval`.
3. Run the independent reviewer-style audit of R1 against §7.3/§7.4 and the §7.2 equivalence evidence.
4. Review B2 against the catch-up-decay equivalence spec (§4.1) before any native training.

**Exact decision thresholds**
- **Nominate:** single-host 20M Δ ≤ −0.0006 with steps within ±10%, or a declared throughput arm.
- **Adopt:** both hosts < 0, mean ≤ −0.0010, |Δ21 − Δ22| ≤ 0.0015, current pooled controls.
- **Discard:** single-host Δ ≥ +0.0005, or two-host mean > −0.0003.
- **Control drift alarm:** >0.0010 from the pooled mean. **Big-win hold:** Δ ≤ −0.010 or public-20M < 0.975.
- **Throughput arms:** exactness gate plus ≥2% same-host step-time gain, then a BPB run.
- **Submission candidate:** 2 clean cold replays within ±0.0008 of the research mean, cold startup ≤ 840 s, and a
  passed audit.

**First point where a submission should be considered:** **S0 on Sep 25 about 10:00–12:00 IST.** It is the cleaned,
numerics-identical 116188 dense file, once lockstep, two cold replays and the independent audit pass. Expected official
≈ 0.991 ± 0.002, against the team's current 1.0246. Its main value is learning whether a review-clean file is accepted
while quota remains.

---

## Addendum (2026-09-24 12:50 UTC): claims now verified by execution

The G2 harness (`workflow/tools/preflight_dryrun.py`) runs the exact `train.py` bytes and argv through the file's
own `__main__` path (install adapters, then `main()`) on CPU/gloo with 4 ranks, for real training steps. Receipts
are in `workflow/evidence/`.

| Claim in §2.3 | Execution result |
|---|---|
| A1/A2 fail on exact 35c9 | **FAIL confirmed** at `make_plan` (full size and m8): `g2-a1-35c9.json`, `g2-a2-35c9.json` |
| `--relu2-tau 0.5` fails | **FAIL confirmed** (adapter assert): `g2-relu2-35c9.json` |
| Width 1152 / MLP ratio 5 fail | **FAIL confirmed** (SBUF assert): `g2-w1152-35c9.json`, `g2-mlp5-35c9.json` |
| Sandwich norm, head gate, x0 gate, out-pool, depth 8×768, epoch-shuffle, WD/LR/warmdown flags run | **PASS** (receipts `g2-*-35c9.json`) |
| Clean 116188 has no arithmetic drift against 35c9 + dense flags | **Bit-identical over 3 steps** (plan, losses, 62/62 parameter hashes): `lockstep3-35c9-vs-116188-base.json`. The research argv is **rejected** by the clean file (`--matrix-lr`), so the launch strings differ |
| IMPL-1 unblocks tables without changing anything else | `--no-ngram-ve` bit-identical to 35c9 over 3 steps; A1/A2 (m8) PASS; exactly one RMSProp update per step; float64 recomputation within BF16 rounding (`workflow/proposals/IMPL-1/`) |

New lead (hypothesis, §4 A-eps): at step 1 the median touched table-gradient element is about 6e-10 with 2,048
tokens/step, against RMSProp ε = 1e-10. At 131,072 tokens/step, rare-row gradients should be about 64× smaller,
so table updates may be ε-dominated early in training. Check the gradient statistics in the IMPL-1 Neuron smoke
before queueing A-eps.

The autonomous Codex workflow built on these tools is in `workflow/AGENTS.md` and `workflow/START_HERE.md`.

## Appendix A: answers to the 17 questions (index)

| # | Answer |
|---|---|
| 1 | §1 table. 0.980: IMPL-1 → A2 (+ LR/warmdown) ≈ Sep 25–26 if tables transfer. 0.975: add B2 scaling ≈ Sep 26–27. 0.960: requires near-full 640-stack transfer plus Recursive-scale tables plus no epoch-2 penalty (≈3%). 0.945: no identified path |
| 2 | §4.3: n-gram value tables (mechanism), with placement on VE layers; sandwich norm; long warmdown. Everything else is small or non-transferable |
| 3 | Yes, as the highest-EV mechanism. But not flag-only (IMPL-1), not single-epoch-safe (run D1/FT alongside), and step cost scales with size until B2 |
| 4 | §4.2 order and EV per host-hour |
| 5 | §4.5: always 20M. 2M misleads for tables, data order, context changes, \|Δ\|<0.001, and cross-depth comparisons |
| 6 | §5.1 and §5.2: screen on one host, replicate on the other, pooled ABBA controls |
| 7 | §5.4 |
| 8 | Lanes L1–L5 in §5.2. B2 yes (static-shape design); clean-port harness yes; async loader only if D2 shows ≥3% host stall |
| 9 | No. Park FP8 (§8) |
| 10 | None proven. D2 profile first. The only probable critical path is the table update path, handled by B2 without new NKI |
| 11 | After adoption: matrix LR ×{0.85, 1.15}; warmdown 0.85; final LR 0.0; table LR only if update/param diagnostics say so (640 was flat over 0.15–0.6); table β₂ keep 0.999; batch needs code (TOTAL must be a multiple of 131,072) and is low priority; capacity via SC-1 only |
| 12 | Plausibly a large part of the *table* question and some of dense's plateau. It cannot explain a 0.041 gap. D1/FT measure it (§3) |
| 13 | §3 diagnostics table |
| 14 | §7.1–7.2 |
| 15 | §6 |
| 16 | §8 |
| 17 | See below |

**Evidence that would change this plan (Q17):**

1. G1 shows dev hosts hold fewer shards than the official default. Redo every epoch argument, and FT becomes
   irrelevant.
2. A1 and A2 are both ≥ 0 on both hosts, *and* A2f ≥ 0. Stop tables and spend the remaining capacity on schedule, FT,
   and flag arms.
3. D1 shows 1 epoch within 0.003 of 1.7 epochs. Demote all throughput work, including B2 speed, and promote capacity
   and memorization control.
4. D1 shows more than 0.010. Promote B2, the async loader and profile-driven kernels.
5. B2 exactness fails or yields under 3% at the A2 layout. Cap tables at the A2 size.
6. S0 is rejected. Freeze the architecture and redirect the effort to the audit.
7. S0's official score differs from public-20M − 0.001 by more than 0.003. Recalibrate every target.
8. A cold startup above 840 s for any table graph. Measure compile time, and cut graphs or table layers first.

---

## Appendix B: arithmetic behind key estimates

- **Epoch size:** 640 runs capped at 2,150 steps × 131,072 tokens ≈ 282M tokens/epoch (35c9 docstring `:94-100`;
  `qkfreeze-frozen-1648s`). Dense 3,654 × 131,072 = 479M tokens ≈ **1.70 epochs**. The boundary falls at about 59%
  of steps. With `WARMDOWN_RATIO=0.75`, the cooldown starts at 25% of time, so all of epoch 2 is cooldown. Verify
  from the dense log's `epoch` field.
- **Dense FLOPs:** matmul params ≈ 6·12·1024² + 8192·1024 ≈ 83.9M → 6N ≈ 503M, plus causal attention
  ≈ 3·4·6·512·1024 ≈ 38M → ≈541M FLOP/token. Throughput ≈ 479M tokens / ~1,785 s ≈ 268k tok/s → ≈145 TFLOP/s
  ≈ **22–23% MFU** (dense BF16 peak 632 TFLOPS from 8 × 79 per the Neuron guide; AWS quotes 655).
- **Table step cost on 640:** +134M params gives 2,139 vs 2,191 steps (−2.4%); +268M gives +6.2% wall at matched
  steps; +537M gives about +13%.
- **Throughput exchange rate (single epoch, 640):** (1.0343 − 1.0813)/log₂2 = −0.047 per doubling → −0.00067 per
  +1% steps.
- **Offset:** 1.0548 − 1.0557118 = −0.0009 (20M); 1.0548 − 1.049535 = +0.0053 (2M).

## 10. Sources

Primary, read for this report:

- Recursive, *First Steps Toward Automated AI Research*, repository `recursive-org/first-steps-toward-automated-ai-research`
  @ `a962ec43e2e3d7c018e59a2ece623fe6e232fdfb` (Apache-2.0): `nanochat_autoresearch/solutions/optimized_from_karpathy.py`
  (SHA-256 `f92ecc81…`), `solutions/lib.py`, `results/val_bpb.csv`, READMEs.
  https://github.com/recursive-org/first-steps-toward-automated-ai-research
- Karpathy, `autoresearch/prepare.py` (`--num-shards` default 10; `EVAL_TOKENS=40*524288`; `VAL_SHARD=6542`; `VOCAB_SIZE=8192`).
  https://github.com/karpathy/autoresearch
- AWS Neuron SDK docs source, `nki/guides/architecture/trainium2_arch.rst` @ `be9037f6fe00b5e881f01a831dbde52be08e90f7`
  (96 GiB HBM, 3 TB/s; 28 MiB SBUF per NeuronCore-v3; 158 FP8 / 79 BF16 dense TFLOPS per core; double-FP8 `double_row` mode).
  https://github.com/aws-neuron/aws-neuron-sdk. Rendered: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html
  (blocked from this environment; read from source). Note: the competition README says 24 MiB SBUF; the arch guide says 28 MiB. 35c9's asserts use 24 MiB.
- AWS Trn2 instance page: https://aws.amazon.com/ec2/instance-types/trn2/
- Keller Jordan et al., modded-nanogpt `README.md` record table (#33 async loading, #62 bigram hash embedding, #71 sparse bigram
  gradient comms) and `train_gpt.py` `sparse_comms_*` (MIT). https://github.com/KellerJordan/modded-nanogpt
- OpenAI Parameter Golf README leaderboard (SmearGate, BigramHash, TTT entries; different rules): https://github.com/openai/parameter-golf
- Competition: https://trainium-frontier.devpost.com/ ; https://www.amazon.science/news/aws-trainium-frontier-competition-co-design-models-and-kernels-on-purpose-built-ai-chips

Papers:

- Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv:2305.16264.
- Hernandez et al., *Scaling Laws and Interpretability of Learning from Repeated Data*, arXiv:2205.10487.
- Kim et al., *Pre-training under infinite compute*, arXiv:2509.14786.
- Bergsma et al., *Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs*, arXiv:2502.15938.
- Amsel et al., *The Polar Express*, arXiv:2505.16932.
- *NorMuon: Making Muon more efficient and scalable*, arXiv:2510.05491.
- DeepSeek-AI, *Conditional Memory via Scalable Lookup (Engram)*, arXiv:2601.07372.
- Huang et al., *Over-Tokenized Transformer*, arXiv:2501.16975.
- Zhang et al., *How Does Critical Batch Size Scale in Pre-training?*, arXiv:2410.21676.
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (FP8 training), arXiv:2412.19437.
