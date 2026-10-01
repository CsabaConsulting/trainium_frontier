# Round19 MLP5r2 — BPB loss; one MLP4.5 branch-C trial

MLP5r2 scored **0.9912812939805707** versus original exact500f **0.9778364059642928**, delta **+0.013444888016277967** (worse). No MLP5 adoption/repeat or MLP6 is selected. Root selected one MLP4.5 full on trn21 under [Round19 branch C](../ROUND19_BRANCH_DECISION.md), using descriptive nearby-token EMA evidence, not a demonstrated causal capacity advantage. Current scientific parent stays500f. Tonight's newer trn22 exact500f score is pending in this report and must be reported separately, never substituted silently.

## Actual full-run metrics

| Field | Actual observation |
|---|---|
| Job / host | trn21-capacity-mlp5-gatefixed-full-r2 / trn21 |
| Source | 17239e61fb21f49bf22a62953bce6a9a616c1a6c1228853166cf812e3055f865 |
| Shape / seed | width1024, hidden5120,32rows/rank,context1024,four ranks/LNC2; seed42 |
| Completed optimizer steps | 2364 |
| Actual training tokens | **470,286,336**, not projected tokens |
| Ramp phases | 1140 accumulation1 steps;1224 accumulation2 steps; first phase2 zero-based step1140 |
| Cold fifth-boundary startup / excess | 705.7420153021812s / 0s |
| Charged through process exit / budget | 1789.7719818269834s / 1800s; not over budget |
| Training harness wall | 2495.5139971291646s |
| Printed phase1 harmonic tok/s | 248,468.22029638404 over steps20,30,…,200; NOT all-run throughput |
| Fresh public evaluation | 20,971,520tokens,context1024,83,462,186bytes,57,347,188nats |
| Eval duration / timeout | 670.2369430970866s /1200s; not over timeout |
| Checkpoint | 520,153,554bytes; SHA45965a8c2866fc4b927bbc712a84d42b1b5625432004e16587063232267a061a |
| Final terminal | 2026-09-27T17:32:58.580451+00:00 |
| Original outer exit / cleanup | exit0; independently authenticated original Guardian release and idle/ownership checks |
| Raw loss / isolated forward+backward ms | UNKNOWN / UNKNOWN |

Token accounting uses the pinned source's one-way131072→262144 token/update ramp,2364updates and the unique compatible integer total behind the archived470.3M display. It is source/log quantization, **not checkpoint-metadata extraction** or rounded tok/s extrapolation. Original500f comparator completed2685steps and531,496,960tokens, with1315/1370 phase steps. The existing full_result reader emits an early-rate projection447,242,796.53349125; it is expressly excluded as an actual-token metric here and set null in the ledger-ready record.

## Nine observed token brackets

Each cell lists lower and upper observed `zero-based step: cumulative tokens / printed EMA`. Arrows indicate token order, not monotonic loss. No target50M multiple was logged exactly; no interpolation is performed.

| Target | MLP5r2 lower → upper | Original500f lower → upper |
|---|---|---|
| 50M | 380: 49,938,432 / 4.4282 → 390: 51,249,152 / 4.3943 | 380: 49,938,432 / 4.4194 → 390: 51,249,152 / 4.3858 |
| 100M | 760: 99,745,792 / 4.1011 → 770: 101,056,512 / 4.0724 | 760: 99,745,792 / 4.0993 → 770: 101,056,512 / 4.0714 |
| 150M | 1140: 149,684,224 / 3.9350 → 1150: 152,305,664 / 3.8514 | 1140: 149,553,152 / 3.9757 → 1150: 150,863,872 / 3.9378 |
| 200M | 1330: 199,491,584 / 3.5909 → 1340: 202,113,024 / 3.5930 | 1410: 197,525,504 / 3.6124 → 1420: 200,146,944 / 3.5965 |
| 250M | 1520: 249,298,944 / 3.3082 → 1530: 251,920,384 / 3.2915 | 1610: 249,954,304 / 3.4410 → 1620: 252,575,744 / 3.4279 |
| 300M | 1710: 299,106,304 / 3.0114 → 1720: 301,727,744 / 2.9377 | 1800: 299,761,664 / 3.2607 → 1810: 302,383,104 / 3.2119 |
| 350M | 1900: 348,913,664 / 2.8733 → 1910: 351,535,104 / 2.8751 | 1990: 349,569,024 / 2.9016 → 2000: 352,190,464 / 2.8372 |
| 400M | 2090: 398,721,024 / 2.7808 → 2100: 401,342,464 / 2.7339 | 2180: 399,376,384 / 2.7763 → 2190: 401,997,824 / 2.7546 |
| 450M | 2280: 448,528,384 / 2.7238 → 2290: 451,149,824 / 2.6969 | 2370: 449,183,744 / 2.7475 → 2380: 451,805,184 / 2.7261 |

Nearby EMA ranges are lower for MLP5 around200M,250M,300M and450M. **350M ranges overlap and400M is mixed**; this is not uniformly lower loss. At50M and100M, shared token-coordinate samples show candidate EMA slightly higher. At150M the models are in different accumulation phases. At200M and later, none of the paired bracket points shares an exact candidate/base token coordinate, so no exact-token delta is claimed.

The logged quantity is rank0's last-microbatch training objective, including scheduled MTP contributions, smoothed by debiased0.9 EMA. It is not raw loss, global accumulated optimizer-step loss, or validation BPB. Different accumulation phases also change the token span represented by an EMA window. Comparisons are **cross-host** (candidate trn21 vs archived500f trn22); at similar token totals, wall-clock-based LR and MTP schedule progress differ. The data do not establish that slower speed, additional capacity or any single mechanism caused the quality result.

## Speed and quality comparators remain separate

The trn21 exact500f native60 archive provides same-host early training samples only, ending7,864,320tokens; it cannot supply50M+ loss or full20M quality. Its sampled steps20/30/40/50 have candidate-minus-control EMA +0.9340,+0.4035,+0.1628,+0.0423 respectively. Native60 timing screens and the248,468tok/s early full-run statistic are not substitutes for the cross-host fresh20M quality comparison above. Whole-step dt includes more than forward/backward work and is not an isolated FB timer.

## Provenance and scope

- Candidate archive: `research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-capacity-mlp5-gatefixed-full-r2/results-bundle.json`, SHAe8b7c5ea8cbe78e6b22315ecea3def4f77be3df4bfc190fed5a80365aaa92414.
- Original500f archive: `research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-r13-gate-adamw-full-r1/results-bundle.json`, SHA9931d177cb25b912a18406dae29d20d79a45c2a402be06ef46417de0d279d57c.
- Existing compare.py SHA9a6806d567e99ab01134a87a903d899e36c506cabd5b0b73dd78c58b812ee261 was run unchanged, without optional checkpoint metadata. Its independent reader review is `verification/20260927-mlp5-matched-token-independent/REVIEW.json`.
- Actual result review: `verification/20260927-round19-mlp5-result-independent/REVIEW.json`, SHA4250552257987020f7ab484aa09da1da726fb95be632c0619f699779d80c1763, authenticated40members and independently recomputed BPB and unique token accounting.
- Root branch decision SHA1818ec9238073fbe3586e4fc8279fa3fb1739728b7a38e2ba3a53257bea8830e; Round19 directive SHAf9aac6cf5cfaf59e26bc30ca731be2aff8d616f0054d2fe07c9885d7538a106c.

Report-only local work. Main ledger, state, queue, source and hosts were not changed. Separate `20260927-round19-mlp5/ledger-ready.json` awaits Root's append decision; `comparison.json` retains all reader brackets and early-reference provenance.
