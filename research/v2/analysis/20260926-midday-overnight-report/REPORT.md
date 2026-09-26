# Overnight evidence update — 26 September 2026, 12:11:38 IST

Read-only Analyst. Canonical evidence: `/home/poojith-devan/Desktop/Trainium`; untracked `runs/` copies excluded. No SSH, portal access, source/control changes, launch or submission. Accounting cutoff is 06:41:38 UTC / 12:11:38 IST; it is not a live-host claim after that time.

## trn21 idle accounting

From midnight IST to the cutoff (12h11m38s), **14 recorded native job intervals cover 7h46m28.367s**. Their complement is **4h25m09.633s with no recorded native job**. This is the useful operational gap figure, **not an exact measurement of accelerator idle**: native intervals include CPU compilation/startup and evaluation; gaps are not continuously observed. The arithmetic must not be presented as hardware utilization.

| Disjoint wall-time category | Duration |
|---|---:|
| Recorded native compile/train/eval/profile/diagnostic job intervals | 7h46m28.367s |
| Four recorded CPU-only validation gates, inside non-native gaps | 6m13.569s |
| Latest no-native-job intervals around the failed 37s diagnostic | 1h51m09.320s |
| Other turnaround/hold/unobserved gaps | 2h27m46.744s |
| Total since midnight | 12h11m38.000s |

The last full train/eval ended **10:19:51.986 IST**. The only later native child ran **11:56:19.842–11:56:56.535 IST**, 36.693714s, then failed its compiler-cache census. Before that attempt, the gap was **1h36m27.855s**; afterward, **14m41.465s** through the cutoff. The parked guardian is a lock holder, **not accelerator work**. At the recorded 11:58:30.285 IST postflight, all 40 observed job/compiler identities were gone, guardian children were empty, Neuron reported no device process and fuser was empty. A continuous whole-day hardware-idle total remains unknown.

The earlier 01:00–07:30 report was deliberately a lower bound. This extension restores the actual P profile and batch timing intervals, adds 00:00–01:00 and later work, and keeps CPU gates separate. Full timestamp rows and receipt paths: [interval-accounting.json](/home/poojith-devan/trainium-overnight-idle-analyst-uacOoaLo/interval-accounting.json). Key final evidence: `research/v2/operator/20260926-clean-native-trn21-launch/RESULT.md`, `explicit-postflight.json`, and `research/v2/ledger.jsonl`.

## Submission usage

README line225 permits **5 submissions/team/week**, best score counts, no retries. It does not define the weekly reset timezone or whether “week” is rolling/calendar. The following IDs/times come from the user's earlier My Submissions screenshot/history, relayed by the Supervisor; only the newest also has a dated canonical submission-vehicle note.

| ID | Displayed upload date/time | Evidence/status scope |
|---|---|---|
| sub-b18271c7 | Sep19 13:20 | User screenshot/history; subsequently corrected to QUEUED |
| sub-a7629c90 | Sep22 18:54 | User screenshot/history; subsequently corrected to QUEUED |
| sub-7be49b6b | Sep23 13:43 | User screenshot/history; subsequently corrected to QUEUED |
| sub-452d2ae4 | Sep24 13:56 IST | User-confirmed SCORED, official0.9901; canonical `research/v2/SUBMISSION_VEHICLE_2026-09-25.md` |

Thus **four known uploads Sep19–24**, of which **three are since Monday Sep21**. A Monday calendar-week interpretation would count3; a rolling-seven-day interpretation would count4 at this cutoff if the screenshot's timezone is IST (the Sep19 row would age out at13:20 IST). These imply2 or1 unused slots only under those assumptions and a complete history. **Exact quota remaining is unknown**, because the reset definition and a complete newer portal history were not observed. Queued submissions are not assumed to refund quota. No portal-side source-byte identity or new upload is claimed.

## Results and major lanes

The overnight quality gain is the confirmed eight-layer clock36 base: **0.9883066128526815 public20M mean**, from trn21 **0.9882310747242088** and trn22 **0.9883821509811542**. It improves the previous confirmed schedule mean by **0.0018608308023681 BPB**. The earlier single0.986340 failed confirmation and was withdrawn. Official best remains the user's reported0.9901.

The exact clean e1e2 port completed a genuine cold replay: **0.9883890652263234**, 3,628steps, **859.473523823s startup**, **1789.768582730s charged**, no over-budget result. The latest human instruction accepts the Claude SOURCE-AUDIT e1e2/nine tests and removes native B1/B2 as an upload gate; older “native consistency and Claude pending” text in state is stale relative to that instruction. Upload itself remains human-only.

| Lane | Measured progress and remaining limit |
|---|---|
| Architecture/schedule | Eight-layer base confirmed. Exact36+Recursive first full result0.9880425232584465; difference versus its historical trn22 base is only-0.000339628, and selector lacks the prospective same-host control. No adopted Recursive gain on eight layers. Latest human instruction schedules one full eight-layer Recursive replicate. |
| Compile/cache | Clean cold replay passes the900s startup cap with40.526s margin. Exact-source anchored warm control startup110.209842s; source-path-only causality and actual cache-hit attribution remain unproved. |
| K1/Muon | Existing batching/ownership already present. Runtime attribution supports only small remaining writeback work; no accepted whole-step>=5% improvement. |
| K2/embedding | Original native ordinary eager and compiled FP32-subnormal control failed (0 versus1408 expected nonzeros); candidate partial exactness is not a full native pass. New trn22 binding was still in CPU/diff review in last canonical state. Latest human deadline16:00 IST requests simplified CPU exactness, flag-off identity and native timing; no timing result exists yet. |
| Batch | Old six-layer262144 screen measured +4.9977495% printed-median throughput, narrowly below its predeclared5% screen. New exact36 batch port has tiny CPU acceptance and full113M CPU plan, not a full result. |
| FP8/K4 | Compiled component dX/dW zero-gradient failure retained; no accepted whole-step>=3% or BPB gain. |
| K3 / tables / seeds | No accepted new flash-attention/ReLU2 result. Tables remain deprioritized after A1 delta+0.0071572. Seed43 was worse by about0.00225; no complete seed44 evidence or threshold change. |

No six-ready-jobs claim, bill/spend estimate or new official score is supported.

## Exact profile numbers and the optimizer claim

These measurements are **the older six-layer Recursive6 source6ef7, step20, four ranks**, not the current eight-layer e1e2/clock36 model. They are coarse runtime graph spans or inclusive host annotations; **there is no current eight-layer exclusive phase breakdown**.

Runtime-observed device-labelled compiled-graph intervals (ms):

| Rank | Forward | Backward | Sum of three Muon graph intervals | Final full parameter writeback |
|---|---:|---:|---:|---:|
| 0 | 138.108883 | 273.642050 | 7.574627 | 6.797362 |
| 1 | 137.073180 | 272.255578 | 10.988568 | 7.974865 |
| 2 | 141.857117 | 281.910434 | 11.611239 | 7.752924 |
| 3 | 141.687720 | 281.273465 | 21.297725 | 8.143385 |

Inclusive host annotations (ms, sums only within the same repeated annotation label):

| Phase | Rank0 | Rank1 | Rank2 | Rank3 |
|---|---:|---:|---:|---:|
| Forward dispatch | 7.897221 | 7.470360 | 7.797104 | 8.396877 |
| Backward dispatch/wait | 146.725391 | 147.012292 | 154.816451 | 150.048712 |
| Data fetch | 275.838232 | 275.177286 | 282.378753 | 281.702138 |
| Gradient unscale | 0.074238 | 0.077516 | 0.087580 | 0.065377 |
| Clip/reduce | 22.428792 | 1268.148260 | 1251.301259 | 1255.640321 |
| Full optimizer | 32.891054 | 31.164747 | 31.152157 | 20.362889 |
| Cache refresh | 5.694834 | 5.484653 | 5.515290 | 4.273343 |
| Existing synchronization | 1.596679 | 2.792512 | 2.410214 | 15.948903 |
| Record boundary | 0.056821 | 0.052138 | 0.371100 | 0.062867 |
| Captured loop wall | 502.096768 | 1747.248569 | 1745.156034 | 1744.044544 |

Nested ranges overlap; graph and host tables must not be summed together or converted into exclusive phase percentages. Capture-start rank skew was about1.24s, explaining why other ranks' clip/reduce windows are inflated. All profiler device-attributed phase fields are null.

Rank0's “data fetch” contains a268.582190ms blocking host copy, of which267.888589ms is waiting before dispatch; the device-labelled transfer is only0.193672ms. It overlaps a273.642050ms backward graph. This supports queued backward synchronization surfacing at data fetch, not a275ms disk/I/O bottleneck.

**“Optimizer <5%” is not established**, especially for the current eight-layer model. Muon graph time alone omits AdamW, packing, gather and writeback; inclusive host optimizer time is not exclusive device time. Even the six-layer rank0 host annotation is32.891/502.097≈6.55%, but that ratio is not an accepted steady-state device-cost estimate. Embedding backward's separate device time, internal matmul/SDPA dtype coverage and MFU remain unknown.

Profile sources: `research/v2/analysis/20260926-p-runtime-interpretation/ANALYSIS.md`, `findings.json`; `research/v2/analysis/20260926-k1-writeback-runtime/ANALYSIS.md`, `result.json`; and `research/v2/operator/20260926-trn21-profile-recursive6ef7/completed/profile/rank{0,1,2,3}/profile.json`. Quality/state sources: `research/agentic/CURRENT_STATE.md`, `research/v2/ledger.jsonl`, `research/v2/analysis/20260926-clean-fullcold-launch/terminal-result.json`, and the prior `research/v2/analysis/20260926-overnight-report/MORNING_REPORT_0745_IST.md`.

