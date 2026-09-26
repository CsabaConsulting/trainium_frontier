# Morning research report — 26 September 2026, 07:45 IST

Prepared from authenticated evidence available at 07:44 IST. This is not a claim of new live leaderboard data.

Delivery: consolidated report was sent to Poojith at approximately07:45IST;
do not send a duplicate scheduled report. The original sections below retain
their preparation-time knowledge.

### Chronological addendum — 07:56 IST

trn22's actual four-arm CPU qualification started07:43:59 and ended07:45:34.
Allfour G2 arms passed, but the checkpoint helper failed before firstforward
with `RuntimeError: NONFINITE_TENSOR`. Its finite-only buffer rule rejects the
intentional negative-infinity attention masks. A verification-only repair is in
fresh independent review; no model/evaluator change or fullCPU pass is claimed.
All observed processes exited, device/original832589 lock idle at07:46:55.

trn21's ordinary-only diagnostic started07:51:19 and ended07:51:42,exit1.
Both ordinary eager and compiled paths zeroed both-sign subnormal gradients on
allfour ranks; normal witnesses/input integrity passed. This is not K2-specific
and does not waive its failed gate. All11 observed identities absent and device/
original596823 lock idle at07:53:53. Raw a236d73e/root normal and optimized-mode
checks authenticate all31files. Both hosts currently HOLD pending next checked
packet. Confirmed BPB and submission status have not changed.

## 1. Best confirmed result

Public-20M/context1024 mean **0.9883066129**: trn21 **0.9882310747**, trn22 **0.9883821510**. Improvement versus the old confirmed schedule mean0.9901674437 is **0.0018608308 BPB**. Both improve on their genuine same-host controls; paired mean delta **-0.0022703270**. Exact36 adopted for research in bdedefe0. The old single0.986340 did not replicate and is withdrawn.

Official score is still the human-reported **0.9901**. No new upload. We are **0.0183066 public BPB above the0.970 freeze minimum**; first place is not close or guaranteed. The current gain is real, but insufficient.

## 2. Host work and utilization

For01:00–07:30IST, the archived timestamped ledger has **6 completed native jobs ontrn21 and9 ontrn22**, including startup/failed diagnostics; **four evaluated full runs per host** are included, one trn21 full being a seed diagnostic. The new trn21 confirmation finished at07:30:51, just outside that counting window.

Authenticated native compile/train/eval intervals establish lower bounds **trn21>=70.65%, trn22>=62.78%** busy wall time. These are NOT exact accelerator utilization: CPU gates and missing intervals are excluded, unknown gaps are not labelled idle. Method/evidence:UTILIZATION_SCOPE.md and recorded-intervals.jsonl.

At report preparation both were not training. trn22's >15-minute idle incident was reported. Its checked full-size schedule CPU qualification is released7be2e829, awaiting the Operator's actual launch receipt; that is CPU validation, not a scored run. trn21 awaits the independently checked ordinary-embedding control diagnostic. We do not have six launch-ready jobs per host.

## 3. Lanes

- **P:** profile captured; bounded decoder retry exceeded scratch allowance271389612bytes versus256MiB. Parked; no trustworthy full phase/MFU attribution yet.
- **CT:** exact36 previously measured cold fifth845.627s, below900; smaller cache-refresh base493.338s. New schedule child requires its own cold measurement. Research cache-hit reuse is not established merely by the directory.
- **K1:** existing Muon batching/ownership confirmed; no new validated whole-step>=5% improvement. Do not count a no-op batching patch as a gain.
- **K2:** registered candidate observations were exact, but required ordinary compiled subnormal control failed all4 ranks: zero versus1408nonzeros. Ordinary-only diagnostic is being qualified; original K2 gate remains FAILED.
- **K3:** no accepted flash-attention/ReLU2 throughput result; separate lane not running.
- **K4:** prior compiled zero-gradient defect remains; no validated whole-step>=3% improvement. Lower priority.
- **V:** warm five-step observation120.33794s missed120s target; no claimed cache-hit proof. Seed43 was worse by about0.00225; seed44 incomplete, no changed selection threshold.
- **X:** no extra AWS instance or qualified16-chip execution; no resource action taken.
- **T:** tables remain deprioritized after A1's~+0.007 loss; no new table win.
- **S:** new confirmed8x1024 clock36 base adopted. Recursive schedule port4bfaec has independent source/small-CPU approval; full-size CPU then separate cold-five/full1800 remain.
- **I:** the exact primary team write-up remains unverified; independent local evidence, not reported competitor causality, controls decisions.
- **C:** no upload-ready improved file. Old clean port3f5035 retains its batch-routing review refusal. A clean116188-based port of the confirmed winner, cold full replay and ClaudeviaPoojith audit still remain.

## 4. Main decisions

Adopt the two-host confirmed0.9883066 base; withdraw unconfirmed0.986340. Preserve every failed native/validator gate rather than count partial exactness as success. Keep tables low priority and K1/K2/compile plus architecture-schedule follow-up critical. Use independent input-only Astra review after the recorded Opus refusal; never retry/reword that refusal. Preserve original per-host exclusive locks and one sole Operator. Details/undo paths are in research/v2/DECISIONS.md.

## 5. Current errors and blockers

K2 actual native error: `torch.distributed.elastic.multiprocessing.errors.ChildFailedError`; ordinary compiled FP32-subnormal gradient max_abs **1.2611686178923354e-44**, expected1408nonzeros/actual0. Exact mechanism is not proved.

New ordinary-only diagnostic's initial validator falsely accepted empty bytes labelled FP32[256,128]. Fresh review refused it; the narrow repair now rejects with `RAW_BYTE_COUNT`, and source/CPU review passes. Mechanical packet review remains separate.

Schedule child must finish full-size checkpoint causality/state tests and its own cold compile before a full score. The retained CPU harness cannot guarantee automatic descendant cleanup after timeout; mandatory explicit Operator postflight/HOLD stays.

## 6. Human actions (none blocks the next research checks)

1. Billing alert: either create/provide the requested $1,000/day Bedrock alert/billing report, or grant suitable billing access later. Recommend the former; no alert is claimed configured.
2. If available, send the exact public Trainium of Thought write-up URL. Otherwise we continue independent profiling; no need to wait.
3. When a qualifying package is ready, obtain the Claude compliance audit then manually upload; do not upload the withdrawn/review-only file now.

## 7. AI spend

Actual overnight billing is **unknown**, not zero. Local GPT-6 Astra implementation/review sessions are documented. No new Bedrock call in this continuation; that does not establish total overnight account cost. The requested alert is unverified.

Evidence anchors: adoption bdedefe0; replica rawcc1fa225; K2 raw30630c84; CPU r2 review0819c28d and release7be2e829; ordinary validator refusalb5b84a0c. No official score, submission acceptance, full-size CPU outcome or future launch is inferred.
