# Verdict: DISCARD — 20260922-weight-ema-fullbudget-r3 (smoke)

- Hypothesis: At the pinned 9a4caa7b defaults (592.9M params, ~2071-2076 optimizer steps in 1800 s), maintaining an EMA of the weights over the last 25% of the budget (decay 0.995, ~518-519 tail updates) and evaluating that EMA state will lower public_val_bpb relative to the raw final weights and relative to the same-host control. Mechanism (hypothesis, not established): late-training weight averaging reduces iterate noise in the low-LR tail; the only prior negative EMA result was a 400-second run with 474 steps and only 103 EMA-tail updates, where 0.995^103 ~= 0.60 of the EMA mass still sat on the snapshot taken at EMA start; at full budget the residual snapshot weight is 0.995^519 ~= 0.07, so averaging lag no longer dominates. The full-budget direction is not closed (verifier finding on r2; no weight-ema row in DECISION_LOG.md).
- Single causal change: Add --weight-ema only. Every other option stays at the 9a4caa7b defaults used by the two pinned controls: identical launch shape (NEURON_LOGICAL_NC_CONFIG=2, torchrun --nproc_per_node=4, --num-steps 5000, --max-train-seconds 1800, --seed 42), identical environment (NEURON_CC_FLAGS exactly '--optlevel=1', NEURON_COMPETITION_R1_SEED=42, PYTHONUNBUFFERED=1), no precision, kernel, optimizer-rule, schedule, batch, architecture, or evaluation change. --weight-ema-decay and --weight-ema-start-frac are NOT passed; both control ledger rows already record the resolved defaults decays=(0.995,) start_frac=0.75 with weight_ema=False updates=0. Per train.py:293-309 and 6218-6233 the only behavioral delta is that EMA shadow weights are maintained after 75% of the budget, saved alongside the raw weights, and used as the preferred evaluation state when present. This is a corrected resubmission of the never-run NEEDS_WORK proposals 20260922-weight-ema-default-shape and 20260922-weight-ema-fullbudget-r2 that fixes each cited defect: (a) historical evidence corrected to 474 steps / 103 EMA-tail updates; (b) averaging lag is stated as a hypothesis; (c) outcome classes are mutually exclusive with explicit precedence and the paired EMA-vs-raw comparison is primary. Pre-full check: the candidate's resolved weight_ema oneliner must show weight_ema=True and, at run end, updates>0 while controls show False/0; otherwise FALSIFIED-INERT without a full run. Definitions: S = candidate steps; C = same-host control public_val_bpb (trn21 1.005734 @ 2071 steps; trn22 1.005967 @ 2076 steps); E = candidate public_val_bpb (EMA-evaluated state); R = candidate raw-weight public_val_bpb if the run logs it. Outcome classes, mutually exclusive, evaluated in order, first match wins: (1) CONFOUNDED-SHORT if S < 0.90 x same-host control steps (experiment-protocol.md section 5; record as throughput regression only). (2) FALSIFIED-INERT if the final weight_ema oneliner shows weight_ema=False or updates=0, or the evaluated state is not the EMA state. (3) FALSIFIED-MECHANISM if R is logged and E >= R (EMA did not beat its own paired raw weights; this takes precedence over any control comparison). (4) SUCCESS if (R not logged or E < R) and E <= C - 0.001. (5) FALSIFIED-QUALITY if E >= C (EMA beat raw or R unavailable, but the run did not beat control, e.g. EMA update overhead cost steps). (6) INCONCLUSIVE otherwise (C - 0.001 < E < C, within the observed ~0.0002-0.0003 control noise band). Trajectory/numerical-equivalence claims are not made from the fixed-budget result; step count is recorded separately as an overhead measurement (EMA overhead annotation if S < 0.97 x control steps).
- Flags: `--weight-ema`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-weight-ema-fullbudget-r3-smoke-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- smoke step rate 1.1006 < floor 1.1513 steps/s

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.174056 | 1.165855 |
| num_steps | 313.0 | 334.0 |
| steps/s | 1.1005625879043601 | 1.1748153359127682 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

DISCARD at the smoke tier: candidate throughput was 1.1006 steps/s, below the required 1.1513 steps/s floor.
The control achieved 1.1748 steps/s; the candidate was about 6.3% slower and completed 313 versus 334 steps in nearly identical training time.
Candidate raw validation BPB was worse: 1.174056 versus 1.165855, a +0.008201 difference.
EMA validation BPB was 1.193370, also worse than the candidate’s raw weights by 0.019314 after 45 EMA updates.
The intended change was only --weight-ema; source hashes, clean commit, seed, host, launch configuration, and 592.9M parameter count matched.
The quality comparison is confounded by unequal training progress under the time cap; these are not equal-step checkpoints.
Runs also occurred at different times, so runtime variability cannot be excluded from this single pair.
Both runs passed causality checks and completed successfully within budget, but the throughput gate determines the discard.
