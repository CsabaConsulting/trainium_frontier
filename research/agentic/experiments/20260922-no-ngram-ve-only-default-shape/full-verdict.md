# Verdict: CONFOUNDED — 20260922-no-ngram-ve-only-default-shape (full)

- Hypothesis: At the pinned 9a4caa7b default shape the model is 592.9M parameters, of which the hashed n-gram value-embedding tables alone are 4 tables x 1,048,576 rows x 128 dim = 536.9M (control ledger oneliner 'ngram_ve: True tables=4 rows=1048576 dim=128 ... opt=rmsprop bf16=True'), leaving roughly 56M dense transformer parameters. Two smoke rows at this commit show that per-step maintenance of that table block is a first-order throughput cost: --ngram-rms-fp32 (only the table RMSprop state math changed) dropped steps from 335 to 316 (-5.7%), and --weight-ema (one extra elementwise pass over all parameters, dominated by the tables) dropped steps from 334 to 313 (-6.3%). Removing the tables entirely (gather, backward scatter, RMSprop update, clip-norm contribution, index computation) should therefore raise the completed step count at the fixed 1800 s budget by at least 8%. Hypothesis: within a 30-minute budget the n-gram tables cost more in lost dense-model optimizer steps than they return in quality, so the dense-only model at seed 42 reaches a lower public_val_bpb than the same-host control. Supporting but confounded prior: the entire fresh-start lineage and the frozen candidate submissions/dense6x1024_public0992851/train.py run with --no-ngram-ve and score 0.979-0.993 public (fs-w65, fs-w98, fs-w146, CURRENT_STATE.md) versus 1.0057 for the ngram-enabled default; those runs differ in depth/width/eval settings, so they motivate but do not establish the isolated effect this run measures.
- Single causal change: Pass --no-ngram-ve only. Every other option stays at the 9a4caa7b defaults used by the two pinned controls: identical launch shape (NEURON_LOGICAL_NC_CONFIG=2, torchrun --nproc_per_node=4, --num-steps 5000, --max-train-seconds 1800, --seed 42), identical environment (NEURON_CC_FLAGS exactly '--optlevel=1', NEURON_COMPETITION_R1_SEED=42, PYTHONUNBUFFERED=1), no precision flag, no kernel/compile flag, no optimizer-rule, schedule, batch, evaluation, or other architecture change. No other --ngram-ve-* option is passed. The only behavioral delta is that the hashed n-gram value-embedding tables and their RMSprop optimizer are not constructed or trained; the dense transformer, its optimizer, data order, and evaluator are unchanged. Pre-full inertness check: the candidate's resolved 'ngram_ve' oneliner must read 'False tables=0' and num_params_M must fall from 592.9 to roughly 56 (>=500M fewer); if either does not hold, classify FALSIFIED-INERT without spending the full run. Comparison: same-host pinned control on public_val_bpb at the fixed 1800 s budget (trn21 -> 2071 steps / 1.005734; trn22 -> 2076 steps / 1.005967). Because the hypothesis is explicitly a fixed-budget tradeoff (fewer parameters, more steps), the primary comparison is the fixed-budget public_val_bpb; no matched-step equivalence is claimed. Outcome classes, mutually exclusive, evaluated in order: (1) FALSIFIED-INERT per the pre-full check above; (2) CONFOUNDED-SHORT if candidate steps < 90% of same-host control steps (<1864 trn21, <1869 trn22; experiment-protocol.md section 5 rule; record as throughput regression only); (3) FALSIFIED-THROUGHPUT if steps < +8% of control (<2237 trn21, <2242 trn22): the table-cost mechanism is falsified regardless of BPB; record BPB for information only; (4) SUCCESS if steps >= +8% and public_val_bpb < same-host control by at least 0.002; (5) FALSIFIED-QUALITY if steps >= +8% and public_val_bpb >= same-host control (the tables return more quality than their step cost at this budget); (6) INCONCLUSIVE if steps >= +8% and the BPB improvement is between 0 and 0.002 (within the 0.00023 inter-host control spread plus margin).
- Flags: `--no-ngram-ve`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-no-ngram-ve-only-default-shape-full-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-full-r1`
- Budget: num_steps=5000 max_train_seconds=1800
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- steps differ by more than 10% (2548.0 vs 2071.0)

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.026131 | 1.005734 |
| num_steps | 2548.0 | 2071.0 |
| steps/s | 1.4275309541150765 | 1.1603541012998655 |
| params (M) | 55.7 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

The full-tier verdict is CONFOUNDED.
Disabling n-gram VE produced validation BPB of 1.026131 versus 1.005734 for the control, a worsening of 0.020397 (lower is better).
The confounder is unequal training steps: 2,548 versus 2,071, about 23% more for the candidate and beyond the 10% tolerance.
Training time was nearly identical, but the candidate ran at 1.428 versus 1.160 steps/second and processed 334.0M versus 271.5M tokens.
Removing n-gram VE also reduced parameter count from 592.9M to 55.7M.
Commit, source hashes, seed, launch configuration, and other recorded options match; both runs passed causality checks and stayed within budget.
The observed BPB difference therefore does not isolate the effect of disabling n-gram VE at matched training steps.
