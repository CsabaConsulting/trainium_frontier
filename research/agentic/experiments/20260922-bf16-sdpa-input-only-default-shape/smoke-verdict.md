# Verdict: KEEP — 20260922-bf16-sdpa-input-only-default-shape (smoke)

- Hypothesis: In the controller's pinned environment the compiler flags are exactly NEURON_CC_FLAGS='--optlevel=1' (no '--auto-cast matmult --auto-cast-type bf16', which every fresh-start recipe run carried). The fs-w15 ledger description ('BF16 SDPA inputs only, retain FP32 normalization outputs and residuals') indicates that the train.py default feeds scaled-dot-product attention FP32 q/k/v. Hypothesis (to be confirmed by the verifier against the argparse default of --bf16-sdpa-input at 9a4caa7b): without compiler auto-cast, the two T x T attention matmuls per layer at seq_len 2048 run in FP32 on the tensor engine, which is a large fraction of the 0.87 s/step control step time. Passing --bf16-sdpa-input casts only the SDPA inputs to BF16, so tokens/step and the optimizer are unchanged while the step rate rises by >=3%; within the fixed 1800 s budget the extra steps lower official public val_bpb. BF16 attention inputs are standard practice and fs-w15 (0.989128 at 2600 steps) shows no quality collapse from this cast. If the default already resolves SDPA inputs to BF16 (or attention is not throughput-bound), the step count will be within +/-1% of control and the hypothesis is falsified as inert.
- Single causal change: Pass --bf16-sdpa-input only. Every other option stays at the 9a4caa7b defaults used by the two pinned controls: identical launch shape (NEURON_LOGICAL_NC_CONFIG=2, torchrun --nproc_per_node=4, --num-steps 5000, --max-train-seconds 1800, --seed 42), identical environment (NEURON_CC_FLAGS exactly '--optlevel=1', NEURON_COMPETITION_R1_SEED=42, PYTHONUNBUFFERED=1), no other precision flag (no --bf16-norm-output, no --ngram-rms-fp32), no kernel flag (no --nki-*, no --compile-sdpa-direct, no --compile-muon), no optimizer, schedule, batch, architecture, or evaluation change. The only behavioral delta is the dtype of the q/k/v tensors entering SDPA (BF16 instead of the default). Comparison: same-host pinned control on public_val_bpb (trn21 -> 2071 steps / 1.005734; trn22 -> 2076 steps / 1.005967). Outcome classes, mutually exclusive, evaluated in order: (1) VOID-PRECONDITION if the verifier finds the argparse default of bf16_sdpa_input at 9a4caa7b is already True (do not launch); (2) CONFOUNDED-SHORT if candidate steps < 90% of same-host control steps (trn21 < 1864, trn22 < 1869; protocol section 5 rule; record as throughput regression only); (3) FALSIFIED-INERT if steps are within +/-1% of same-host control (trn21 2050-2092, trn22 2055-2097): SDPA inputs already effectively BF16 or attention is not throughput-bound; (4) FALSIFIED-QUALITY if steps >= +1% and public_val_bpb >= same-host control (extra steps do not translate, or BF16 attention numerics cost more than they buy); (5) SUCCESS if steps >= +1% and public_val_bpb is lower than the same-host control by at least 0.0005 (larger than the 0.0003 cross-run spread observed among control/no-qk-shift/clip-after-reduce full runs); (6) INCONCLUSIVE otherwise (bpb lower but by < 0.0005, or steps between -10% and -1%). Because tokens/step are unchanged (unlike --batch-ramp, whose step gain came from smaller early batches), a step gain above +10% does not confound the quality comparison; a matched-step sanity check on the logged loss curve is recorded as secondary evidence only.
- Flags: `--bf16-sdpa-input`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn22 (`ip-172-31-32-253`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-bf16-sdpa-input-only-default-shape-smoke-r1`; control label: `al-20260922-control-trn22-lnc2x4-9a4caa7b-smoke-r2`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- correctness gates passed; step rate within tolerance; eligible for full run

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.156334 | 1.165604 |
| num_steps | 360.0 | 335.0 |
| steps/s | 1.2644889357218125 | 1.1775043936731107 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

KEEP at smoke tier; correctness gates passed and the candidate is eligible for a full run.
The candidate adds only --bf16-sdpa-input; source hashes, seed, host, launch shape, and parameter count (592.9M) match the control.
Validation BPB improved from 1.165604 to 1.156334, a delta of −0.009270, passing the +0.001 limit.
Step rate rose from 1.1775 to 1.2645 steps/s, about 7.4%, exceeding the 1.2011 floor.
The candidate completed 360 steps versus 335, clearing the 341-step minimum.
Both runs exited successfully, passed causality with zero maximum deviation, and stayed within budget.
The main confounder is unequal training exposure: the candidate processed 47.2M versus 43.9M tokens in nearly equal training time, so the BPB gain does not isolate a per-step quality improvement.
These are separate short runs, so run-to-run timing variability remains unmeasured; this is smoke-level evidence, not full-run confirmation.
