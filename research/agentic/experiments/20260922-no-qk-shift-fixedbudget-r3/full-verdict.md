# Verdict: DISCARD — 20260922-no-qk-shift-fixedbudget-r3 (full)

- Hypothesis: Both pinned controls resolve to qk_shift=True qk_beta=0.0 qk_freeze=True (ledger oneliners for al-20260922-control-trn21/trn22-lnc2x4-9a4caa7b-full-r1). HYPOTHESIS H1 (throughput): the frozen-zero QK token-shift branch still executes per-layer shift/scale ops on q and k in forward and backward because beta is a runtime tensor, not a compile-time constant, so the compiler cannot eliminate them; passing --no-qk-shift removes that work and yields a small but measurable step-count gain (+1% to +3%) in the fixed 1800 s budget. HYPOTHESIS H2 (near-equivalence, weaker): with beta frozen at exactly zero the branch contributes nothing to model outputs, so at equal step count the loss trajectory should be indistinguishable up to seed-42 init/RNG-consumption drift; the extra steps then translate into a small public_val_bpb reduction. Neither H1 nor H2 is asserted as established fact: the exact mixing equation, whether a frozen beta parameter exists in the optimizer/param count, and compiler DCE behavior are not verified from source here and are exactly what the run tests. Related but non-isolating evidence: every fs-w* row from fs-w6 onward ran with --no-qk-shift bundled with other flags and completed with causality_passed=true, so the code path is stable; the proposal verifier for r2 confirmed the isolated default-shape direction is not closed by the supplied decision log. This r3 corrects the r2 defects: mechanism framed as hypotheses, fixed-budget BPB test separated from equivalence attribution, and protocol section 5 CONFOUNDED rule applied as written (candidate >10% SHORT on steps).
- Single causal change: Pass --no-qk-shift only. Every other option stays at the 9a4caa7b defaults used by the two pinned controls: identical launch shape (NEURON_LOGICAL_NC_CONFIG=2, torchrun --standalone --nproc_per_node=4, --num-steps 5000, --max-train-seconds 1800, --seed 42), identical environment (NEURON_CC_FLAGS exactly '--optlevel=1', NEURON_COMPETITION_R1_SEED=42, PYTHONUNBUFFERED=1), no precision, kernel, optimizer, schedule, batch, evaluation, or other architecture switch. --qk-shift-freeze and any qk beta option are not passed. Matched comparison is the same-host pinned control (trn21: 2071 steps / 1.005734; trn22: 2076 steps / 1.005967) on public_val_bpb from the controller's shared scorer; no private-shard claim. Outcome classes, mutually exclusive and evaluated in order: (1) INVALID if crash, non-finite loss, causality check failure, or any launch/environment mismatch; (2) CONFOUNDED (protocol section 5) if candidate steps < 0.90 x same-host control steps; (3) RATE-ANOMALY if candidate steps > 1.10 x control steps: record as a throughput-only result, BPB comparison provisional pending a same-host control repeat (separate conservative upper rule, not the protocol rule); (4) FALSIFIED-INERT if steps are within +/-1.0% of control AND |delta BPB| <= 0.001: the branch costs nothing measurable, direction closed; (5) SUCCESS if steps >= 1.01 x control AND public_val_bpb <= control - 0.001; (6) FALSIFIED-QUALITY if public_val_bpb > control + 0.001 at any step count in [0.90,1.10]: the branch is not output-inert or init/RNG drift dominates, direction closed; (7) INCONCLUSIVE otherwise. Secondary, non-scoring attribution: compare logged training loss at matched step indices over the common prefix (noting the time-based schedule gives slightly different LR at equal step index, so this is provisional support for or against H2, never a verdict). Also record num_params_M versus the controls' 592.9 as an observation (a change would indicate the flag removes trainable capacity and would be reported, not used as an abort).
- Flags: `--no-qk-shift`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn22 (`ip-172-31-32-253`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-no-qk-shift-fixedbudget-r3-full-r1`; control label: `al-20260922-control-trn22-lnc2x4-9a4caa7b-full-r1`
- Budget: num_steps=5000 max_train_seconds=1800
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- public_val_bpb delta -0.00020 did not meet threshold -0.00200

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.005766 | 1.005967 |
| num_steps | 2080.0 | 2076.0 |
| steps/s | 1.1653313911143481 | 1.1634814773300455 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

DISCARD at the full tier: disabling QK shift reduced public_val_bpb from 1.005967 to 1.005766.
The candidate-minus-control delta was -0.000201, short of the required -0.002000 threshold.
Both runs used the same clean source commit, seed, host, launch configuration, and 592.9M parameters; QK shift was the only recorded model-option difference.
The main confounder is unequal training progress under the fixed 1,800-second budget: the candidate completed 2,080 steps versus 2,076.
Step rates were 1.16533 versus 1.16348 steps/s, so the small BPB gain cannot be attributed solely to QK shift at equal training steps.
Both runs passed causality checks and stayed within the charged budget, with no evaluation timeout.
