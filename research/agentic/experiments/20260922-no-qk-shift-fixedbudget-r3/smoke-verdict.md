# Verdict: KEEP — 20260922-no-qk-shift-fixedbudget-r3 (smoke)

- Hypothesis: Both pinned controls resolve to qk_shift=True qk_beta=0.0 qk_freeze=True (ledger oneliners for al-20260922-control-trn21/trn22-lnc2x4-9a4caa7b-full-r1). HYPOTHESIS H1 (throughput): the frozen-zero QK token-shift branch still executes per-layer shift/scale ops on q and k in forward and backward because beta is a runtime tensor, not a compile-time constant, so the compiler cannot eliminate them; passing --no-qk-shift removes that work and yields a small but measurable step-count gain (+1% to +3%) in the fixed 1800 s budget. HYPOTHESIS H2 (near-equivalence, weaker): with beta frozen at exactly zero the branch contributes nothing to model outputs, so at equal step count the loss trajectory should be indistinguishable up to seed-42 init/RNG-consumption drift; the extra steps then translate into a small public_val_bpb reduction. Neither H1 nor H2 is asserted as established fact: the exact mixing equation, whether a frozen beta parameter exists in the optimizer/param count, and compiler DCE behavior are not verified from source here and are exactly what the run tests. Related but non-isolating evidence: every fs-w* row from fs-w6 onward ran with --no-qk-shift bundled with other flags and completed with causality_passed=true, so the code path is stable; the proposal verifier for r2 confirmed the isolated default-shape direction is not closed by the supplied decision log. This r3 corrects the r2 defects: mechanism framed as hypotheses, fixed-budget BPB test separated from equivalence attribution, and protocol section 5 CONFOUNDED rule applied as written (candidate >10% SHORT on steps).
- Single causal change: Pass --no-qk-shift only. Every other option stays at the 9a4caa7b defaults used by the two pinned controls: identical launch shape (NEURON_LOGICAL_NC_CONFIG=2, torchrun --standalone --nproc_per_node=4, --num-steps 5000, --max-train-seconds 1800, --seed 42), identical environment (NEURON_CC_FLAGS exactly '--optlevel=1', NEURON_COMPETITION_R1_SEED=42, PYTHONUNBUFFERED=1), no precision, kernel, optimizer, schedule, batch, evaluation, or other architecture switch. --qk-shift-freeze and any qk beta option are not passed. Matched comparison is the same-host pinned control (trn21: 2071 steps / 1.005734; trn22: 2076 steps / 1.005967) on public_val_bpb from the controller's shared scorer; no private-shard claim. Outcome classes, mutually exclusive and evaluated in order: (1) INVALID if crash, non-finite loss, causality check failure, or any launch/environment mismatch; (2) CONFOUNDED (protocol section 5) if candidate steps < 0.90 x same-host control steps; (3) RATE-ANOMALY if candidate steps > 1.10 x control steps: record as a throughput-only result, BPB comparison provisional pending a same-host control repeat (separate conservative upper rule, not the protocol rule); (4) FALSIFIED-INERT if steps are within +/-1.0% of control AND |delta BPB| <= 0.001: the branch costs nothing measurable, direction closed; (5) SUCCESS if steps >= 1.01 x control AND public_val_bpb <= control - 0.001; (6) FALSIFIED-QUALITY if public_val_bpb > control + 0.001 at any step count in [0.90,1.10]: the branch is not output-inert or init/RNG drift dominates, direction closed; (7) INCONCLUSIVE otherwise. Secondary, non-scoring attribution: compare logged training loss at matched step indices over the common prefix (noting the time-based schedule gives slightly different LR at equal step index, so this is provisional support for or against H2, never a verdict). Also record num_params_M versus the controls' 592.9 as an observation (a change would indicate the flag removes trainable capacity and would be reported, not used as an abort).
- Flags: `--no-qk-shift`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn22 (`ip-172-31-32-253`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-no-qk-shift-fixedbudget-r3-smoke-r1`; control label: `al-20260922-control-trn22-lnc2x4-9a4caa7b-smoke-r2`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- correctness gates passed; step rate within tolerance; eligible for full run

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.165482 | 1.165604 |
| num_steps | 335.0 | 335.0 |
| steps/s | 1.178747361013371 | 1.1775043936731107 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

KEEP at smoke tier: correctness gates passed, step rate met the floor, and the candidate is eligible for a full run.
Disabling QK shift yielded validation BPB of 1.165482 versus 1.165604, a decrease of 0.000122, comfortably within the allowed +0.003 delta.
Both runs completed 335 steps, exceeding the 334-step minimum, with identical 592.9M parameter counts.
Candidate throughput was 1.178747 versus 1.177504 steps/s, approximately 0.106% higher.
Both runs passed causality checks with zero maximum absolute error, exited successfully, and stayed within budget and evaluation timeout.
The comparison matches source hashes, clean commit, seed, host, compiler settings, and launch shape; QK shift is the only recorded model-option change.
Potential confounder: the runs occurred roughly ten hours apart, so transient system conditions could explain the tiny throughput difference.
This single short comparison establishes smoke eligibility, not a reliable quality or speed advantage.
