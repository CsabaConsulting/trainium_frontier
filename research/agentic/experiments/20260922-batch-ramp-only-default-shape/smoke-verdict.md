# Verdict: KEEP — 20260922-batch-ramp-only-default-shape (smoke)

- Hypothesis: At the fixed 1800 s budget the default-shape model (6x640 + hashed n-gram tables, 592.9M params) is update-count-limited early and noise-limited late: the two pinned controls complete only ~2071/2076 optimizer steps at a constant grad_accum=4. Enabling the already-implemented but default-off batch ramp (grad_accum 2 -> 4 -> 8 at 35%/75% of the time budget with sqrt LR scaling, all resolved values already recorded in the control rows) roughly doubles the optimizer-update rate during the first 35% of the budget and doubles the effective batch (lower gradient noise) during the low-LR tail while keeping total tokens, precision, kernels, optimizer and architecture unchanged. This should lower final public val_bpb versus the same-host control.
- Single causal change: Pass --batch-ramp only. Nothing else changes: no precision flag, no kernel flag, no optimizer/architecture switch, no --batch-ramp-grad-accum/--batch-ramp-bounds/--batch-ramp-lr-exp overrides (their resolved defaults are already recorded in both pinned control rows as grad_accum=(2, 4, 8) bounds=(0.35, 0.75) lr_exp=0.5 ref_grad_accum=4, so the only behavioral delta is the ramp being active instead of a constant grad_accum=4). Launch must otherwise be byte-identical to the same-host pinned control: NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --num-steps 5000 --max-train-seconds 1800 --seed 42, NEURON_CC_FLAGS='--optlevel=1' exactly, NEURON_COMPETITION_R1_SEED=42, commit 9a4caa7b.
- Flags: `--batch-ramp`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-batch-ramp-only-default-shape-smoke-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- correctness gates passed; step rate within tolerance; eligible for full run

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.163452 | 1.165855 |
| num_steps | 395.0 | 334.0 |
| steps/s | 1.387912860154603 | 1.1748153359127682 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

KEEP at smoke tier: correctness gates passed and step rate met tolerance; the candidate is eligible for a full run.
Candidate validation BPB was 1.163452 versus 1.165855 for control, an improvement of 0.002403.
It completed 395 versus 334 steps in nearly equal training time (284.6 versus 284.3 seconds).
Step rate was 1.3879 versus 1.1748 steps/s, about 18.1% higher and above the 1.1513 floor.
Both runs passed causality checks, exited successfully, and stayed within budget and evaluation timeout.
Model size (592.9M), seed, source hashes, host, and execution configuration matched; the intended change was enabling default batch ramp.
The key comparison confounder is variable gradient accumulation: optimizer steps represent different token counts, so higher steps/s does not imply higher token throughput.
The candidate processed 42.3M tokens versus 43.8M for control; this is not an equal-token or equal-update comparison.
These single-seed smoke results support proceeding, not a confirmed full-run quality gain.
