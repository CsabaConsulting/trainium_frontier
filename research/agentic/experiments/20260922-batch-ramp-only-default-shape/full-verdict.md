# Verdict: CONFOUNDED — 20260922-batch-ramp-only-default-shape (full)

- Hypothesis: At the fixed 1800 s budget the default-shape model (6x640 + hashed n-gram tables, 592.9M params) is update-count-limited early and noise-limited late: the two pinned controls complete only ~2071/2076 optimizer steps at a constant grad_accum=4. Enabling the already-implemented but default-off batch ramp (grad_accum 2 -> 4 -> 8 at 35%/75% of the time budget with sqrt LR scaling, all resolved values already recorded in the control rows) roughly doubles the optimizer-update rate during the first 35% of the budget and doubles the effective batch (lower gradient noise) during the low-LR tail while keeping total tokens, precision, kernels, optimizer and architecture unchanged. This should lower final public val_bpb versus the same-host control.
- Single causal change: Pass --batch-ramp only. Nothing else changes: no precision flag, no kernel flag, no optimizer/architecture switch, no --batch-ramp-grad-accum/--batch-ramp-bounds/--batch-ramp-lr-exp overrides (their resolved defaults are already recorded in both pinned control rows as grad_accum=(2, 4, 8) bounds=(0.35, 0.75) lr_exp=0.5 ref_grad_accum=4, so the only behavioral delta is the ramp being active instead of a constant grad_accum=4). Launch must otherwise be byte-identical to the same-host pinned control: NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --num-steps 5000 --max-train-seconds 1800 --seed 42, NEURON_CC_FLAGS='--optlevel=1' exactly, NEURON_COMPETITION_R1_SEED=42, commit 9a4caa7b.
- Flags: `--batch-ramp`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-batch-ramp-only-default-shape-full-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-full-r1`
- Budget: num_steps=5000 max_train_seconds=1800
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- steps differ by more than 10% (2397.0 vs 2071.0)

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.01219 | 1.005734 |
| num_steps | 2397.0 | 2071.0 |
| steps/s | 1.343609865470852 | 1.1603541012998655 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

The full-tier verdict is CONFOUNDED.
The batch-ramp candidate scored 1.012190 BPB versus 1.005734 for control, an observed worsening of 0.006456 BPB (lower is better).
The named confounder is step count: 2,397 versus 2,071, about 15.7% more steps, exceeding the 10% tolerance.
Step rates were 1.344 versus 1.160 steps/second under nearly identical training times of about 1,784 seconds.
Batch ramp changed gradient accumulation; more steps did not mean more tokens (264.9M versus 271.5M).
Both runs used 592.9M parameters, the same seed, commit, source hashes, and hardware configuration.
Both passed causality checks and stayed within the charged budget, but the step mismatch prevents an unconfounded attribution of the BPB difference to batch ramp.
