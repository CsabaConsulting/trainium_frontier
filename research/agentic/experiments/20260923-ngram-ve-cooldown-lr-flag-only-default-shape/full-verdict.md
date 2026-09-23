# Verdict: DISCARD — 20260923-ngram-ve-cooldown-lr-flag-only-default-shape (full)

- Hypothesis: The hashed n-gram value-embedding tables (the single largest measured win in the repo, 4 tables of 1048576x128 on layers 1 and -1, RMSProp-trained) are the only large parameter family whose learning rate is held FLAT through the 75%-of-budget LR warmdown (NGRAM_VE_FLAT_LR default True). Every other tensor's step size decays to FINAL_LR_FRAC=0.05 of peak, which is what lets the final ~1500 cooldown steps converge instead of bounce; the tables never get that annealing, so their entries carry full-amplitude RMSProp noise into the final checkpoint. Letting the tables follow the same cooldown multiplier (--no-ngram-ve-flat-lr) should reduce end-of-run table noise and lower val_bpb at an unchanged step count, because it is a pure per-step schedule change that adds no ops, tensors, or parameters.
- Single causal change: Pass --no-ngram-ve-flat-lr only (NGRAM_VE_FLAT_LR False: the n-gram VE table learning rate follows the run's LR warmdown instead of being held flat). Every other option stays at the 9a4caa7b defaults used by the two pinned controls: identical launch shape (NEURON_LOGICAL_NC_CONFIG=2, torchrun --nproc_per_node=4, --num-steps 5000 --max-train-seconds 1800 --seed 42, NEURON_CC_FLAGS='--optlevel=1'). No precision flag, no kernel flag, no optimizer switch, no architecture switch, no --ngram-ve-lr/--ngram-ve-opt/--ngram-ve-beta2 change, no data/evaluator change.
- Flags: `--no-ngram-ve-flat-lr`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260923-ngram-ve-cooldown-lr-flag-only-default-shape-full-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-full-r1`
- Budget: num_steps=5000 max_train_seconds=1800
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- public_val_bpb delta +0.00210 did not meet threshold -0.00200

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.007831 | 1.005734 |
| num_steps | 2072.0 | 2071.0 |
| steps/s | 1.161109554497058 | 1.1603541012998655 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

DISCARD at the full tier: candidate public_val_bpb was 1.007831 versus control 1.005734.
The candidate-minus-control delta was +0.002097 (worse), failing the required threshold of −0.002000.
The candidate disabled flat learning rate for the n-gram VE tables, making their learning rate follow the run’s warmdown.
Both runs used identical source hashes, seed 42, hardware host, compiler settings, and 592.9M parameters.
Throughput was nearly identical: 1.16111 versus 1.16035 steps/s.
A minor confounder is unequal training exposure under the time budget: 2,072 candidate steps versus 2,071 control steps; runs also occurred on different dates.
Both passed causality checks and stayed within the charged budget, with no evaluation timeout.
This single-seed comparison does not establish run-to-run variability, but it fails the specified acceptance criterion.
