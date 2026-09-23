# Verdict: KEEP — 20260923-ngram-ve-cooldown-lr-flag-only-default-shape (smoke)

- Hypothesis: The hashed n-gram value-embedding tables (the single largest measured win in the repo, 4 tables of 1048576x128 on layers 1 and -1, RMSProp-trained) are the only large parameter family whose learning rate is held FLAT through the 75%-of-budget LR warmdown (NGRAM_VE_FLAT_LR default True). Every other tensor's step size decays to FINAL_LR_FRAC=0.05 of peak, which is what lets the final ~1500 cooldown steps converge instead of bounce; the tables never get that annealing, so their entries carry full-amplitude RMSProp noise into the final checkpoint. Letting the tables follow the same cooldown multiplier (--no-ngram-ve-flat-lr) should reduce end-of-run table noise and lower val_bpb at an unchanged step count, because it is a pure per-step schedule change that adds no ops, tensors, or parameters.
- Single causal change: Pass --no-ngram-ve-flat-lr only (NGRAM_VE_FLAT_LR False: the n-gram VE table learning rate follows the run's LR warmdown instead of being held flat). Every other option stays at the 9a4caa7b defaults used by the two pinned controls: identical launch shape (NEURON_LOGICAL_NC_CONFIG=2, torchrun --nproc_per_node=4, --num-steps 5000 --max-train-seconds 1800 --seed 42, NEURON_CC_FLAGS='--optlevel=1'). No precision flag, no kernel flag, no optimizer switch, no architecture switch, no --ngram-ve-lr/--ngram-ve-opt/--ngram-ve-beta2 change, no data/evaluator change.
- Flags: `--no-ngram-ve-flat-lr`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260923-ngram-ve-cooldown-lr-flag-only-default-shape-smoke-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- correctness gates passed; step rate within tolerance; eligible for full run

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.168643 | 1.165855 |
| num_steps | 334.0 | 334.0 |
| steps/s | 1.175228712174525 | 1.1748153359127682 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

KEEP at smoke tier: correctness gates passed and step rate was within tolerance, making the candidate eligible for a full run.
The candidate changes only n-gram VE flat_lr from True to False, so its table learning rate follows the run’s warmdown.
Validation BPB was slightly worse: 1.168643 versus 1.165855, a delta of +0.002788.
Both runs completed 334 steps; candidate throughput was 1.17523 versus 1.17482 steps/s, above the supplied floor of 1.15132.
Parameter count, seed, source hashes, host, and execution configuration matched; both runs exited successfully and passed causality checks.
Potential timing confounder: the runs occurred on different dates and had different startup overheads, so wall-time differences should not be attributed to the flag.
This short, single-seed smoke comparison establishes eligibility, not a quality improvement or full-run outcome.
