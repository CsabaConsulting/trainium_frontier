# Verdict: DISCARD — 20260923-engram-only-default-shape (smoke)

- Hypothesis: The hashed n-gram memory is the largest fixed-budget quality lever measured on this commit: removing it (--no-ngram-ve) gained +23% steps yet worsened val_bpb by +0.0204 (1.026131 vs 1.005734), and halving its dim gained +6.3% steps yet worsened val_bpb by +0.0011 (1.007083 vs 1.005967). train.py at 9a4caa7b already contains a complete gated conditional-memory read-out of the same mechanism family (--engram with --engram-heads/--engram-key-dim/--engram-mem-dim/--engram-orders/--engram-conv-kernel/--engram-share-ngram-tables/--engram-site/--engram-table-mult/--engram-bf16) that has never been launched in the autoloop ledger or closed in DECISION_LOG.md. If the untested Engram read-out (context-gated multi-head lookup plus short causal conv) is enabled at its own defaults while every other option stays at the pinned-control defaults, the model will extract more loss-per-token benefit from n-gram memory than the plain ngram-ve path alone, lowering official val_bpb at comparable step count; if step rate drops more than ~10% or the memory read-out adds no quality, the hypothesis is falsified.
- Single causal change: Pass --engram only. Every other option stays at the 9a4caa7b defaults used by the two pinned controls: identical launch shape (NEURON_LOGICAL_NC_CONFIG=2, torchrun --nproc_per_node=4), seed 42, --num-steps 5000, --max-train-seconds 1800, NEURON_CC_FLAGS='--optlevel=1'. No engram sub-option is set (all --engram-* values remain at their code defaults), no precision flag, no kernel flag, no optimizer or schedule switch, no other architecture switch, no change to ngram-ve, eval, data, or tokenizer.
- Flags: `--engram`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260923-engram-only-default-shape-smoke-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- smoke step rate 1.1232 < floor 1.1513 steps/s

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.187724 | 1.165855 |
| num_steps | 320.0 | 334.0 |
| steps/s | 1.1232011232011232 | 1.1748153359127682 |
| params (M) | 660.4 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

DISCARD at the smoke tier: candidate throughput was 1.1232 steps/s, below the required 1.1513 steps/s floor and about 4.4% slower than control.
Validation BPB was also worse: 1.187724 versus 1.165855, an increase of 0.021869.
Enabling engram increased parameters from 592.9M to 660.4M, with source hashes, seed, host, compiler flags, and launch shape otherwise matched.
The quality comparison is confounded by unequal training progress: candidate completed 320 steps versus 334 in nearly identical training time, so the BPB gap does not isolate engram’s effect at equal steps.
Both runs completed successfully, passed causality checks, and stayed within budget; these do not override the throughput gate.
