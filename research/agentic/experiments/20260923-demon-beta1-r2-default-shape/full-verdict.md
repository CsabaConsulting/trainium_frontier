# Verdict: DISCARD — 20260923-demon-beta1-r2-default-shape (full)

- Hypothesis: Annealing the AdamW groups' beta1 across the LR warmdown (Recursive 'Demon' schedule, train.py DEMON_BETA1 block, lines ~414-448) lets the embedding-family updates (wte, lm_head, unigram value embeddings, adamw fallback) and the 0.9-beta1 scalar/gate/local-conv groups track the sharply decaying LR instead of averaging over a stale first-moment window during the final ~1500 low-LR steps that shape the final weights most. The repo already adopted the same tail-momentum-reduction mechanism for Muon (MUON_MOMENTUM_COOLDOWN_FRAC 0.045 -> MUON_MOMENTUM_FINAL 0.85, part of the confirmed T0.4 stack, train.py lines 508-526), but the AdamW groups still run a fixed beta1 to the last step. Mechanism as the source actually implements it (correcting the prior proposal's wrong claim): the training loop writes per-step Python-float betas into group['betas']; _adamw_step_foreach consumes 1-beta1 as the scalar weight of lerp_ (which the muon_momentum docstring records as measured NOT to bake into the Neuron graph), and _adamw_state computes scalar bias corrections that already vary every step in the baseline (they depend on the step count), so the schedule changes values, not the kind of op or scalar plumbing. Throughput neutrality is therefore expected but is a testable prediction checked by the smoke step-rate gate, not a guarantee. The rmsprop n-gram tables have no first moment and are excluded by kind; Muon is untouched. Prior verifier (20260923-demon-beta1-only-default-shape) found no closed equivalent in the decision log or source annotations and judged the flag correctly isolated; this r2 only corrects the two flagged claims.
- Single causal change: Pass --demon-beta1 only (DEMON_BETA1 True with in-source defaults: DEMON_BETA1_FINAL 0.55 quoted against DEMON_BETA1_REF 0.8, applied as a multiplicative shrink of each AdamW group's own initial beta1, DEMON_BETA1_START_FRAC -1.0 i.e. the anneal starts where the LR warmdown starts at progress 1 - WARMDOWN_RATIO = 0.25). Every other option stays at the 9a4caa7b defaults of the two pinned controls: NEURON_LOGICAL_NC_CONFIG=2, torchrun --nproc_per_node=4, --num-steps 5000 --max-train-seconds 1800 --seed 42, NEURON_CC_FLAGS='--optlevel=1'. No precision flag, no kernel flag, no other optimizer switch (--cautious-wd, --wd-decay-to-zero, --muon-lr-fp32, --adam-every-n-lr-comp, --ngram-ve-*), no architecture, data or evaluator change. The AdamW beta1 schedule is the sole causal option.
- Flags: `--demon-beta1`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260923-demon-beta1-r2-default-shape-full-r3`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-full-r1`
- Budget: num_steps=5000 max_train_seconds=1800
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- public_val_bpb delta +0.00019 did not meet threshold -0.00200

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.005924 | 1.005734 |
| num_steps | 2071.0 | 2071.0 |
| steps/s | 1.1602890918258726 | 1.1603541012998655 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

DISCARD at the full tier: enabling --demon-beta1 produced public_val_bpb 1.005924 versus 1.005734 for control.
The candidate-minus-control delta was +0.000190, a slight regression rather than the required improvement of at least 0.002000.
Both runs completed 2,071 steps and processed 271.5M tokens with 592.9M parameters; throughput was essentially identical at 1.1603 steps/s.
Seed, source hashes, commit, host, compiler settings, and other recorded model settings matched; the intended change was DEMON beta1.
Both runs passed causality checks and stayed within the charged budget.
No material recorded configuration confounder is evident. Separate run times and a single seed leave run-to-run variability unquantified.
