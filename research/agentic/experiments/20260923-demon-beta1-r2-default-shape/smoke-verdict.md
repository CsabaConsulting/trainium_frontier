# Verdict: KEEP — 20260923-demon-beta1-r2-default-shape (smoke)

- Hypothesis: Annealing the AdamW groups' beta1 across the LR warmdown (Recursive 'Demon' schedule, train.py DEMON_BETA1 block, lines ~414-448) lets the embedding-family updates (wte, lm_head, unigram value embeddings, adamw fallback) and the 0.9-beta1 scalar/gate/local-conv groups track the sharply decaying LR instead of averaging over a stale first-moment window during the final ~1500 low-LR steps that shape the final weights most. The repo already adopted the same tail-momentum-reduction mechanism for Muon (MUON_MOMENTUM_COOLDOWN_FRAC 0.045 -> MUON_MOMENTUM_FINAL 0.85, part of the confirmed T0.4 stack, train.py lines 508-526), but the AdamW groups still run a fixed beta1 to the last step. Mechanism as the source actually implements it (correcting the prior proposal's wrong claim): the training loop writes per-step Python-float betas into group['betas']; _adamw_step_foreach consumes 1-beta1 as the scalar weight of lerp_ (which the muon_momentum docstring records as measured NOT to bake into the Neuron graph), and _adamw_state computes scalar bias corrections that already vary every step in the baseline (they depend on the step count), so the schedule changes values, not the kind of op or scalar plumbing. Throughput neutrality is therefore expected but is a testable prediction checked by the smoke step-rate gate, not a guarantee. The rmsprop n-gram tables have no first moment and are excluded by kind; Muon is untouched. Prior verifier (20260923-demon-beta1-only-default-shape) found no closed equivalent in the decision log or source annotations and judged the flag correctly isolated; this r2 only corrects the two flagged claims.
- Single causal change: Pass --demon-beta1 only (DEMON_BETA1 True with in-source defaults: DEMON_BETA1_FINAL 0.55 quoted against DEMON_BETA1_REF 0.8, applied as a multiplicative shrink of each AdamW group's own initial beta1, DEMON_BETA1_START_FRAC -1.0 i.e. the anneal starts where the LR warmdown starts at progress 1 - WARMDOWN_RATIO = 0.25). Every other option stays at the 9a4caa7b defaults of the two pinned controls: NEURON_LOGICAL_NC_CONFIG=2, torchrun --nproc_per_node=4, --num-steps 5000 --max-train-seconds 1800 --seed 42, NEURON_CC_FLAGS='--optlevel=1'. No precision flag, no kernel flag, no other optimizer switch (--cautious-wd, --wd-decay-to-zero, --muon-lr-fp32, --adam-every-n-lr-comp, --ngram-ve-*), no architecture, data or evaluator change. The AdamW beta1 schedule is the sole causal option.
- Flags: `--demon-beta1`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260923-demon-beta1-r2-default-shape-smoke-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- correctness gates passed; step rate within tolerance; eligible for full run

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.166347 | 1.165855 |
| num_steps | 334.0 | 334.0 |
| steps/s | 1.1744022503516176 | 1.1748153359127682 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

KEEP at smoke tier: correctness gates passed, step rate is within tolerance, and the candidate is eligible for a full run.
The candidate enables --demon-beta1 with default final/reference settings of 0.55/0.8; the control disables it.
Candidate BPB is 1.166347 versus 1.165855, a worsening of 0.000492, below the allowed 0.003 delta.
Both completed 334 steps, exceeding the 320-step minimum, with identical 592.9M parameter counts.
Candidate throughput is 1.174402 steps/s versus 1.174815, about 0.035% slower and above the supplied 1.151319 floor.
Both passed causality checks, exited successfully, and stayed within budget and evaluation timeout.
Source hashes, commit, seed, host, and listed training configuration otherwise match.
Potential confounder: the runs occurred on different days, so runtime conditions and startup overhead may differ.
This single-seed smoke result supports progression, not a demonstrated quality improvement or full-run benefit.
