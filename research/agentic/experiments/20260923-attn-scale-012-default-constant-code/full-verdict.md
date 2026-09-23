# Verdict: DISCARD — 20260923-attn-scale-012-default-constant-code (full)

- Hypothesis: At the pinned config every attention layer applies parameter-free per-head normalization to q and k (rope_norm inside CausalSelfAttention.forward, per the 2026-09-23 verifier's source reading), so the attainable logit magnitude is bounded by the fixed softmax scale: with |q|,|k| RMS-normalized over head_dim d, the largest logit is d*scale, i.e. sqrt(d) at the backend default 1/sqrt(d). The optimizer therefore cannot sharpen attention beyond that ceiling except by distorting q/k directions. modded-nanogpt (records with QK-norm) found a fixed scale of 0.12 (~1.36x sharper than 1/sqrt(128)) improves loss per step at zero cost. train.py already carries the complete checkpoint-safe plumbing for this (module-level ATTN_SCALE=0.0 meaning 'backend default', GPTConfig.attn_scale, CausalSelfAttention.self.attn_scale forwarded as scale= to every SDPA call and to nki_attention) but the default is the backend value and no CLI flag reaches it at 9a4caa7b (deterministic check: --attn-scale does not exist). Setting ATTN_SCALE to 0.12 is a pure per-step-quality change: the scale is a scalar kwarg of the same SDPA graph, so step count should stay at ~2071 and the comparison against the pinned trn21/trn22 controls is not step-confounded (unlike every recent throughput-driven variant that crossed the ~2195-step data wall and got worse).
- Single causal change: One architecture/numerics constant change confined to train.py: the existing module-level ATTN_SCALE default changes from 0.0 (use backend default 1/sqrt(head_dim)) to 0.12, flowing through the EXISTING GPTConfig.attn_scale -> CausalSelfAttention.self.attn_scale -> scale= plumbing to every attention backend call. No new tensor, parameter, op, or scale path is added; no second scale is introduced; the '<= 0 means backend default' semantics are preserved so the control value remains reachable. Everything else stays at the 9a4caa7b defaults and the pinned launch shape (NEURON_LOGICAL_NC_CONFIG=2, torchrun --nproc_per_node=4, --num-steps 5000 --max-train-seconds 1800 --seed 42, NEURON_CC_FLAGS='--optlevel=1'): no precision flag, no kernel flag, no optimizer or schedule switch, no data/eval change.
- Flags: ``
- Commit: `7c7449648547692e0877cc26b6b4a70c29cbf251`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260923-attn-scale-012-default-constant-code-full-r2`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-full-r1`
- Budget: num_steps=5000 max_train_seconds=1800
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- public_val_bpb delta +0.00403 did not meet threshold -0.00200

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.009764 | 1.005734 |
| num_steps | 2071.0 | 2071.0 |
| steps/s | 1.1604191180590575 | 1.1603541012998655 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

DISCARD at the full tier: candidate public_val_bpb was 1.009764 versus control 1.005734, a worsening of +0.004030 rather than the required delta of −0.002000 or better.
Both runs completed 2,071 steps and 271.5M tokens with 592.9M parameters; throughput was effectively identical at about 1.1604 steps/s.
Seed, host, compiler flags, and logged training settings matched. Both passed causality checks and stayed within the charged budget.
Potential confounder: the runs used different commits on different dates. Only train.py differs among the recorded source hashes, but the ledger alone cannot verify that the attention-scale constant was the sole code change.
The candidate ledger’s “keep” status does not override the deterministic DISCARD verdict.
