# Verdict: DISCARD — 20260923-unet-skip-zero-init-r2 (full)

- Hypothesis: At the fixed 1800 s budget the control (2071 steps) already sits just under the ~2195-step epoch boundary, and every throughput-only variant that crossed it (2346-2716 steps) came back CONFOUNDED with worse BPB, so the remaining lever is loss-per-token, not steps. Zero-initialized U-Net style skip connections (modded-nanogpt records 8+: encoder block outputs added, LIFO, into the decoder blocks' residual stream through a learnable per-channel vector) shorten the gradient path to shallow features and let late blocks read early representations before they were overwritten. Because the skip vectors start at exactly zero the forward graph is bit-identical to the control at step 0 (same init RNG draws, no new torch RNG), so any BPB delta is causally attributable to the learned skips. Distinct from the existing input-dependent x0 gate (reads the embedding, not block outputs), the frozen block nudge, and the final-output pooling; the previous r1 proposal was judged architecturally bounded and non-duplicate and was returned only for (a) an incorrect optimizer-routing plan and (b) an over-broad data-wall claim, both fixed here.
- Single causal change: One architecture change confined to train.py: new module-level toggle USE_UNET_SKIP (default True; env NEURON_COMPETITION_R1_UNET_SKIP='0' disables; CLI --unet-skip/--no-unet-skip; appended to ARCH_TOGGLE_NAMES so the checkpoint persists it and load_for_eval rebuilds the identical graph). When on, GPT owns n_enc = depth // 2 learnable FP32 vectors of shape (n_embd,) initialized to ZERO (nn.ParameterList self.unet_skip_gates). In GPT.forward the outputs of the first n_enc blocks are stacked; immediately before decoder block n_enc + j (j = 0..n_enc-1) the residual becomes x = x + unet_skip_gates[j] * (output of encoder block n_enc-1-j), i.e. the usual LIFO pairing. These vectors are EXPLICITLY appended to the existing gate/conv AdamW parameter list inside setup_optimizer (the group that already holds the 1-D gate tensors), inheriting its lr/betas/wd; they are never placed in the scalar group nor in Muon. Nothing else changes: block-call signature, all norm/dtype policies, x0 gate, value embeddings, n-gram VE, optimizer hyperparameters, schedule, data, evaluator all untouched.
- Flags: ``
- Commit: `7e500778265cf2c55b39ab00666b73072970f089`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260923-unet-skip-zero-init-r2-full-r3`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-full-r1`
- Budget: num_steps=5000 max_train_seconds=1800
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- public_val_bpb delta +0.00266 did not meet threshold -0.00200

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.008391 | 1.005734 |
| num_steps | 2041.0 | 2071.0 |
| steps/s | 1.1434814275309542 | 1.1603541012998655 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

DISCARD at the full tier: candidate public_val_bpb was 1.008391 versus control 1.005734.
The delta was +0.002657 (worse), failing the required improvement threshold of −0.002000.
The candidate enabled U-Net skips (unet_skip=3); both runs reported 592.9M parameters and matched seed, host, compiler flags, and training budget.
A training-exposure confounder is the candidate’s lower throughput: 1.14348 versus 1.16035 steps/s, yielding 2041 versus 2071 steps and 267.5M versus 271.5M tokens.
Thus, this fixed-time comparison does not isolate the architecture’s effect at equal training exposure.
Both runs passed causality checks, completed successfully, and stayed within the charged budget.
