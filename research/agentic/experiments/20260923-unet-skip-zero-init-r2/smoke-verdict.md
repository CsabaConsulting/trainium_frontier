# Verdict: KEEP — 20260923-unet-skip-zero-init-r2 (smoke)

- Hypothesis: At the fixed 1800 s budget the control (2071 steps) already sits just under the ~2195-step epoch boundary, and every throughput-only variant that crossed it (2346-2716 steps) came back CONFOUNDED with worse BPB, so the remaining lever is loss-per-token, not steps. Zero-initialized U-Net style skip connections (modded-nanogpt records 8+: encoder block outputs added, LIFO, into the decoder blocks' residual stream through a learnable per-channel vector) shorten the gradient path to shallow features and let late blocks read early representations before they were overwritten. Because the skip vectors start at exactly zero the forward graph is bit-identical to the control at step 0 (same init RNG draws, no new torch RNG), so any BPB delta is causally attributable to the learned skips. Distinct from the existing input-dependent x0 gate (reads the embedding, not block outputs), the frozen block nudge, and the final-output pooling; the previous r1 proposal was judged architecturally bounded and non-duplicate and was returned only for (a) an incorrect optimizer-routing plan and (b) an over-broad data-wall claim, both fixed here.
- Single causal change: One architecture change confined to train.py: new module-level toggle USE_UNET_SKIP (default True; env NEURON_COMPETITION_R1_UNET_SKIP='0' disables; CLI --unet-skip/--no-unet-skip; appended to ARCH_TOGGLE_NAMES so the checkpoint persists it and load_for_eval rebuilds the identical graph). When on, GPT owns n_enc = depth // 2 learnable FP32 vectors of shape (n_embd,) initialized to ZERO (nn.ParameterList self.unet_skip_gates). In GPT.forward the outputs of the first n_enc blocks are stacked; immediately before decoder block n_enc + j (j = 0..n_enc-1) the residual becomes x = x + unet_skip_gates[j] * (output of encoder block n_enc-1-j), i.e. the usual LIFO pairing. These vectors are EXPLICITLY appended to the existing gate/conv AdamW parameter list inside setup_optimizer (the group that already holds the 1-D gate tensors), inheriting its lr/betas/wd; they are never placed in the scalar group nor in Muon. Nothing else changes: block-call signature, all norm/dtype policies, x0 gate, value embeddings, n-gram VE, optimizer hyperparameters, schedule, data, evaluator all untouched.
- Flags: ``
- Commit: `7e500778265cf2c55b39ab00666b73072970f089`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260923-unet-skip-zero-init-r2-smoke-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- correctness gates passed; step rate within tolerance; eligible for full run

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.167766 | 1.165855 |
| num_steps | 330.0 | 334.0 |
| steps/s | 1.1583011583011584 | 1.1748153359127682 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

KEEP at smoke tier: correctness gates passed and step rate remained within tolerance, making the candidate eligible for a full run.
The U-Net skip candidate recorded 1.167766 validation BPB versus 1.165855 for control, a worsening of 0.001911.
Its step rate was 1.158301 versus 1.174815 steps/s, about 1.41% slower but above the 1.151319 floor.
Both runs passed causality with zero maximum absolute error, exited successfully, and stayed within budget and evaluation timeout.
Reported parameter count, seed, host, compiler flags, sequence length, and world size matched.
The main confounder for the BPB comparison is unequal training exposure: the candidate completed 330 steps and 43.3M tokens versus 334 steps and 43.8M tokens.
Runs also occurred on different dates and commits; the listed source hashes differ only for train.py.
This smoke result supports proceeding, not a claim of improved validation quality.
