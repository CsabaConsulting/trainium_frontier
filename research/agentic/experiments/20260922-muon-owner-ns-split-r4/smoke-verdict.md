# Verdict: DISCARD — 20260922-muon-owner-ns-split-r4 (smoke)

- Hypothesis: In the pinned control every rank redundantly executes the compiled Muon Newton-Schulz (NS) orthogonalization for every matrix of every same-shape batch (MuonAdamW._muon_step -> fn from _MUON_COMPILED, i.e. torch.compile(_muon_math, backend='neuron', dynamic=False)). NS is the only super-linear part of the optimizer (BF16 matmul chain on g.bfloat16() per iteration) and it is replicated 4x across the LNC=2 nproc=4 ranks. Partitioning only the NS stage by contiguous ownership and reconstructing the full orthogonalized batch with one all_gather removes ~3/4 of that work per rank at the cost of one small BF16 collective per shape group. fs-w65 measured this exact mechanism at +5.06245% complete steps after an exact numerical gate on a 117.5M dense model; fs-w70/w98/w146 retained it. Because train.py schedules by elapsed time, more steps in 1800 s should lower fixed-budget public_val_bpb, as the bf16-sdpa-input row did (2233 vs 2076 steps -> 1.002028 vs 1.005967).
- Single causal change: One optimizer-implementation change confined to train.py, inserted ONLY inside the existing compiled branch of MuonAdamW._muon_step (the branch that fetches fn from _MUON_COMPILED); the eager and NKI branches and main()'s existing compile-muon-over-nki-muon backend precedence are untouched. When the new module-level toggle USE_MUON_OWNER_COMPUTE is on (default True; env NEURON_COMPETITION_R1_MUON_OWNER_COMPUTE='0' disables; CLI --muon-owner-compute/--no-muon-owner-compute), torch.distributed is initialized with world_size > 1, and the batch has n >= 2 matrices, the compiled branch runs the SAME arithmetic as _muon_math split at the NS stage into three identically compiled helpers: (pre) out-of-place momentum/Nesterov statements and the g.bfloat16() NS-input cast on every rank, returning the new momentum buffer (rebound exactly as today) and the BF16 NS input G; (ns) the NS-input normalization plus both NS loops, moved verbatim and taking tall and ns_steps as explicit arguments, executed on every rank only for its contiguous owned slice (rank r owns [r*per_rank, min((r+1)*per_rank, n)), per_rank = ceil(n/world_size); local stack has static shape [per_rank, *shape]; unowned slots are filled with copies of G[0:1] and discarded), followed by ONE torch.distributed all_gather that reconstructs the full orthogonalized batch in rank order sliced to n; (post) the remaining statements (shape-dependent scale, second-moment update/normalization, LR/weight-decay scaling, parameter update, return tuple) on every rank on the reconstructed batch, returning exactly what _muon_math returns in the same order. When the toggle is off, world_size == 1, n < 2, or the compiled branch is not the selected backend, the unchanged _MUON_COMPILED fn executes as today. Momentum/second-moment layout, optimizer state and checkpoint format, gradient all-reduce, clipping, AdamW/ngram-ve optimizers, schedules, model, forward/eval code, data, seed, evaluator, launch shape and environment are unchanged. Full-run classification vs the same-host pinned control (trn21 2071/1.005734; trn22 2076/1.005967), evaluated in order: (1) HALT-INERT if static inspection finds NS already partitioned across ranks in the compiled branch (no patch, no run); (2) HALT-NUMERICS if the smoke log lacks a 'muon_owner_check' line for any partitioned (shape, n) group, or any such line reports ns_rel > 1.5625e-2 (2 BF16 ulps; the NS chain is BF16), delta_rel > 3.125e-2, mom_rel > 1e-5, sm_rel > 1e-3, or a non-finite value, or smoke public_val_bpb > same-host control smoke (1.165855 trn21 / 1.165604 trn22) + 0.003, or smoke causality_passed is False; (3) CONFOUNDED-SHORT if full-run steps < 90% of same-host control (<1864 trn21, <1869 trn22); (4) FALSIFIED-INERT if steps within +/-1% of control; (5) SUCCESS if steps >= +2% of control and public_val_bpb < same-host control by >= 0.001; (6) FALSIFIED-QUALITY if steps >= +2% and public_val_bpb >= control; (7) INCONCLUSIVE otherwise (including +1..2% steps, or >= +2% steps with a positive gain smaller than 0.001). If steps exceed +10% (>2278 trn21, >2283 trn22) the verdict is still assigned by (5)-(7) on the fixed-budget public_val_bpb; the Analyst additionally records the muon_owner_check lines and the smoothed training-loss log value at the control's final step count as attribution notes only (not validation BPB; they do not alter the verdict).
- Flags: ``
- Commit: `7adfe78f8324cb1f33588e4d8d63d978cd2d36b5`
- Box: trn22 (`ip-172-31-32-253`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-muon-owner-ns-split-r4-smoke-r1`; control label: `al-20260922-control-trn22-lnc2x4-9a4caa7b-smoke-r2`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- smoke steps 325 < proposal minimum 340

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.173345 | 1.165604 |
| num_steps | 325.0 | 335.0 |
| steps/s | 1.14315863524446 | 1.1775043936731107 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

DISCARD at the smoke tier: the candidate completed 325 steps, below the proposal minimum of 340 and the control’s 335.
Candidate throughput was 1.1432 steps/s versus 1.1775 for control, about 2.9% slower and below the required 1.1952 steps/s floor.
Validation BPB was worse: 1.173345 versus 1.165604, a delta of +0.007741.
The BPB comparison is confounded by unequal training progress: the candidate processed 42.6M tokens versus 43.9M for control.
Both runs used 592.9M parameters, seed 42, the same host and listed runtime settings, and nearly identical training durations.
The candidate also had a larger estimated startup allowance (174.9s versus 119.7s), complicating wall-time comparisons but not explaining the measured training-throughput deficit.
Both runs completed successfully and passed causality checks; these do not override the smoke-step failure.
