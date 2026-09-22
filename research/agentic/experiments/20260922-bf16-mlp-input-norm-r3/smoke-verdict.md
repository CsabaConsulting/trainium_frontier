# Verdict: KEEP — 20260922-bf16-mlp-input-norm-r3 (smoke)

- Hypothesis: In the pinned default configuration the transformer Block feeds an FP32 norm(x) output into the MLP, so the MLP up/down projections run as FP32 matmuls. The ledger shows this class of boundary is the dominant per-step cost: the global --bf16-norm-output policy (all norm outputs BF16) reached 2716 steps vs 2071 (+31%) but regressed public_val_bpb to 1.016279, while the single isolated --bf16-sdpa-input boundary reached 2233 vs 2076 (+7.6%) and improved BPB to 1.002028 (KEEP). BF16 inputs to hidden-layer matmuls are standard mixed-precision practice; the plausible quality culprit in the global policy is the final norm -> lm_head/softcap-CE path (and/or other non-matmul norm sites), not the MLP. Casting exactly one tensor per Block -- the MLP-input norm output -- to BF16 should recover a large share of the step-rate gain (predicted +8..18% steps) while preserving per-step learning, lowering fixed-budget public_val_bpb. This is the narrowest untested precision boundary; the previous attempt (20260922-bf16-mlp-input-norm-only) PASSED proposal review and failed only implementation review because the parallel_block code path bypassed the toggle, which this revision specifies explicitly.
- Single causal change: One numerical-format boundary change confined to train.py: a new module-level toggle USE_BF16_MLP_NORM_OUTPUT (default True; env NEURON_COMPETITION_R1_BF16_MLP_NORM_OUTPUT='0' disables; CLI --bf16-mlp-norm-output/--no-bf16-mlp-norm-output; appended to ARCH_TOGGLE_NAMES so checkpoints persist it and load_for_eval restores it) casts exactly ONE tensor per transformer Block to torch.bfloat16 (the identical cast target the existing --bf16-norm-output policy uses): the norm output that is the direct input of the MLP module. In the default serial path (parallel_block=False, mlp_sandwich=False, the path exercised by the candidate) this is the result of the norm(x) call passed to self.mlp(...). The cast is guarded by a dtype check so that under --bf16-norm-output (norm already returns BF16) behaviour is byte-for-byte unchanged. For consistency of the toggle's meaning across Block.forward variants: (i) the mlp_sandwich path casts its own MLP-input norm output the same way (post-MLP sandwich norm untouched); (ii) the parallel_block path with non-coalesced projections keeps the shared h = norm(x) untouched for attention and passes a separately cast copy of h to self.mlp only; (iii) the parallel_block path with coalesced projections (where the MLP consumes a projection computed jointly with qkv from the shared h) is declared inapplicable: no cast is applied and one explicit startup line is printed. Acknowledged downstream propagation: MLP up-projection/activation/down-projection run on BF16 activations via the already-exercised BF16-input Linear path (Linear.forward selects the weight cache matching input dtype), and the MLP output is added to the FP32 residual with the existing promotion semantics -- the same consumer path the --bf16-norm-output full run exercised end-to-end. NOT changed: norm(q_in)/norm(k_in)/norm(v_in), the norm feeding local_conv, QK norms/rope_norm, the GPT.forward initial embedding norm, the final norm before lm_head, lm_head, logits/softcap/CE, ngram-ve, optimizer, schedule, batch, data, eval, causality, launch shape or environment. num_params_M must remain 592.9 and every existing config oneliner must be identical to the control except for one added field. Launch smoke and full with the exact control command/environment and no additional flags. Full-run classification vs same-host pinned control (trn21 2071 / 1.005734; trn22 2076 / 1.005967), evaluated in order: (1) HALT-INERT if static inspection finds the MLP-input norm output is already BF16 under default flags (no patch, no run); (2) CONFOUNDED-SHORT if full-run steps < 90% of same-host control (<1864 trn21, <1869 trn22); (3) FALSIFIED-INERT if steps within +/-2% of control (the MLP-input FP32 boundary is not a measurable cost); (4) SUCCESS if steps >= +3% of control and public_val_bpb < same-host control by >= 0.002; (5) FALSIFIED-QUALITY if steps >= +3% and public_val_bpb >= control (BF16 at the MLP input costs more quality than the extra steps buy); (6) INCONCLUSIVE otherwise (+2..3% steps, or >= +3% with a gain smaller than 0.002). Verdicts use only fixed-budget public_val_bpb; the Analyst records as attribution notes only (not verdict inputs) the fraction of steps beyond the ~2195-step data wall documented in train.py and the smoothed training-loss log value at the control's final step count.
- Flags: ``
- Commit: `b2a5bf5ba7e6f3c071a4fa941f95b7723534237b`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-bf16-mlp-input-norm-r3-smoke-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- correctness gates passed; step rate within tolerance; eligible for full run

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.150459 | 1.165855 |
| num_steps | 379.0 | 334.0 |
| steps/s | 1.3321616871704745 | 1.1748153359127682 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

KEEP at smoke tier; the candidate passed the gates and is eligible for a full run.
Validation BPB improved from 1.165855 to 1.150459, a decrease of 0.015396.
Step rate rose from 1.1748 to 1.3322 steps/s, about 13.4%, exceeding the 1.2101 floor.
The candidate completed 379 steps versus 334 for control, clearing the 345-step minimum.
Both runs passed causality checks, exited successfully, and stayed within budget and evaluation timeout.
Parameter count, seed, host, compiler flags, and distributed configuration matched; both source trees were clean.
The main confounder is unequal training exposure: the candidate processed 49.7M tokens versus 43.8M in nearly identical training time, so the BPB gain does not isolate a per-step quality benefit.
These single-seed smoke runs support full-run evaluation, not a definitive quality claim.
