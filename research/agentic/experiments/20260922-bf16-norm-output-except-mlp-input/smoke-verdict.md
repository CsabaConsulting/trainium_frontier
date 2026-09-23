# Verdict: KEEP — 20260922-bf16-norm-output-except-mlp-input (smoke)

- Hypothesis: The two bracketing runs at 9a4caa7b decompose the --bf16-norm-output result: BF16 at ONLY the MLP-input norm gave +13% steps (2346 vs 2071) but +0.014 BPB (1.020187 / 1.019731 vs 1.005734), while BF16 at ALL norm sites gave +31% steps (2716) and 1.016279 -- i.e. adding the non-MLP sites (q/k/v norms, local-conv norm, final norm) on top of the MLP site added ~370 steps and LOWERED BPB by 0.0035. Together with the --bf16-sdpa-input KEEP (attention-side BF16, 2233 steps, 1.002028), this says the quality damage of the full policy is localized to the MLP-input boundary, whereas the attention/final-norm BF16 boundaries are quality-neutral speedups. Therefore the complement policy -- BF16 norm outputs at every norm site EXCEPT the norm whose result enters self.mlp -- should retain roughly half of the full policy's step gain (~0.10 s/step: control 0.869 s/step, mlp-only 0.767, full 0.663; predicted ~0.765 s/step => ~2350 steps) with no quality penalty, and thus lower val_bpb below the control.
- Single causal change: One numerical-format policy change confined to train.py: a new module-level toggle USE_BF16_NORM_OUTPUT_EXCEPT_MLP (default True; env NEURON_COMPETITION_R1_BF16_NORM_OUTPUT_EXCEPT_MLP='0' disables; CLI --bf16-norm-output-except-mlp/--no-bf16-norm-output-except-mlp; appended to ARCH_TOGGLE_NAMES so checkpoints persist it and load_for_eval restores it). When True it (i) makes the existing norm-output policy constant USE_BF16_NORM_OUTPUT effective (exactly the state produced by passing --bf16-norm-output, so every code path that reads that constant behaves exactly as in the already-run al-20260922-bf16-norm-output-policy-r3-full-r1 arm) and (ii) exempts exactly the norm() call(s) in Block.forward whose result is the direct input of self.mlp(...): those calls execute the control's non-cast path and return the input dtype (FP32 in the default serial path). Relative to the already-run --bf16-norm-output arm the only difference is the dtype of the MLP-input norm output; relative to the already-run mlp-input-norm-only arm it is the exact complement site set. No optimizer, schedule, kernel, data, evaluator, or shape change; with the toggle off the program is behavior-identical to 9a4caa7b defaults.
- Flags: ``
- Commit: `fff4b9ba808e18f8012537cb80b445889f2a9608`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-bf16-norm-output-except-mlp-input-smoke-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- correctness gates passed; step rate within tolerance; eligible for full run

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.151434 | 1.165855 |
| num_steps | 378.0 | 334.0 |
| steps/s | 1.329113924050633 | 1.1748153359127682 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

KEEP at smoke tier; the candidate is eligible for a full run.
Validation BPB improved from 1.165855 to 1.151434, a delta of −0.014421, exceeding the required 0.005 reduction.
The candidate completed 378 steps versus 334, clearing the 355-step minimum.
Step rate rose from 1.1748 to 1.3291 steps/s, about 13.1% faster and above the 1.2453 floor.
Both runs passed causality checks, exited successfully, and stayed within budget without evaluation timeouts.
Parameter count, seed, host, compiler flags, and distributed configuration matched; both source trees were clean.
The main confounder is unequal training exposure: the candidate processed 49.5M versus 43.8M tokens in nearly identical training time, so the BPB gain does not isolate a per-step numerical-format benefit.
This single-seed smoke comparison supports proceeding, not a confirmed full-run quality improvement.
