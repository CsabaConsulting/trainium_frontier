# Verdict: DISCARD — 20260922-nki-local-conv-only-default-shape (smoke)

- Hypothesis: At the pinned default shape (6 layers, 592.9M params incl. 4x1M-row BF16 n-gram tables, FP32 matmul, CC='--optlevel=1'), the compiled two-tap local convolution is a non-trivial fraction of the ~0.87 s/step budget (2071-2076 steps in 1800 s). Replacing it with the already-integrated, gate-validated fused NKI local-conv kernel (--nki-local-conv), which is designed to be numerically equivalent to the compiled conv ('corrected integration passed all four native gates', fs-w12), will raise completed steps in the fixed 1800 s budget and therefore lower public val_bpb, with no change to the model function, optimizer, schedule, precision boundaries (FP32 norm outputs and residuals retained, exactly as fs-w15 demonstrated NKI conv works with FP32 normalization), data, or evaluator. This is the isolated arm the proposal verifier explicitly requested when rejecting the bundled --nki-local-conv+--bf16-norm-output proposal ('Test --nki-local-conv alone against defaults').
- Single causal change: Pass --nki-local-conv only. Every other option stays at the 9a4caa7b defaults used by the two pinned controls: no --bf16-norm-output, no --bf16-sdpa-input, no --compile-sdpa-direct, no optimizer/clipping/schedule/shape/precision switch, no env change (NEURON_CC_FLAGS='--optlevel=1', NEURON_LOGICAL_NC_CONFIG=2, NEURON_COMPETITION_R1_SEED=42, nproc 4, --num-steps 5000 --max-train-seconds 1800 --seed 42). The only behavioral delta is which implementation computes the same two-tap local convolution (fused NKI kernel instead of the compiled PyTorch two-tap path). Inference on the control default: neither pinned control launch passes --nki-local-conv; every fresh-start run that used the kernel (fs-w9, fs-w12, fs-w14, fs-w15, fs-w16, fs-w18, fs-w50..fs-w146, commits 2026-09-18..19) passed it explicitly, and the paired --no-nki-local-conv flag exists, so the default is off; preflight MUST confirm this from the resolved-config/log line printed at startup (see abort conditions) rather than assume it.
- Flags: `--nki-local-conv`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn22 (`ip-172-31-32-253`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-nki-local-conv-only-default-shape-smoke-r1`; control label: `al-20260922-control-trn22-lnc2x4-9a4caa7b-smoke-r2`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- smoke steps 330 < proposal minimum 336

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.167854 | 1.165604 |
| num_steps | 330.0 | 335.0 |
| steps/s | 1.1587078651685392 | 1.1775043936731107 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Gate-audit correction

The original generic comparator marked this smoke `KEEP`, but the frozen proposal explicitly
prohibited promotion at 335 or fewer candidate steps. The candidate completed 330 steps versus
335 for the matched control and its smoke BPB was worse by 0.00225. The proposal's existing prose
gate is now represented as a machine-readable minimum of 336 steps. No full run was launched; two
attempts stopped at read-only inventory before staging because trn22 was below the 8 GiB disk floor.
