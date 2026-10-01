# Independent Round13 C4 schedule-bundle verification

Recorded 2026-09-27 08:39:54 UTC. **PASS_SOURCE_STDLIB_CPU_ONLY**, with no source/scalar blocker, for exact source `308ae5ae65d6138819a6e4e8dc296dd80ab9a152bc984d496f33efc771d94a77` against c239 parent `c23935404ac44314e333ddf3608a6c671f4526263a3348eb901bb3f8d4ca05b0`.

This verifies the expressly defined three-part bundle; it is not one-scalar attribution, native qualification, quality evidence, adoption, or launch authority. The candidate does not include the separate gate fix or C1/C2/C3. The known parent VE-gate-in-Muon routing is retained; a gate-fixed rebase/combined source needs a new hash-bound review. No new legality hold is introduced for the exact reviewed diff. The normal completed proposal and Operator gates remain; no extra native experiment is required by this receipt.

## Source checks

Independently reversed six precise text replacements and recovered the parent byte-for-byte. AST comparison permits exactly five changed top-level nodes: `WARMUP_STEPS`, `MUON_MOMENTUM_COOLDOWN_FRAC`, `MUON_MOMENTUM_FINAL`, `muon_momentum`, and `main`. Within `main`, the sole changed expression is scheduled Muon weight decay. Model and optimizer classes, initializer, eval/checkpoint/data functions, charged-time computation and stop logic are unchanged. The existing cautious sign mask and once-per-step update structure are untouched.

Actual source-defined scalar optimizer policies were extracted from `GPT.setup_optimizer`: five AdamW templates and one Muon template (the latter instantiated per matrix shape in production). Initial policies are exactly parent-equal. No author test was executed or counted as independent proof.

## Independent scalar test

Script: `research/v2/research_cycles/20260927-r13-c4-reference-verification.py`, SHA256 `4a55100b1a540fe7aa671d6c43c67edeaca1e1a22cb57a67f782119ebb574f39`.

```sh
/home/poojith-devan/venv/bin/python -B research/v2/research_cycles/20260927-r13-c4-reference-verification.py
```

First run exited 0; reported body time 0.2997 seconds. Pure standard library, no Torch import. Hash-bound actual `lr_multiplier`, `muon_momentum` and actual training schedule statements were executed with independent algebraic references.

- **8,704 group cases:** six actual policy templates plus synthetic AdamW/Muon `no_lr_decay` templates; 16 step values surrounding steps 39/40 and 299/300; 17 progress values including immediate neighbors of 0.25, 0.5 and 1; budget inputs 0, 1e-9, 1 and 1800 seconds. Nonnegative step/progress is the valid domain. Charged progress is upper-clamped exactly as in the unchanged parent.
- Max absolute reference errors: LR `5.551115123125783e-17`; momentum `2.220446049250313e-16`; Muon weight decay `2.7755575615628914e-17`. The decay reference used the independent half-angle identity `0.2*cos(pi*p/2)^2`.
- Dense 1,001-point checks show Muon cosine decay remains in `[0,0.2]` and is nonincreasing on progress `[0,1]`. After step 300, scheduled momentum is in `[0.90,0.97]` and nonincreasing with charged progress.
- All peak LRs, the trailing-75% LR cooldown and final fraction 0.05 are unchanged. At step 40 and later, actual LR values are exactly parent-equal. AdamW fixed weight decay and betas are exactly parent-equal in every case; Muon beta2 stays 0.9. The `no_lr_decay` override is retained, now with the requested 40-step warmup.

## Boundary and overlap semantics

For zero-indexed steps 0–39, **all groups** use `(step+1)/40`. Step 40 resumes the original charged-time LR schedule. Thus LR warmup takes precedence if charged-time cooldown is already underway: at progress 0.6, step 39 has multiplier 1.0 and step 40 has 0.5566667. If progress reaches 1 during warmup, the warmup branch still wins; the final 0.05 multiplier applies after warmup. These are the declared existing branch semantics, not an added clock change or hidden smoothing.

Momentum uses `warm=0.85+0.12*min(step/300,1)` and blends the **current** warm value toward 0.90 over charged progress 0.25–1. The two clocks overlap; the code does not wait for step 300 before beginning the blend, nor guarantee a 0.97 plateau. If the current warm value is below 0.90, the charged-time blend increases it toward 0.90 despite the legacy “cooldown” name.

| Step | Charged progress | LR multiplier | Muon momentum | Muon WD |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.025 | 0.85 | 0.2 |
| 150 | 0.5 | 0.6833333 | 0.9066667 | 0.1 |
| 0 | 0.5 | 0.025 | 0.8666667 | 0.1 |
| 300 | 0.25 | 1.0 | 0.97 | 0.1707107 |
| 300 | 0.5 | 0.6833333 | 0.9466667 | 0.1 |
| 300 | 1 | 0.05 | 0.90 | 0 |

The optimizer constructor's nominal momentum 0.95 is unchanged, but the actual scheduling loop assigns the verified value before each update. C4 is **not** C3's AdamW-only 40%-length cooldown: C4 changes both kinds' initial LR warmup and retains both kinds' original 75%-length cooldown.

## Inputs and limits

Reviewed `SPEC.md` SHA `93af082a790e963fedf25611bf1e33533bfc17aff28a7b1d8394ee951883e164`; author `CPU_RECEIPT.json` SHA `6c71894b9aea84930ae0c57ecad14e8e542909e9d4db3a1da116831ebf52ee67`; proposal stub SHA `de1a524dd0f9e63cbd2adfafe8abf18c87d08c3988ac0d9a7c7c794d4b61b32a`. Source SHA was rechecked after testing.

No model forward/update, native kernel, distributed collective, host, dataset or evaluator was exercised. No local throughput/BPB gain is asserted. The proposal remains an implementation stub, not a completed machine run packet. Verifier writes were limited to this review and its own bounded reference script; no candidate or operational files were edited.
