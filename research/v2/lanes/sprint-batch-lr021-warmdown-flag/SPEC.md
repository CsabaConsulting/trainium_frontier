# Expose the existing warmdown ratio on exact adopted8c6

Implementer delivery for the Researcher's
`research/v2/proposals/20260926-batch-lr021-warmdown-flag/IMPLEMENTATION_SPEC.md`.
This is one minimal shared flag patch for prospective .60/.85 full arms, not
a selected run, native result, independent acceptance or BPB improvement.

Frozen candidate SHA256:
`4ec8b1b46562612cdcbfe3d7f66b0c8d25666bdbac3516b30bd9149c2dc08d2d`.
Parent exact8c6:
`8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee`,
`research/v2/proposals/20260926-sprint-p0-clean-batch/train.py`.

## Full delta

Only main() changes: add WARMDOWN_RATIO to the global declaration; add
`--warmdown-ratio` with float type and existing .75 default; reject nonfinite
values and values outside0<r<=1 with argparse; assign the parsed scalar.
No environment-variable override or new dependency. Original batch/LR errors
remain intact. `SOURCE.diff` is the entire five-added/one-removed-line patch.

The module default and `lr_multiplier` function remain byte-identical, as do
all107 non-main top-level nodes. Removing the four permitted main AST changes
reproduces the parent main AST exactly. Model, initialization, optimizer,
momentum, beta2, weight decay, flat-LR groups, warmup handling, loader, clock,
startup allowance/predictor, reserve, checkpoint and evaluator are unchanged.
No Recursive, K2 or tables. No compile/cache speed claim.

The parent source's bare defaults remain batch131072/LR.015. The adopted
research configuration is selected by the existing batch262144/LR.021 flags;
the prospective warmdown arms keep those flags fixed and change only ratio.
The ratio controls the existing LR-decayed groups, including AdamW and Muon;
it does not control Muon's separate .045 momentum cooldown or weight decay.

## Executed bounded author evidence

One 17.7689-second CPU process, exit0, one thread, 4GiB address-space cap and
180s outer timeout. Peak RSS394752KiB. Python3.14.4/Torch2.13.0+cpu.
The existing hash-pinned stack fixture's seven helper functions were reused
unchanged by AST extraction; none of its unrelated Recursive tests executed.

- Actual parser prefix accepts omission/.75/.60/.85/1 and all normal launch
  flags with batch262144/LR.021/seed42/context1024/pack32. Rejects0, negative,
  >1, NaN and both infinities. Existing invalid batch/LR rejection preserved.
- 360 scalar cases over .60/.75/.85/1, warmup0/300, boundary-adjacent elapsed
  values, progress0/1 and endpoint saturation match the parent hex-exactly
  when given the same existing scalar. An independent exact-rational scalar
  oracle also agrees within four ULP; largest absolute error1.11e-16.
- 2880 actual group-routing cases match the parent. Non-LR fields are unchanged
  across ratios; `no_lr_decay` groups retain their original flat schedule.
- Three-step parent identity passes for omitted and explicit.75 at both
  inherited batch131072/LR.015/reducedaccum1 and adopted batch262144/LR.021/
  reducedaccum2. Losses, input order, clipped gradients, parameters, optimizer
  state, RNG, configuration and group membership match bitwise.

Fixture: actual eight-layer GPT and MuonAdamW, width128/vocab256/context32,
1,475,892 parameters, BF16 compute/FP32 parameters, synthetic inputs only.
No113M CPU run, dataset/evaluator execution, native compiler/simulator, host,
distributed ownership, real262144-token materialization or checkpoint I/O.
The changed ratios have scalar/parser/routing evidence, not full-run BPB.

## Documentation-only spec correction after the test

The raw receipt and unchanged author script pin the specification that was
actually read:61ea9e1df60b47f0c8097ad4f02dcb16c05ecc7dbb6aa60185e9fbf89330e8d2.
Its exact bytes are archived as `IMPLEMENTATION_SPEC.tested.md`, exported
from Researcher commitd08ce1130. The Researcher then corrected only the phrase
`source-size` to `model-size`, commitcf871b00a, current spec SHA
`cf1e9240920bca5041aa6d0370190edef0bf05a41e34eea2d73dff65a4b67e95`.
Source, arithmetic, requirements and tests did not change. Supervisor expressly
directed no redundant CPU rerun; no claim is made that the old receipt ran
against the newer document. MANIFEST.json preserves both identities.

## Handoff

One fresh scoped independent review remains root-owned for this shared patch,
not one review per ratio. The first selected full run is the native check; no
added smoke, NKI test or framework-forward gate. Root binds exact source/commit,
flags, actual comparator and normal operational checks; the sole Operator
launches. No proposal, queue, ledger, state or launch authority was edited.
