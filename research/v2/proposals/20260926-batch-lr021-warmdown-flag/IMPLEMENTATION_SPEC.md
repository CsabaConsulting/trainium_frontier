# Minimal warmdown flag on the adopted batch/LR base

Researcher specification only, 26 September2026. This does not implement a
source, pass a test, allocate a host or authorize a run. The Supervisor chooses
the order of the two full arms; neither is ready until the delivered exact
source and its ordinary bounded evidence are bound.

## Exact parent and existing behavior

- Parent: `research/v2/proposals/20260926-sprint-p0-clean-batch/train.py`.
- SHA-256: `8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee`.
- Parent commit: `7117d57b92e19e8437158164dbce94495389d7b6`.
- New implementation lane: `research/v2/lanes/sprint-batch-lr021-warmdown-flag/train.py`.
- Line60 is `WARMDOWN_RATIO = 0.75`; line61 is `FINAL_LR_FRAC = 0.05`.
- `lr_multiplier`, lines1310–1318, already reads WARMDOWN_RATIO. It returns
  1 before progress <1-r, then `(1-progress)/r` interpolated to FINAL_LR_FRAC,
  retaining exact arithmetic/order and existing progress clamp/denominator.
- `main`, lines1331/1357–1363, already exposes batch and matrix LR but has no
  warmdown flag. It validates batch>0 and finite positive matrix LR.
- Existing line1461 routes this multiplier to every LR-decayed optimizer
  group. `no_lr_decay` groups keep `lrm_flat`. Matrix LR also applies to the
  local-convolution AdamW group; this is not a Muon-only schedule.
- Muon momentum's separate .045 cooldown at lines1320–1328, beta2=.9 and
  weight-decay schedule at1462–1465 are NOT warmdown-ratio controls.

## Only permitted source change

Expose the existing scalar; do not rewrite the schedule:

1. Add WARMDOWN_RATIO to the main function's global declarations.
2. Add `--warmdown-ratio`, type=float, default=WARMDOWN_RATIO, beside the
   existing batch/LR arguments. No new environment-variable override.
3. After parse_args, validate finite **0 < value <= 1** with a clear parser
   error. Preserve the existing batch/LR validation/error behavior.
4. Assign `WARMDOWN_RATIO = args.warmdown_ratio` before training setup.

Keep module default0.75 and `lr_multiplier` byte-identical. No helper wrapper,
new interpolation, clamping change, operation reorder, schedule tensor, hidden
eval behavior, initialization or model-state/config change. No changes to
architecture, batch/LR handling, data, seed/RNG, forward/backward, optimizer
groups, momentum/beta2/weight decay, source clock/startup/predictor/checkpoint
reserves, distributed ownership, reduction/clipping, loader/evaluator or
checkpoint semantics. No Recursive, K2, tables or model-size changes.

The default or explicit0.75 must produce the same globals and arithmetic as
the parent; the two proposed arms change only this existing scalar to0.60
or0.85. Model tensor shapes and graph topology are unchanged by design, but
actual compiler/cache reuse remains to be observed rather than guaranteed.

## Bounded evidence, once for the shared patch

- Freeze exact delivered source/commit/diff hashes; demonstrate the diff is
  restricted to those main-parser/global/validation/assignment changes.
- Parser checks: omitted flag and explicit0.75 preserve default;0.60,0.85
  and1 are accepted;0, negative values, >1, NaN and both infinities rejected.
  Existing batch262144/matrixLR0.021 plus all normal launch flags still parse.
- Existing scalar schedule reference must agree at default and at each arm,
  including just below/at/above1-r, progress0/1 and existing warmup handling.
  Do not change the function to satisfy the test or invent a different clamp.
- Reuse the available bounded default/OFF identity harness once at its actual
  documented scope. Do not turn it into a113M CPU run. Report which evidence
  was executed, versus inherited source identity; parser-only evidence is not
  a claim of full-model/native training equivalence.
- One scoped independent review for this one shared minimal patch; no review
  per scalar value. The first full run itself is the native check; no added
  smoke/forward-compile/second-review cycle. Researcher runs no such tests.

## Two prospective full arms, not an automatic stack

Both retain the adopted ordinary eight-layer source configuration, total
batch262144, matrixLR0.021, seed42, full-chip LNC2/four ranks, pack32/context1024,
fixed1800 official charged seconds and fresh public20M/context1024/batch1.

- **0.60:** keep full LR for40% instead of25% of the charged budget, then
  shorter/steeper cooldown. Hypothesis: more useful optimization before late
  cooling on the larger-batch run; risks larger late updates/overfitting.
- **0.85:** start cooldown after15% instead of25%, and cool more gradually.
  Hypothesis: lower late-run update noise; risks under-training. This does not
  implement the Recursive package or change Muon momentum/beta2.

Predicted BPB gains and probabilities are **unknown**, not fitted promises.
These are separate one-factor comparisons to0.75, never a joint .60/.85 run
or a combination with the simultaneous LR0.02625 experiment.

Actual scientific comparators remain trn21 batch/LR.021 public0.9869854734569898
and trn22 public0.9867730678453942. Their original batch_arm/basee1e2 rows are
immutable. Future arms use base8c6 and kind=schedule_arm. Unchanged selector
may report INSUFFICIENT without an actual prospectively declared
kind=control/base8c6; do not relabel old rows, create duplicate records or
silently allocate extra control runs. Raw same-host deltas remain reportable.

Supervisor selects host/order only after the actual source/evidence is bound.
Source SHA/commit and native PASS stay unknown in both proposals now. Existing
original-lock/sole-Operator/preflight/STOP/resources, startup-over900 charging,
checkpoint/fresh20M and human-only upload protections remain. Warm research
cache use, if selected, must be recorded and never called genuine cold proof.
