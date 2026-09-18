# MLP compilation boundary and gradient rounding

Updated2026-09-18,19:28UTC. This is a diagnosis of a failed gate, not an accepted
precision change or an SDK bug claim. Production code is unchanged.

## Observations

A separately compiled packed MLP failed the existing native-gradient comparison
on the second, changed microbatch:9 input-gradient elements exceeded the bound,
relative RMS0.003586663. Using native forward inside the same boundary and
forming dx directly as TxC both reproduced the identical failure. All failed
jobs completed automatic cleanup with no health failures.

Wave46 keeps calls uninstrumented, reads outputs/gradients after execution, and
compares five variants with FP64 matmuls under explicit rounding hypotheses.
The strict reference rounds dHidden to BF16 before the ReLU derivative; the
alternate keeps that intermediate unrounded, then rounds dPre to BF16. Neither
reference widens the existing native-comparison tolerance.

| Uncached input/variant | Relative RMS dx versus strict BF16 | Versus unrounded dHidden |
|---|---:|---:|
| First base/native |0.003562280|0.000113975|
| Changed/native |0.000049690|0.003586575|
| Changed/native joined custom autograd |0.000049983|0.003586591|
| Changed/NKI joined |0.000049983|0.003586591|
| Changed/NKI separate |0.003586737|0.000054447|

The separated native and NKI variants match each other's dx, and repeat their
base/changed outputs and dx exactly, including changed input after zero input.
Only the native and joined-native variants differ on the initial base versus a
later base repeat in the five-model process. The test performs no weight updates.
Repeating with parameter gradients cleared before every call preserves this
pattern, so accumulated gradients are not required.

The numerical pattern strongly supports different intermediate rounding. It
does not yet establish why the native reference changes after its first call.
Do not label this stale NKI saved data: the NKI separated repeats are exact.
Also do not assume the multi-model diagnostic itself cannot influence compiler
state. Single-variant processes and autograd/cache metadata are the next controls.

## Installed backend check

Actual native package source is under
`/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx/`.
`neuron_dynamo_backend/backend.py:250` and`:309` contain conversion/execution
fallback paths controlled by `TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION`.
A new diagnostic with this set to1 plus Dynamo recompilation logging completed
without a compiler/execution exception and reproduces the same numerical pattern.
This excludes those exception-driven fallback paths in that run. It does not
prove that every operation across an explicit graph boundary is compiled; the
next diagnostic records the actual autograd-node types and compiled-cache keys.
The installed SDK has not been modified.

## Required next decisions

1. Run native alone and the separated NKI variant alone with unchanged seed,
   weights and input sequence. Record weight identity, autograd-node types,
   compiled-cache keys and repeated-call exactness.
2. If the native transition needs another model's compilation, isolate cache/
   graph interaction. If it occurs alone, inspect first-call versus later
   execution/rounding. Do not infer a specific implementation bug from the RMS
   fit alone.
3. A separated training implementation still needs a declared numerical contract,
   actual master/cache/update validation and complete-step performance. The old
   native comparison failed and remains recorded as failed. No full-budget
   training or speed promotion follows from a diagnostic exit code0.

Raw evidence: `run-logs/fs-w45-mlp-*-integration-r1.log` and
`run-logs/fs-w46-mlp-boundary-{rounding,noaccum,strict-compile}-diagnosis-r1.analysis.json`.

## Wave47: the alternating reference becomes eager

The isolated native process and isolated separated-NKI process each completed
all12 cases with exact repeats, unchanged FP32 weights, and compiled autograd
nodes. Both fit the unrounded-dHidden numerical reference. The minimal two-model
process reproduces the original discrepancy and reveals a more direct cause:

- Native first call: `CompiledFunctionBackward`, matching unrounded dHidden.
- After the separated model is called, native's next call: `LinearBackward0`,
  `PowBackward0`, and `ReluBackward0`, with no compiled autograd node. It fits
  strict BF16 intermediate rounding.
- The separated NKI model retains two `CompiledFunctionBackward` nodes and
  exact repeated results throughout.

Thus the original comparison changed from compiled-versus-compiled to
compiled-versus-eager inside the alternating-model harness. Disabling the
backend's exception fallback does not prevent this Dynamo-level execution-path
change. Shared forward code and an explicit graph break are involved; the exact
Dynamo implementation mechanism is not yet established, so this is not a
specific SDK bug attribution. No NKI derivative change is justified by the old
rounding discrepancy alone.

A new isolated integration gate captures native tensors in one guarded process
and compares the candidate in a second. It retains the original bounds,
accumulated microbatches, FP32 masters, BF16 caches, matched reference updates,
evaluation and strict checkpoint load checks. Every call must contain a compiled
autograd node. The old failed comparisons remain recorded; this gate is pending.
Source: `scratch/mlp_isolated_integration_check.py`. Evidence labels
`fs-w47-mlp-{native-only,nki-separate-only,native-plus-separated}-diagnosis-r1`.

## Wave48: unchanged gates pass with isolated compiled references

Native capture and candidate comparison completed cleanly as separate guarded
jobs ontrn22. All48 output/dx/accumulated-master-gradient comparisons pass the
original bounds at256x256 and8192x1024. Every reference and candidate call has
compiled autograd; cache refresh, matched updates, exact native evaluation and
strict ordinary checkpoint loading pass. Forward outputs match exactly at both
shapes. See `fs-w48-mlp-isolated-gate-summary.json` for maxima. The605MB-class
reference tensor artifacts are being preserved locally with hashes.

The former failed gate was comparing different execution paths. The same NKI
implementation now qualifies for a complete-step phase; this is not a speed
result and does not establish a full-training quality win. A native-forward
control behind the same boundary is the next mechanism comparison.
