# MLP compilation boundary and gradient rounding

Updated2026-09-18,19:15UTC. This is a diagnosis of a failed gate, not an accepted
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
