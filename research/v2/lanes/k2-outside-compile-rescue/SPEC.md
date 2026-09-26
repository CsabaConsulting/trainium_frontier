# One authorized K2 rescue — outside AOT graph

User-authorized single local implementation; no agents or host release.
Deadline: usable cold Trn2 timing with net tokens>=+3% by27 September12:00IST.
Safe-stack host launches and attached evaluations take priority.

Use exact f804 runtime-group kernel, metadata and registered backward unchanged.
Compose only the already accepted8c6 batch/LR parser into that source, so OFF
matches current batch262144/LR.021/ordinary.75/momentum300 baseline. Do not
silently include the new unscored momentum150 or either warmdown arm.

Move metadata plus all used embedding lookups into one compiler-disabled
helper before the transformer body. Return the main and VE tensors, allowing
ordinary autograd to invoke their registered NKI backward outside the captured
transformer AOT backward. Do not disable the transformer/optimizer or suppress
compiler errors. Verify that exclusion with captured CPU AOT forward/backward
graphs and actual autograd invocations, not decorator assumptions alone.

PyTorch documentation describes an explicit disabled function as a graph break
with tracing resumed afterwards; Neuron-specific behavior remains unmeasured:
https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.compiler_disable.html

Local evidence: kernel/metadata/registration AST identity; no new backward
math or BF16 accumulation; flag-OFF three-step reduced-model identity, CPU
ON gradients and loader-reuse/accumulation, captured-graph exclusion. Existing
local compiler/simulator receipts cover identical kernel code only, NOT the
new whole-model integration. One local heavy process at a time.

No host launch without local correctness and applicable source review. New
code is not the human-selected one-literal arm; no self-independent review
claim. No reviewer subagent is started under the conservation instruction.
Native test remains required: cold startup, charge, tok/s, first200step loss;
net=max(0,1800-max(0,startup-900))*steady_tok_s vs actual matched baseline.
Existing479791343.282 tokens are the historical small-batch reference, not an
automatically matched batch262144 reference. Do not qualify using wrong base.
No table/full continuation or speed claim until measured net>=+3%.
