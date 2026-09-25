# Pure architecture cross-host replicate — preparation only

Assigned by Supervisor after the first3e79 architecture result:0.9863403052402583
on trn22, delta-0.0051736530902537 versus that host's c5c control. This packet
prepares the next trn21 slot AFTER its active seed43 training AND attached
fresh20M evaluation terminate. No remote staging or native launch performed.
Separate Supervisor binding review/release is required.

Science is identical to the successful3e79 run: seed42,1800s,4ranks/LNC2,
8x1024/U-Net/attention0,2,4,6,7/initial-cache-refresh; **no Recursive flag**.
Exact source3e79a7c1 from b4d83b5d. Compare to trn21 c5c0.9914739939934344,
not to the Recursive seed43 diagnostic or the other host's control.

The reviewed two-phase driver changes only eight host-bound constants:
ROOT, PREP, HARNESS, LABEL, TRAIN_COMMAND, EVAL_COMMAND, TRAIN_ENV, EVAL_ENV.
Control flow, source/prepare/harness checks,1800 accounting, original lifetime
flock, inherited FD, fresh ps/neuron-ls before train and eval, and failure
handling are unchanged. Local AST comparison and exact argv/source/hash
checks passed; these are Operator mechanical checks, not independent review.

Existing trn21 warm-intended train/eval directories come from the already
running seed43 binding. No reset or deletion. Such paths prove no cache hit.
The prior844.353s truly cold five-step gate was on trn22, not trn21;
this replicate will record actual uncapped fifth-boundary and process budget.
No full-cold or submission claim follows from this packet.

Pins:
- driver b4647403e8338cea1bc9d16b7077315d9f3705e29c4bb364a869ee30f398fe6c
- binding 7fea022a302ae4626cee1a463b75a9d40ac0fc6267750977ad5f7bd43021cce6
- proposal6194e2786a52699d8f7829d896f4c7e526408b85c202bbfacda425175c7e4c6e

A local orchestration helper initially raised ReferenceError: structuredClone
is not defined before any file or host action; replaced with a standard JSON
clone for mechanical packet preparation. This was not a model or host failure.

