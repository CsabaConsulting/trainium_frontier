# K2 rescue status

26 September17:42IST: matched batch262144 native comparator fixed in
NATIVE_TIMING_REFERENCE.json before any candidate native result. Boundary9..200
aggregate281291.27/trn21 and281392.04/trn22 tok/s. Conservative zero-charge
baseline506.324M/506.506M; required+3% candidate521.514M/521.701M net tokens.
Existing batch runs are WARM: no cold-start claim. Zero charge gives a stronger
baseline upper bound, not a warm-start substitution in submission scoring.
Historical479.79M used batch131072 and is not the matched rescue comparator.

26 September17:36IST: extended CPU suite also PASS. Actual compiled-AOT vs
eager losses and every parameter gradient bit-identical across3passes. Reused
ID storage, two embeddings and two accumulation microsteps match framework
FP32 reference for both int32/int64.12 observed eager kernel calls in extended
AOT section comprise6reference +6compiled-continuation calls; previous6count
was the first version without eager reference. Source bytes unchanged949f262c.
No native execution or speed/startup qualification yet. Exact source diff and
CPU/AOT receipts are in this same folder; no extra review/test directories.

26 September17:35IST: source949f262c CPU PASS on installed torch2.13.0+cpu.
Actual reduced8-layer/width64 model, int32 AND int64: OFF3 and ON3 steps have
bit-identical loss/gradients/SGD-updated state vs8c6; ordinary eval unchanged.
Captured AOT graphs: forward5nodes, forward830nodes, backward1612nodes.
None contains K2 op; actual custom backward executes eagerly6times/3passes.
Kernel, metadata and registration ASTs match existing locally compiled/simulated
f804 code. No production-shape/native or speed PASS follows this CPU result.
Model import uses existing CPU prepare/pyarrow test shims, no data/eval access.

Next: eligible source review and native timing only. New-code
independent review remains pending; no new reviewer agent authorized/spawned.
Original300-step momentum on confirmed batch/LR base; do not include unscored150.
Native reference must match batch262144, not automatically old479.79M reference.

26 September17:32IST: single main implementer started after both warmdowns
were safely launched and the newly requested momentum150 packets prepared.
Source/local tests in progress; no native slot, source review or speed PASS.
Old variants and native failure remain preserved; no retry of consumed labels.
