# Last-layer FP32 tables with the compact K2 component

Supervisor-selected off-host composition, 26 September 2026. No native release,
proposal, command, queue, host or launcher change is part of this delivery.

Frozen candidate SHA-256:
`4b74ed48c772a5e82807d5509529704deaf1c8cd56692705efe9047ea7ae4a7b`.
Table parent: `../tables-runtime-select/train.py`, SHA-256
`414cdd76df85450659375c0928b7f0c882b963088529ca049c19be32814fec34`.
Compact component donor: `../k2-runtime-groups/train.py`, SHA-256
`f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e`.

## Only permitted source change

Replace `_k2_build_kernel` in the table parent with the byte-identical function
from compact f804. The complete `SOURCE.diff` has seven added/eight removed
lines: outer row groups become runtime rather than statically unrolled and
use the already verified grouped reshape/select HBM accesses, plus one comment.
Original K2 geometry remains V8192/T32768/C1024, row partitions128, features512
per LNC owner, 64 groups. The arithmetic and within-row ordering are unchanged.

The table kernel itself remains byte-identical to 414c: V524288/T32768/C128,
4096 runtime groups, feature halves64. All table FP32 master parameters,
gradients, collision/microstep accumulation, replicated AVG and RMS state/update
remain unchanged. BF16 gathered activations, hashes, metadata, lookup callers,
flags, model, optimizer, clock, checkpoint and evaluation remain unchanged.

This file deliberately does **not** incorporate the separately adopted batch/LR
base or the running Recursive stack. Its inherited constants remain batch131072,
matrix LR.015 and ordinary warmdown. Composing or selecting a new stack would
be a separate Supervisor-scoped change, not silently folded into this patch.

## Actual local evidence and reuse

The existing reduced table CPU suite ran sequentially on the exact new file
for INT32 and INT64. Each passed the FP32 reference, six collision/order cases,
RMS/ownership routing, clipping, OFF3, ON2, common initialization, checkpoint
roundtrip and same-shape causality/batch independence. Full result and limits
are in `COMPOSITION.md`; source/test/base hashes are in both CPU receipts.

`check_composition.py` independently checks actual byte ranges and AST bodies:
everything outside the one K2 factory is byte-identical to 414c; the factory
is byte-identical to f804; all119 other top-level AST nodes match 414c.
It also verifies all inherited component artifact hashes and fresh CPU hashes.
This is Implementer binding evidence, **not another independent review**.

The already completed real backend and official full-shape simulator receipts
for each byte-identical component are reused, preserving their original source
pins. No component is relabelled as a newly compiled combined model. No full
production compile/simulation, native action or new forward gate was run here.
Existing component reviews and Supervisor compositions apply to their exact
scopes; no new independent reviewer or review cycle is introduced.

## Conditional status

This is speculative local preparation, not a measured speed or BPB win.
Current compact f804 native201 on trn22 determines continuation: genuine
cold-start-adjusted net tokens must improve at least3% versus the declared
same-host base. Startup maximum1300s remains the human's current operational
limit for that running compact test. Missing metrics or failure do not pass.

Any later table run must itself record genuine cold startup, startup charge
and net tokens; full quality remains fixed1800 official accounting and fresh
public20M/context1024. The combination has no native timing, cold whole-model
startup or BPB result. Root and the sole Operator retain all selection,
binding, original-lock/resource checks and release authority. Human-only
submission, AWS and legality rules are unchanged.
