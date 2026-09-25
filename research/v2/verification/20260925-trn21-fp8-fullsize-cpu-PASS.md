# Exact FP8 full-parameter CPU evidence — independent PASS

Reviewer: root, distinct from source Implementer and trn21 Operator.
Raw Operator archive03e4a616 (canonical152f211e), exact source921271d7.

Independently parsed all three G2 reports and twelve raw rank files. ON, OFF
and fresh matchedc5c each have4ranks,3finite optimizer steps,117,453,580
parameter elements,62updated tensors, matching rank ownership plans and no
cross-rank mismatch. Checkpoint exists and deterministic reload passes. Every
raw rank file exactly equals its corresponding aggregated report entry.

Default-OFF andc5c have identical12losses, all62final parameter digests and
ownership plans. ON uses the same science argv plus `--fp8-proj`; OFF uses
`--no-fp8-proj`. Depth6,width1024,pack32,context1024,LR0.015,seed42,100-step
native timing cap are pinned. G2 intentionally overrides to3CPU steps with
synthetic1x16training microbatches and full model/optimizer sizes. Its fake
clock, NKI-CE bypass and CPU reference arithmetic are NOT native evidence.

Recomputed G2 file hashes:

- OFF: aa3e7ac9d2c591bd39c1e3a1fcca7e9eb227f6bbcf0e85d5737e34079266047a
- ON: a822c4058539eb43a88a89099e846ed2791f4a604a98bc901f59c5a825799c44
- c5c:571c1e8bc2403e1000c2c3f5c93bfe230b1a865bee530485d2880847bd339044

This resolves the exact-size CPU/default-OFF gate of reviewed timing packet
30177711, whose independent packet PASS isb880cc4a. Source/timer reviews are
unchanged. No native kernel, speedup or BPB result is claimed.

## Supervisor native assignment

After fresh normal preflight, the distinct trn21 Operator may execute the
already prepared immutable NATIVE_BINDING-v2 (a46e562c): one component check,
OFF100, ON100, then ON checkpoint frozen-forward check. All phases remain
serial under the common device lock with unique existing labels/exact9212
bytes. A failed component gate stops the sequence; no timing or quality result
may be inferred. Keep the six fallback entries and prompt serial hand-back.

Use the reviewed20warmup+80measured whole-step window. At least3% accepted
step-time improvement is still required before a full quality run. These
100-step diagnostics are not fixed-budget BPB candidates. Actual startup,
process-through-exit and budget evidence, retained logs and optional exit-only
metrics are required as already specified; no extra conditioning runs or new
human approval ceremony is introduced.
