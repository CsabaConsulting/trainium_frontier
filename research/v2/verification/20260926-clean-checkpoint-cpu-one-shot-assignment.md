# Conditional one-shot CPU checkpoint diagnostic

Supervisor issued 2026-09-25 21:49 UTC (26 September 03:19 IST).
Sole Operator: `/root/trn22_queue_operator`.

Run only after trn22's current architectureLR0.012 job AND attached fresh20M
evaluation naturally terminate, their receipts are retrieved, the original
driver exits, and fresh process/device checks prove quiescence. Do not interrupt
either host, overlap a job, or restart an old queue. The current job's result
need not be favourable; this diagnostic uses the previously scored and preserved
architecture checkpoint, not the LR candidate's checkpoint.

## Frozen scope

- Packet `135b19e8` (raw `d524de5cb974aa77883037cdf858fcfddba3113d`).
- Label `fs-20260926-clean-checkpoint-cpu-trn22-r1`, host trn22 only.
- Binding `research/v2/proposals/20260926-clean-checkpoint-route-diagnostic/cpu/BINDING.disabled.json`.
- Binding SHA `0e5fd285080e57d0b03de80c927d89b87d1f35d094e82baf16af7bb3a8964e3e`.
- Driver SHA `a2620e94474aeda9463baa173233ac43ac7074ce03c04ba48fdf641518faeb9d`.
- Probe SHA `d21d35c68440e6a2b62abf5338603f0c8a4891a454bb61d3df0fdc420d7682a1`.
- Validator SHA `9565ad654d75a897bc2d8c3f39872c4af54763996af827beca87b0b8e34a5c19`.
- Manifest SHA `251ef57349d3449c49f7c81980c0e974f5c42eb0d9d81534a45fc3ac88a1d10a`.
- Candidate unchanged `3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f`.
- Reference unchanged `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`.
- Retained checkpoint `068749c2559f7c68f6acc76d558caff9d3806d85273fa8562483d87bbad50d25`,453048594bytes, at the exact bound existing path.

Independent review774cb8cc (canonicalab0289b2) passes CPU diagnostic mechanics
and tiny exactness/refusal tests only. Supervisor read its complete report and
probe/driver/binding and replayed94+57 checks normally and under optimized Python.
The documentation-only prior-result citation is transposed: the real REFUSE
archive is `research/v2/verification/20260926-fresh-astra-architecture-clean/`,
canonical527db181. No runtime pin changes are needed or authorized.

## Execution permission

This assignment permits one exact packet stage and one CPU inference attempt,
not mutation of the frozen disabled binding. Verify all pins, unique absent
root, STOP/breaker state, >=64GiB available RAM and >=6.5GiB free disk before
dispatch. Use the original common device0 lifetime flock, the pinned guard and
dispatch driver, and fresh in-lock ps AND neuron-ls. A busy/unknown check refuses.
Preserve lock identity. No independent second launcher is allowed.

Use exactly the bound single-thread600-second command plus15-second termination
grace. Load both real unchanged loaders with the SAME retained trained weights.
SyntheticT1024/B1-B2 only, nine calls per source, all25 strict cases required.
No training/update, dtype adjustment, data access, forward wrapper, model edit,
checkpoint save, evaluator invocation or validation-path alteration is permitted.
The six new private cache directories and artifact caps are exactly as reviewed.

Retrieve source/argv/env/guard/launch/exit/raw cases/state hashes/checkpoint
before-after hashes and terminal/quiescence evidence, including failures.
Never report this as quality, cold-start or general source-acceptance evidence.
Any mismatch is HOLD; any timeout, crash, missing receipt or incomplete result
is UNRESOLVED. Do not extend the deadline or loosen exactness. No automatic
retry/native/fullCPU-training/cold-full job follows this assignment.

Independent result acceptance and a separate assignment are required before
native diagnostics. Original clean-source REFUSE and full-training/upload HOLD
remain unchanged. Human Claude compliance audit and human submission remain.

Undo before execution by revoking this assignment. If it fails, preserve the
actual error, prove process/device quiescence and select another qualified job.
