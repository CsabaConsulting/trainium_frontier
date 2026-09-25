# Frozen clean cold-five/full binding review

Verdict: **PASS for disabled binding mechanics only; source and launch HOLD remain.**

Reviewed exact `ce73fa8a54860d309c4ced52e50a5fb7016bb8ae` (canonical `d9f372dc` byte identity checked). This is not a fresh source review, route-refusal resolution, full-size CPU result, native correctness/cold acceptance, full-run permission, or submission approval. No host access, tensor allocation, model import or candidate modification occurred.

## Independent evidence

`audit.py` completed **114 checks** both normally and with `python3 -O`; the extracted candidate assertions were deliberately compiled with `optimize=0`, matching the drivers' explicit refusal of optimized execution. The script authenticates both full packet manifests (21 entries), source/prepare/harness/guard/dispatch pins, and immutable canonical imports. It executes exact driver control flow against synthetic filesystem/process observations, exact guard parsers and direct-parent refusal, and exact rehearsal budget arithmetic. It does not rely on the Implementer's test results.

Own preparation failure retained: first local test invocation returned `SyntaxError: unmatched ')'` at audit.py line109, before any test ran. One extra parenthesis in this verifier-only fixture was removed; the next normal run and the optimized run each passed114 checks. No candidate/binding repair was needed.

The full package command is byte-for-byte the requested long116188 command. Cold-five differs only by `--num-steps 5`; MAX_TRAIN_SECONDS remains1800, seed42, four ranks/LNC2, seq1024/pack32. Exact source SHA is `3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f`; the proposal113,256,724-parameter/65-tensor scope is preserved. Command agreement does not independently prove the effective source configuration: the fresh source REFUSE remains unchanged.

Both detached routes retain the exact common `/home/ubuntu/.cache/trainium_freshstart/device0.lock`, authenticated outer parent/open FD, fresh ps and neuron-ls under that lock, inherited FD in the immediate timeout child, and no release/reacquire between synchronous phases. Full checks again before fresh eval and requires the same FD. Refusals exercised: wrong host, optimized interpreter, STOP, low disk, changed source/launch/provenance/prepare/harness/guard/dispatch, existing sandbox, busy train/eval, train failure, missing/false budget, missing checkpoint, changed FD and eval failure. Unrelated host_driver, torchrun, malformed neuron schema and occupied device are rejected; direct invocation without the exact flock parent is rejected.

The unchanged harness has a previously documented limitation: its deeper torchrun/eval child does not inherit this FD. Normal synchronous driver/timeout/harness lifetime is protected, and immediate-driver death is covered by the inherited timeout FD; arbitrary death of all lock-holding ancestors must not be described as end-to-end descendant lock survival. Fresh ps AND neuron-ls quiescence is still mandatory before any reuse, including after failures. No broader lock/runner redesign is requested by this review.

Training/eval create separate absent phase sandboxes and six absent empty SDK roots. Compiler URL is prebound to each matching compiler DIR; data cache is unchanged. Actual effective cache routing still needs retrieved evidence. Clean cold-five must supply its own uncapped fifth<=900s; target870s is not the hard cutoff and prior research844s does not transfer. Cold terminal `CT_HARNESS_EXITED` is only successful harness execution, not threshold acceptance; no automatic eval/full follows.

Full uses the fixed outer2730s watchdog (+60s kill grace), NOT a2730s legal training allowance. The exact pinned rehearsal checks `process_wall - min(observed_fifth,900) <= 1800` through training process exit. Synthetic boundary cases prove excess over900 is charged and1800.1 refuses. Fifth observation includes nominal50ms polling and scheduling delay; preflight/cache setup before torchrun is outside this observer clock and must remain separately labelled. Full refuses eval on failed/missing budget evidence or missing checkpoint. Fresh unchanged prepare eval is20,971,520 tokens, context1024, batch1, Neuron, internal1200s timeout under an outer1800s watchdog, with separate fresh caches. No quality result is claimed.

## Exact support pins

- Cold binding `f8a677b645f579f5fca269e32d61a8bae8614aa2eb4cc201b4ac0c5ed3628ab9`; driver `13696a59827c59d6a34edff0960dfbe5777c4c32f863f9834671793da9d92dbc`.
- Full binding `cac9f4aef20d63c1cc081ddeae1f7271c0d36f18b244f4d09b92899bbef6558d`; driver `0b0f34cba9757f9371ab9c0b6e2b33adc6294023f5ea7f61464eaf96f2b9a511`.
- Prepare `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`; rehearsal `7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b`.
- Shared guard `adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737`; dispatch `c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a`.

Both proposals are explicitly unassigned/unstaged/disabled, with all gate booleans false. These are control-plane gates, not an assertion that the executable would authenticate a future Supervisor approval document: they must remain unstaged until the sole Operator receives an exact scoped assignment after the required evidence. The root/source REFUSE, missing full-size CPU and separate checkpoint-behaviour diagnostic are not waived by this mechanical PASS. Full additionally requires accepted clean-source cold-five evidence. No automatic next job, evaluator edit, wrapper restoration, queue mutation, artifact deletion or upload is authorized.

Reproduce from this worktree:

```text
python3 research/v2/verification/20260926-clean-cold-full-binding-independent/audit.py
python3 -O research/v2/verification/20260926-clean-cold-full-binding-independent/audit.py
```
