# K2 SDK cache timeout and concurrent compiler evidence

Read-only local analysis of the10:25UTC observation,26September2026.
No host call, compiler/test process, source/timeout/environment/cache/lock
change, or additional gate was performed or proposed.

**Conclusion:** three distinct new compilation pipelines were active after
three workers reported timeout on the same SDK cache lock. This is consistent
with repeated compilation after coordination timed out. Matching logical graph
statistics strengthen that inference, but identical compiler inputs and exact
duplicate work are not proved. Compactf804 directly reduces the standalone
kernel's generated code size; whether that avoids full-model graph expansion
and the1200-second wait remains the prospective native201 measurement.

## What the raw records establish

The complete7544-byte training log in
`research/v2/operator/20260926-k2-native201-handoff/trn21-boundary-1025-live.json`
reports the following SDK events, all on source662c:

| UTC | Observation |
|---|---|
| 10:01:01 | Original sampled compiler invocation starts in debug work directory `6229ea5a83fb744eaaea0363e021b4cb`. |
| 10:18:52 | Original walrus3282079 is CPU-active in dependency reduction; its sample reports25 CPU ticks over0.250328 seconds. |
| 10:21:01.247 | rank1 reports `FileLock.cpp:182`, timeout1200s, breaking SDK local-cache lock `3fb4539f12a14621c2cff4a12c9ad36f.lock`. |
| 10:21:02.098 / .905 | rank2 and rank3 report the same path and1200-second timeout. Log line order is interleaved; these timestamps order the events. |
| 10:21:02 | New sampled compiler invocation starts in directory `977eeb73b96c00a71c2ff09b9c7b5c1a`. |
| 10:24:16 | rank0's CCOM log warns that RX waited120s and is retrying. This is not a terminal error. |
| 10:25:09 | Four original train workers remain; no logged completed step0, terminal or rehearsal result. Three new walrus processes are captured in stateR. Original device lock66305:596823 is held. |
| 10:25:40 | walrus3282560 adds50 CPU ticks over0.250267 seconds, about2.0 CPU-seconds per wall second during this bounded sample. Its log reaches DMA spill/reload optimization at10:25:40. |

At10:25 the actual compiler ancestry is:

| Root invocation → compiler child → walrus | Debug work directory |
|---|---|
| 3282446 → 3282447 → 3282574 | `08ca3c550f4fbbf16870d7714efc7a22` |
| 3282463 → 3282465 → 3282560 | `977eeb73b96c00a71c2ff09b9c7b5c1a` |
| 3282466 → 3282473 → 3282588 | `7476032544c1eab05160f8ecd21248ec` |

All three root invocations have parent3280913, one of the train workers.
Therefore these are three distinct pipelines, not six compiler wrapper PIDs
mistaken for six jobs. Their parentage does not identify which request/rank
caused each compile: the archived SDK supports a shared compilation service.
No service-request mapping is present here.

Original compiler PIDs3281899/3281906/3282079 do not appear in the272-row
10:25 census. Its completion/exit status and NEFF result are not in this
capture. Do not claim four concurrent compilers at10:25 or infer that the
original compile failed. Only one of the three new pipelines has the detailed
CPU/prefix/tail sample; stateR alone is not a measured CPU rate for the others.

The sampled original and new prefixes match these statistics exactly:
7,192,275,058,688 MACs;536 nontrivial operations;1291 initial HLO instructions;
1252 pre-optimized instructions;1172 after graph optimization;23 custom calls;
226 inputs;65 outputs. This is strong evidence of the same logical graph
class. It is not a byte-identity check. Unique debug directory names are job
identifiers, not authenticated content hashes. InputMLIR bytes, graph-key-to-
work-directory bindings and completed per-graph counters are missing.

## What the warnings do not establish

The SDK's `cold-training-caches/local/locks/...` file is distinct from
`/home/ubuntu/.cache/trainium_freshstart/device0.lock`. The original device
lifetime lock remained held. The Operator did not break/reset either lock;
the cache warning is emitted by the installed SDK. The SDK calls its lock
stale, but this record does not independently establish stale-owner status.
Original compiler liveness was observed before the warning; its exact exit
time is unknown. There is no basis for manually resetting the lock or editing
the timeout/environment/cache policy.

Archived `kernels/cache_utils.py:41–43` declares a default cache-lock timeout
of1200, matching the observed warning numerically. The full installed C++
`FileLock.cpp` implementation is not in the inspected excerpts; do not infer
its complete stale-lock protocol from that Python getter. The shared-service
and local-cache excerpts are historical installed-source evidence, not a
fresh complete runtime attestation. The CCOM warning also does not prove a
compiler deadlock, loss of original device-lock ownership or terminal failure.

## Does the compact loop address the graph expansion?

Source662c statically emits64 row-group bodies. Sourcef804 changes only this
outer loop to runtime iteration and equivalent grouped HBM reshape/select.
The unchanged inner occurrence loop preserves ordered FP32 accumulation of
BF16 tangents. The existing independent review and production simulator
receipts support this address/order transformation; they do not predict speed.

Standalone production backward artifacts provide direct structural evidence:

| Artifact | Static662c | Runtimef804 |
|---|---:|---:|
| NEFF bytes | 349,692 | 61,338 |
| Sum of recorded core0 opcode counts | 8,903 | 251 |
| Sum of recorded core1 opcode counts | 8,903 | 252 |

The new NEFF is82.46% smaller. The opcode-count reductions measure the emitted
standalone code, including pseudo/unknown opcodes, not executed iterations,
device cycles or whole-model instructions. Runtime looping still performs the
required64 logical groups. These sums were calculated from the two archived
instruction-stat tables per source using a text-only awk reduction.

The whole-model static662c log is much larger: at10:10:46 sg00 has two
modules/functions,694,828 allocation records,1,026 blocks and1,895,470
instructions before `extend_shared_lifetimes`; at10:18:52 the per-core view
has373,120 allocations,513 blocks and985,265 instructions. These are
different stages/scopes and cannot be subtracted as a growth or savings
measurement. The sampled new duplicate662c compile later reports834,125
instructions and513 blocks after allocation; it is still source662c,
**not** a native result for compactf804.

Thus f804 addresses a demonstrated static expansion mechanism inside the K2
kernel and plausibly reduces the full-model compiler burden. We have not
attributed every one of the1.9M whole-model instructions to K2, measured an
f804 native graph or shown that its compilation finishes before1200s. If it
does shorten the relevant compile below the SDK wait, it may also avoid this
timeout-triggered extra work. That is a conditional mechanism, not a claim
that the patch changes or repairs the SDK lock implementation.

The separately frozen proposal1211b34a8 requests one coldf804 native201,
with unchanged t9..t44 timing, source/argv/machine protections and>=3% net
tokens after startup charging. Existing662c continues to its actual terminal
or original budget/timeout. No added smoke, gate or review is needed here.

## Exact evidence

All paths below are under
`research/v2/operator/20260926-k2-native201-handoff/`:

- `trn21-boundary-1025-live.json`:
  `4d282facbbfc8d9282e0ab83d7e6a5c67fd91ceeee38ed23394bffbcf25a947c`.
- `trn21-compiler-progress-1025.json`:
  `edcd175011bb11a9924cc94c9b16e924dbcec35031e471fb7aed391bf0f21266`.
- `trn21-live-compiler-progress.json`:
  `e3fb1c04efd92e7c07834ad23bff8555c107ef0391d86887120a9d9deeafc1fb`.
- `trn21-compiler-progress-1018.json`:
  `232a18859553790c749accf99cb152668a159980feb60587e4bb55da8b248bfe`.

Operator archive commit606224793 contains the10:25 receipts. The full
training-log bytes inside the boundary bundle hash to
`7261669adefba5e93883c8e05062ce4fc237e15aef40ad3fb1343fece268e9ed`.
Detailed compiler prefix/tail samples are explicitly partial.

Standalone artifacts:
`research/v2/analysis/20260926-k2-full-local-compile/artifacts/` and
`research/v2/lanes/k2-runtime-groups/artifacts/`, both using
`core0-instruction-stats.txt` and `core1-instruction-stats.txt`.
The f804 production receipt and review are pinned by
`research/v2/proposals/20260926-k2-runtime-groups-native201/INPUTS.sha256.json`.

Historical SDK excerpts:
`research/v2/operator/20260926-trn22-arch-initial-cache-cpu-r1/native/cache-readers/kernels__cache_utils.py.txt`
and `csrc__core__compilation__CompilationCache.cpp.txt` in that same directory.
They distinguish local cache locks, shared compilation-service selection and
the default timeout; no policy change follows from inspecting them.
