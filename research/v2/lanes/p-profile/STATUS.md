# P status — 2026-09-26 IST

## 02:12 IST capability and K1 attribution update

Read-only installed-tool inventorycd298eba found the removed neuron-profile
shim and installed neuron-explorer2.32.0.498-b1f3998. Its complete help supports
session-dir and console summary-json; no decoder has been run on the4.13GB
trace. Correct session/NEFF pairing and bounded, non-competing CPU execution
remain prerequisites. Presence/help is not proof of decode compatibility.

The local K1 follow-up correlates exact-source pack/gather/final-writeback
events on all four SHA-verified runtime traces. Final full writeback spans
6.8–8.2ms, with overlapping intervals, so the proposed quarter-copy omission
does not currently justify a native5%screen. See
`research/v2/analysis/20260926-k1-writeback-runtime/`; no new source or host edit.
Current active jobs are architecture and seed43; preserve both and their evals.

## Native capture and bounded analysis completed (01:45 IST update)

Exact6ef7 P ran on trn21 from19:38:22 to19:52:38UTC,22steps,exit0.
Four runtime traces were retrieved with verified hashes; rank0's4.13GB raw
device trace remains on the host, undecoded. Independent local interpretation
is archived in `research/v2/analysis/20260926-p-runtime-interpretation/`
(04c6ff92). This is not a new source review or a performance win.

Coarse runtime-observed forward graphs span137.073–141.857ms and backward
graphs272.256–281.910ms. Three already-batched Muon graphs per rank total
7.575–21.298ms. These are runtime graph intervals, not per-instruction active
times. Embedding backward and internal matmul/SDPA dtypes remain unresolved.
No MFU, K1/K2 speedup or claim of every matmul being BF16 follows.

Rank0's275.838ms data-fetch annotation overlaps backward by271.683ms; its
blocking token copy waits267.889ms before dispatch, while the correlated
device-labeled copy lasts0.194ms. Do not label this a275ms loader-I/O bottleneck.
Capture-start rank skew also makes the profiled loop unsuitable as a clean
throughput comparison. Both findings and exact event identities are retained.

Next analysis: bounded installed-tool capability and correct NEFF/NTFF pairing
before any device decode, on a non-competing CPU slot. Do not disturb the
running batch timing/full architecture jobs or copy/parse the large trace on
the RAM-limited laptop. Preserve UNKNOWN until decoded evidence is available.

The earlier pending-launch/source snapshots below are historical.

Latest handoff, 19:27 UTC: fresh independent session
`01a0da02-1e9c-7771-8c4e-22560f61836f` completed source/CPU-only PASS for
exact6ef7 and this wrapper, including its own 22-step FP32/BF16 reference.
Supervisor acceptance is in
`research/v2/verification/20260926-fresh-review-supervisor-acceptance.md`.
Native launch remains pending the reviewed lifetime-lock/in-lock-idle guard
and a scoped Operator assignment. No native profile or performance result.

Implementation snapshot below is historical; its pending source-review line
is superseded by the receipt above, not by a claimed native pass.

Implemented, pending independent review and Operator binding. No host access or
native capture by Implementer. Wrapper SHA:
`eaad6aff2e18a825eb8ddca91aed5efa72c91a385eb02cbdaee36b01c2644f7d`.

Eight stdlib/receipt tests pass normally and under `-O`. Exact6ef7/93ae/3e79 bytes
and landmarks verified from Git. A sequential tiny actual6ef7 CPU OFF/ON pair
passes: fixed-clock3-step losses, all22 parameter digests and ownership identical
with actual P hooks plus CPU profiler. Depth2/width128,4 ranks,synthetic1x8;
test capture step1 only. Full-size/native/architecture identity remain unproven.
Shared laptop CPU slot was released immediately afterward. No source,
data, evaluator, queue, ledger or submission changes. Default OFF invokes exact
source without profiler setup/hooks; ordinary quality launches do not use wrapper.

CPU/runtime capture: four ranks. Device capture: rank0 only, explicitly limited.
No invented device phase times, kernel dtypes, MFU or profiling speedup. See
`RUNBOOK.md` for exact interpretation, known failed describe attempt and caveats.

Next: independent source review, Operator chooses exact source/host and binds
the existing manual launch route, then one captured steady step after20 warmups.
No new approval service/launcher and no automatic second architecture run.
