# P r2 outcome audit: authenticated refusal, parked

Disposition: **AUTHENTICATED_REFUSED_SCRATCH_CAP_PARKED**. This is a raw-result Analyst audit of Operator commit `72087392` (canonical `8422c6cc`), not a successful decode or source approval. **Role disclosure:** this agent also authored the r2 readiness repair; this outcome audit must not be used as independent source review of that repair. Its separate review/release is not replaced here.

Own read-only stdlib audit passes **248 checks**, identically under normal Python and `-O`. It decodes and rehashes all **22 archived files /104,923 bytes**, authenticates frozen proposal/binding/manifest and actual staged runtime pins, compares all86 input/copy records, checks failure arithmetic and inspects the complete external process census. No host call, source change, model/tensor allocation, retry, cap increase or cleanup occurred.

## Findings

- Both initial and final shared readiness guards actually passed on installed trn21: fresh ps and neuron-ls returned0/idle under the same original lock device66305/inode596823, FD3, driver3101006/outer3101005. Both AS receipts record temporary16GiB and restoration to parent soft512MiB/hard16GiB, without guard errors. The static receipt's `go_16gib_compatibility=UNPROVEN` string remains unchanged, but these two actual calls demonstrate compatibility **for this installed command and run**, not arbitrary future Go work.
- Exact version succeeded: `neuron-explorer 2.32.0.498-b1f3998 built on 2026-08-10T05:09:49Z (commit b1f3998a)`. The observed pre-decode binary SHA was `f1b609564aca32f0a1dc7768aa0389e0d3f724cbf0598a7fade5a057cbc2c008`, inode862663/126,932,944bytes. This was an observed runtime identity, not a previously known binary content pin or completed posthash.
- The86 named inputs, including all80NEFFs and4.13GB NTFF, were copied in their prescribed full relative layout; the recorded source/copy SHA and sizes agree for every file, totaling4,310,546,731bytes. Copies are separate inodes, linkcount1, regular0444 files. Original content hashes were first observed during this run; do not call them historical pre-pins. This audit authenticates **recorded identities**, not a new local rehash of the multi-GB originals, which were not downloaded.
- Decoder began00:05:37.094050UTC with the exact fixed summary-json command, CPU0/GOMAXPROCS1,16GiB AS,64MiB file ceiling,300+10s decoder wall cap,256MiB generated-scratch stop, and stricter207s residual CPU allowance. At the sampled stop, generated scratch was **271,389,612bytes**, exceeding268,435,456 by **2,954,156bytes**. `RuntimeError: GENERATED_SCRATCH_CAP` is the precise error. This is a sampled0.1s stop with a disclosed overshoot, **not successful hard-quota enforcement**; the archive does not establish which generated file consumed those bytes or successful model ingestion.
- Tracked decoder3101016/start92152355 receivedTERM (recorded PID/PPID/PGID/SID), exited−15 after1.44574183598s; final kill-target census was empty. stdout/stderr were both0bytes; no decoded summary, model mapping, device attribution, dtype, MFU, throughput or quality evidence exists.
- Driver ended00:05:38.607968UTC,8.791105207s, exit1/`DECODER_FAILED_OR_LIMITED`; outer exit1 followed00:05:38.615010UTC. Parent plus reaped-child CPU8.781054s was within300s. The **source/copy/binary posthash stage was not reached**; `readonly-postflight.json` is absent. Initial read-only copy evidence cannot be promoted into successful end-to-end original preservation verification.
- External postflight00:06:30.536803UTC contains258 unfiltered PID/PPID/PGID/SID/comm rows. All observed dispatch/flock/driver/version/decoder PIDs3101004/3101005/3101006/3101012/3101016 are absent, with no matching PID/PPID/PGID/SID rows, related compiler/explorer work, or device holders. Fresh external ps succeeded; device idle; original lock free. This is quiescence at the recorded postflight time, not a claim about later assigned work.

Original r1 failure remains preserved. P stays parked; no third attempt or relaxed cap is authorized by this audit. Other independently assigned work is unaffected.

## Exact archive pins and reproduction

- `research/v2/operator/20260926-trn21-p-offline-decoder-r2/completed/results-bundle.json`: `44d6f1abe2dee31f041f0bb529ce43d2d44eb05d76db185109cba693e2fe2038`.
- `explicit-process-postflight.json`: `293679fdfbaee597b87706a01259431bd3a0d24ca4a32d779e9af3d319d1ecb7`.
- Inherited r2 driver `616fc37c756a5be2cc04f8fb4c13e551a8e0bae4e07e2eeda00475fd1db5e1df`; helper `96aebc1e9a43843f57691e67053c75ecbe904f2cec6899688a9e3036c863efd9`; binding `3c08ce4b48d98596bd8c8594fc2f9f8fcb92e41669779cd93732de335ce856cb`.

```
python3 -I -B research/v2/verification/20260926-p-r2-outcome-audit/audit.py
python3 -I -B -O research/v2/verification/20260926-p-r2-outcome-audit/audit.py
```

The script reads frozen Git bytes; it never invokes the host driver, decoder, guard, provider or remote tools.
