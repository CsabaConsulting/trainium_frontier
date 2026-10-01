# R136 final decode attempt — 2026-09-30 08:28 IST

PARTIAL FORWARD ONLY; BACKWARD FAILED, PERMANENTLY PARKED. Exact source
e1bd539f and R119 standalone captures, not current LaneB or a whole training step.
Archivecb8339b7b1af1c13da152843a5c1f28e3754d61f1b18dfc27e0f4beb9e52371c;
forwardsummary0dfd55a64fdefd48b4ff5db571f719e7bc94a4ed2d52de44dc8585502e811254.

| Forward aggregate only | Value |
| --- | ---: |
| Execution | 106.717 ms |
| DMA transfers / average bytes | 19,368 / 140,747.157 |
| DMA active / estimated MBU | 87.112% / 49.070% |
| HBM read / write bytes | 19,882,090,096 / 17,617,658,814 |
| Spill reload / save bytes | 13,517,497,344 / 7,750,517,760 |
| Spill bytes / total HBM bytes (derived) | 56.715% |
| Transpose / hardware FLOPs (derived) | 9.170% |
| Named top operations / embedding-scatter share | NOT FOUND / UNKNOWN |

Backward error: `Failed to unpack NEFF ... neff482131665/neff.tgz: file too large`.
The64MiB file-size guard applies to unpacked temporary files too. Worker ended
after339.374s with no remaining descendants; no devicecalls/inputcleanup.
No further decode/cap change underR136. The embedding>=5% gate is NOT proven;
these forward aggregates cannot justify a backward-kernel decision.

## Historical, separate capture below — unchanged

| Evidence | Availability / result |
|---|---|
| Capture | Exact500f raw session preserved on trn22:20 files,1,238,671,265bytes; one rank,batch16/context1024 forward/backward,not whole-step. |
| CPU ingest | FAILED:Explorer exit2 after54.1626s; driver CPU_WORKER_EXIT_1 after57.0136s. No accepted completed decode. |
| Allocation / bounds | Go could not allocate4,194,304bytes (11,406,213,120 in use);16GiB address-space cap,8GiB Go soft limit; peakRSS8,923,640KiB. |
| Host OOM | NOT established; MemAvailable at admission129,026,469,888bytes. Exact allocator/cap interaction unproven. |
| Top named operations | UNAVAILABLE:partial Parquets are not accepted evidence; no top15 or warm-window attribution. |
| DMA / spill | UNKNOWN: no trusted traffic,spill,bandwidth,utilization or production-step estimate. |
| Cleanup | Original Guardian release_ready; all7 observed process identities absent; no device lock/API calls; private CPU lock only. |
| Disposition / archive | DROPPED per human:no retry,no cap increase,no partial-output summary. ArchiveSHA58803f4e05984434516a29919af8810109f2f0efbe983e0150dad580949cc7e0. |
