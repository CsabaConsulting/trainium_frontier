# K2 namespace retry: compiler failure, route parked

Actual one-shot native child started 2026-09-25 23:06:48.856645 UTC and exited 1 at23:07:14.675852 UTC after25.8187616559444s. Exact source d0f0d7fb/consumer4939265b, unchanged checker89189024/reference14572563 and original four-rank guard.

The module-global alias repair passed the earlier name-resolution point, but the first BF16 cross_tile_large consumer compile failed with:

> [INTERNAL_ERROR] [NCC_IXGM002] Expected function sg0000 in subgraph 0 to have 9 basic blocks, but on core 1 it has 1 basic blocks

Rank0 wrote REFUSE_OR_UNSUPPORTED and wrapper exit1. Torchrun terminated ranks1–3 with SIGTERM; their missing terminal checker receipts are preserved as missing, not filled or treated as success. All four pre-import environment receipts exist. Only upstream_roundtrip completed; no consumer exactness, gradient, component timing or whole-step result exists.

All raw JSON/logs available in this packet and the6461-byte compiler log were retrieved (22files/70941bytes). The configured bounded debug inventory contains one1168-byte module.mlir and that log; module contents were not needed to identify the compiler error and were not copied. The private caches and failed artifacts remain untouched.

Explicit postflight23:08:34.750819 UTC verifies all nine observed dispatch/flock/driver/timeout/torchrun/workers absent, no PPID/PGID/SID descendants or compiler commands, idle Neuron device and free original lock. No manual kill, cleanup or breaker reset occurred. This was the single authorized diagnosed retry; the tiny K2 route is now PARKED. No further retry/full/eval/adoption is authorized.

