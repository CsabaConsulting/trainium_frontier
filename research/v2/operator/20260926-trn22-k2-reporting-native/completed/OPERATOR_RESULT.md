# K2 native diagnostic — compiler/API refusal

The exact four-rank diagnostic failed exit1 at22:45:12.709986UTC. All four checker receipts report REFUSE_OR_UNSUPPORTED on the first `cross_tile_large` BF16 case. The real compiler diagnostic is:

`error: failed to compile NKI kernel: ... error: failed to resolve name 'nl.bfloat16'`

It arose at unchanged checker line133 calling the NKI consumer. All four upstream transfer roundtrips were exact, but no consumer comparison or timing completed. STARTED cases and absent timings are not passes. No numerical mismatch, whole-step gain or BPB evidence was obtained. No SDK/source change, retry or timeout extension occurred.

`results-bundle.json` preserves27 raw remote files (including four refusal receipts, four wrapper exits, environment/cache/guard/launch/child/outer exits and complete error log). No metrics CSV existed at this frontend failure. Explicit22:47:18UTC postflight checked all nine observed dispatch/flock/driver/timeout/torchrun/worker PIDs, PPID/PGID/SID descendants and compiler commands: all absent, device idle, original common lock free. No kill or cleanup was necessary.

This failed packet remains consumed. The separate stable-source-pair release can only proceed under fresh unchanged STOP/breaker and idle checks; this record grants no new authority.
