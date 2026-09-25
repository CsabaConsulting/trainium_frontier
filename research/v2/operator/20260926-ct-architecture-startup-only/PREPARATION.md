# CT: exact architecture cold startup, preparation only

No host files have been staged and no job is authorized by this packet. The
newest CT directive supersedes the earlier full1800 architecture rerun; its
unconsumed full label/binding remain parked. This unique label runs exactly
five optimizer steps with unchanged MAX_TRAIN_SECONDS1800, then exits normally
and preserves its checkpoint. There is no evaluation or automatic full run.

Use the already-proven Sep24 rehearsal harness unchanged (7a55cb87), not a
new timing mechanism. It measures monotonic launch-to-first-observation of
each append-only rank0 line emitted by unchanged prepare.record_step_boundary.
The fifth observations entry is uncapped startup. Its separate allowance field
is min(observed startup,900), and neither is the trainer's estimate. Polling is
50ms plus scheduling delay; do not invent a sub-poll precise or absolute UTC
arrival. Exact observed elapsed startup <=900 plus clean source/cache/exit
evidence is required before any separately assigned full architecture run.

The harness creates six private empty SDK cache roots before launching the
training child. The clean env sets NEURON_COMPILE_CACHE_URL to that same new
compiler directory. No warm cache is copied/deleted and the trusted data/tokenizer
cache is unchanged. Verify actual worker env and cache objects after launch.
The six-dir creation/empty assertions live in the pinned harness, so the future
sandbox must remain absent before it runs.

Static num_steps audit is not independent acceptance: every occurrence is
recorded. None feeds model/config/loss/LR/optimizer. Fifth-step stop-control
changes only after the fifth record_step_boundary, then synchronized exit and
checkpoint saving proceed. Actual native graph/compiler behavior stays a
measurement, not a promised equivalence. The outer1800s+60grace diagnostic
timeout is distinct from unchanged1800 source budget; missing fifth boundary
or excess900 startup parks the architecture, never generates a quality score.

Existing root queue remains paused and old scalar fallbacks remain parked.
All STOP/breaker/hash/idle/device/commonlock/disk guards still apply. Supervisor
must attach fresh independent acceptance and a completed scoped assignment
before this Operator may stage/launch.
