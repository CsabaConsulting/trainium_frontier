# One read-only dual-host snapshot

Actual observations: **2026-09-26 03:07:02 UTC**. No staging, launch, kill, source/driver/cache change, restart, cleanup or follow-on occurred.

- **trn21 LR0.012:** actively training at printed step **1870**, loss **2.9340**, **265,327 tokens/s**, source LR multiplier **0.671**, **882 seconds remaining**, epoch 1. Four original workers3152652–3152655 remain stable. No current compiler descendants. Original `66305:596823` lock held. Disk **15,723,225,088 bytes** free. First five printed losses and steps0..4 are retained; exact uncapped harness fifth measurement remains unavailable until rehearsal finalization, so the printed775.76-second first step is not substituted for it.
- **trn22 Recursive child:** still in initial compilation, no completed printed step yet. Four original workers3252248–3252251 remain stable. Actual compiler descendants3252754,3252756,3252771–3252774 were live;3252771 used99.9% CPU in this snapshot. Original `66305:832589` lock held. Disk **20,795,445,248 bytes** free. No cold or cache-hit inference from its reused cache paths.

No traceback/runtime/assertion failure was present in the captured training logs. Historical queue PIDs remain absent on both hosts. Attached fresh20M phases are preserved and have not started. Both free-disk values remain above their launch/runtime floors, below the25GB notification threshold; no deletion is authorized by this observation.

Raw receipt SHA-256:

- trn21: `270fa74014876bfbffd2296644c7ad8584dd534550863e22b2f015a9dbedc6b0`.
- trn22: `7cd03eb03a96a712c7c5bcad08b49034696d789ab8eb524aa8f489d373bfbca7`.

All selected processes include actual PID/starttime/PPID/PGID/SID, argv and cwd for later union-based explicit postflight. The new compiler IDs must not be lost at terminal retrieval.
