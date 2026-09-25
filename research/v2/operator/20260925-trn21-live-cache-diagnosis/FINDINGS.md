# trn21 live-cache read-only diagnosis

Observed 2026-09-25 16:32–16:33 UTC while the approved dense Recursive run was compiling. No SDK, source, environment, cache, or active process was changed.

- Queue PID 2930000; workers 2930018, 2930019, 2930020, 2930021 remained alive. Exact launch and current source are archived in `../20260925-trn21-l3-timer-dense-full-r1/`.
- All four workers inherit the same persisted Sep24 train cache roots for NEFF, local lock files and HLO. `TORCH_NEURONX_HLO_CACHE=1`; `TORCH_NEURONX_NEFF_DISABLE_CACHE` absent. Only the explicit cache/compiler whitelist was inspected, never the general environment.
- Active train.py line23 uses `os.environ.setdefault('NEURON_COMPILE_CACHE_DIR','/tmp/neuron_cache')`; it does not override the supplied value or the NEFF/HLO paths.
- Installed SDK origin is `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx`. Its `csrc/core/cache/PersistentCacheBackend.cpp:23` uses `TORCH_NEURONX_NEFF_CACHE_DIR`. Read/write implementation uses sharded `.neff` files, while `CompilationCache.cpp:51` uses `TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR` for locking. These match the live environment and actual files. Absence of `NEURON_COMPILE_CACHE_URL` is therefore not evidence that this eager backend has no persistent cache.
- At 16:32:28 UTC: NEFF207 files,748755698bytes; one newly written28052596byte NEFF since launch. HLO492files,590071bytes, no writes since launch. Local208zero-byte lock files, two new. Compiler and inductor directories empty. Debug140files,966500654bytes modified since launch, including an active neuronx-cc log. This proves compilation is still occurring despite persistent storage; it does not quantify a cache hit rate.
- Installed persistent backend emits hits/misses at DEBUG level. No explicit runtime hit/miss counts were retrieved; do not infer total misses from empty generic compiler directories, nor infer complete reuse from the old HLO files.
- A first source-tree traversal encountered permission denied within a build subtree; the successful read-only retry excluded build/. Remote `rg` was unavailable, so source discovery used grep and bounded Python reads. Neither affected training.

Next diagnostic, if needed: compare cache keys/compiler inputs of freshly generated large NEFF graphs with previous runs and inspect existing debug logs. No cache flags or SDK edits are justified from directory existence alone. Keep controls matched if changing any effective compiler setting.
