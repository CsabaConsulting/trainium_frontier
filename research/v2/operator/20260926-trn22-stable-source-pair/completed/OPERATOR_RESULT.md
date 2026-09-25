# Stable-source pair: S1 cold pass, comparison refused

Exact source3e79, frozen pair plan518c039d. S1 launched22:47:39.841696 UTC and completed five steps with training exit0. The uncapped fifth boundary was 845.5936225099722s, process wall 849.8144521720242s and charged-through-exit 4.220829662051983s. Retained checkpoint SHA a4e11a3d781dd85667e69abbc5ceba9a23cbf57eeb5762a38dff5c3984ba0a07, 453048594 bytes. No checkpoint was downloaded or deleted.

The reviewed driver then exited at23:01:52.975813 UTC with `MATCHED_PRINCIPAL_COMPILER_BYTES_MISSING`. S2 was not launched; no manual bypass, retry, evaluation or full follow-on occurred. This is an incomplete compile-reuse experiment, not a model-quality failure or a reuse success.

Raw archive contains 26 files/90280 bytes, including all four completed graph_stats CSVs, source/provenance guards, five step observations, training log and exact terminal/outer-exit errors. Explicit postflight at23:03:45 UTC proved all26 observed dispatch/worker/compiler PIDs absent, no related PPID/PGID/SID rows, idle Neuron device and free original lock. The lower-priority read-only training-shard hash check then ran; it is recorded separately.

## Why compiler bytes were unavailable

The post-exit helper required these principal input hashes:

- aa53f72d8a4c45dbc0ba1784f3536f8803525417f6de4d3f8fb82332aa2ef58e
- d81a59fd782a752806c228735a21928133bf2d21c5d59344e2e0e94fb3db8c25

No exact matching input bytes were found by its bounded `debug/neuronx-cc/*/module.mlir` inspection. The live observations preserve compiler command lines and named debug/log/NEFF output paths. They did not preserve an earlier directory inventory, a stat/hash of those module.mlir files, or principal raw MLIR bytes. The post-exit receipt also does not inventory nonmatching MLIR files. Therefore **disappearance at exit versus a path/hash mismatch is UNKNOWN from the captured evidence**. SDK cleanup is a hypothesis, not a proved cause. No broad cache search or mutation was performed to fill this gap.

The four graph tables prove compilation was recorded; they do not prove a cache hit. S1 alone supplies no stable-path repeat comparison, no path-only causality and no new BPB. The original no-S2-on-missing-evidence decision is preserved.

