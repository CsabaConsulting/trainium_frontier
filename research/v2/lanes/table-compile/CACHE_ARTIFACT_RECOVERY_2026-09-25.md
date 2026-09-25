# Actual A2 archived-input recovery: matched pair unavailable

Read-only follow-up on trn22, 2026-09-25 approximately 16:49–16:53 UTC. No SDK
import, cache flag, host write, training run, process attachment or signal was
performed. Pinned transport was the repository's existing trn22 SSH destination
`ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com`.

## Observed result

The proposed retrospective comparison cannot be completed from the inspected
artifacts: **A2-r1's exact compiler input and matching key record are missing**,
and the only surviving A2-r2 input is an interrupted later graph, not a retained
first-step forward input. This is a concrete observability limitation, not a
diagnosis that cache keys differ.

Both actual sandbox sources were rehashed read-only and match:
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.

Inspected run directories (file inventory to depth four):

- `/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260925-v2-c5c-a2-m64d128-trn22-r1-full-work/train`
  contained train.py, prepare.py, prepare bytecode, rehearsal.json, steps.jsonl
  and out/final.pt; no saved compiler input/key artifact.
- `/home/ubuntu/trainium_finalpush_queue_20260925_trn22_wait_r1/runs/fs-20260925-v2-c5c-a2-m64d128-trn22-r2-warm-full`
  contained its train.py, proposal, launch/process/preflight/attempt/exit/terminal
  records, training.log and human-stop receipts; no saved compiler input/key
  artifact.
- The shared `.../fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train/inductor`
  yielded no files to depth four.
- A targeted search of r1 source-package train.log and r2 training.log found no
  `Persistent cache`, `Cache miss`, `Cache file empty`, `Failed to read cache`,
  `Skipping persistent` or `execution_cache_key` messages. This means unknown
  hit/miss identity, not zero hits.
- `/tmp/tn_op_dumps` does not exist. No custom dump root was asserted absent.

The shared debug root has only two immediate `neuronx-cc` work directories.
Its complete module.mlir inventory to depth three was:

| Timestamp UTC | Bytes | Relative path under shared train cache |
| --- | ---: | --- |
| 2026-09-24 21:27:20.473998490 | 16,992 | `debug/neuronx-cc/e663fb9eb13303d684bf533d53d32e87/module.mlir` |
| 2026-09-25 16:26:04.231844028 | 24,208 | `debug/neuronx-cc/67a18319e025bd80fe75224a7fb7a342/module.mlir` |

The latter is MLIR bytecode, SHA256
`3033d4097dd626a41405d698876f679fb74e641b689e02594e64c1730a54c871`.
Its directory contains a 2,388,727-byte log-neuron-cc.txt and generated attention
backward / conv backward kernel artifacts timestamped 16:26–16:29, consistent
with A2-r2's interrupted backward compilation. At directory depth one, only
module.mlir and log-neuron-cc.txt exist: no metadata.txt/compiler.txt key/config
record. The September-24 input is not a matched A2-r1 control. No semantic
comparison between these unrelated files was used to infer a cache failure.

## Installed-source explanation of missing temporary inputs

Read-only source observations on trn22:

- `csrc/core/compilation/NeuronCompiler.cpp:50–71` obtains a compilation workdir,
  calls `ShouldPreserveCompilationArtifacts`, constructs a TempDirectory and
  writes the supplied HLO bytes to module.mlir. Thus a live file here is an exact
  compiler input, not the earlier Python pre-strip debug dump.
- Lines 113–137 write metadata.txt (including cache_key) and compiler.txt **only
  when preserve is true**. Lines 140–144 execute the compiler and read its NEFF.
- `csrc/core/cache/CacheUtils.cpp:42–44` formats two random numbers into the
  directory suffix and returns `<debug>/neuronx-cc/<random suffix>`.
  **Those 32-hex directory names are not NEFF keys.**
- `CacheUtils.cpp:47–55` returns false when the preservation environment variable
  is absent; its header identifies that variable as
  `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS`.
- `csrc/core/utils/TempDirectory.h:88–97` removes the directory tree during
  cleanup when `preserve_` is false. This default lifecycle explains why a
  successful compile can leave no debug input while an interrupted compile does.
  It is source-level explanation consistent with the observed remnants, not an
  independently reconstructed history of every removed directory.

Observed installed-source SHA256:

```
2ee0b89ae000f3f1b070190e09d8a3e7048670a2e6b3444abf6554ce7ddab556  csrc/core/compilation/NeuronCompiler.cpp
a5e25ce81413a88503e9d8504900af77f2d1ed8031f4a5eda9bbcec3964d0690  csrc/core/utils/TempDirectory.h
```

## Minimum next observational collection

Do not change flags or rerun aborted A2. During two otherwise-needed,
independently approved same-source/config launches, have the existing operator
observer retrieve **one** already-created live module.mlir and its existing
compiler log before successful compilation removes the temporary directory.
Record source hash, launch label, graph role, UTC, compiler command and exact
input SHA256. Stream the small input to the controller; do not write a copy on
the host, preserve entire compiler directories, or read multi-megabyte NKI dumps.
If no compile directory appears, that is not automatically a hit: pair it with
the existing compile/cache counters or key-specific logs if available.

This uses the existing observation/retrieval path and changes no model or SDK
behavior. One bounded capture per job is sufficient; it is not a new monitor
framework or launch requirement. Without a second otherwise-needed matching
launch, retain the first input and leave the causal diagnosis open.

If existing records do not expose actual persistent keys/hit counters, the
default-OFF train.py reporting proposal in CACHE_DIAGNOSIS remains the minimal
separately reviewed instrumentation option. No cache-behavior remedy is justified
until that evidence identifies the failure. In particular, neither NKI dump
persistence nor raw-HLO key mode is approved by this recovery attempt.
