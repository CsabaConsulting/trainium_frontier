# Persistent-cache diagnosis: cause not yet established

Researcher, 2026-09-25 16:46 UTC. This is an evidence report and bounded read-only
probe proposal, not launch authority. The latest
`workflow/POST_A1_PRIORITY_CORRECTION_2026-09-25.md` controls: tables and the earlier
gather-boundary proposal in this directory remain parked. Do not interrupt either
live job, alter the SDK, set a new compiler/cache flag, or clear a cache.

## Conclusion

The configured NEFF/HLO roots are correct. There is **no established root cause
for the repeated large startup cost**. The available evidence does not prove
that relocated train.py paths cause misses, that persistent NEFF caching is
disabled, or that NKI dump regeneration accounts for the whole delay.

The cheapest decisive next check is one already-produced, matched large-graph
artifact/key pair from A2-r1 and A2-r2 on trn22. Compare actual key inputs and the
corresponding NEFF lookup result, without rerunning training. A0/A1/A2 or a
changed optimizer schedule are poorer controls because genuinely new graphs are
expected. A2-r2 is human-aborted, not a quality result; do not resume it for this
diagnostic.

Follow-up: the bounded read-only recovery was attempted; the exact matched pair
is unavailable in the inspected run/cache directories. See
`CACHE_ARTIFACT_RECOVERY_2026-09-25.md` for actual file inventory, input/source
hashes, the installed temporary-directory cleanup lifecycle, and a smaller
next-job collection plan. Random compiler work-directory names are not cache
keys. The cause of repeated startup cost remains unproven.

## What the evidence establishes

Primary archive:
`research/v2/operator/20260925-trn21-live-cache-diagnosis/`.
Additional installed-source excerpts and hashes are in
`CACHE_SOURCE_ADDENDUM_2026-09-25.txt` beside this report.

| Observation | Supported interpretation / limit |
| --- | --- |
| Four live trn21 workers inherited the intended NEFF, HLO, local-lock, debug and generic compiler roots. train.py uses `setdefault` for `NEURON_COMPILE_CACHE_DIR`. | No override-to-/tmp diagnosis. The generic compiler directory being empty does not mean the actual NEFF cache is unused. |
| At 16:32:28 UTC, NEFF had 207 files / 748,755,698 bytes, one new 28,052,596-byte NEFF since this launch; HLO had 492 files / 590,071 bytes and no new files; local had 208 zero-byte lock files. | These are occupancy/delta observations, not graph-specific hits or misses. Empty locks are normal lock artifacts, not empty NEFFs. Some new compilation is consistent with a new model/optimizer graph. |
| `compile.py:20–21,111–155` strips only the top-level `torch.debug_dump_path` attribute, then serializes MLIR bytecode. | The actual default base key is SHA256(bytecode) + `:` + `global_version_hash()`. It is **not** the location-free textual hash in `utils/stablehlo_utils.py::compute_cache_key`. Earlier saved debug artifacts can precede stripping and are not automatically exact key inputs. |
| `utils.py:27–70` hashes SDK `python_ops` and `neuron_dynamo_backend` Python relative paths and contents, plus package versions. | A caller train.py filename/run directory is not directly in this version component. SDK changes can legitimately invalidate the cache. Caller locations could still enter serialized MLIR; that separate hypothesis is unproved. |
| `CompilationCacheKey.cpp:76–105` adds platform, logical core count, optimization level, compiler arguments and compiler version before XXH3-128. | Matching source and apparent tensor dimensions alone is insufficient. Compare this complete configuration and the persistent key, not just the Python execution handle. |
| `nki_kernel.py:147–150` leaves dump-config disk persistence OFF pending robust multi-file source hashing; `201–223` builds BIR/backend configuration when that layer misses. | This upstream layer is distinct from the final NEFF cache. Its `_cached_compile_to_bir` call may itself cache work; do not equate method entry with a full NEFF compilation. Enabling this guard-disabled cache is not a safe remedy established here. |
| `hlo_cache.py:114–119` documents nondeterministic NKI backend_config bytes across ranks. | Byte instability is a credible mechanism, not proof of the same failure between these two runs. It requires the matched input comparison below. |
| `PersistentCacheBackend.cpp:79–105` reads `<neff>/<key[0:2]>/<key[2:4]>/<key>.neff`, logging open miss, empty/read error or hit. `CompilationCache.cpp:248–288` counts actual persistent hits/misses. | Correct roots and NEFF existence alone do not establish a successful lookup for the graph being discussed. A persisted hit is represented with zero compilation time in its new CacheEntry. |

Relevant A2 control evidence:

- A2-r1 `a2-startup-truncated/train.log:1,84–88` records c5c source, its command,
  1056.75 s / 903.47 s initial steps and then ~0.62 s steps.
- A2-r2 `a2-r2-launch/training-launch.json` records the same causal model flags,
  seed 42, shared trn22 train-cache roots, and launch at 16:01:54 UTC. The latest
  correction records its first completed step at 1040.11 s before human abort.
- Those step durations are not per-graph compiler timers. Similar duration does
  not prove identical compiler work, keys or inputs.

## Smallest decisive nonmutating probe

Operator/researcher read-only scope: **one pair of large first-step forward
graphs** from A2-r1/r2; add the backward pair only if the forward pair genuinely
hit. Stop after 60 seconds of host inspection or four candidate IR/metadata
files. Do not scan/hash every NEFF, import torch/Neuron, call an SDK diagnostic,
attach to a process, set logging flags, or launch a test. Do not deserialize a
cached Python object on the host merely to inspect its fields.

1. Use the two existing launch receipts and preserved source hashes to establish
   matching c5c source, causal argv, SDK/version manifest, compiler configuration,
   and NEFF root. The exact shared trn22 root from r2 is
   `/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train`.
   The r1 launch is at 15:12:50 UTC; r2 at 16:01:54 UTC. Do not silently assume
   the installed SDK now proves its state at both launches.
2. Search only the preserved A2 stdout/compiler metadata for these existing SDK
   messages: `Persistent cache hit:`, `Persistent cache miss:`, `Cache miss`,
   `Cache file empty or error`, `Failed to read cache file`,
   `Skipping persistent cache`, and `execution_cache_key=`. Retain a graph's
   full persistent key, resolved NEFF path and compiler config where present.
   Absence of debug-level messages is **unknown**, not a miss.
3. Resolve only that matched pair of saved compiler inputs. Preferred exact
   artifacts, **if already saved**, are C++ `module.mlir` with `meta_data.txt`
   and `compiler.txt`, or the `NeuronOutputCode` embedded bytecode and key.
   `backend.py:269–295` serializes the same module after CompileGraph strips the
   volatile attribute and stores `cache_key` plus `hlo_bytes`. This source
   capability does not establish that Dynamo artifacts were actually persisted.
   The ordinary earlier debug dump may still contain `torch.debug_dump_path`;
   do not hash it and label the result the actual key. If exact artifacts cannot
   be resolved within the bound, stop and report missing evidence.
4. Read the two files only. Compare exact SHA256/size, base/version/persistent
   keys, and corresponding NEFF size/mtime/open-error evidence. Compare decoded
   MLIR structure and backend_config on retrieved local artifacts using an
   already-available non-executing reader; no imports or object unpickling on a
   busy host. A normalized text comparison is explanatory only, never a
   replacement for the actual raw key. If backend_config differs, inspect its
   meaning before declaring it harmless; constants/kernel code may differ too.

| Result | Conclusion / falsifier |
| --- | --- |
| Shapes, operations, constants, training branch, aliasing, compiler config or SDK version genuinely differ | Legitimate new graph/key. Falsifies the claim of an identical-input persistent miss for this pair. |
| Exact post-strip bytes differ while complete semantic IR and configuration match; differences are verified volatile metadata/backend_config only | Establishes unstable key material for this pair. Identify the exact changing field; merely different hashes or a path substring is insufficient. |
| Exact bytes and version/config match, but persistent keys differ | Key reconstruction or identity is wrong, or a key-construction defect exists; inspect exact C++ inputs before attributing it to filesystem paths. |
| Same persistent key, valid earlier NEFF at the worker-consumed path, plus an explicit open/read/bypass failure | Establishes cache access/bypass failure for this graph. Fix only the demonstrated path/permission/config defect. |
| Explicit persistent hit (or recorded persistent-loaded zero-time entry) for the large graph, yet startup remains large | That graph did not NEFF-miss. Inspect already-recorded upstream NKI/lowering timing and other graphs; regenerate-vs-hit is now separated. |
| No key-specific messages and no exact saved input pair | Retrospective diagnosis is blocked by observability, not proof that any hypothesis is true. No new training solely to fill this gap. |

If the last row applies, the minimal *future* implementation proposal is one
default-OFF train.py diagnostic that reports already-existing per-process
persistent hit/miss/compile counters and graph-key/timing records at the normal
first-step boundary. A separate implementer must first verify the installed
public reporting API without clearing/resetting counters. Use the next otherwise
approved job; do not inject into live workers, wrap compilation functions, or
create a parallel launcher/framework. Counter-only logging distinguishes NEFF
hits from regeneration but cannot prove raw-byte instability: that still needs
the paired exact input artifacts. This report does not authorize such a patch.

## Remedy policy

No cache-behavior change is supported by the current evidence. Preserve the
already-correct roots. Pin SDK/compiler config across comparisons if drift is
actually found. A stable launcher source path would be justified only after
proving caller-path-only key variation, and must never overwrite a live source
file. No safe train.py-only correction for arbitrary unstable SDK backend_config
has been established. Leave `NKI_DUMP_PERSISTENT_CACHE` OFF.

`TORCH_NEURONX_NEFF_KEY_FROM_HLO=1` is not a general solution: installed code labels
it dev/CI, it still hashes raw bytes, removes the SDK-source version safeguard,
and may skip persistent lookup for paths without HLO bytes. The current
observations do not justify it.
