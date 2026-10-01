# R144 — installed backend, compiler and profiler inventory

Observed on trn21 on 2026-09-30; collected at 04:38:07 UTC (10:08:07 IST).
Read-only CPU inventory completed. No optimization flag was tried, model compiled,
profile captured/decoded, trainer imported, or device called. This report creates
no source, timing, launch or scored authority. CS3 remains REFUSE_PENDING_SESSION_REUSE_AUTHORITY.

## Evidence and current versions

Exact collection:
`research/v2/operator/20260930-r144-installed-inventory-cpu21-preparation/collect-20260930T043807461006Z.json`
SHA256 `36860c8ef13f4e98d346b6cea7542c07efd98bf76eb1f7aa9a3f47f5353793f9`.

All complete embedded file byte counts/SHA256 values, three CLI stdout/stderr
pairs, and retained full-source text hashes were locally authenticated. The original
worker stdout is 330,794 bytes, SHA256
`748cf9e6685cbc7a438a8ea6d2169968a0cddb99204a809ff85b08f0b905c4bf`.
Terminal `1d20c23917e11095c70ea1ddd40d4d5f1a2b9d764d37ae2aae846e8d332a8027`;
binding `42e597fb67d56ba59f468dfd153b6d410db53f6b37f178f4fd0176f535d0f989`.

| Installed item | Observed version |
|---|---|
| neuronx-cc | 2.26.6360.0+6f180f47 |
| torch-neuronx | 2.11.3.0.0+unknown.dev |
| NKI | 0.5.0+28631259367.ga768afa6 |
| neuron-explorer | 2.32.0.498-b1f3998; built 2026-08-10T05:09:49Z; commit b1f3998a |

The worker statically scanned 48 Python source files / 480,485 bytes: every
`.py` under installed `torch_neuronx/neuron_dynamo_backend`, plus
`__init__.py`, `utils.py`, `kernels/compiler_config.py` and
`kernels/compiler_subprocess.py`. All four optional support files and both
settings/config files existed. The receipt retains 56 syntactic key-access
records, 68 assignment records, 275 feature contexts and all scanned hashes.
Annotation `Optional['NKIArray']`, duplicate helper accesses, and environment
writes were not misclassified as extra user options in the lists below.

This is exhaustive for literal Python key reads within that stated scan, not a
claim to enumerate controls hidden in compiled C++, arbitrary dynamic keys or
unscanned imported packages. No process environment values were dumped.
Defaults below are source expressions, not values obtained by importing the SDK.
Four package defaults depend on optional `_build_config.py`, which was not read:
their exact editable-install fallbacks are labeled, not asserted as effective defaults.

## A. Installed torch.compile backend "neuron"

Direct backend option-dictionary accesses in current `backend.py`:

| Key / parameter | Source default | Actual observed handling |
|---|---|---|
| `options["model_name"]` | `"model_default"` | Read by `options.get`; the internal compiler's separate `options is None` route uses global `get_model_name()`, initially `model_default`. |
| `options["use_tensorizer_backend"]` | `False` | Passed to `CompileGraph(..., use_tensorizer=...)`. |
| `options["dynamic"]` | `False` via `setdefault` | Inserted when absent. `setdefault` does not overwrite an explicit True; no other read of this dictionary key was found in the scanned backend. This does not establish support for dynamic graphs. |
| wrapper parameter `mode` | `"default"` | Not an options-dictionary key. Nondefault values warn and revert to default. |

No complete validating dictionary schema is present. Passing an unused key
without rejection is not proof of a working feature. The three dictionary keys
above are the actual consumed/inserted literal keys, not an invented list of
PyTorch frontend arguments.

Hardcoded backend behavior includes pre-/post-AOT processing, separate forward
and backward compiler wrappers, selective decompositions, and
`keep_inference_input_mutations=True`. No public option key for a fusion budget,
partition size, graph-size limit, memory schedule or spill threshold was found.

### Backend-subtree environment reads

| Environment name | Default / source behavior | Location |
|---|---|---|
| `TORCH_NEURONX_RETAIN_DEVICE_MODE` | Reader default `"0"`, enabled by equality to `"1"`; package initialization can supply a build-config default below. | backend.py:221 |
| `TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION` | Boolean reader fallback False, **but package initialization sets an absent variable to `"1"`**. These are distinct layers, not an effective False default. | backend.py:251,313; package __init__.py:72 |
| `TORCH_NEURONX_NEFF_KEY_FROM_HLO` | Unset returns None; enabled only by equality to `"1"`. Otherwise cache key also uses global source-version hash. | compile.py:138 |
| `RANK` | `"0"` | cache_artifact.py:203 |
| `LOCAL_WORLD_SIZE` | `torch.neuron.device_count()` expression; not evaluated in inventory. | config.py:77 |
| `TORCH_NEURONX_ENABLE_NKI_SDPA` | Boolean True | decompositions.py:548,736 |
| `TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY` | Boolean False | executor.py:105 |
| `TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE` | Unset returns None; no explicit default argument. | fx/passes/vocab_sharding_spmd.py:348 |
| `TORCH_NEURONX_ENABLED_METRIC_TABLES` | Empty string | metrics.py:24,104 |
| `TORCH_NEURONX_METRICS_DIR` | `"/tmp"` | metrics.py:27–28,232 |

`TORCHINDUCTOR_CACHE_DIR` also appears as an environment subscript assignment
in cache_artifact.py:205; the scan is not evidence of a user-option default for
that write. The generic `settings.py` helpers accept a dynamic `key`; their
callers' concrete names are listed above. Its Boolean parser lowercases and
recognizes `1/true/yes` as True; absent values return the caller's default.
Integer parsing raises on invalid input; path/string helpers return their
caller defaults when absent, and the flags helper splits on whitespace.

### Related package initialization/support defaults

These are package/support settings, **not** extra `torch.compile(options=...)`
keys. `setdefault` preserves an existing environment value.

| Environment name | Initialization or reader default |
|---|---|
| `NEURON_LAUNCH_BLOCKING` | Build-config `DEFAULT_SYNC_MODE`; fallback `"0"`. utils reader fallback `"0"`. |
| `TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS` | Build-config default; fallback `"1"`. |
| `TORCH_NEURONX_MLIR_ATEN_OPS` | Build-config default; fallback `"1"`. utils reader alone falls back to `"0"`. |
| `TORCH_NEURONX_RETAIN_DEVICE_MODE` | Build-config default; fallback `"0"`. |
| `TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION` | Package `setdefault("1")`; backend Boolean reader fallback False as explained above. |
| `NEURON_RT_ALLOW_LEGACY_NEFF` | `"1"` |
| `NEURON_RT_DBG_ZEROCOPY` | `"1"` |
| `NEURON_RT_MAP_HBM` | `"1"` |
| `TORCH_NEURONX_ENABLE_HOST_CC` | `"0"` |
| `NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE` | With host-CC enabled on trn2, absent becomes `"true"`; explicit `"false"` raises an incompatibility error. No general absent-value default is asserted outside that branch. |
| `NEURON_RT_EXEC_TIMEOUT` | With host-CC enabled, absent becomes `"300"`; no general default asserted. |
| `NEURON_RT_DISABLE_EXECUTION_BARRIER` | Absent becomes `"1"` with host-CC enabled, otherwise `"0"`. |
| `TORCH_NEURONX_CPP_PROCESS_GROUP` | `"1"` |
| `TORCH_NEURONX_ENABLE_STACK_TRACE` | `"0"` |
| `TORCH_NEURONX_DEBUG_DIR` | `"/tmp/neuron_backend"` |
| `NEURONXCC_VERSION` | If absent, assigned compiler module version; empty string on import/error fallback. Inventory itself does not execute this code. |
| `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS` | `"0"` |
| `TORCH_NEURONX_ENABLE_LAZY_ALLOC` | `"1"` |
| `TORCH_NEURONX_HLO_CACHE` | `"1"` |
| `TORCH_NEURONX_ENABLE_CONCATENATION` | Reader fallback `"0"` |
| `NEURON_LOGICAL_NC_CONFIG` | `get_lnc_setting()` expression; not evaluated. |
| `TORCH_NEURONX_DISABLE_NEFF_CACHE` | Reader fallback empty string, lowercased. |
| `TORCH_NEURONX_ATEN_GMM_ALIGN` | Reader fallback `"128"` |
| `TORCH_NEURONX_SKIP_OP_PRECONDITIONS` | Reader fallback `"0"` |
| `TORCH_NEURONX_NATIVE_INT64` | Reader fallback `"0"` |

The initialization source is SHA256
`91f3fd461b084f1426fbf1a2df9a910950d37c662e52270c1822a6338f31d99c`.
Its complete source already retained in the Sep28 API receipt matches the fresh
R144 file hash exactly; that matching source resolves the conditional defaults
above. Optional `_build_config.py` remains the only explicitly unobserved
override for the four build-config-dependent defaults.

The supporting `CompilerConfig` dataclass has: target =
`get_platform_target()`, model_type = `"transformer"`, optimization_level =
`"-O1"`, auto_cast = `"none"`, framework = `"XLA"`, lnc =
`get_logical_neuron_cores()`, and extra_flags = new empty list. These defaults
belong to that support class, not necessarily to the C++ dynamo compilation path
or the active training process. They must not be substituted for CLI defaults.

Fusion/partition/graph-size-related findings: NKI SDPA and vocab-sharding
environment gates exist as above. Flex-attention lowering has fixed Python
constants `_Q_GRP_SZ=128`, `_V_TILE_SZ=128`, `_SECTION_LEN=8192`;
these are implementation constants, not exposed options. No scanned public
fusion/partition/maximum-graph-size option was found. No constant was edited.

## B. Installed neuronx-cc compile help

Executed exactly:
`/home/ubuntu/workspace/native_venv/bin/neuronx-cc compile --help`.
Exit0; stderr empty. Full stdout 7,064 bytes, SHA256
`769b881872123634d444850335b6a792bb5dbe96a9d17bdab0e4261c87c8826f`.
Entrypoint SHA256
`6f68bf2a5fa3caedc4520795f0ef8e4bac039e3cee4036c704e67ee613ae4c9e`.

Every printed flag whose description matches tiling, blocking, spill, SBUF,
allocation, scheduling, pipelining or memory:

| Flag | Printed default | Relevant description |
|---|---|---|
| `--optlevel {1,2,3}`, `-O {1,2,3}` | 2 | O2 balances compile time/performance; O1 reduces compile time; O3 may increase compile time and exceed device memory limits. |
| `--hbm-scratchpad-page-size` | 512 MB | HBM scratchpad page size; must match `NEURON_SCRATCHPAD_PAGE_SIZE` at execution. |

No `--internal-*` or `--tensorizer-*` option is printed. No additional matching
tiling/blocking/spill/SBUF/allocation/scheduling/pipelining description is printed.
Absence from help is not proof that no private implementation option exists;
none is guessed or tried. `--enable-fast-context-switch` mentions model
switching and deferred weight-constant loading, but not the requested terms.
The full help, including other flags, remains in the exact receipt.

## C. Installed neuron-explorer and every view option

`which` over explicit SDK search path
`native_venv/bin:/opt/aws/neuron/bin:/usr/bin:/bin` resolves to
`/opt/aws/neuron/bin/neuron-explorer`. Binary size126,932,944 bytes; SHA256
`f1b609564aca32f0a1dc7768aa0389e0d3f724cbf0598a7fade5a057cbc2c008`.
Version is **2.32.0.498-b1f3998**, built 2026-08-10T05:09:49Z, commit b1f3998a.

Both `--version` and `view --help` exited0 with empty stderr. Complete view
help is62,764 bytes, SHA256
`7712ceda8e7b4539e03ff9627ea4d30d45a5d8f55cbc7030eea6a334122115b9`.
Its prefix says "exited with an error" even though help exits0; this is preserved,
not mistaken for a successful profile decode.

| Group | All printed options |
|---|---|
| Global | `-v/--version`, `-h/--help` |
| Input | `-n/--neff-path`, `-s/--session-file`, `-d/--session-dir`, `--source-code-root`, `-o/--cc-only` |
| Names/database/UI | `-m/--session-name`, `--db-endpoint`, `--db-org`, `--db-bucket`, `--display-name`, `-f/--force`, `-p/--port`, `--data-path`, `--disable-ui`, `--ingest-only` |
| Device detail | `--ignore-nc-buf-usage`, `--ignore-instruction-trace`, `--ignore-event-trace`, `--ignore-dma-trace`, `--ignore-instruction-hierarchy`, `--enable-memory-tracker`, `--enable-memory-analyzer`, `--dma-throughput-sample-interval-ns` |
| Output/schema | `--output-format`, `--terminology`, `--show-profile-schema`, `--show-device-profile-schema`, `--output-file`, `--processing-status-file`, `--json-pretty-print` |
| Sampling | `--fgtrace-sample-rate`, `--pprof-sample-rate` |
| System selection/grouping | `--ignore-device-profile`, `--ignore-system-profile`, `--system-trace-primary-group`, `--system-trace-secondary-group` |
| System detail/filtering | `--system-trace-ignore-host-memory`, `--system-trace-ignore-cpu`, `--system-trace-filter-instance-id`, `--system-trace-filter-neuron-core`, `--system-trace-filter-event-type`, `--system-trace-filter-ultraserver-instance-index`, `--system-trace-filter-ultraserver-id` |

Printed defaults: db endpoint `http://localhost:8086`; db org `AWS`; UI port3001;
DMA throughput sample interval10ns; fgtrace/pprof150Hz; primary system grouping
`instance_id,process_id`; secondary `worker_gid,lnc_idx,thread_id`.
Output-format choices are `db|summary-text|summary-json|json|perfetto|parquet`.
JSON defaults to `ntff.json` (device) / `system_profile.json` (system), and
Perfetto to `ntff.pftrace` / `system_profile.pftrace`; parquet treats output path
as a folder. No unprinted flag defaults are inferred.

`--ignore-instruction-hierarchy` explicitly trades detail for processing time/RAM.
`--enable-memory-tracker` requires XLA_IR_DEBUG=1 and XLA_HLO_DEBUG=1 before
compilation plus capture's `--enable-dge-notifs`; `--enable-memory-analyzer`
requests detailed allocation intervals/addresses. These are capability facts,
not authorization to recapture, decode or recompile. Summary format alone is
not a proof of bounded decoder RAM or successful model/trace matching.

## Execution containment and limits

Original warm CS2 source/ranks/start ticks and held lock remained unchanged.
Final observed warm step2590 had254 seconds remaining. One CPU (0), nice19,
autoload disabled, AS/aggregate RSS2GiB, timeout60s, CPU45/50s and output2MiB
were enforced; no cgroup memory enforcement is claimed. Terminal exit0,
elapsed1.615178289823234s, sampled peak descendant RSS157,143,040 bytes,
own child reaped, no remaining descendants, protected inputs unchanged,
empty stderr, no device lock acquired and no training signal sent.
The previous failed CS12 CPU was used only as authenticated released-occupancy
evidence; its failure was not converted into PASS.

Frozen worker:
`a6b831343bd2288846d1694e8a010ac0632ed9e893d1b23c90c897b0f7c3cc4e`.
Generated supervisor:
`d24e1968e3fa8c1a7cd0625f69ef13329f9bd9b71c3cc1f6e7b3c923c45a2b04`.
Root performed the host execution. This reviewer performed only local
metadata/hash/source interpretation, not laptop numerical tests or host calls.

