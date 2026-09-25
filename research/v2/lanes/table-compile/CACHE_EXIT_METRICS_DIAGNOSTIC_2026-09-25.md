# Existing SDK exit metrics: non-blocking observability, not a cache fix

Read-only installed-source inspection by root, 2026-09-25 17:00–17:09 UTC.
No live process was imported into, attached to, signalled or reconfigured.
The failed initial search used an extra package-directory component; the
correct installed package is `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx`.
An empty result from that first search is not evidence of a missing API.

## Supported source-level facts

- `neuron_dynamo_backend/metrics.py` defines
  `TORCH_NEURONX_ENABLED_METRIC_TABLES` and `TORCH_NEURONX_METRICS_DIR`.
  Its registered `graph_stats` table includes graph ID/name, collective flag,
  node count, timestamp, FX/lowering time and NEFF compilation time.
- `record_compilation` already appends these records unconditionally. The two
  export settings do not enable new instrumentation in the measured step.
- `_write_csv_at_exit`, registered through `atexit`, checks the enabled tables,
  synchronizes, resolves compile times and writes
  `metric_table_graph_stats_rank<rank>.csv` to the requested directory.
  It does not create that output directory; the Operator must precreate a new
  owned per-phase receipt directory and archive the files after process exit.
- `get_dynamo_metrics()` is also exposed through the Neuron API. It synchronizes;
  do not inject a call into a running job or the timed training window.
- General `torch_neuronx.metrics` collection has a separate enable flag and
  defaults disabled. Enabling broad counters or concatenation profiling is NOT
  part of this diagnostic. No counter clearing or reset is authorized.
- `_get_compilation_time_ms` returns zero both for a zero-time matching cache
  entry and when no entry matches. Therefore a CSV zero alone cannot establish
  a persistent-cache hit. Nonzero graph compile and lowering times can still
  separate where startup time was spent. Graph IDs are not assumed equivalent
  to C++ persistent filenames without their key transformation evidence.

## Narrow Operator instruction

After confirming the installed implementation, include these logging-only
settings in the otherwise-approved unlaunched architecture run and BOTH FP8
timing arms, if it is a straightforward addition to their existing bindings:

```
TORCH_NEURONX_ENABLED_METRIC_TABLES=graph_stats
TORCH_NEURONX_METRICS_DIR=<new existing per-phase receipt directory>
```

Use the same policy for OFF and ON; only the output directory differs. Record
the effective environment. Keep train.py bytes, compiler/cache flags, source
path, native kernels, measured step window and all charging unchanged.
Include exit synchronization and CSV overhead in actual process-through-exit
timing. This is not an extra allowance, performance result or launch approval.

No separate training run, cold replay, SDK edit, monkeypatch or logging framework
is needed. If unavailable or it would delay the next launch, omit it and record
that limitation. It is optional evidence collection, not a new gate. The
existing exact-size CPU/native correctness gates still apply to new candidates.

At this writing Operators have been asked to inspect/prepare this configuration;
no export from a real native run is claimed yet. The repeated-startup root cause
and a behavior-changing cache fix remain unproven.
