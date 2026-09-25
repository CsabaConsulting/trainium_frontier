# P: one step of the actual training loop

This is a diagnostic wrapper, not a new launcher, model, optimizer, timing win,
or quality run. `train.py`, `prepare.py`, tokenizer, evaluator and launch science
are unchanged. The wrapper is default OFF and uses `runpy` on exact source.
The ON path adds filtered Python trace hooks and profiler annotations only.
Its commit and `BINDING.json` require independent review before Operator use.

## Bound execution

Use the existing manual Operator route and shared host job lock, not a second
launcher. Check host identity, unique unused label/output, source/support hashes,
the relevant source gates, no competing Neuron job/compiler, STOP state and disk
headroom under `research/experiment-protocol.md`. Preserve the exact reviewed
environment/cache paths; in particular the recorded compiler setting already is
`--optlevel=1 --auto-cast matmult --auto-cast-type bf16`. Do not add debug flags,
clear caches or independently prewarm. Retain environment, preflight, command,
PID/process-group, raw rank output, compiler/graph statistics, wallclock/startup,
exit status, artifacts and quiescence receipts. Allow at most 2700 wall seconds
for this bounded diagnostic, then the existing owned-process cleanup procedure.
The source still declares and enforces its 1800-second training budget; startup,
profile preparation/export and exit are separately visible and never erased.

Operator stages exact source from the commit in `BINDING.json`, unchanged
`prepare.py` through the proven import path, and this exact wrapper. Its existing
job driver substitutes source, profile and output paths below and preserves the
outer lock/preflight/containment. Create only the fresh profile parent before
launch; per-rank directories are exclusive-created by the wrapper. Never reuse a
failed label or overwrite an existing profile/output. Example bound inner argv:

```text
torchrun --standalone --nproc_per_node=4 profile_actual_loop.py
  --train-py /OPERATOR_FRESH_STAGE/train.py
  --capture --out /OPERATOR_FRESH_STAGE/profile
  --target-step 20 --device-ranks 0 --capture-nc 0-7 --max-events-per-nc 200000 --
  --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct
  --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce
  --seq-len 1024 --no-eval-public --matrix-lr 0.015 --seed 42
  --max-train-seconds 1800 --num-steps 22 --out-dir /OPERATOR_FRESH_STAGE/out
  --recursive-schedule
```

That is exact 6ef7 / `fs-20260926-profile-recursive6ef7-trn21-r1`. For the one
Supervisor-selected architecture profile, replace only the final
`--recursive-schedule` with `--unet-8x1024 --attention-layers 0,2,4,6,7` and bind
93ae / `fs-20260926-profile-architecture93ae-trn22-r1`. If the independently gated
3e79 port is selected instead, use its exact source and additionally append
`--initial-cache-refresh`, label `fs-20260926-profile-architecture3e79-trn22-r1`.
This is a choice of source, not permission to run both architecture alternatives.
The wrapper rejects extra/reordered/overriding train arguments in capture mode.

No public evaluation is requested. The real loop still saves its ordinary final
checkpoint. Preserve it under existing policy; do not treat it as a quality
candidate or perform an extra scored evaluation. Production quality launches
continue to invoke unchanged `train.py` directly, without this wrapper.

## Coverage and interpretation

The installed API source is archived at commit `63ebb7f0`,
`research/v2/operator/20260926-profile-capability-trn22/profiling.py.txt`.
The wrapper verifies its exact installed SHA. Operator confirmed the CPU and
PrivateUse1 enum values exist; no successful profiling session is claimed yet.
The [AWS native profiling guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/tools/neuron-explorer/how-to-profile-workload.html)
specifies CPU plus PrivateUse1 activity and the runtime/device modes. DEVICE
capture can consume about 5 GB of Trn2 HBM, so rank0 device capture is the bounded
default; all four ranks collect CPU/runtime traces in separate output directories.
Do not expand device ranks without a revised binding/review.

The config is created inside the genuine initialized main before step0, enabling
metadata precaching during the real warmup. Steps0–19 execute normally; precisely
step20 is captured; stop/export occurs before step21, then the unmodified source
finishes step21 and exits normally. There is no extra sync or Tensor dispatch
mode and no module/optimizer/data replacement. The source's existing syncs and
control-tensor reads delimit real work. The hook is removed after export.

CPU `P/` ranges annotate forward, backward, data fetch, gradient unscale,
clip/reduce, optimizer, weight-cache refresh, existing syncs and boundary marker.
Actual calls to Muon groups, scalar uploads, AdamW groups, all-reduce and
all-gather have nested inclusive ranges. An eager Muon math range exists only if
that function actually executes in Python; compiled Newton–Schulz must be
identified from correlated runtime/device kernels. These are host-dispatch
ranges, **not exclusive device durations**. Summing nested ranges double-counts.
Data-fetch duration is the observed call/wait, not automatically all loader work.
Async CPU loader work may occur in other threads or outside the selected step.

Required analysis reads the actual exported trace and nonempty NTFF/NEFF/runtime
files, correlating kernel and collective intervals to the original source ranges.
Report forward/backward, optimizer including NS, all-reduce/all-gather, embedding
backward and data; report unattributed/overlapped time explicitly. Rank0 DEVICE
coverage cannot establish the other ranks' device workload or their critical
path. Report every observed matmul/SDPA dtype from trace/device IR metadata,
separate FP32 masters/residuals/optimizer arithmetic, and retain UNKNOWN for
compiled inner operations without metadata. Static casts and missing events are
not dtype proof; absent embedding events do not mean zero embedding work.

`profile.json` contains actual event shapes/dtypes if the installed profiler
exposes them, phase CPU times, inventory and model/FLOP metadata, but intentionally
does not manufacture a completed device breakdown or an MFU number. Independent
analysis should estimate FLOPs using the actual layer/attention configuration
(eight-layer partial attention is only layers0,2,4,6,7). The source's
`estimate_flops()` is an explicitly imperfect estimate, not executed FLOPs.
For an approximate training MFU, name the reviewed FLOP convention and divide
FLOPs/update by **667e12 × unprofiled seconds/update** for the one full Trn2 chip;
667 TFLOPS is the documented dense BF16 peak in the
[AWS Trainium2 specification](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/about-neuron/arch/neuron-hardware/trainium2.html).
Use steps10–19 after rejecting compile/stall contamination, aligned across ranks
(maximum per-step rank time, then report distribution), not the captured duration.
The wrapper records these pre-capture boundary intervals. They still include
filtered Python tracing/metadata-precache overhead; compare original source log
step durations and disclose that bias. Do not call this uninstrumented throughput.

Exit0 / `PROFILE_DIAGNOSTIC_COMPLETED` means only the wrapper window and source
completed. Acceptance as profile evidence additionally needs four rank receipts,
correct step/argv/hash, nonempty relevant native artifacts, a genuinely steady
window (compiler evidence plus timing distribution), no NaN/error/overflow, exit
and ordinary checkpoint evidence, and independent attribution review. Missing
data, truncated event buffers, recompile, early budget exit or device API failure
is incomplete/failed evidence, not a zero-time phase or a performance win.

## Self-checks and known limitations

```text
python3 -m unittest discover -s research/v2/lanes/p-profile -p test_profile_actual_loop.py -v
python3 -O -m unittest discover -s research/v2/lanes/p-profile -p test_profile_actual_loop.py -v
```

Eight stdlib/receipt tests pass in both modes. They validate all three immutable source
landmarks, exact bounded argv, refusal of overriding flags/existing trace hooks,
one balanced step20 window, unchanged fixture arithmetic and restored argv/path,
and incomplete-window behavior, plus the archived tiny actual-source CPU pair.
After the fresh reviewer released the laptop slot, one sequential exact6ef7
OFF/ON pair ran: depth2/width128, four ranks, three synthetic1x8 steps, fixed clock,
OMP/MKL/OpenBLAS1. ON uses the actual P hook plus CPU profiler at test-only step1.
All ranks' losses, all22 final parameter digests and ownership are bit-identical;
strict checkpoint reload passes. `cpu_g2_hooks.py` is a **test-only** bridge to the
existing G2 shims; it never runs natively. This is not the full-size launch graph,
native profiler, compiled backend or architecture8 proof. Tensor runs used normal
Python; only their receipt assertions were repeated under `-O`. Slot released.
No full-size laptop model, host call, training launch or SDK change.

One initial `--describe` attempt accidentally selected Recursive **primed bcbd**
rather than allowed6ef7; it failed `UNREVIEWED_SOURCE` as intended. Correct6ef7 and
3e79 describe checks passed; the source tests also validate exact93ae. No native
retry or swallowed capture failure occurred.

Two import-only CPU attempts are also preserved: `tiny-off.json` used a nonexistent
old PyArrow test-support directory and failed `No module named pyarrow`;
`tiny-off-r2.json` corrected that path but the CPU runtime lacks `requests` needed
by real prepare. The successful `tiny-off-r3.json` / `tiny-on.json` explicitly use
existing `--stub-prepare` and the existing Recursive test-only PyArrow/Gloo support.
The stub never reads real data. None of these CPU shims enters the native wrapper,
train.py, prepare.py or production import path. See `SELF_CHECK.md` for exact argv.

Profiler preparation, sys.settrace overhead, record-shapes/stack capture and
export can perturb timing, memory and actual elapsed-based LR/WD. The fixed-clock
stdlib arithmetic check cannot prove native update identity or no Dynamo/backend
interaction. Inspect compiler evidence for retracing before calling step20 steady.
No change is moved outside official accounting and no quality conclusion follows.
