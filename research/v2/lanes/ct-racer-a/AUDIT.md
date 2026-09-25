# CT racer A: specialization audit, no new production patch

Exact adopted6ef7 and original93ae were audited, with their existing default-OFF
initial-cache candidates bcbd/3e79. `graph-source-audit.json` binds immutable Git
objects, AST references, native graph IDs and the captured compiler input. The
stdlib checker passes normally and under `-O`; no tensor job or host access.

## Findings and selected first experiment

The smallest existing candidate remains the already-implemented differentiable
initial cache refresh, not a new compiler flag or optimizer change. Test exact
bcbd on6ef7 and exact3e79 on93ae only after the separate source/CPU/fresh-review
gates. Each adds four executable lines and leaves every subsequent refresh,
update and accounting hook unchanged. The priming packet's actual gradient,
optimizer, checkpoint and OFF proofs remain its evidence, not a new independent
replication by this audit. No duplicate flag or claim of a new implementation.

| Native original93ae evidence | Observed count |
| --- | ---: |
| Rank0/1/2/3 recorded graph rows | 7 / 8 / 7 / 7 |
| Initial model forward/backward nodes, every rank | 704 / 1217 |
| Later model forward/backward nodes, every rank | 666 / 1169 |
| Distinct model graph IDs across all ranks | 4, each shared by all four ranks |
| Other no-backward/inference rows | 13, with8 distinct IDs, all194 nodes |
| Total rows / distinct IDs | 29 /12 |

The 194-node `inference` labels **do not mean public evaluation**. They occur
between initial backward and the second training forward, while actual argv
has `--no-eval-public`. Rank0 training log reports three compiled Muon groups
`(6,1024,1024)`, `(2,1024,4096)`, `(2,4096,1024)`. This is strong timing/source
evidence for optimizer regions, not exact call-site attribution of every ID.
Other ranks own different matrix batches. Do not remove these as duplicate
evaluation graphs or assume equal node count means equal shape/work.

Archive `c2c47f7d`, directory
`research/v2/operator/20260925-trn22-l2-93ae-cpu-r1/native/completed-training/metrics-training/`
contains the four graph tables. The previously generated, source-bound JSON
inventory is used here; no new hardware measurement. First step durations were
847.90s and695.40s, then about0.49s. These are not per-graph compile times.
The recorded first and later model pairs support two specializations; no causal
claim that cache priming removes them has yet been measured.

The first forward bytecode captured at17:43:12UTC is16,658 bytes, SHA
`927f008f870ad145057d17cdb33f66dfeee965cd8392db47cf7063ff4b7d838d`, exactly the
input component of all four ranks' initial-forward graph ID. Shared input IDs
do not prove concurrent duplicate compilation or successful cache reuse. The
snapshot contains **six** compiler command entries with identical argv, not a
verified four-rank-to-four-compiler mapping; it lacks each process cwd/input and
parentage. Partition/compiler workers can also appear. Zero compile_us means
unknown and cannot identify a hit. Full persistent keys additionally depend on
platform, compiler config and versions, not only the recorded graph ID.

## Exact source paths that can and cannot specialize

Line numbers below refer to source6ef7 unless stated; machine-readable mappings
for93ae/3e79 are in the audit JSON.

- `Linear:1444`, specifically initial `_cached_weight=None` at1461 and branch
  1468–1471: first forward casts the FP32 master internally; later forwards
  receive the differentiable cached BF16 cast. MLP `make_forward:4446` has the
  same cache-versus-cast choice. Post-update refresh is4039; ordinary current
  initialization does not refresh before the first model call. This is the
  concrete first-to-second graph difference targeted by priming. Architecture
  has38 cached Linear instances, consistent with the forward node reduction38;
  backward reduction48 is not explained solely by that count. Correlation is
  not an operator-level diff or proof of cause.
- Full model compile is3863 (`dynamic=False`). The real loop3953 calls
  `model(x,y)` at3977, with stable packed shape32x1024 per rank, GA1. It does not
  pass LR, elapsed time, step index or the optimizer state into this compiled
  model. `zero_grad(set_to_none=True)` already repeats every step. No training
  mode switch, batch ramp, table freeze or EMA is enabled in these profiles.
- `_muon_step:2861`, compiled call2911: same-shape stacks are already batched;
  cached function key is `(tall,ns_steps,red_dim,tuple(g.shape))`. Lazy moments
  and scratch buffers are allocated **outside** `_muon_math`. Rebound output
  buffers could in principle change metadata/alias guards, but the archive
  shows no later optimizer graph records beyond startup. Eager preallocation
  alone is therefore a weak candidate and might only move allocation cost.
- `_muon_compile_scalars:2586` already uploads a persistent5-element FP32
  tensor each step. `lr`, `lr*wd`, momentum and its complement enter as device
  tensors, preserving the deliberate float32/float64 host rounding contract.
  With RecursiveON, `one_beta2` is a tensor from the **first call**, not a float
  that switches type when cooldown starts (2915–2916). With original93ae
  RecursiveOFF it stays the same constant Python value. Schedule changes do
  not introduce a new Python-float argument into the compiled Muon function.
- LR/momentum/beta2 helpers3248/3316/3375 execute in the Python outer loop.
  AdamW's `_device_scalar:2449` likewise stages changing LR; AdamW runs the
  existing eager foreach path2673, not a compiled optimizer wrapper. Native
  eager scalar-list behavior still needs observed cache/trace evidence; these
  source facts do not prove every runtime NEFF stays constant forever.
- There are three syntactic `torch.compile` calls in each source. Besides
  model and Muon, `make_forward` has an optional separate-MLP-region compile
  at4479. The installed training adapter4862 passes `forward_backend='native'`
  and leaves `separate_region=False`; that third compile site is inactive.
  Do not count six/eight mandatory per-layer compiled MLP regions from it.
- Public evaluation is guarded after the loop at4101 and disabled by the exact
  training argv. `load_for_eval:3211` constructs the uncompiled evaluation
  model; it contains no explicit `torch.compile` call. Fresh public20M is a
  separate process and remains required. Native eager eval kernels may compile
  for scorer shapes, but they cannot explain the two startup training graphs.
  No evaluator, causality, context widening, attention or load behavior is
  changed to reduce compilation.

No exact6ef7 graph_stats archive was found in the canonical archived `research/`
inventory; thus native6x model/optimizer graph counts and later schedule-switch
graphs are UNKNOWN, not copied from architecture. Archive coverage here does not
include all eager operators, NKI internals or compiler partition NEFFs.

## Flags and prior trials: avoid duplicates

Source6ef7 module lines20–28 already default `NEURON_CC_FLAGS` to optlevel1/BF16
matmul autocast for bare main. Actual captured compiler argv is:

```text
neuronx-cc compile module.mlir --framework XLA --target trn2 --lnc 2
 -O2 --model-type transformer --auto-cast=none
 --optlevel=1 --auto-cast matmult --auto-cast-type bf16 --output module.neff
```

Both `--model-type transformer` and the trailing optlevel1 override are already
present. Adding their spelling is not a causal optimization. Exact parsing/
resolved options, not a leading `-O2` substring alone, determine compiler mode.
No flags or SDK/cache implementation are modified by this audit.

Previous first-cache audit at`b635c8b0` searched ledger, decision log, results and
Git history; T1.10/`2f32abc2` batched **post-update** cache refresh saved only
~0.17% of an older step and was not adopted except lookup hoisting. That is not
initial priming. `_device_scalar` and compiled Muon already fix historical
per-step LR specialization; K1 audit`b2bcd793` proves batching and four-rank
ownership already shipped. Native caches have real installed read/lock paths;
the earlier A2 matched-input pair was unavailable. Do not resurrect generic
location/hash stripping or infer a root cause from directory occupancy.

## Ranked next actions / bounded falsifiers

1. **Existing priming candidate**, normal grad mode inside counted step0.
   After independent gates, true-cold5-step base/candidate source bindings.
   Falsifier: primed source still records two model forward/backward pairs, or
   invalid gradients/checkpoint, or startup fails the predeclared threshold.
   No new patch is necessary before that evidence arrives.
2. **Only if duplicate compilers are established**, investigate serialized
   same-key compile/cache loading. Current identical graph IDs + six generic
   process entries are insufficient. Next already-approved cold test should
   associate compiler PID/parent/cwd/exact input key with per-rank graph role.
   No rank gating, extra training forward, barriers or dummy backward is added
   here; such changes can deadlock collectives or alter first updates.
3. **Only if priming leaves a distinct non-cache specialization**, inspect actual
   first/steady exported graph inputs/guards, especially saved BF16 cast alias
   metadata and Muon moment outputs. Do not add initializer work outside the
   official clock or preallocate blindly.
4. **Schedule/eval removal** presently has no supported target in exact sources.
   Observe any new graph at warmup/cooldown transition on an otherwise-required
   run before selecting a change. No table/batch-ramp/eval bypass or new LR mode.

Native layer-loop/scan support is not established by this source audit; no
torch_xla API is assumed available in the native backend. That direction and
CPU compiler-flag racing require their own actual installed-API evidence.
