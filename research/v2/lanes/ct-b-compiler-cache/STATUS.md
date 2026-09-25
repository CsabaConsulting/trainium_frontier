# CT-B: compiler/cache coordination, bounded research

04:22IST update: stable-source S1 actually began22:47:39.841696UTC ontrn22;
fourworkers, original lifetimeflock and freshps/neuron-ls archived7b16ffbd.
S2 notyetlaunched: complete S1gate/receipts/quiescence required first.
Different75b9Recursive architecture cold-five ontrn21 passed831.421556s;
this is NOT stable-source reuse evidence. Cache benefit remains unmeasured.

Latest26September04:08IST: separate stable-source S1/S2 operational diagnostic
af4f986c now has independent90-check mechanical PASS, rootreplayed normal/-O,
and conditional releasecdbb183e behind trn22currenteval/checkpointCPU/K2.
See20260926-stable-source-pair-assignment.md. No candidate arithmetic or
submissioncommand edit. Native cache reuse remains UNMEASURED; missing actual
NEFF reuse evidence must remainUNRESOLVED even with equalkeys/zerocompiletime.
Current trn21is running75b9 cold-five, not this pair; old report below retained.

2026-09-25 19:14 UTC. Separate racer B; no candidate edits, compiler execution,
host calls, cache writes or tensor allocations. This is not independent review
of a patch I implemented. Current authority is `workflow/COMPILE_TIME_GATE_2026-09-26.md`:
cold five-step startup <=900s before any full architecture rerun; target870s.

## Result

No justified new production patch emerged from the advertised compiler/cache
shortcuts. Prioritize the independently reviewed first/steady graph-state fix
and CT-A graph specialization work. Do not spend a native slot adding already
present flags or implementing naive rank-zero priming.

`audit_static.py` ran normally and with `-O`: 21 checks PASS, stdlib only.
It independently hashes the raw CSVs/input, groups graph IDs, and executes the
exact original93ae `make_plan` on shape-only Python objects. Its reconstructed
65-parameter ownership summary equals the raw full-parameter G2 summary exactly.
`STATIC_RESULT.json` preserves all input hashes and calculated facts.

## Concrete findings

1. Actual compiler argv already contains `--model-type transformer` and
   `--optlevel=1`. It also contains an earlier `-O2` (and `--auto-cast=none`
   before later BF16 autocast arguments). Adding the same flags is not a new
   experiment. The captured compiler is2.26.6360.0+6f180f47. Current official
   documentation says the right-most competing option wins, but its public
   version is newer; installed parser/effective-config evidence would settle
   exact duplicate handling. Do not infer an O2 misconfiguration from the first
   occurrence alone. Raw `compiler-log-prefix.txt:1-12` also proves transformer
   partitioning was actually selected.

2. Four distinct model graph IDs occur on every rank: node counts704/1217
   then666/1169. Each ID has exactly one nonzero reported compiler timer:
   rank0 first F/B205.380/521.889s, rank1 later F202.538s, rank3 later B465.466s.
   Sum1395.273s is a sum of compiler counters, NOT launch-to-fifth-boundary
   startup. All graph CSV rows report no in-graph collectives. Other ranks'
   zero timers alone do not prove persistent cache hits, but the records do
   not substantiate four concurrent duplicate large compilations.

3. There are13 small194-node inference occurrences /8 distinct IDs, with
   total reported compile44.471s. The shape-only ownership reconstruction
   explains why rank-local compiled Muon shapes cannot all be replaced by
   rank0's graph set. Numbers below are stack counts after the existing
   singleton-to-two padding, not new padding or altered updates:

   | Rank | 8x32 | 1024x1024 | 1024x4096 | 4096x1024 |
   |---|---:|---:|---:|---:|
   | 0 | absent | 6 | 2 | 2 (one real) |
   | 1 | 2 (one real) | 3 | 2 (one real) | 3 |
   | 2 | absent | 6 | 2 (one real) | 2 |
   | 3 | absent | 5 | 4 | 2 |

   This exactly gives8 distinct shape signatures, consistent with the small
   graphs; exact key-to-stack assignment remains an inference without their
   individual IR. No claimed one-to-one native binding is manufactured.

4. Archived installed `CompilationCache.cpp:84-120` already selects a shared
   service socket keyed by UID and distributed MASTER_PORT by default, with
   fallback to local compilation. Separate per-rank Dynamo/AOT cache paths are
   intentional because guards can hold rank-specific DeviceMesh data; merging
   them is not the same as sharing NEFFs. The persistent NEFF/local-lock roots
   were already shared. Exact full service locking/thread-pool implementation
   is absent from these excerpts, so no arbitrary concurrency variable is
   proposed. At the18:04 snapshot, one compiler root is under rank3, followed
   by compiler descendants in the same work directory; counting those as
   independent training-rank compilers is wrong.

5. Naive `if rank==0: first_training_step(); barrier()` is unsafe: gradient
   all-reduce and `OwnedOptimizer.step` all-gather require all ranks in matching
   order. Even moving only model F/B ahead of other ranks would need explicit
   restoration of gradients, RNG/state/cache lifetimes, on-clock timing and a
   proof it saves anything beyond existing coordination. It would not prime
   the other optimizer shapes. Do not build this speculative arm now.

## Smallest feasible next CT-B candidate

An OFF-by-default **generic-versus-transformer compiler specialization** is
a real contrast, unlike adding transformer again. First falsify it without a
training run: replay the already-captured first-forward MLIR (SHA927f008f…)
with the SAME installed2.26.6360 compiler, targettrn2/LNC2, BF16 policy and
effective optlevel1, changing only model type. Use private scratch outputs,
not live persistent caches. The literal original compiler invocation is
archived in `native/live-compiler-input/capture.json`; retain its base argv
for the control and replace only transformer for the arm after checking the
installed parser. Record wall/peakRSS/NEFF/compiler flags. This is a proposal,
not an execution assignment, and not a guaranteed improvement.

The original binary MLIR is already archived (16,658bytes), so no training
graph recollection is necessary for that forward comparison. Backward IR is
not in this packet: a forward-only compile saving cannot establish the cold
five-step gate. A later full source arm would still need defaultOFF identity,
native correctness, actual cold startup and steady whole-step measurement.
Official budget charging cannot move compilation off-clock by shipping this
research precompile. The existing compiler is not installed in this laptop
runtime, and archived backward compile RSS reached15,415MB; no local full
compiler job or package installation is proposed.

Public `neuron_parallel_compile`/XLA and vLLM CPU-capture docs do not establish
support in the competition's native backend. Likewise there is no installed
evidence here that a `scan` layer loop is supported. Do not import those APIs
or worker-count environment variables by analogy.

## Evidence and next

- Raw: `research/v2/operator/20260925-trn22-l2-93ae-cpu-r1/native/`:
  `completed-training/metrics-training/metric_table_graph_stats_rank*.csv`,
  `live-compiler-input/{capture.json,module.mlir,compiler-log-prefix.txt}`,
  `startup-alarm-1804/{observation.json,compiler-log-tail-0.txt}`.
- Installed-source excerpts: `research/v2/operator/20260925-trn21-live-cache-diagnosis/`
  and `research/v2/operator/20260926-trn22-arch-initial-cache-cpu-r1/native/cache-readers/`.
- [AWS compiler flags](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/index.html):
  generic/transformer, levels1–3, competing option precedence. Accessed2026-09-25UTC.
- [AWS compiler FAQ](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/faq.html):
  compilation can run on CPU without accelerator execution. This capability
  does not establish availability of the exact installed SDK locally.
- [AWS parallel compile](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.25.0/frameworks/torch/torch-neuronx/api-reference-guide/training/pytorch-neuron-parallel-compile.html)
  and [vLLM CPU compile](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/compilation/cpu_compilation.html)
  are distinct flows, not evidence for native training integration.

Next: Supervisor can select the bounded same-SDK offline flag falsifier if a
CPU/compiler slot is useful. CT-A/current priming startup test remains the
better-supported immediate path. No new gate, queue, framework, full rerun or
claimed compile/performance gain is created here.
