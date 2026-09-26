# P3 bounded component diagnostic on audited e1e2

Implementer preparation only. No native execution, launch permission, selected
kernel optimization or performance result. K2 source63925519 remains frozen.

This single-file train.py derivative adds an explicit `--p3-component-timing`
development mode. Without that flag, original code is unchanged, including
training, model/eval, data, optimizer, compile settings and clocks. The diagnostic
returns before data loading/model construction/optimizer/checkpoint/evaluation.
It emits no fake optimizer-step or official-budget markers and must not be run
through a scorer expecting training. Main/sole Operator supplies the proposal
and ordinary lifetime-lock/preflight/direct development launch protocol.

## Measured operations and boundaries

Fixed production geometry: four ranks/LNC2, BF16, batch32/context1024,
32768 tokens/rank, C1024, 8 attention heads of128, MLP width4096.
Source records five attention and eight MLP calls per microstep. It times one
component call at a time under four-rank contention; these counts are context,
not a claim that the isolated timings sum to production graph time.

1. `sdpa_core`: exact `F.scaled_dot_product_attention(..., is_causal=True,
   scale=None)` used in e1e2. Q/K/V originate as contiguous B,T,H,D and are
   transposed to B,H,T,D, stride (1048576,128,1024,1). Upstream gradient has
   the same transposed-BTHD stride, reflecting the caller's transpose/materialize
   boundary. Native autograd supplies the actual backward; no guessed flash
   backward primitive. Q/K projections, RoPE/norm and output projection excluded.
2. `mlp_complete_original_custom_autograd`: direct `_FusedMLP.apply` with
   BF16 x[32768,1024], up[4096,1024], down[1024,4096]. These are the actual
   eligible training boundary and cached-weight dtypes. Forward invokes the
   unchanged `_fused_mlp_kernel`; backward invokes unchanged `_FusedMLP.backward`.
   Weight-cache refresh/FP32 Parameter cast-back, normalization and residuals
   are outside this boundary and excluded.
3. Activation forward: `relu(pre)`, its square, and saved
   `relu.T.contiguous()`, with pre[32768,4096] BF16 contiguous.
4. Activation backward: saved relu[4096,32768] square plus
   `dhidden_t * (relu_t * 2.0)`, both inputs contiguous BF16.
5. Matrix forward: the two separate projection products, supplied with an
   explicitly materialized activated tensor. Both products are returned.
6. Matrix backward: the exact four products for dhidden, dX, dUp and dDown,
   supplied with explicitly materialized dpre_t and activated_t.

The decompositions are intentionally **isolated materialized operations**.
They are not exclusive slices of the fused full MLP graph. Compiler fusion,
layout, scheduling and reuse may make their costs nonadditive. No subtraction
or multiplication is reported as whole-step saving; SDPA and activation results
can only motivate a subsequent measured optimization.

## Timing and bounded output

Every callable uses `torch.compile(..., backend='neuron')` with unchanged
production compiler environment. Ranks barrier and synchronize before each
block; timing starts after that barrier and ends after device synchronization.
Records are host wall-clock component times, including dispatch overhead, not
exclusive device instruction durations. No trace/profiler/NTFF/decoder is used.

First forward and first backward are measured independently and explicitly
recorded as **compilation-and-execution first-call seconds**, not pure compiler
duration. They are excluded from steady measurements. To measure genuinely
cold compilation, the Operator must select one new isolated empty compilation
cache for the whole diagnostic and record that environment. The source records
the cache path and makes no unsupported claim that a pre-existing cache is cold.

Then two warmup blocks precede four timed blocks by default, three repetitions
per block. Complete SDPA/MLP forward and backward have separate records. Each
backward block repeats the same saved forward graph using retain_graph=True;
this executes the original VJP without recomputing/charging another forward.
An installed-backend failure of repeated saved-graph backward is a failed
diagnostic, not permission to substitute another primitive or conceal overhead.
For named backward-only decompositions, the `forward_ms_per_call` field means
invocation time of that explicit backward-stage callable, not model forward.

Finite outputs/gradients are checked outside timed blocks; nonfinite results
fail. Synthetic random inputs use seed42 and bounded scales. JSON records shapes,
strides, dtypes, raw block times, medians, first calls, rank, config and status.
No synthetic data represents the training distribution. Rank files remain
separate; report the slowest-rank result when assessing chip contention.

Default maximum is900s from diagnostic entry, allowed range60..1200s. A daemon
watchdog emits TIMEOUT and exits124 at the limit, including compiler or teardown
stalls. The Operator still owns the whole job, holds its lock through cleanup,
and must reap its compiler children after failure; source does not kill other
processes or touch lock files. A process exit alone does not establish host idle.
Each completed component persists a small per-rank JSON snapshot in out-dir;
timeout during a component leaves prior snapshots plus the terminal log record.
No large tensors/checkpoints are written. Expect only kilobytes/rank.

Prospective direct diagnostic argv (Operator assigns unique label/out-dir,
empty cache, exact source hash and hard process supervision; not a launch):

```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --p3-component-timing --device-type neuron --seed 42 --p3-max-seconds 900 --p3-blocks 4 --p3-repeats 3 --out-dir <unique-diagnostic-output>
```

One graph per exact callable plus its backward may compile; startup feasibility
is unknown. If the deadline expires, report incomplete components and stop.
No BPB, throughput or kernel-win inference follows automatically.

## Local evidence

`test_p3.py` imports actual source with disclosed unused parquet/test-prepare
import shims because the local venv lacks pyarrow. No source function is replaced.
Small BF16 tests match SDPA forward/backward to plain Torch, full MLP forward
to simple linear/ReLU²/linear, original MLP backward to its literal matrix/
activation boundary decomposition, and each split operation bit-for-bit.
The repeated saved-graph VJP pattern passes on CPU, not yet on Neuron.

The actual eight-layer width64/context8/batch2 model matches base losses,
gradients and updates for three SGD .01 steps with default OFF. An AST check
removes only diagnostic functions/parser options/early dispatch and confirms
the entire remaining source matches audited e1e2. This is stronger structural
scope evidence, not full-size/native equivalence.

Run in this folder with the existing venv:

```
/home/poojith-devan/venv/bin/python -B test_p3.py --base /home/poojith-devan/Desktop/Trainium/workflow/candidates/confirmed-clock36-116188/train.py
```

`cpu-result.json` is Implementer evidence. One independent reviewer follows;
no extra gate or host action is added by this package.
