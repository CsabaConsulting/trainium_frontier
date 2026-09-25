# Initial differentiable weight-cache refresh: two isolated source ports

Status: **Implementer tiny self-check PASS. Independent source review,
full-size CPU G2, native correctness/startup and quality gates PENDING.**
No host calls, native launches, queue/ledger edits or submission changes.

## Exact sources and scope

| Profile | Exact base commit / source SHA-256 | Candidate source SHA-256 |
| --- | --- | --- |
| Adopted Recursive6x1024 | `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b` / `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3` | `bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522` |
| Standalone architecture8x1024 | `1e21c3d797eed764f108c41546b270065ad5650c` / `93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5` | `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c` |

Both source paths are `workflow/proposals/research-v2/train.py` on separate
branches. This packet is on `codex/20260925-recursive-initial-cache-refresh`.
The architecture source is committed at
`b4d83b5db66971f3c348837f04f53cba65aa6d33` on
`codex/20260925-architecture-initial-cache-refresh`. Extract that complete
file for architecture; applying only its four-line patch to6ef7 is not a
valid architecture source. No combined architecture+Recursive stack exists.

Each port adds exactly four executable lines:

- One `--initial-cache-refresh` BooleanOptionalAction, defaultFalse; explicit
  OFF is `--no-initial-cache-refresh`. No environment override.
- At `step == 0` when ON, call the existing `refresh_weight_cache` once for
  each existing cached Linear, in normal grad mode, after the existing `t0`
  and before the first training forward.

Removing those four lines reconstructs its pinned base byte-for-byte. All
existing arithmetic, post-update refresh, optimizer/ownership/state, data,
compiler flags, timer reads/resets/charging, checkpoint and evaluator code is
untouched. The flag is training-only, not a checkpoint model setting. It does
not affect the separate pre-existing `--compile-only` path; that mode is not
part of either proposal and must not be used as external prewarming.

## Source-supported mechanism and prior trials

Exact93ae `Linear` starts `_cached_weight=None` (1480), selects internal
`weight.to(x.dtype)` on its first forward (1487–1490), then the training loop
refreshes the differentiable BF16 casts after its optimizer update (4092–4093).
The MLP adapter makes the same cached-versus-cast choice (4536–4540).
`USE_WEIGHT_CACHE=True` is active. On6ef7, `Linear` is AST-identical
(1444–1474), initial list construction is3916 and post-update refresh4039.
Its MLP installation and training/ownership adapters are also AST-identical.
The only Muon class arithmetic difference is the already-adopted Recursive
beta2 runtime-tensor selection; this patch does not touch it.

This is a concrete first-to-second-forward path change. It does not prove
the cache transition caused the observed native compilation, nor that
priming removes it. Lazy Adam/Muon state initialization and rebinding of
compiled Muon outputs are weaker alternate specialization hypotheses.
Changing schedule values already use persistent device scalars; no Python
float-to-tensor conversion or optimizer initialization change is proposed.

Search covered the decision log, experiment ledger/results, workflow notes,
and Git history for weight-cache/refresh/initialization/priming/first-forward
aliases. No prior initial-cache-priming experiment was found. T1.10
(`research/trn2-experiment-results.md:2013`, commit2f32abc2) instead measured
per-step batched refresh:2.65→1.20ms,0.17% of the old whole step, not adopted.
Only hoisting the module lookup was adopted. This is not that old proposal.
The old116188 smoke's480.53/359.84s first steps support a shared startup
pattern, but that receipt explicitly failed true-cold attribution.

## Native graph inventory: observed, not a causal proof

`graph-inventory.json` pins every recorded graph ID/name/node count/timestamp
and reported compiler time on every rank to archive
`c2c47f7dcc1717eb3d176b1bb01622efe891c09f`. Source is exact93ae, no tables,
RecursiveOFF/FP8OFF. Two model forward/backward specializations occur on
all four ranks. Initial graph node counts are704/1217; later666/1169.

| Rank | Recorded graphs | Initial forward/backward reported compile seconds | Other194-node inference graphs: reported seconds | Later forward/backward reported seconds |
| --- | ---: | --- | --- | --- |
| 0 | 7 | 205.380 /521.889 | 5.404,5.099,5.746 | 0 /0 |
| 1 | 8 | 0 /0 | 3.128,5.283,0,6.052 | 202.538 /0 |
| 2 | 7 | 0 /0 | 0,0,0 | 0 /0 |
| 3 | 7 | 0 /0 | 5.577,8.182,0 | 0 /465.466 |

Zero is an unpopulated/unknown timing, NOT a cache-hit certificate. In
particular rank1 and rank3 report real compilation for the later model
graphs even though rank0 reports zero. Do not sum cross-rank times as wall
time or call recorded FX graph count the count of all eager NEFFs/internal
compiler partitions. The inference records are consistent with compiled
optimizer regions; their exact call sites were not captured by this table.

First forward timestamp is17:42:59UTC; later forward17:56:04; later backward
17:59:40–43. The original step durations are847.90/695.40/.49/.49/.49s.
The live input captured before step0 had740 HLO instructions and explicitly
triggered partitioning (3.3866T MACs vs200G threshold). During step1 another
compiler remained active. Warm directory reuse is not true-cold provenance.
No6ef7 graph_stats CSV exists among the archived `research/` paths searched;
a graph-by-graph6x/8x comparison is unavailable, not inferred from durations.

## Bounded CPU evidence

Existing G2 driver is reused byte-unchanged, with `g2_observe.py` adding only
test-side observations. Eight sequential jobs completed on Torch2.13.0+cpu,
one OMP/MKL/OpenBLAS thread per worker; no concurrent tensor job or full-size
laptop model. Both profiles use4 ranks and3 real optimizer steps on exact
source bytes, seeded synthetic inputs and the existing deterministic clock.

| Test profile | Base/default/explicitOFF/ON durations | Live Parameters | Initial caches | Checkpoint SHA-256, identical in all four arms |
| --- | --- | ---: | ---: | --- |
| 6x128 RecursiveON | 9.1/8.7/8.5/9.1s | 62 | 40 | `1bc4752a2f82da10dd944c1a58bcdc2859ccb4bedea59222d3e77bb41efe098c` |
| 8x128 U-Net/partial attention, RecursiveOFF | 9.5/9.5/9.8/10.3s | 65 | 38 | `01ce0f7b8a9ecd76db109fc2510af1efdb058d47ac388a54f319e705bb2617be` |

For each rank and each step, receipts include full SHA256/shape/dtype records
for EVERY named Parameter's raw gradient before clipping, reduced/clipped
gradient, parameter after update and all serialized owned optimizer state.
Every field is identical across its four arms, including missing gradients
(none on first step). Three per-rank losses and optimizer ownership plans
match exactly. Rank0 checkpoint payloads AND actual serialized file hashes
match; unchanged `load_for_eval` strict-loads and produces deterministic
finite logits. Caches are absent from checkpoint state keys.

Before the first ON forward, every cached value equals the direct BF16 cast
of its FP32 master; `autograd.grad(cache.sum(), master, retain_graph=True)`
returns all ones without modifying `Parameter.grad`. All refreshes observe
grad mode enabled. Default/OFF first caches are allNone; later caches match
the control and remain fresh. ON refresh count is4x cache count vs3x for
default/OFF. This checks gradient connectivity rather than assuming it.

Actual synthetic micro-inputs are2x128=256 tokens/rank, enough to enter the
real custom MLP autograd path at width128. Each loss graph contains6 or8
`FusedMLPBackward` nodes and the unchanged main loop executes backward.
The historical receipt field `mlp_custom_backward_calls` counts these nodes
across forward graphs (18/24), not a separately instrumented native callback.
No native backend/NKI CE or real production-token update is represented.

CPU-only shims: BF16 compute on CPU, no compile, synthetic data/tokenizer,
existing stub prepare, import-only rejecting PyArrow stub, fake clock,
Neuron-only CE install bypass, and G2's micro-input slice. The stub excludes
two startup boundaries rather than real prepare's five; all comparisons
share it, but this is not an official timing test. Native/full-size gates
must use real prepare and no test_support path. Architecture tiny tests use
existing `--depth 8 --n-embd 128 --unet-skip --partial-attn-8` because the
real `--unet-8x1024` preset forcibly allocates width1024. Production stays
8x1024 with the exact original preset/attention layers.

Five source/receipt tests PASS normally and under `-O` for both profiles.
All eight tensor runs passed on their first attempts. The initial static
test run intentionally skipped the not-yet-created tensor receipts; no
failure is omitted. The support-only parser/observation code never ships in
the candidate. Commands and pending native/full-size commands are in RUNBOOK.md.

## Interpretation and remaining gates

The CPU result proves tiny fixed-clock arithmetic/gradient/update identity,
not Neuron equivalence or compile savings. A shorter startup changes the
actual elapsed values seen by the existing wall-clock LR/WD schedule; do not
promise native first50-step identity. Do not alter that schedule or move
casts outside official accounting to conceal this consequence.

Independent review and exact-full-size CPU G2 must pass before native use.
Keep source/argv/control binding, unique proposals, full chip, seed42,
declared1800 budget, checkpoint/exit checks, actual startup/capped allowance/
charged excess and fresh public20971520/context1024. Native graph/startup
evidence must demonstrate the intended effect; one versus two MODEL
specializations is a hypothesis, not a total-graph promise. Warm quality is
research evidence only. Cold measurement, unchanged BPB selection/replication
rules and human-only submission remain separate gates. Newest Supervisor
priority is architecture compile-cut;6ef7 is separately prepared, not queued.
