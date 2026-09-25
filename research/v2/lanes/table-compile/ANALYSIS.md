# Table gather compilation boundary: source diagnosis

2026-09-25. Researcher-only proposal, not a patch, launch receipt, or result.
Controlling instruction: `workflow/COMPILE_CONSTRAINT_CORRECTION_2026-09-25.md`.
Priority is (1) gather outside the compiled model, (2) distinct-graph reduction,
(3) compiler flags. No host contact, model edit, SDK edit or cache mutation occurred.

## Finding and selected experiment

The current training entry point compiles the whole GPT. Hashing and advanced
indexing of every n-gram table therefore occur inside its traced forward. Add one
default-OFF training flag, `--ngram-gather-outside-compile`: calculate the same
on-device hashes and gather the raw narrow rows in an uncompiled training caller,
then pass a fixed-order tuple of those tensors to the compiled GPT core. Preserve
autograd links to the original table Parameters; leave each projection inside
the model, and preserve all dense gradient, reduction, clipping and RMSProp work.

This can remove the large table gathers from the model's forward/backward compiler
regions. It does **not** prove that the Neuron eager gather/backward is cheap, that
total graph count falls, or that cross-core resharding disappears. Those are native
measurements. A compiler-boundary change can worsen launches or live memory.

## Exact evidence

Source snapshots read in the canonical Desktop checkout:

- `submissions/research_v2/train.py`: SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
- `workflow/proposals/research-v2/train.py`: SHA-256
  `f9b6a9cbb1a1760fb8db422ffd79ffb91e9280899c40aa86028d935acf355f67`,
  installed at `e273b97c42a2d9cc86124e1c5cbbe2a54c75bab1`. Its Recursive
  schedule is OFF for this proposal. If the independent predictor patch becomes
  the selected base, pin that exact new base for both ON/OFF arms before launch;
  do not silently combine code changes or reuse the old file hash.

Line references below are to the exact c5c snapshot (the f9b6 file has offsets).

| Source | Observation | Consequence |
|---|---|---|
| 3830–3833; 3846; 3947 | `torch.compile(model, backend=..., dynamic=False)` and ordinary model calls in compile-only/training | Whole GPT, not just blocks, is the present compile scope. Both training call sites must use the proposed boundary. |
| 2038–2085 | Hashes are causal int32 device operations over current/previous tokens with fixed per-layer primes | Reuse these operations exactly. Moving hashing to a CPU/data-loader approximation would be a different change. |
| 2087–2115 | `_ngram_lookup`: `weight[rows].to(COMPUTE_DTYPE)`, followed by table-specific projection | Move only the hash/raw-row materialization boundary; preserve the operation spelling, dtype conversion and projection. |
| 2226–2250 | GPT hashes once, looks up `b{i}`/`t{i}` at their layer, then passes them to the block | Optional pre-gathered raw rows must be matched to the exact same keys and used once at the same model sites. |
| 1915–1930 | M64/D128 has 524,288 rows/table; zero-initialized BF16 table Parameters; independent projections | Each table is 128 MiB. Four A2 tables total 512 MiB before gradients/state. No new Parameters are needed. |
| 2167; 2202–2212 | Projection matrices join Muon; tables join their existing RMSProp group | Do not reroute optimizer ownership, table state, bias correction or LR. |
| 3016–3095 | Table clipping has its own group; each gradient is AVG-reduced; clip-after-reduce is selected | Keep parameter enumeration/order and dense gradients unchanged on every rank. |
| 2807–2855; 4591–4612 | RMSProp state/math remains BF16 for the actual arm; tables update identically on all ranks, separate from owned FP32 parameters | Boundary extraction is not sparse/touched-row optimization. |
| 2906–2916 | Muon caches compiled bodies by shape/orientation/NS parameters; tables add projection matrices | A2 can also add/change optimizer specializations. Two slow step durations are not two identifiable graph compilations. |
| 4403–4406; 4435–4445 | Existing MLP helper uses a deliberate disabled wrapper around a separately compiled callable | A local precedent for boundaries, not proof for table gather performance. Do not enable MLP separate-region mode in this arm. |
| 3179–3184; 3208–3240; 4807–4822 | Checkpoint saves original model state; scorer reconstructs frozen stateless model | Keep state keys and eval arithmetic unchanged. Training raw-row inputs must never become persisted or eval-time cached state. |

PyTorch's [compiler-disable API](https://docs.pytorch.org/docs/2.14/generated/torch.compiler.disable.html)
documents disabling compilation of a function and optionally its callees. This
supports the frontend boundary concept only; the locally installed native backend
and its generated forward/backward graphs remain the controlling performance test.

## Retrieved timing facts, with limits

Raw A1 log, evidence commit `121d715aa7d903205df9b3745ff8d2cb7a5e1424`:
`research/v2/operator/20260925-trn21-a1-full-final/training.log`, SHA-256
`b80583b9ae274e67a30e66d3bf69a59d73ab34956554512bcfa889253f7c20e5`.
It records 3,051 steps, `training_seconds=1784.9`, first durations 541.61,
448.56, 0.54, 0.54, 0.54 seconds, and capped startup allowance 900. The five
durations sum to 991.79 seconds; import/model/loader time is additional. They are
step durations dominated by startup, not compiler-phase attribution or a clean
cold-cache measurement. A1's final public-20M was pending at this diagnosis.

Raw A2 evidence commit `cf731b636588119661cdbc0704a6ba4da3b0290f`:
`research/v2/operator/20260925-trn22-conditional-queue-r1/a2-startup-truncated/train.log`
(SHA-256 `5bb7ddda0a609633ddadea8986d3b034c82ed0310e028c3befce8f2ea3ac1675`)
and sibling `rehearsal.json`
(SHA-256 `381e4981953b446021f842f7f2a38ee53772c27d2d2bd9e413a6eb7a297bb930`).
Five steps completed; durations begin 1056.75/903.47 seconds. The independent
harness observes the fifth boundary at 1988.6475348719396 seconds, process wall
1997.294311970938, allowance 900, and charged-through-exit 1097.294311970938.
`fresh_empty_sdk_caches` is empty: do not call this a proven cold experiment.
It is invalid as a full quality trial. Its warm same-c5c rerun is research-only.

The actual `prepare.py` constants (lines 40–41) exclude five startup boundaries
with a **900-second maximum allowance**. c5c separately sets a 30-second launch
margin and charges `max(0, observed_startup + 30 - 900)` after boundary five.
Thus **870 seconds** is the conservative planning target, not a changed cap.
The old predictor retains startup durations in `max(recent_step_times[-20:])`
(4023), explaining the separate A2 five-step stop defect. Clearing that history
does not remove charged startup and is outside this proposal.

## Duplicate/flag audit and alternatives

Searched current c5c/f9b6 source, `research/v2/ledger.jsonl`, old
`research/experiment-log.jsonl`, `research/trn2-experiment-results.md`,
`research/agentic/DECISION_LOG.md`, and the report stop list using gather,
pre-gather/external gather, gather-outside-compile, graph boundary/count,
compile scope, and touched-row aliases. The exact *idea* was already proposed
on September 12 (`research/trn2-experiment-results.md:3854`), but no recorded
experiment implementing this raw-row training boundary was found. It is a
revisited proposal, not a newly invented technique. It needs the new cold-start
hypothesis and current human selection to justify work.

The most important negative is the September 15 **reshard hunt**
(`research/trn2-experiment-results.md:7095–7320`). At the real four-rank
640-wide/context2048 geometry, replacing dynamic gathers with static-address
slices recovered **0.0 +/- 1.6 ms**, across injection sites. The old one-rank
47.6-ms cross-core penalty was therefore falsified as a competition-shape
throughput explanation. Later blocks also closed the apparent single-layer-1
schedule penalty as a property the shipped multiple-injection profile did not pay.
These are strong negative priors for a claimed steady-state resharding fix.
Primary four-rank JSONs are preserved in `research/run-logs/reshard-a/`; e.g.
`a3-l1-dyn.r0.json` records world4, LNC2, width640, context2048 and a 138.1053-ms
forward. This lane does not claim that historical mechanism returned on c5c.
The new falsifiable target is **cold compiler-region complexity at the exact
width1024/context1024/pack32 A2 profile**, where current startup is severe.

- `research/scratch/prof_driver.py` already has `--ngram-hoist`,
  `--ngram-merge-gather`, and `--ngram-gather {index,static}` (lines 66–88).
  Hoist/merge are applied *before* the whole-model compile at lines 415–423;
  they never created the requested eager boundary. Their historical nulls argue
  against Python-only hoisting, merge/resizing and an assumed steady-state win.
  The static arm changes values and is only a timing diagnostic, not valid model
  arithmetic and not evidence that a one-hot replacement is fast.

- `NGRAM_VE_INDEX_GATHER=True` already uses advanced indexing. Its source comment
  cites `research/run-logs/embed-gather-bench.log`: at 262144x128, `weight[idx]`
  was 2.21 ms versus 72.43/72.49 ms for other forward+backward spellings. This
  historical microbenchmark is not an exact-size cold-start result. Retain the
  current spelling; merely switching to advanced indexing is a duplicate.
- `--ngram-ve-dim`, table count/placement and BF16/RMS flags already exist.
  Reducing capacity or changing optimizer dtype would confound this experiment.
- B2 sparse candidate `b2e61508` was rejected before hardware for dynamic sparse
  paths, duplicate reduction and BF16 deferred-decay exactness; see
  `research/agentic/TRACK_B2_SPARSE_CANDIDATE_REJECTION_2026-09-24.md`.
  This boundary proposal does not reopen that candidate.
- Distinct-graph reduction is second: first inventory actual unique graphs,
  compile/cache hits, shapes, ranks and call sites. Reuse one compiled core per
  process and a fixed row tuple; never create a compiled function per table or
  microstep. Only then propose a separate shared-region experiment if remaining
  compile time is dominated by duplicated graphs. Do not change Muon batching
  or layer regions speculatively in this arm.
- Compiler flags are third and unchanged here. Existing `--optlevel=1` must stay.
  `NKI_DUMP_PERSISTENT_CACHE` is not enabled or proposed; there is no archived SDK
  source proof in this lane, and enabling a cache switch is not a gather fix.

## Implementation contract for a separate Implementer

1. Add one default-OFF flag in the research file. OFF must take the existing
   whole-model compiled route with no initialization/RNG/optimizer/state changes.
2. ON supports the exact A1/A2 table configurations with Engram disabled. Fail
   clearly for ON without n-gram tables, with Engram, or with `--no-compile`;
   additional configurations can be separately tested later.
3. Use the original model object and table Parameters. An uncompiled training
   callable computes original hashes and a stable tuple of raw narrow rows,
   then calls a compiled core with `(idx, targets, raw_rows)`. No detach,
   `no_grad`, sparse gradient, CPU copy, manual table-gradient injection, or
   cross-microbatch row cache. Autograd must return gradients to the original
   table through the original advanced-indexing backward.
4. The compiled core must bypass its own hash/gather when rows are supplied, but
   apply exactly the original projection, table gates, attention and loss.
   Ordinary `GPT.forward(idx)` still constructs rows itself. Prefer a shared
   projection helper rather than a copied model-forward implementation.
5. A tensor dictionary created *inside* a compiled caller is not the requested
   boundary. Place the outer call outside `torch.compile`; if a disabled wrapper
   is used, ensure it does not recursively disable the intended inner core.
6. Keep setup_optimizer, clip lists, AVG reductions, replicated RMS updates,
   original-model checkpoint saving and `load_for_eval` unchanged. The flag
   changes training execution only and needs no new checkpoint tensor/state.
7. Gathered BF16 inputs are 8 MiB per table at B32/T1024/D128: A1 16 MiB, A2
   32 MiB. This is HBM materialization, not an SBUF allocation; lifetime/layout
   effects still require measurement. Full dense gradient/state traffic remains.

The complete proposal and cheapest falsifier are in the two sibling files.
No speed, BPB, resharding, submission eligibility or independent-PASS claim is made.
