# Research handoff audit — 2026-09-21

This document converts the latest repository evidence into a bounded starting queue. It does not
authorize a device run, AWS mutation, merge, or leaderboard submission.

## Source of truth

- Local `main` is stale at `7f9aaaf5` and must not be treated as the latest experiment history.
- Latest completed research handoff: `origin/research/fresh-start-20260918` at
  `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`.
- `origin/backup/fresh-start-20260918` is a later recovery snapshot of that state, not a replacement
  research line.
- Primary handoff files are `research/RESUME.md` and
  `research/HANDOFF-CLAUDE-2026-09-20.md` on the authoritative branch.

Before editing research code, create a separate clean worktree from the authoritative branch. Keep
the present `main` checkout intact because it contains the local agent-control files.

## What `train.py` means

The competition accepts one `train.py`. The canonical research `train.py` contains switches and
experimental paths; it is not automatically the file to submit. The current frozen candidate is:

`submissions/dense6x1024_public0992851/train.py`

It is self-contained because the selected helper code was vendored into that single file and then
replayed. Never copy the exploratory canonical file to the submission page without rebuilding and
revalidating the one-file contract.

The frozen launch configuration is:

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

## Ground truth at handoff

- Best preserved comparable public BPB: `0.9928513116311253`.
- Independent cold replay: `0.9938556748843932`.
- Frozen model: dense 6-layer, width 1024, sequence 1024, pack factor 32, seed 42.
- Official accepted leaderboard score visible in the supplied screenshot: `1.0246`.
- The mapping from that official submission to a commit/config is still missing.
- The reason a later submission was rejected is still missing.
- Handoff said both Trn2 hosts were idle at 2026-09-20 21:53 UTC. This is stale historical evidence;
  every future operation needs a fresh read-only inventory and the full experiment protocol.

## Ranked research queue

### 0. Reconcile official results

Record the submission ID, exact one-file hash, commit/configuration, and score for `1.0246`, plus the
complete reviewer reason for the rejected submission. Until this is known, public-eval improvements
cannot be reliably connected to leaderboard behavior.

### 1. Complete source197 profile attribution — analysis only

The matched full-step measurements have a large position effect despite the same number and shape of
custom calls:

| Replacement | Mean step time |
|---|---:|
| Zero-library controls | 499.109 / 498.149 ms |
| First layer only | 496.617 ms |
| Last layer only | 682.661 ms |
| All six layers | 564.512 ms |

The raw first- and last-layer captures already exist and were ingested. Match runtime dispatches to
their NEFFs and identify the ordinary compiled work surrounding each custom call. Instrumented
latency is attribution evidence, not speed evidence. Do not infer exact DMA traffic where profile
notifications were dropped.

Why first: this costs no new 30-minute training run and may prevent implementing a backward path that
cannot integrate efficiently into the compiled model.

### 2. Complete saved ROW-FP8 backward — conditional implementation

Source199 measured the saved forward with all preparation and 436,469,760 bytes of saved state
charged:

| Concurrent ranks | Native BF16 | Saved ROW FP8 | Reduction |
|---|---:|---:|---:|
| 1 | 5.314 ms | 4.812 ms | 9.46% |
| 4 | 5.974 ms | 4.831 ms | 19.14% |
| 1 repeat | 5.321 ms | 4.819 ms | 9.43% |

This is the strongest open systems lead, but it proves only a forward component. The implementation
must follow the explicit scale-axis and straight-through-VJP contract in
`research/fresh-start-library-fusion-next-2026-09-20.md`, including:

- saved `Qx/sx`, BF16 pre-square ReLU, and `Qh/sh`;
- current FP32 master parameters and refreshed forward/backward layouts;
- all four data/weight-gradient products;
- FP32 gradient accumulation;
- independent stagewise numerical references, tiny/zero/outlier and changed-input cases;
- cache lifecycle, causal independence, native evaluation, and checkpoint restoration;
- all preparation, state writes, layout changes, communication, optimizer, and refresh costs.

Promotion gate: a repeatable whole-step improvement of at least 3%. Only after that gate may a
separate human-approved 30-minute quality proposal be created.

### 3. Profile-directed NKI work only

NKI is a tool, not a research direction by itself. Only write a new kernel when an existing trace
identifies a dominant operation and the complete caller boundary can be measured. Prefer changes
that reduce layout conversion, HBM materialization, cache refresh, or boundary overhead. Isolated
kernel speedups do not qualify.

### 4. Capacity or hyperparameter sweeps only after a systems change

Width 1152 scored `0.9938092802993084`, worse than the width-1024 public best despite passing the
budget. Earlier depth increases also lost in their recorded configurations. A broad sweep now would
spend scarce Trn2 time without a new premise. Revisit width/context/packing only when a measured
whole-step improvement changes the capacity-versus-update tradeoff, and change one causal factor at
a time.

### 5. Deep research questions for Bedrock agents

Use agents to answer bounded questions against repository evidence and primary technical sources:

1. How should the ROW-FP8 weight-gradient scale be represented when token rows are the contraction
   axis, and what NKI layout minimizes the required transpose/reduction cost?
2. Which dispatch/NEFF differences explain the source197 first-layer versus last-layer timing gap?
3. Can the complete backward use a mixed policy—FP8 data gradients but BF16 weight gradients—and
   still clear the 3% whole-step gate after every conversion is charged?
4. Which documented NKI primitives in the installed version can implement those layouts without
   unsupported assumptions?

Every research answer must cite repository files and primary documentation, list conflicts with
prior experiments, and end in a falsifiable proposal—not edit or run code automatically.

## Closed or parked directions

Do not repeat these without materially new evidence:

- corrected NKI-attention design: approximately 1.9x slower;
- generic/full-model FP8 composition: whole step roughly 563–568 ms versus 489 ms native;
- BF16 NKI-library MLP integration: isolated gain but complete-model regression;
- Vector-engine square: 2.13% slower than Scalar;
- compiler SBUF placement: about 15.25% slower at full-model level;
- width 1152: matched public BPB worse than width 1024;
- speculative decoding: cannot improve the evaluator's full-vocabulary BPB;
- old count mixer and narrow tokenizer continuation rule: measured gains too small;
- source198: failed harness binding; source199 is the corrected completed experiment;
- source159: explicitly marked not to dispatch in the handoff.

## Controlled agent flow

`Researcher -> proposal -> human selection -> Implementer -> independent Verifier -> human launch approval -> Operator -> Analyst`

The supervisor may keep this queue moving, but no agent may combine proposal, implementation,
verification, and operation authority for the same run. The current next action is source197 offline
analysis, not a training launch.
