# Fresh-start campaign — 2026-09-18

**Current campaign status, 14:47 UTC.** Minimize official30-minute BPB on a
full Trainium2 chip. The working competitive target is around **0.9749**, with
margin for leaderboard movement. Our best larger public check is
**1.001551299**, on20,971,520 tokens/B1/T1024; public/private data differences
prevent an exact leaderboard prediction. Lower BPB is better.

The reference is dense6×1024 without hashed n-grams,117.45M parameters, NKI
causal convolution and BF16 normalization outputs. Clipping after gradient
averaging completed3,005 budgeted updates versus2,977 for the earlier control.
Its larger score improves by0.000515388, with exposure also differing. One
predeclared seed43 repeat is complete and its checkpoint is local/hash-verified:
2,959 updates, small-prefix1.000259@1024 versus seed42's0.996049. Different
hosts and46 fewer updates confound a seed-only attribution. **No seed sweep or
selection is planned.** See the measured-results ledger for exact manifests.

**Enforced experiment protocol:** every new device job uses the host-wide lock,
owned-process cleanup, immutable source/environment manifest, bounded command,
automatic pre/postflight health checks and local log mirror. Failures receive
the same preservation and cleanup as successes. Heavy artifacts have tracked
hash receipts. The current organizer README and protected prepare.py were
freshly read through GitHub and match the local copies exactly.

**Kernel priorities and current evidence:**

1. Ordinary embedding backward: compact3072-row native accumulation plus NKI
   expansion gives only1.09% whole-step improvement (582.43ms vs588.84ms nearby
   same-host control), below the570ms advancement gate. It is not promoted.
   The alternative custom segmented scan is now numerically repaired: explicit
   gathers of512 outputs pass all original/uninstrumented shape/distribution
   tests, including actual training IDs. Tiled FP32 transfers reduce its prepared
   cost from151ms to4.61ms, but native is3.17ms. Next gate: BF16-rounded input
   gradients matching native, FP32 accumulation, followed by profiling if still
   slow. CPU metadata and its transfers must remain charged and shared honestly.
2. Fused BF16 MLP training forward: numerical and backward gates pass. Replacing
   scalar weight DMA and strided activation transfers cut the flawed785ms
   implementation to1.69ms, yet native forward is1.35ms. Cached BF16 weights do
   not repair this gap (1.68ms; complete forward/backward also slower). Inspect
   the newly captured actual-shape profile before choosing another layout. No
   whole-model integration or speed claim yet.
3. Preserve validated production improvements and prioritize genuinely material
   full-chip gains. Component timings omit optimizer/control costs; every
   candidate must pass whole-model finite/causal/batch/repeatability checks and
   measured full-step speed before a30-minute quality comparison. Small public
   prefixes, context lengths and private leaderboard numbers are not interchangeable.

The live hypotheses, failures and advancement decisions are recorded in
[fresh-start-results-2026-09-18.md](fresh-start-results-2026-09-18.md). The broader
[deep research report](fresh-start-deep-research-2026-09-18.md) retains the earlier
architecture, precision, data-exposure and optimizer evidence. The following
sections are historical context and the campaign's original predeclared gates.

The user's13:30 leaderboard snapshot has a tenth-place score of0.9847. That is
0.016851299 below our best larger public score, on different validation data.
The research target is a material, repeatable improvement with margin below the
moving cutoff. Sub-1.0 alone is no longer a useful competitive endpoint.

FP32 normalized activations with BF16 only at SDPA loses the full-budget test:
2,600updates,0.989128/1.004102 small-prefix BPB. Keep the faster BF16-normalized
path. A fixed zero-byte output-class mask gives a paired0.000357046 improvement
on2.1M public tokens; it is a small secondary opportunity, not a gap-closing
result. The early ordinary-embedding gradient prototypes failed native numerical gates;
subsequent debugging and compact-gradient outcomes are summarized above.
No unvalidated kernel enters training.

**Earlier update, 13:15 UTC:** the width896 direct-attention control completed
2,188 updates at **0.987568 public BPB @2048 / 0.999880 @1024**. Its fused
FP32 table-optimizer counterpart completed 2,204 updates at **0.986947 /
0.999735**. Both fit the budget and use four ranks and 2,097,152 evaluation
tokens. Both checkpoints are locally preserved. The small difference does not
establish an optimizer quality win. The fused optimizer checkpoint now scores
**1.004452862** on 20.97M public tokens at context1024, versus **1.009889395**
for the first-wave checkpoint at the identical evaluation geometry. This is a
useful improvement but not a robust sub-1.0 result. Repeatability remains necessary. See the
[measured-results ledger](fresh-start-results-2026-09-18.md).

The dense 6×1024 control **without hashed n-grams** completed 1,805 updates at
**1.012886 BPB @2048 / 1.027777 @1024**, in budget, with its checkpoint locally
preserved. Removing the frozen zero Q/K shift reduced step time from1.417s to
0.986s; the same-host repeat on trn21 measured0.979s. Custom NKI convolution with
BF16 normalized activations now measures **0.588s** through the corrected
integrated entrypoint and passes finite parameters/gradients/logits and exact
causality/repetition checks on all four ranks. Its full1,800s run `fs-w12-dense1024-nki-norm-1800-r1` (source`c61bfca`)
completed2,977updates/390,201,344targets at **0.981710BPB@2048 /0.996689@1024**,
using the same four-rank2.1M-token diagnostic geometry as the dense control.
Estimated charged time1785.3s, causality0, checkpoint locally hash-verified.
The standalone20.97M-token/context1024/O1 check completed at **1.002066687 BPB**,
83,462,186bytes,709.243s. This improves on the similarly evaluated table model
by0.002386175 but remains above1.0. The larger sample, rather than the small
diagnostic's0.996689, is the current dense quality reference.

The NKI-only configuration with original FP32 normalized activations is **not
promoted**: its full run and integrated phase fail with NaNs, localized before
the very first backward/update. The earlier713ms scratch timing lacked the
newer finite-value gate and cannot establish a usable speedup. Independent FP32
and BF16 component gates, including zero weights at B4T2048C1024, both pass.
The uninstrumented one-program compiled forward also fails; the two-program
eager forward and the compiled forward retaining intermediate tensors are finite.
This points to a full compiled-graph interaction, not an isolated arithmetic or
two-program-only failure. Retaining tensors changes lifetimes and compilation;
it is diagnostic evidence, not a production fix. No scored FP32 run is promoted.
The BF16 combination is separately gated and completes the full run with finite
losses; the larger evaluation is required before a robust sub-1.0 claim.

Trn21 passed the **8×1024 dense capacity** gate at the same BF16/NKI policy,
four ranks, and131,072 targets/update: **743.742ms median**, all four updated
finite/causal/batch/repetition gates pass,151,012,368parameters. Its full1,800s
run `fs-w14-dense1024-depth8-nki-norm-1800-r1`, source`e6f6139`, completed
**2,365updates/309,985,280targets**,1785.3s charged, at **0.990702BPB@2048 /
1.004083@1024**, four-rank2.1M-token diagnostics. This is0.007394 worse than
six layers at1024, so six layers remains the reference. The604,075,069-byte
checkpoint is locally hash-verified; a larger eight-layer evaluation is not
prioritized. The predeclared speed gate was <=0.95s median update. This is
not a repeat of the old table-model depth sweep: that implementation paid nearly
2× step time for 33% more layers and used a different activation/layout path.
The old short matched-exposure depth result actually favored depth by ~0.003 BPB;
the throughput cost defeated it. Full-budget BPB is now the decision gate.

A separate CPU research probe examines conservative canonical-BPE continuation
constraints using only the fixed tokenizer and provided training shard0. It
does not change the tokenizer, data loader, trainer, or scorer. A generic GPT-2
mask is not safe to transplant because the regex differs. The native paired
diagnostic `fs-w15-dense1024-canonical-paired-eval-r1`, source`4f0e77a`, passed
all causal/reference gates and scored0.996184154 unmodified versus0.996108324
constrained, identical2.1M-token/world1/B1/context1024 geometry. The **0.0000758305
BPB gain** is small; this conservative mask is deprioritized without a training
integration. This does not close every stronger tokenizer constraint.

The cached-weight FP8 component probe on trn21, source`8704139`, completed all
four numerical gates. At the real MLP input/output projection shapes, native
BF16 matmul measures0.556/0.548ms; FP8 including both operand layouts and casts
measures0.633/1.096ms. Even with pretransposed weights, the second projection
loses. Do not integrate this implementation. Its empirical pair-rounding
reference is not a hardware specification; online scale reduction and backward
are not included, so this is not evidence for an end-to-end FP8 speedup.

The updated dense device capture is complete, with an unprofiled585.183ms
control agreeing with the earlier587.999ms. All63raw trace/binary files
(1,198,602,679bytes) are locally preserved with independently matched SHA256s;
The reader additionally required seven warmed eager scalar/cast binaries,
now separately recovered and hash-verified. Local ingestion was SIGKILLed during
severe memory pressure; trn22 completed processing the trace between device
jobs with a40GiB Go memory target. The current backward graph has only
0.199 GFLOPs of regular FP32 matrix work across its two physical cores; the
old approximately614 GFLOPs attention-backward problem is removed. Ordinary
input/value embeddings now account for587.204 GFLOPs of attributed regular
matrix work. This is a FLOP attribution, not an exclusive elapsed-time share.
It motivates a bounded segmented-gradient component comparison, including
sorting, metadata upload and layouts, before any model integration.
The old FP32 attention
trace cannot identify the current bottleneck after changing activation precision.

Trn22's B8/GA2 phase passes all four updated-model checks but measures576.865ms,
only about1.9% faster. It misses the predeclared <=570ms full-run gate. The
B16/GA1 phase also passes but regresses to614.578ms; close this packing sweep
and retain B4. The T1024/pack8/GA4 phase passes correctness but measures577.148ms,
missing its predeclared558.6ms gate. No shorter-context full run is launched.
The SDPA-only BF16/NKI production arm passes all four gates at678.258ms and is
running its full-budget quality comparison since12:45 on trn22. The
clipping-order full-budget control completes3,005 updates at0.981174/0.996049 BPB,
with1784.9s charged and causality0. This is a modest small-prefix improvement
with28 more updates; the larger evaluation is active on trn21. The checkpoint
is locally preserved. The segmented-gradient device probe passes small cases
but fails the real shape, so it is not integrated. Next isolate its stages and
test a distinct compact-row/native-matmul/unique-NKI-scatter implementation.

### Next decisions after the active dense runs

1. Preserve each completed checkpoint locally with size/SHA receipts. Run the
   unchanged standalone scorer at B1/context1024/20,971,520tokens with O1 and no
   global autocast for a promising model; compare only identical geometry.
2. Complete the clipping-order full-budget control after trace ingestion. Keep
   the conservative canonical-token mask deprioritized following its measured
   0.00007583BPB paired improvement. A broader constraint would be a separate
   hypothesis with its own proof and training-cost requirements.
3. Run the gated **SDPA-only BF16 + NKI convolution** precision arm after CPU
   ingestion completes on trn22. Retaining FP32 normalization costs about15%
   per update; only full-budget BPB can justify that cost. Do not infer quality
   from dtype or the short phase's training loss.
4. Keep B4/T2048/GA4 as the geometry reference. B8/GA2, B16/GA1 and
   B8/T1024/GA4 phases did not meet their full-run speed gates. Further geometry
   tests need a new measured mechanism, not another broad sweep.
5. Test ordinary-embedding backward with stable token sorting and an NKI FP32
   segmented scan. The current graph supplies the motivation; the old width640
   benchmark favored native embedding over indexing and remains a negative
   result for that shape. The new width1024 probe must pass duplicate/zero/
   cancellation/real-training-row references and beat current native backward
   by20% after estimating shared preparation across four tables before an
   integrated phase is justified. All actual integrated preparation will be
   charged. Optimizer time remains~55ms/588ms, limiting its overall upside.

The earlier FP32 device trace identifies approximately614 GFLOPs of attention-backward
matrix work in one profiled microbatch across rank0's two physical cores. BF16
at the SDPA boundary alone also passes updated-model checks and is an independent
candidate. No public evidence establishes which architectures the leaderboard
teams use. N-gram capacity is optional.
The original launch plan below remains as provenance.

The first dense full-run dispatch (`r1`) pre-created the runner's own log and
manifest. Its duplicate-label guard refused to launch training. The failure is
preserved; `r2` routes outer stdout to a separate `.driver.log` and lets the
runner own its official artifacts. No device work from r1 overlapped r2.

## Evidence that changes the next experiments

1. The current public record is reproducible around 1.004–1.006 at ~2,100 steps,
   seed 42. This is already a six-layer, width-640 transformer, augmented with
   large trainable hashed n-gram tables. It is not the old four-layer starter.
2. **Capacity can reach below 1.0:** `tc-sh-896x6-2135.log` records 0.987716 at
   2,137 steps, but charged 2,488.7 seconds, so it is NOT a valid competition win.
   A recovered BF16 compiler speedup must be tested jointly with this capacity.
3. Global BF16 compiler autocast gives ~27–30% more updates but lost quality at
   width 640. That simultaneously changed arithmetic, update count, and dataset
   exposure (including a second pass). The failed LR cuts do not isolate these
   causes. Treat the prior “numerics guilty / throughput dead” conclusion as
   stronger than the evidence supports.
4. The training code clips gradients on each rank **before** averaging. Clipping
   the averaged global gradient is a different, untested algorithm. It is a
   controlled experiment, not an automatic correction to a proven winner.
5. NKI softcap/cross-entropy and RoPE/QK normalization already run by default.
   Existing NKI attention passes numerical and causality checks but is slower;
   enabling every NKI flag is not an optimization strategy. The baseline Muon
   Newton–Schulz GEMMs are already BF16. Find the actual changed matrix products
   before attributing the global compiler-autocast penalty to optimizer precision.
6. Byte-weighting each token loss is not an exact BPB objective. For a fixed
   evaluation corpus, total bytes is a constant denominator; ordinary summed
   token NLL already minimizes the numerator. Do not reopen that flag as a
   fundamental fix without a distinct statistical hypothesis.

Detailed supporting audits are recorded alongside this plan. Retain negative
results and distinguish measured facts, inferred causes, and untested hypotheses.

## First wave

All runs use seed 42, `NEURON_LOGICAL_NC_CONFIG=2`, four workers, all chip cores,
`--optlevel=1`, and a separate immutable checkout. Public eval uses identical
token count/context for comparisons. A shortened diagnostic is not a full-budget
competition score.

| Host | Experiment | Purpose | Decision |
| --- | --- | --- | --- |
| trn21 | width 896, depth 6, BF16 matmult autocast, 1800s | Buy capacity with recovered speed | If competitive, repeat at the same budget and validate through `load_for_eval` |
| trn22 | width 640, BF16 matmult autocast, ~1395s | Approximate the baseline's ~2100 updates/exposure with compressed cooldown | Compare actual updates and LR trace; this is approximate, not exact matched-step evidence |
| Next available | width 640, no global autocast, clip after averaging, 1800s | Test global versus local gradient clipping | CPU distributed math gate, short device smoke, then full-budget comparison |

Locked shared arguments: `--compile-muon --qk-shift --qk-shift-beta 0.0
--qk-shift-freeze --ngram-ve-layers 1,-1 --ngram-ve-table-mult 128 --seed 42`.
BF16 flags: `--optlevel=1 --auto-cast matmult --auto-cast-type bf16`.
Ordinary control flags: `--optlevel=1` (in-code default).
Use an unreachable step cap and the time limit to ensure a completed cooldown.

**Follow-up audit correction:** the attention wrapper was introduced after the
historical record and BF16 comparisons. Both first-wave runs use the newer
disabled wrapper even with custom attention off. Their historical comparisons
therefore change compiler graph boundaries as well as the advertised geometry or
duration. Preserve these runs, but do not use them as a causal matched-source
precision experiment. The next available host first tests the direct-SDPA option
against the same current source. See the code audit and evidence-history appendix.

## Systems experiments after first-wave launch

Measure the **current full model**, not the old phase-timer defaults (which remove
the large n-gram tables). Profile forward, backward, collectives, optimizer and
table refresh separately. Optimize the largest avoidable component, and evaluate
whole-step time including layout changes and compilation.

1. **Attention graph and scoped precision:** current `nki_attention` is always
   Dynamo-disabled, including its ordinary SDPA fallback. Restore direct SDPA
   behind an opt-in flag and measure graph capture, actual native kernel selection,
   throughput and gradients. The installed backend has no per-region compiler
   flag override; unsupported `torch.compile(options=...)` is not a solution.
   Baseline Muon Newton–Schulz already uses BF16. Localize any other precision
   differences at matched inputs/updates before changing arithmetic.
2. **Native Trainium2 FP8 GEMM:** independent of failed global compiler autocast.
   Probe native NKI e4m3/e5m2 casting with explicit scaling, FP32 accumulation,
   supported double-row contraction. Start with MLP forward only; retain BF16
   backward and FP32 optimizer. Include cast/layout/scale overhead in the test.
   Do not assume Trn3 MXFP8/OCP support applies to these Trn2/NKI-0.5 hosts.
3. **Attention redesign:** only after a trace identifies an avoidable bottleneck.
   Existing per-query backward spills dK/dV to HBM; a tiled reduction/SPMD redesign
   needs gradient and future-token perturbation checks before throughput claims.
4. **N-gram sparse update/communication:** these tables dominate parameter count.
   Establish actual time spent in dense zeroing, reductions, optimizer updates and
   BF16 refresh before pursuing sparse or row-sharded training. The final model
   must still load as a replicated, stateless single-device model.

Candidate promotion requires numerical forward/backward checks on representative
and boundary shapes, a device smoke, a same-host timing improvement, and full
1800s quality. Keep exact- and approximate-math changes distinct in interpretation.

## Statistical and architecture followups

- Short-tail averaging of the **current** flat-LR n-gram tables: old EMA negatives
  were primarily a different small dense model with decaying learning rates.
  In-budget averaging and the final saved model must be part of one training run;
  offline averaging is diagnostic only until integrated and timed.
- Couple width/depth, precision, and schedule. Reopening a shape is justified when
  a measured speed improvement changes its feasible token exposure.
- If second-pass behavior is the issue, test training-only sampling/reweighting
  within the organizer's allowed API and explicit sampling permission. Keep
  `prepare.py`, tokenizer, evaluation and provided data unchanged.
- Higher-risk option: allocate capacity to a cheap learned causal memory and fewer
  dense layers, using the current n-gram winner as the control. Measure marginal
  BPB gain per second and per additional table row, not total parameter count.
- Do not seed-hunt, fit to held-out data, or treat a short smoke's final BPB as a
  full-run architecture ranking.

## Operations and preservation

Observed 2026-09-18: both hosts idle, no `/dev/neuron0` holders, >50GB free.
Actual old checkout is `/home/ubuntu/trainium_frontier` (the older runbook's
`/home/ubuntu/workspace/trainium_frontier` path was stale).
Venv is `/home/ubuntu/workspace/native_venv`: torch 2.11.0+cpu, native
torch-neuronx 2.11.3.0.0 dev, neuronx-cc 2.26.6360, NKI 0.5.0.

Both old checkouts contain uncommitted work and different train.py hashes.
Preserve them without overwriting. Local snapshots (excluding heavyweight
checkpoints/cache) are in sibling `trainium_frontier_backups/2026-09-18/trn21`
and `trn22`. Fresh experiments use `/home/ubuntu/trainium_freshstart_20260918`,
cloned from a local git bundle, so GitHub access is unnecessary and source commit
identity is verifiable. Preserve record checkpoints separately as capacity allows.

Runner changes: stream stdout/stderr directly to disk with unbuffered Python;
write launch manifest before starting; capture source hashes, commit, host and
Neuron flags; refuse duplicate labels; hold a run lock; flush final JSONL result.
The old runner buffered the entire run until exit and could lose all logs on crash.
Code is committed locally before transfer. Retrieve live logs repeatedly, final
results and useful checkpoints after completion, and commit local research updates.

Between every device job: verify no workers or device holders remain, check free
disk, source identity and flags; allow only one job per host. A normal process exit
releases Neuron memory. Do not stop/reprovision instances or delete caches to
manufacture a “clean state.” A failure is diagnosed before the next launch.

## Rule anchors

- [Organizer repository](https://github.com/aws-neuron/trainium-frontier): allowed
  architecture/kernel/optimizer/sampling changes; fixed preparation/tokenizer/eval;
  replicated stateless causal `load_for_eval`; provided training assets only.
- [Competition rules](https://trainium-frontier.devpost.com/rules): Round 1 30-minute
  training on the full chip. The trusted `prepare.py` timer excludes the first five
  step boundaries up to 900s; later compilation and checkpoint completion must be
  accounted for. Always retain charged-time evidence, not just steps/s.

## Results and live state

Initial engineering checks: runner regression passes (log visible before exit,
crash output retained, duplicate overwrite refused, typed summary parsing).
Device runs and detailed verdicts will be appended as measured.
