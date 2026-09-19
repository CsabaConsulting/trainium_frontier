# Fresh-start campaign — 2026-09-18

**Current result, 2026-09-19 09:51 UTC:** best matched public BPB is
**0.9928513116311253**, source98 dense 6x1024 with pack32/GA1, no ngrams and
seed42. The frozen standalone source129 package reproduces those checkpoint
logits exactly; an independent cold 30-minute rehearsal scored 0.9938556748843932
and exited within 1790.34 charged seconds. The 245-character launch is pushed
in `b857d13`. Private leaderboard performance remains unknown.

Source137's complete isolated FP8 MLP is 3.006% faster than its native component
control after exact dual-cache preparation. Real-Linear integration passed.
Source140's complete model instead takes 567.095 ms/step versus native 488.702 ms;
all four numerical checks pass, but its disk-reserve failure is retained. The
caller profiles now locate most additional time in forward and confirm unused
BF16 saved activations are still written. The compact-output version passes
all11 numerical cases and real-Linear cache/master/restore gates. Its nearby
same-host comparison is clean: native488.702ms, legacy FP8568.354ms and compact
FP8562.902ms. That small gain leaves15.18% of full-step regression. Source147
streamed-weight correctness gates are now active on trn22. Ordinary BF16 Linear
tiles shrink around FP8 in the observed profile; weight lifetime and output
layout are separate follow-up hypotheses, not proven compiler causes.
No FP8 speed or BPB promotion. Dropped profile notifications prohibit exact
traffic/bandwidth claims; raw captures and parquet are preserved locally.

Both hosts recovered disk reserve by archiving old compiler diagnostics locally
and on GitHub, verifying every file before remote retirement. Shared caches
remain intact. Width1152 native MLP and exact measured optimizer-ownership
gates passed, followed by a clean587.592ms whole-model phase. Its single seed42
30-minute quality trial is active on trn21, with matched public evaluation still
required. The kernel queue, capacity revisit
and `RESUME.md` record the executable next steps and exact job identities.
The competitive target remains about 0.9749; no seed sweep is planned.
Serial QKV execution with unchanged Parameters/Muon states and contiguous
RoPE inputs is queued separately from the older negative architecture trials.

**Earlier result, 2026-09-19 00:21 UTC:** matched larger public BPB is
**0.9938106631483421**, improved from0.9973302213667847. The dense6x1024 model
uses no n-grams and seed42, T1024/pack16/GA2, padded exact optimizer ownership,
native-forward/custom-backward MLP and exact adaptive two-core CE/target gather.
It completed3,548 updates /465,043,456 targets in1,784.9 training seconds; charged
estimate1,785.3 is within1,800. The matched evaluation uses20,971,520 tokens,
83,462,186 bytes and57,493,516 nats, world1/B1/T1024/O1 without global autocast;
688.38 seconds, clean. This is a public result, not a private leaderboard score.
Checkpoint local and GitHub-digest-verified. Exact source/checkpoint/evaluation
identity is in `fresh-start-best-public.json`. Working target remains0.9749.

The CE change passed exact compiled-autograd gates and improved a same-source
whole step500.372595ms versus519.395419ms (3.6625%). Quality combines CE with
shorter context/packing and optimizer ownership; its gain is not attributed to
CE alone. The previous larger results0.9973302214 (T2048/pack8/ownership) and
0.9974687137 (T1024/pack16/original fused AdamW) remain reproducible checkpoints.
No seed sweep or new function approximation is in the winning run.

FP8 token-row and fixed-input-scale variants pass declared forward/STE backward
and master-accumulation gates, but still lose complete component timing. Fixed
input gives17.926315ms versus18.019795ms adaptive token rows (0.519% gain),
15.672957ms native BF16 and14.629534ms NKI BF16 cache. The trace exposes an
additional compiler-generated transpose and reported spills around the token-row
kernel; a matching backward orientation is the next bounded experiment.
A separate BF16 head/loss memory-boundary probe keeps FP32 arithmetic and
requires exactly matching loss/gradients. Full FP8 backward remains unimplemented.
Compact full-value tables passed accuracy but lost19.04x; stronger rsqrt
corrections and standalone exponential polynomials also lost timing. Reciprocal
refinement passed optimizer integration but is absent from the quality run.

Current requests and recovery steps are in `RESUME.md` and the kernel queue.

**Earlier status, 21:59 UTC.** Minimize official30-minute BPB on a full Trainium2
chip. Working competitive target approximately0.9749; best larger public check
is **1.0010172426032236**, 20,971,520tokens/B1/T1024. The private dataset differs.
Dense6×1024 without hashed n-grams now completes3,183updates/417.2Mtargets using
NKI causal convolution, BF16 normalized branches, native-forward/custom-backward
MLP and pack8. Its small public gain over the3,005-update reference is measured;
its checkpoint is local and cloud-digest-verified. No seed sweep.

Every device job uses immutable source, a host-wide lock, bounded command,
owned-process cleanup and automatic pre/postflight. Source and logs are pushed
through a separate-index recovery path; all31 heavy artifacts have verified
GitHub draft-release receipts. Source5dcbf6a and its recovery snapshot were
verified at21:50. See RESUME.md for exact jobs and newer backup receipts.

Earlier priorities (completion states superseded by the queue above):

1. Complete both active quality runs and preserve their checkpoints. Exact
   padded ownership saves5.06245% of the matched whole step; its T2048/pack8
   run is on trn21. T1024/pack16 plus original fused AdamW saves3.24837%; its
   run is on trn22. Both keep seed42,131072targets/update and clip-after-reduce.
   Evaluate promising outcomes using the same larger public configuration.
2. Gate the mode2 reciprocal refinement through repeated optimizer updates,
   schedules, BF16 caches and restore. It passes34 component checks and is
   16.77% faster than original fused AdamW in a paired test; it has no whole-step
   claim. Diagnose the failed scaled-rsqrt variant by exposing its seed and
   correction stages. Do not relax the original numerical bounds.
3. Test exact loss work elimination: explicit two-core row partitioning and
   SBUF target gather, keeping current primitives, reduction width and backward
   math. Require exact current-kernel agreement, real-shape timing, then complete
   compiled-model gates. Follow up composed approximations only with measured
   costs and an explicit forward/backward accuracy contract.
4. Stop advancing isolated kernels that fail complete-step gates. Degree5/7
   exp polynomials pass sampled accuracy but cost3.57–4.19times the built-in exp.
   The packed/
   streaming MLP variants, standalone normalization and embedding paths retain
   useful diagnostics but have no new quality claim. FP8 remains a moonshot
   requiring a complete forward/backward island with layout/cast costs charged.

Require numerical gates and at least3% complete-step improvement before a
30-minute quality comparison. Evaluate with the unchanged larger public scorer
at identical geometry. Private validation is inaccessible. Kernel work that
changes rounding must disclose and test it, not relabel a failed comparison.

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
