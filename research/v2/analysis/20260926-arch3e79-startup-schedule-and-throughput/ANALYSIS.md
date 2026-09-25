# Architecture3e79: startup-schedule confounder and one residual throughput hypothesis

Researcher/Analyst only, 26 September 2026 IST. No training-source edit, tensor
allocation, provider call, host operation, enqueue or launch. Isolated note base:
`18422b4c`. Existing K2, CT, Recursive and clean-submission packets remain frozen.
The Supervisor's later request to explain the cross-host trajectory takes priority
over generic throughput suggestions. This note neither changes a verdict nor grants
implementation authority.

## 1. Priority finding: the early optimizer schedules demonstrably differ

Both retained logs authenticate the same train SHA
`3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`,
prepare SHA `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
seed42, LR0.015, batch131072, pack32, T1024, world4/LNC2, eight layers,
113256724 parameters and the same attention/UNet/cache flags. The archived
launch environments both use optlevel1/BF16 matmult auto-cast. Their cache paths
and source directories differ; no claim of identical compiler products follows.

The following are exact **printed, rounded, smoothed** losses and LR multipliers,
not raw minibatch losses or parameter checksums. Rows84–90 of each log:

| Step | trn22 loss | trn21 loss | trn22 lrm | trn21 lrm |
|---|---:|---:|---:|---:|
| 0 | 9.0101 | 9.0101 | 1.000 | 1.000 |
| 1 | 8.7779 | 8.7779 | 0.760 | 0.767 |
| 2 | 8.4665 | 8.4646 | 0.760 | 0.767 |
| 3 | 8.1275 | 8.1226 | 0.760 | 0.766 |
| 4 | 8.2179 | 8.2456 | 0.759 | 0.766 |
| 10 | 6.9511 | 6.9945 | 1.000 | 1.000 |
| 20 | 6.0954 | 6.1104 | 1.000 | 1.000 |

Step0 takes790.37s on trn22 and779.52s on trn21. Step1's printed multiplier
differs **before** the first visible printed loss difference at step2. Later
matched-step loss differences trn21−trn22 are +0.0137/+0.0195/+0.0183/+0.0187
at500/1000/2000/3000; epoch transitions also match at the sampled points.
These numbers establish a schedule confounder, not its contribution to final BPB.

All train line references here refer to immutable Git blob
`b4d83b5db66971f3c348837f04f53cba65aa6d33:workflow/proposals/research-v2/train.py`:

- `_PROCESS_STARTED` is sampled at19. The separate loop/budget clocks start
  at3993–3994. Each step samples `elapsed` **before** its forward at4012.
- `lr_multiplier`3302–3314 uses this elapsed/1800, warmdown0.75 and floor0.05
  (constants197–198); step warmup is zero at188. Once elapsed exceeds450s,
  the old schedule is already cooling down.
- At4070–4080 that pre-step elapsed controls the LR and Muon weight decay,
  momentum/beta2 progress inputs. The loss printed for step1 was calculated
  before step1's differing update. WD is also affected even though not printed.
- Only after the fifth completed update,4104–4111, is budget origin reset using
  `excess=max(0, process_age + 30s - 900s)`. This restores subsequent lrm near1.
  The30s source margin is at174; `prepare.py:40` fixes five excluded steps.
- Consequently the first four post-step0 updates see excluded compile time as
  optimizer progress; the later schedule does not. Both recorded startups are
  below the exclusion cap, so the mismatch is not explained by charged excess.

This was anticipated as a possibility in
`research/v2/lanes/ct-base-cache-research/PROPOSAL.md:137`, but the two actual
architecture logs now demonstrate different early LR values. Same seed is not
the same update sequence. Do not silently call these deterministic replications.

### Initialization/data audit and limits

`init_runtime:511` seeds Torch with the supplied42, with no rank/host offset.
Construction/init occur at3898–3901 before `.to(device)`3907. The same explicit
initializer at2006–2057 initializes all active matrices; training-adapter install
asserts RNG preservation4928–4933. Logs show the same parameter count, model
configuration and owner inventory. This supports the intended same initialization
procedure, **not** observed byte equality of actual initial weights/RNG states.

Neither log enables epoch shuffle; ordinary `make_dataloader` is selected3951.
Unchanged prepare409–441 uses the ordered public shard list and deterministic
rank-strided row groups, and468–519 performs the same best-fit packing. There is
no source-level host-specific shuffle. This audit did not read training data or
authenticate cross-host data/tokenizer bytes or batch digests; those remain
unknown. Rounded equal initial losses are not proof of identical inputs/weights.
Native reductions, compiler products and later real-time schedule differences
are additional unquantified sources of trajectory variation.

The Supervisor now reports trn21 public20M0.9931291951444726 versus trn22
0.9863403052402583 (spread0.0067888899042143); the new raw eval archive was not
in this note's frozen base and was not independently validated here. This note
does **not** attribute that spread to startup LR, erase either result, establish
an architecture win, or authorize stopping anything. Final BPB and the original
same-host controls still govern adoption.

### One isolated CPU-first proposal, not an implementation

The excluded portion of startup should not advance an optimizer schedule meant
to track the charged budget. Correcting that inconsistency is nevertheless a
**new scientific schedule factor**, not bit-equivalent cleanup or proof of a
better model. Proposed default-OFF flag: `--startup-adjusted-schedule`, on exact
old-schedule3e79 only, not stacked with Recursive in this diagnostic.

Use one optimizer-schedule elapsed value. Before the fifth boundary, derive it
from the same sampled monotonic instant as existing elapsed:
`max(0, sampled_now - _PROCESS_STARTED + existing_30s_margin - existing_900s_cap)`.
After that boundary, use **exact existing `elapsed`**. This provisionally removes
only the capped allowance; it must not blindly force progress to zero when a
startup exceeds the cap. Feed this shared value to LR and `wd_progress` and hence
the existing optimizer progress consumers. Step-number momentum/warmup stays
unchanged. The selected profile has batch-ramp/EMA/dynamic loss scaling OFF;
their machinery, data order, accumulation and numerical optimizer formulas are
not part of this proposal. A broader scheduler abstraction is unnecessary.

Keep byte-unchanged: actual budget origin/reset and excess calculation,
five recorded boundaries, process/startup accounting, initial cache placement,
1800s declared limit, wallclock stop/predictor,15s checkpoint reserve, control
collectives, checkpoint/exit and unchanged harness/prepare. No prewarming or
work moved outside accounting. This is not permission to alter official timing.

Fail-fast CPU evidence, if separately selected:

1. Stdlib exact-function replay with fake sampled clocks must reproduce current
   lrm divergence for the two observed startup durations and show matching ON
   optimizer scalar traces for matched subsequent clocks below the cap.
   Test cap/margin boundaries, startup>900, step4→5 reset and warmup/cooldown
   boundaries. Excess must still advance the schedule after the allowance.
2. Original/default/OFF three-step losses and every update/optimizer/checkpoint
   digest remain bit-identical. ON is intentionally **not** equivalent to OFF.
   Use a small actual-model multi-step CPU comparison between two synthetic
   startup histories, identical synthetic inputs and post-reset clock increments;
   test all gradients/missing-gradient status and state. Explicitly disclose the
   fixed-clock shims; no full model on this laptop.
3. Replay the same clocks through ON/OFF accounting: recorded boundary times,
   charged-through-exit calculation, stop decision and checkpoint reserve must
   remain identical for identical simulated durations. Assert the patch has no
   reach into prepare/evaluator/data or compiler flags. Independent/fresh source
   review and exact full-size G2 still precede any Operator assignment.
4. Only then propose a short unchanged-shape native diagnostic with finite/update
   receipts and measured first-five/cold graph cost. Fail on any correctness,
   accounting or unexpected graph-count regression. No automatic full run.
   A later same-host fixed1800/public20M comparison would test quality, not a
   promised speedup; a single outcome cannot establish the old spread's cause.

This patch removes one known startup-clock sensitivity below the cap. It cannot
make later elapsed-based schedules identical across hosts with different step
times, nor promise native first50 identity. Expected device-speed gain: none
claimed. Expected BPB effect and fraction of the observed spread explained: unknown.

## 2. Lower-priority throughput research: paired pointwise MLP-backward outputs

No strong measured >=3% or >=5% untried throughput winner emerged from this audit.
One concrete new variant is worth retaining **only as a bounded falsifier after
K2/CT**, not as a native slot or implementation assignment.

The actual3e79 training installation selects `forward_backend='native'`4929;
eligible training enters `make_forward`4546–4561 and its custom backward, not the
flag-controlled original `MLP.forward`. At4472–4474 it computes three BF16
pointwise expressions around an unchanged GEMM:
`activated_t=relu.square()`, `twice_relu=relu*2`, `dpre_t=dhidden_t*twice_relu`.
All four GEMMs, saved tensors and transposed layout are explicit4471–4480.

Proposed single factor: one small two-output elementwise helper consumes
`(saved_relu, dhidden_t)` and returns `(activated_t, dpre_t)` in the existing H×T
layout, with the **same intermediate BF16 roundings**, signed-zero/nonfinite
behavior and outputs. Keep forward, saved tensors, four backward GEMMs,
FP32-master gradients, ownership and checkpoint layout unchanged. No FP8,
recompute, fused-GEMM replacement, gradient approximation or activation change.
This is not turning on the inactive `--nki-relu2` flag.

At32768 tokens ×4096 hidden units a BF16 tensor is256MiB. If all three native
pointwise expressions independently materialized in HBM, their logical reads+
writes would be7tensor-volumes versus4for the joint helper, a conditional
768MiB/layer or6GiB/rank/update difference across8layers. **This is only source
traffic algebra**, not measured traffic or an achievable bound: existing compile
may already fuse them, preserve a different physical layout or overlap them with
GEMMs. Explicit helper boundaries can add traffic/spills and make the graph slower.

Current P is exact6ef7, not3e79. Its runtime-observed forward137.1–141.9ms and
backward272.3–281.9ms are whole-graph intervals. They expose no per-pointwise or
embedding active cost. They cannot support a milliseconds/throughput prediction
for this helper. The prior Source197 library-placement study actually observed
193.388ms added forward time and16.344GB extra spill reloads from a boundary
change; treat that as a serious whole-graph risk, not a component-only model.

Novelty audit: no matching two-output backward-only helper found in current lane
notes, `research/experiment-log.jsonl`, `research/trn2-experiment-results.md`, or
Git subjects searched for pointwise/epilogue/dual-output. This is bounded negative
search evidence, not proof no historical file exists. Related d595ce0e tried
reusing library forward activation transposes; its documented bulk-transpose
component was15.992ms vs15.252ms native. That is not this proposed backward-only
helper, and neither result predicts its gain. K3 active-path audit1bcf203c already
warns that simply adding the activation flag is a no-op in eligible training.

Smallest falsifier, only after selection: exact BF16 pointwise oracle including
rounding edges/zero/NaN/Inf, adversarial upstream values, retained full custom-VJP
checks; default-OFF identity; then native **compiled enclosing MLP** with all
gradients/nonzero checks, not just eager/component output. If correct, measure
whole-step OFF100/ON100 with20warm+80measured on exact3e79, unchanged science,
matched conditions and cold graph cost. Require the existing K3 >=3% gate; a
component win, CPU speed or omitted work is not acceptance. Stop on correctness,
compile/cold-gate failure or failure to clear the timing threshold. No predicted
gain is assigned; if decoded evidence shows the pointwise work already fused or
too small, discard this hypothesis before implementation.

## Duplicate and low-value directions explicitly excluded

- K1 batching/whole-parameter ownership already exists. P shows three compiled
  Muon group calls/rank; its6.8–8.2ms full writeback intervals do not justify
  the already-proposed quarter-copy omission's >=5%whole-step gate.
- Batch262144 already missed its prospective5%screen at4.9977495155%; no rounding,
  retry or LR compensation proposed. Tables/model enlargement are out of scope.
- QKV norm CSE already has a flag and old proposal (Decision Log Sep24;
  results4366): exact3e79 could remove ten repeated norms at five attention sites,
  but it is not a genuinely new implementation direction. Do not rebrand it.
- Old flat-gradient bucket hangs on native hardware and changes clipping
  reduction order (`trn2-experiment-results.md:1648–1686,1814–1850`); not a clean
  untried collective optimization. Clip-foreach was already neutral.
- P's275ms fetch annotation mainly waits for pending backward; it is not evidence
  for a275ms loader optimization. Exact3e79 has just WTE plus layer7 VE, not the
  four ordinary embedding tables in6ef7; do not transfer K2 costs without checking.

## Evidence pins and reproducibility

Paths are repository-relative. No checkpoint/data contents were opened.

- `research/v2/operator/20260926-trn21-arch3e79-replicate/completed-training/training.log`:
  SHA256 `98a201bd36f17f03cf791dc39040873c9dc50d9538cabb01c696c015f11a8ff7`.
- `research/v2/operator/20260926-trn22-arch3e79-full/completed-training/training.log`:
  SHA256 `19a0b5a906c04cd799d7393fbe51e6f6dcd23a937be30226b12727af1360016e`.
- `research/v2/analysis/20260926-p-runtime-interpretation/findings.json`:
  SHA256 `ffead6b1397dcfc60eb0830f0e0c1f3ef4a581e01a8ecccc7d7a00eb94751f62`.
- `research/v2/analysis/20260926-k1-writeback-runtime/result.json`:
  SHA256 `b1c17393a573f3945d233514d88399da6d0e24e68958fcdd4c798c61783a190b`.
- Prior caller risk: `research/agentic/SOURCE197_OFFLINE_ATTRIBUTION.md`;
  actual mechanisms: `research/v2/lanes/k1-muon-batching/AUDIT.md` and
  `research/v2/analysis/20260926-k3-active-path/AUDIT.md`.

Reproduce with `git show b4d83b5d:workflow/proposals/research-v2/train.py` plus
numbered line ranges above, the literal `step` rows in the two pinned logs, and
the archived P/K1 analysis. No import of the candidate is necessary. Only this
note is added; `git diff --check` is the applicable local check, not CPU/native
correctness evidence.

Decision/time: 26September03:54IST (25September22:24UTC), retain the startup-clock diagnostic as
priority research and the pointwise helper as an unselected, measurement-dependent
idea. Reason: a concrete schedule mismatch is demonstrated; a device bottleneck
is not. Undo: discard this note/proposal; runtime behavior was never changed.
