# Resume the Trainium kernel campaign

Updated 2026-09-20 08:48 UTC. Authorization: danger-full-access, approval never.
Use committed manifests/postflights before dispatch; never restart completed jobs.

## Current work — 2026-09-20 08:48 UTC

- Trn21 RUNNING source196d8461533 at remote wave196. Coordinator13612,
  fs-w196-nkilib-vector-square-sequence-r1, started08:43:43UTC, plan9ddd3fe5,
  max5100s. Nine small/medium cases pass all-six-output equality against the
  matched Scalar square; production compilation/timing pending. New explicit
  Vector square is the only arithmetic-engine change. All194 gates retained,
  plus ten exact comparisons; no backward/BPB claim. Do not restart it.
- Trn22 RUNNING source195282935ca at remote wave195. Coordinator14184,
  fs-w195-nkilib-partial-position-sequence-r2, started08:46:21UTC, plan6b545049,
  max11900s. Runs first-only, last-only, all-six, then zero-library repeat,
  with separate cleanup/ownership each time. First-only model compiling now.
  Original three requests3300s each; final identical zero control1200s.
  The never-launched position-sequence-r1 is SUPERSEDED; DO NOT launch it.
- Source195 routing and initial zero-library control COMPLETE/CLEAN. Four
  selections pass256 native gradient comparisons/20exact gates. Zero model
  loss5.126208782196045 exactly matches the original native control. Mean
 499.107528ms, median488.653169ms. Final sample805.567272ms includes343.200688ms
  optimizer time; every sample is retained. The added final control brackets
  the comparisons because this spike can distort a small apparent speed gain.
- Source194 COMPLETE/CLEAN08:33:51UTC; every original frozen requirement locally
  verified. All10 cases/310exact checks. Saved forward4.771002ms vs5.323111ms
  native HxT (-10.3719%), including online master casts/preparation and
 436,469,760B of actual state saves. Full backward and model speed unmeasured.
- Source193 independently replicated191 no-save gain:3.908544ms vs4.595774ms
  BF16, all preparation charged. Raw/derived profiles verified locally/cloud.
  Large products areFP8. Missing dynamic DMA metadata prevents exact traffic
  claims; source/engine intervals overlap and cannot be added.
- Source192 compiler SBUF full model563.005751ms vs488.528818ms (+15.2451%);
  all gates pass, no promotion. Smaller fixed budgets also lost.195 isolates
  layer-count/position effects with native HxT retained in untouched blocks.
- Full ROW backward is designed, not implemented. Per-token scales cannot be
  factored outside weight-gradient contractions over tokens. Use explicitly
  gated global/channel scaling or a BF16 weight-gradient control; charge all
  producers and transposes. Preserve tiny gradients, FP32 masters/accumulation,
  cache refresh and native evaluation/checkpoint contracts. See fusion-next.
- Trn22 maintenance r4 archived1829old files8.595GB, verified every local hash,
  live GitHub digests and remote contents before removal. Free19.556GB before
 195 resumed. Sources/shared caches/checkpoints/current profiles retained.
- Best public BPB0.9928513116311253, cold replay0.9938556748843932; no new quality
  run this library wave. Frozen train.py/prepare.py/submission unchanged.
  Source159 T512 MUST NOT dispatch. Both active sequences are finite and stop
  on failed gates; no automatic quality promotion or further experiment exists.
- Cron every10min mirrors/pushes source and backup refs; no merging or job
  selection. Active sequences mirror45s. Consult last-success.json for exact
  verified refs. Keep8GiB reserve, immutable sources, Git/host locks and
  independent pre/postflight. Do not edit running plans/requests/helpers.

## Previous checkpoint — 2026-09-20 08:42 UTC

- Trn21 source194 COMPLETE/CLEAN08:33:51UTC, every frozen requirement verified:
  all10 cases and310 exact checks. Saved forward4.771002ms vs5.323111ms native
  HxT (-10.3719%), charging all preparation and436,469,760B of real saved state.
  Backward/model/BPB pending. Host idle; source196 Vector-square comparison is
  implemented and locally AST checked, NOT DISPATCHED. Retain the Scalar arm
  and all original194 gates; only the square engine changes.
- Trn22 source195 RUNNING, remote wave195, immutable282935ca. Coordinator21953,
  fs-w195-nkilib-partial-gate-control-sequence-r1, started08:29:14UTC, max6300s.
  Routing gate COMPLETE/CLEAN: zero/first/last/all-six,256 original native
  coordinate/gradient comparisons and20 exact checks, all independently
  verified. Current full-model zero-library control retains native HxT in
  every layer. Fixed seed42,4ranks/LNC2,pack32,5warmups/30updates.
- After195 gate/control BOTH complete cleanly, explicitly start the already
  committed fs-w195-nkilib-partial-position-sequence-r1. It runs first, last,
  all-six sequentially, each with separate pre/postflight and3300s bound;
  total10800s. Every request re-verifies the gate/control and source bytes.
  This second sequence is PREPARED, NOT LAUNCHED. No automatic watcher exists.
- Trn22 old compiler scratch maintenance r4 COMPLETE/CLEAN before195 launch.
 1829files8,594,799,997B archived, every local hash checked, live GitHub asset
  digests verified, remote contents reverified before retirement. Free space
 10,946,162,688B to19,556,073,472B. Cutoff07:40UTC preserves latest compiles.
  Shared caches, sources, checkpoints and current profiles retained.
- Source193 forward replication: native4.595774ms/preparedFP84.688562ms/
  inlineFP83.908544ms, all online preparation charged. Original19114.6885%
  gain replicated. All trace/NEFF and decoded table identities verified locally
  and in GitHub. No saved-activation/backward or BPB claim from193.
- Source192 complete model563.005751ms vs488.528818ms (+15.2451%), all original
  gates pass. Compiler SBUF allocation does not fix the regression; fixed
  smaller budgets also lost.195 diagnoses layer-count/position causality.
- Full ROW backward still needs its own scale/rounding contract, tiny-gradient
  handling and complete-step tests. Per-token scales cannot be moved outside
  a weight-gradient contraction over tokens; see library-fusion-next.
- Best public BPB0.9928513116311253, cold replay0.9938556748843932. Frozen
  submission/train.py/prepare.py unchanged. Source159 T512 MUST NOT dispatch.
- Consult out/fresh-start-backup/last-success.json for exact verified refs.
  Cron mirrors/pushes, never merges or selects jobs; sequences mirror45s.
  Keep8GiB reserve, finite requests, immutable sources, Git/host locks and
  independent pre/postflight cleanup. Do not edit active plans or helpers.

## Previous checkpoint — 2026-09-20 08:20 UTC

- Trn21 source194 RUNNING from725a385c, remote wave194, coordinator6760,
  sequence fs-w194-nkilib-row-training-save-sequence-r1, started08:16:05UTC.
  All small/medium saved-output gates pass; production compilation pending.
  Plan2b447dfc; max5100s. Never restart a launched request.
- Trn22 idle after192 COMPLETE/CLEAN. Source195 partial-layer routing code
  and full default-phase AST proof prepared locally; NOT DISPATCHED.
- Source191 forward win independently replicated by193 with unchanged kernels:
  native4.595774ms, preparedFP84.688562ms, inlineFP83.908544ms. Online input
  and FP32-master weight preparation included. Source191 original14.6885%
  lower latency survives this separate measurement. No backward/BPB claim.
- Source193 raw12files122,444,398B and derived28files69,259,863B all verified
  against remote SHA256s. GitHub payload/manifests digest-verified08:08:53UTC.
  Local analysis matches all3 dispatches to preserved binaries. Both large
  projections are FP8; Tensor active time falls about4.04ms to2.46ms/core.
  Scalar/Vector work remains; intervals overlap and cannot be added. Missing
  dynamic DMA metadata means no exact traffic/bandwidth conclusions.
- Source192 compiler-managed SBUF fails to recover full-model speed:
 563.005751ms vs488.528818ms native (+15.2451%). All original numerical,
  cache, checkpoint and four-rank postchecks pass. Source190 smaller fixed
  budgets also lost. No whole-training promotion or quality run warranted.
- Source194 will save exact BF16 pre-square ReLU, consumed input/activation
  FP8 bytes and both row scales inside the existing kernel. Original191
  forward controls/gates remain unchanged. Native saved-ReLU HxT control and
  all candidate saves are timed. No raw FP32 up diagnostic store or extra
  SBUF allocation. Full backward still requires a declared approximate VJP.
- Next model diagnosis: zero/first/last/all-six library replacements while
  untouched blocks retain the native HxT custom backward. This has a causal
  premise after the allocator tests. Source195 routing gate precedes timings;
  default adapters/kernel arithmetic unchanged. Plans not dispatched yet.
- Best public BPB0.9928513116311253, cold replay0.9938556748843932. Frozen
  submission/train.py/prepare.py unchanged. Source159 T512 MUST NOT dispatch.
- Latest verified push is recorded in out/fresh-start-backup/last-success.json; consult it
  for exact refs. Manual source2b447dfc/snapshot86b0c2ad push verified08:16:47UTC;
  it explicitly skipped retrieval, while194 mirrors logs every45seconds. Cron mirrors/pushes,
  never merges or selects jobs. Keep8GiB reserve, finite requests, immutable
  sources, Git/host locks, and separate pre/postflight cleanup.

## Previous checkpoint — 2026-09-20 07:59 UTC

- Trn21 RUNNING source193317b99c3 at remote wave193. Sequence
  fs-w193-nkilib-row-forward-profile-sequence-r1, coordinator19207, started
  07:56:08UTC. Plan d70c083d. One same-host three-arm forward capture followed
  by raw-preserving Explorer ingestion. Requires every source191 gate and
  unchanged kernel bytes. All preparation stays inside each compiled call.
  Max6000s. Preserve raw traces/NEFFs and decoded tables locally and in cloud.
- Source191 COMPLETE/CLEAN; every frozen requirement independently verified.
  Ten cases pass. Production forward: inline3.926421ms, nativeBF164.602453ms,
  preparedFP84.687184ms.14.6885% faster thanBF16,16.2307% faster than previous
  ROW control; actual bytes/scales/stages/repetition/changed-input checks pass.
  Includes all online input/weight preparation. No training saves/backward.
  Relative RMS versusBF16 remains5.76445%, so no quality-equivalence claim.
- Trn22 RUNNING source1928b1a3fc0 at remote wave192; coordinator4195, sequence
  fs-w192-nkilib-compiler-sbuf-sequence-r1, started07:42:29UTC. All ten full
  component and both Linear/cache gates COMPLETE/CLEAN, verified locally.
  Complete component14.924422ms vs15.204608ms native. Current native model
  control will be followed by compiler-SBUF library model timing. Max12000s.
- Both source190 fixed-budget sequences COMPLETE/CLEAN;128KiB full step
  562.878526ms vs486.986693ms (+15.584%);64KiB566.748940ms vs488.645588ms
  (+15.984%). Smaller fixed bounds did not fix the model regression.
- Source180 caller/instruction analysis and raw/derived profiles are preserved
  locally and cloud-verified. Ordinary Linear tile work units shrink and
  copies/reloads grow. Missing DMA notifications prohibit exact traffic claims.
- ROW training design is now explicit in library-fusion-next: keep/save actual
  quantized input and activation plus row scales, channel-scaled weights, BF16
  pre-square ReLU, and a separately declared approximate VJP. The older fixed
  input/weight-scale FP8 backward cannot be plugged in unchanged. Charge all
  saves, dual weight layouts, gradient calibration and four backward products.
- Conditional graph diagnosis if192 loses: zero/first/last/all-six replacements,
  keeping the same native HxT custom backward in untouched blocks. This can
  distinguish global graph interference from per-layer cost. Not dispatched.
- Latest verified cron push07:51:05UTC sourcedbb20338, both hosts retrieved;
  later source193 commits and191 result await next push. Consult last-success.
  Cron preserves/pushes, never merges or chooses experiments. Active logs
  mirror every45s. Never edit running plans/requests/protected helpers.
- Best public BPB0.9928513116311253, cold replay0.9938556748843932. Frozen
  submission/train.py/prepare.py unchanged. Source159 T512 MUST NOT run.
  Keep8GiB free reserve, finite bounds, immutable sources, Git/host locks,
  and separate pre/postflight cleanup. No automatic quality run is queued.

## Previous checkpoint — 2026-09-20 07:19 UTC

- Source190 BOTH RUNNING from immutable dad228df, remote wave190 on each host.
  Trn21 sequence fs-w190-nkilib-sbuf128-sequence-r1, coordinator2513, started
  07:11:59UTC. Trn22 fs-w190-nkilib-sbuf64-sequence-r1, coordinator9028, started
  07:13:17UTC. Both completed all10 full-gradient component gates and real
  cached/uncached Linear integration CLEAN, independently verified locally.
  Both now execute native whole-model control, followed by budgeted candidate.
  Do not edit frozen plans, requests or protected launch/sequence/sync helpers.
- Component128KiB14.884966ms vs native15.202572ms;64KiB15.648807ms vs
  15.196156ms. No component-speed requirement for this model-scheduling test.
  Same full-gradient/FP32 accumulation, cache refresh, native eval and exact
  checkpoint gates passed. Model timing remains pending.
- Source191 PREPARING, NOT DISPATCHED: BF16 input quantized directly into
  library FP8 tiles, eliminating input HBM materialization. New scratch is
  allocated through the original manager and charged to its tile planner.
  Original fused-weight/prepared-input control retained; inverse AST proof
  verifies unchanged original fixtures/bounds plus actual input-byte, scale,
  all-intermediate-stage and changed-input checks. Device validation pending.
- Source189 COMPLETE/CLEAN. Exact fused weight preparation29.85%/32.06% faster;
  complete ROW forward4.686795ms vs4.596355msBF16, still1.968% slower. All10
  original gates pass. No saved training activations/backward or quality claim.
- Source180 both raw traces and both derived tables are locally SHA-verified
  and GitHub-digest-verified. Paired device forward137.850536 vs209.325622ms;
  ordinary Linear activity/spill attribution is a major new cost. FP32 matmul
  work falls. Dropped DMA notifications prohibit exact traffic/bandwidth claims.
- Trn22 scratch maintenance r3 COMPLETE/CLEAN07:13UTC:2,381 old files,
  6.448GB preserved, every archived file verified, live GitHub digests checked,
  then retired remotely. Free11.058 to17.524GB. Sources, shared caches,
  checkpoints, profiles and newer compiler diagnostics retained.
- Best public BPB0.9928513116311253, cold replay0.9938556748843932. Frozen
  submission/train.py/prepare.py unchanged. Source159 T512 MUST NOT dispatch.
- Last verified push07:12:48UTC source d2e10195/snapshot615a3943. That manual
  push explicitly skipped remote retrieval; sequences mirror active logs every
  45seconds and cron retrieves/pushes every10minutes. Consult
  out/fresh-start-backup/last-success.json for newer receipts. No auto-merging.
- Use research/fresh_start_sequence_start.py, which rejects uncommitted or
  changed request/plan bytes. Explicit Git/host locks, finite timeouts,
  immutable checkouts and pre/postflight cleanup. Always set -e for dependent
  shell operations. No automatic retry of a failed or ambiguous experiment.

## Previous checkpoint — 2026-09-20 06:25 UTC (superseded host status)

- Trn21 source185 RUNNING, sequence fs-w185-nkilib-row-mlp-parser-sequence-r1,
  coordinator12737, immutable source3d4c8c91, started06:24UTC. Same ten-case
  ROW FP8 forward gate as181/184; only two tuple-attribute assignments split
  into four NKI0.5-compatible statements. AST equivalence proved. No bounds
  or data fixtures changed. No saved training activations/backward yet.
- Source183 byte transpose COMPLETE/CLEAN, all frozen gates independently
  checked. NKI is exact for composed FP8 preparation and all256-byte fixtures,
  including both production weight shapes. Ordinary byte transpose corrupts
  3,391,488/4,194,304 coordinates; source182's float workaround also fails.
  Preserve181/182 failures. The corrected184 preparation passes input/up/down
  bytes/scales, then fails NKI parser before MLP execution; this is185's premise.
- Trn22 source180 r2 finite sequence RUNNING since06:21UTC, coordinator11330,
  immutable source9e103ea6: native caller ingestion, library joined forward
  capture r2, library ingestion. Four ranks/whole chip/real data;30 steps per
  capture. Original180 r1 remains stopped on the NEFF_ARTIFACT substring bug:
  it counted TRACE_NEFF_ARTIFACT too. Exact record-boundary proof validates the
  native capture with unchanged numeric/payload fields. Do not restart r1.
- Native raw trace2,508,972,448B mirrored locally and every remote/local hash
  verified. GitHub asset upload is running; consult its artifact receipt.
  No DGE packet flag; inspect warnings before profile-derived conclusions.
- Trn22 scratch maintenance r2 COMPLETE:2,071 files/6.46GB archived and every
  file checked locally, live GitHub asset digests verified, then old remote
  copies retired. Free space11.64 to18.12GB, clean host. Shared NEFF caches,
  sources, checkpoints and current diagnostic artifacts retained.
- Both BF16 library whole-model tests COMPLETE/CLEAN with every original
  frozen requirement verified: joined561.502963ms vs489.888026ms (+14.619%);
  separate611.722864ms vs488.809250ms (+25.146%). No quality promotion.
  The isolated complete component's2.404% gain does not survive the caller.
- Full FP8 separate-region r2:592.811543ms vs488.469301ms (+21.361%), CLEAN.
  Its original r1 disk failure remains. Source179 ROW producer passes14 cases.
- Best public BPB0.9928513116311253, cold replay0.9938556748843932. Frozen
  submission/train.py/prepare.py unchanged. Source159 T512 MUST NOT dispatch.
- Last verified push06:20:55UTC sourcecc44c646/snapshotf90cb8ae. Consult
  out/fresh-start-backup/last-success.json for later refs. Cron preserves and
  pushes; it does not select experiments or merge branches. Continue explicit
  git/host locks, finite timeouts, immutable checkouts and pre/postflight cleanup.

## Previous source169/170 checkpoint (superseded host status)

- Source169 BOTH COMPLETE/CLEAN; original frozen plan requirements independently
  verified locally. MLP production forward4.049636ms vs native5.129885ms,
 21.058% faster including layouts. All6 FP64/repeat/batch checks pass. No
  saved ReLU/backward yet: do not claim a training or BPB improvement.
- Attention diagnostic completes all6 cases and original new-vs-bundled gates,
  while retaining1514 CPU coordinate failures (both arms/heads/fields). The
  global-RMS heuristic failure is not relabeled as passing. No promotion.
- Source170 PREPARED: token-row-sharded MLP (also requires unchanged native
  coordinate/RMS gate) and explicitly non-promoting attention paired timing
  with exact structural gates. Inspect fs-w170 receipts for actual status.
- Last169 free space16.77GB/10.26GB;8GiB reserve remains. Next MLP training
  implementation must explicitly save ReLU and charge its stores/backward.

## Previous source168/169 checkpoint (superseded host status)

- Source168 both STOPPED CLEAN after passing the four small fixtures. MLP
  outputs exactly native; large shape hits upstream B*S/unflattened-S input
  bug. Attention output/three gradients exactly bundled; T1024 hits NKI0.5
  rejection of intermediate reduce_res=None. CPU diagnostic explains the
  first FP64 mismatch exactly via BF16 exponential rounding/cancellation.
  It retains all355 coordinate failures per version and permits no promotion.
- Source169 PREPARED: explicit flat-row MLP view and attention row-max helper
  retaining intermediate readouts for the older API. No original numerical
  thresholds changed. Inspect fs-w169 receipts for subsequent actual state.
- Cron04:30:52 pushed sourcec36a1d14 and snapshotca6bc01f; both hosts retrieved.
  No full-model or quality gain claimed from this investigation so far.

## Previous source167/168 checkpoint (superseded host status)

- Source166 and167 finite library queues have STOPPED, both hosts clean.
  Do not restart labels. Source167 parser compatibility worked: attention
  output/dQ/dK/dV exactly match bundled for the first fixture. The FP64
  coordinate screen rejects the bundled control itself at one coordinate;
  investigate rounding/cancellation before promoting anything. MLP compiled
  but its wrapper indexed away a tensor batch axis; source168 fixes unpacking.
- Source168 PREPARED: trn21 repeat the unchanged six-case MLP screen with
  corrected return handling. Trn22 run an explicitly non-promoting attention
  numerical diagnostic retaining every original bound failure, with staged
  BF16 estimates from independent CPU algebra. No thresholds are changed.
  Check fs-w168 receipts for subsequent actual launch/completion status.
- Cron04:20:48 pushed source6ee56994 and snapshotf7297448, both hosts retrieved.
  Source167 plans/results and168 preparations need the next push; consult
  out/fresh-start-backup/last-success.json for authoritative later receipts.

## Source165 results and library audit

- Source165 BOTH finite queues completed cleanly by02:29UTC. Do not restart.
  The complete component win survives numerically but not in the model:
  all-scale550.129347ms vs native488.587103ms (+12.596%); backward-scale
  556.930141ms vs488.616140ms (+13.981%). All original numerical/rank/cache
  gates pass. No quality promotion. Comparison JSON fs-w165-scale-fusion-
  final-comparison-r1.json includes exact source/log/postflight identities.
- User supplied newly cloned nki-library. Audit and next experiments are in
  fresh-start-nki-library-audit-2026-09-20.md. Pinned upstream92d11f63;
  installed selected attention/MLP files match2.31_release, while Adam/CE
  already match main. Native SDPA already uses bundled NKI attention.
- Prepared isolated66-file1.18MB library source subset; no package install,
  backend replacement, canonical train.py edit or SDK upgrade. The CTE MLP
  needs an explicit ReLU2 hook despite its enum; separate scratch adapter
  uses upstream's two-instruction helper without changing pinned files.
- Source167 postflight found both hosts idle. Free17.19GB/11.26GB. Preserve
 8GiB reserve; use only guarded launches and immutable committed checkouts.
- Best public BPB remains0.9928513116311253; frozen submission unchanged.

## Previous source165 launch checkpoint (superseded host status)

- BOTH hosts started clean source165 queues at01:52:07 UTC. Source500ba34,
  immutable wave165 checkouts. Plan commit03ede7a. Do not edit a running
  plan/request/helper or restart its completed labels; inspect live state first.
- trn21 coordinator fs-w165-fp8-all-scale-model-sequence-r1 PID5712:
  complete forward+backward scale-fusion component (11original cases,32768
  tokens, exact retained outputs/gradients), actual Linear integration, fresh
  native full-step control, then FP8 candidate full-step timing. Bound10000s;
  requests3000/1500/1500/1500+30s each. Remote guard1615520 started01:52:13,
  first preflight and source16312case prerequisite passed; logs mirror every45s.
- trn22 coordinator fs-w165-fp8-backward-scale-model-sequence-r1 PID5676:
  backward-scale actual Linear integration, fresh native full-step control,
  then backward-scale FP8 full-model timing. Bound6000s; three1500+30s requests.
  Remote guard1931233 started01:52:13; first preflight and source163 numerical
  prerequisite passed. No other accelerator experiment may run on this host.
- Both model comparisons use same host/source, seed42, real data, dense6x1024,
  pack32/GA1,131072global tokens/update,5warmups30measured steps. All optimizer,
  communication and fresh weight-cache costs are charged. Every step needs
  its exact numerical records and clean postflight before the next starts.
  Each candidate must refresh all12MLP weight caches35times. Queues STOP after
  timing; no quality training or submission is automatically launched.
- Source163 MLP data-gradient win:4.706015ms vs prior FP85.548443ms,15.18%
  faster including extra backward preparation. All12cases pass original CPU
  bounds and exact outputs. Shared forward calibration is still borrowed;
  isolated data-gradient timing is NOT a whole-model or BPB result.
- Source164 narrow square-cache negative:19cases pass,1.631882ms vs prior
  dual1.628419ms/native1.576031ms. No promotion. Source159 shorter-context
  screen also loses0.662%; prepared30-minute T512 request MUST NOT dispatch.
- trn22 verified profile retirement completed clean01:49:39, label
  fs-maintenance-preserved-wave140-trn22-20260920T014925Z. Rehashed local/remote
  raw+parquet and verified live GitHub digests before removing only redundant
  remote copies and matching staging hardlinks. Free9.03GB→15.00GB. Intent,
  result, three progress records and script hash are committed. Sources,
  checkpoints, compilation caches and local/cloud originals are retained.
- Source162 DMA/source163 compute traces and all decoded tables are local-
  hash/cloud-digest verified. Non-DGE trace confirms Scalar activity halves;
  DMA notifications add substantial host overhead, so their timings are not
  speed evidence. Detailed results in fresh-start-continuous-work-2026-09-20.md.
- Latest verified cron01:50:55 pushed source500ba34. Plan03ede7a and these
  active launch receipts need the next deliberate push; check
  out/fresh-start-backup/last-success.json for newer exact source/snapshot refs.
- Canonical train.py/prepare.py and frozen submission unchanged (rechecked
 01:52UTC). Best matched public BPB0.9928513116311253; cold replay0.993855675.
  Cron preserves and pushes every10minutes; it selects no experiments.

## Previous production QKV checkpoint (superseded host status)

- trn21 source156 ed3a53e three-step sequence is COMPLETE. All 9 BF16 and
  10 FP32-V component cases passed. The production diagnostic completed cleanly
  at 00:25:41 UTC; coordinator completion is 00:26:02 UTC. Direct SSH at
  00:26:57 UTC confirms no device holders. This finite queue has no next job.
- At B32/T1024/C1024, cached differentiable inputs and seed152032, no-VE
  adapter comparisons are exact. With VE, Q-only, K-only, V-only and native
  preprojection are EACH exactly native, but joint BF16 replacement fails
  81 coordinate bounds (relative RMS 0.0017290317). Joint FP32-V replacement
  reproduces the original 47,860 failures (RMS 0.0041308524). Frozen BF16
  controls reproduce the 81 failures; frozen FP32-V has 47,839. These are
  diagnostic observations, not a pass of the original attention gate.
- The small-shape V precision explanation is insufficient at production
  shape. NEXT: isolate pairwise QK/QV/KV and the joint consumer graph or
  inspect compiled intermediates while preserving the native authority.
  Do not fit a shape-dependent precision switch or loosen error bounds.
  Broader FP8 projection coverage remains queued after a bounded diagnosis;
  neither BF16 component layout has demonstrated a speed win.
- trn22 source156 ed3a53e runs the focused production VE-on diagnostic in
  wave156. Coordinator PID5754 started 00:22:29 UTC, plan commit5b85c3c,
  bound2400seconds; remote guard1905745 started 00:22:35 UTC. Direct SSH
  at 00:26:57 UTC confirms device holder1905747 and no postflight yet.
  Label: `fs-w156-serial-qkv-channels-production-ve-boundary-r1`.
  It reuses completed source155 component gates with primitive/probe byte
  identity checked before execution. Initial joint BF16 result independently
  reproduces the token layout's 81 failures; remaining arms are running.
- Source154 token and source155 channel original attention gates both pass
  small-shape checks but fail production with exactly 47,860 coordinates,
  RMS0.0041308524. Both stopped cleanly before timing. All prior compiler,
  logger and numerical failures are preserved; no integration promotion.
- Latest token FP32-V component is 5.728514ms versus its matched explicit
  FP32-V native control 5.632454ms (1.705% slower). Earlier BF16 component
  layouts were 5.358%/10.478% slower. Distinct contracts must not be mixed.
  Source148 output-rank0.385% and reciprocal pack32 0.052% whole-step gains
  remain below3%. No whole-model or quality trial is currently queued.
- Frozen matched public best remains0.9928513116311253, dense6x1024/no ngrams/
  seed42; independent cold replay0.9938556748843932. No competition submission.
- Last verified cron push00:20:47 UTC: source4b29db3, snapshot119d34b2, both
  hosts retrieved successfully. This completion/launch checkpoint follows;
  read `out/fresh-start-backup/last-success.json` for the exact newer receipt.
  Live coordinators sync every45seconds; backup cron preserves work every10
  minutes but does not select or launch experiments. Queues are finite.

## FP8 output-rank timing and pack32 optimizer gates — 17:42 UTC

- Source147 streaming completed clean13:34:10.966945 while the session was
  idle. Median570.476580ms versus same-source resident562.968046ms,1.334%
  slower; identical loss5.120891094207764 and all four numerical postchecks.
  New summary links source/raw-log hash, timing and comparison. No promotion.
- Source148 is committed as60f5374bece8714e2bf2a13212a91ba953201074, deployed
  cleanly on both hosts at `/home/ubuntu/trainium_freshstart_20260918_wave148`:
  optional direct BxTxC NKI output for unpadded
  full-FP8 MLP calls, unchanged arithmetic/saves/four backward products. Input
  gradient shape restored explicitly; short padded calls retain the FP8 row
  path. Original11-case numerical and real-Linear gates extended, no relaxed
  tolerances. Local syntax passes; hardware checks and speed remain pending.
- trn21 native control COMPLETE/clean17:36:34.847072,488.035319ms median,
  5warmups/30timed updates and all four model postchecks. Its summary links
  raw-log/postflight hashes; the native coordinator completed17:36:45.898.
- trn22 output-rank gates COMPLETE/clean17:36:47.362014:11 independent cases
  including32768 tokens and real-Linear cached/uncached widths256/1024, exact
  retained outputs/master/input gradients, accumulation, cache refresh, short
  padding/row independence and native evaluation/restore. Summary preserved.
- Prepared fixed trn22 sequence: adopt those gates, then source148 resident
  compact FP8 control, then direct-rank FP8 candidate. These two timings share
  host/source; trn21 native is not a matched FP8 reference. Each request has a
  bounded remote guard, host lock and independent cleanup; advancement requires
  source-matched clean postflight plus all declared numerical result records.
  `fresh_start_sequence.py` and two exact request-digest plans are locally
  validated against completed hardware records and21 invalid cases. Coordinator
  source975ef2a is committed. Detached local coordinators STARTED17:35:45UTC:
  native PID6907 (completed), trn22 PID6908 (90-minute limit). Both adopted
  the existing dispatches and synced successfully17:35:52. The trn22 coordinator
  verified gate completion and launched resident control with clean preflight:
  `fs-w148-fp8-resident-control-phase-r1`, guard1842012,17:37:46.858660.
  Direct-output-rank candidate remains queued behind its clean/four-check result.
  Exact plan/helper
  hashes, PIDs, logs and state are recorded in their coordinator-launch/sequence
  receipts. Do not change a running plan, prepared request or coordinator/launch/
  sync helper: hash mismatch stops advancement. The native sequence only
  collects/verifies its one control and stops. Backup cron does not select jobs.
- New trn21 three-step plan adopts the completed native control, then runs
  current-inventory reciprocal/ownership gates and the pack32/GA1 candidate.
  Source98's prior reciprocal comparison usedpack16/GA2 (1.128% improvement);
  no pack32 reciprocal timing was found. Kernel/adapter bytes match source98.
  The new real-inventory gate must pass24 rank/stage checks plus four composition
  records before timing. Plan/wrapper checks pass locally and are committed in
  fa0d52a. Local coordinator PID16921 STARTED17:41:21.909988, one-hour limit.
  It revalidated the completed native control and launched
  `fs-w148-owned-reciprocal-current-gates-r1`, guard1530847,17:41:37.585278,
  clean preflight. Candidate timing is queued behind those gates/clean postflight.
  See `fresh-start-reciprocal-pack32-2026-09-19.md`. This is bounded composition
  alongside FP8, not a seed sweep or automatic quality promotion.
- Remaining priorities: serial QKV preserving Parameters/Muon states/fast RoPE,
  then wider FP8 projection coverage and measured optimizer/loss combinations.
  NLP ideas remain feasibility candidates alongside the kernel campaign.
- Last verified source push884645a, snapshot86618ecf,17:40:44UTC, includes
  bounded coordinators and both complete remote retrievals. Later completion
  summaries, reciprocal plans and launch state need the next deliberate push.
  Live coordinators keep mirroring exact job logs. Frozen public best
  remains0.9928513116311253; no competition submission or seed sweep.

## Matched width1152 evaluation and FP8 consumer comparison — 13:28 UTC

- Width1152 training COMPLETE/clean10:18:20.635599:3046 updates,399.2M target
  presentations,1784.7 training seconds/1785.2 charged estimate, no overrun,
  causality passed. Its small public0.988808 is not a matched20M result.
  Checkpoint571,028,961bytes, SHA256
  `aa57fe35f10ca2070499b68a3d3fc38a5d04d8bac0bae4d0722ab907d3a0afb3`.
  Local independent hash verified13:08:35 and GitHub asset digests13:11:10.
- trn21 matched width1152 eval COMPLETE/clean13:24:40.784776, source1c70da11:
  **0.9938092802993084**,20,971,520tokens/83,462,186bytes,57,493,436nats,
  711.94seconds, world1/B1/T1024/O1 without global autocast. All limits pass.
  This loses to frozen0.9928513116311253 by0.000957969; no promotion/seed sweep.
  Training and eval summaries link exact identities; checkpoint already cloud
  verified. trn21 is now idle/clean; do not rerun the completed capacity trial.
- trn22 source147 streamed correctness COMPLETE/clean09:59:18.490964:
  all11 independent cases including32768tokens, exact retained tensors and
  master gradients, cached/uncached real-Linear widths256/1024, accumulation,
  row/padding independence and native evaluation/restore. Summary preserved.
- trn22 resident compact phase COMPLETE/clean13:20:28.440699, all four model
  postchecks. Median562.968046ms, source145 repeat differs only0.012%; final
  diagnostic loss identical5.120891094207764. Summary and raw log preserved.
- trn22 ACTIVE `fs-w147-full-fp8-streamed-phase-r1`, sourceac4f789,
  guard1808156, started13:22:11.276554, clean preflight. The request verified
  all independent/real-Linear gates and resident clean/four postchecks before
  starting5warmups+30timed full-model updates. No stream speed result yet.
  NEXT after complete result/four checks/clean postflight: compare same-source
  resident timing. If promising, a new native control precedes any quality
  trial; otherwise inspect the separately documented output-layout hypothesis.
- Both prior90-minute watchers expired normally while the session was idle;
  new trn21=146/trn22=147 watchers started around13:09,90rounds/45s each.
  No existing remote device job was interrupted. Cron continued preserving
  completed logs and cloud assets between turns.
- New user ideas and exact FP8 scope are recorded in
  `fresh-start-structured-priors-and-fp8-scope-2026-09-19.md`:
  full MLP FP8 is not the full transformer; wider matmul coverage stays open.
  Generation-only speculation does not change teacher-forced BPB. English
  corpus metadata,20K-training-document structure audit, active package
  metadata and historical count-mixer/startup rulings are checked. New priors
  are design/feasibility candidates, not implemented submission changes.
  Hierarchical output factorization adds a training-cost interpretation of
  vocabulary organization; CPU normalization/routing feasibility comes first.
- The follow-up classic-NLP audit checked81/82 installed distributions and a
  broad selected package set. Only regex/tokenizers/tiktoken/transformers are
  present from that set. Whisper has number/contraction normalization source,
  but its spelling mapping is an external constructor input; no ready LM.
  The source hash and exact metadata are preserved. Older byte-feature WTE
  initialization is a known negative, not a new untried morphology direction.
- Source943c39f4 and snapshot67177429 pushed/ref-verified13:24:03, including
  resident completion, stream dispatch and the new head feasibility design.
  Width matched eval and expanded NLP audit need the next source commit/push.
  Frozen train.py/prepare.py/submission hashes
  were rechecked unchanged13:16. No competition submission.

## Width1152 quality and streamed FP8 correctness active — 09:51 UTC

- trn21 ACTIVE `fs-w146-dense1152-context1024-pack32-owned-ce-1800-r1`,
  source1c70da11, guard1442054, started09:45:14.368755, clean preflight.
  Compilation finished; finite training losses and about0.59s/step observed.
  One seed42,1800-second capacity trial; no ngrams. Preserve/check budget and
  postflight when finished, then retrieve/hash/cloud-backup final.pt and run
  the matched20,971,520-token public evaluation. No BPB claim before that.
- The source146 width1152 phase completed clean09:44:32.092383, all four
  numerical checks. Median587.591986ms versus width1024 control487.434961ms:
  20.55% more time for21.54% more parameters (142,752,108). This meets the
  predeclared700ms capacity-trial threshold; it is not a pure speed win.
- Source145 same-host three-arm sequence is COMPLETE, all clean/four checks:
  native488.702096ms, legacy FP8568.353812ms, compact FP8562.901787ms.
  Compact is0.959% faster than legacy but15.183% slower than native. Both FP8
  final diagnostic losses are exactly5.120891094207764. Unused-save removal
  helps only slightly; it does not repair the full-model regression.
  Compact postflight completed09:48:00.203536 with25.22GB free.
- trn22 source147 (`ac4f7899a17904591f9a02d1a30fcd4017fc79e9`) deployed clean
  at09:50:01.100742. ACTIVE `fs-w147-fp8-streamed-weight-gates-r1`,
  guard1769835, started09:50:09.448909, clean preflight. Early fixture outputs
  pass; wait for ALL11 cases, real-Linear gates and clean postflight.
  NEXT only after those pass: prepared source147 compact resident phase,
  then streamed phase on the same host/source, each5warmups+30timed updates.
  No speed or allocation gain claimed for streamed storage yet.
- Source140 caller analysis finds observed ordinary BF16 Linear tiles changing
  from mostly128x128 to128x32/32x128 around FP8. Weight lifetime and output
  layout are separate hypotheses. A rank-preserving output-layout control is
  designed, not implemented; see `fresh-start-fp8-output-layout-2026-09-19.md`.
- Watchers now retrieve trn21=146 and trn22=147 every45s for120 rounds each.
  Their replaced local processes were identity-checked and idle; remote device
  jobs were untouched. The10-minute cron recovery push remains separate.
- Source7629ffed and recovery snapshot03f9c12a pushed/ref-verified09:40:19.
  Current results/active manifests/design note await the next source push.
  All four raw/decoded caller datasets and maintenance archives are already
  local/cloud verified. Frozen best0.99285131 and cold replay0.99385567 stand.

## Compact FP8 and width1152 phases active — 09:39 UTC

- trn22 source145 legacy full-FP8 phase COMPLETE/clean09:34:54.637342:
  median568.353812ms, FW/BW509.880698ms, cache refresh5.257469ms,
  last loss5.120891094207764, all four numerical postchecks pass.
  Same-host native control488.702096ms; this reproduces the ~16.3% regression.
- trn22 ACTIVE `fs-w145-full-fp8-compact-phase-r1`, guard1766041, started
  09:36:11.069769, clean preflight, source63f6cce. Wait for its complete30-step
  result/four postchecks and clean postflight. Summarize all three source145 arms.
- trn21 ACTIVE `fs-w146-native-width1152-phase-r1`, guard1436331, started
  09:31:17.262621, clean preflight, source1c70da11. First warmup398.277s.
  NEXT after command0/clean/four model checks and phase median <=700ms:
  `fs-w146-dense1152-context1024-pack32-owned-ce-1800-r1.request.json`.
  This is one fixed-seed capacity trial, intentionally allowing a slower larger
  model. Its wrapper enforces prerequisites. Production training keeps winning
  clip-after-reduce; the diagnostic phase clips before reduction.
- trn21 maintenance-r2 COMPLETE/clean09:31:00.495730:6,277 original files,
  17,182,348,355bytes were archive/hash/cloud verified before retirement.
  Free disk15.176GB ->32.406GB. Local archive711,686,967bytes; cloud payload
  581,728,416bytes verified09:30:15. Shared caches/newer work remain.
- Prepared source147 is `ac4f7899a17904591f9a02d1a30fcd4017fc79e9`, not current
  HEAD. `out/fresh-start-wave147.bundle` and `out/deploy-wave147.py trn22` are
  ready; NOT deployed. Finish source145 compact comparison before deployment.
  Its next numerical request is
  `fs-w147-fp8-streamed-weight-gates-r1.request.json` (1800s bound, exact compact
  resident/original outputs, all four products/real-Linear lifecycle checks).
  Do not launch a streamed full model before these gates pass.
- Local hash-verified opcode/operand analysis finds changed ordinary-Linear
  TensorEngine geometry in source140 FP8 forward:16,384 observed128x32 and
  49,152 observed32x128 BF16 operations with512 free elements, versus mostly
  128x128 in native. This is a tiling/layout lead, not a layer identification
  or proof that weight residency causes it. EVENT/DMA warnings remain.
  Records: `fs-w140-shared-linear-opcode-attribution-r2.json` and
  `fs-w140-shared-linear-forward-tiles-r1.json`; analysis code is tracked.
- Source de6d6805 and snapshot2a942b88 were pushed/ref-verified09:30:00.
  Current attribution/new results/maintenance completion need the next push.
- Both raw and both decoded caller datasets are local/cloud verified. Watchers
  remain trn21=146/trn22=145. Frozen public best0.99285131 and cold replay
  0.99385567 are unchanged; no new FP8/width BPB or competition submission.

## Nearby controls complete; legacy FP8 and staged maintenance active — 09:29 UTC

- trn22 source145 native control COMPLETE/clean09:20:30.022658, all four model
  postchecks pass. Median488.702096ms, last loss5.126208782196045, consistent with
  source140. Summary preserves all phase medians, exact source and raw-log hash.
- trn22 ACTIVE `fs-w145-full-fp8-legacy-phase-r1`, guard1761315, started
  09:21:59.988638, clean preflight. NEXT after command0/clean/four postchecks:
  `fs-w145-full-fp8-compact-phase-r1.request.json` on the SAME source145 worktree.
- trn21 source146 D1024 control COMPLETE/clean09:23:29.405196, all four model
  checks pass. Median487.434961ms, FW/BW433.103516ms, loss5.126208782196045.
  Its15.18GB free disk is above reserve, but larger compilation needs headroom.
- trn21 maintenance `fs-maintenance-nki-scratch-trn21-20260919-r2` is capturing
  the oldest16GiB of ordinary idle compiler diagnostics before08:00UTC. It holds
  the host lock. Wait for local per-file verification, then put this artifact
  first in artifact-backups.json and upload/verify cloud digests, then retire
  ONLY its reverified original files. Existing shared caches/newer work remain.
  NEXT after clean retirement: prepared source146 width1152 phase request.
- Source147 candidate is prepared locally: optional sequential packed weight
  groups in compact FP8 forward. It preserves all paired contraction/rescaling
  arithmetic and trades repeated HBM loads for a lower source-level weight
  footprint. Numerical/full-model tests are PENDING; do not claim allocation
  or speed gains. See `fresh-start-fp8-weight-residency-2026-09-19.md`.
  Finish source145 three-arm comparison before deploying/running this candidate.
- Both raw caller captures AND both decoded32-table Parquet datasets are now
  locally hash verified and GitHub asset-digest verified. Artifact backup09:05
  completed63entries without errors. Native Parquet payload1,604,252,780bytes;
  FP8 Parquet payload1,961,821,219bytes. Receipts are under research/run-logs.
- Source27495b09 and snapshot92c1bb11 were pushed/ref-verified09:19:21. Current
  source147 preparations/controls/maintenance intent require the next commit.
- Watchers remain trn21=146, trn22=145. No new BPB: frozen package0.99285131,
  independent cold replay0.99385567. No private leaderboard/submission action.

## Compact/ownership gates passed; nearby whole-model controls active — 09:18 UTC

- trn22 source145 `fs-w145-fp8-compact-forward-saves-gates-r1` COMPLETE/clean
  09:09:36.852509. All11 cases, real32768x1024x4096, both256/1024 Linear gates,
  exact retained tensors/gradients, cache updates, row/padding independence,
  native evaluation/checkpoint restore pass. Summary JSON preserves the gates.
- trn22 ACTIVE `fs-w145-native-control-phase-r1`, guard1755778, started
  09:10:10.242854, clean preflight, source63f6cce. First warmup327.080s includes
  compilation. Process inspection confirms active work, no hang diagnosis.
  NEXT after command0/clean postflight/four numerical checks:
  `fs-w145-full-fp8-legacy-phase-r1.request.json`, then
  `fs-w145-full-fp8-compact-phase-r1.request.json`. Both requests enforce prior
  gates. Check rank records with JSONDecoder.raw_decode: rank stdout can
  concatenate complete JSON objects on one line.
- trn21 source146 `fs-w146-width1152-optimizer-ownership-gate-r1` COMPLETE/clean
  09:11:01.480307. All28 exact comparisons across four ranks/seven stages pass,
  including schedules, None/zero gradients, grouped state, padding, refreshed
  caches, continued and restored updates, and all-rank weight-byte hashes.
  Owned optimizer including exchange31.90–31.92ms versus replicated68.67–69.33ms.
  This compares ownership with replication, not the former fallback plan.
- trn21 ACTIVE `fs-w146-native-width1024-phase-r1`, guard1429666, started
  09:13:27.594494, clean preflight. Source146 is
  `1c70da11c4dd9bb789afca0fa3088842874a1efd`. NEXT after clean control:
  `fs-w146-native-width1152-phase-r1.request.json`. Inspect remaining disk
  before the new larger compile; last read-only snapshot was13.89GB free.
  Additional verified old-diagnostic retirement may be needed for headroom.
  Do not overlap maintenance with an active job or erase shared caches.
- Source146 and snapshot53f2b803 were pushed/ref-verified09:08:07. Current
  completed gate evidence/new plans need the next source commit and push.
- Raw native and FP8 profiles are local/cloud verified. Both32-table parquets
  are locally hash verified; the automatic artifact backup is uploading them,
  with no reported errors. Do not claim a cloud copy until its receipt verifies.
- Serial QKV execution is queued in `fresh-start-serial-qkv-revisit-2026-09-19.md`.
  It preserves separate Parameters/Muon states, serial block order, original
  attention inputs and contiguous RoPE fast path. It is a design, not a new win.
- Watchers: trn21=146, trn22=145. Frozen submission/public best0.99285131 and
  independent cold replay0.99385567 are unchanged. No FP8 or wider-model BPB yet.

## Retained FP8 saves confirmed; compact gate and width costs active — 09:02 UTC

- trn21 ACTIVE source144 `fs-w144-width1152-optimizer-group-costs-r1`, guard
  1417752, started08:58:35.465128, clean preflight.13 cases derive from the
  actual142,752,108-parameter model. Eight costs have appeared; wait for command0
  and clean postflight before using the grid. New costs/ownership are not yet
  installed or numerically gated. Input config is the byte-verified committed
  source144 mirror of the trn22 profile log, not a missing cross-host path.
- trn22 ACTIVE source145 `fs-w145-fp8-compact-forward-saves-gates-r1`, guard
  1748935, started09:00:04.923393, clean preflight. Immutable source145 is
  `63f6cceca2e70bafaa747c0c9901844276433403`. First numerical probe covers ten
  small edge cases plus32768x1024x4096; second covers real-Linear masters,
  caches, padding and native restore. No timing claim from this job. Check both
  summary gates and clean postflight before a whole-model phase.
- Source144 caller analyses both COMPLETE/clean: FP8 at08:53:12.272984, native
  at08:57:06.594962. All runtime dispatches (45native/33FP8) plus each startup barrier match. Forward device
  intervals137.156557(native)/209.748307(FP8)ms; backward272.922323/275.310033ms.
  Same-host unprofiled source140 phase remains authoritative for speed.
- Retained BF16 writes CONFIRMED:3,013 observed DMA rows at cached kernel
  lines149/186,1,973,157,888 observed write bytes. Source135 file equals source140
  byte-for-byte. Exact parquet identity verified locally. FP8 dropped EVENT and
  DMA notifications; native dropped DMA notifications. No exact traffic/bandwidth
  inference or claim that shared Linear line2853 identifies a particular layer.
  Evidence: `run-logs/fs-w144-pack32-caller-comparison-r1.json` and
  `run-logs/fs-w140-full-fp8-retained-forward-saves-r1.json`.
- Both raw caller captures are locally hash verified. Full-FP8 raw is also
  GitHub asset-digest verified. Native raw asset upload is ACTIVE; verify final
  backup receipt. Full-FP8 parquet is locally verified32files/2,149,367,880bytes;
  native parquet transfer is ACTIVE. Both ingests match every locally preserved
  raw hash and leave originals unchanged. Cloud-back up both parquets next.
- Watchers are trn21=144 and trn22=145. Only the completed local trn22 wave140
  watcher was stopped via verified PID identity; no remote experiment was touched.
- After compact gates and transfers finish: nearby same-host source145 native,
  legacy-full-FP8 and compact-full-FP8 phase measurements, each with its own
  pre/post cleanup. Compact is default OFF. Frozen submission/public best
  0.99285131 and independent cold replay0.99385567 remain unchanged.

## FP8 caller analysis active; native decode and compact preparation — 08:53 UTC

- trn21 source140 full-FP8 ingest COMPLETE/clean08:51:43.042083. Raw capture
  (70files/4,941,989,339bytes) is local and GitHub asset-digest verified. Ingest
  retained source hashes. Its parquet directory is now copying locally.
- trn21 source144 `fs-w144-full-fp8-pack32-caller-analysis-r1` is dispatching/running;
  verify its exact process/postflight before another job. It uses the exact
  source140 trace and parquet, checking every binary dispatch and startup barrier
  before attributing source families, engine unions, precision and spill counts.
  The added families include all current FP8 backward/preparation kernels.
- trn22 source140 `fs-w140-native-pack32-caller-remote-ingest-r1` ACTIVE,
  guard1745363, started08:45:05.285630, clean preflight. Its raw profile is
  copying locally with the explicit clean maintenance receipt; the original
  disk-only failed profile postflight remains intact. Native analysis request
  `fs-w144-native-pack32-caller-analysis-r1.request.json` is ready AFTER clean
  ingestion. Do not skip its runtime/NEFF identity checks.
- trn22 maintenance `fs-maintenance-nki-scratch-trn22-20260919-r1` COMPLETE/clean
  08:43:38.274715.12,549 files/25,775,762,804 original bytes were reverified
  locally and in current GitHub assets before remote retirement. Free disk
  7,152,185,344 -> 33,026,797,568bytes. Shared caches/newest profile/compile remain.
- Width1024 and1152 pack32/T1024 native MLP integration BOTH PASS on source143,
  clean08:35:45.433370. Source144 `b291b930ca07da483bde0746d0ddce2d1627048c`
  is deployed on both hosts; its real-inventory optimizer cost probe and bounded
  request `fs-w144-width1152-optimizer-group-costs-r1.request.json` are ready.
  Launch on trn21 after caller analysis and profile transfers there finish.
  No new optimizer costs or ownership plan have yet been validated.
- Compact full-FP8 forward saves are prepared locally, default OFF. The new
  five-output mode removes only discarded BF16 activation saves. Independent
  four-product/byte gates and real-Linear gates are prepared, not run. Inspect
  the source140 trace for retained writes before spending device time on it;
  commit/deploy the new source before a gate. No FP8/BPB promotion.
- Both ingests report incomplete DMA timing metadata. Do not report exact
  bandwidth. Full-FP8 ingestion took about13minutes but completed within its
 1200s limit; do not terminate the still-progressing native ingest. The installed
  Explorer also documents `--ignore-instruction-hierarchy` if future decoding
  needs a bounded alternative. It is not enabled in these captures/ingests.
- Watchers: trn21=144, trn22=140. Source144 and snapshot79592171 were pushed
  and ref-verified08:43:13; newer prepared source/receipts need the next commit.
  Frozen package/public best0.99285131 and cold replay0.99385567 are unchanged.

## Both caller captures complete; width gate and preservation — 08:36 UTC

- Source140 full-FP8 profile COMPLETE/clean 08:29:08.062744, command0, all four
  numerical postchecks pass. Native profile-r2 COMPLETE 08:31:50.417156,
  command0, all four postchecks pass; its only postflight failure is disk reserve
  (7,263,350,784 bytes free). Preserve the original failed health receipt.
  `run-logs/fs-w140-pack32-caller-capture-summary-r1.json` records the exact
  summaries. Native r2 completed without NRT1204 after removal of the optional
  DGE-notification flag, supporting an instrumentation-triggered failure in r1.
- Full-FP8 raw-directory preservation is active locally. trn22 maintenance
  `fs-maintenance-nki-scratch-trn22-20260919-r1` is archiving the oldest24GiB of
  same-UID `/tmp/nki_*` files modified before07:00UTC. Retirement must wait for
  per-file local verification and current GitHub asset digests. No current
  profile, newest compilation or shared immutable cache is selected.
- trn21 ACTIVE `fs-w143-native-mlp-widths1024-1152-integration-r1`, guardPID
  1408379, started08:32:10.827313, clean preflight. Immutable source143 is
  `2f826dc2e8ae244f67e91e128e6692b52b99cddc`. Native MLP guard is now correctly
  scoped to the resident NKI forward; two separate subprocesses gate widths1024
  and1152 at pack32/T1024, with actual Linear caches/master gradients, two
  accumulated microbatches/two updates, exact evaluation and native checkpoints.
  This is correctness-only work while raw trace copies proceed, not timing.
- Log watchers: trn21=143, trn22=140. Caller analysis now recognizes the complete
  FP8 data-gradient, calibration, weight-gradient and dual-cache source families.
  Next: verified raw preservation, CPU-only64GiB/4thread ingestion, then exact
  runtime-to-NEFF matching and source/allocation attribution. Do not launch
  ingest on trn21 until its width gate has completed with clean postflight.
- Source143 and all preceding maintenance/status source commits are pushed and
  ref-verified at08:31:21. Snapshot branch was verified at `31d4488ff7e3e6ef67c39a720b53058751b339fc`.
  No new BPB result; frozen submission remains unchanged.

## Both source140 caller profiles active — 2026-09-19 08:26 UTC

- trn21 ACTIVE `fs-w140-full-fp8-pack32-micro-device-profile-r1`, guard PID
  1404853, started 08:25:23.258911; clean preflight on immutable source140.
- trn22 ACTIVE `fs-w140-native-pack32-micro-device-profile-r2`, guard PID
  1738521, started 08:25:04.208775; clean preflight on the same source140.
- Native profile-r1 failed during warmup with NRT1204 software notification
  queue overflow. No numerical/profile result. Automatic postflight is clean,
  command 1, no leftover workers/device holders. The added global DGE-notification
  flag is the leading hypothesis. r2 removes only that optional environment
  flag; the model, compiler flags, data and shape are unchanged. The candidate
  profile also uses the previous working native-profiler settings. Do not claim
  exact DMA traffic when the resulting schema/diagnostics lacks that evidence.
- Maintenance `fs-maintenance-nki-scratch-trn21-20260919-r1` COMPLETE/clean at
  08:24:39.373976. All 8,926 original files / 17,183,487,985 bytes were verified
  in the 714,780,213-byte local archive and current GitHub assets before remote
  copies were retired. Free disk 8,221,212,672 -> 25,478,021,120 bytes. Shared
  caches and newest candidate compile were excluded. Receipt and original
  failed candidate postflight are retained; do not rerun the retirement.
- Retrieve and preserve both profiles only after command 0 / clean postflight
  and four numerical postchecks. Then ingest their exact runtime/NEFF identities
  before judging unused BF16 writes, ordinary Linear spills or precision changes.
  The previous 3% isolated MLP gain is still not a whole-model win.
- Source/status commit `e6e03a0` and snapshot `4d132690` pushed and ref-verified
  at 08:19:29. New profile/maintenance records need the next regular backup.

## Resume kernel investigation after submission handoff — 2026-09-19 08:20 UTC

- Short launch/package update is pushed as `b857d13`; frozen submission remains
  public best 0.99285131 / independent cold replay 0.99385567. No new BPB result.
- Source140 full FP8 candidate finished at 07:50:45, command 0, all four numerical
  postchecks passed. Median step 567.094872 ms versus native control-r2
  488.701909 ms: 16.041% slower. Forward/backward accounts for 74.667012 ms of
  the difference; cache refresh adds 2.070754 ms. The candidate postflight failed
  only the disk reserve (8,224,735,232 bytes free); no device holders, remaining
  workers or ECC errors. Treat it as diagnostic and retain the original failure.
  Exact records: `run-logs/fs-w140-full-fp8-phase-comparison-r1.json`.
- trn22 source140 now has a native pack32 microstep device-profile request:
  `run-logs/fs-w140-native-pack32-micro-device-profile-r1.request.json`.
  Its first bundle transfer failed because the incremental prerequisite existed
  only on trn21. The r2 deployment uses trn22's source129 base and verifies the
  exact source140 worktree. Both receipts are retained. Check dispatch/postflight
  before launching anything else there; preserve every binary and trace.
- trn21 maintenance request:
  `run-logs/fs-maintenance-nki-scratch-trn21-20260919-r1.request.json`.
  New `scratch/maintenance_archive_nki_scratch.py` captures the oldest 16 GiB of
  same-UID `/tmp/nki_*` diagnostics modified before 07:00 UTC while holding the
  host lock. It verifies every source hash inside the local archive. Retirement
  is a separate stage requiring current GitHub asset digests and remote hashes.
  Shared caches, newest candidate compile, source and checkpoints are excluded.
  Do not dispatch the FP8 profile until maintenance is complete and health clean.
- Next kernel decision: compare actual native and full-FP8 caller traces. Check
  whether unused BF16 forward saves are retained and attribute the 75 ms model
  cost before implementing the explicitly scoped output-elimination mode.
  Capacity/optimizer balancing and composed reciprocal updates remain queued.


## Submission launch fits the 256-character field — 2026-09-19 07:45 UTC

- `submissions/dense6x1024_public0992851/launch-command.txt` is now 245
  characters excluding its trailing newline. Packaged `train.py` is unchanged at
  SHA256 `35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2`.
- Omitted values are frozen source defaults: exact `NEURON_CC_FLAGS`, one million
  maximum steps, 1,800-second budget, depth 6, seed 42, `--compile`, and
  `--compile-muon`. Every retained flag differs from the packaged source default
  or is required to establish the four-rank LNC2 launch.
- On idle trn22, isolated parser probes showed the old and short commands have
  exactly equal complete argument namespaces and compiler environments. The
  organizer `prepare.py check` also passed the shortened line. These checks did
  not start training or touch Neuron devices. Evidence is in
  `research/run-logs/fs-submission-short-launch-equivalence-20260919.json`.
- Rebuild/package verification is complete. Commit and push this record before
  reporting the short command to the user. No competition submission was made.

## Current handoff: submission ready, full FP8 phase active — 2026-09-19 07:39 UTC

- User's requested upload is submissions/dense6x1024_public0992851/train.py,
  SHA25635c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2.
  Paste the complete launch-command.txt line. All helpers are inline; upload
  requires no research modules. Package plus complete validation is already
  pushed in d46bdb7. Best public0.9928513116311253; independently cold-trained
  package0.9938556748843932 over20,971,520tokens, charged1790.341476s. Both
  rehearsal postflights are clean. No competition submission has been made.
- Native control-r2 COMPLETE/clean07:35:36.482392, all four postchecks exact,
  median488.701909ms, last loss5.126208782196045, free10,251,456,512B. Use this
  control for comparison. The earlier reserve failure and recovery are retained.
- trn21 ACTIVE: fs-w140-full-fp8-candidate-phase-r1, guardPID1392505,
  started07:37:54.937632, clean preflight, immutable source140
  2a6f267b443e262bb9d7955e06d539f603dc4edc. Same T1024/pack32/GA1, optimizer
  ownership, data and seed. First full-GPT timing of four-product FP8 backward
  with compiled dual weight preparation. Thirty timed steps follow five warmups;
  require command0, clean health and four numerical postchecks before judging.
  Phase JSON is ONE record containing a30-element steps list, not30 records.
  Compare against control-r2, not its predecessor. No FP8 BPB claim yet.
- trn22 is idle after the clean source129 public evaluation. Do not restart
  its completed cold training or evaluation. trn21 watcher follows140 and is
  active; trn22's129 watcher may expire, but scheduled backups discover all jobs.
- If the full-GPT FP8 result disappoints, the next bounded source audit is in
  fresh-start-full-fp8-mlp-diagnosis-2026-09-19.md: determine whether discarded
  BF16 activation/weight outputs still cause writes in the actual compiled
  caller before adding a full-VJP-only save mode. No kernel change for this
  idea has been made. Capacity guard/ownership issues have a separate note.
- All five selected old trn21 checkpoint copies have now been retired with
  local/cloud preservation intact. The one-off recovery must not be rerun.
  Source142cf6c12b/snapshot35c95cb pushed and verified07:33:16; newer status and
  final phase records are covered by the next explicit/scheduled backup.

## Research host recovery complete; clean control repeat — 2026-09-19 07:32 UTC

- Submission is fully ready and already pushed in d46bdb7, verified07:25:46.
  Fresh public0.9938556748843932; exact old-checkpoint replay0.9928513116311253;
  cold charged1790.341476s. Give the standalone train.py and launch-command.txt.
  Private leaderboard result unknown. trn22 is idle after clean public scoring.
- Native control140-r1 completed command0 and all four numerical postchecks,
  median489.272651ms, exact historical last loss5.126208782196045. Its postflight
  failed only disk reserve (7,746,342,912B free). Keep it diagnostic, not the
  authoritative paired control. A fresh control-r2 request is now dispatching
  on the same immutable source140, with the exact command and initialization.
- Five old checkpoints remain fully preserved locally and in verified GitHub
  assets; only their redundant trn21 copies were retired. The first maintenance
  attempt ending072852Z failed after two retirements because write_json
  intentionally refuses overwriting an existing progress record. Preserve its
  failure, intent and remote evidence. Source142cf6c12b changed progress to
  unique numbered records and restricted recovery to the remaining three.
  The original writer was exercised locally for three successive unique records.
- Recovery fs-maintenance-full-fp8-phase-headroom-20260919T073150Z COMPLETE/clean:
  free9,669,758,976 -> 11,079,245,824B. All remaining local/cloud/remote hashes
  were reverified under the host lock. Currentbest/checkpoints on trn22,
  source, logs, profiles and shared caches were untouched. Do not rerun this
  one-off recovery; its selected files are now absent by design.
- Launch research/run-logs/fs-w140-full-fp8-candidate-phase-r1.request.json ONLY
  after control-r2 is clean with all four postchecks. Watcher trn21=140 is active;
  trn22 watcher=129. Latest maintenance/results still need final commit/push.

## Submission is ready — 2026-09-19 07:25 UTC

Exact packaged cold launch scored 0.9938556748843932 public BPB on the protected
20,971,520-token scorer, evaluation697.480227s. Both training and evaluation
command0/clean; cold charged time through process exit1790.341476s. Original
public best remains0.9928513116311253, exactly replayed by this package.
Give the user submissions/dense6x1024_public0992851/train.py and the exact
launch-command.txt line. All NKI helpers are inline. No competition submission
has been made. Package README/validation and best-public pointer now record
readiness; commit/push these final records before the final response.

trn21 control140 remains active: warmup0 took316.18s, further compilation may
continue. During compilation free disk dropped to7,099,265,024B. DO NOT delete
artifacts while the host lock is held. Prepared one-off
research/scratch/maintenance_full_fp8_phase_headroom.py retires exactly five
old checkpoint copies (about3.333GB), each preserved locally/cloud, after
control140 exits successfully. It re-verifies all local/cloud/remote hashes,
excludes currentbest, holds the host lock, and permits only disk-reserve health
failures while retaining the original postflight. Run only after completion;
then require clean actual health and four native control postchecks before
launching the already prepared FP8 candidate. No shared cache/source/log purge.

Current dispatch update: trn21 source140
2a6f267b443e262bb9d7955e06d539f603dc4edc is running
fs-w140-full-fp8-native-control-phase-r1, PID1383339, started07:16:23, clean
preflight. Candidate request research/run-logs/fs-w140-full-fp8-candidate-phase-r1.request.json
is prepared; launch only after control completes clean with four postchecks.
Watcher trn21=140 from07:16:23, trn22=129. GitHub source140/snapshot22cd791
verified07:16:04. Heavy backup completed07:14:17:57 artifacts verified, no errors.
New dead-BF16-output investigation is recorded in the full-FP8 diagnosis note;
it is queued behind current comparison, not yet implemented or measured.

## Submission cold launch passes; fresh public evaluation active — 2026-09-19 07:15 UTC

- Exact packaged launch, six empty SDK caches, completed clean on trn22 at
  07:08:13.561380: 3,634 updates, 476,315,648 targets, 2,645.428628 wall seconds.
  Observed first-five-boundary allowance 855.087152s; charged through process
  exit 1,790.341476s. Checkpoint SHA256
  f985887995c9ffe9d7efb6124926d842aac399059f23f04c1295fb09a145d45a,
  469,834,849 bytes. Local/remote hashes match; GitHub asset digests verified
  07:13:37. Root train.py, prepare.py and packaged train.py unchanged.
- trn22 source129 c51f6fdc5ca77793a1524148a563cafac0ab22b0 ACTIVE:
  fs-w129-submission-cold-public20m-r1, guard PID1725340, started 07:11:35.
  Protected scorer B1/T1024/O1, 20,971,520 tokens, timeout1200. Need its fresh
  checkpoint score before final readiness report. Retrieve nested rehearsal.json,
  require clean postflight, update package README/validation and commit/push.
  Original packaged-checkpoint replay already exactly reproduced 0.99285131.
- Before evaluation, one redundant wave4 raw-profile archive was retired after
  local/current-cloud/remote SHA verification under host lock. Archive remains
  local/cloud; no checkpoint, source, log or shared cache removed. trn22 clean,
  free 8,633,110,528 -> 13,742,022,656 bytes. Maintenance receipt
  fs-maintenance-submission-eval-headroom-20260919T071044Z.
- trn21 source139 full FP8 real-Linear integration gate COMPLETE/clean 07:11:27.
  Widths256/1024, cached/uncached, two microbatches/four reference updates,
  exact full outputs/input/master gradients, cache refresh/restore, training
  padding/causality/row independence, native eval/checkpoint restore pass.
  No whole-GPT speed or BPB claim. Prepared distinct --fp8-mlp-full phase flag;
  next compare native/custom-backward and full FP8 at T1024 pack32/GA1 on the
  same host/source. Existing phase timer clips before reduction on BOTH arms;
  it is a matched phase comparison, not the winning clip-after quality loop.
- Watchers: trn21 wave139, trn22 wave129. Source139 and snapshot e231aec pushed
  and verified by 07:10 cron. Cold score and newer notes still need final push.
  User asks for submission file/command; no competition submission authorized.

## Fused FP8 producer passes; final submission training minutes — 2026-09-19 07:06 UTC

- trn21 source137552672ff705f30662f230d1c30c165a5404909c7 both resident
  and streamed dual-cache probes COMPLETE/clean. All8 cases include independent
  metadata/QDQ/layout references, exact old/new cache bytes and full derivatives,
  repeated calls and refreshed/restored FP32 masters. Current32768x1shape.
- Resident preparation0.294823ms vs previous compiled0.541435ms; full FP8 MLP
  14.886892ms vs compiled native15.348333ms (~3.01% faster component).
  Streamed14.906892ms vs native15.355214ms (~2.92%). Resident slightly ahead;
  no whole-model throughput or BPB claim. Next: real-Linear full-VJP wiring gate.
- Prepared maintenance_submission_eval_headroom.py for AFTER cold training:
  only one old wave4 raw-profile archive may be retired from trn22 after fresh
  local/current-cloud/remote verification under lock.5.1GB stays local/cloud.
  Requires exact completed cold job, command0 and no non-disk health failures.
  Do not run while training is active; no checkpoint or shared cache deletion.
- Cold standalone training still finite, past3000steps and second data pass.
  Final checkpoint is out/<cold-label>/out/final.pt onwave129. Preserve using
  --relative-checkpoint, then score with same standalone package and protected
  20M scorer. Private submission has NOT been made.
- GitHub source137/snapshot6c8ae2f verified06:57:07;07:00cron also pushed source137
  and snapshot58b6022. Watchers trn21=137, trn22=129. Best public0.99285131.

## FP8 complete component wins at pack32 — 2026-09-19 06:56 UTC

- Source135 pack32 gate COMPLETE/clean. Independent forward and saved-byte
  checks, all4 gradient products with negative controls, repeated calls,
  refreshed/restored masters, FP32 accumulation stress test and native FP64
  rounding-bound control pass at32768tokens. Final timing uses ONE microbatch.
- Complete FP8 with compiled preparation15.110784ms vs compiled native15.382169ms
  (1.76% faster), previous eager FP8 path15.565401ms. Prep1.005657→0.549269ms.
  The former16384x2shape remained0.485%slower. No whole-model/BPB claim.
- Prepared new dual weight producer: online calibration plus FW/BW packed
  layouts from one quantization, no BF16 HBM transpose, optional resident8MiB
  weight cache. The separate probe retains previous compiled and native controls,
  all independent cache gates, exact complete gradients and refreshed masters.
  Source135 large-shape arithmetic is prerequisite; do not skip it on failure.
- trn22 submission cold rehearsal remains active,~0.49s/update,~13minutes left
  at last check. Package source129 unchanged. Need final checkpoint/time gate
  and fresh20M public evaluation. Watchers trn21=135,trn22=129.
- The06:50cron backup safely skipped because our Git index changed during its
  snapshot. An explicit retry started06:53; check latest.json before claiming
  current source and notes are pushed. No concurrent merge was performed.

## Current pack32 FP8 gate and recovery headroom — 2026-09-19 06:51 UTC

- trn21 source13560e8d96f56dbfc566523ce38261b9d2a64264e21 ACTIVE:
  fs-w135-fp8-compiled-cache-pack32-component-r1, guardPID1368846,
  started06:46:26.849805. Full independent32768-token gates precede the actual
  single-microbatch measurements. No result/whole-model promotion yet.
- Before deployment, source3old table-model checkpoint copy1,453,684,039B was
  retired from trn21 only. Full local SHA and current GitHub asset digests plus
  remote hash were reverified under host lock; free8,926,478,336→10,380,165,120B,
  clean. Original model remains locally/cloud preserved; currentbest excluded.
  Generic procedure is fresh_start_retire_checkpoint.py; exact prior clean job
  and explicit path are mandatory. No active-job data/cache/source/log removal.
- trn22 cold standalone training still active,~0.49s/step, no NaNs. Need final
  timing/checkpoint preservation plus protected20M evaluation before readiness.
- Submission rebuild into a new ignored directory matched train.py, launch and
  provenance byte for byte. Source129 train SHA35c9ef79... remains unchanged.
- New capacity note records a native-path SBUF assertion and width-specific
  ownership-cost fallback. Metadata-only illustration uses original make_plan
  AST and shape stubs (local Torch unavailable); no device or BPB claim.
- Watchers trn21=135 (from06:48:47), trn22=129. Latest committed source135;
  latest explicitly verified GitHub source133 from06:41:38. Push pending notes.

## Compiled FP8 preparation completes; submission training continues — 2026-09-19 06:44 UTC

- trn21 source13210946f008e5680d2071a93f36fd61dbf53dc3100 component
  COMPLETE/clean: all8 cases, exact eager/compiled packed bytes/metadata/qdq,
  native and FP8 complete outputs/gradients, updated and restored masters.
  Cache prep1.006624→0.549580ms (45.40%); complete FP8 MLP15.863606→15.443611ms
  (2.65%). Compiled native15.369117ms: FP8 still0.485%slower. No BPB promotion.
- New prepared extension adds32768tokens/one microbatch, matching pack32.
  Requires independent forward + all4 gradient product bounds at new shape,
  not just compiled/eager self-comparison. Numeric gate retains two distinct
  microbatches as an accumulation stress test; final timing uses requested one.
- trn21 successful replay's redundant transfer copy retired after local/current
  cloud/remote hashes; host lock held, clean, free9,107,853,312B at06:40:55.
  Earlier retirement attempt failed before mutation by reading command status
  from condensed state. Fix reads the full postflight; both records retained.
- trn22 cold rehearsal remains active with finite loss at steady~0.49s/step.
  Source1338aa2498 and snapshotdb3165e were verified on GitHub06:41:38.
  Watchers trn21=132, trn22=129. Package source129 is unchanged.

## Standalone scorer replay exact; cold training underway — 2026-09-19 06:40 UTC

- Packaged source129 replay COMPLETE on trn21, clean06:35:54.439321:
  20,971,520tokens,83,462,186bytes,57,438,016nats, BPB0.9928513116311253
  EXACT to source98, scorer692.860s. Protected scorer, B1/T1024/O1. The
  standalone train.py reproduces the score; this is not a private submission.
- trn22 cold training reached fifth boundary within900s and is now steady at
  about0.49s/update. First step474.47s; subsequent startup compilation is over.
  Final independently observed allowance/checkpoint deadline await completion.
  Original source98 training loop unchanged, including its early wall-clock LR
  behavior. Cold startup can change early LR; do not claim identical weights.
- One-off maintenance script may retire ONLY the redundant source98 checkpoint
  transferred to trn21 after verifying local/current cloud/remote hashes and
  exact successful replay state under host lock. Original source98 preserved.
- Source131 probe prepared; plan deploy after replay cleanup. Both wave129
  watchers still live. Source13159120a8 and snapshot6e11d0b pushed06:34:47.

## Cold startup underway — 2026-09-19 06:34 UTC

- trn22 cold rehearsal source129 started06:23:59, guardPID1688427. First
  synchronized optimizer boundary completed, logged loss9.0112, stepdt474.47s.
  Initial full-graph compilation is real, not a warmed-cache test. Further
  startup compilation still active; fifth-boundary allowance900s not yet known.
  Free space at06:32:34 was8,919,302,144B (during compilation previously8.46GB).
  Do not prune files/caches while its device lock is held. If postflight fails
  disk reserve only, preserve original failure and perform verified recovery.
- trn21 source12920M replay still running, guardPID1359747; began06:23:59.
- Source13159120a8127fb4a40a1a1f33caff28329ad7b6c7f adds prepared
  `fp8_compiled_cache_probe.py` but NOT yet deployed/run. It keeps source124's
  16384x2micro shape for comparison. Before promotion, explicitly test current
  winning32768x1micro shape too; do not extrapolate the compiler ranking.
- Both source129 watchers active. Source13054b0bf8/snapshot76e68d5 pushed
  and verified06:24:33. Latest probe and validation manifest await backup.

## Submission contract verified; final rehearsal — 2026-09-19 06:24 UTC

- Final candidate source129c51f6fdc5ca77793a1524148a563cafac0ab22b0
  deployed to bothhosts. `submissions/dense6x1024_public0992851/train.py`
  SHA25635c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2.
- CPU and native contract129 BOTH PASS, clean06:23:06/06:23:18 respectively.
  Original B1 checkpoint logits, rowwise reference, causal future independence,
  batch size/permutation/other-row independence, repeated calls all EXACT.
  Parameters frozen/no hooks; weights and buffers unchanged.117,453,580params.
- Source127 CPU batch-size gate failed;128 reproduced identical original-model
  behavior: max0.0872438, no dependence on other-row content; FP32 diagnostic
  reduced difference to2.64645e-5. Original native127 passed every gate exactly.
  Preserve both failed CPU records. Final evaluator uses a GPT subclass that
  executes each independent sequence at B1, retaining original B1 logits and
  checkpoint keys while eliminating backend batch-shape rounding differences.
  Training is unchanged; no tolerance relaxation. Native129 B2 also exact to
  original B2. Original CPU B2 may differ; documented in diagnostic, not hidden.
- trn22 job requested `fs-w129-submission-cold-rehearsal-1800-r1`: actual
  launch-command.txt parsed without research flag injection, onlytrain.py plus
  organizerprepare.py in sandbox. Six fresh job-owned SDK cache dirs, no shared
  cache purge. Independently observe first five official boundaries and time
  through process exit. New checkpoint will be nestedout/<label>/out/final.pt;
  localpreserver now accepts explicit--relative-checkpoint for that layout.
- trn21 requested `fs-w129-submission-source98-public20m-r1`: protected20M
  public replay using packagedtrain.py, original preserved source98checkpoint,
  B1T1024/O1/no globalautocast. No private validation or competition submission.
- Watchers both129 from06:22:49 for90x45s. Need retrieve final logs/checkpoint,
  validate live postflights, update packageREADME/provenance validation status,
  commit/push and give user concrete file plus launchcommand. Publicbest0.99285131.

## Submission packaging — 2026-09-19 06:18 UTC

- User asks for the sub-1.0 train.py and submission launch command. Public best
  remains0.9928513116311253 (source98,20,971,520tokens); private result unknown.
- Built `submissions/dense6x1024_public0992851/train.py` with all required
  source98 helper/kernel code inline. SHA25695be3ac7d55d99ae45fec25de319557e7084173e41072d812294007533e26f06.
  All72base definitions except eval freezing are AST-identical; roottrain.py
  and prepare.py unchanged. Launch includes compiler BF16 matmul flags and
  every winning CLI override, no in-process public evaluation.
- Actual organizer static checker functions return zero warnings. Hardware
  contract/replay and exact-command cold-cache full-budget rehearsal pending.
- Authenticated upstream main c907da4c74bbaafeca7d13765c49610b7e77436c;
  README/prepare.py byte-identical. Formal rules exclude startup/compilation;
  actual five-step/capped900s contract is retained. No competition submission.
- Both hosts idle after125. Pre-test disk maintenance only retires profile
  files verified locally and against current GitHub asset digests. trn21 clean,
  free9,405,628,416B after499MB actual reclaimed (wave16 profile still has
  hardlinks elsewhere). trn22 maintenance pending. Never rerun one-off cleanup
  without checking its expected prior job and existing receipts.
- Campaign FP8 next design remains compiled cache preparation. Submission
  packaging is the active immediate task; no source127 deployment yet.

## Complete profile attribution — 2026-09-19 04:57 UTC

- Both source125 captures and ingestions COMPLETE and clean. trn21 ingestion
  finished04:52:25, free8,907,345,920B; trn22 finished04:52:24,
  free9,371,713,536B. No device job is currently active; both watchers are125.
  Native raw15files/108,307,982B; full raw21files/93,192,642B. Parquet native
  27files/63,518,360B, full26files/50,996,951B. All local/remote hashes match.
- Local source126 analysis now supports all annotated phases. All8native and
  16FP8 runtime dispatches match their binaries. Cache preparation has8extra
  FP8 dispatches; device intervals0.757307 vs0.112367ms. Micro0 backward Tensor
  activity/core0 drops4.059482→2.181706ms, while Scalar activity rises
  0.987601→2.436987ms and sync activity0.038757→0.815935ms. Different hosts and
  phase barriers make these attribution evidence, not speed certification.
- Next DESIGN: compile the unchanged cache producer as one graph, with exact
  metadata/byte/refresh gates and a compiled-native control; then fuse current
  calibration plus both weight orientations in NKI if justified. Gradient
  per-output calibration stays a separate later precision experiment. Nothing
  beyond source124 has earned a full-model FP8 or30minute quality trial.
- Detailed evidence: fresh-start-full-fp8-mlp-diagnosis-2026-09-19.md.
  Raw captures are now cloud verified; all54inventoried assets passed04:53:20.
  New parquet receipts are locally verified and discovered by the next backup.
  Source125/snapshot were verified04:49:51; latest analysis/notes await commit.
  Protected train.py/prepare.py hashes are unchanged. Public best0.99285131.

## Active profile captures — 2026-09-19 04:50 UTC

- Source1255029bd01c85202b2334c5aa8ae8d8fd3780cd1e9 deployed to both hosts
  with Git automatic maintenance disabled. Both preflights passed. trn21 runs
  `fs-w125-native-complete-mlp-component-profile-r1`, PID1334147; trn22 runs
  `fs-w125-full-complete-mlp-component-profile-r1`, PID1667900. Both started
  04:49:29, under900second bounds. Watchers repointed to125 at04:49:35.
- Capture one current-cache/two-distinct-microbatch update per arm, exporting
  every compiled/eager NEFF. Extra headroom checks before/after compilation.
  Numerical and speed authority remains source124 (same host, unprofiled).
  Once clean complete, preserve raw files immediately, then ingest/analyze.
  No full-model FP8 or quality trial is justified by the current numbers.
- Source125 and snapshot169429bd09b671bab25a6e0fe9173b3112568cc0 pushed and
  remote-ref verified04:49:51. Original failed postflights and clean maintenance
  receipts are committed; do not rerun either one-off maintenance script.

## Completed comparisons and clean recovery — 2026-09-19 04:48 UTC

- Source124 COMPLETE: all11 full FP8 MLP cases pass, including the actual
  16384x1024x4096 shape, both distinct microbatches, FP32 master accumulation,
  cache refresh/restoration, independent CPU quantizers and all4 pair-product
  bounds. Complete cache preparation +2FW/BW: native15.485907ms,
  FP8-forward/BF16-backward15.756617ms, all4backward FP8 products15.937871ms.
  The full FP8 component still loses2.9186%; no model or BPB promotion.
- Source123 three same-host phases COMPLETE: original501.265347ms,
  split-native684.669608ms, split-cached681.025798ms. All4 native postchecks
  pass per arm. Custom head saves0.5322% within its split scope, but loses
  35.8613% to the original complete step. The separate boundary fails the
  advancement gate. Original exact-monolithic numerical failures are retained.
- Both commands succeeded but their postflights failed only the8GiB disk
  reserve. Explicit maintenance reverified local files, cloud manifest/hash
  identities and remote hashes under the host locks. trn22 clean04:45:19,
  free8,369,889,280→9,690,304,512B; trn21 clean04:45:35,
  free8,103,407,616→9,180,405,760B. Original failed postflights are unchanged.
  trn22's earlier maintenance attempt stopped before mutation because an old
  raw-profile root had already been removed; its failed intent is preserved.
  Successful cleanup removes wave36 raw/ingestion copies +wave39 parquet on22,
  and already-preserved wave112 parquet on21. Checkpoints/caches/source remain.
- Source125 prepares SINGLE-arm native/full FP8 component captures on separate
  hosts. Independent waveform attribution only; source124 same-host timing is
  authoritative. Extra384/512MiB disk headroom is required before and after
  compilation, beyond the8GiB reserve. Binaries are exported before exit.
  No dispatch exists yet at this note's creation; inspect manifests/watchers.
- All52 heavy artifacts were reverified by the scheduled backup at04:35.
  Best public BPB remains0.9928513116311253. No seed hunt or ngrams were added.

## Complete FP8 component and split-head controls — 2026-09-19 04:34 UTC

- Source123 separate native/cached head gate COMPLETE/clean04:21:33:196 exact
  checks across D256/D1024 and two master updates. This proves equivalence only
  within the same separate region, not to the original monolithic model.
- trn21 original native phase COMPLETE/clean04:30:56, all4 postchecks pass:
  mean501.265347ms for30 timed updates. Split-native control ACTIVE since04:32:12,
  PID1326730, source123. Only after its clean/pass run split-cached request.
  Three arms separate boundary overhead from the kernel's contribution.
- trn22 source122 forward saved-byte probe COMPLETE/clean04:19:53, all10cases
  pass. Complete online cache preparation plus two producers:6.898546ms with
  repacking versus5.918992ms directly saving bytes (14.20% reduction). Ordinary
  forward5.793787ms does not supply the required backward inputs. No BPB claim.
- Source124 prototype composes FP8 forward and all four backward products with
  these saved bytes. Exact compiled/staged gradients plus independent CPU
  quantizers/pair bounds, two different microbatches and cache refresh precede
  timing. All current casts/calibration/transposes/packing and accumulation are
  charged. The VJP is explicitly new; no original-BF16 equivalence or quality.
  trn22 ACTIVE `fs-w124-fp8-complete-mlp-component-r1` since04:33:36,
  PID1661083, source34b6f24affcf03f421637a20bae62cda747af3e0. Both deployment
  and preflight passed; watcher repointed122→124 at04:33:49. No result yet.
- Source124/snapshot were explicitly pushed and remote-ref verified04:33:49:
  source34b6f24affcf03f421637a20bae62cda747af3e0;
  snapshotdffea9cd0dafec841be55478cbcba9cda3566299.
  All52 heavy assets were GitHub-digest verified04:10:10 with zero errors.
  Both host watchers continue123/124. Canonical best remains0.9928513116311253.

## Separate precision variant — 2026-09-19 04:20 UTC

- Source122 final-norm boundary also FAILS exact monolithic loss (9.5367e-7).
  Command1, clean04:17:44. No original-equivalence claim or quality promotion.
- Source123 adds a DISTINCT same-scope native reference gate for the original
  BF16 split head. All196 checks remain exact; report explicitly denies proven
  monolithic equivalence. Original native splitting alone already caused the
  source121 discrepancy. After passing, three matched phases must separate the
  compiler boundary from kernel gains, then quality requires a fresh timed run.
- trn22 source122 FP8 forward-byte gate active, watcher122; small cases pass,
  real-shape independent gates running. trn21 clean and ready for123 scope gate.
- Future remote fetch/worktree commands disable automatic Git maintenance per
  command. Wave122 trn22 fetch triggered background Git packing; a read-only
  check at04:18 found no git process/gc.pid while real-shape gates were still
  running, before timing. Do not run Git maintenance concurrently with phases.

## Next immutable source122 — 2026-09-19 04:16 UTC

- Source121 original-head diagnostic proves the graph boundary itself changes
  gradients: native/cached are exact within both scopes. FP32 carrier passes20
  component checks but FAILS the original GPT exact gate; no full phase allowed.
  Source122 moves final normalization into the separate head region, retaining
  the FP32 residual boundary and original norm/head arithmetic. Gate first.
- Source122 FP8 producer probe saves already computed activation bytes/scales;
  compare against actual forward plus repacking, charging online cache work and
  both forward producers. All original independent gates are retained. Current
  saved values and derivative are unchanged; no all-four-gradient claim.
- All52 checkpoint/profile artifacts GitHub-digest verified04:10:10, no errors;
  includes latest head parquet980,330,644B tar. Source/snapshot verified04:07:36;
  regular10minute source backup continues. Both hosts' latest121 jobs have clean
  postflights, including numerical failures. Check fresh manifests before launch.

## Boundary failure and next diagnostic — 2026-09-19 04:10 UTC

- Source120 split-head exact gate FAILED: same loss, token-embedding gradient
  difference2.861e-6; command1 but clean04:07:19. Do not run its phase requests.
  Next source121 diagnostic compares original native heads across the same
  boundary, then a separately gated FP32 carrier tests extra gradient rounding.
  Original196 exact GPT checks and failed evidence remain unchanged.
- Source119 per-row fused abs data backward COMPLETE/clean04:07:05: all11cases
  pass.5.654671ms versus original6.200740ms and native5.510375ms. No fullFP8 gain.
- Source120/snapshot pushed and remote-ref verified04:07:36. Artifact uploader
  continues under its lock. Both current hosts clean; source121 two independent
  bounded component/diagnostic requests are being prepared. Verify their actual
  dispatch and postflight before relying on this prose.

## Active trials — 2026-09-19 04:06 UTC

- trn22 source119 `3ae1861f48094429d842ac79a493c022ee846d02`, watcher119:
  `fs-w119-fp8-data-backward-fused-abs-component-r1` ACTIVE since04:05:03.
  Both per-row calibrations use fused abs/max. Eleven fixtures retain original
  stage equality and independent CPU bounds; actual timing charges cache work.
- trn21 source113 saved-softcap component completed clean. Source120 prepares
  a SINGLE separate compiled head region, leaving cached-head arithmetic alone.
  First require all196 GPT exact checks plus explicit compiled-boundary checks;
  only then same-source native/monolithic/separate full phases. Original model
  evaluation/checkpoint behavior and protected train.py remain unchanged.
- Caller diagnosis is recorded in fresh-start-head-caller-diagnosis-2026-09-19.md.
  The head's residency coincides with much slower native body matmuls. This is
  a hypothesis for the boundary experiment, not a proven allocator mechanism.
- Artifact uploader is already active under its lock at04:05; the attempted
  second uploader safely skipped. Latest source commit awaits the next source
  push; retain the existing regular backup schedule.

## Current dispatch update — 2026-09-19 04:04 UTC

- Both hosts have clean completed postflights; check live manifests before new
  dispatch. trn21 watcher113, trn22 watcher118. Source118 is pushed and tracked.
- Saved-softcap head passed all20 component/autograd checks;13.850659ms versus
  native18.219967ms (24.0% component gain), clean03:59:53. Not integrated into
  the model. The older cached head still loses37.791% whole-step.
- Head caller ingestion retry2 completed345.61s,32tables/1,076,930,574B;
  original command0 disk-reserve failure retained. Verified maintenance restored
  clean health03:57:40 after pruning only locally/cloud-backed-up raw/hardlink
  copies. Parquet retrieval matched all32 hashes. Recovery receipt is explicit,
  canonical-hash matched locally/remotely; original postflight is never rewritten.
- Local head caller analysis accounts for45 runtime dispatches plus the preserved
  startup barrier. Forward161.379941ms versus prior native68.001178ms;
  backward135.365241 versus138.211172ms. The forward reports14.26GB spill reloads,
  including6.52GB attributed to ordinary Linear.forward. These cross-host profiles
  locate a compiler/layout problem; only same-host unprofiled runs measure speed.
- Two-core FP8 gradient preparation passed18 cases and composition passed10:
  down1.547501ms/up1.695063ms versus native1.094661/1.074635ms. Still loses BF16;
  no full backward/autograd or training integration. Fused per-row magnitude/max
  is the next data-backward trial; retain all original independent stage checks.
- Public best remains0.9928513116311253. All51 raw/checkpoint artifacts were
  cloud-digest verified03:52:35; new head parquet is locally verified and pending
  the next artifact backup. No30-minute FP8 or head-quality run is authorized
  by these component numbers alone.

## New dispatch update — 2026-09-19 03:53 UTC

- trn22 source116 `b435826feb6c7cfd9316605e22a7a0e84fba1250`, watcher116,
  fused-amax gradient composition COMPLETE/clean; all10cases pass, up1.987077ms
  still loses native1.080190ms. All three source115 amax alternatives passed13
  exact cases including subnormals; fused abs is fastest (0.197/0.650ms).
  Source117 two-core calibration/recomputed row scaling prototype is prepared.
- trn21 source112, watcher112, caller ingestion retry2 ACTIVE with64GiB Go soft
  limit/4threads/GC trace. Retry1 was deliberately SIGQUIT-interrupted at440.58s
  after>300s empty outputs and22GiB RSS under24GiB limit. Exact UID/job marker/
  argv/cwd/start_ticks match is recorded. Stack shows processing workers in
  gcAssistAlloc; command2, raw hashes unchanged, postflight clean03:50:20.
  Retry1 stack/log/empty output preserved, no cache or source purge.
- Raw cached-head trace70files/2,384,899,172B is locally hash verified and now
  GitHub-digest verified as729,901,043B tar asset. All51heavy artifacts verified
  with no errors03:52:35. Once ingestion completes, preserve/analyze parquet;
  expect disk-reserve check and reclaim only backed-up redundant files if needed.
- Current public best remains0.9928513116311253; identity in best manifest.

## Dispatch state — 2026-09-19 03:40 UTC

- trn21: source112 `5abe6708fbc29ff9c1ec5f0d0ca2c9e58f96a5ec`, watcher112.
  `fs-w112-head-loss-cached-micro-device-profile-r1` active, started03:39:24.
  Complete cached-head component gained22.56%, but the matched model step
  regresses37.791%:689.956164 versus500.726543ms. All4native postchecks pass;
  last losses5.068909/5.064923 differ. No quality promotion. Profile caller
  allocation/spills/dispatch before redesign; root cause not established.
  Original candidate postflight command0/disk-reserve-only failure is retained.
  `fs-maintenance-preserved-wave89-trn21-20260919T033824Z` reverified local and
  cloud copies, removed only matching raw/parquet/staging files, restored clean
  host03:38:38; free8,356,802,560→12,027,482,112B. No shared cache purge.
  Source113 is also deployed. Saved-softcap20gate component request waits
  until this profile completes and has a clean postflight.
- trn22: source113 `bcd317f1c86bf4855659b7dc23e5f88492048a51`, watcher113.
  FP8 parallel-calibration experiment passed10cases and exact original scales;
  full down1.909484/up3.167645ms still lose native1.090103/1.108476ms.
  `fs-w113-fp8-weight-gradient-balanced-component-r1` is dispatched next;
  combined calibration/balanced request waits for its clean numerical result.
  These component gates do not establish full FP8 backward/autograd quality.
- Pack32 quality and larger public evaluation are COMPLETE. New matched best
  is0.9928513116311253. Source98/checkpoint/evaluation identity and the original
  training disk failure plus clean recovery are in fresh-start-best-public.json.
  No additional30minute run is justified solely by component speed.
- Source113 saved-softcap prototype saves exact FP32 tanh values instead of
  logits, removes backward tanh recomputation at unchanged tensor bytes. It
  remains untested on hardware; run its original20head checks before promotion.
- Latest source113 and snapshot were pushed/remote-ref-verified03:30:42.
  All50inventoried heavy artifacts were GitHub-digest-verified03:24:02, no errors.

## Current result and active work

**Current result, 2026-09-19 03:38 UTC:** matched larger public BPB is
**0.9928513116311253**, improved from0.9938106631483421. Dense6x1024, no ngrams,
seed42, T1024/pack32/GA1,131072global targets/update, padded exact optimizer
ownership, native-forward/custom-backward MLP, exact adaptive two-core CE and
clip-after-gradient-averaging. Native AdamW; no reciprocal approximation or FP8.
It completed3660updates /479,723,520targets in1784.6training seconds; charged
estimate1785.0 is within1800. Matched evaluation:20,971,520tokens,83,462,186bytes,
57,438,016nats, world1/B1/T1024/O1 without global autocast;702.43seconds, clean.
This is a public result, not a private leaderboard score. The training command
succeeded, but its original postflight failed the8GiB disk reserve; verified
redundant-profile cleanup restored clean health before evaluation. The original
failure remains recorded. Checkpoint SHA80fab9b63c935d54638cf3eb07d884c9b6e869550297bbcfe6b2f4f13991029d
is locally and GitHub-digest verified. Full identity: fresh-start-best-public.json.
Working target remains0.9749; the public improvement is0.0009593515172168221.

The CE change passed exact compiled-autograd gates and improved a same-source
whole step500.372595ms versus519.395419ms (3.6625%). Quality combines CE with
shorter context/packing and optimizer ownership; its gain is not attributed to
CE alone. The previous larger results0.9973302214 (T2048/pack8/ownership) and
0.9974687137 (T1024/pack16/original fused AdamW) remain reproducible checkpoints.
No seed sweep or new function approximation is in the winning run.

The packed FP8 forward/BF16 backward passes numerical and real-Linear/cache
integration gates but loses the complete model comparison: **587.956385 ms**
versus **499.384107 ms** native BF16, a **17.7363% regression**. Same immutable
source dcd6b80, same trn21 host, seed/data/geometry/ownership/adaptive CE,
5 warmup plus 30 timed updates. The two-microbatch compute phase adds85.0014 ms;
extra cache refresh adds2.52845 ms. All four native-evaluation postchecks pass,
and all12 FP8 cache counters equal35. Approximate training losses are finite
(5.067037 FP8 versus5.064923 BF16); numerical correctness is not a quality win.
The3% advancement gate fails: do not run a30-minute FP8 quality experiment yet.
See `run-logs/fs-w89-packed-fp8-whole-step-comparison-r1.json`.

Both wave89 caller captures and CPU ingestions are COMPLETE and clean. Raw
FP8 trace68files/2,539,181,724B; native70files/2,066,921,123B. Parquet respectively
32files/1,131,372,239B and32files/878,108,743B. All are locally hash-verified;
All47 inventoried heavy artifacts, including both raw profiles and their
parquet directories, are GitHub-digest-verified at02:13:37 with no errors.

Source96 caller analysis accounts for every device execution:33 FP8 or45 BF16
ordinary runtime dispatches plus one explicitly matched preserved c10d startup
barrier. The initial source94 count mismatch is retained, not silently ignored.
Forward109.520190ms FP8 versus68.001178ms native; backward137.589812 versus
138.211172ms. The largest source change is ordinary Linear.forward(train.py:2853):
reported spill reloads6,492,786,688B versus1,835,008B, and Tensor activity about
22.6ms versus7.9ms per core. These cross-host profiled intervals are attribution;
only the same-host unprofiled phase comparison establishes the17.7363% regression.
Missing dynamic DMA metadata forbids exact physical bandwidth claims. Read
`fresh-start-fp8-caller-diagnosis-2026-09-19.md` for the evidence and limitations.
Source96 has a REPORTING bug: integer division prints FP8 cost multiplier0
instead of0.5. Raw and adjusted FLOPs, timings, counters and identity matching
are unaffected. Both analysis scripts now cast to float64 before division;
corrected source97 analyses are COMPLETE, and original reports remain intact.
Native analysis clean02:10:35. Local FP8 reanalysis took15.05s and2,089,104KiB
peak RSS. All nonprecision graph values and ordered execution identities match
source96; only the FP8 ratio changes0 to0.5. Tied Arrow aggregate row ordering
is canonicalized. Checks: fs-w97-packed-fp8-caller-local-analysis-r3.checks.json.

Source95 d14298ea22ab33ff8be47afc9cba4a81ca8967ac separate compiled FP8 region
passes BOTH real-Linear widths256/1024, numerical/master-gradient/cache/update/
row-independence/native-restore checks. Clean02:00:13. No speed promotion yet.
Source95 paired phases COMPLETE and clean. Native499.641620ms, monolithic
FP8586.422830ms, separate-region FP8619.975017ms. Separate adds5.7215% versus
monolithic and loses24.0839% versus native. All three pass four postchecks and
FP8 cache counts are35x12. Last losses5.064923/5.067037/5.063094 differ; no
full-model bitwise-equivalence claim. Reject this boundary strategy for speed;
no FP8 quality run. It does not falsify the original source-level spill lead.
See fs-w95-fp8-compilation-boundary-comparison-r1.json.

Source98 `1da6c0714c0755fd3b078e972fbb0aefbb43bf6d` mode2 AdamW/ownership
composition passes24 exact same-math parameter/state/rank checks,13 total owner
table updates, cache/skip/schedule/restore and wrong-math rejection. Its whole
step is497.650549ms versus503.328809ms control (1.12814% gain), optimizer
23.619889 versus26.362835ms. Both clean/all four postchecks pass. Rank0 performs
70 NKI table updates. Below3%: retain for combination, no standalone quality.
**Pack32 phase complete clean02:42:22:**487.405374 versus503.328809ms control,
3.16362% gain. All four postchecks pass; same131072 global targets/update.
trn22 quality request `fs-w98-dense1024-context1024-pack32-owned-ce-1800-r1`
launched with source98, seed42, no ngrams, nativeMLP/ownership/adaptiveCE and
clip-after-reduce, changing only pack32/GA1. Watcher98. No BPB result yet.
Maintenance02:44:11 removed two verified redundant wave89 remote trace/parquet
directories under host lock, after102 local/remote file hashes and four GitHub
asset digests passed. Logical files total2,945,029,866B; observed free space
rose8,806,248,448 to9,684,447,232B. Ingestion hardlinks retain raw trace blocks;
local/GitHub files and remote staging remain. No cache/source/checkpoint purge.

Source99 `94fa40ae3c5b5b93ba27f8f53d318ff14b07afaa` head/CE forward gate FAILED
clean02:30:35 on confident-correct loss sign. All preceding projection/CE cases
pass; candidate/native/frozen-reference logits/loss/lse agree exactly. Tiny
negative CE appears in the original too (about-5.8e-8 versus tiny positive
FP64). The original nonnegative gate remains FAILED in the wave99 evidence. Source100
diagnostic completes clean02:36:09: all10 cases exactly match original CE.
The fused15*u+bias retains a multiplication residual against the rounded target
bias; at strength80 this is-5.96046e-8, yielding exp/denominator0.9999999404
and loss-5.7742e-8. Separately centered exp is1; runtime log(1) is positive.
Four cases reproduce negative original losses. Kernel math is unchanged.
Source101 corrects the incompatible sign assertion to exact reference
signs/values, retaining negative counts/minima and FP64 diagnostics. No equality
tolerance or projection bound changes. All14 fixtures pass, clean02:41:17;
actual-row gates also pass. Forward-only timing clean02:42:38: native two
forwards12.819835ms versus16.254153ms candidate including transpose and saves,
26.7891% slower. No backward or quality promotion. Prepared head device profile
separates native forward, weight transpose and fused forward; source102 pending.
trn21 clean, watcher101.



Source102 head profile clean02:47:40 and CPU ingestion clean02:49:12. Raw12files/
87,043,943B is local/hash-verified; parquet26tables retrieving. Profile-script
actual-shape native/candidate logits/loss/lse all exactly match. Local analysis maps all3 dispatches: native6.444392ms, transpose0.118419ms,
fused8.322044ms; weight-load GpSIMD source activity5.388/5.381ms. No populated
fused spills, sampled SBUF0.507. Both raw and45,027,599B parquet are local and
GitHub-digest-verified; all49 artifacts verified02:54:08. Source104 head layout
options are prepared: in-place softcap/exp after logits save; then full16MiB
BF16 weight cache. Gate separately; no forward/backward promotion.
Source103 FP8 weight-gradient gate PASS10cases, clean02:55:33. Component
down6.431302/native1.125327ms; up7.069405/native1.084679ms, clean02:58:04.
Ordinary real gradients differ2.65/2.73% RMS from BF16 surrogate. No promotion.
Grouped operand/PSUM layout prepared for another exact stage gate.
Source104 in-place head gates14fixtures/actual rows, but loses16.606820 versus
12.764331ms native two forwards, clean03:01:12. Cached-weight variant now
launched trn21, watcher104. Source103 implementation contract: row-scale transfer,
current global calibration, both byte preparations and long-token FP32 outputs.
Eight small/two real cases; no backward-autograd integration. Gate before timing.

Wave92 generic FP8 gradient GEMMs pass all9 independent cases. Repacked byte
mismatches are only107/103 signed-zero encodings under the ORIGINAL zero
contract; no changed nonzero value. Extra packing plus two products saves only
1.6518% for G@Wdown and0.5554% for dPre@Wup. Wave93 fused data backward passes
all9 cases and matches the separate FP8 algorithm exactly, but takes6.205545 ms
versus5.505396 ms BF16 (12.7175% slower). It improves over separate FP8 by9.5398%.
The17.350630 ms fused timing outlier remains in the raw samples. No backward
autograd integration, weight-gradient FP8, quality run, or promotion follows.

To recover trace headroom, maintenance01:37 reverified all17 local/remote
hashes and the GitHub digest receipt for the old wave5 raw profile before
removing ONLY its4,112,012,976-byte redundant remote copy. Local and GitHub
copies remain. Whole-host lock held; no device holders; no source/cache change.
trn22 free disk rose11,478,880,256 to15,590,940,672 bytes. Exact paths and hashes:
`run-logs/fs-maintenance-preserved-wave5-profile-20260919T013701Z.json`.

Wave89 wrapper rerun passes BOTH D256/D1024 all checks, cached/uncached gradients
and real updates, FP32 accumulation, cache freshness/nonmutation, FP8 training
future/batch/repeat/padding independence, exact native BF16 checkpoint restore.
Clean01:20:21. Process-local Dynamo reset BETWEEN independent model widths fixes
wave88's eight-specialization harness limit; no shared device purge or loosened
numerical check. The earlier failed attempt remains in the ledger.

Wave87 packed FP8 trace32files /499268795B is local/hash-verified. Parquet304690666B is local/hash-verified; analysis maps all70 dispatches.
Packed forward2.535229ms versus2.803872ms previous FP8; sampled SBUF peak
0.130903 versus0.364328; backward unchanged. Device trace is attribution only.
All43 inventoried heavy artifacts verified on GitHub01:26:21 with no errors.

Wave85 exact head-gradient-only CE complete step497.864910ms versus500.517682ms
control (0.5300% improvement), all four postchecks pass, last loss matches. Below
3% advancement gate: no quality run. Original failed full BF16 boundary remains
blocked. Wave86 native BF16 token-row component17.434858ms loses11.6268% versus
deployed contiguous HxT15.618884ms (view15.605782ms). Keep original BF16 control;
the same layout direction helped FP8, so the BF16 loss is not an FP8 veto.

FP8 wrapper uses quantized training and original BF16 evaluation. All calibration
and packing are charged in ordinary refresh; missing initial caches are charged
in forward. Short calls pad inside MLP to512 multiples, preserving per-token
independence. No quality-trainer change yet.

Wave83's six diagnostic cases establish the failed forward-boundary cause:
compiled original head logits retain FP32 accumulator precision. The BF16-input
candidate adds rounding. Exposing logits leaves both losses unchanged, and
frozen-input CE reproduces each result exactly. MLIR and diagnostic output are
local/hash-verified; the exact gate remains failed. Trace inputs and all41 inventoried heavy artifacts are offsite verified.

The installed CE/autograd gate passes36 eager/compiled exact loss/gradient
checks. Fresh same-source whole-step control519.3954189890064ms versus candidate
500.372594979126ms:3.6624936% incremental gain, above the predeclared3% gate.
All four ranks pass finite/causality/batch/repetition checks; final measured
loss5.064923286437988 matches exactly. No complete parameter-equality claim.
Combined gain versus earlier owned T2048/pack8 control528.3530275ms is5.2958%.
The phase clips before averaging; quality clips after. Protected trainer unchanged.

**Completed wave85 packed MLP:** all ten forward cases and eight
training cases pass, with exact packed-versus-old saves/gradients, independent
FP64 bounds and FP32 master accumulation. Packed complete update 15.957811 ms,
previous token-row FP8 16.442287 ms, native BF16 15.625856 ms. Packing improves
2.9465% versus the previous FP8 layout but remains 2.1244% slower than native.
No quality promotion. This small isolated loss does justify a bounded whole-
model allocation probe after wrapper/cache/prediction gates; it is not a veto.

The next bounded control is `mlp_native_token_rows_probe.py`: test whether the
same layout lesson also improves native BF16. It distinguishes the deployed
contiguous HxT save from the isolated benchmark's HxT view and a natural TxH
save/backward. Independent FP64 forward/backward bounds and exact master
accumulation precede timing. No full-model native-layout change is installed yet.

Wave82 profile raw 429,091,735 bytes and parquet 262,392,836 bytes are locally
hash-verified. All 57 dispatches map to their binaries. Same fixed-input FP8
forward: old backward causes 8,704 private compiler transposes and reported
64 MiB spill saves / 128 MiB reloads; matching token-row backward removes these
populated counters and transposes. Forward device time falls 3.568860 to
2.801243 ms; backward is essentially unchanged at about 4.84 ms. Trace times
are attribution, and missing dynamic DMA metadata prevents exact bandwidth
claims. Unprofiled wave81 timing remains the speed evidence.

Wave78 raw profile358627902B and parquet219879131B are local/hash-verified.
All44 runtime dispatches map to binaries. Device FW: oldFP83.686925ms, token
rows3.603983ms; BF16 backward remains about4.8ms. Token rows introduce8,704
Tensor transpose instructions in compiler private_nkl/transpose.py on core0,
and reported67,108,864B spill saves/134,217,728B reloads. The saved-ReLU layout
is a hypothesis, not yet proven as the source tensor. Profile warning about
dynamic DMA metadata prevents exact bandwidth claims. Analysis is
`research/run-logs/fs-w78-fp8-token-rows-profile-analysis-r1.json`.

The compact joint table passes hardware accuracy but loses 47.200750 ms versus
2.478831 ms Scalar composition (19.04x). Close this full-value/GpSIMD-gather
layout; full loss remains unimplemented. Its scalar-source frontend error was
resolved without changing any accuracy bound.

Actual FP8 up/down forward plus BF16 backward passed wave73 numerical gates,
but lost complete-component timing18.1811702vs15.6627304ms native. The wave74
profile is now local/hash-verified, ingested and registered for artifact backup.
All31 runtime dispatches map to ingested binaries. Analysis:
`research/run-logs/fs-w74-fp8-component-profile-analysis-r1.json`.
Forward device times: native2.63ms, NKI BF16 cache2.08ms, FP8 island3.68ms;
backward approximately4.8ms for all. Profile times are attribution, not new
unprofiled speed results. Reported SBUF peak drops0.7734→0.5080 with FP8;
Scalar/Vector activity rises sharply and Tensor transpose activity rises to
about0.98ms/core. Missing dynamic DMA metadata prevents exact bandwidth claims.
No populated spill counters for either NKI component does not prove zero spills
in a surrounding training graph. The profile is isolated, not full-model.

`fp8_mlp_token_rows_nki.py` is now gated and measured, not merely a proposal.
It keeps ReLU/hidden/output in token rows, omits the nonnegative hidden minimum
reduction, and preserves two inverse rescalings. The profile will determine
whether backward layout offsets a forward benefit. Full FP8 backward remains
separate. The next input-scale invariant contract is in the kernel queue.

## Latest numerical findings

FP8 online power-of-two calibration uses exponent bits, avoiding log2/floor/exp2
but charging amax reduction and cache refresh. Wave67 passes42 exact checks
across7 cases, including all finite BF16 codes in the declared calibration
range, zero/tiny/outlier/endpoints, repeat/idempotence/no mutation. The first
small MLP inference outputs and saved activations agree exactly between FP8
storage, BF16 storage and native. Training mode then exposed3 native output
changes near BF16 rounding boundaries, max0.0009765625; all sampled down-GEMM
FP64 bounds pass. CPU BF16 native input-gradient comparison fails9 pointwise
checks, RMS0.003536. All36 subsequent FP64-propagated backward records and all
paired NKI storage output/gradient checks pass across four cases. The original
CPU BF16 and native cross-compilation failures remain in the evidence. No
quantized-model equivalence or quality claim follows from these bounds.

CE wave68 tests96 variants: every mismatch has exactly one outer forward-loop
iteration (256rows/grid2), under both general-FP32 and BF16-valued input. All
repeated128-row blocks agree across physical cores. Backward using native lse
is exact for every case.512/768/1024-row variants are exact, as is the earlier
16384-row test. This localizes a forward shape/loop compilation effect; the
specific generated instruction is not identified. Do not attribute it to input
quantization or hardware-core variation. Adaptive dispatch now passes the fresh exact gates without changing tolerances;
compiled public-autograd integration and whole-model gates now pass; quality run is active.

Reciprocal-only Adam mode2 passes34 original component gates and full repeated
optimizer/cache/checkpoint-restore integration. Paired component1.085410 vs
1.304101ms (16.77% faster); no whole-step claim. Raw Scalar sqrt(mode3) and one
scaled-rsqrt Newton correction(mode4) fail original weight bounds. Seed-stage
probe measures0.00319824 maximum seed relative error,1.52741e-5 after one
correction,1.38269e-7 after two. Both stronger updates now pass all34 gates:
mode5 two corrections1.363799vs1.262366ms, mode6 quadratic correction
1.255049vs1.239520ms. Both lose timing; no integration. Mode2 remains preferred.
Standalone degree5/7 exp polynomials pass accuracy but are3.57–4.19times slower.

`compact_ce_table_study.py` enumerates65280 finite BF16 codes. Joint
exp(15*tanh(x/15)) and derivative tables with analytic tails pass CPU bounds:
max exponential relative error6.37e-8, derivative absolute error2.98e-8.
4/5/6MiB replicated-table designs and13 loss/gradient cases pass, including a
stable tiny-loss log1p path. CPU division/log are ideal FP32; device lookup primitive is now correct but19.04x slower than Scalar functions;
full loss reduction/reciprocal/log/backward remains unimplemented.
All13 user retro links and primary-source research are in the retro-numerics note.

See `fresh-start-kernel-queue-2026-09-18.md`, results ledger and source/log summaries.
Never veto an FP8 layout solely because its BF16 version lost: compressed storage
can change allocation/spills. The storage-only and actual-FP8-forward component variants both lose so far.
Profile their costs before choosing the next layout; full FP8 backward is separate.

## Operational protocol and backups

Canonical checkout trainium_frontier, branch research/fresh-start-20260918,
origin https://github.com/CsabaConsulting/trainium_frontier. Side directories
trainium_frontier_backups/2026-09-18/{trn21,trn22} are recovery snapshots without
Git databases, not forks to merge wholesale. Original dirty EC2 repos untouched.
Commit locally, bundle to immutable detached remote worktrees, return logs and
checkpoints, then commit/push. No automatic host-source merge.

Local Devuan/SystemV, usercsaba cron; no root/systemd:
- Every10minutes: source/recovery backup with alternate Git index and atomic
  fast-forward pushes. Skips merge/rebase/cherry-pick/index locks; no pull,
  force push, merge or active HEAD/index mutation. Source03a9d902e10212e671e4918a261d1b71cf4e87e1 and recovery
  fbd7f48a0f1eb7f573abc72e1ecc629c46b40de3 were verified01:10:40. Check newer
  exact refs in out/fresh-start-backup/{latest,last-success}.json.
- Minutes5/35: bounded digest-verified heavy-artifact upload to draft release
  research-backup-20260918. The01:05 backup verified all41 artifacts, including wave70 checkpoint and
  wave82 raw/parquet plus wave83 compiler inputs, without errors. State is
  out/fresh-start-artifact-backup/latest.json; receipts under research/run-logs.
  Newly produced remote checkpoints still need retrieval to checkpoints/ first.
- Local log watchers currently follow trn21/wave99 and trn22/wave98,90rounds at
 45seconds. Repoint when the active remote worktree changes. Only stop exact
  owned watcher PID/UID/cwd/argv/startticks; remote training jobs are separate.

SSH key /home/csaba/.ssh/trn2.pem, BatchModeyes, ConnectTimeout12:
trn21 ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com
trn22 ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com
Venv /home/ubuntu/workspace/native_venv; native PyTorch2.11, NKI0.5, LNC2,
4logical/8physical cores. Common remote Git directory is wave4. Component jobs
use visible logical0; full training/phase jobs use4ranks without a core override.
Keep LNC2; prior host-wide LNC1 attempts hung. A kernel[1] launch inside LNC2 is
allowed and works. No package/SDK changes, provisioning, shared-cache purges,
driver resets, broad process kills or submissions. No private validation access.

Use research/fresh_start_launch.py request.json. The experiment guard holds
/home/ubuntu/.cache/trainium_freshstart/device0.lock across pre/postflight,
records immutable source, environment, boot health and disk/device holders,
and cleans only exact owned sameUID processes with matching markers/startticks.
Every failure receives postflight. Fields are command_returncode/host_clean/
failures, not passed. A dispatch receipt is not a completion. Unknown work is
not killed. Disk must stay above8GiB; trn22 had about17GiB after wave66 evaluation.

Protected train.py SHA256 bc1d0b75e9964c2cccf27865d22ef287f7032984be4b10a8f8490b95e2ed1e7c.
Protected prepare.py SHA2565807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb.
Training flags --optlevel=1 --auto-cast matmult --auto-cast-type bf16.
Matched evaluation uses --optlevel=1 only. No loader/scorer/tokenizer/timing changes.

Owner optimizer requires explicit rank-local state for optimizer resume; full
model checkpoints remain ordinary parameters. Padded singleton Muon groups pass
exact native/state/cache/restore tests. Do not combine ownership with NKI AdamW
until its separate integration gate passes; the trainer currently rejects it.

Closed kernel variants and profiling details are in the MLP spill, norm,
embedding and optimizer notes. Smaller source-level buffers do not establish
smaller caller allocation. Full-graph profiling remains mandatory before a
component win can be treated as a whole-training speedup.

Source104 cached head forward COMPLETE clean03:02:25. All14 fixtures and actual-row
gates pass. Native two forwards12.787697ms versus8.443401ms including transpose
and FP32 logits saves (33.97247% forward-only gain). Full backward/master-gradient
and caller integration remain required; no training speed or BPB promotion.
Source105 grouped FP8 weight-gradient gate FAILED clean03:04 on small up shape
[512,512,256];65520 coordinate bound failures and0.500913 relative RMS. Seven
small down cases pass. Suspected narrow PSUM-bank alias: width256 allocation
places two independent accumulators in the same512-element bank. Source106
changes only the accumulator bank stride, retaining original pair/error gates.
All10cases now PASS, including exact original FP8 products; clean03:09:16.
Source106 grouped component timing launched, watcher106. Source107
b65d61e4ff8ada778ae4c84e0c2a509410dca17e cached-head autograd probe is deployed
trn21, request ready after106 finishes. Source/backup refs verified GitHub03:08:42.
Matched pack32 larger-evaluation request is prepared under out/; launch ONLY
after clean quality completion and checkpoint preservation.
trn22 pack32 quality remains active on source98, about0.49s/update.

Source106 grouped component COMPLETE clean03:10:51: down1.919147/native1.095259ms;
up3.177483/native1.093426ms. Exact10case gate passes. No promotion.
Source107 cached-head autograd gate launched trn21, watcher107.
FP8 per-stage cost diagnostic is prepared locally; source108 pending.

Source107 all14small head loss/dx/master gradients exactly match original.
Accumulation harness fails on unsupported retain_graph=True with AOT donated
buffers (clean). Correct harness to ordinary backward, matching trainer; no
compiler setting or equality changes. Source109 rerun prepared; source108 FP8
component stage-cost diagnostic now active trn21, watcher108.

Source108 stage costs COMPLETE clean03:13:33. Up calibration1.176563ms and
prepared product1.471031/native1.074664ms; down calibration0.344010ms and
product0.856747/native1.085396ms. Preparation erases down's product advantage.
Source109 head exact-autograd/component rerun active trn21, watcher109.
Source110 per-instance GPT training adapter/full caller gate prepared; do not
run phases before autograd and caller gates pass.

Source109 head autograd COMPLETE clean03:15:40.142026:20exact comparisons,
real16384x1024head FW/BW14.095440ms versus18.202054native (22.5613% gain),
every transpose charged. Source110 actual GPT/all-master-gradient gate launched
trn21, watcher110. Pack32 quality should finish around03:17–03:19; preserve
checkpoint after clean postflight then launch prepared matched larger eval.
Source111 FP8 parallel-tile calibration and balanced8-bank grouping are prepared
for separate controlled stage/component tests after head work.

Source110 GPT gate fails clean while compiling the ORIGINAL model: test omitted
COMPILE_SDPA_DIRECT=True and selected a compiler-disabled attention wrapper.
Source112 aligns gate configuration with the production native SDPA/QK flags;
no head kernel or equality change. Source111 was not deployed; deploy112 bundle
from known remote110. FP8 controlled variants also available in112.
Pack32 quality finishes3660updates/479.7Mtargets, small public0.987784; await
clean postflight, preservation and larger matched evaluation before promotion.
