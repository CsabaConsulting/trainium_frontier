# Resume the Trainium kernel campaign

Updated 2026-09-19, 00:33 UTC. Authorization: danger-full-access, approval never.
Use current manifests/postflights before dispatch; do not restart completed jobs.

## Current result and active work

**Current result, 2026-09-19 00:21 UTC:** matched larger public BPB is
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

**trn21 CLEAN after wave80:** BF16 CE memory boundary passes all72 exact
eager/compiled comparisons. Complete CE+casts10.071585ms adaptiveFP32 versus
6.233179ms BF16. The new head/caller adapter is prepared; commit/deploy wave82
and run the real Linear/cache/head gradient gate before the prepared phase
control and candidate. Whole-step gate remains3%. Watcher follows wave80.

The installed CE/autograd gate passes36 eager/compiled exact loss/gradient
checks. Fresh same-source whole-step control519.3954189890064ms versus candidate
500.372594979126ms:3.6624936% incremental gain, above the predeclared3% gate.
All four ranks pass finite/causality/batch/repetition checks; final measured
loss5.064923286437988 matches exactly. No complete parameter-equality claim.
Combined gain versus earlier owned T2048/pack8 control528.3530275ms is5.2958%.
The phase clips before averaging; quality clips after. Protected trainer unchanged.

**trn22 CLEAN after wave81:** token-row backward passes all small/real
FP64 surrogate gradient bounds and accumulated-master checks. Complete
component16.515157ms versus17.920212ms previous backward with same forward
(7.84% gain), but15.675626ms native (5.36% slower). No training promotion.
Next wave82 profile pairs both backward orientations and the same fixed-input
forward to check compiler transpose/spill removal. Watcher follows wave81.

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
  force push, merge or active HEAD/index mutation. Source0eae485 and recovery
  259e0491cfeb1f392bd19d713b29c0a7f315d42b were verified00:00:42. Check newer
  exact refs in out/fresh-start-backup/{latest,last-success}.json.
- Minutes5/35: bounded digest-verified heavy-artifact upload to draft release
  research-backup-20260918. The00:05 backup planned35 artifacts including the new FP8 raw trace/parquet;
  inspect its completion. The new wave70 checkpoint arrived after that inventory. State is
  out/fresh-start-artifact-backup/latest.json; receipts under research/run-logs.
  Newly produced remote checkpoints still need retrieval to checkpoints/ first.
- Local log watchers currently follow trn21/wave80 and trn22/wave81,90rounds at
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
