# Resume the Trainium kernel campaign

Updated 2026-09-19 02:09 UTC. Authorization: danger-full-access, approval never.
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
raw archives have GitHub digest receipts. The02:05 heavy-artifact backup is
processing47 artifacts including these two new parquet directories. Check its
latest receipt before claiming all47 are uploaded.

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
corrected source97 analyses are planned, and the original reports remain intact.

Source95 d14298ea22ab33ff8be47afc9cba4a81ca8967ac separate compiled FP8 region
passes BOTH real-Linear widths256/1024, numerical/master-gradient/cache/update/
row-independence/native-restore checks. Clean02:00:13. No speed promotion yet.
**trn21 ACTIVE:** `fs-w95-native-owned-fp8-control-phase-r1`, started02:07:45.
Then run `fs-w95-packed-fp8-monolithic-control-phase-r1` and
`fs-w95-packed-fp8-separate-candidate-phase-r1` sequentially, only after each
clean postflight. Immutable source95 worktree, same real data/seed/geometry,
5 warmup +30 timed updates. Requests are in out/. This isolates the compilation
boundary; math and caches are unchanged. trn22 is clean after caller analysis.
Watchers currently trn21/wave95 and trn22/wave96. Repoint only exact owned
watchers when moving worktrees; never interrupt a running experiment.

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
- Local log watchers currently follow trn21/wave95 and trn22/wave96,90rounds at
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
