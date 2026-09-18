# Resume the Trainium kernel campaign

Updated2026-09-18,23:10UTC. Authorization: danger-full-access, approval never.
Use current manifests/postflights before dispatch; do not restart completed jobs.

## Current result and active work

Best matched larger public score: **0.9973302213667847 BPB**, not a private
leaderboard result. Working competitive target remains about0.9749. No seed sweep.
Dense6x1024,117453580parameters, no n-grams, BF16 branch normalization, NKI causal
conv, native-forward/custom-backward MLP, T2048/pack8/GA2, padded exact optimizer
ownership, seed42, clip after gradient averaging. It completed3377updates/
442.6Mtargets in1784.8training seconds; charged estimate1785.2, clean/within budget.
Small public scores0.979318/T2048 and0.992072/T1024 are separate diagnostics.

Larger evaluation `fs-w65-eval-owned-pack8-s1024-20m-o1-r1` is COMPLETE:
20,971,520tokens,83,462,186bytes,57,697,128nats, world1/B1/T1024/O1 without global
autocast, protected scorer. Checkpoint local and GitHub-digest-verified:
`checkpoints/fs-w65-dense1024-native-mlp-pack8-owned-1800-r1/final.pt`,469834849B,
SHA256 a8a2c91e1886681d03996f7930710c944d1b0fd71e7c6a7936f35bd343745743.
The previous T1024/pack16/original-fused-AdamW result is0.9974687136975224,
3282updates/430.2Mtargets. Its checkpoint SHA is
9eabd33e6925052091f0238629f9192096c42f1310cf3127286898621932cd27 and is also
local/cloud verified. Earlier native pack8 result1.0010172426032236 is retained.

**Both hosts are clean after wave69/sourcef3a67c9e28028150cab07751ba89d29981448975.**
trn21 adaptive/gather CE passes original exact gates, shape boundaries and row
independence.16384x8192 FW+BW is5.679064700962044ms versus12.339531350880861ms
original (53.9767% component gain). Next source adds a scratch adapter and
compiled installed-autograd gate before whole-model timing. It preserves the
original loss, reduction and checkpoint interfaces; train.py remains untouched.

trn22 FP8-storage/BF16-math full gate passes under explicit FP64-propagated
bounds and exact paired NKI outputs/gradients. It loses isolated timing:
17.898401897400618ms versus14.989582099951804ms NKI BF16 storage and
15.641983901150525ms native original per cache refresh plus two FW/BW calls.
No actual FP8 matmuls, optimizer, collectives or caller spill measurements.
Full FP8 forward/backward islands remain open; no training promotion here.

The T1024/pack16+ownership phase finished clean at515.4737504781224ms versus
fresh owned T2048/pack8 control528.353027504636ms:2.4376%, below predeclared3%.
All four model checks pass. Combine gated CE before another30-minute quality
run. These phases clip before averaging; real quality recipes clip after.
Current source changes include adapter/check, phase flag and explicit training
flag, all default off. Commit/deploy before any launch; requests in ignored out/.

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
compiled public-autograd integration and whole-model measurement are next.

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
stable tiny-loss log1p path. CPU division/log are ideal FP32; device lookup,
reduction, reciprocal/log error, backward and timing are still untested.
All13 user retro links and primary-source research are in the retro-numerics note.

See `fresh-start-kernel-queue-2026-09-18.md`, results ledger and source/log summaries.
Never veto an FP8 layout solely because its BF16 version lost: compressed storage
can change allocation/spills. Current test isolates that capacity effect with
BF16 math. Full FP8 forward/backward islands remain a distinct queued direction.

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
  force push, merge or active HEAD/index mutation. Sourcef3a67c9 and recovery
  0fe8ae883de1c4b2947ba9113399698f98237ca6 were verified23:00:30. Check newer
  exact refs in out/fresh-start-backup/{latest,last-success}.json.
- Minutes5/35: bounded digest-verified heavy-artifact upload to draft release
  research-backup-20260918. Latest completed23:05:18:33/33 artifacts verified,
  zero errors, including both new quality checkpoints. State is
  out/fresh-start-artifact-backup/latest.json; receipts under research/run-logs.
  Newly produced remote checkpoints still need retrieval to checkpoints/ first.
- Local log watchers currently follow trn21/wave69 and trn22/wave69,90rounds at
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
