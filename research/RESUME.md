# Resume the Trainium kernel campaign

Updated 2026-09-18, 21:48 UTC. Active permissions: danger-full-access, approval never.
This replaces the accumulated stale resume instructions; their history remains
in Git. Do not restart completed experiments from earlier revisions.

## Live work

- **trn22 active:** `fs-w63-dense1024-native-mlp-context1024-pack16-adam-1800-r1`,
  source18b416d, remote wave63, started21:25:56UTC. Dense6x1024, no n-grams,
  native-forward/custom-backward MLP, T1024/pack16/GA2, original fused FP32
  AdamW mode0, seed42, clip-after-reduce, unchanged131072targets/update.
  Predeclared phase gate passed:540.481049 vs558.627329ms (3.24837%). Training
  is finite near0.54s/update. Wait for the real1800s result and postflight;
  preserve checkpoint locally/cloud, then evaluate at the same20M/T1024/O1
  public settings if promising. It contains none of the new approximations.
- **trn21 active:** `fs-w65-adamw-scaled-rsqrt-newton-gate-r1`, source5dcbf6a,
  remote wave65. Mode4 uses a power-of-two-scaled Scalar rsqrt seed and Newton
  refinement, plus refined Scalar reciprocal. Original34 numerical gates and
  a same-job mode0 NKI timing control. Inspect completion before launching more.
- **Next trn21 quality run:** padded optimizer ownership with the wave50 geometry
  (T2048/pack8/GA2) and clip-after-reduce. It has passed exact CPU/Gloo and native
  optimizer/cache/restore/schedule gates and the matched full-model phase:
  **526.782934 vs554.873134ms (5.06245%)**,30timed/5warmups, all four model checks,
  clean postflight. Control/candidate last loss and six attention-weight norms
  also match exactly. These checks are not a checksum of every full-model weight.
  The wrapper flag `--experimental-optimizer-ownership` installs singleton
  padding. Combine with `--experimental-native-mlp`; do not combine with NKI
  AdamW yet. Preserve seed42, loader, global batch, scorer and timing hooks.
- Ownership wave61 failure was batch-one tall Muon numerics, fixed by appending
  an independent zero matrix. Wave62 passed28 exact native records, including
 63extra timing/warmup updates. Full optimizer inclFP32 exchange25.35–25.38ms
  versus52.29–52.84ms. Dummy parameter/grad/state remain zero. All owner states
  need explicit rank-local checkpoints for optimizer resume; ordinary full model
  checkpoint format remains unchanged. See ownership note and run summaries.
- User steering explicitly covers **all useful arithmetic approximations and
  function compositions**, not only sqrt or tables. Read
  `fresh-start-retro-numerics-2026-09-18.md`. All13 provided links attempted;
  primary Doom/Quake source, author papers/blogs, SLEEF/Sollya and AWS docs read.
  Raw web data stays in ignored out/retro-approximations-20260918.
- Wave64 raw Scalar sqrt + refined reciprocal failed one8M-table first-step
  weight at g=-7.8730639785e-9, the same value that exposed the native sqrt error.
  Moments/finite checks pass; no tolerance relaxation. Reciprocal-only mode2
  **passes all34 gates**, functional component1.188021ms vsforeach4.968162ms,
  but that run lacks a paired mode0 NKI control. No integration/promotion yet.
- Wave65 polynomial exp degree5/7 both pass boundary, exp0, repeat and FP64
  accuracy gates. Real8M maxrelative errors1.366e-5/1.183e-7 vsScalar1.134e-5.
  **They lose timing:** .446690/.524650ms vsScalar.125238ms, including HBM.
  Do not replace standalone exp or run training with it. Retain as a precise
  diagnostic and evidence that hardware already approximates this function.
  Composed functions, layout/traffic and engine balance remain separate questions.
- Best larger public result remains **1.0010172426032236 BPB** from wave50,
  20,971,520tokens/B1/T1024/O1,83,462,186bytes, clean. The checkpoint SHA is
 9399bc62561cec110856424701a40fc628db48a63bca37006cb033544fec6289,
  local and GitHub-digest-verified. Wave50 completed3183updates/417.2Mtargets,
  1784.7s charged. Its small-public .983219/T2048 is not a larger/private score.
- Source and receipts are committed regularly; cron pushes canonical research
  and recovery refs every10minutes. Source commits through92b8ad6 were available
  by21:40; verify last-success for the latest exact pushed hash. Latest artifact
  cycle21:35:18 verified31/31 assets with zero errors. No automatic host-source
  merges. Latest detached sources differ by host by design.
- Whole-model MLP fusion variants, standalone branch norms and separate FP8
  projections have recorded losses. Do not restart closed variants blindly.
  Current native MLP with custom backward passed real-shape boundary/cache gates;
  old same-process equivalence failures were compiler fallback artifacts.

Use each `.dispatch.json`, `.job.json`, `.log`, `.postflight.json` under
`research/run-logs/` for exact launch state. Requests are local under `out/`.
A launch receipt is not a completion; postflight uses `command_returncode` and
`failures`, not a `passed` field. Phase8steps/3warmups may compile for4–6minutes.

## Current evidence and decisions

Reference dense6x1024,117,453,580parameters, no hashed n-grams, NKI causal conv
and BF16 normalization outputs. Best larger public BPB is
**1.0010172426032236**,20,971,520tokens/B1/T1024 from wave50's native-MLP/pack8
checkpoint; this is not the private leaderboard. Working target approximately0.9749.
No seed sweep. Previous wave16 result1.0015512989000874 is retained; its
checkpoint469,834,849bytes is local and cloud-digest-verified:
`checkpoints/fs-w16-dense1024-nki-norm-clipafter-1800-r1/final.pt`, SHA256
`f5c32a4a61bd5adff168e6a74e0f85da8b2f9580b6d89a3348f069ebf5fd7c06`.
It completed3,005updates; larger evaluation label
`fs-w17-eval-dense1024-clipafter-s1024-20m-o1-r1`.

Current same-host native full-step medians590.315111 ms/trn21 and599.574587 ms/
trn22. Grouped MLP665.456528 ms, tile256686.589837 ms, streaming-down673.016550 ms,
streaming-both693.390713 ms and prepacked678.041824 ms all lose despite correctness.
All have clean postflights. Prepacked isolated FW/BW3.541410 versus native4.229880 ms
excludes0.287952 ms layout refresh; full-step includes it. No variant is promoted.

Old fused full-forward capture58.761415 versus native36.514939 ms. Recorded spill
reload counters2.340GB native versus7.265GB fused.5.386GB is on a shared Linear
source line, not established as the LM head. Penguin groups span most of graph.
Sparse counters/dropped DMA metadata prevent complete traffic or exclusive
latency attribution. TE transpose time rises despite less adjusted work.
Smaller declared source-level buffers have not proven smaller caller allocation.
See `fresh-start-mlp-spill-investigation-2026-09-18.md` and wave40/42 analysis JSON.

Embedding three-kernel DMA gains only1.42% in the full model; two-kernel follow-up
has no component advantage. Separate FP8 projection probes lose after layout/
cast costs. Whole-island FP8 remains open after diagnosing graph integration.
Require numerical gates and at least3% total-step gain before a new30minute run.

## Continuity and off-machine backups

Canonical checkout `trainium_frontier`, branch `research/fresh-start-20260918`,
origin `https://github.com/CsabaConsulting/trainium_frontier`.
The sibling `trainium_frontier_backups/2026-09-18/{trn21,trn22}` are recovery
snapshots without Git databases, not forks to merge wholesale. Original dirty
EC2 repositories remain untouched. Source is committed locally, transferred as
bundles to immutable detached worktrees, then logs/artifacts return here.

Local OS **Devuan/SystemV**. Usercsaba cron, no sudo/root, no systemd:
- Every10minutes: `research/fresh_start_backup_cron.sh`, separate Git index and
  recovery branch `backup/fresh-start-20260918`, atomically pushes source+WIP and
  verifies both refs. Skips merge/rebase/cherry-pick/index locks, does not pull,
  merge, force-push or change active HEAD/index. Latest verified source
  `92b8ad63ee58ea134249a1e7a44122616297f2e8` is the expected21:40 source;
  inspect last-success to verify the latest exact hash before reporting it.
- Minutes5/35: `research/fresh_start_artifact_backup_cron.sh`, own lock, nice10,
  bounded uploader, digest-verified immutable assets in draft release
  `research-backup-20260918`. The latest cycle completed at21:35:18UTC: **31of31
  artifacts verified, zero errors**, including the new wave50 checkpoint.
  All queued
  checkpoints/profiles have actual GitHub assets and matching digests. New ignored
  artifacts still require discovery/retrieval before a future backup receipt.

States `out/fresh-start-backup/{latest,last-success}.json` and
`out/fresh-start-artifact-backup/latest.json`; check process/lock if status stale.
Receipts `run-logs/fs-artifact-backup-*.json`. Actual download/restore check of
original trn21 snapshot verified1,550files. See `continuity-and-backups.md`.

## Runtime, protocol and artifacts

SSH key `/home/csaba/.ssh/trn2.pem`, BatchModeyes, ConnectTimeout12:
- trn21: ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com
- trn22: ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com
Venv `/home/ubuntu/workspace/native_venv`, native PyTorch2.11, NKI0.5, LNC2,
4ranks/8physical cores. Common Git directory is remote wave4. No package/SDK
changes, broad kills, cache purges, driver resets, provisioning or submissions.
Protected prepare/tokenizer/loader/scorer remain unchanged. No private-data access.

Use `fresh_start_launch.py request.json`. Protocolv3 (`experiment-protocol.md`)
holds `/home/ubuntu/.cache/trainium_freshstart/device0.lock` across pre/post,
records immutable source/environment/boot health and cleans only owned sameUID
processes with matching run marker and unchanged startticks. Unknown work is
never killed. Every failure gets postflight. Remote disk capacity is monitored;
trn22 had about28GiBfree before wave43. Read-only SSH does not require a device job.

`fresh_start_sync.py --host trn21 --remote /home/ubuntu/trainium_freshstart_20260918_waveNN --prefix fs-wNN- --rounds 1`
(and trn22). Zero registry rows for probes is normal; inspect raw logs/postflights.
Environment: NKI_VENV_PATH=venv, NEURON_LOGICAL_NC_CONFIG=2,
NEURON_CC_FLAGS='--optlevel=1 --auto-cast matmult --auto-cast-type bf16'.
Component visible core0; full4rank requests leave visible/num-core overrides out.

Fused forward raw profile: local `out/fresh-start-profiles/fs-w36-dense1024-fused-mlp-forward-device-r1`.
Ingested32tables299,505,407bytes under local `out/fresh-start-profiles/fs-w39-mlp-forward-parquet-r2/`;
remote wave39/out/fs-w39-mlp-forward-parquet-r2/i-0673e54d9d66e6a1e_pid_1335537_nc_0_session_0.
Baseline tables local `out/fresh-start-profiles/dense1024-nki-norm-current-parquet`,
remote trn22 wave40/out/dense1024-nki-norm-current-parquet. Use local out/fresh-start-tools/profile-analysis-venv/bin/python (Python3.12,
pyarrow25.0.0) for bounded analysis. Avoid memory-heavy local re-ingestion. Explorer is
`/opt/aws/neuron/bin/neuron-explorer`, not venv/bin.
