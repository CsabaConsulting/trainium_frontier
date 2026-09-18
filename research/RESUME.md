# Resume the Trainium kernel campaign

Updated 2026-09-18, 21:12 UTC. Active permissions: danger-full-access, approval never.
This replaces the accumulated stale resume instructions; their history remains
in Git. Do not restart completed experiments from earlier revisions.

## Live work

- **trn21:** wave50 native-MLP/pack8 full-budget run finished cleanly: 3,183
  updates, 417.2M targets, 1,784.7s charged. Small public .983219/T2048 and
  .995565/T1024 are not larger-eval or private scores. Its checkpoint is local
  and GitHub-digest-verified (SHA9399bc62561cec110856424701a40fc628db48a63bca37006cb033544fec6289).
  `fs-w57-eval-native-mlp-pack8-s1024-20m-o1-r1` completed cleanly:
  **1.0010172426032236 BPB**, 20,971,520tokens, 694.054s, same settings and
  83,462,186byte denominator as the previous best. This is a small public gain,
  not sub1.0 or a private-leaderboard score. Wave60 native FP32 pack/all-gather/
  copyback passes all four ranks and costs6.403158ms worst-rank median.
  Wave61 CPU/Gloo ownership is exact through cache/restore on all ranks. Native
  fails only singleton tall Muon batches (matrices58/59 on ranks3/0); all other
  parameters/states and cross-rank byte equality pass. Both postflights clean.
  Wave62 prepares zero-matrix padding of singleton groups, keeping exact gates.
  No owner-update speedup has been measured.
- **trn22:** wave58 Scalar square+reduction norm passed but loses
  (.543885ms vs native .440652ms); neither normalization variant is promoted.
  AdamW functional component passes all34 original cases and saves69.277%.
  Wave58 proves the original iteration6 weight bound omitted propagated moment
  rounding: native and CPU FP32 reproduce the flagged values. Kernel/adapter
  unchanged. Wave59 introduces independently checked moment stages and a weight
  stage conditioned on actual stored moments, with unchanged constants and
  whole-update/trajectory RMS gates. CPU counterexamples and negative controls
  must pass, then the full integration/cache/restore gate before full-model timing.
  Wave59 failed before integration because the two selected cancellation values
  are not a full-table RMS sample. Wave60 corrects that fixture and **passes the
  complete integration/cache/skip/restore gate**, clean postflight.
  `fs-w60-native-mlp-pack8-adam-fused-phase-r1` completes551.895444ms median,
  1.205076% below558.627329ms, all four postchecks and clean postflight. Below3%
  gate; no30minute AdamW-only run. New `fs-w60-native-mlp-context1024-pack16-phase-r1`
  is active from21:09:59UTC, sourceefaebf1. Same16384tokens/microstep and GA2,
  shorter attention context, native AdamW. Check quality separately if it wins.
  See fresh-start-adamw-rounding-2026-09-18.md for the corrected contract.
  The same-host30-step native-MLP/pack8 control is558.627329ms median, clean.
- Preflight backup false positive is fixed by exact-script recognition, with
  five passing regression checks. Device-holder and runtime checks are intact.
- Next implementation target: fused FP32 AdamW. Current full-step trace has
  56AdamW NEFF dispatches before16Muon dispatches; do not sum overlapping driver
  waits as device time. Compare NKI and pure-tensor compiled controls against
  actual foreach math, including state/bias/LR and all copy costs.
- The wave49 native full-step runtime trace is now local and hash-verified:
  88files/68,923,863bytes, `out/fresh-start-profiles/` plus its label directory.
  The 20:05 artifact backup verified its GitHub asset. A trace summary and
  optimizer dispatch attribution are in run-logs/.
- The runner's explicit repository-local --entrypoint is source-fingerprinted.
  Production train.py and prepare.py remain unchanged by these experiments.
- Separated packed MLP688.922656ms and native-forward separated control
  700.449689ms both lose, despite passing the isolated numerical gates and
  all four whole-model checks. Do not train either separated variant.
- Native references inwave48/out/fs-w48-mlp-isolated-reference-r1 are preserved
  locally underout/fresh-start-profiles/ and cloud-digest-verified. All48
  comparisons pass unchanged tolerance; all real-shape outputs are exact,
  while small-shape outputs are within bounds, not all exact.
- The native comparison's old failure came from switching to eager execution
  after another model compiled in the same process. See the boundary-rounding
  note. Separate-process gates fix the comparison without changing derivatives
  or tolerances.
- Layer5-only packed MLP814.094ms loses even to all-six678.042ms; all checks clean.
  Longer combined native-MLP+embedding pair remains588.186/571.188ms,2.889898%,
  below3%. That combination was not advanced; the active quality run tests the
  separately gated native-MLP + pack8 candidate.

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
  `458d1f8e3faf161f5b2aef58538d2aff4a0e7b39` at21:06, with both refs verified.
- Minutes5/35: `research/fresh_start_artifact_backup_cron.sh`, own lock, nice10,
  bounded uploader, digest-verified immutable assets in draft release
  `research-backup-20260918`. The latest manual cycle completed at20:40:19UTC: **31of31
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
