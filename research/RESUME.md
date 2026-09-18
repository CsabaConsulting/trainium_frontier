# Resume the Trainium kernel campaign

Updated2026-09-18,19:00UTC. Active permissions: danger-full-access, approval never.
This replaces the accumulated stale resume instructions; their history remains
in Git. Do not restart completed experiments from earlier revisions.

## Live work

- **trn21:** `fs-w44-dense1024-native-control-phase-r2` is running after the
  native-forward/custom-autograd phase completed574.075248 ms with all checks
  and cleanup passing. Earlier native590.315111 ms makes this2.751% faster,
  short of3%. Sourceaec170ff810424a98841a59ee5ffe7c5f9362351, remote wave44.
  After the fresh control, the prepared request
  `out/fs-w44-dense1024-native-mlp-embedding-phase-r1.request.json` combines
  this formulation with the previously validated three-kernel embedding path.
  Check clean postflight before launching, and require3% full-step gain.
- **trn22:** wave45 separated MLP gate failed on changed microbatch1/small shape.
  Native forward inside the same separated region fails identically:9 input
  gradient elements exceed the unchanged bound, relative RMS0.003586663.
  Both jobs cleaned automatically. No separate-region full-model job launched.
  Latest deployed worktree wave45/sourcefa9ca234fb0ef41e612c90d75e51675e79e0e333.
- **Next:** deploy newly committed wave46 diagnostic to trn22 and run
  `scratch/mlp_separate_diagnose.py`. Keep calls uninstrumented; compare repeated
  base/changed/zero inputs, CPU FP64 GEMMs with explicit BF16 rounding and other
  rounding hypotheses. It records failures, never relaxes the existing gate and
  cannot authorize promotion on its own. Inspect whether repeats are exact.
- Wave43 explicit7.125MiB SBUF phase finished705.225310 ms versus599.574587 ms;
  rejected, clean. Wave42 prepacked phase678.041824 ms also rejected, clean.
- Source scripts: `scratch/mlp_separate_diagnose.py`,
  `scratch/mlp_fused_integration.py`, `scratch/mlp_fused_integration_check.py`,
  `scratch/phase_timer.py`. Production train.py is unchanged.

Use each `.dispatch.json`, `.job.json`, `.log`, `.postflight.json` under
`research/run-logs/` for exact launch state. Requests are local under `out/`.
A launch receipt is not a completion; postflight uses `command_returncode` and
`failures`, not a `passed` field. Phase8steps/3warmups may compile for4–6minutes.

## Current evidence and decisions

Reference dense6x1024,117,453,580parameters, no hashed n-grams, NKI causal conv
and BF16 normalization outputs. Best larger public BPB is
**1.0015512989000874**,20,971,520tokens/B1/T1024; this is not the private
leaderboard. Working target approximately0.9749. No seed sweep, no new training
claim. Wave16 best checkpoint469,834,849bytes is local and cloud-digest-verified:
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
  `aec170ff810424a98841a59ee5ffe7c5f9362351` at18:50, with both refs verified.
- Minutes5/35: `research/fresh_start_artifact_backup_cron.sh`, own lock, nice10,
  bounded uploader, digest-verified immutable assets in draft release
  `research-backup-20260918`.18:05 uploader finished25of26items at18:46. A detached retry started18:52
  using the committed fix for owned starter placeholders with declared nonzero
  size. It passed the previous failure and is reuploading the old wave4 archive.
  The archive is still unverified; do not describe it as backed up. Do not kill
  the active uploader. Receiptfs-artifact-uploader-restart-20260918T1852.json.

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
remote trn22 wave40/out/dense1024-nki-norm-current-parquet. LocalPy3.14 lacks
pyarrow; remote venv has it. Avoid memory-heavy local re-ingestion. Explorer is
`/opt/aws/neuron/bin/neuron-explorer`, not venv/bin.
