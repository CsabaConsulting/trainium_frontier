# Resume the Trainium kernel campaign

Updated 2026-09-18, 22:37 UTC. Active permissions: danger-full-access, approval never.
This replaces the accumulated stale resume instructions; their history remains
in Git. Do not restart completed experiments from earlier revisions.

## Live work

- **Latest live override,22:37:** trn21 is evaluating the completed ownership
  checkpoint under label `fs-w65-eval-owned-pack8-s1024-20m-o1-r1`, source5dcbf6a,
  remote wave65, started22:33:25. Training finished3377updates/442.6Mtargets,
 1784.8seconds/1785.2charged estimate, clean, small BPB0.979318/T2048 and
 0.992072/T1024. Checkpoint469834849bytes preserved locally at22:33:18, SHA256
 a8a2c91e1886681d03996f7930710c944d1b0fd71e7c6a7936f35bd343745743.
  GitHub artifact cycle22:37:08 verified33/33 assets with zero errors, including
  this checkpoint. Do not restart training.
- **trn22 current source dc2373b/wave67:** online exponent-bit calibration and
  FP8 conversion pass42 exact gates across7 cases. Storage-only MLP inference
  agrees exactly between NKI BF16/FP8 caches and native on the first small case.
  The job stopped on a native inference-versus-training exactness difference
 0.0009765625, before timing. New diagnostic code preserves this failure and
  inspects independently summed dots, paired NKI training outputs and gradients.
  Do not bypass it or claim a complete MLP pass.
- Adam mode5(two rsqrt Newton corrections) passes all34 original gates but loses
  paired component timing:1.363799 vs1.262366ms. Mode6(quadratic correction) also
  passes34 gates but loses:1.255049vs1.239520ms. Both postflights are clean;
  neither is promoted, and reciprocal-only mode2 remains the math winner.
  Next source adds `softcap_ce_shape_diagnostic.py`, repeating128-row blocks
  across256/512/768/1024 rows and FP32/BF16-valued inputs, plus one-core gather.
  This distinguishes a shape/loop effect from a physical-core difference.
  Local log watchers were replaced at22:32:57 for90rounds/45seconds on both
  current waves. Remote jobs were unchanged. Commitdc2373b and recoveryf7fc797
  were GitHub-ref-verified22:31:45. The older bullets below retain earlier
  milestone detail; completed wave65 training must not be treated as active.

- **trn22 latest completed:** `fs-w66-eval-context1024-pack16-adam-s1024-20m-o1-r1`,
  sourcea1625d9, remote wave66, clean22:24:04UTC. Protected public evaluation:
 20,971,520tokens, B1/T1024/world1/O1, no global autocast. **0.9974687136975224
  BPB**,57,705,140nats/83,462,186bytes,705.573659s. First sub1.0 on this larger
  public check; not a private score or the competitive target.
  Training wave63 finished clean22:04:07:3282updates/430.2Mtargets, small public
  BPB0.992175, exact causality check. Dense6x1024, no n-grams, native-forward/
  custom-backward MLP, T1024/pack16/GA2, original fused AdamW mode0, seed42,
  clip-after-reduce and unchanged131072targets/update. Its checkpoint is local
  and GitHub-digest-verified (32/32 artifact cycle22:11:55). SHA256
 9eabd33e6925052091f0238629f9192096c42f1310cf3127286898621932cd27.
  It contains none of the new approximations. Do not label the small score as
  the larger/public or private leaderboard score.
- **trn21 active:** `fs-w65-dense1024-native-mlp-pack8-owned-1800-r1`,
  source5dcbf6a, remote wave65, started21:52:43UTC. Padded optimizer ownership,
  native-MLP, T2048/pack8/GA2, seed42, clip-after-reduce and131072targets/update.
  Exact native optimizer/cache/restore/schedule gates and matched full-model
  phase passed: **526.782934 vs554.873134ms (5.06245%)**,30timed/5warmups,
  all four model checks and clean postflights. Control/candidate last loss and
  six attention-weight norms match; this is not a checksum of every weight.
  No new arithmetic approximations or NKI AdamW combination. Await full quality,
  preserve checkpoint and compare larger public eval when justified.
- **trn22 CE probe also complete:** `fs-w66-softcap-ce-parallel-gather-gate-r1`.
  One-core mask port agrees exactly;12.2223ms vs original12.3398ms FW+BW.
  Both two-core arms agree exactly at real16384x8192 BF16-valued logits but
  fail the256-row general-FP32 stress cases, loss differences up to about4e-4.
  They were excluded from timing, not promoted. Diagnose dtype/loop-length/
  stage differences and add the missing one-core gather arm. Keep strict gates.
  Host environment remains LNC2; no host-wide LNC1 retry.
- **Completed wave66 diagnostics:** reciprocal mode2 integration passes all
  repeated optimizer/None+zero+scalar fallback/real cache/CPU checkpoint restore
  checks. The stagewise rsqrt diagnostic measures raw scaled Scalar seed error
 0.00319824, one correction1.52741e-5, two corrections1.38269e-7. The discrepancy
  from FP64 recurrence is only1.50e-7; seed accuracy explains the earlier failure.
  Both jobs have clean postflights. Do not rerun them as if pending.
- **Local next-wave code:** Adam modes5(two Newton corrections) and6(quadratic
  inverse-root correction) need the original34 device gates and paired mode0
  timing. FP8 weight-storage/BF16-compute control and online exponent-bit scale
  calibration and its independent probe driver are implemented as prototypes.
  Syntax checks pass; device gates are pending. Neither is a hardware result
  or active training change. Start calibration-only before the full MLP probe.
  See `fresh-start-kernel-queue-2026-09-18.md` for the full precision-specific
  investigation and compact-table/composed-function queue. A BF16 timing loss
  is not a veto on a layout whose FP8 storage fits on chip differently.
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
  and wave65 adds a paired mode0 NKI control: **1.085410 vs1.304101ms**,16.77%
  component gain, all34gates pass. No whole-optimizer/step claim or promotion.
  Wave65 mode4 fails12 large-gradient weights, moments/finite checks pass;
  one rsqrt Newton step is insufficient there. Wave66 diagnosed the seed across
 786432 values; preserve the failures and unchanged thresholds for modes5/6.
- Wave65 polynomial exp degree5/7 both pass boundary, exp0, repeat and FP64
  accuracy gates. Real8M maxrelative errors1.366e-5/1.183e-7 vsScalar1.134e-5.
  **They lose timing:** .446690/.524650ms vsScalar.125238ms, including HBM.
  Do not replace standalone exp or run training with it. Retain as a precise
  diagnostic and evidence that hardware already approximates this function.
  Composed functions, layout/traffic and engine balance remain separate questions.
- Best larger public result is now **0.9974687136975224 BPB** from wave63,
  20,971,520tokens/B1/T1024/O1,83,462,186bytes, clean (details above). The prior
  wave50 result1.0010172426032236 and its checkpoint are retained; its SHA is
 9399bc62561cec110856424701a40fc628db48a63bca37006cb033544fec6289,
  local and GitHub-digest-verified. Wave50 completed3183updates/417.2Mtargets,
  1784.7s charged. Its small-public .983219/T2048 is not a larger/private score.
- Source and receipts are committed regularly; cron pushes canonical research
  and recovery refs every10minutes. Sourcea1625d9 and WIP recoveryfb95f39 were
  verified22:20:32; verify last-success for later exact hashes. Latest artifact
  cycle22:11:55 verified32/32 assets with zero errors. No automatic host-source
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
**0.9974687136975224**,20,971,520tokens/B1/T1024 from wave63's native-MLP/T1024/
pack16/original-fused-AdamW checkpoint; this is not the private leaderboard.
Working target approximately0.9749.
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
  `a1625d936b8650e8ec61e4c13d303a9af7d4085d` is the verified22:20 source;
  inspect last-success to verify the latest exact hash before reporting it.
- Minutes5/35: `research/fresh_start_artifact_backup_cron.sh`, own lock, nice10,
  bounded uploader, digest-verified immutable assets in draft release
  `research-backup-20260918`. The latest cycle completed at22:11:55UTC: **32of32
  artifacts verified, zero errors**, including the new wave63 checkpoint.
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
