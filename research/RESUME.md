# Restart the Trainium kernel campaign here

Update2026-09-18 after the full-access restart: actual active permissions are
`danger-full-access`, approval `never`. The saved checkpoint was recovered cleanly.
Wave40 spill analysis and both grouped/tile256 integration gates completed
cleanly. Recorded spill reloads rise2.340GB to7.265GB, mostly in surrounding
Linear ops. Nearby native controls completed at590.315ms/trn21 and599.575ms/trn22.
Grouped MLP is665.457ms,12.73percent slower, with all postchecks/cleanup passing.
Tile256 completed686.590ms,14.51percent slower; clean and not promoted.
Both streaming modes passed component gates. Both streamed variants passed actual integration but lost full-chip timing:
673.017ms/stream-down and693.391ms/stream-both. Clean postchecks and cleanup;
neither is promoted. Wave42 prepacked mode passes component and actual cache
gates, and now runs full-chip phase on trn21. Trn22 has completed the CPU-only
Linear spill attribution refinement and is available for wave43 fixed-SBUF gates. Wave41 source7886eab; worktree
`/home/ubuntu/trainium_freshstart_20260918_wave41`. All labels are fs-w41-*. See `fresh-start-mlp-spill-investigation-2026-09-18.md`.
The integration now exposes token tiles128/256/512 and grouped I/O as scratch
options; production `train.py` is unchanged. No new training result is claimed.
Use `fs-w40-*.dispatch.json` and `fs-w41-*.dispatch.json` for exact launch state.
Wave42 prepares a lossless prepacked-weight cache: same gradients go through
ordinary weights, while auxiliary packed inputs receive no duplicate gradient.
Packing is charged once per optimizer update in full-step timing. Wave42 component and actual-cache gates passed. Full-model measurement is
pending; mode3 forward/backward3.541410ms versus native4.229880ms excludes
0.287952ms of separately measured per-MLP layout preparation. Wave43 adds a
separate explicitly allocated7.125MiB SBUF kernel, reusing five buffers and
sequential outer loops. It is not yet validated.
The bulk artifact uploader did not survive the restart; the scheduled artifact
backup resumed at18:05UTC and is uploading the remaining queue. One interrupted
old trace chunk is still unverified; inspect its GitHub state before retrying. Do not interpret
its stale `latest.json` current item as proof that a process is still running.


Updated2026-09-18 at17:52UTC. The user is choosing a deliberate restart to
activate full-access/no-prompts mode. This is a safe research checkpoint.

## Current state for the next session

- Both Trn2 hosts are idle after completed, automatically cleaned wave39 jobs.
  Both latest worktrees are `/home/ubuntu/trainium_freshstart_20260918_wave39`,
  source `fb6f0b719524dc6b00ed77b7058ca694707e104e`. No training job needs
  interruption or relaunch. Still check the host lock/status before new work.
- All source history is now on GitHub, branch `research/fresh-start-20260918`.
  A separate `backup/fresh-start-20260918` preserves WIP and remote evidence.
  Cron has actually run successfully at17:40 and17:50, with both refs verified.
  See `continuity-and-backups.md`, `continuity.json`, and
  `out/fresh-start-backup/{latest,last-success}.json`.
- The local machine is **Devuan/SystemV**, not systemd. User `csaba`'s crontab
  has two entries: Git recovery every10minutes; artifact backup at minutes5/35.
  No sudo/root shell was used. Backups skip merges/rebases/cherry-picks/index
  locks, use a separate Git index, preserve staged additions and staged.patch,
  never merge/pull/force-push, and leave the active branch/index alone. A real
  conflicted-merge test passed. Do not replace this with systemd instructions.
- The best checkpoint, both original EC2 snapshots, the full MLP forward raw
  capture, baseline raw capture and extra NEFFs, and the early runtime trace
  already have actual GitHub draft-release assets with matching SHA256 digests.
  Draft tag `research-backup-20260918`; receipts `fs-artifact-backup-*.json`.
  A real download/restore check verified all1550 files of the trn21 snapshot.
- A bulk artifact uploader started17:47UTC, command
  `python3 research/fresh_start_artifact_backup.py`. At17:52 it had verified
  7 of24 queued artifacts and was processing the older5.1GB wave4 archive.
  Check `out/fresh-start-artifact-backup/latest.json` and its lock; do not
  assume it died with the GUI or start a concurrent upload. Cron resumes the
  queue if interrupted. Only completed digest receipts mean off-laptop backup;
  the remaining archive/checkpoint queue is not all uploaded yet. Newly retrieved
  wave37/wave39 profiles will be discovered on the next artifact cycle.
- User-requested defaults were saved in `/home/csaba/.codex/config.toml`:
  `approval_policy="never"`, `sandbox_mode="danger-full-access"`. Previous
  config is `/home/csaba/.codex/config.toml.before-full-access-20260918`.
  Installed CLI0.155.0 accepts `--yolo` (verified with `codex --yolo --version`).
  The old active GUI session still had managed permissions. After restart,
  inspect the actual supplied permission policy; never assume a config edit
  changed permissions of a running session.

## Immediate scientific resume point

The interrupted full-model MLP trace **has now ingested successfully** on trn22
with installed `/opt/aws/neuron/bin/neuron-explorer`, GOMEMLIMIT24GiB,95.28s.
The first retry used an incorrect venv executable path and failed before
ingestion; its failure and clean postflight are preserved. Successful retry:
`fs-w39-mlp-forward-remote-ingest-r2`. All32 Parquet files (299,505,407 bytes)
are locally hash-verified under
`out/fresh-start-profiles/fs-w39-mlp-forward-parquet-r2/` and remain remotely in
wave39's corresponding `out/` directory. There are879,720 instruction rows.

The initial summary is
`run-logs/fs-w39-mlp-forward-profile-analysis-r1.analysis.json`.
Fused forward graph `6ec5dd5b3e70cf667b3952ff25804a5c.neff` takes58.761415ms
in this device capture; the baseline forward
`e496117f74af27424b6c982ea962b625.neff` took36.514939ms. Fused per-core Tensor
active~27.37ms versus baseline~18.8ms; Scalar24.41/21.19ms versus12.33/9.44;
GpSIMD10.75/9.32ms versus6.21/4.78. One leading DMA instruction explicitly has
`SpillSave` in its name. **This is a lead, not a quantified spill attribution.**
Do not add overlapping engine/instruction durations or claim total bandwidth:
the trace warns about dropped DMA packets and missing dynamic DMA metadata.

Next: compare `Instruction.spill_save_bytes`, `spill_reload_bytes`, source lines,
layer/BIR names and graph windows with the baseline, and inspect descriptor
patterns. Local baseline tables are
`out/fresh-start-profiles/dense1024-nki-norm-current-parquet/` (11 files), with
summary `fs-w16-dense1024-nki-norm-device-micro-r1.device-summary.json`.
The fresh local system Python3.14 lacks pyarrow; the remote native venv has it.
Analyze on a clean remote host or use a known existing analysis environment;
do not reinstall device packages or retry the memory-heavy local ingestion.
If spill pressure is confirmed, test reduced SBUF footprint/streamed weight
tiles or smaller token tiles against the full-step control. Grouped MLP I/O
remains only an isolated component win, not an integrated training improvement.

The resumed two-kernel embedding passed all11 numerical cases, including every
FP64 scan position, instrumented/uninstrumented eager and compiled composition,
native controls and zero/repeat. Training-row median0.902076ms prepared and
1.734378ms with metadata, versus native3.150630ms. The existing three-kernel
version was0.855589ms prepared and1.303211ms with metadata. These are different
runs; the new version has no promotion-worthy component advantage. No full
training run follows it. Label `fs-w39-embedding-permute-scan-component-r1`.
The corrected-metadata small trace from wave37 is also now locally verified:
11 files/420,881 bytes under its matching `out/fresh-start-profiles/` directory.

The rest of this note records the original crash context and invariants; where
it describes then-pending recovery, the completed status above takes precedence.

The canonical checkout is `trainium_frontier`, branch
`research/fresh-start-20260918`, remote `origin` =
`https://github.com/CsabaConsulting/trainium_frontier`. The two sibling directories
`../trainium_frontier_backups/2026-09-18/{trn21,trn22}` are recovery snapshots of
the old EC2 directories, without Git object databases. They are not active forks
to merge wholesale. New source is committed here and sent as a Git bundle to
immutable remote worktrees. Run records and artifacts flow back here.

## Authorization and invariants

The user authorizes long-running research on the two existing Trn2 instances,
custom kernels, retrieval and commits, and now **regular pushes to their GitHub
fork**. Preserve failures and negative results. No seed sweep; no speculative
full training before correctness and full-step speed gates. Do not submit to the
competition, provision hosts, change installed drivers/packages, overwrite active
source, or broadly kill processes. Follow `experiment-protocol.md` for automatic
preflight, owned-process cleanup, host lock and postflight on every device job.
Prepare/tokenizer/loader/scorer remain unchanged. Native PyTorch/NKI0.5, LNC2.

## Crash recovery findings

- Both hosts are reachable and idle. The wave37 embedding profile and wave38
  four-rank MLP probe finished before the GUI crash; returncode0, healthy
  automatic postflights, no surviving experiment workers or cleanup actions.
- Local source history through `052c83966456e37335dfe42d14fa016e4290944f`
  survived (68 commits beyond origin/main). GitHub still had only the older
  branches at recovery time. The immediate action is to commit the recovered
  work and push this research branch, followed by scheduled backups.
- The last local edit survived: `scratch/embedding_permute_scan_nki.py` and the
  `--fused-permute-scan` probe arm. This two-kernel prototype fuses wide indirect
  permutation plus scan, retaining fast DMA expansion. **Hardware validation
  pending; it is not a promoted model change.**
- The 1,020,828,335-byte wave36 full-model MLP forward capture is locally
  hash-verified. Local Explorer ingestion ended with timeout124 at15:39:32,
  leaving partial Parquet files (Instruction.parquet is zero bytes). Do not
  treat that output as a successful profile analysis. Raw files are intact.

## Current scientific position

Best preserved dense6x1024 model has no hashed n-grams. The larger public
evaluation is **1.0015512989000874 BPB**, B1/T1024,20,971,520 tokens. This is not
the private leaderboard score; do not substitute the smaller T2048 score0.981174.
Training run `fs-w16-dense1024-nki-norm-clipafter-1800-r1` performed3005 updates.
Checkpoint: `checkpoints/fs-w16-dense1024-nki-norm-clipafter-1800-r1/final.pt`,
469,834,849 bytes, SHA256
`f5c32a4a61bd5adff168e6a74e0f85da8b2f9580b6d89a3348f069ebf5fd7c06`.
Larger eval: `fs-w17-eval-dense1024-clipafter-s1024-20m-o1-r1`.

The MLP kernel is numerically correct and faster in isolated forward/backward,
but its full-model integration regresses from586.46 to681.97ms per step.
Runtime attribution places the extra time in forward (61.35 vs38.73ms), not
backward. Wave38 now confirms four concurrent ranks retain the component gain:
native forward/backward4.70–4.74ms, fused4.38–4.39ms. Ordinary chip contention
therefore does not explain the large integration regression. Investigate caller
graph layout and SBUF spills in the preserved full forward device trace.

Grouped MLP I/O improves isolated forward/backward to3.65 vs4.22ms, but is not
yet integrated or proven in the model. Embedding three-kernel DMA is0.856ms
isolated versus native3.13ms, but the complete step improves only1.42%
(578.15 vs586.46ms), missing the570ms promotion gate. Fully fused embedding
metadata repair improved31.48 to3.88ms, still slower than three kernels. This
motivates the unvalidated two-kernel design above. No new30-minute run is yet
justified by these candidates.

## Exact resume points

1. Finish regular GitHub backups and preserve the old EC2 snapshots and important
   ignored artifacts off the laptop. A local commit or SHA receipt alone does
   not protect against SSD failure. Keep ordinary Git commits separate from
   automatic WIP recovery snapshots; never let a scheduler alter the active
   branch/index or overwrite an experiment's source.
2. Retrieve the completed wave37 small profile from trn21 and hash-verify it.
   Remote worktree `/home/ubuntu/trainium_freshstart_20260918_wave37`, source
   `a1729bcdd6362a095cd2e6d8b2cadfcc4fd356a8`, label
   `fs-w37-embedding-contiguous-small-device-profile-r1`.
3. Ingest the large MLP capture on a host with enough memory, bounded and separate
   from device experiments; the laptop attempt timed out. Raw local directory:
   `out/fresh-start-profiles/fs-w36-dense1024-fused-mlp-forward-device-r1`.
   Matching remote source is trn22wave36, commit55383db. Preserve all originals.
   Use Explorer2.32, `--ingest-only`, unique output, NEFFs beside NTFFs, cwd at
   staging; never re-use the incomplete output as evidence. Inspect spill
   counters, NKI source locations and DMA descriptors, not just aggregate time.
4. Validate the two-kernel embedding prototype through all11 existing cases:
   `research/scratch/embedding_segmented_dma_probe.py --device neuron
   --real-shape --training-row --tiled-io --fused-permute-scan`. Use the launcher,
   new committed worktree/unique label and exact source/environment, then compare
   every FP64 scan position, instrumented/uninstrumented calls, compiled
   composition, zero/repeat and native controls before interpreting timings.

SSH identities and host mapping are in `fresh_start_sync.py`; key remains outside
the repository at `/home/csaba/.ssh/trn2.pem`. Both use
`/home/ubuntu/workspace/native_venv`. Last deployed worktrees: trn21wave37 above,
trn22wave38 source052c839. Remote common Git directory is wave4; original dirty
`/home/ubuntu/trainium_frontier` is preserved and untouched. No active jobs at
the recovery inspection. Each subsequent launch must update this resume note
or the machine-readable continuity status before depending on chat memory.

Detailed history: `fresh-start-results-2026-09-18.md`,
`fresh-start-deep-research-2026-09-18.md`,
`fresh-start-kernel-primary-research-2026-09-18.md`,
`fresh-start-preservation-2026-09-18.md`, and `run-logs/fs-*` receipts. Raw logs
and manifests are authoritative; a completed component job is not a promoted
training improvement.
