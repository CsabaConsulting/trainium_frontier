# Restart the Trainium kernel campaign here

Updated2026-09-18, after the local GUI crash; recovery began17:07UTC.

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
