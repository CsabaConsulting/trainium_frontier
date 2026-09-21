# Source197 evidence pack

Purpose: offline attribution of the large position-dependent library-MLP timing effect. This is
analysis evidence only; no profile latency is a new speed claim and this pack authorizes no machine
operation.

## Immutable comparison

- Source commit: `282935ca9b1e30f6c21c397143770b03034363f9`
- Host used for both captures: `trn22`
- Four ranks, `NEURON_LOGICAL_NC_CONFIG=2`, seed 42, sequence length 1024, pack factor 32
- Native controls mean: 499.108 ms and 498.149 ms
- Library replacement in layer 0 only: 496.617 ms
- Library replacement in layer 5 only: 682.661 ms
- Library replacement in all six layers: 564.512 ms

All variants retain original BF16 custom backward in every layer. The custom library replacement
count and shape are held constant; only layer position changes.

## Captures and integrity

| Evidence | Label | Status | Ingestion |
|---|---|---|---|
| First-layer profile | `fs-w197-nkilib-partial-first-profile-r1` | complete, clean | `fs-w197-nkilib-partial-first-ingest-r2` — 31 Parquet tables |
| Last-layer profile | `fs-w197-nkilib-partial-last-profile-r1` | complete, clean | `fs-w197-nkilib-partial-last-ingest-r1` — 32 Parquet tables |

The continuation sequence is complete at `2026-09-20T12:57:39Z`; its source, postflight, and log
hashes are in `research/run-logs/fs-w197-nkilib-partial-profile-sequence-r2.sequence.json` on the
fresh-start worktree. Raw profile artifacts and derived Parquet were hash-preserved locally and in
cloud backups. Do not restart `r1`; the older sequence metadata's `running` marker is stale.

## What the analyst must do

1. Compare dispatch and compiled binary/NEFF identity between first- and last-layer captures.
2. Identify which ordinary compiled regions surround the custom MLP call in each case.
3. Separate proven dispatch differences from inferences. Dropped DMA notifications mean exact
   traffic/bandwidth claims are not allowed.
4. State whether the evidence supports implementing the saved ROW-FP8 backward path, or whether a
   caller-boundary fix should be investigated first.

## Primary files

All following paths are under `.worktrees/fresh-start/`:

- `research/HANDOFF-CLAUDE-2026-09-20.md`
- `research/RESUME.md`
- `research/fresh-start-library-followup-2026-09-20.md`
- `research/fresh-start-library-fusion-next-2026-09-20.md`
- `research/run-logs/fs-w197-nkilib-partial-profile-sequence-r2.sequence.json`
- `research/run-logs/fs-w197-nkilib-partial-first-profile-r1.request.json`
- `research/run-logs/fs-w197-nkilib-partial-last-profile-r1.request.json`
- `research/run-logs/fs-w197-nkilib-partial-first-ingest-r2.log`
- `research/run-logs/fs-w197-nkilib-partial-last-ingest-r1.log`

The profile records reference the exact remote Parquet paths and artifact hashes. Retrieve or query
them only through the established evidence-preservation workflow, after a fresh read-only inventory.
