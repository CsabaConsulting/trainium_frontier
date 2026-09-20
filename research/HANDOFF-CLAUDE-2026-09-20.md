# Trainium research handoff to Claude Code

Verified 2026-09-20 at 21:53 UTC (14:53 PDT). The user is considering transferring
control. Both Trainium2 hosts are idle; no experiment coordinator or profile
preservation worker remains active. No further experiments were dispatched while
preparing this handoff. A new owner should check current ownership before starting
work, then continue under the existing experiment protocol.

## Start here

Canonical checkout:
`/home/csaba/repos/AWS/TrainiumFrontierCompetition/trainium_frontier`

Working branch: `research/fresh-start-20260918`.
Push destination: `origin`, `https://github.com/CsabaConsulting/trainium_frontier`.
Do not merge historical EC2 snapshots over this checkout. The original side
clones are recovery evidence; new experiments use committed, immutable remote
worktrees. Read [RESUME.md](RESUME.md),
[experiment-protocol.md](experiment-protocol.md), and
[continuity-and-backups.md](continuity-and-backups.md).

The machine-readable handoff receipt is
[fs-handoff-claude-20260920-r1.json](run-logs/fs-handoff-claude-20260920-r1.json).
It records direct SSH observations, available host locks, last clean postflights,
seven independently revalidated experiment results, original manifest/log hashes,
profile preservation receipts, and the active backup schedules.

## Host ownership and completed work

| Host | Last completed queue | Queue finished (UTC) | Current state |
| --- | --- | --- | --- |
| trn21 | `fs-w199-nkilib-saved-contention-sequence-r1` | 12:49:53 | Idle, last postflight clean, lock available |
| trn22 | `fs-w197-nkilib-partial-profile-sequence-r2` | 12:57:39 | Idle, last postflight clean, lock available |

No training, compilation, profiling processes, or `/dev/neuron0` holders were
found on either host. Free root space is about 16.01 GB and 15.14 GB respectively;
available RAM is about 131.5 GB on each. These are observations, not a substitute
for the next experiment's preflight or a capacity estimate for a large capture.

SSH uses `/home/csaba/.ssh/trn2.pem`, `BatchMode=yes`, `ConnectTimeout=12`:

- trn21: `ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com`
- trn22: `ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com`

See `research/fresh_start_sync.py` for the exact reusable SSH configuration.
Remote native Python is `/home/ubuntu/workspace/native_venv/bin/python`.
Keep the installed native PyTorch/NKI environment and LNC2; four ranks for model
experiments. Do not apply XLA instructions or replace the installed SDK.

Latest immutable sources:

- trn21: `/home/ubuntu/trainium_freshstart_20260918_wave199`, source
  `5d1ff50e33683d7ed15eb86d1b9d1deac28833b8`.
- trn22: `/home/ubuntu/trainium_freshstart_20260918_wave195`, source
  `282935ca9b1e30f6c21c397143770b03034363f9`; source197 reused this code.
- Remote Git manager: `/home/ubuntu/trainium_freshstart_20260918_wave4`.
  Hosts have no direct GitHub access. Transfer committed Git bundles and create
  a fresh worktree for changed code; preserve old checkouts and result labels.

## Latest result: saved FP8 forward survives four-rank contention

Source199 completed all three requests with clean postflights. The handoff audit
rechecked every original frozen completion rule, request hash, and final log hash.
The unchanged source194 saved-forward arithmetic is compared with native BF16
at shape 32768 × 1024 × 4096. Six balanced alternating-order rounds measure
50 calls per arm; report the median of maximum-rank latency, with collectives
outside the timer. Online master casts, quantization/preparation and all
436,469,760 bytes of saved state are charged.

| Arm | Native BF16 (ms) | Saved ROW FP8 (ms) | FP8 latency reduction |
| --- | ---: | ---: | ---: |
| One rank, first | 5.314404 | 4.811585 | 9.46% |
| Four concurrent ranks | 5.973897 | 4.830518 | 19.14% |
| One rank, repeat | 5.320863 | 4.818947 | 9.43% |

This supports continuing the ROW FP8 training implementation. It measures forward
with saved state only: no backward, complete model speed, or BPB result follows.
Correctness is against the specified quantized arithmetic; unquantized BF16
equivalence and training quality are separate questions.

Files: `scratch/nki_library_row_training_forward.py`, its probe, and
`scratch/nki_library_row_saved_contention_modules.py`. The last file fixes only
the source198 diagnostic's Python module registration; the failed original is
retained. Do not repeat source198's failed manifest.

Source196's explicit Vector-engine square was 2.13% slower than the matched
Scalar-engine square at production shape despite exact results. Retain Scalar.
See [library follow-up](fresh-start-library-followup-2026-09-20.md).

## Next work: full backward and analysis of the surrounding model

1. Implement and validate the complete ROW FP8 backward. It is designed but
   **not implemented**. Read the saved-state and scale algebra in
   [library-fusion-next](fresh-start-library-fusion-next-2026-09-20.md).
   Per-token scales cannot be factored outside weight-gradient contractions over
   tokens. Use a valid global/channel-scaled gradient path, retain scales inside
   the contraction, or start with a BF16 weight-gradient control. Keep FP32
   masters/accumulation, cache refresh, tiny-gradient and zero-initialized-weight
   cases, independent VJP references, native evaluation and checkpoint gates.
   Charge all four products, layout conversions, calibration and saves.
2. Analyze source197's completed first-layer versus last-layer captures. This
   isolates a large surrounding-model effect at equal library-call count.
   Source195's unprofiled means were 496.617 ms first-only, 682.661 ms last-only,
   564.512 ms all-six, and 499.108/498.149 ms for bracketing zero-library controls.
   Every sample is retained, including final-step optimizer spikes. Source197
   captures and Explorer ingestion are complete, but detailed runtime-to-NEFF
   attribution and comparison remain undone. The existing analyzer is
   `scratch/model_caller_profile_analysis.py`; local analysis Python is
   `out/fresh-start-tools/profile-analysis-venv/bin/python`. Check its input
   assumptions before adapting it. Profiled latency is not speed evidence;
   overlapping engine intervals cannot be added, and missing DMA metadata
   cannot establish exact traffic or bandwidth.
3. Require a repeatable complete-step gain of at least 3%, plus all numerical,
   cache and evaluation gates, before a new 30-minute quality run. Fixed seed42
   and nearby controls remain the default. The user prioritizes kernel work
   over seed searching and wants root-cause investigation of NaNs/crashes.

The NKI Library audit and extracted techniques are documented in
[fresh-start-nki-library-audit-2026-09-20.md](fresh-start-nki-library-audit-2026-09-20.md).
The vendored 66-file library is pinned to upstream
`92d11f63a9a8ec1ade34e6e1a3b8db66ef31307e`, in an isolated licensed namespace.
Successful transfers include fused FP8 weight preparation and input quantization
inside the tile loader. Smaller SBUF budgets and automatic compiler placement
did not resolve the BF16 whole-model regression. Revisit a negative result only
with a specific new premise; do not treat component gains as model gains.

## Profiles and recovery

The finite preservation sidecar finished at 13:20:42 UTC (06:20:42 PDT).
All four accepted artifacts were copied locally, checked file by file against
the remote hashes, and uploaded to the user's GitHub draft backup release with
matching payload/manifest digests:

- `fs-w197-nkilib-partial-first-profile-r1`
- `fs-w197-nkilib-partial-first-ingest-r2-parquet`
- `fs-w197-nkilib-partial-last-profile-r1`
- `fs-w197-nkilib-partial-last-ingest-r1-parquet`

Local directories are `out/fresh-start-profiles/<artifact-id>`.
Tracked cloud receipts are `run-logs/fs-artifact-backup-<artifact-id>.json`.
The authenticated draft release is `research-backup-20260918`, release ID
391665500 in `CsabaConsulting/trainium_frontier`. Receipts contain exact asset
IDs and digests; draft assets may require authentication. Do not upload duplicates
or delete source evidence just because a receipt exists.

Important old states:

- Original source197 `profile-sequence-r1.sequence.json` still says `running`.
  Its coordinator was deliberately stopped through a verified pidfd before
  another capture could consume the disk reserve. The authoritative hold is
  `fs-w197-coordinator-disk-hold-r1.json`. Its replacement r2 is complete.
  **Do not restart either sequence.**
- Original first ingestion returned zero but failed the 8 GiB free-disk reserve.
  Keep that failed postflight. Its output is also backed up. Maintenance r5
  archived and verified 12.89 GB of old NKI scratch before retiring the exact
  remote inventory, then the fresh r2 ingestion passed normally. No threshold
  was waived and the old failure was not reclassified.
- Source198 stopped with returncode1 and a clean postflight; later requests in
  its plan never ran. Corrected source199 is complete.
- Source195 position-sequence-r1 was never launched and is superseded by its
  completed r2. Source159's prepared T512 quality run **must not dispatch**:
  its preceding screen failed the advancement gate (0.662% slower).

## Submission remains protected

The best public development score is **0.9928513116311253 BPB**; the independent
cold training replay scored **0.9938556748843932**. No newer BPB result exists.
This is a dense six-layer, width1024 model, with no ngrams and no FP8. The private
leaderboard score is unknown and no competition submission has been made here.

The self-contained submission is
[submissions/dense6x1024_public0992851](../submissions/dense6x1024_public0992851/README.md).
Use that directory's `train.py` and `launch-command.txt` together. The command is
245 characters and its omitted defaults were independently checked. Do not
submit the canonical experimental root `train.py` in its place. The package's
README/provenance/validation contain the full cold-replay and contract checks.
Canonical train.py/prepare.py and this submission were unchanged at handoff.

## Cleanup, commits and automatic backup

Each experiment must use `fresh_start_launch.py` with a new label, committed
source fingerprints and a finite timeout. The host-wide
`/home/ubuntu/.cache/trainium_freshstart/device0.lock` covers setup through
postflight. Run marked-process pre/post cleanup and save both health receipts.
No broad process killing, driver resets, cache purges, or deletion of failed
compiler evidence. Disk archival uses capture, independent local/cloud
verification, and exact inventory retirement under the host lock.

Two existing **user crontab** entries continue independently of the GUI:

- Every 10 minutes: `fresh_start_backup_cron.sh` retrieves remote records,
  preserves a recovery snapshot using an alternate Git index, and pushes the
  research and `backup/fresh-start-20260918` branches. No source-branch merging
  or experiment selection occurs. It skips active Git operations and never
  force-pushes. Snapshotting uncommitted work does not replace deliberate source
  commits on the research branch.
- At minutes 5 and 35: `fresh_start_artifact_backup_cron.sh` preserves/verifies
  manifested large artifacts under its own lock. It does not launch experiments.

Local Linux is Devuan with SysV init; no systemd service is involved. Leave these
backups in place. Neither job requires Codex to stay open or transfers experiment
ownership. For commits, cooperate with `out/fresh-start-backup/backup.lock`, call
`fresh_start_backup.require_idle_git`, check the real index, and stage explicit
owned paths. The current working tree can contain automatic artifact-receipt
timestamp updates; preserve them instead of blanket reset/clean operations.

After committing, `python3 research/fresh_start_backup.py --skip-remote` pushes
and verifies both refs. Normal cron performs full remote retrieval. Read
`out/fresh-start-backup/last-success.json` for exact completed push evidence.
Do not run a second experiment owner alongside Claude Code on the same host.
