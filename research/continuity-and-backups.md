# Continuity and GitHub backups

The user's2026-09-18 crash-recovery request authorizes regular commits and pushes
to `CsabaConsulting/trainium_frontier`. `trainium_frontier` is the canonical
checkout. The two sibling `trainium_frontier_backups/2026-09-18/` directories
preserve original EC2 files, without independent Git histories. Do not overlay
them on newer canonical code. Reconcile a genuinely new remote source change
with a reviewed patch; immutable experiment sources normally need no merge.

## What runs independently of the GUI

`fresh_start_backup_cron.sh` runs `fresh_start_backup.py` every10minutes from the
local user's cron daemon. Each invocation is bounded to480seconds, with a lock
preventing overlap. No computer can run the schedule while powered off or without
network access; the next interval retries after recovery. Credentials stay in the
existing SSH key and GitHub keyring. The helper discovers the user's existing
desktop bus for cron, without exporting a token to a file. A locked keyring or
network failure is recorded, never described as a successful backup.

The helper:

1. Reads local dispatch receipts, which are written before remote launches.
   Fetches each manifested job's logs, setup/cleanup receipts, complete remote
   experiment registry, and any unexpected source edits from both hosts. It
   stages remote data under ignored `out/fresh-start-backup/hosts/`. It never
   writes to or deletes a file on EC2 and never touches a device process.
2. Creates a recovery commit using a temporary alternate Git index. Captures all
   tracked working-file changes and new text/code/research files (up to50MiB per
   file); ignored weights/profiles and unsupported new files are not silently
   treated as backed up. Skipped new files are listed in the snapshot manifest.
   The active branch, working files, real index and staged changes are untouched.
3. Records remote evidence under `research/continuity/hosts/` in the recovery
   commit. These files cannot overwrite canonical source or the canonical
   experiment log. `research/continuity/latest.json` records source identity,
   unfinished jobs, remote source differences, retrieval failures and the previous
   push receipt. These are WIP recovery snapshots, not tested/promoted changes.
4. Atomically pushes ordinary committed source to
   `research/fresh-start-20260918` and the recovery snapshot to
   `backup/fresh-start-20260918`. Uses normal fast-forward checks, no force.
   Verifies both remote hashes using `git ls-remote`. A dead EC2 host does not
   prevent backing up available local work; the incomplete retrieval is explicit.

Manual run: `python3 research/fresh_start_backup.py`.
Local verification/status: `out/fresh-start-backup/latest.json`, historical
`receipts/*.json`, and `cron.log`. Scheduler inspection: `crontab -l`.
Committed config: `research/continuity.json`. The cron installation receipt and
subsequent successful runs are recorded separately; this document alone does
not prove the scheduler is installed or a push succeeded.

The script was tested against a temporary repository and bare Git remote for
staged/unstaged separation, deletion capture, new-file inclusion, ignored/unsafe
suffix exclusion, separate remote evidence, linear snapshot history, atomic
two-ref pushes and readback. Neither the active HEAD nor the real index changed.

## Routine research checkpoints

Before deployment, commit the exact source. After a job completes, synchronize
its evidence with `fresh_start_sync.py`, update the results note and `RESUME.md`,
commit and push the research branch. The10minute recovery branch bounds lost
uncommitted work between deliberate checkpoints. It is not a replacement for
reading results, recording failures, or maintaining the human resume note.

## Restore after a lost laptop

Clone the user's GitHub fork and check out `research/fresh-start-20260918`.
Read `research/RESUME.md`, then compare it with
`origin/backup/fresh-start-20260918:research/continuity/latest.json`. Create a
separate checkout of the recovery branch to examine later WIP changes and remote
evidence; do not blindly merge its machine-status directories into source.
Recover desired edits with explicit paths or a reviewed patch. Import remote
experiment registry records by provenance using the same deduplication rule as
`fresh_start_sync.py`. A missing postflight means inspect the host before launching
another job, not that the process is absent.

Checkpoints and raw profiles are ignored by Git. Their SHA256 receipts establish
identity but do not contain the data. Off-machine artifact backups must carry the
actual files, with upload and digest verification receipts. GitHub draft release
assets provide this separate channel; keep every asset below2GiB and retain a
chunk manifest when splitting a larger archive. See GitHub's
[release storage limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
and [release asset digest fields](https://docs.github.com/en/rest/releases/assets).
Do not label a pending or failed upload as protected against SSD failure.
