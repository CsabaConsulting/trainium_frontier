# Continuity and GitHub backups

The user's2026-09-18 crash-recovery request authorizes regular commits and pushes
to `CsabaConsulting/trainium_frontier`. `trainium_frontier` is the canonical
checkout. The two sibling `trainium_frontier_backups/2026-09-18/` directories
preserve original EC2 files, without independent Git histories. Do not overlay
them on newer canonical code. Reconcile a genuinely new remote source change
with a reviewed patch; immutable experiment sources normally need no merge.

## Host ordering and promotion policy

The ten-minute job retrieves trn21 first, then trn22, sequentially. Each host and
immutable worktree has a separate destination; the second retrieval cannot
overlay the first. These are evidence retrievals, not Git merges. The helper
then assembles one recovery snapshot and atomically pushes the source and
backup refs. It never merges either host into the canonical checkout.

All experiment source is authored and committed in the canonical research
branch before deployment. Each host receives a detached worktree at an exact
commit; device jobs never share a mutable source checkout. Failed candidates
and their evidence remain on the research branch, normally behind explicit
scratch entrypoints or opt-in flags. They do not become the main branch merely
because a backup ran. A winning change requires deliberate integration and
validation before promotion. A genuinely new remote edit is retrieved into its
own recovery path, reviewed, and applied one patch at a time. Separate feature
branches are appropriate for competing production implementations; separate
per-host branches are unnecessary for immutable experiment deployments.

Host-wide device locks serialize experiments on each individual instance while
allowing the two instances to run independently. The backup has its own overlap
lock and does not acquire device ownership. Explicit log synchronization uses
another lock when adding provenance-keyed records to the experiment registry;
that JSON-record reconciliation is not a Git source merge.

An observed overlap at19:58UTC exposed a preflight false positive: the inline
backup inventory source contains the filename train.py and matched the old
process-name heuristic. The launch was refused before device work began. The
updated host check recognizes only the exact read-only inventory script under
Python; arbitrary inline programs and altered scripts stay conservative. Device
holder and Neuron-runtime checks remain mandatory even for a recognized backup.
The regression check replays the actual failed receipt and verifies those cases.

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
   Newly staged files are included; `research/continuity/staged.patch` separately
   preserves the staging intent as a binary-safe patch against the source commit.
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

Merge/rebase/cherry-pick/revert/sequencer/bisect state and Git index/ref locks
cause the cycle to skip and retry later. These are checked before the cycle,
after snapshot assembly, before publishing its local recovery ref, and before
pushing. The real index identity and HEAD must remain stable during assembly.
The helper does not acquire or delete Git's own locks. A merge beginning after
the final check can at most overlap a push of an already captured commit; the
helper still cannot edit the merge's working files/index, and ordinary remote
fast-forward checks remain enforced. Recovery snapshots are WIP, not an atomic
transaction covering arbitrary edits by an editor that does not cooperate with
Git locks.

Manual run: `python3 research/fresh_start_backup.py`.
Local verification/status: `out/fresh-start-backup/latest.json`, historical
`receipts/*.json`, `last-success.json`, and `cron.log`. Scheduler inspection:
`crontab -l`. The installed entry belongs to user `csaba`; installation uses
neither `sudo` nor a root shell. Codex sandbox escalation does not change UID.
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

`fresh_start_artifact_backup.py` uses a draft release with immutable filenames
containing content hashes, inventories every source file, and verifies GitHub's
SHA256 digest and size after each upload. Every retained checkpoint and raw
profile identified by a preservation receipt is discovered automatically; the
priority catalog also covers both original EC2 snapshots and older raw captures.
Archives exceeding1.5GiB are streamed into ordered chunks. Preparation and verified
uploads are cached so a restart resumes the remaining items. Only an owned,
zero-byte `starter` placeholder from an interrupted upload can be removed; its
metadata is recorded first. Completed or differing assets are never overwritten.

After a real upload is verified, the installer can add a second user cron entry
at minutes5 and35 of each hour. It runs at nice10, has its own overlap lock and
3500second bound, and does not manipulate Git. Upload receipts are picked up by
the ten-minute Git recovery backup. Status is in
`out/fresh-start-artifact-backup/latest.json`; files are protected only when their
tracked receipt has `all_github_digests_match: true`.
