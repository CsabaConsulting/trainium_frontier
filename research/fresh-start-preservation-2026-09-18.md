# Preserved artifacts — 2026-09-18

Both pre-existing remote working directories were copied to the local sibling
`../trainium_frontier_backups/2026-09-18/{trn21,trn22}` before new runs. These are
recovery snapshots, not replacements for the canonical repository. They include
remote code/research/log changes; caches, Git objects and heavyweight checkpoint
directories were excluded. Neither old remote checkout was overwritten.

The record checkpoint from trn21 was separately copied to
`checkpoints/fresh-start-preserved-t17-l1-m128/final.pt` (1,297,949,831 bytes).
The local and remote SHA256 hashes were independently checked and match:

```
55295ee555d4742ff606dd43fc04b02cf94362da0d06fc930e7a174b3cd017f5
```

Original remote path: `/home/ubuntu/trainium_frontier/out/t17-l1-m128/final.pt`.
Weights remain outside Git under the existing checkpoint ignore rule. This note
records their recovery location; code, logs, research and provenance are committed.

The new width896 checkpoint is also preserved locally at
`checkpoints/fs-w896-bf16-1800-r1/final.pt` (1,453,683,975 bytes). Its remote
source is `/home/ubuntu/trainium_freshstart_20260918/out/fs-w896-bf16-1800-r1/final.pt`.
Independent local and remote SHA256 calculations match:

```
fe4a2bcb153a3a4920cb92bda15f05b5030da8a391742dd92092e68910828c6d
```

Its source commit is `ca635670f67b067201fb9b6167eccf414b70bacd`. See the
[results note](fresh-start-results-2026-09-18.md) for the exact evaluation geometry.

Fresh training wave1 uses `/home/ubuntu/trainium_freshstart_20260918`, source
commit `ca635670f67b067201fb9b6167eccf414b70bacd`. It was installed from a locally
created Git bundle. Source files are not changed while a run is active. New waves
use a new checkout and distinct log prefix so copied historical logs cannot
overwrite newer results during synchronization.

Run logs and launch manifests are mirrored continuously by `fresh_start_sync.py`.
Completed JSONL records are appended using provenance-based deduplication. The
runner's old `status=keep` convention means a successfully completed process; it
does not mean the research has selected that model as a winner. Raw logs are the
authority for completeness, budget, causality and evaluation geometry.

`git_dirty` includes newly generated research artifacts. The added `source_dirty`
field checks the three hashed launch sources independently. Subsequent runs in
the same immutable checkout may have dirty result files while all training code
still matches its recorded commit. Device-holder/process preflight remains
host-wide; a checkout lock alone cannot prevent overlap from another checkout.
