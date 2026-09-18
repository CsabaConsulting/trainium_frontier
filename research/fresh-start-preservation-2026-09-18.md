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

The first complete dense-model native trace is also copied locally to
`out/fresh-start-profiles/fs-w4-dense1024-native-trace-r1/`. Its 20 files total
25,547,468 bytes. Per-file SHA256 hashes and source-relative locations are in
`research/run-logs/fs-w4-dense1024-native-trace-r1.summary.json`; the raw profile
files stay outside Git. The source run's launch manifest records the installed
checkout and command. Captures are diagnostic artifacts, not trained checkpoints.

The completed direct-attention width896 control is preserved at
`checkpoints/fs-w3-w896-direct-control-r1/final.pt` (1,453,684,039 bytes).
Source commit `00f82d8`; remote source:
`/home/ubuntu/trainium_freshstart_20260918_wave3/out/fs-w3-w896-direct-control-r1/final.pt`.
Independent local/remote SHA256 values match:

```
2217907e4b876f772d742a5e9234191848d05be7273df86334a64b721bd109c5
```

The older four-rank device capture is archived locally at
`out/fresh-start-profiles/fs-w4-dense1024-device-micro-r2.tar.gz` and remotely
beside its original output directory. The archive is 5,108,910,766 bytes;
independent archive hashes match:

```
a4fdb15ab120db62a5835c1f6c61b401551ba0d3a7bb981287c96903d3232da3
```

All 28 members (16,240,114,289 uncompressed bytes) were streamed locally and
checked against their recorded sizes/SHA256 hashes. After that completed,
only the remote raw r2 directory was removed to recover disk space; both archives
remain. See `run-logs/fs-w4-dense1024-device-micro-r2.archive-verification.json`.
This old capture lacks matching compiled NEFFs and is not the ingested r3 capture.

Selected ingested r3 Parquet tables are already local under
`out/fresh-start-profiles/dense1024-device-micro-r3-parquet/`, with hashes in
`run-logs/fs-w5-dense1024-device-micro-r3.summary.json`. The raw r3 trace and
same-process NEFFs are now also preserved locally, as recorded below.

The completed fused-FP32-optimizer width896 checkpoint is preserved at
`checkpoints/fs-w4-w896-nki-rms-fp32-r1/final.pt` (1,453,684,039 bytes).
Source `a62df2734b5de7e33afa40d74a690fae6404fefc`; remote source:
`/home/ubuntu/trainium_freshstart_20260918_wave4/out/fs-w4-w896-nki-rms-fp32-r1/final.pt`.
Independent local/remote SHA256 values match:

```
6a16f3d490964c201a488e487d407ca18e94b9b23a82a2a97abd35c5ecfd64aa
```

The raw r3 profile plus its same-process NEFFs are preserved at
`out/fresh-start-profiles/fs-w5-dense1024-device-micro-r3/`. All **17 files,
4,112,012,976 bytes**, have independently matching local and remote SHA256
hashes and sizes, verified at 10:14:50 UTC. The capped 8 MiB/s transfer is complete;
the remote originals remain. See the per-file
[preservation receipt](run-logs/fs-w5-dense1024-device-micro-r3.preservation.json).

After final successful synchronization, the old wave4 and trn21-wave6 mirrors
were stopped. Separate mirrors cover active trn22-wave6 and trn21-wave7 work.
New wave8 checkouts at `0016ef8` contain the two precision candidates and post-timing
native evaluation checks; no active run's source was overwritten.

The first completed no-table dense1024 checkpoint is local at
`checkpoints/fs-w6-dense1024-noqk-1800-r2/final.pt`, **469,834,785 bytes**.
Remote source is trn22 wave6's `out/fs-w6-dense1024-noqk-1800-r2/final.pt`,
source commit`c333adb`. Independent hashes match:

```
d4baf47851ecf46100f45de306d5c13e8939c6d06e69a4f1717710aeef659c33
```

See its [receipt](run-logs/fs-w6-dense1024-noqk-1800-r2.preservation.json).
After final successful sync at10:39 UTC, the trn22-wave6 and trn21-wave7 mirror
processes were stopped. Wave8/9/10 mirrors cover the subsequent diagnostic and
training checkouts; each wave retains its immutable source commit.


The optimized dense6×1024 checkpoint is now local at
`checkpoints/fs-w12-dense1024-nki-norm-1800-r1/final.pt`, **469,834,849bytes**.
Remote source is trn22 wave12's corresponding `out/` path, training source
`c61bfca7f2ea2c75cb177c0793c08649065e2a3a`. Independent hashes match:

```
d05929eddd9a0dc313ebb750b01008f6c3f83386f371114532291a42c5e48755
```

The [receipt](run-logs/fs-w12-dense1024-nki-norm-1800-r1.preservation.json) was
written11:43:47UTC, before dispatching the larger standalone evaluation. The
active mirrors now cover wave12 evaluation and wave14 depth8 training; old
wave8–11/13 mirrors were finally synced and stopped at11:11. The new wave15
checkout contains committed follow-up diagnostics and does not change either
active source checkout.
