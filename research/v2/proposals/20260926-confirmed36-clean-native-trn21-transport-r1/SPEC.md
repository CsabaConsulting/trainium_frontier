# Fixed trn21 disabled staging and retrieval mechanics

## Scope and authority

This additive packet provides only fixed small-file staging and streaming artifact
retrieval for the original reviewed clean-native trn21 packet. It is not a launcher,
new run framework, cleanup tool, or native/scientific approval. All authority flags
in TRANSPORT_BINDING.disabled.json are false. There is no RELEASE.json in this
proposal. The trn22 alternative is unchanged and parked, not staged or reserved.

The unchanged original packet is authoritative for scientific computation,
checkpoint pair, all seventeen retained input paths and hashes, six shapes,
four-rank LNC2 settings, tolerances, masking, no-hook settings, exact native argv and
environment, lifetime guardian, and 900-second child plus 15-second TERM limits.
PACKET_SPEC.md and SCIENCE_SPEC.md are byte-preserved copies of the two controlling
specifications. This transport neither edits those files nor enables the original
disabled binding.

## Fixed identity and preserved payload

- Host: trn21, hostname ip-172-31-34-85, uid 1000.
- Label: fs-20260926-confirmed36-clean-native-consistency-trn21-r1.
- New-only root: /home/ubuntu/trainium_confirmed36_clean_native_consistency_20260926_trn21_r1.
- Existing lock: /home/ubuntu/.cache/trainium_freshstart/device0.lock,
  device 66305, inode 596823; never create, replace, truncate, or unlink it.
- Packet manifest: 905eb5fb73281c4058c52f6d16f4824dd1146929fa80b64a7b9d8d7d41d61f8f.
- Original disabled binding: c29fe99533c21cae045a4cd7b5116f40808feda65bdab14e5ed38e1ab0b11760.
- Driver: 4a062c97107e3afd9f0061824074e9f151644e204d2957dd28ddfd6b372e98f7.
- Helper: 937e2f239ad90b5aa61e0b40ccf067b19ac02febe246492735e948a35c6b86af.
- Foundation: 65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8.
- INPUTS: 0cc0569936b0b1653bcc946edcdedd1bf91d3f671fabbc205a0a7b883bf9a15e.
- Guardian: e461ec5c76c784fd51471c9e8c268cdde42fe1def68b31bfc9524480f980c312.

Only the exact nine existing small packet files (the manifest and its eight
entries) are staged. The aggregate static payload must be at most 1 MiB.
The already-present checkpoints and seventeen inputs are never copied.

The implementation reuses the independently reviewed full-cold exclusive staging
pattern (source 94524dd3c7fd25f01952147f766bb0c6ecd75274ae0b198847a41d656782c338),
but omits its dispatch step entirely. It imports only the authenticated stdlib
driver/guardian/idle module definitions to reuse their checks. It does not import
the scientific helper/foundation, instantiate a guardian, or spawn a native child.

## Runtime facts, not permission

evidence/RUNTIME_INSPECTION.json is the exact sole-Operator read-only observation
from canonical commit 608f2dd016b13e965ff34ad496c56063145e510c, SHA256
8ca95247b8f6a0699a4c198c514a4d6ff458e7f322963180d448fdf2af56e9da.
Its observation ended at 2026-09-26T06:05:54.020812+00:00. The eight executable
records are copied exactly into this transport's disabled factual binding.
The original native binding remains byte-identical, including its null runtime
fields. Native SDK package versions were intentionally not gathered; no SDK
imports were performed by that observation.

The observation reported 14,746,206,208 free disk bytes and 131,413,176,320
available RAM bytes, with disk headroom 250,691,584 bytes above the staging floor.
These facts are historical, not reservation, freshness, or release evidence.

## External authorization and connection boundary

No SSH invocation, credentials, connection, stage, retrieval, or launch is executed
during this local preparation. The transport has no connection code.
Only the sole assigned Operator may bind and use the existing authenticated
strict-host-key connector (source
7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899)
and its sensitive-output checks. An independently checked connection receipt must
freeze the actual strict connector argv/host identity and exact remote command
[/usr/bin/python3, -I, -B, -]; private credentials must never enter receipts.

Future wire construction requires a separately supplied release with the original
driver's exact release schema plus a transport object containing the applicable
stage_authorized or retrieve_authorized true, the exact transport source SHA256,
independent review SHA256, Operator assignment SHA256, connection receipt SHA256,
and exact remote_command above. Stage additionally needs the runtime inspection
bytes and matching release pin. The supplied transport source must match this
disabled binding. Hash-shape checks are mechanics only, not authentication of
Supervisor decisions: the sole Operator must substantiate those release records.
Missing review, connection, assignment, fresh controller state and external
release remain explicit nulls. No connection or release is created here.

## Staging transaction

The future stage operation refuses an existing root, symlinks, unexpected packet
names/bytes, wrong host or uid, wrong original lock identity, and a busy lock.
It holds an exclusive nonblocking flock on the original inode throughout staging,
using the unchanged driver's staging_preflight before any remote root creation.
That preflight checks host, STOP, original lock, all seventeen input identities,
the eight runtime executable records, ps plus neuron-ls plus fuser device idleness,
and a fresh (at most 60 seconds) untripped controller record.

Before and after staging, free disk must be at least 14,495,514,624 bytes
(13.5 GiB), and available RAM at least 68,719,476,736 bytes (64 GiB). The native
driver's unchanged runtime disk floor remains 11,274,289,152 bytes (10.5 GiB).
No cleanup is attempted if a floor fails.

The exact nine files are exclusive-created, fsynced, made read-only and rehashed.
The historical runtime observation and staging receipts are also retained.
All seventeen inputs are rehashed again; resource, STOP, release freshness,
runtime, device-idle, lock-identity and artifact checks repeat before the completion
receipt. The separately authorized RELEASE.json is published last. There is no
dispatch after publication, even if a future supplied release permits launch.
The original native driver must independently enforce its launch gates.

Any failure returns HOLD and preserves the partially created root plus any
failure receipt that can be written. A failed root is not retried or cleaned.
The staging process releases only its own flock when bounded synchronous checks
have returned; it has spawned no native child. Once the unchanged native guardian
is separately launched, its lifetime and retained-lock behavior are untouched.

## Retrieval transaction

Retrieval is separately authorized and read-only on the host. It may collect
failure evidence while the original guardian retains its lock; it never bypasses
or changes that guardian. It records before/after resources, complete ps,
neuron-ls and fuser results, original inode and nonblocking busy/free observations.
These observations do not prove quiescence or native acceptance.

Every non-cache regular file under the fixed root is inventoried, hashed and
streamed, including hidden receipt files, partial receipts, hidden temporary files,
logs and resource evidence. Scan errors at root or nested directories, symlinks,
special files and read errors fail completeness rather than producing zero sizes.
Cache contents are not copied; individual cache sizes and total bytes are retained.
The two exact retained checkpoint hashes are explicitly rejected before contents
are emitted, as well as by the local receiver.

Raw chunks are bounded to 256 KiB, encoded wire lines to 512 KiB. Each file's bytes
are read through one retained no-follow descriptor. Before emission each immutable
chunk bytes object is authenticated to the approved whole-file SHA256 by bounded
64-way range hashes: a parent's bytes are hashed before its child hashes are trusted,
and leaf bytes are hashed before those same bytes are emitted. Path replacement or
in-place mutation cannot substitute unapproved bytes between verification and send.
Memory is bounded to two chunk buffers and at most 64 child hashes per level (at
most eight levels for signed 64-bit file lengths); no host snapshot/temp writes or
new artifact-size ceiling is added. Late source changes still fail completeness.
The whole-file hash is also checked after transfer; the final inventory is rescanned
and rehashed. File changes, additions or removals fail completeness. The seventeen
retained inputs and eight runtime identities are checked without importing native
packages. Input or runtime mismatches are preserved as failed gates while recovery
continues. Artifact limits remain 16 MiB per rank and 256 MiB total, including
hidden/partial/temp files. A cap breach is recorded as failure, never used to
truncate the copied evidence.

The local receiver creates a new absolute non-symlink destination exclusively.
It rejects traversal, duplicates, wrong order, malformed hashes, size/hash mismatch,
checkpoint contents, trailing records, or a missing complete record. It fsyncs
completed files and saves the final retrieval receipt. On failure it retains
partial files and a failure receipt; it does not overwrite or delete any destination.
A metadata line exceeding the explicit wire bound fails as incomplete, not success.

A complete copy is labeled either COMPLETE_COPY_FAILED_GATE or
COMPLETE_COPY_PENDING_INDEPENDENT_REVIEW, with native_acceptance false.
Neither status is a scientific PASS. Connection exit status, original guardian's
terminal/quiescence records, native receipts and scientific comparisons still
require the separate Operator and independent review workflow.

## Local author checks and remaining gates

The author tests use stdlib-only local tiny fixtures, real local flock, injected
filesystem failures and mocked host observations. They do not use SSH, host
credentials, native packages, full tensors, training or evaluation. Both normal
and optimized interpreter logs accompany the factual receipt. Author test code
is committed for audit but excluded from the input-only independent review bundle.
Original independent refusal evidence is preserved.

Fresh independent transport review, separate Supervisor release, exact authenticated
connection binding, current controller checks, fresh stage/launch resource and
machine checks, and the actual native run with complete artifact retrieval remain
outstanding. This local packet author cannot approve its own patch for launch.
