# Narrow retrieval byte-authentication repair receipt

Status: DISABLED_REPAIR_PREPARED_FOR_FRESH_INDEPENDENT_REVIEW.
This is author evidence, not independent verification or launch authority.

Repair base: 3d9021845db1ff57a98c237757e75955a66dd467.
The earlier FACTUAL_RECEIPT.md and author logs are historical pre-repair evidence;
their source pin is superseded by this receipt for the repaired transport.

Preserved independent refusal report SHA256:
052959d576cb1aa169c267a792dd41745156b88c75d23e13c5d184e78bd703b9.
The report, exact independent test source, logs, original inventory and final tiny
race evidence are retained under evidence/prior-transport-refusal. No previous
evidence was edited. Reviewer code is preserved for audit but excluded, together
with author test source, from the new input-only review bundle.

The demonstrated defect was real: the old transport could send all 32 synthetic
prohibited checkpoint bytes after a pathname replacement, then detect the mismatch
too late. A post-emission exception cannot undo that copy.

Changed production behavior only:

- Retain one no-follow file descriptor before emitting the file header.
- Authenticate every immutable chunk bytes object to the approved whole-file hash
  before passing those same bytes to the emitter. A bounded 64-way range-hash
  decomposition authenticates child hashes against their parent before use.
- A pathname swap can only leave the approved original descriptor's bytes available;
  in-place changes must still satisfy the pre-authenticated hash chain before send.
- Maximum memory is two 256 KiB read buffers plus at most 64 child hashes per level
  (at most eight levels for signed 64-bit file sizes). No host writes, snapshots,
  artifact-size limit, release field, packet byte, resource threshold or timeout
  was added. The wire format and local receiver are unchanged.
- Post-read pathname and final-inventory checks still reject changed sources.
  Hidden/partial/temp artifacts and failed-cap complete-copy behavior are preserved.

New transport.py SHA256:
51e37ab4ca72a1cb48682a50dcf1eec64be101182f0c41dede5680a923997a8e.
Updated disabled binding SHA256:
40a838ba331baf20de6e24f0f4f90cb15947f61e555f55ac8489e3f6dddb6f32.
The binding differs only in transport_source_sha256; every authority remains false.
The original trn21 packet and all seventeen inputs are unchanged.

Sixteen author unittest groups pass normally and with -O. New checks include
replacement before descriptor open, the exact header-callback replacement window,
in-place mutation after the header, mutation immediately after a real descriptor
read, mutation between multilevel chunks, bounded reads, and exact multilevel
round-trip. The 32-byte prohibited fixture is never present in emitted wire or
retained receiver partials. Existing hidden/partial/cache, scan-error, cap-failure,
input mismatch, staging, lock and receiver tests still pass. Both logs are preserved.

No SSH, host access, stage, cleanup, native import, full tensor, training, evaluation,
reservation, release or launch occurred. Fresh independent review and every
previous external Operator/Supervisor gate remain outstanding.
