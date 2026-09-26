# Compiler-cache symlink accounting repair; disabled candidates

Scope selected by the Supervisor: repair only the observed compiler-cache
accounting/retrieval incompatibility; prepare a separately disabled clean-native
r2 retry and a K2 monitor candidate. This is not a release or result acceptance.
Base is 20576a9dba83b3f7f83a73765fde8d90f279a41d. Historical r1 files and receipts
are immutable. The live failure observation is commit
45859da1c901651297ed7241e84aeb5400f932e4,
research/v2/operator/20260926-clean-native-trn21-launch/cache-entry-metadata.json:
the compiler table link has lstat size 121, inode 3187099, and a literal SDK
target. Its 4096-byte target contents were not read by that observation.

## Narrow amendment

Only symlinks below the clean diagnostic's six caches/{neff,local,hlo,inductor,
compiler,debug}/ directories, or within K2's dedicated cache/tmp trees, receive
opaque metadata treatment. Roots themselves must be real directories. All other
symlinks and every special file remain a failed gate. K2's regular-file hardlink
rejection and every existing cap/floor remain.

Do not resolve a link, stat its target, open its target, traverse it, or copy its
contents. Count the link's own lstat st_size in cache bytes. Record literal target,
device, inode, mode, size, mtime and ctime. A link to a directory, outside root,
cycle, or missing target is handled identically as an opaque entry; never recurse
through it. Metadata does not claim target bytes are allocated to the job.
Regular cache files are still counted, without reading their contents.

Descriptor-relative traversal opens only real directories with O_NOFOLLOW,
authenticates their identities, and stats children without following links.
Check link identity/timestamps around readlink; reject observed changes. Retain
scan, permission, vanished-entry and root/ancestor-replacement errors. This is a
sampled census, not an atomic filesystem snapshot or adversarial same-UID proof.

Transport preserves its existing immutable-chunk authentication and whole-file
rehash protocol for all noncache artifacts. Retrieve hidden/temporary/partial
artifacts losslessly; caps fail acceptance and never truncate evidence. Check
cache metadata again at final census; a changed link or cache size/set prevents
completeness. The receiver writes no cache symlinks or cache-target contents.

## Three separate disabled deliverables

1. ../20260926-confirmed36-clean-native-trn21-packet-r2/packet is a separate
   complete packet at the new root
   /home/ubuntu/trainium_confirmed36_clean_native_consistency_20260926_trn21_r2,
   label fs-20260926-confirmed36-clean-native-consistency-trn21-r2.
   The probe changes only its consumed-label literal r1 to r2. Reverse that one
   substitution to recover donor SHA937e2f239ad90b5aa61e0b40ccf067b19ac02febe246492735e948a35c6b86af.
   Rebind driver/binding/proposal root and label, and recalculate affected pins.
   r2-transport binds that exact packet. All authority flags remain false.
2. r1-recovery/transport.py binds only the original consumed r1 root, label,
   manifest, driver and binding hashes. Its stage entry and stage wire are
   explicitly disabled. A separate checked retrieve release is still required.
   It can inspect/copy existing noncache artifacts read-only while a guardian
   retains its lock, without modifying the old packet or releasing the guardian.
3. k2/monitor.py and k2/SOURCE_PINS.json are a two-file candidate overlay onto
   the exact 20576a9d
   research/v2/proposals/20260926-k2-f0f2-native-evidence/packet donor. Only the
   monitor pin changes. The trn22 binding/project/label/source anchor remain
   byte-identical; no new packet installation or runtime binding is performed.

## Unchanged contracts and review boundary

The original transport SPEC, PACKET_SPEC and SCIENCE_SPEC control except the
explicit cache/consumed-label amendment above. K2's selected SPEC, BINDING_SPEC,
partial IMPLEMENTATION_SPEC and REPAIR_SPEC continue to control. Science, all17
inputs, source/checkpoint bytes, ordinary loaders, tensor checks, 900s/15s,
four ranks, 13.5GiB clean-native staging floor, and K2's28 observations,
1200s/16GiB/24GiB remain unchanged. No training/evaluator/selector/guardian edit.

Fresh independent review must inspect exact input-only diffs and frozen source,
execute its own small real-symlink fixtures in normal and optimized Python,
test broken/outside/directory targets without reading targets, noncache refusal,
scan failures, link/directory swaps, caps and immutable-chunk/checkpoint races.
Author code/logs/notes are excluded from the fresh-review input bundle.
Author checks are neither independent acceptance nor host/native evidence.

No SSH, staging, launch, cleanup, guardian recovery, lock release, AWS operation,
or source adoption is authorized. Guardian3202398 and original trn21 lock596823
remain outside this implementation. Runtime/host preflight, all independent
gates, separate sole-Operator assignment/releases and result retrieval remain
required. No automatic retry or successor.
