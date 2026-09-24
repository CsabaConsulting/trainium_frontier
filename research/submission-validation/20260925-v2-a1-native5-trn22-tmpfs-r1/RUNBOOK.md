# Trn22 tmpfs native-five gate — pending runbook

Status: **NON-EXECUTABLE / NO HOST AUTHORITY**. This document prepares a contract; it is not an
Operator command. No concrete launcher request exists. G1 and G2 have neither request nor
authorization files.

## Fixed order and identity

The only order is G0 observation-off hybrid, G1 observation-on hybrid, then G2 observation-on
ordinary, all on `trn22` / `ip-172-31-32-253`. Exact labels, argv, environment, paths and source
hashes are in `g0-request.pending.json`, `requests.pending.json`, and `EXPERIMENT_PROPOSAL.json`.
The candidate, optimizer, data, seed 42, LNC2, world size four and five-update arithmetic remain
unchanged. Historical trn21 requests and labels are consumed and must fail the observer contract.

## Required external finalization sequence

1. An independent Verifier reviews this commit, every protected hash, the root/tmpfs guard, the
   same-fd checkpoint verifier, exact observer wiring, ownership manifest and comparator binding.
2. A distinct Supervisor may create a later G0-only request-bearing child commit with external
   literal HEAD, request SHA-256 and authorization SHA-256. The pending files here are never
   rewritten into authority in place.
3. The later finalizer must translate the frozen G0 template into the standard
   `fresh_start_launch.py` schema without changing any train argv/environment/path field. It must
   preserve the reviewed private process-group TERM/KILL/reap/quiescence lifecycle and exact
   launcher/guard hashes. This package deliberately contains no materializer and cannot launch.
4. The Operator must freshly prove host identity, one-run lock, device idleness, exact clean source,
   initial path absence, mount identity/options, no swap, RAM and both filesystem floors. No cleanup
   or remount is permitted to make a gate pass.
5. G0 remains a single cold representative run. During it, the runtime must sample root free,
   tmpfs free, `MemAvailable`, per-arm out/cache allocation, capped root logs and all enumerated root
   cache snapshots. The watcher must begin before child start, emit at least one authenticated
   runtime sample no more than 15 seconds apart, and remain active through the final artifact seal;
   a zero-sample or incomplete series is invalid. A violation terminates the owned group through the
   reviewed bounded lifecycle.
6. After quiescence, `checkpoint_guard.py` must hash and validate `final.pt` through one pinned file
   descriptor. It requires the model-only weights-only schema, finite state, strict load, exact
   logits and causal prefix behavior; exact mode `0600`, one link, current uid/gid, pinned output
   parent identity and the authenticated tmpfs device/mount; the checkpoint remains on tmpfs.
   Any later independently reviewed runtime must establish a private `0077` child umask before
   creation so the unmodified `torch.save` produces mode `0600`; changing the checkpoint mode after
   creation is not an accepted substitute.
7. `storage_contract.adjudicate_storage` must return `PASS`. Empty/4-KiB declared cache, root cache
   mutation, mount drift, root growth/cap failure or ambiguity returns `CACHE_ROUTING_INVALID` and
   G1 stays locked. A new evidence-bound G1 authorization chain is required; advancement is never
   automatic. The same rule applies before G2.
8. Retain all three tmpfs outputs, caches and checkpoints through final exact comparison. No root
   checkpoint copy, cleanup, prune, full A1 training or submission is authorized.

## Frozen resource gates

- Root breaker: `7,247,757,312` bytes; operational floor `6,979,321,856`; emergency hard abort
  `5,368,709,120`; capped root logs/reports `8,388,608` bytes per arm.
- Before each arm, root free must reserve every remaining `192,167,936`-byte source stage and
  `8,388,608`-byte log allowance above the breaker: G0 `7,849,426,944`, G1 `7,648,870,400`, G2
  `7,448,313,856`. Root and tmpfs must each report at least 10,000 free inodes.
- `/dev/shm` pre-arm floors: G0 `59,055,800,320`, G1 `45,097,156,608`, G2
  `31,138,512,896`; retained final floor `17,179,869,184` bytes.
- `MemAvailable` planning floors: G0 `93,415,538,688`, G1 `79,456,894,976`, G2
  `65,498,251,264`; live breaker `34,359,738,368`; swap total and use must both be zero.
- Per arm: output at most 1 GiB, cache at most 12 GiB, combined at most 13 GiB. The declared cache
  must allocate more than an empty 4-KiB directory and every enumerated root cache must be exactly
  unchanged.

Every cache/output tree is an exact closed-schema, anchored no-follow recursive snapshot containing
mount/device/inode/type/mode/owner/link/size/allocation/timestamp and same-fd content SHA-256 for
every regular file. Every opened directory/file descriptor must also report the authenticated mount
ID. Traversal starts at a pinned `/` descriptor: every lexical ancestor before the authenticated
mount boundary is identity- and mount-ID-pinned and revalidated, while the exact authenticated
device/mount-ID pair is mandatory at the boundary and for every descendant. An absence observed
before that boundary is not evidence about a tmpfs path. Symlinks, parent swaps, same-device bind
mounts and nested submounts are invalid. While the authenticated host/run lock remains held and the
owned process group, device and unknown-holder postflight are exactly quiescent, two complete
recursive observations must match. The terminal canonical manifest is an authenticated observation,
not a claim that a mutable same-UID tree remains perpetually current.

The later independently authorized runtime must bind exact request/job/postflight, lock ownership,
process birth/TERM/KILL/reap, device-holder, watcher-coverage, mount/root, tree and checkpoint receipt
hashes into the bounded canonical observation receipt at the exact request path. It must publish that
small JSON only with anchored `O_EXCL`, mode `0600`, one link, current ownership, full-write, file and
parent `fsync`, and same-descriptor readback/hash verification. It must not copy cache or checkpoint
bytes to root. This package contains no executable finalizer or materializer.

Before G1, G2, final comparison, or any tmpfs lifecycle action, a consumer must retain the lock,
freshly re-open and reauthenticate the exact live mount-aware tree and checkpoint, and compare their
canonical identities/digests with the immutable receipt. Any later writer, process, holder, mount,
content, membership or checkpoint drift is STOP. Privileged freezing, remounting, ownership changes,
cache copying, cleanup and prune remain outside authority.
Retained G0/G1 evidence is the full authenticated arm record, never a label or asserted boolean.
The final checkpoint receipt binds its model/logit/causal digests to the four rank-report hashes,
frozen source/observer, runtime job, postflight and exact output-parent identity.

Any missing evidence, changed retained arm, non-quiescent process, holder, timeout, crash, NaN,
checkpoint instability, unexpected root write or mount/lifecycle change is STOP, not a retry.
