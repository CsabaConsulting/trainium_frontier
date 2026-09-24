# Trn22 tmpfs G0/G1/G2 native-gate proposal analysis

Date: 2026-09-25  
Role: Analyst  
Status: **NON-EXECUTABLE DRAFT — HOLD**  
Recommendation: **prepare a fresh trn22 G0 -> G1 -> G2 package using `/dev/shm` for every
checkpoint/output and compile cache; authorize only G0 after independent implementation review**

This document is a storage and gate-design analysis. It authorizes no implementation, host access,
request materialization, launch, cleanup, deletion, training, AWS action, G1/G2 advancement, or
submission. The read-only host observations below are not a current preflight and must be freshly
re-proved by an Operator under a separately verified request.

## Decision

The complete same-host three-arm gate fits conservatively only if large mutable artifacts are kept
off the root filesystem. Use the separate `/dev/shm` tmpfs for each arm's initially absent output,
checkpoint, and declared unique compile-cache path, retain all three arm directories through final
comparison, and leave only immutable source staging plus capped small logs/receipts on root.

Do **not** run a separate small tmpfs compatibility probe. G0 is the only representative probe: it
exercises the exact compiler, graph, environment, output, checkpoint verification and five-update
path. A smaller probe could succeed on a different cache path while adding another cold identity and
host action. G0 must therefore be both the observation-off control and a fail-closed storage-routing
gate. G1 stays locked unless G0 proves that its declared tmpfs cache was actually used and the root
global cache was not modified. G2 stays locked until the same checks and scientific gates pass for
G1.

This changes storage plumbing, host identity and arm identities only. It must not change candidate,
optimizer, model, data, seed, update count, compiler flags, gate arithmetic, record cardinality or
exact-byte acceptance criteria.

## Evidence and fixed observations

Controlling scientific/protocol evidence:

- proposal and runbook under this directory;
- historical trn21 G0 evidence commit
  `37aa3644e63e20a1427cfa69b44fd6de7e25cd81` and corrected replay verification
  `8add2585d30b11ac2c50fb2701957657a1a6f30d`;
- G1 r3 disk-failure evidence commit
  `5e45a564bba0d04b052345a7c9c1812ddb71ce51` and receipt SHA-256
  `2325354ee0660fe713a64b23e102c2b53b9c8cab4ca6a5ab081f8850cb322946`;
- cleanup r2 independent FAIL commit
  `5af3dd88d3e312cfda9e771ad67f93a4f89e26e0` and receipt SHA-256
  `8db725310ba9dcfabc62904891ca3b54affef6830ad6d8382270cc6e04103f20`;
- prior path decision commit
  `1ee3253f0dff161654f082b352cf897406d98284`.

Read-only trn22 observations supplied for this analysis:

```text
root free before fresh staging                         8,010,768,384 bytes
representative root staging                              192,167,936 bytes
root free after one representative staging             7,818,600,448 bytes
margin over 6.5-GiB operational floor                    839,278,592 bytes
/dev/shm type/options                     tmpfs; rw,nosuid,nodev; no noexec
/dev/shm size/free                                     66,987,511,808 bytes
RAM installed/available                         approximately 134/131.6 GB
swap                                                        none reported
historical G0 checkpoint allocated                       739,328,000 bytes
historical G0 small logs/reports allocated                    647,168 bytes
historical declared custom cache allocated                     4,096 bytes
existing root /tmp/neff_cache allocated              approximately 9.848 GB
```

The root global cache size and the 4,096-byte declared custom cache make routing **unknown**, not
proved. No proposal may assume that `NEURON_COMPILE_CACHE_DIR` controlled the actual write path.

## Immutable science and fresh identities

Freeze these existing science bytes:

```text
production candidate
  9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae
prepare.py
  5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb
hybrid gate wrapper
  908431d7e1da8a9110e6e6005de5ecbe3dba13d798bf2f37f2443404dd8d7d94
ordinary gate wrapper
  6ed69d2356c28e0dd8f88f494d988c0b87f7f86426a001945bfd958ed6f8f6d0
checkpoint verifier
  e1ead481f9786ec02a3c60d88e72820f084274200bccd1f7eeb16da108cb6e2e
launcher baseline
  7620fea10bb51c54481b8347ec9e97ebd40ce8f5ee206dc3f203c9483c35ee2f
```

Freeze seed 42, world size 4, LNC2, five updates, all model/data/optimizer arithmetic, exact arm
flags, separate cold processes, exact report schema/cardinality, exact-byte comparisons, and the
existing strict checkpoint/causality rules. Any implementation-produced observer, comparator,
manifest, finalizer or launcher hash must be pinned in the new proposal and independently verified
as identity/storage instrumentation only.

Fresh arm identities:

```text
G0 fs-20260925-v2-a1-native5-g0-hybrid-observation-off-trn22-r1
G1 fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn22-r1
G2 fs-20260925-v2-a1-native5-g2-ordinary-observation-on-trn22-r1
host trn22
expected hostname ip-172-31-32-253
```

The observer must accept only that host/hostname and the exact arm/label relation above. It must
reject trn21 and every consumed trn21 label. G0 must be rerun on trn22: historical trn21 G0 is
regression evidence only and is not a causal control for trn22 G1/G2.

## Exact path and retention contract

Use one fresh, initially absent tmpfs parent:

```text
/dev/shm/trainium_native5_20260925_trn22
```

Bind each arm to an initially absent, non-symlink child and never reuse it:

```text
/dev/shm/trainium_native5_20260925_trn22/g0-r1/out
/dev/shm/trainium_native5_20260925_trn22/g0-r1/cache
/dev/shm/trainium_native5_20260925_trn22/g1-r1/out
/dev/shm/trainium_native5_20260925_trn22/g1-r1/cache
/dev/shm/trainium_native5_20260925_trn22/g2-r1/out
/dev/shm/trainium_native5_20260925_trn22/g2-r1/cache
```

The exact arm output argument must name its `out` path and its exact
`NEURON_COMPILE_CACHE_DIR` value must name its `cache` path. The request, job, launch record,
environment, observer report and postflight must all agree on those literal absolute paths. No
fallback path, shared cache, relative path, symlink, bind mount or previously existing child is
allowed.

Each arm also needs a distinct, initially absent root bundle, checkout, small-log/report, job,
result, dispatch, receipt and controller-request path derived from its full label. Source/checkouts
and bounded text/JSON evidence are the only planned root writes. No checkpoint, tensor dump,
compiler cache, core, temporary model or copied tmpfs tree may be archived on root.

Retain all six tmpfs children and all three checkpoints unchanged through G0/G1/G2 final
comparison. No cleanup/prune/deletion is authorized. `/dev/shm` is volatile: any reboot, remount,
instance lifecycle change, mount-identity change, loss of an arm child, or inability to authenticate
the retained artifacts invalidates the sequence. Durable evidence consists of controller-retrieved
small records plus checkpoint/cache metadata and hashes; the checkpoint bytes themselves need not
be copied to root.

## Conservative storage budget

### Root filesystem

Freeze the proposal's root accounting as follows:

```text
observed initial root free                             8,010,768,384
three source stages, 3 x 192,167,936                    576,503,808
three capped small-log/report sets, 3 x 8 MiB             25,165,824
maximum planned root growth                              601,669,632
projected root free after all planned growth           7,409,098,752
frozen operational floor                               6,979,321,856
projected margin over operational floor                  429,776,896
proposal root breaker = floor + 256 MiB                7,247,757,312
projected margin over proposal breaker                   161,341,440
frozen hard-abort floor                                5,368,709,120
```

The 8-MiB per-arm cap is a conservative proposal cap, not the historical 647,168-byte observation.
Before each stage and each dispatch, require:

```text
current root free - all remaining source/log allowances >= 7,247,757,312
```

The higher proposal breaker controls even though it exceeds the existing 6.5-GiB operational
floor. Root free below `7,247,757,312`, any per-arm root log/report set above 8 MiB, or any
unallowlisted root growth is an operational failure. The existing approximately-5-GiB hard floor
remains an emergency abort condition, not a usable planning threshold.

### `/dev/shm` tmpfs

Freeze these independent per-arm caps:

```text
one out/checkpoint tree cap                  1 GiB =  1,073,741,824
one compile-cache tree cap                  12 GiB = 12,884,901,888
one arm total cap                           13 GiB = 13,958,643,712
three known checkpoints, 3 x 739,328,000             2,217,984,000
three compile-cache allowances, 3 x 12 GiB          38,654,705,664
three complete arm caps, 3 x 13 GiB                 41,875,931,136
observed tmpfs size/free                             66,987,511,808
projected tmpfs free after all arm caps              25,111,580,672
retained tmpfs free floor                  16 GiB = 17,179,869,184
projected final margin over tmpfs floor                7,931,711,488
```

The 12-GiB cache allowance is conservative relative to the observed approximately-9.848-GB root
global cache; it does not assert that a single graph needs that space. Required tmpfs free before
each arm, including every remaining 13-GiB arm cap and the 16-GiB final floor:

```text
before G0: 55 GiB = 59,055,800,320
before G1: 42 GiB = 45,097,156,608
before G2: 29 GiB = 31,138,512,896
after G2 and throughout final comparison: >= 17,179,869,184
```

Any arm cache above 12 GiB, output tree above 1 GiB, tmpfs free below the arm-specific requirement,
or tmpfs free below 16 GiB is a fail-closed operational condition. Do not remove an earlier arm to
make a later arm fit.

### Memory and swap

Tmpfs allocation consumes memory, so tmpfs capacity alone is insufficient. Require no configured
or active swap and freeze a 32-GiB minimum runtime `MemAvailable` breaker. Conservatively reserve
the remaining arm allocation, 32 GiB runtime headroom, and an additional 16 GiB uncommitted memory
reserve:

```text
before G0: 3 x 13 + 32 + 16 = 87 GiB = 93,415,538,688
before G1: 2 x 13 + 32 + 16 = 74 GiB = 79,456,894,976
before G2: 1 x 13 + 32 + 16 = 61 GiB = 65,498,251,264
live runtime breaker: MemAvailable < 32 GiB = 34,359,738,368
```

The reported approximately-131.6-GB availability clears the planning threshold, but it must be
re-measured immediately before staging and dispatch. Any swap appearance/use, memory-threshold
failure, or OOM evidence invalidates the arm and locks downstream arms.

## Mount, identity and absence gates

Before creating the parent and again immediately before each dispatch, record and require:

1. `/` and `/dev/shm` have distinct mount IDs and backing devices; `/dev/shm` resolves to the same
   mount entry across all arms.
2. `/dev/shm` is `tmpfs`, mounted `rw,nosuid,nodev`, and does not have `noexec`; record full mount
   source, target, filesystem type, options, device, inode and filesystem identity.
3. Root and tmpfs `statvfs` totals/free/inodes independently satisfy their applicable floors. Never
   combine or infer one filesystem's capacity from the other.
4. The arm root paths, tmpfs arm child, `out`, `cache`, checkpoint, reports and labels are absent,
   with no symlink or ancestor replacement. Earlier completed tmpfs arms remain present, unchanged
   and on the same tmpfs mount.
5. Host/hostname, lock, device idleness/holders, process absence, exact source head/hashes, clean
   scoped tree, environment, no swap and memory gates pass the existing protocol.

A mount disappearance, mount-ID/options change, root/tmpfs alias, insufficient inodes, unexpected
pre-existing path or retained-arm drift is STOP. Do not remount, clean or repair in place.

## G0 cache-routing and root-growth breaker

G0 is not valid merely because training reaches step 5. Its storage evidence must prove all of the
following:

- the exact declared G0 tmpfs cache path began absent, was created on the authenticated tmpfs mount,
  and acquired actual cache content/allocated bytes beyond an empty 4,096-byte directory, or exact
  supported compiler telemetry independently proves writes to that literal path;
- the root `/tmp/neff_cache` path is anchored before launch by device/inode/type plus recursive
  membership, metadata and allocated-byte snapshot, and remains byte/entry/metadata unchanged;
- root growth is limited to the already staged source and the exact capped allowlisted log/report
  files; no compiler cache, checkpoint or large temporary output appears anywhere on root;
- tmpfs/output/cache allocation, root free, tmpfs free and `MemAvailable` stay within all caps and
  floors through quiescence, checkpoint verification and postflight.

If the declared tmpfs cache remains an empty 4,096-byte directory with no authoritative routing
telemetry, cache routing is **unproved** even when `/tmp/neff_cache` also appears unchanged. If
`/tmp/neff_cache` changes at all, or any other root cache/temp growth is attributable to the arm,
classify G0 as `CACHE_ROUTING_INVALID`, stop the owned process group using the already reviewed
bounded TERM/KILL/reap path, capture postflight evidence, and do not authorize G1. Free space still
above a threshold does not cure wrong-path writes.

The runtime guard must sample root free, tmpfs free, `MemAvailable`, exact arm out/cache allocations,
allowlisted log allocations and `/tmp/neff_cache` identity/membership/allocated bytes at a declared
bounded interval. A semantic cache-routing violation, root free below `7,247,757,312`, root logs
above cap, tmpfs arm cap/floor violation, or memory breaker must cause bounded termination and a
failure receipt. Postflight rechecking is required even if the live watcher observed no violation.
An independent verifier must prove the watcher/termination behavior with temporary-filesystem and
mutation tests before any request is executable.

Apply the same routing and breaker gates independently to G1 and G2. A cache hit is not permission
to share a cache: each arm must start with its own absent cache child and prove its own routing.

## Checkpoint verification without a root copy

For each arm, after process-group quiescence and before advancement:

1. resolve the exact expected checkpoint under that arm's tmpfs `out` child without following a
   symlink;
2. open once read-only with close-on-exec and no-follow semantics; require regular file, expected
   ownership/mode, one link, correct tmpfs device/mount and stable same-file-descriptor metadata;
3. from that same open file description, stream SHA-256 over the exact bytes and run the existing
   model-only, weights-only strict load, finite-state/model-digest and fixed-evaluation-logit checks;
4. run the frozen causal-prefix check and require all rank/record/checkpoint identities to agree;
5. re-read same-descriptor metadata after verification and revalidate path name, inode, device,
   size, mtime and link count before accepting the receipt.

Archive only the authenticated SHA-256, byte/allocated size, device/inode/link/mode/owner/timestamps,
strict-load/model/logit/causality results, source/environment identities and report hashes to small
durable evidence. Keep the tmpfs checkpoint itself untouched until final comparison. Any unstable
metadata, replacement, partial checkpoint, strict-load failure, nonfinite value, causal failure or
missing artifact fails the arm. Controller streaming of checkpoint bytes, if later desired, requires
an explicit separately reviewed bounded path; it is not needed for this proposal and must not be
silently added to root.

## Ordered non-executable checklist

1. **Supervisor selection only.** Record trn22 tmpfs as the selected preparation lane and keep all
   run, cleanup, G1, G2, full-A1 and submission booleans false.
2. **Implementer package.** In a fresh branch, change only trn22 host/hostname/labels/path bindings,
   observer allowlists, tmpfs accounting/routing instrumentation and request/finalizer schemas.
   Pin every resulting hash. Do not alter frozen science bytes or thresholds.
3. **Independent static/CPU verification.** Review the exact diff; exercise real observer
   initialization for all three exact runtime argv/environment combinations; require trn22 labels
   to pass and trn21/consumed/wrong labels to fail. Test mount identity, path absence/no-follow,
   root/tmpfs/RAM arithmetic, cache-routing ambiguity, root-global-cache mutation, cap crossing,
   checkpoint same-file-descriptor verification and bounded termination. Verify no action is
   executable and no concrete request exists before state-only finalization.
4. **G0-only authority chain.** A Supervisor may prepare a G0-only pending state after PASS. A
   distinct Verifier must authenticate exact implementation/proposal/manifest hashes and only then
   permit one O_EXCL concrete G0 request. G1/G2 remain false.
5. **Operator G0.** Re-prove host, mount, storage, no-swap, memory, idleness/lock/holder, source,
   path-absence and root-global-cache baselines; stage only capped source/small evidence; dispatch
   exactly the authenticated G0 request once. Retrieve small evidence and complete postflight. No
   automatic next-arm action.
6. **Independent G0 adjudication.** Accept G0 only if science, checkpoint and causality pass and
   tmpfs cache routing/root non-growth are positively proved. Authenticate retained tmpfs state and
   all budgets. Any ambiguity is FAIL/STOP, not permission to repeat under the same label.
7. **Fresh G1 authority chain and run.** Only after accepted G0, repeat independent state-only
   finalization, request verification, preflight, one run, checkpoint/storage verification and
   G0/G1 observation-non-perturbation adjudication. Keep G2 locked throughout.
8. **Fresh G2 authority chain and run.** Only after accepted G1 and valid G0/G1 comparator, repeat
   the complete process for G2 and run the frozen hybrid-versus-ordinary exact comparator.
9. **Final evidence seal.** Before any tmpfs lifecycle action, authenticate all three retained arm
   paths/checkpoints/caches, small durable evidence, same-host comparator output and clean
   postflights. A PASS proves only the declared native gate. Cleanup, full A1 training and submission
   remain separately locked and require new decisions.

Implementation and independent review may be prepared in parallel, but no authorization,
materialized request, host action or reservation may be created during preparation. Each arm needs
its own unique authority/request/receipt chain. A failed or consumed identity is never retried.

## Fail-closed verdicts

Immediate STOP with no downstream advancement applies to any of the following:

- any fixed science hash, flag, seed, topology, update count, threshold or report schema drifts;
- root/tmpfs mount identity, options, path absence, storage, inode, memory or swap gate fails;
- unique tmpfs cache use is not positively proved, or `/tmp/neff_cache`/another root cache changes;
- a root/tmpfs/tree cap or live breaker is crossed;
- any earlier arm is missing or changes before final comparison;
- checkpoint same-file-descriptor hash/strict-load/logit/causality verification fails;
- source/evidence is dirty, incomplete, unauthenticated or mutable;
- an arm crashes, times out, leaves a holder/process, or lacks clean postflight;
- an unauthorized cleanup, prune, cache deletion, root checkpoint copy or automatic next-arm launch
  occurs.

## Final recommendation

Prepare this trn22 tmpfs design as the primary gate lane. The supplied observations support the
complete conservative budget: projected root free is `7,409,098,752` bytes after all planned root
writes, and projected tmpfs free is `25,111,580,672` bytes after three full 13-GiB arm caps. The
remaining uncertainty is not nominal capacity; it is whether Neuron honors the declared cache path.
Resolve that uncertainty through representative G0, not a small side probe, and make positive tmpfs
cache use plus zero root-global-cache mutation an independent G0 pass condition.

Current state remains **HOLD / NOT AUTHORIZED**. Fresh Operator measurements, implementation, an
independent verifier PASS, and a G0-only state authority are still required before any host action.
