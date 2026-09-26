# K2 F0+F2 real-geometry native falsifier

Completed Researcher proposal only; **NON-LAUNCHABLE**. No harness, host slot,
release, source adoption or native result is supplied. Select, implement and
independently review this separate diagnostic before any sole-Operator action.

## Question and immutable scope

Can the unchanged default-disabled supplied-metadata consumer execute full
V8192/T32768/C1024 BF16-to-FP32 F0 and F2 on all four LNC2 ranks and exactly match
both an independent ordered binary32 oracle and ordinary native embedding
backward? The cheapest rejection is the complete F0 result, including all1024
distinct channel tags and both512-channel owners. F2 then tests rounding order,
reachable signed subnormals, boundary rows and zero signs.

Consumer `research/v2/lanes/k2-bf16-production/consumer_nki.py` SHA256:
`9425390c96053a1b24d4addf8b61b1a87c4993f3551f08c7cdcefcf142d84dcd`.
Use exactly `build_kernel(enabled=True)` and the returned `resident_consumer[2]`.
No rewrap, model import, adapter, metadata producer, dtype shim or arithmetic edit.
All calls have the full geometry; CPU synthetic supplied metadata is diagnostic
input only. Seed42; identical canonical fixture bytes on all four actual ranks.

The six-fixture proposal at
`research/v2/proposals/20260926-k2-realshape-supplied-metadata/SPEC.md`, SHA256
`c5ffec92cbe0d3036b1a740e6d42429917453eb06de75ad35cd6728fb213b485`,
is unchanged. Its explicit partial alternative motivates this separately scoped
design. Its full qualification, exact memory evidence and F1/F3/F4/F5 requirements
are NOT satisfied or amended here. Consumer fresh review remains
SOURCE_MATH_ACCEPTED_NATIVE_COMPONENT_UNQUALIFIED. Original FP32-subnormal and
required ordinary-control failures remain FAILED_PRESERVED.

## Exact fixtures, routes and evidence

Freeze full little-endian canonical blobs and their hashes before the first
native call. Unspecified tangents are positive BF16 zero. Let q=2^-133.

| Fixture | Frozen recipe |
| --- | --- |
| F0 | id[t]=t mod8192; g[t,c] has BF16 rawbits0x3800+c, t0..32767/c0..1023. Four occurrences/row; all1024 channels distinct. |
| F2 | Begin id[t]=t mod8192; override t0/128/129 to row0, t1/130 to127, t2/131 to128, t3/132 to8191, t4/133 to4095. For a=c mod8: a0 sets t0/128/129=[2^24,1,-2^24]; a1 uses [1,2^-24,-1]; a2 sets t1/130=[2^-126+q,-2^-126]; a3 sets t2/131=[-2^-126-q,2^-126]; a4 sets t3=q; a5 sets t3=-q; a6 sets t4/133=[-0,+0]; a7 sets t4/133=[-0,-0]. |

On EACH rank run in this order: F0 direct, F0 direct repeat using the same
callable/input tensors, F0 ordinary eager, F0 ordinary compiled, F2 direct,
F2 ordinary eager, F2 ordinary compiled. All four ranks finish each route and
report their local comparison before the next route is scheduled. First failure
stops further scheduling; remaining routes are NOT_EXECUTED. Do not use an
unbounded collective for failure rendezvous. Exceptions/worker loss use the
bounded owned-process termination protocol, preserving any partial receipts.

Ordinary routes use FP32 weight[8192,1024], ordinary embedding immediately cast
to BF16, and actual autograd backward with the identical BF16 tangent. For a
fully specified finite weight, use w[r,c]=(((17*r+3*c) mod257)-128)/128, exactly
representable in FP32 and BF16. Check the entire forward against an independent
ordinary gather/cast reference. Record actual tangent and gradient dtypes.
No consumer/custom backward is allowed in this control. Reuse one ordinary
compiled callable from F0 to F2. An ordinary-route mismatch fails this partial
diagnostic even if the direct consumer matches; never replace expected bits
with native output or apply tolerance.

The harness author supplies an integer-exact IEEE binary32 round-to-nearest,
ties-even recurrence, beginning with positive zero and widening each raw BF16
operand exactly before each ordered addition. Derive row occurrence order from
canonical IDs independently of consumer metadata. F0 needs1024 four-addition
recurrences; F2 has eight channel classes. Validated identical whole timelines
may share an oracle result, but no occurrence regrouping/tree sum/FP64 sum or
torch index_add may replace the ordered oracle. Expand ALL8192x1024 FP32 output
words and compare all33554432 bytes for every route, including zero signs and
both owners. Independent review must execute its own reference/counterchecks.

Construct metadata separately from its validator. Validator proves counts,
exclusive starts, ascending positions, full token permutation, ID correspondence
and exact128-row maxima. Reject swapped ordering, wrong in-range row, duplicate
or missing position, staleF0 metadata forF2, wrong counts/starts/maxima and invalid
addresses before upload. Never send corrupt metadata to this unvalidated kernel.

Before AND after every route, round-trip and byte-compare complete actual inputs
(tangent, IDs, applicable metadata, and ordinary weights), revalidating returned
metadata against returned IDs. No shape/hash-only transport check. Retain direct
F0 output storage through the repeat and later F2 call; re-read ALL its bytes
after both to prove it was not overwritten. Route receipts must bind actual
call output to its full retained blob, rank/PID/starttime, input hashes, source
hash, invocation ordinal and synchronization timestamps. A supplied oracle blob
presented as an actual output is insufficient provenance.

Before release, CPU harness negatives must reject zero/stale output, missing
owner, one-bit or signed-zero changes, wrong dtype/shape, nonfinite output,
missing route/blob, and a fabricated actual-output association. A fresh reviewer
must demonstrate that bypassing the consumer cannot create an accepted receipt.
Wrong-BF16-input specialization refusal on hardware is explicitly NOT tested
here; it remains a full-suite gate. This diagnostic does not claim that refusal
or hardware kernel-execution counts without evidence.

## Feasibility: enforce guards, report unknown precision

Existing mechanisms were read at snapshot
`94fe5ae6f1d665e7bd36093d1a9aefe84c3c4ebf`:

| Evidence | What it establishes / does not establish |
| --- | --- |
| `research/v2/proposals/20260926-k2-registered-native-trn22/host_driver.py:58` and `dispatch_driver.py:11` | Source/STOP/host checks, inherited original flock, fresh ps+neuron-ls, six absent private caches and timeout wrapper are implemented. Disk is prelaunch-only; current timeout600+60 is child-only. No total-driver deadline or running RAM/cache/artifact cap. |
| Same directory `idle_guard.py:34` and `:70` | Lock FD/inode/parent identity plus both parsed idle checks, each bounded15s. These remain mandatory. |
| `research/v2/proposals/20260926-p-offline-decoder-trn21/decoder_support.py:132` | Bounded symlink-rejecting scratch walk, PID/starttime descendant census, Linux subreaper, owned-session TERM/KILL, monotonic deadlines and bounded stdout capture are existing standard-library/Linux mechanisms. |
| Same decoder helper `:233`, `:266`, `:311` | Decoder-specific address-space/CPU/affinity limits and rejection of every descendant must NOT be copied to torchrun. Post-hoc ru_maxrss is NOT summed concurrent RSS or a hard memory ceiling. |
| Same decoder `host_driver.py:66` | Existing statvfs and MemAvailable parsing; thresholds are frozen afresh below. No SDK memory API required. |
| Harness core commit `3f6bf53b31786a148222193452ef727a897c7ec9` | Portable fixtures/oracle/blob comparison only; actual route provenance, multi-rank safe writes, native transport and resource watchdog remain unimplemented. Reuse is optional after independent checks; it is not a runnable packet. |

Separate Implementer prepares a small fixed-command packet using these existing
lock/idle/containment patterns, with reviewed monitoring adaptations; no new
general launcher or SDK instrumentation discovery is part of this proposal.
It must not copy historical hostnames/labels or activate two launchers.

Limits for THIS partial diagnostic, in binary GiB/MiB:

- One actual host, four ranks/LNC2, CPU/BLAS threads1 per rank; no full model,
  train step, checkpoint or evaluator. Same tensor ceilings as full proposal:
  448MiB/rank, parent256MiB, aggregate2GiB. Freeze a concrete tensor lifetime
  ledger and enforce harness allocations; these are not runtime/HBM totals.
- Initial free disk>=16GiB and MemAvailable>=24GiB, checked inside the original
  lock on each relevant filesystem. No cleanup or lowering a floor to fit.
- Independently running host supervisor samples at most1s apart: global
  MemAvailable, free bytes via statvfs, and summed VmRSS for itself plus all
  authenticated job-owned workers/descendants (including compiler descendants).
  Stop own job on MemAvailable<4GiB, sampled summedRSS>20GiB, disk<6.5GiB,
  unreadable/ambiguous live-process accounting, or sampling gap>2s. Preserve
  PID/starttime associations and account for adopted/double-fork descendants.
  A disappeared PID requires demonstrated exit, not assumed zero memory.
- Sample new job-owned cache/tmp bytes, cap6GiB; artifacts cap2GiB including
  temporary copies; combined log/metric cap128MiB. Enforce before harness writes
  with one serialized blob writer and reserve space before each write. Compiler
  cache/tmp growth remains sampled, with at most65536 entries and no symlinks,
  special files or external hardlinks. Unexpected paths/accounting fail closed.
  Bound pipe capture; never silently truncate a log and call it complete.
- Sampled guards are stop triggers, NOT hard kernel-enforced RSS/disk ceilings
  or exact peaks. There can be between-sample overshoot; record intervals and
  observed overshoot. No cgroup/virtual-address/RLIMIT_AS claim substitutes for
  physical RSS. If the independently reviewed monitor cannot satisfy this
  limited safety contract, remain unlaunchable; do not omit it.

Exact per-invocation HBM peak, SBUF allocation/spill, separate compiler/worker
RSS peaks, complete graph counts and hardware owner/execution traces are
UNAVAILABLE unless actual frozen-SDK logs establish them. Store explicit null
values with reason and scope, never zero. Retain emitted graph_stats/compiler
logs and timestamps, but do not infer no recompilation/no spill/cache hit from
silence. Source scratch1,052,164B/owner is a logical estimate only. F0 complete
channel-byte equality is behavioral two-owner coverage, not a hardware trace.
Missing those precision metrics is an explicitly declared limitation of this
PARTIAL_DIAGNOSTIC, and still prevents the unchanged full-suite qualification.

One monotonic deadline spans preparation, generation, native work, comparisons,
finalization and shutdown: **1200s total**, not1200+60. Work stops by t0+1100;
at most60s TERM grace, then bounded KILL/reaping, with the remaining40s reserved
for postflight/terminal receipt. Within the1100s work window: oracle/generation
<=180s; combined first-use synchronized calls<=600s; later calls plus their
comparisons<=240s; remaining80s for preflight/manifests. These sublimits do not
extend/reset the common deadline. First-use time conservatively includes
compilation+execution+sync, not a claimed pure compiler duration. Any later
synchronized call has60s maximum. Compile/transfer that fails to fit is a
diagnostic failure, never justification for skipping full comparison.

The packet must show enforcement with tiny synthetic stuck-child, worker-loss,
double-fork, budget, output-cap and missing-sample tests before review. Account
for KILL/reap and two15s postflight checks inside1200, rather than copying a
1140+60 wrapper that leaves no finalization time. An unkillable survivor or failed
postflight is CONTAINMENT_FAILED: retain protection, report it, and permit no
successor. This is an attempted execution bound, never a promise the OS can kill
an uninterruptible process. Later network artifact retrieval duration is reported
separately and authorizes no additional computation or next run.

## Artifacts, verdict and release boundary

Use lossless content-addressed full raw blobs with single-writer/dedup bookkeeping.
Canonical tangents total128MiB; two FP32 expected outputs64MiB, plus metadata,
IDs, weights, forwards and actual outputs. Reuse matching blobs only after full
comparison; every rank/route association remains. Mismatches retain complete
expected/actual bytes plus bounded coordinate witnesses. A digest without a
complete retained blob is missing evidence. Validate and retrieve the entire
manifest, four rank receipts, invocation provenance, actual source/environment,
CPU/byte negatives, raw resource samples, logs, timers and explicit all-observed
process/device/lock postflight. Missing/over-cap artifacts fail the diagnostic.

Allowed result: PARTIAL_DIAGNOSTIC_F0_F2_EXACT_OBSERVED, or an explicit failed/
incomplete/not-executed result. None is full K2 qualification, a production path,
speed/BPB evidence, an ordinary-path failure waiver, or permission for the next
run. F1/interleaving/empty groups, F3 maximum32768 skew, F4 changed-metadata replay,
F5 exceptions, actual wrong-dtype refusal, full instrumentation and eventual
producer/integration/full-step gates remain pending. This does not update
decide.py, controller state, submission bytes, train.py, adapter or tiny kernel.

Prospective unique label: `fs-20260926-k2-realshape-f0f2-partial-r1`.
No host is selected. Before launch a separate immutable binding must resolve
host/paths, exact source+support hashes/commit, complete argv/environment,
work/monitor/termination mechanics, clean tree and separate review receipts.
Required source environment shape is existing native_venv, LNC2/world4,
NEURON_CC_FLAGS=`--optlevel=1 --auto-cast matmult --auto-cast-type bf16`, seed42,
one thread/rank and new job-private caches; actual versions still require capture.
No executable launch command is invented before that implementation exists.

Only the Supervisor can select and later release the completed reviewed packet
to the existing sole Operator after current jobs/evals and all descendants exit,
STOP/breakers are honored, original device0 flock is acquired, and fresh ps AND
neuron-ls plus device-holder and resource checks pass inside it. Retain the same
lock through the job and postflight. Higher-priority clean/cold work keeps its
slot. This Researcher neither verifies its own proposal nor implements/operates it.
