# F0/F2 fixed-purpose packet — implementation candidate, default disabled

Mechanics repair note: REPAIR_SPEC.md supersedes the original helper lifetime,
finalization and peer-admission descriptions below. The original independent
refusal remains preserved; repair is not approval or launch authority.

Authority is the separately selected `20260926-k2-f0f2-partial` SPEC/JSON at
Researcher ccece0f3 / canonical24b6b096 and selection922fcbf3. This implementation
does not edit that contract or the full six-fixture contract. It is not a release,
independent verification, native result, model integration, or speed claim.

`binding.py:BINDING = None` refuses before host checks, output creation, SDK
imports or child creation. A future independent host-specific binding/review must
freeze hostname, absent absolute output_root, project_root/project_commit,
scientific_source path, complete stop_roots, exact label, independent_review_sha256
and supervisor_release_sha256. Binding is source-pinned too: filling it changes
the packet and requires renewed immutable pins/review/release. No host is chosen.

## Components and donor deltas

- `consumer_nki.py` is a byte-identical copy, SHA
  `9425390c96053a1b24d4addf8b61b1a87c4993f3551f08c7cdcefcf142d84dcd`.
- `idle_guard.py` is a byte-identical donor copy, SHA
  `adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737`, from
  `20260926-k2-registered-native-trn22`. The original outer lifetime flock argv,
  inode/FD/parent authentication and both parsed ps+neuron-ls checks are retained.
  The driver supplies a bounded-pipe runner instead of unrestricted capture:
  two checks at most15s each, 2MiB combined output per check.
- `monitor.py` adapts the census/owned descendants, subreaper, bounded directory
  walk and TERM/KILL ideas in `20260926-p-offline-decoder-trn21/decoder_support.py`.
  New concrete deltas: pidfd identity/exited evidence; concurrent VmRSS sum;
  MemAvailable/statvfs samples; separate accounted artifact/cache/tmp/log roots;
  symlink/special/hardlink refusal; observed escaping writable regular-file FD
  refusal; nominal0.5s sampling with >2s gap failure and >1s scan-duration failure.
  No donor CPU, address-space, CPU-affinity or no-descendant restrictions remain.
- `host_driver.py` adapts the fixed native driver's explicit native_venv argv,
  six job-private cache variables, fresh metrics and original lock. It spawns
  only the coordinator, which spawns only the fixed four-rank torchrun worker.
  There is no arbitrary argv runner exposed by the native entrypoint. The tiny
  bounded helper accepts argv only for preflight git/idle checks and tests.
- Portable `oracle.py`, `fixtures.py`, `generate.py` are byte-identical to the
  committed core3f6bf53b. `artifacts.py` starts from that core but adds unbuffered
  writes, full-byte duplicate comparison and read-only finalized blobs.
  `test_oracle_core.py` is the original tiny core test unchanged.

All Python sources have explicit hashes in SOURCE_PINS.json. Host, coordinator
and workers check every pin before native work; every rank receipt includes pins.
The external reviewed packet digest/commit remains necessary: a self-reported
mutable manifest alone is not source authentication.

## Actual routes, references and provenance

Coordinator freezes both full canonical fixtures, metadata, exact integer-RN32
expanded outputs, deterministic weights and independent gather/cast forwards
before creating torchrun. The weight formula is exactly
`(((17*r+3*c)%257)-128)/128`, not the older optional core weight helper (unused).
F0/F2 gradient recipes and whole ordered-timeline class expansion are unchanged.
One oracle-addition budget and180s absolute generation deadline span both cases.
No full-size generation has been executed locally.

Each worker imports only the pinned consumer and executes exactly
`build_kernel(enabled=True)` followed by the returned kernel. There is no
autograd adapter, wrapper around kernel arithmetic, dtype shim or model import.
Ordinary eager/compiled controls use `F.embedding(weight/ids)` in the correct
PyTorch argument order (`F.embedding(ids, weight)`), immediate BF16 cast and
actual backward with the same BF16 tangent. One compiled callable spans F0/F2.
The seven frozen routes execute in order on four ranks; the coordinator waits
for all four exact receipts before advancing. FIRST_EXCEPTION aborts scheduling;
socket closure and the independent host supervisor contain failures. There are
no route-rendezvous device collectives. SDK process-group init/destroy may still
block and are covered by the external work/termination bound.

Complete actual input tensors are round-tripped before/after every route. The
single writer stores every returned byte before comparison; returned metadata
is independently revalidated against returned IDs. Direct F0 output stays live
through repeat and F2; all its bytes are rechecked after both. Input/callable and
retained object identities are checked in addition to byte equality.

The coordinator authenticates Unix peer UID/PID/starttime and ancestry to its
exact torchrun, rank/world identity and all source hashes. Each route has a fresh
challenge and required pre-input -> call-start -> call-end -> actual-byte capture
-> post-input sequence. The worker checks that captured objects are native-device
tensors; CPU oracle objects cannot be captured as native outputs. Workers are
not sent oracle/reference output paths. A reference blob ID is not accepted as
an actual result: the channel must stream full bytes and the parent records its
own call boundaries and per-blob associations before comparison. Every mismatch
retains its full actual blob and expected association, plus bounded witnesses.

This provenance is source/process/control-flow evidence, NOT a hardware execution
counter or defense against arbitrary same-UID code that can rewrite the reviewed
packet, monkeypatch the trusted SDK or forge all source bindings. The independent
reviewer must test bypass/counterexamples against the frozen source and external
pin, not accept a dictionary merely claiming native provenance. Unknown hardware
execution/owner traces remain null. F0 all-channel bytes provide behavioral
coverage only.

## Storage and tensor lifetime ledger

Only the coordinator writes artifact blobs/manifests. Four socket handlers share
one serialized writer lock; a complete incoming object's size is reserved before
the ready-to-send reply, then every chunk is reserved before writing. Canonical
chunks are <=256KiB, transfer chunks<=1MiB. Full-byte duplicates share one finalized
read-only blob only after comparison. Complete temporary duplicates count toward
2GiB before their own duplicate scratch is removed. No previous run is cleaned.
The entire job artifact tree is additionally sampled, including pending files.

| Harness-owned live item | Logical maximum |
| --- | ---: |
| Current BF16 tangent, native | 64MiB |
| Current IDs and four metadata tensors | <0.5MiB |
| Retained first F0 gradient | 32MiB |
| Deterministic FP32 native weight | 32MiB |
| Current gradient, distinct from retained | 32MiB |
| Ordinary BF16 forward | 64MiB |
| One CPU transport copy | 64MiB |
| Load CPU mapping while native input allocated | 64MiB, occurs before forward/gradient |

Each worker reserves known storage before native allocation/copy and records
allocate/release events;448MiB is a fail-closed cap. F0 tangent and metadata stay
identical across repeat; old nonweight inputs are released before F2 upload.
Autograd gradient and forward references are released before the next route.
The coordinator uses streamed numeric arrays plus one64MiB read-only gradient
mapping; no torch/device tensors. Its numeric work buffers are well below256MiB;
four448MiB worker reservations plus256MiB parent equals2GiB. These are logical
harness allocations, not hidden compiler/runtime/allocator/HBM totals. Sampled
all-owned RSS is the separate physical safety trigger. Four raw transfers never
become four64MiB coordinator buffers.

## Time and resource safety contract

One inherited t0 reaches coordinator/driver. Work stops by t0+1100, TERM grace is
at most60s and ends no later than1160, KILL/reap attempt ends no later than1165.
Two15s postflight checks must start before1167 and finish by1198, leaving terminal
receipt headroom under1200. One generation budget180s; combined first-use routes
600s; later routes plus their comparisons240s (individual later route deadline
is conservatively60s including transfers); other work80s. These never extend the
common deadline. First-use wall time includes transport/compare overhead as well
as compilation/execution/synchronization; it is not pure compilation time.

Initial16GiB free and24GiB MemAvailable are mandatory inside the original lock.
Running stop triggers: MemAvailable<4GiB, sampled sumRSS>20GiB, any relevant job
filesystem free<6.5GiB, cache+tmp>6GiB, artifacts>2GiB, logs+metrics>128MiB,
ambiguous process accounting, unsafe paths/entries, or missing/slow samples.
At most65536 entries across accounted trees; external hardlinks are rejected
conservatively by refusing all hardlinked regular scratch files. Driver-owned
pipe/monitor output reserves a120MiB subcap, leaving8MiB for other metrics; the
combined128MiB total is sampled. Output-cap violations explicitly mark failure,
not a silently truncated success.

Sampled writable-FD census catches observed writable regular files outside the
accounted roots, but is not syscall interception and cannot observe a file opened
and closed wholly between samples. Likewise sampled RSS/disk may overshoot between
samples. All raw sample intervals/values and exits are retained; no exact peak or
hard memory-ceiling claim is made. Exact HBM/SBUF/spill, split compiler/worker
peaks, complete graph counts and owner/execution traces are explicit nulls with
reason. Emitted graph_stats and compiler output are retained for independent
inspection; silence does not establish cache hit/no recompilation/no spill.

If containment/postflight fails, terminal status is CONTAINMENT_FAILED and the
driver deliberately remains idle holding the original lock for human intervention.
This is a protection hold, not extra computation or a successor. The1200s number
is attempted work/termination/postflight, not a guarantee Linux can kill an
uninterruptible process. No automatic retry, breaker reset or lock release follows.

## Local checks and remaining review

`python test_packet.py` and `python -O test_packet.py` exercise tiny byte negatives,
metadata corruptions, forged/missing association, CPU-output bypass rejection,
monitor thresholds/gaps, tensor/write/output caps, actual PID/start/RSS sampling,
stuck-child TERM/KILL, actual worker loss and double-fork adoption/reaping. The
largest numeric output fixture is eight words; process/output fixtures are small
bounded standard-library helpers. Retained raw receipts name all temporary evidence.
`python test_oracle_core.py` and `python -O test_oracle_core.py` each check20,006
scalar RN32 pairs against NumPy, order/subnormal witnesses and core negatives.
Normal/optimized raw outputs are included. Dependencies: Linux Python with pidfd
support, standard library, NumPy for these checks; no SDK/install needed locally.

The actual native packet has NOT run. Full-size disk/time/transfer feasibility,
frozen native SDK compatibility (including the fixed512-channel consumer),
ordinary-control exactness, complete four-rank evidence and native containment
remain unobserved. Only fresh independent diff/harness review can judge whether
the implementation is ready for a separate host-specific release. It cannot
qualify the unchanged full six-fixture gate or authorize production integration.
