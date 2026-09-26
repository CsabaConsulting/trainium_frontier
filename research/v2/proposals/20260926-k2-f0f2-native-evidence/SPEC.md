# K2 F0/F2 native evidence collection with an explicit runtime trust boundary

Completed Researcher design; **NON-LAUNCHABLE**. This adds a separate proposal,
not a retrospective PASS or an edit to any failed gate. Only proposal/spec files
are changed. No host was contacted, no source or fixture was changed, and no
tensor, staging packet, host reservation or launch is supplied.

## Decision and evidence ordering

Recommend a host-only trn22 binding of the repaired seven-route packet, subject
to separate selection, implementation, fresh independent review and sole-Operator
release. The question remains whether the exact consumer and required ordinary
controls produce the exact frozen F0/F2 results on four real LNC2 ranks.

The selected partial SPEC, SHA256
`777cebb9bb9b4d191eccc21541b727b5470f23921d3ca316c8487b5eb74d1519`,
requires a fresh reviewer to demonstrate that bypassing the consumer cannot
create an accepted receipt. Its unconditional interpretation is **NOT_ESTABLISHED**.
That historical status and the fresh review's HOLD remain unchanged.

The fresh review at `research/v2/verification/20260926-k2-bound-worker-fresh/REPORT.md`,
SHA256 `7faad08ee432c4c318e3bec55d1e086cac041523b7997347aca8eab73b5068b5`,
establishes two different facts. Its own CPU tests reject missing invocation,
CPU oracle return objects, missing/forged blob associations, domain errors and
byte corruptions in the actual worker/receiver path. Its positive routing test
also accepts an explicitly substituted native-looking facade. The latter changes
the trusted SDK/kernel seam; the review explicitly does not establish an
exploitable bypass in the unchanged production path. It also cannot attest to
hardware execution. Do not relabel the positive mock as a native observation.

For this new diagnostic only, propose the following auditable ordering. Fresh
review must accept this distinction before a release can exist; the Researcher
cannot approve it. A genuine defect reachable through the frozen packet's
supported inputs, without replacing trusted source/runtime behavior, remains a
pre-launch failure. Nothing moves a discovered numerical or containment failure
to an optional post-launch check.

| Required before native release | Measured by this native diagnostic | Outside either claim |
| --- | --- | --- |
| Reviewed immutable source and import/call path; exact scalar oracle counterchecks; malformed metadata/receipt/blob and supported-path bypass negatives; distinct-worker admission; repaired containment/deadline/cap mechanics. CPU mocks must be disclosed. | Actual installed native runtime, four distinct admitted workers, three direct and four ordinary invocations per rank, complete full-shape inputs/forwards/gradients, synchronization and retained-F0 evidence, measured feasibility and containment. | An unconditional proof against a substituted/hostile SDK or arbitrary same-UID source rewrite; hardware execution counters, exact HBM/SBUF/spill/owner traces without emitted evidence. |

The trust boundary is the externally pinned packet, expected interpreter and
installed native SDK, and OS process/device observations. The real run must use
that unmodified environment: no monkeypatch, fake tensor facade, import hook,
alternate kernel, oracle-return shim, simulator, runtime replacement or extra
test-only path. No user-supplied expected blob may masquerade as a call result.
The reviewed worker's direct assignment from the consumer, real native tensor
semantics, synchronized capture, admitted PID/starttime and independently
retrieved full blob form **source/process/behavioral evidence under that trust
boundary**. Even a successful real run is not unconditional hardware attestation.

The old unconditional clause is neither silently marked passed nor offered as
the authority for this new run. This proposal must be separately selected and
reviewed as a partial evidence-collection experiment. If a fresh reviewer finds
an unresolved source-preserving bypass, unsafe release or unverifiable runtime
binding, keep HOLD and return a concrete defect for a separately reviewed repair.

## Immutable science and all 28 required observations

Consumer `research/v2/lanes/k2-bf16-production/consumer_nki.py` remains SHA256
`9425390c96053a1b24d4addf8b61b1a87c4993f3551f08c7cdcefcf142d84dcd`.
Each rank calls `build_kernel(enabled=True)` once and the returned
`resident_consumer[2]` directly. No wrapper, arithmetic, dtype, adapter,
registration or metadata-producer change is proposed.

Domain: V8192/T32768/C1024; gradient[32768,1024] BF16; counts/starts[8192,1],
positions[32768,1], maxima[64,1] int32; output[8192,1024] FP32; 64 groups of
128 rows, two owners of 512 channels; world4/LNC2; seed42. Supplied synthetic
metadata is diagnostic input only. No checkpoint, full model, training step,
dataset, scorer, optimizer or evaluator is needed.

F0 and F2 are exactly the existing recipes in the pinned partial SPEC and the
unchanged six-fixture SPEC (`c5ffec92cbe0d3036b1a740e6d42429917453eb06de75ad35cd6728fb213b485`).
F0 uses id[t]=t mod8192 and raw BF16 tangent bits0x3800+c. F2 retains every
original ID override, eight channel classes, ordered cancellation, reachable
signed subnormal and signed-zero witness. No new, removed, shortened or repaired
fixture is permitted. All four ranks receive identical complete canonical bytes.

On each rank, the mandatory order remains:

1. F0 direct.
2. F0 direct repeat with the same callable and input tensors.
3. F0 ordinary eager.
4. F0 ordinary compiled.
5. F2 direct.
6. F2 ordinary eager.
7. F2 ordinary compiled, reusing the compiled callable.

All four ranks complete an exact route before the next route is scheduled.
Success requires all 28 actual rank-route observations. First failure stops
scheduling; preserve completed and failing evidence and mark remaining routes
NOT_EXECUTED. Never run remaining routes to manufacture a full count after a
failed gate, and never reduce the required count for a success claim.

The integer-exact ordered RN32 oracle, canonical generation, separate metadata
validator, full 33,554,432-byte gradient comparisons, zero signs, both owners and
all 1024 F0 tags remain unchanged. Expected bits come from IDs/tangents, not from
native outputs or consumer metadata. The ordinary routes retain FP32 weights
`w[r,c]=(((17*r+3*c)%257)-128)/128`, ordinary embedding, immediate BF16 cast and
actual backward with the same BF16 tangent; full independent gather/cast forward
comparison is mandatory. An ordinary-control mismatch fails this diagnostic.

Every actual input is round-tripped and fully compared before and after each
route; returned metadata is revalidated. The first direct F0 output stays live
and is fully reread after the repeat and after F2 direct. Preserve the repaired
four-distinct-worker admission, source pins, invocation ordinal/challenge,
actual call boundaries, streamed bytes and complete retained blobs. A passing
scalar oracle, CPU mock or hash without its complete actual blob cannot fill a
missing native observation. Full-size expansion is unexecuted until this job;
the independent scalar review does not assert all expanded words are correct.

## Smallest implementation delta and fresh review

Use the existing disabled trn21 copy as the mechanical donor, not as launch
authority: `research/v2/proposals/20260926-k2-f0f2-trn21-binding/packet/` at
snapshot `608f2dd016b13e965ff34ad496c56063145e510c`.
Its SOURCE_PINS.json SHA256 is
`aae834cbb39538f9778f1329ca0b176fc2f92bc81d7c535926e2b5adc3c4ffbe`;
worker SHA256 `711378567cf7e7a7af6aabf5bb2dc023ce79620ac91a205020f482596b5df588`;
coordinator SHA256 `3d0fe301b55a84f64722bdea644ce1a020f8af9c4f5cd52f9dc49cdea5846912`.

A separate Implementer may prepare a new disabled copy with host/path/label,
remote project/source-anchor and receipt binding fields, corresponding exact
argv/environment manifests and externally authenticated pins. Preserve consumer,
oracle, fixtures, generation, worker, coordinator, evidence/artifacts, monitor and
idle guard byte-for-byte. Preserve the four repaired lifecycle/admission fixes
and all limits. The only additional host-driver delta is the read-only anchor
guard's literal SHA36a862 -> SHAe1e2f686, matching the known trn22 file below;
neither train.py is executed by this diagnostic. This is an explicit identity
binding change, not a consumer or ordinary-control change. No runtime patch or
numerical helper change is authorized by this design. A discovered incompatibility
requires an explicit separate repair
diff, review and new label if a native attempt already consumed this label.

Fresh review uses only the new exact diff/spec/receipts, an actual fresh session,
and its own executed small exact reference under the repository amendment. It
must explicitly state which archived narrow receipts it relies on, inspect their
pins, and check donor identity and host-binding negatives; archived CPU acceptance
is not whole-packet/native acceptance. In particular retain both the original
mechanics refusal and four-repair acceptance, plus the latest narrow seven-route
review. Do not rerun a full model on CPU or introduce a new scientific fixture
merely to demonstrate call-path behavior.

## Explicit intended host, ungranted staging and resource budget

Proposed host: **trn22**, hostname `ip-172-31-32-253`, original common lock
`/home/ubuntu/.cache/trainium_freshstart/device0.lock`, device66305/inode832589.
Unique prospective label: `fs-20260926-k2-realshape-f0f2-trn22-evidence-r1`.
Proposed absent roots: `/home/ubuntu/k2-f0f2-trn22-evidence-r1-packet` and
`/home/ubuntu/k2-f0f2-trn22-evidence-r1-output`. Use the existing project
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1`.
Its actual clean HEAD must be obtained and explicitly bound before staging.
Use the retained, independently authenticated read-only anchor
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet/train/train.py`,
SHA256 `e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43`.
The accepted terminal archive binds this exact path's bytes; the Operator must
recheck their current existence/hash before staging and under the launch lock.
Do not copy or alter this protected source, write into its old packet, or confuse
its retained checkpoint with a K2 input. The anchor is a hash check only and is
never executed by K2; the numerical consumer/control sources remain unchanged.

At the frozen snapshot, the accepted clean full-cold run and attached eval have
ended. Retained postflight records all702 observed identities absent at
05:58:18.149622UTC, no device holder and original832589 free; postflight SHA256
`ebbbf6ce4732cc01ed6ead64787239d1208d5da3fa8db120f8ae82d0576da418`.
Last recorded trn22 free bytes17,294,692,352 exceed the unchanged16GiB
(17,179,869,184B) floor by114,823,168B (about109.50MiB). trn21's approved-cleanup
result14,746,214,400B remains2,433,654,784B below that floor. These are recorded
facts, not a new current host observation or a reservation. No checkpoint
transfer is required; donor packet logical size118,386B is only a local planning
observation, not staged allocated size or a present disk pass.

Before even staging, the Supervisor must issue a separate scoped staging
authorization to the sole Operator for the exact frozen bundle/absent paths;
this proposal supplies none. The Operator must confirm current target absence,
exact clean project/source/runtime facts, available space and protected-job
quiescence. Recheck the full16GiB floor after staging, inside the original lock,
immediately before computation. No saved postflight or projected footprint can
replace a fresh check. Any floor failure parks this host variant. No cleanup,
checkpoint/cache deletion, threshold reduction or automatic trn21 fallback.

All original partial-proposal limits are copied verbatim into the JSON. In brief:
1200s attempted total lifetime,1100s work stop, at most60s TERM and40s reserved
KILL/reap/postflight/terminal work;180s generation,600s cumulative first use,
240s later routes/comparisons,80s other work,60s individual later call. Work
windows never reset or extend the common deadline. Four448MiB logical worker
tensor caps,256MiB parent and2GiB aggregate;24GiB initial MemAvailable; sampled
stop at <4GiB MemAvailable, >20GiB owned summedRSS, <6.5GiB disk, >6GiB cache/tmp,
>2GiB artifacts including temporaries, >128MiB logs/metrics, >65536 entries or
ambiguous/missing accounting; sample interval<=1s, gap>2s fails. No hidden
compiler/runtime allocations are counted as zero or as logical tensor storage.

Native_venv remains `/home/ubuntu/workspace/native_venv`, four actual ranks/LNC2,
seed42, CPU/BLAS threads1, compiler flags exactly
`--optlevel=1 --auto-cast matmult --auto-cast-type bf16`. The separate binding
must expand every argv/environment path and six fresh private cache paths;
only authenticated inherited lock FD and common monotonic t0 are dynamic.
Expected interpreter/SDK versions and import origins need read-only evidence
from the actual installed environment and must agree with the executed worker
receipts. Do not install packages or assert that metadata is SDK attestation.

Distinct Supervisor launch release follows fresh independent packet acceptance.
The sole Operator obtains original exclusive nonblocking flock, fresh successful
ps AND neuron-ls plus device-holder checks, STOP/breaker and prior-job checks,
and all initial resource/source/runtime gates inside that lock. Hold it through
native work and final all-observed-process/device postflight. No second launcher,
automatic next run, lock replacement, AWS action or breaker reset. Containment
or final evidence failure retains the protection hold; preserve all raw errors.

## Results and remaining gates

Retrieve complete lossless canonical/reference/actual blobs, all rank-route
associations, sources/argv/environment, worker admission/process identities,
full pre/post transport and retained-F0 evidence, raw resource samples,
timestamps/compiler logs and all-observed-process/device/original-lock postflight.
Independent actual-result review must authenticate those artifacts before use.
Failure/incomplete/not-executed outcomes remain durable; no digest-only success.

The sole positive result allowed is
`PARTIAL_DIAGNOSTIC_F0_F2_EXACT_OBSERVED`, explicitly scoped to the reviewed
source/runtime and these28 observations. It is not native-production or full K2
qualification, unconditional consumer attestation, speed, BPB or submission
readiness. Exact HBM/SBUF/spill, separate compiler/worker RSS peaks, complete graph
count and hardware-owner/execution trace remain null with their actual reason
unless emitted evidence establishes them. Sampled maxima are not exact peaks.

Preserve FAILED_PRESERVED unrestricted FP32/custom and required ordinary-control
subnormal outcomes, original mechanics REFUSE, and unconditional bypass proof
NOT_ESTABLISHED. F1/F3/F4/F5, worst32768 skew, other replays, exceptional values,
actual wrong-dtype specialization refusal, full instrumentation, metadata
producer, integration/autograd and whole-step>=5% timing remain separate future
gates. No training/evaluator/selector/submission file or controller release bit
is changed by this proposal. A success here authorizes no successor.
