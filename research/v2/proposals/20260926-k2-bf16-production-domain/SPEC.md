# K2 BF16-output production-domain design — Researcher/Analyst only

Decision: a separate narrow contract is supportable. Production implementation,
native exactness and useful speed are unresolved. **HOLD; not launchable.**
This document selects no job, changes no gate, and approves no implementation.
The unrestricted FP32 control failure remains FAILED. A new patch must receive
a genuinely fresh independent review; this author cannot provide that review.

## Frozen evidence and applicability

Research snapshot: `e018498e4bdd91ce5c93a184c3e7602a42228311`.
Actual adopted model: `7419927c7417b5666a984ec9c2b849feadabaaa9`,
`workflow/proposals/research-v2/train.py`, SHA256
`36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb`.
Use the exact argv in `research/v2/adoptions/20260926-clock36-confirmed.json`:
8x1024, attention0/2/4/6/7, initial cache, startup-adjusted schedule, LR.015,
seed42, pack32, context1024, batch131072, world4/LNC2. Recursive is absent.

`GPT.forward` immediately casts `self.transformer.wte(idx)` and
`self.value_embeds[str(i)](idx)` to COMPUTE_DTYPE. The Neuron dtype function
returns BF16. The constructor's `has_ve`/`has_attention` and adopted architecture
select **two** ordinary tables: WTE and VE7. Each is FP32 [8192,1024]; each
rank's one microbatch has [32,1024] IDs, T=32768. Both use the same idx.
These are direct source/config facts, not a full-model native trace.

Protected tiny diagnostic kernel:
`research/v2/lanes/k2-symmetric-consumer/consumer_nki.py`, SHA256
`2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae`.
Registration adapter:
`research/v2/lanes/k2-registered-forward-autograd/adapter.py`, SHA256
`d761a7274e50c21d5930bfa8b984da710ed257be73381fc17ea5f062c03f190a`.
The supplied-metadata consumer has hard assertions V256/T256/C128, two
programs, two row groups and feature halves64. It has no production metadata
builder. It cannot be enabled unchanged for the shapes above.

`analysis/20260926-k2-cast-boundary/{README.md,audit.py,result.json}` supplies
24 CPU cast cases plus authenticated native rank results. This proposal's
`domain_probe.py` independently checks 16 tiny CPU cases and all65,280 finite
BF16 encodings, authenticates the original four raw rank receipts, and records
all hashes in both normal/-O outputs. It imports no candidate or installed SDK.
Largest allocated test tensor has three elements. This is research evidence,
not an independent gate for a future implementation.

Original native archive SHA256:
`a210263a3f0cd67f4896cb76c55ae67198331053220432800639b4483b1d2563`.
Allfour rank aggregates remain `REGISTERED_FULL_CASES_MISMATCH_OR_ERROR`.
The unrestricted FP32-minimum ordinary compiled control returned0 vs1408
nonzeros. Native BF16-subnormal inputs were nonzero (32768 elements) and all
five tested routes matched (1408 nonzero outputs); FP32-minimum values passed
as BF16 were already zero. The later ordinary-only eager AND compiled failure
remains failed too. None proves a universal or documented FTZ behavior.

## Exact proposed contract

One causal change: a default-OFF training-only registered backward for **only
the two ordinary embedding lookups whose forward result is immediately BF16**.
Keep the forward gather and its existing BF16 cast, Parameter identities,
initialization/RNG order, checkpoint keys, optimizer, clipping, ownership,
collectives, schedule, data and ordinary evaluation route unchanged. No ngram,
RMSProp/table path, arbitrary FP32-output embedding, or eval adapter is included.
Future control is the same new source with the flag OFF, verified against36.

The registration output must have BF16 dtype by construction. Save per-call
immutable metadata. Backward accepts the actual BF16 output tangent and
widens each element exactly into FP32. Native compiled evidence must show this
boundary is reached and consumer output is used; the compiler may otherwise
change the apparent graph. Do not infer reachability from ambient BF16 mode.
Higher-order gradients and changed embedding options are outside the enabled
shape/call contract and must retain the ordinary path or explicitly reject.

For each destination row r and channel c, enumerate original flattened token
positions t with idx[t]=r in ascending order. Start A=positive FP32 zero and
perform **one correctly rounded FP32 addition per occurrence**:
`A <- RN32(A + widen_BF16(g[t,c]))`. Return a fresh dense FP32 dW. Untouched
rows stay positive zero. Preserve operation order across every tile, repeated
invocation, accumulation microstep and all channel owners. No grouped sums,
tile partial sums, reassociation, atomic duplicate updates, averaging, sparse
optimizer, FP64 alternate oracle or tolerance is allowed.

The finite input lattice has spacing q=2^-133. Each finite widened BF16 is
an integer multiple of q. Adding two finite lattice members yields another
multiple; when representable FP32 spacing is finer than q the sum is exact,
and when coarser its rounded value is still on that lattice. Thus any finite
intermediate of this recurrence is a q multiple. It cannot produce a nonzero
FP32 value of magnitude below q. **It can and does produce FP32 subnormals:**
`00810000 + 80800000 -> 00010000`. Both signs must remain tested. This closure
argument relies on correctly rounded addition with gradual underflow; it is
not a claim that the installed backend implements it.

Retain BF16-minimum inputs, normal-to-subnormal cancellation, cancellation
after large partial sums, and the old cross-tile counterexamples
`[2^24,1,-2^24]` and `[1,2^-24,-1]`. Those BF16-representable inputs still
distinguish token order from grouped accumulation. Retain both zero signs in
inputs and intermediate/result raw bits; inactive lanes preserve accumulator
bits by selection, never by multiplication or synthetic additions.

NaN, both infinities, finite-input overflow and infinity cancellation remain
in diagnostic coverage. Do not clamp, zero, filter, replace, or silently fall
back based on their values. Preserve raw inputs/results and exact exceptional
classification, infinity sign and zero sign; report NaN payload differences
rather than claiming bit identity. Existing finite-only gate must refuse all
nonfinite outputs, even if both routes agree. NaN equivalence is not a training
PASS. Any nonfinite ordinary/candidate whole step is a failed gate. If payload
behavior needs an explicit criterion, leave it unresolved for the fresh review,
not an automatic equal-NaN tolerance. The research probe records those cases as
FAIL_NONFINITE while verifying propagation; it does not approve them.

Only static dtype, shape and supported option guards may choose the ordinary
fallback. Arbitrary FP32 tangents must not be cast down to fit this contract.
Unsupported paths remain unchanged, including the known unrestricted FP32
failure; keeping that path is not qualification. Invalid IDs, corrupt/stale
metadata or observed output-dtype drift must fail closed before unsafe indirect
reads. No hidden CPU validation/data transfer may be omitted from cost accounting.

## Production design and costs — not an implementation

Retain the resident-row algorithm's ordering and two symmetric feature owners.
Generalize each owner to512 channels and64 groups of128 rows, with shared runtime
group maximums. Use runtime loops and bounded scratch, not32768 static bodies.
Each active row gathers its next ascending token position and one gradient
slice, adds once into resident FP32 state, then stores its512-channel result
once. Both programs visit identical row-group schedules; write ownership is
disjoint and complete. Existing native support at tiny dimensions does not
establish legal access descriptors, scratch allocation or compilation at512.

Metadata must be produced on device from the actual unchanged loader's device
idx, once per forward and reused by WTE/VE7. No generator introspection, CPU-ID
side channel, prepared-token mutation or dynamic Python control is permitted.
The earlier resident-row proposal's stable counting route is the bounded design
basis: counts[V], exclusive starts[V], ascending positions[T], maxima[V/128].
A [8192,256] row/tile prefix table costs8MiB. Persistent int32 metadata is
196,864 bytes (counts32KiB, starts32KiB, positions128KiB, maxima256B; +4B if
an explicit starts[V] endpoint is stored). Straight comparison counting does
268,435,456 row/token comparisons plus4,194,304 within-tile ordinal comparisons,
16384 row-group/tile bodies. These costs are paid once, not zero times.
A legal efficient producer still DOES NOT EXIST in the tiny consumer; it must
be independently implemented/reviewed before integration or useful timing.

Per table per rank: input tangent64MiB BF16, output32MiB FP32, FP32 weights32MiB.
Two tables mean128MiB useful tangent reads and64MiB output stores, excluding
forward gathers, IDs, metadata and ordinary gradient reduction/update traffic.
Let Q=sum of the64 row-group maximum counts. For T32768/V8192,
256<=Q<=32768, active fraction T/(128Q) lies between1 and1/128.
Padded adds/table are128*1024*Q: 33,554,432 to4,294,967,296. Without effective
inactive skip, BF16 requested gradient payload spans64MiB–8GiB/table; skip
can preserve64MiB useful payload but not remove descriptor/control overhead.
Both tables still require separate tangents and recurrences.

Per program a128x512 FP32 accumulator is256KiB. With FP32 value, updated and
selected temporaries, obvious live tiles total about1MiB plus metadata and
compiler overhead; BF16 staging may add128KiB. This arithmetic fits the24MiB
nominal SBUF budget but is not an allocation/spill/liveness proof. Metadata8MiB
is HBM scratch and must be tiled, not assumed globally SBUF-resident. Dynamic
loop lengths and512-channel descriptors are unresolved compile/performance
risks. Dense two-table all-reduce and optimizer work are unchanged.

Old four-table amortization does not transfer: this source has two lookups,
so the same metadata producer cost is amortized across half as many gradients.
The old segmented DMA whole-step gain1.417% used another source/shape and
cannot predict a >=5% win here. Current P profile exposes no separate embedding
device timing, so its backward graph cannot be treated as an embedding budget.

## Cheapest useful next gates and STOP rules

1. First, CPU-only independent review of the new BF16-only contract, scalar
   recurrence, casting provenance, metadata stability and exact default/OFF
   behavior. Implementer may only build after Supervisor selection. Any future
   patch needs its own fresh reviewer; this proposal is not its approval.
2. After separately reviewed implementation/packet, use a native **component**
   at real V8192/T32768/C1024, never a full model initially. Start consumer with
   authenticated supplied metadata, then measure real producer+consumer including
   uploads/launch/dependencies. One table's live weight+tangent+gradient plus
   metadata/scratch is roughly<200MiB before compiler/runtime; use sequential
   fixtures and a declared memory limit. Test balanced, interleaved duplicates,
   boundary IDs, all-one-row, changed-metadata/tangent replay, all required
   subnormal/cancellation/exceptional cases and unused rows. Component synthetic
   skew is a stress test, not a training-frequency estimate. Record cold compile
   cost, peak memory, graph count, route traces and exact outputs on allfour ranks.
   STOP on any mismatch, wrong dtype, invalid address, graph recompilation by ID
   content, zero/stale-consumer false acceptance, compiler failure/spill explosion,
   timeout or missing artifacts. Do not shorten hard fixtures to get a PASS.
3. Before a whole model, compare the complete two-table component (metadata
   amortized only twice) against ordinary embedding gather/cast/backward for
   the same shapes and inputs. If it is consistently slower on both balanced
   and interleaved fixtures, park; no full-model experiment is justified by a
   mathematical-domain proof alone. A faster component remains inconclusive:
   no full-model embedding fraction is known. All-one-row timeout parks a
   universal production claim unless a separately proposed semantic fallback
   is reviewed; do not add such a fallback ad hoc.
4. Only after all source/default-OFF, full-size CPU where required, native
   correctness and role checks, propose one matched whole-step timing screen.
   Minimum informative design: same host/day, A(control)-B(candidate)-A(control),
   each fresh seed42 process, same exact source/argv except the single flag,
   production world4/LNC2/pack32,5 startup boundaries +5 settling steps +30
   timed complete steps. Report raw per-rank step walls and the rank-max critical
   path; include loader fetch, metadata, copies, backward, clipping/collectives,
   optimizer and cache refresh. No prepared metadata or cached tangents. Exact
   numerical parity uses a separate replay with identical prescribed schedule
   samples; real wall-clock scheduling can differ in the timed screen and must
   not be misreported as matched optimizer inputs. No scorer needed for timing.
   Require candidate token rate>=1.05x **each** bracketing control (equivalently
   step duration<=control/1.05), stable windows and no new stalls. If control
   medians drift>2%, mark timing inconclusive, not a win; do not launch quality
   on a noisy interval. Report medians/p95/windows, not isolated kernel speed.
5. A timing win only earns consideration of separately approved true cold-five
   (uncapped fifth<=900s, planning870), cold fixed1800 and unchanged fresh public20M/
   context1024 against comparable same-host controls. Official charged budget
   and checkpoint reserve remain unchanged. BPB selection uses existing exact
   rules, mean delta<=-0.0010. No automatic full run, adoption or submission.

No native executable command, host assignment, cache action or breaker reset
is issued here. Every proposed native stage requires its own immutable complete
launch proposal, independently reviewed packet and sole Operator interlock.
Abort/timeout in one label remains failure; any retry needs a new labeled proposal.

## Unresolved before implementation or host spending

The new contract needs independent acceptance. Production metadata generation,
512-channel native legality, loop liveness and worst-skew runtime are unproven.
Native actual cast-domain reachability and ordinary-vs-candidate recurrence over
the BF16 reachable domain require new evidence. NaN payload policy must be
explicitly resolved without accepting nonfinite training. Current profile cannot
quantify an attainable5% whole-step saving. No production hash/launcher can be
frozen until implementation exists; the completed deliverable is this design,
not a completed launch proposal. Suggested next action is CPU-only specification
review and implementation scoping, not a repeat of the failed unrestricted gate.

## Environment incident and reproducibility

Initial `git worktree add -b codex/k2-bf16-production-contract
/tmp/trainium-k2-bf16-production-contract e018498e` failed with repeated
`unable to write file` and `fatal: Could not reset index file to revision HEAD`.
No cleanup/deletion was performed. Subsequent df showed /tmp tmpfs7.5G/5.9G
used/1.6G available, root filesystem466G/420G used/25G available. Capacity
during checkout was not captured; insufficient tmpfs is a hypothesis, not proven.
A no-checkout worktree on the root filesystem was then restricted to this output
directory using non-cone sparse checkout. Only new proposal artifacts are changed.
CPU command is `/home/poojith-devan/venv/bin/python -B [domain_probe.py]`, repeated
with `-O`; no host/network/SDK import or full-model tensor was involved. The
original source, evaluator and all failure receipts remain unchanged.
