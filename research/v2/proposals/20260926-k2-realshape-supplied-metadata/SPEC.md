# K2 real-shape supplied-metadata correctness falsifier

Completed Researcher design only. **NON-LAUNCHABLE; selection, separate harness
implementation and independent harness review are missing.** No production,
autograd adapter, metadata producer, model, training data or speed experiment.
One prospective four-rank/LNC2 synthetic component; no automatic next stage.

## Immutable subjects and question

Consumer source `research/v2/lanes/k2-bf16-production/consumer_nki.py` SHA256
`9425390c96053a1b24d4addf8b61b1a87c4993f3551f08c7cdcefcf142d84dcd`;
CONSUMER_SPEC SHA256
`ca4b1c6c00c70ccf9e43282951b4251b39882f091d7601ff957c576765573d7b`.
Production-domain specification SHA256
`5ae7e3b5cc4c072f0f72b0314d736bf8a65285722239866a9d1cbf057285090b`.
Source snapshot for this design: `f89f35a5a242b85cbab805fc91a26c90ce29432f`.
Protected tiny consumer2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae
and all original unrestricted FP32/ordinary-control failures remain unchanged.

Independent source-math report
`/home/poojith-devan/k2-consumer-independent-review-PJCqBe/REVIEW.md`, SHA256
`695c45cc868098bd613250317976673ca555be9fc8fc4520f5a86c31628b5cdf`,
is evidence of SOURCE_MATH_ACCEPTED_NATIVE_COMPONENT_UNQUALIFIED, not authority.
It specifically leaves actual512-channel descriptors, SBUF allocation/liveness,
rounding, inactive skips and32768-round skew runtime unresolved.

Question: does the unchanged real-shape supplied-metadata consumer execute
correctly at V8192/T32768/C1024, BF16 input, FP32 output, both512-channel owners,
on allfour actual ranks, within the declared component budget? A finite exact
component result cannot establish registration/autograd integration, production
metadata cost, whole-step improvement, BPB, cold training or submission readiness.

## Exact call and reference domains

Each rank uses one callable `build_kernel(enabled=True)` -> resident_consumer[2].
No explicit rewrap, new kernel specialization, arithmetic edit or dtype shim.
Gradient[32768,1024] BF16; counts/starts[8192,1], positions[32768,1], maxima[64,1]
int32; output[8192,1024] FP32. Owner0 channels0..511, owner1 channels512..1023.
Allcalls remain full geometry. CPU fixtures are synthetic diagnostic metadata,
never a permitted production route. Seed42 fixes any harness randomness; no
rank-dependent seed hunt. Use identical canonical fixture bytes on allfour ranks
and authenticate actual worker identity/invocation receipts separately.

The harness author must implement its **own** ordered binary32 oracle, not import
consumer tests, source prototype or the reviewer's script. Decode BF16 raw bits,
widen exactly, accumulate per destination/channel in ascending original token
order starting from positive zero; after each occurrence apply integer-exact
IEEE binary32 round-to-nearest/ties-even. No torch sum/index_add, segment/tree
sum, FP64 rounded-once sum or tolerance substitutes for that oracle. Independently
check it on small ordinary CPU embeddings and explicit rounding/subnormal witnesses
before packet review. Fresh Verifier must supply its own reference/counterchecks.

Full geometry does NOT require33,554,432 Python scalar additions for every
route. The fixtures below have explicit repeated channel/row timelines. The
oracle can compute one recurrence per rigorously identical timeline and expand
its bits to the full32MiB expected output. F0 requires1024 distinct channel
recurrences of four additions, then8192 copies of each row. F1–F5 have at most
eight channel classes and therefore at most32768*8 ordered additions each.
Total oracle arithmetic is <=1,314,816 additions for six fixtures, plus output
expansion/full byte comparisons. This reuses identical **whole recurrences**;
it never groups occurrence sums or changes order. The implementation must derive
row membership independently from IDs, not trust consumer counts/positions or
assume equivalence classes whose input bytes have not been validated.

## Six frozen fixture recipes and call order

All unspecified tangent entries are positive BF16 zero. Flattened token order
is0..32767; channels0..1023. q=2^-133. Integer division/fixed bit recipes require
no RNG. Harness must freeze complete recipes, little-endian raw inputs, expected
outputs, metadata hashes and all witness coordinates before native execution.

| ID | IDs and BF16 tangent recipe | Required witness |
|---|---|---|
| F0 balanced/channel tags | id[t]=t mod8192. Every g[t,c] has BF16 raw bits0x3800+c. | Four occurrences/row; all8192 rows, every distinct channel including0/511/512/1023; both-owner descriptor coverage. |
| F1 interleaved/unused | j=t mod4096; id=j except j4095 maps8191. Let k=floor(t/4096), a=c mod8; g=P[(k+a) mod8], P=[1,1/2,-1/4,1/8,q,-q,2^-24,-2^-25]. | Eight occurrences/used row; row127/128 and8191, manyunused rows and completelyempty middle groups, ordering with nonzero outputs. |
| F2 exact boundary witnesses | Begin id[t]=t mod8192; override t0/128/129->row0; t1/130->row127; t2/131->row128; t3/132->row8191; t4/133->row4095. For c mod8=0 set g at t0/128/129=[2^24,1,-2^24]; class1=[1,2^-24,-1]. Class2 at t1/130=[2^-126+q,-2^-126]; class3 at t2/131=[-2^-126-q,2^-126]. Class4 at t3=q; class5 at t3=-q. Class6 at t4/133=[-0,+0], class7=[-0,-0]. | Cross-token128 cancellation; positive/negative reachable subnormals and minimum inputs; zeros/inactive accumulator preservation. Same classes cover both owners. |
| F3 maximum skew | Every id=8191, exactly32768 occurrences. For a=c mod8 choose repeated four-value pattern: a0=[2^24,1,-2^24,q]; a1=[1,2^-24,-1,0]; a2=[q,-q,q,q]; a3=negative of a2; a4=[1/256]*4; a5=[-1/256]*4; a6=[-0,+0,-0,+0]; a7=[1,-1,1/2,-1/4]. | One row must execute all32768 rounds; all63 earlier groups empty; no fixture shortening or changed loop cap. |
| F4 changed metadata+tangent | id[t]=(3*(t mod8192)+17) mod8192. k=floor(t/8192), a=c mod8; g=R[(k+a) mod4] with R=[2,-1/2,1/4,q]. | Same input shapes, changed counts/starts/positions/maxima and tangent bytes; stale/F1 output cannot pass. |
| F5 exceptional diagnostics | Default id=t mod8192. All channelclasses use c mod8. class0: t0 quietNaN(raw0x7fc1); class1:t127 +inf; class2:t128 -inf; class3:t8191 +inf andt16383 -inf; class4:t4095/t12287 maxBF16(raw0x7f7f); class5:t4096/t12288 negative max(raw0xff7f); class6:t0=-0; class7:t1=1. | NaN input, signedinf, infcancellation, finite positive/negative overflow, zero signs and unaffected finite/unused rows. Nonfinite outputs remain failed finite-training gates. |

Direct consumer order: F0, F0 repeated, F1, F2, F3, F4, F1 replay, F5.
Reuse the exact same callable and input tensors for repeated F0; F1 replay uses
its immutable original bytes and new return allocation after intervening changes.
For every transition retain preceding direct output until after the next call
and prove its full bytes/storage contents remain unchanged. Then release it.
F0 tags every channel uniquely; a zero/stale/missing-owner route cannot hide
behind repeated eight-channel motifs in later fixtures.

Ordinary native controls: on EACH rank, EACH of F0–F5, and the changed F4->F1
replay, execute ordinary eager AND torch.compile ordinary forward embedding
with FP32 [8192,1024] deterministic weights followed immediately by BF16 cast,
then actual autograd backward with the same BF16 tangent. Reuse the ordinary
compiled callable across content changes. No custom backward/consumer inside
these controls. Forward values must match ordinary gather/cast and weights/IDs
must be unchanged. Compiler environment and gradient dtype are recorded; no
downcast shim may hide an unexpected FP32 tangent. Both ordinary native dW and
direct consumer dW must match the independent finite oracle byte-for-byte.
If ordinary fails but consumer succeeds, the required component gate FAILED;
do not replace the oracle with native zeros or waive the control.

F5 comparisons preserve all raw outputs, nonfinite masks, infinity/zero signs
and NaN payload differences. Compare unaffected finite coordinates exactly.
Every F5 nonfinite result is labelled FAIL_NONFINITE_DIAGNOSTIC, never equal-NaN
training PASS. Correct classification is a diagnostic observation, not finite
numerical qualification. Report finite-suite result and exceptional diagnostic
completion separately, with no generic model/native-production PASS.

Wrong tangent dtype: one separate bounded call attempt passes actual full-shape
FP32 input to this unchanged callable and must refuse at its specialization
guard before a device kernel runs; capture the real error. Host-only pretend
rejection is insufficient. Release ordinary output/input temporaries first.
If dtype assertion unexpectedly allows execution, record failure immediately.

## Metadata and transport are part of correctness

An independent checker constructs counts from canonical IDs, exclusive starts,
strictly ascending positions and exact128-row maxima. Validate full range,
counts sum32768, positions permutation0..32767, every row's count/offset segment,
IDs matching each listed position, and maxima equality. Separate construction
and validation paths are required. Reject before upload: swapped duplicate
order, wrong in-range row, duplicate/missing position, off-by-one start/count,
stale F1 metadata againstF4 IDs, wrong maxima, invalid ID/negative/sentinel address.
Do not dispatch corrupt metadata to a kernel that does not validate it.

All metadata/IDs/tangents/weights have immutable CPU canonical raw bytes/hash.
Before each call, round-trip device inputs and byte-compare **all values**, not
only shapes or checksums. Revalidate returned metadata against returned IDs.
After every direct/control call, repeat full input round-trip and comparison,
including weights; input mutation fails. Save pre/post hashes, counts and raw
canonical/transport blobs in the content-addressed manifest. Guard lifetime:
retain references through synchronization, comparisons and prior-output check.
The full comparisons are diagnostic overhead; do not include/exclude them to
claim production speed. Native results must actually feed the comparator.

Required harness negatives run without corrupting production/source: feed zero,
stale F0/F1, one-owner-missing, finite bitflip, wrong dtype/shape, signed-zero
mismatch, missing record and nonfinite outputs to its acceptance path and require
refusal. These validate the comparator only; they are not alternate native
consumers. The fresh independent harness reviewer must also show that bypassing
the actual consumer call cannot manufacture a successful route receipt.

## Explicit limits and practical stopping

Prospective one label `fs-20260926-k2-realshape-consumer-correctness-r1`, one host
selected later, four actual ranks/LNC2, one component process group. No fullmodel.
Worker CPU threads1 each; no nested BLAS thread pools. Tensor storage ceiling
448MiB/rank,2GiB aggregate including the parent; one gradient64MiB, FP32 weight32MiB, each expected/dW
32MiB and ordinary BF16 output64MiB are reused sequentially. Never retain all
fixture tensors or three control outputs simultaneously. Parent oracle/generator
live storage<=256MiB. Full output expansion is chunked; Python scalar tensors
or millions of Python float objects are forbidden.

Machine RAM prerequisites: MemAvailable>=24GiB before launch; combined actual
driver/workers/compiler RSS ceiling20GiB (worker total<=8GiB, compiler subtree
<=12GiB), in addition to the explicit tensor bounds. Actual HBM peak, SBUF/spill
diagnostics and process RAM must be recorded; source's1,052,164B logical scratch
perowner is not a measured memory claim. Missing reliable memory accounting
prevents qualification. The packet must choose supported existing accounting
mechanisms, freeze them, and be independently reviewed; no guessed API/tool patch.

Predeclared component wall budget1200s plus existing60s TERM grace, not training's
1800s allowance. CPU oracle/fixture validation<=180s; cumulative first-use native
compiler interval<=600s; entire post-compilation call/comparison sequence<=300s;
retrieval/finalization reserve120s. Each synchronized consumer/control invocation
has60s ceiling, including all-one-row32768. Actual deadlines are measured from
phase starts, cannot be reset by warnings/recompiles. First compile may use its
600s allowance; the60s invocation bound applies after compilation, measured
separately. If instrumentation cannot separate these safely, count conservatively
against both outer/phase limits and report unknown rather than fabricate timing.

New job-owned diagnostic cache ceiling6GiB; no shared-cache cleanup. Complete
artifact ceiling2GiB and logs/metrics128MiB, floor6.5GiB free disk at alltimes;
require>=16GiB free before launch. No archive/checkpoint deletion to fit. A source
hash/specialization change, additional content-dependent compilation, spill or
compiler output exceeding those budgets, OOM/timeout/unsafe access or missing
artifact fails. Monitoring may stop only its own process tree via the reviewed
existing containment, never kill another job to gain resources.

Raw artifacts are practical through **lossless content-addressed deduplication**:
canonical full BF16 input perfixture64MiB (~384MiB for6), full FP32 reference
perfixture32MiB (~192MiB), metadata/IDs, and native full output blobs. Every
rank/route/call maps to its actual captured blob hash, shape/dtype/byte count.
If bytes match an already captured blob, retain one copy plus every authenticated
route association; do not pretend a digest without an existing complete blob
is a retained output. Lossless compression is permitted; report compressed and
uncompressed sizes. Mismatches retain complete actual/expected blobs plus bounded
coordinate witnesses, not just summary errors. No huge JSON float arrays.

After first finite mismatch or mechanical failure, allranks rendezvous and stop
further fixture scheduling, preserving artifacts and labelling remaining cases
NOT_EXECUTED. This is an intentional fail-fast falsifier, never full-suite PASS.
An asynchronous failure/timeout uses existing containment and explicit descendant
postflight; no rank may hang indefinitely at a collective. Exceptional diagnostic
nonfinites are expected failures of the finite gate, so they do not erase prior
finite observations or masquerade as a successful trained result.

At completion retrieve four raw rank receipts, canonical inputs/oracles and all
actual output blobs, transport/input-immutability checks, ordered witness traces,
compile/call timestamps, actual two-owner output coverage, runtime versions,
graph/specialization records where emitted, memory/spill logs and postflight.
No empty field may mean zero/hit/PASS. Both-owner correctness is evidenced by
full-channel F0 tags and exhaustive raw output comparison; sourcegrid2 plus
compiled source/launch metadata is recorded without inventing hardware owner
traces when unavailable. Unavailable allocation/graph evidence stays unresolved.

## If the complete gate does not fit

The bounded reference/duplicate raw-output scheme makes full required witnesses
plausible without fullmodel or giant oracle arrays; actual host feasibility is
unmeasured. If independent harness preparation cannot meet memory, artifact or
deadline accounting, STOP before dispatch. A separate smaller **coverage** stage
may run full-shape F0/direct+ordinary controls and F2 witnesses only, with the same
strict bytes and allfour ranks, to falsify descriptor/rounding basics. It remains
PARTIAL_DIAGNOSTIC, not this component qualification. Do not shrink geometry,
drop all-one-row, weaken controls or relabel partial success. The eventual full
gate still needs F1/F3/F4/F5, replays, transport and all limits above in a newly
completed proposal. No automatic retry or continued stage is included here.

## Authority and isolation

Sole Operator only after selection, completed immutable harness/command/environment
binding and genuinely independent mechanics review. Original shared device0 lock
held throughout; fresh successful ps AND neuron-ls under lock; prior train/eval
terminal and explicit descendant/device quiescence; STOP/breakers respected;
source/support pins and clean tree; absent unique output root; exact four-worker
identity; final artifact retrieval and explicit all-observed PID/device/lock
postflight. No instance/cache/protected-source manipulation or second launcher.
Clean submission qualification retains priority. No actual hostname/slot/release
is assigned by this document. The author is Researcher, not Verifier/Operator.
