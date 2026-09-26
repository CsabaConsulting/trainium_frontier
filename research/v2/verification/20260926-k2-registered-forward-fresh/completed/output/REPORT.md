# Independent registered-forward review

**Verdict: PASS for the limited source/CPU suitability gate.** The candidate
remains disabled and is not qualified for native execution, training, adoption,
performance or BPB claims. No source edit is requested by this review.

Reviewed proposal `20260926-k2-registered-forward-autograd`, canonical
`a6ebb4c2a8e165b4e11bce684f4b9dc544992bd4`; the manifest identifies candidate commit
`aa3d47629730c4f467b97d4193090af6f15d40f3`. Commit history was not accessed.
I worked only from this isolated packet, including the subsequently supplied
neutral provenance addendum, and wrote only in `output/`. I did not inspect the
repository, implementation branch, another agent's worktree, prior conclusions,
or a native host.

## Authentication and source findings

The original manifest SHA256 is
`05c1440c0f908b87ceeb1b7d029098fced9013bd0faaa9460a65b108084a1847`.
All 20 listed files match their byte counts and hashes. Reconstructing the
candidate from `adapter.diff` gives exactly 44 lines and SHA256
`d761a7274e50c21d5930bfa8b984da710ed257be73381fc17ea5f062c03f190a`.
The reconstruction in `output/adapter.py` is byte-identical, not a candidate edit.

The adapter defines one six-Tensor operator per output dtype, with the prescribed
names and `mutates_args=()`. Its body is exactly embedding followed by the output
cast. Setup saves only counts, starts, positions and maximums. Backward passes
the contiguous upstream and those four saved tensors directly to the supplied
callable and returns one result followed by five `None` values. There is no
second consumer registration, explicit NKI rewrap, tensor cache, arithmetic or
shape change in the adapter. FP32 dW is supplied by the protected consumer;
the adapter adds no different gradient cast. The factory only accepts the
specified FP32/BF16 output contracts. Default factory invocation and script
execution both reject activation without importing Torch/NKI.

The protected consumer and reference match the specified hashes:

- Consumer: `2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae`.
- Reference: `14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f`.

The consumer still returns `resident_consumer[2]`, accepts native V256/T256/C128,
and writes FP32 accumulation results. Its ordered additions, metadata reads,
loop bounds and feature partition are untouched. The source requires a caller
to supply that protected callable; this factory does not authenticate an arbitrary
callable itself. Native caller binding therefore remains a future gate.

## Archived decorator and automatic callable contract

I read the archived `nki_hop.py`, `nki_kernel.py`, framework `kernel.py` and
`compiled.py`, authenticated through the manifest. The exact `nki_op` FunctionDef
from `nki_hop.py` lines 569–634 was executed unchanged with the real local
`torch._library.custom_ops.custom_op`, `infer_schema`, and standard typing
dependencies. No `nki` or `torch_neuronx` module was imported. The complete
archived file hash is
`a8b7bffbf461378b54e6f5b4ad3406b80e069a3d89122712c40df79ebef08914`;
my AST hash including source-position attributes is
`87684aba759b00a4cdf9f1250079316f60b7f252c47f88122543518abceac3e1`.

This decorator constructs a real custom operator using the inferred schema,
registers the original body as its fake implementation, and installs a
FunctionalTensorMode decomposition that executes that body. Both private
registrations and the adapter's setup/backward registrations were inspected
successfully. The decomposition makes poisoned backward tests necessary;
positive ordinary embedding results alone would not distinguish fallback.

The archived automatic native bridge preserves LNC through frozen Kernel fields:
`resident_consumer[2]` sets LNC2; native tensor detection selects
`wrap_nki_to_torch_hop`, which supplies `NKIHOPCaller` with `[kernel.lnc]`.
Real Neuron eager tensors can take the C++ dispatch path; compile/fake/proxy
contexts use the HOP path and its config/lowering machinery. The adapter calls
the supplied callable without replacing this mechanism. The SDK contains
compilation/configuration caches, distinct from prohibited adapter tensor-value
caching. This is source evidence only: automatic capture and lowering were not
executed against the target SDK.

## Independent CPU results

`review_cpu.py` is my own test source, with a scalar token-ordered FP32 oracle
using `struct.pack/unpack` after each addition and a separate traceable ordered
consumer that uses runtime metadata. Equality requires finite values, exact
shape/dtype and identical bytes, including signed zero; no tolerance is used.
Runs used Python 3.14.4 / Torch 2.13.0+cpu, one intra-op and one inter-op thread,
V8/T8/C4, noncontiguous upstreams, FP32 weights/dW, and both output dtypes.
The explicit tensors are tiny; the largest is 272 bytes. No model was allocated.

| Independent run | Registered gradients exact | Registered gradients rejected | Other controls exact |
| --- | ---: | ---: | ---: |
| Ordered consumer | 28 | 0 | 42 |
| Zero consumer | 8 | 20 | 42 |
| Stale initial-gradient consumer | 8 | 20 | 42 |

Each run contains 70 routes: registered eager/AOT, equivalent original Function
eager/AOT, and ordinary eager embedding across seven fixtures and both dtypes.
The compiled callable is reused for unique-first, changed nonzero tangents,
changed valid routing metadata with duplicates/unused rows, cancellation,
signed zero, and an initial-fixture revisit. Logical-empty autograd is an extra
local shape specialization; a separate fixed-storage/nonzero-upstream empty
metadata consumer check confirms zero-trip behavior. Neither empty check is
claimed as native coverage.

Every route preserves all seven input hashes before/after. Registered eager
setup saves exactly the four supplied metadata tensor identities. Forward bytes
and dtypes match the independent indexed/cast reference. Zero/stale results are
produced inside the actual registered backward, including the AOT route; they
are not injected after execution. Each poison causes five failures per
registered route per dtype while all ordinary/original controls stay exact.
This excludes ordinary-embedding fallback as the explanation for these local
registered-backward results. Zero/stale equality on fixtures where those values
are correct is retained, not mislabeled a detection failure.

The validator independently rejects missing output, wrong dtype, wrong shape,
NaN, positive and negative infinity, and signed-zero byte differences.
`review_context.py` also executes the unchanged original DiagnosticEmbedding and
enclosing FunctionDef AST from supplied source context, with my CPU consumer.
All 12 additional eager/AOT routes across dtypes and changed tangents/metadata
are exact. That confirms the actual supplied Function control, beyond the
equivalent Function implemented in the main independent test.

One reviewer test attempt failed: its local empty-input substitute padded to
eight rows and then indexed sentinel eight. The complete source, traceback,
command and correction are retained in `review_cpu_attempt1.py`,
`own-ordered-attempt1.log` and `FAILED_ATTEMPTS.md`. The corrected test pads nine
rows and all later runs pass. No candidate change or relaxed comparison was
used. The developer's earlier captured-real-Tensor stale attempt is likewise
retained as original packet evidence and is not counted as a success.

## Provenance addendum and preserved historical native failure

The initial supplied original-20 checker has hash `89189024...`; the specification
names the later FP32-first checker `bd2bca7c...`. I raised that identity ambiguity.
The original packet was not relabeled or changed. A separate addendum manifest
with SHA256
`3a964d71e5ed1b9d800e3bfc0a07b9ba6a5cbc48155cc8fda9b6a52b6e5ab0f0`
authenticates nine additions, including the exact later checker and raw historical
rank receipts. `review_addendum.py` authenticates every added byte count/hash.
The two checker `make_wrapper` ASTs are identical, hash
`7f5f67df39f616d784cf8afc3dd32d1b1744265508823796053c2887f808c36a`.
The initially recorded distinct-file hashes in `own-context-attempt1.log` remain
correct; the addendum resolves their relationship.

The four archived receipts report Torch 2.11.0+cpu with
NKI 0.5.0+28631259367.ga768afa6 on the native path. For the sole FP32/unique
fixture, direct consumer, eager custom Function and compiled ordinary embedding
have exact dW, while compiled custom Function has finite all-zero dW instead of
32,709 nonzero values. All forwards and the seven before/after input records
agree. I recomputed the expected unique-fixture gradient hash using the pinned
reference and the all-positive-zero hash independently. They match the recorded
expected and failed actual hashes respectively on all four ranks. The archived
child and terminal return codes are 1. This failure remains a failure.

These are authenticated archived records, not a native run by me. They concern
the old Function integration, not the registered candidate. Host before/after
hashes do not expose values inside the compiled VJP or establish the root cause;
an observer effect is not ruled out. The outer bundle and host were not accessed.

## Scope remaining

Torch 2.13 CPU AOT plus a test double does not establish Torch 2.11/NKI0.5 behavior.
No candidate native execution, automatic bridge capture, full-model routing,
higher-order gradients, launch binding, speed or BPB improvement has passed here.
A separately reviewed disabled native packet and applicable machine/role gates
remain required, followed by localization and all 20 original fixtures on all
four ranks. Sentinel/localization success cannot substitute for that coverage.
Performance, end-to-end cost, training and public-20M decisions are future gates.

Exact commands, runtime details and exits are in `COMMANDS.md`; complete raw
outputs are `own-*.log`. The original developer receipts were authenticated as
supporting inputs, but this verdict rests on my independent source review and
executed tests. This PASS grants no launch, submission, adoption or hardware
authority.
