# Six-fixture harness preparation — incomplete, default REFUSE

Selected context: canonical proposal commit `2d81736ec55c4ecba362548c8622aba1ebc97553`,
`research/v2/proposals/20260926-k2-realshape-supplied-metadata/`, selection
Supervisor791e2b07. This document does not replace or relax that proposal.

This directory contains a portable preparation core, **not the selected native
harness**. There is no native dispatcher, executable launcher, qualification
function, model flag, or launch authority. Full-size generation has not been
executed on the laptop. The source cannot produce a native PASS.

## Implemented scope

`fixtures.py` defines scalar recipes F0–F5 at V8192/T32768/C1024, direct order,
count/prefix/fill metadata construction and a separate original-ID occurrence-list
validator. `oracle.py` performs one integer binary32 round-to-nearest-ties-even
addition per BF16 input, including subnormals, ordered cancellation, signed zero,
overflow and exceptional diagnostic classification. NaN payload canonicalization
is diagnostic only, not an equal-NaN pass.

`generate.py` streams the full gradient recipe to lossless blobs, validates every
retained gradient byte against the scalar recipe/class expansion, derives row
membership independently from original IDs, then caches only identical complete
ordered timelines. It expands every expected output coordinate. One caller-owned
OracleBudget must span all six fixtures and one caller-provided monotonic deadline
must span the whole oracle phase. No full native caller exists yet to enforce this
or the 180-second aggregate deadline. Recipe validation uses the same recipe
function as generation; independent fixture review is still necessary.

`artifacts.py` retains raw little-endian SHA256-addressed blobs, validates full
retained files and compares every FP32 word. Finite values and signed zeros require
exact bits. Exceptional comparisons separately count NaN payload differences,
classification and infinity-sign differences and can never yield an exceptional
PASS. `compare_record_bytes` checks a supplied record's domain and bytes only;
it does NOT authenticate where those bytes came from. Supplying reference bytes
as actual bytes can satisfy that comparison. This is deliberately not a route
acceptance function. Storage is single-writer only; four-rank locking/accounting
is not implemented. Its 2 GiB ceiling does not account for logs/cache elsewhere.

## Mandatory missing work / evidence

1. Actual consumer invocation and ordinary eager/compiled BF16-output autograd
   routes, reuse/replay, full forward checks, full before/after transport blobs,
   independently revalidated returned IDs/metadata, prior-output immutability,
   and authenticated invocation-to-artifact associations are absent.
2. Actual unchanged-callable FP32-tangent refusal, including evidence that no
   device kernel executed, is absent. A host dtype check is not a substitute.
3. No reviewed four-rank fail-fast rendezvous/containment implementation is bound.
   Existing registered-native `check_native.py:observe_route` continues after
   failure; it is not reusable as the required fail-fast controller. Existing
   host/rank wrappers do not provide the new independent phase budgets or
   20/8/12 GiB combined/worker/compiler-subtree RAM enforcement.
4. No supported frozen-SDK extraction binding has been established for actual
   per-invocation HBM peak, SBUF allocation/spill evidence, or kernel-execution
   counts. This is missing evidence, NOT a hardware incompatibility conclusion.
   Official Neuron Monitor documentation describes current memory usage sampled
   over a period; taking its maximum is not proof of an exact transient peak.
   Official newer Neuron Explorer schema documents spill counters and its Memory
   Viewer documents allocation analysis, but those pages do not establish their
   availability, capture overhead, association, or schema on the frozen run SDK.
5. Graph/specialization/cold-start evidence, actual two-owner F0 coverage records,
   full negative-control suite, complete lossless four-rank route manifests and
   atomic bounded finalization are not implemented. The primitive comparison
   tests below do not stand in for those end-to-end controls.
6. The proposal's initial 16 GiB free disk, 6 GiB cache, 2 GiB artifact, 128 MiB
   logs, 6.5 GiB floor; 24 GiB MemAvailable; 448 MiB/rank tensors, 2 GiB aggregate
   including parent and 256 MiB parent; outer 1200+60 seconds, 600 compile,
   300 postcompile, 60 per warm invocation and 120 finalization reserve remain
   unchanged and unproved. No host was inspected or selected. No cleanup is
   authorized by this preparation.

Official evidence references (not version-compatibility endorsements):

- https://awsdocs-neuron.readthedocs-hosted.com/en/v2.21.0.beta/tools/neuron-sys-tools/neuron-monitor-user-guide.html
- https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/tools/neuron-explorer/profile-schema-reference.html
- https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/tools/neuron-explorer/overview-memory-viewer.html

## Local checks only

Dependencies: Python standard library and NumPy; receipt records installed NumPy
2.5.1. From this directory run `python test_core.py` and `python -O test_core.py`.
The test writes retained tiny raw blobs in a new `/home/poojith-devan/k2-harness-core-*`
directory. Both receipts are retained here. Checks include 20,006 scalar RN32
pairs against independent NumPy binary32 addition, order/subnormal witnesses,
12 metadata corruption/rejection cases, scalar recipe boundary samples, raw
comparison bitflip/signedzero/zero/owner-like missing coordinate negatives,
missing record/blob, byte-size mismatch, diagnostic NaN payloads and budget caps.
Largest numeric test array has four elements. No full fixture generation, native
SDK execution, real transport, full route suite, independent verification, or
runtime/resource guarantee was tested. These are implementer development checks.

Consumer SHA remains `9425390c96053a1b24d4addf8b61b1a87c4993f3551f08c7cdcefcf142d84dcd`;
train.py SHA remains `36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb`.
Protected tiny source, adapter, model defaults and canonical state are untouched.

## Separate unselected partial diagnostic proposal

If the supervisor elects to narrow future preparation, propose only F0+F2 at the
same full geometry, all four ranks/LNC2, actual direct and ordinary routes,
full ordered oracle/transport/immutability/negative controls and lossless outputs.
Verdict must be `PARTIAL_DIAGNOSTIC`, never full qualification. This does not solve
missing instrumentation, does not waive reliable accounting or the 16 GiB
requirement, and is not selected or launchable. It deliberately omits F3 worst
skew, F1/F4 stale-replay coverage and F5 nonfinite diagnostics; those omissions
must remain explicit. A completed separate proposal and fresh review would be
required before any native test. No automatic successor is created here.
