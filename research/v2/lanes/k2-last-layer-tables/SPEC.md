# Fixed last-layer tables with explicit FP32 master arithmetic

Implementer delivery under the human's 13:26IST FP32 precision amendment in
workflow/FINAL_SPRINT_GOAL_2026-09-26_1326_IST.md and updated DESIGN.md.
Native/full execution still follows K2 timing qualification and this patch's
one independent review/native check. No host, launch or self-review performed.
Frozen parent K2 SHA63925519 and P3 source are unmodified.

## Actual single-file implementation

New train.py derives directly from exact K2 parent. Existing `--ngram-ve` is
default OFF and enables only b7/t7 at absolute layer7. GPTConfig persists one
default-false ngram_ve boolean. Old parent checkpoints omit it and load false;
enabled checkpoints reconstruct the same trained modules with strict state load.
The enabled native training argv requires --k2-embedding. Evaluation follows
ordinary Torch with no table metadata/native backward, including longer contexts.

Each table is a zero-initialized FP32 Parameter[524288,128]. Ordinary and
registered forward both gather FP32 master values first, then convert gathered
activations to BF16. Never convert the entire table before gathering. Parameter
gradients, cross-microstep accumulation, AVG-reduction and RMS state remain FP32.
This deliberately replaces historical BF16 arithmetic; the archived BF16
rounding falsifier is not the reference for this patch.

Layer7 hashing uses the fixed A1 signed-int32 products/XOR and modulo524287,
padding8192, bigram primes120193/128201 and trigram144203/152197/160201. Length1/2
are safe. No document/BOS reset. Hashes derive from the immutable token snapshot
when K2 is on, then receive separate fresh stable-sort metadata per table/call.
Original WTE/VE K2 helper ASTs are unchanged and checked in the CPU suite.

Two RNG-neutral 128-to-model-width projections are initialized after existing
random tensors at the original uniform scale. Layer7's new32-to-head gates
are initialized zero without disturbing common RNG state. Bigram consumes
normalized V channels32:64; trigram64:96. Injection order is ordinary V,
unigram VE, bigram, trigram, then the original Q-dtype conversion and attention.
Projection/gate weights join ordinary Muon groups and cached Linear refresh.

Dedicated table kernel has V524288/T32768/C128, two64-feature owners, FP32
dense output, sequential FP32 additions of actual BF16 tangents in token order.
4096 row groups use a runtime outer loop; it does not unroll4096 Python bodies
or pad feature width to1024. Inner loops use group maximum occurrence count.
All rows, including unused rows, receive fresh output stores. Native legality
of runtime-group indexing and dynamic DMA slices is specifically untested.

The table RMS group uses beta1=0, beta2=.999, eps1e-10, WD0, width-scaled LR
.3*(1024/768)^(-.5)=.2598076211353316, and no_lr_decay. FP32 second moment uses
lerp(g.square(),1-beta2), bias correction1-beta2^step, sqrt+eps and FP32 update.
Ownership excludes tables from packed/all-gather buffers but updates the full
RMS group once per rank after AVG. Both tables are clipped jointly at norm1,
separately from all ordinary parameters. Hybrid optimizer-resume methods fail
explicitly; production final checkpoints remain the original model-only format.

## Supplied CPU evidence and limits

Run test_tables.py with --parent pointing to frozen K2 train.py and --repo to
the canonical repository using the existing laptop venv. test_fp32_reference.py
is the first cheap FP32 contract check. cpu-result.json contains source hashes.

* Actual inlined table consumer is CPU-interpreted with geometry constants
  reduced; actual producer and registered autograd execute. This is not NKI
  simulation or compilation. Tests cover the historical [1,2^-8,-1] witness
  under the new FP32 reference, FP32 order witnesses, BF16 subnormal signs,
  normal cancellation, duplicate/interleaved/unused rows and two microsteps.
* Signed-int32 production hash calculations match an independent Python oracle
  at lengths1/2/4 and boundary IDs. No full tables are allocated on CPU.
* Three FP32 RMS updates match direct Torch arithmetic. Four mocked ownership
  ranks prove replicated routing and one RMS update each, not native collective
  correctness. Separate main/table clipping matches independent Torch calls;
  AVG routing observes FP32 gradients for every parameter.
* Actual eight-layer width128/context8/batch2 model uses table32x128 in the CPU
  fixture. Three tables-OFF steps match exact parent losses/gradients/updates
  by bytes. Two ON steps match an ordinary FP32-master/BF16-gather model with
  nonzero tables, actual optimizer/RMS and separate clipping, also by bytes.
  Common initial parameters remain identical when tables are enabled.
* Same-shape future-token changes leave prefix outputs bit-identical; changes
  to another batch row leave this row bit-identical. Repeated calls are exact.
  Strict ON checkpoint and old-parent checkpoint load both roundtrip exactly
  using the production BF16 compute policy in the CPU test.

Disclosed shims: existing test-only prepare import, unused parquet placeholder,
table capacity32 (actual width128 retained), consumer geometry interpretation,
fake ownership collectives and BF16 compute policy for CPU checkpoint load.
No official prepare/evaluator/data source was modified.

Initial optional cross-shape prefix comparison at atol1e-6/rtol1e-5 failed at
one logit with max absolute7.62939453125e-6 after context shape changed. This
observation is preserved in the receipt. The fixed-shape causality tests above
avoid changing GEMM shapes and pass by bytes; no gradient/operation exactness
tolerance was relaxed. This is not native causality/scoring evidence.

## Remaining native costs and uncertainty

Each rank adds512MiB table weights,512MiB dense table gradients and512MiB RMS
state, before update temporaries and activations. Table checkpoint growth is
512MiB. Two producers perform searches over524288 rows each; stable-sort/
searchsorted native support remains inherited K2 uncertainty. Runtime-group
address lowering, consumer liveness, subnormal/rounding semantics, dense FP32
AVG/RMS bandwidth and compilation duration require the native check.

Metadata/two table backward/FP32 RMS and checkpoint costs must be included in
any whole-step timing. There is no component/native/timing/BPB win claim.
Use the existing unique-label proposal and sole Operator path; the prospective
scientific flag delta is --ngram-ve on the qualified --k2-embedding stack.
Table addition is evaluated for final BPB, not required to itself gain3% speed.

All patch artifacts remain in this one folder. train.py is self-contained.
parent.diff is a convenience source diff; tests do not import another candidate
as a production dependency. One independent reviewer remains pending.
