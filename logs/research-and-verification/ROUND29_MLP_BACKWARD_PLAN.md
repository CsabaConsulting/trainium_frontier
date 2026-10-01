# Round29 — conditional full MLP backward plan

## Superseded integration/gates — Round29b, Sep28 02:40 IST

Read ROUND29B_REPORT.md first. Full build now also requires SDK>=2.32
and attributed MLP-backward small-DMA or separate-elementwise-pass evidence.
Actual training NKI0.5 is SDK2.31 generation; profile attribution is absent.
NO BUILD selected. If reopened legitimately, separate fwd/bwd custom ops,
fake kernels, register_autograd and opcheck replace this document's old
CE-style Function integration. Normal-call numerical validation remains
required. Device timing deadline Mon Sep28 16:00 IST; <=12h timebox remains.
No SDK upgrade. Detailed design below is historical analysis, not authority.

2026-09-28. Source/research only, under
[the Round29 directive](../../../workflow/ROUND29_2026-09-28_0225.md).
No kernel, proposal, launch packet, model test, compiler invocation, host call,
queue change or SDK change was made. Root owns profile decoding and operation.

## Decision and time estimate

**CURRENT NO-GO: DMA HYPOTHESIS NOT CONFIRMED; NO BUILD SELECTED.** Root's
decode reports an incomplete trace, described in section5. The plan remains
conditional on valid DMA evidence; no implementation follows this document.
A narrowly scoped, one-design attempt is
estimated at **10–12 focused engineer-hours**, including CPU/simulator checks
and one matched native60 pair, excluding waiting for a host. This is an
estimate for a bounded attempt, not a guarantee of compiling, passing gradients
or being faster. The present decoder result does not satisfy the DMA condition.

The design below avoids new cross-PNC communication by duplicating one GEMM.
It is worth a conditional attempt only if Root's decoded evidence supports
eliminating the large backward intermediates despite that extra work. If the
trace does not establish a material MLP-backward DMA cost, **NO-GO**. If an
installed-stack incompatibility, allocation problem or full-graph liveness
failure needs open-ended repair, **stop this attempt**, rather than treating
12 hours as permission for compiler research. A successful implementation
including such repair is plausibly 16–24+ hours, with low confidence.

The prior wall estimated 6–12+ hours for monolithic implementation alone.
The narrower estimate here relies on unchanged forward/CE/cache behavior,
reuse of the already-written dpre tile arithmetic, one fixed tile size, and
exclusive output-channel ownership instead of new collectives or atomics.
Those simplifications make a bounded attempt plausible; they do not resolve
the existing native integration failure or prove a speed opportunity.

## 1. Old evidence: component success is not full integration success

The [old dpre kernel](../proposals/20260927-mlp-dpre-epilogue-gatefixed/dpre_kernel.py)
fuses **only** `D.T @ G.T` and the ReLU-squared derivative. It returns a full
BF16 `dpre[H,N]` in shared HBM; framework matmuls still compute dX/dU/dD.
It uses hidden128/token512 tiles, a runtime token loop, a full 8 MiB BF16
down-weight SBUF cache, and disjoint token ownership across two PNCs. Q,
twice-ReLU and the product each have explicit BF16 storage boundaries.
Its [registration helper](../proposals/20260927-mlp-dpre-epilogue-gatefixed/mlp_ops.py)
registers a separate framework forward custom op and its autograd backward.
That enclosing route is not the requested CE-style Function route.

The [historical component status](../kernel_swarm/20260927-dpre/STATUS.md)
records simulator/offline component compilation and a 104810-byte NEFF.
Those are not full-model or native-gradient passes. Its prospective
"ready" language predates the failures below and is not current authority.

Last actual **full training-graph native60 attempt**: source
`0bc973832a707893c73304449e219759f5431c672a919103e1247ffabc91c869`, archive
`322aaab9e1ea14156d7b8e46fe315975340ae27b0db6f030fed2be8826526d4c`.
It failed before any training step in `model_default_0_backward_3`, 1206 nodes.
The [failure wall](DPRE_OWNED_INPUTS_20260927.md) records exactly:

> NCC_IESL012: Did not expect any shared internal tensors to be live on input to function sg0000 in subgraph 0, but found custom_call.10, custom_call.13, custom_call.14, custom_call.17, custom_call.18, custom_call.3, custom_call.6, custom_call.9 live.

`prepare.py inputs.copy_` was the asynchronous detection location, not evidence
of a loader/evaluator bug. Input aliasing or shared storage is a hypothesis,
not an established cause. No loader/evaluator edit is proposed.

The three-owned-input-clone repair source
`b8d5315aab86db010a1bfba45828c8f27d318d5c3f3da6595fb495f7bcec5bae`
passed CPU boundary/fallback checks, but the later **Round14 triage** ended
`FAILED_TRIAGE_WITH_AUTHENTICATED_SAFE_RELEASE`,
`RuntimeError: TRIAGE_PAYLOAD_FAILED_124`, payload return code124,
driver23.586864s. The worker printed `OUTPUT_SYMLINK`; the original Guardian
terminated descendants safely. Its actual
[ROOT_OUTCOME.json](../operator/20260927-round14-kernel-triage-preparation/dpre-dispatch-after-qk22r4/ROOT_OUTCOME.json)
says `native_compile_result: UNKNOWN; no rank success receipts` and
`native_gradient_pass: false`. Archive SHA256 is
`d3a3a92af764d8b1a8eb0bbf8b3beed764a8a4ca89f778f56ca1dca80324c14b`.
This is a **harness-output guard stop, neither a native compile verdict nor a
gradient pass**. The exact transient symlink path/target remained unknown.
It does not supersede the earlier actual NCC_IESL012 with a success or a new
kernel failure. No retry is inferred from this plan.

## 2. Current CLEAN contract and production layout

Read actual [CLEAN train.py](../proposals/20260928-clean-gatefix/train.py), SHA256
`881a4f491d256fb89af33246a3b568fc3ffead9616070eea1860d7d8f7f9f749`.
`_FusedMLP` lines742–759 and `MLP.forward` lines769–778 are the current path.
Width1024, ratio4, sequence1024 and 32 rows/rank give N32768, C1024, H4096
per microstep; four LNC2 ranks. Ramp accumulation1→2 does not change these
per-microstep dimensions. Training forward remains ordinary
`F.linear -> ReLU -> square -> F.linear`, saving `relu.T.contiguous()`.
The named `_fused_mlp_kernel` is not an active NKI forward. The wrapper's
`tile=min(512,tokens//2)` is an eligibility check, not forward tiling.

| Tensor / expression | Actual logical shape and dtype | Contract to preserve |
| --- | --- | --- |
| X; incoming G | `[N,C]` BF16 | X is the flattened normalized activation; G may arrive as a view and must not be assumed contiguous. |
| Cached U; D | `[H,C]`; `[C,H]` BF16 | Differentiable casts from FP32 masters; no detach or master-dtype change. |
| Saved R | `[H,N]` contiguous BF16 | Nonnegative ReLU, 256 MiB; same saved forward tensor. |
| Q = D.T @ G.T | `[H,N]` BF16 | Round Q before derivative multiplication. |
| P = Q * (R * 2.0) | `[H,N]` BF16 | Separate BF16 twice-R and product boundaries. |
| A = R.square() | `[H,N]` BF16 | BF16 square before dD matmul. |
| dX = (U.T @ P).T | `[N,C]` BF16 | Baseline is a transpose view of `[C,N]`, not a required HBM transpose. |
| dU = P @ X; dD = G.T @ A.T | `[H,C]`; `[C,H]` BF16 | FP32 matmul accumulation, then original BF16 output casts and master-cast gradients. |

Q/P/A each have a logical 256 MiB extent, dX64 MiB, U/D8 MiB each.
These are arithmetic sizes, **not measured HBM allocations or traffic**.
`.T` is a view; `.to(x.dtype)` can be a no-op. Compiler fusion/spilling and
physical transposes require trace evidence. R must still survive in HBM
between forward and backward; this proposal does not change checkpointing.

## 3. Conditional schedule: one full backward, exclusive C/2 ownership

This is a design inference, not implemented or allocation-proven. Grid size2;
PNC p owns output channels `c in [p*C/2, (p+1)*C/2)`. Each PNC reads all G/D
to form Q, but owns only its output subset in **all three final gradients**:

- dX-transpose rows `c,:`;
- dU columns `:,c`;
- dD rows `c,:`.

Thus every final output byte has exactly one writer. No atomics, partial-HBM
gradient reduction, cross-PNC barrier or send/receive primitive is introduced.
Start with token tileT128, hidden tile128, channel tile128, BF16 inputs and
FP32 accumulators. C/2 must be a multiple of128. No padded production recipe,
FP8 conversion, activation change or runtime autotuning.

`nc_matmul` computes stationary.T × moving; for ordinary BF16 on Trn2 use
partition/contractionK<=128, stationary freeM<=128, moving freeN<=512.
Inputs are SBUF, result is FP32 PSUM. Initialize a PSUM reduction on its
first contraction tile, accumulate only afterwards.
[AWS nc_matmul](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.isa.nc_matmul.html)

| Result tile | Stationary `[K,M]` | Moving `[K,N]` | Reduction and rounding |
| --- | --- | --- | --- |
| Q[h,t] | D[c,h], `[128,128]` | G.T[c,t], `[128,128]` | C=8 tiles; Q PSUM→BF16 SBUF. |
| P[h,t], A[h,t] | Elementwise R/Q in SBUF | Not a GEMM | BF16(2R), BF16(Q*BF16(2R)), BF16(R²), not one FP32 fused expression. |
| dX.T[c,t] | U[h,c], `[128,128]` | P[h,t], `[128,128]` | H=32 tiles; retain four owned-C FP32 PSUM tiles across H; cast at final store. |
| dU[h,c] | P.T[t,h], `[128,128]` | X[t,c], `[128,128]` | N=256 token tiles; FP32 partial plus owned FP32 SBUF accumulator, BF16 only after all N. |
| dD[c,h] | G[t,c], `[128,128]` | A.T[t,h], `[128,128]` | Same N reduction and final BF16 cast. |

Outer token loop retains the old runtime-loop approach, not an unrolled
N32768 loop. For one token tile, stage G and owned X once. Transpose G into
channel-partition tiles once. Loop over hidden blocks: form Q/P/A, consume P
immediately in dX and dU and A in dD; do not store Q/P/A to HBM. After all H,
store the owned dX-transpose rows for that token tile. After all N, write the
owned complete dU/dD slices. Only small P/A tiles need their token-partition
transpose for dW. No full activation/layout transpose is introduced.

Tensor-Engine `nc_transpose` on Trn2 uses SBUF→PSUM, then a copy back to SBUF;
its tile is at most128×128. At T128 this is directly usable and bit-accurate
on v3. This is not a claim that every transpose can disappear: G, P and A
require different partition axes for their consumers.
[AWS nc_transpose](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_transpose.html)

Return fresh real output allocations `dxt[C,N], du[H,C], dd[C,H]` in shared
HBM; the Function returns `dxt.T, du, dd`, retaining the baseline dX view
orientation. No input mutation, output/input alias or full hidden scratch
returned. All temporary tensors are SBUF/PSUM; if a truly necessary HBM
temporary is later identified, it must be **private_hbm**, used only by its
own kernel instance, and its traffic must count against the hypothesis.
Private HBM is not a cross-PNC communication channel. Shared HBM is visible
across instances; SBUF/PSUM and private HBM are instance-local.
[AWS memory hierarchy](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/api/nki.language.html#memory-hierarchy)
Fresh output ownership is a design rule, not proof that NCC_IESL012 is fixed.

### Memory and work costs that can defeat the proposal

At production shape the two owned FP32 dW accumulators occupy8+8=16 MiB
SBUF per PNC. Possible blocked layouts are `[128,32,512]` for dU and
`[128,4,4096]` for dD. Four dX PSUM tiles total256 KiB at T128; Q, transpose
and transient dW tiles add pressure. G/X staging, P/R/A and weight tiles are
additional. Do not retain both full weights. A later optional D cache alone
would add8 MiB and is outside this initial fixed design.

Trn2 has28 MiB SBUF and2 MiB PSUM **per physical core**, not a pooled56 MiB
SBUF allocation. These are hardware ceilings, not promised compiler-usable
capacity; partition pressure, liveness, banking and compiler buffers matter.
[AWS Trn2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/nki/guides/architecture/trainium2_arch.html)

The schedule duplicates Q across the two PNCs. Baseline has four N*C*H GEMMs;
this has five across the pair: approximately **25% extra backward GEMM
arithmetic**, excluding transposes and elementwise work. It also reloads
weight tiles and updates16 MiB of FP32 SBUF dW state per token tile. At T128,
256 token tiles imply roughly4 GiB of accumulator elements read and4 GiB
written per PNC, before other on-chip traffic. These are derived work estimates,
not trace counters. Small strided R/weight loads may recreate the very DMA
problem being targeted. Allocation/spill inspection and native timing decide.

Rejected initial alternatives: token ownership overlaps dW and requires a
reduction; hidden ownership overlaps dX and requires a reduction; retaining
both whole FP32 dW tensors without partitioning needs32 MiB SBUF. New
collectives, HBM partials, multi-kernel finalizers or tile tuning would expand
this attempt. They are not hidden fallback paths within its12-hour estimate.

## 4. Autograd boundary: reuse the existing CE pattern, not CE math

CLEAN lines441–465 wrap NKI kernels with `torch_neuronx.wrap_nki` and call
them inside `_SoftcapCrossEntropyNKI(torch.autograd.Function)` forward/backward.
Reuse that **integration style** in the already-existing `_FusedMLP` Function:
keep forward and `ctx.save_for_backward(X,U,D,R)` unchanged; backward invokes
the wrapped full-backward kernel on two lanes and returns the same three
gradients. Keep all eligibility/fallback behavior outside the native path
explicit, preserve same dtype/shape/stride metadata, and include any required
`grad.contiguous()` copy in timing. Do not copy the historical registered
framework-forward op into this proposal or add speculative owned-input clones.

CE kernels, CE reduction, softcap, MTP objective, loss weights, model init/RNG,
attention, cache refresh, FP32 optimizer/master weights, gradient accumulation,
all-reduce/clipping, clock/ramp, checkpoint loading and causal eval all remain
unchanged. No forward-kernel restoration or residual/layout change is allowed.
Cached U/D stay differentiable; no `detach`, `no_grad` cache or extra refresh
between accumulation microsteps.

The public native guide describes NKI custom-op/autograd integration, but it
does not establish this exact Function boundary on the frozen development
stack. The current CE implementation is the local precedent, not proof that
a three-output MLP backward compiles.
[AWS native NKI integration](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html#neuron-kernel-interface-nki-integration)

Archived installed versions are Torch2.11.0+cpu, torch-neuronx
2.11.3.0.0+unknown.dev, neuronx-cc2.26.6360.0+6f180f47 and
NKI0.5.0+28631259367.ga768afa6, per [source research](mlp_backward_sources.md).
No host version was re-queried. Current public docs include newer APIs; use
only installed-supported primitives verified against those archives. Do not
infer an SDK release from the compiler component number or install anything.
AWS's [experimental MoE backward source](https://raw.githubusercontent.com/aws-neuron/nki-library/main/src/nkilib_src/nkilib/experimental/moe/bwd/bwmm_bwd_dropless.py)
is a tile/activation-derivative design reference, not a drop-in BF16 dense
ReLU-squared backward. Public main is not pinned-stack compatibility evidence;
AWS documents compiler-specific bundled-library compatibility.
[AWS nki-library policy](https://github.com/aws-neuron/nki-library/blob/main/README.md#integration-with-the-neuron-compiler)

## 5. Evidence required before selecting the conditional attempt

Actual Root-reported decoder handoff, retained with the
[Root decode artifacts](../operator/20260928-round29-dma-decode/results-bundle.json):
DMA notifications were **dropped**;
additional dynamic metadata was absent (DGE disabled), as were compiler
metrics/HLO statistics. The DPA query returned1485 transfers,440398208 bytes,
p10=2048 bytes, median=131072 bytes;600 transpose=true and885 unknown.
Unknown is not transpose=false. Packet counts were1681128 total, including
716765 swdge and822370 hwdge. Those packet counts are not an alternative
complete DMA-transfer count. Framework timestamp attribution was0/1485;
name coverage was still being checked. Raw SDK MBU=0.006629726% and
DMA-active=0.013783347%, using the query's declared units, are **not reliable
utilization evidence** for this incomplete capture. These values are Root's
reported query output, not recomputed measurements in this source-only task;
do not divide them into guessed microsteps or infer a ranked framework table.
The [Root report](../ROUND29_REPORT.md) identifies the source capture as500f,
batch1/context1024, one captured forward/backward after one uncaptured warmup,
no optimizer. Thus it is not a production pack32 microstep even before
accounting for the missing metadata.
Therefore the small/transposed-transfer hypothesis is **NOT CONFIRMED** and
Root has selected **no full-kernel build**. Missing metadata is not a speedup
opportunity established by the presence of small transfers.

Root supplies the decoded captured source/shape/step/rank context, transfer
count per captured microstep, median/p10 transfer bytes, DMA-transpose count,
MBU versus DMA-active, and top-ten framework operations by DMA time where
attribution exists. Missing fields remain missing; aggregate DMA activity is
not automatically MLP cost or bandwidth saturation. Engine overlap is not
summed into whole-step critical-path time. A one-rank/two-step trace is not
a four-rank speed measurement. Source-level `.T` counts and256 MiB tensors
are not substitutes for those measurements.

Admission inference: the avoidable, attributed MLP backward time must plausibly
exceed the extra Q work, local transpose/accumulator traffic and integration
overheads. Even a3% whole-step rate gain needs at least
`1 - 1/1.03 = 2.9126%` reduction in whole-step time; that is an ideal lower
bound, not a sufficient DMA threshold. If the trace cannot resolve the
hypothesis or shows a compute-bound backward, do not build under Round29.
Root records that decision; this document does not supply invented metrics.

## 6. Exact validation scope if selected later

All Torch, simulator, compiler and native execution is **instance-only**, by
Root under an actual completed proposal and existing guarded machinery.
Laptop work remains source/AST/stdlib metadata only. Tests below are prospective;
no PASS is claimed. They reuse existing safety controls, not new launchers.

1. **Static proof / ownership model.** Pin CLEAN and candidate. AST/diff permits
   only the backward kernel/binding and existing Function backward dispatch.
   Prove the two C-owner intervals cover every dxt/du/dd output exactly once,
   and every N/H/C contraction tile exactly once except deliberate duplicated
   Q. Check initialization before accumulation, final BF16 cast positions,
   no input mutation/shared scratch and unchanged forward/eval/recipe.
2. **CPU wrapper/reference, seed42.** Independent reference is the exact
   expression table above, not the candidate implementation. Shapes
   `(N,C,H)=(256,256,1024),(2048,256,1024),(256,1024,4096)` cover ownership,
   multiple token iterations and production feature dimensions. Inject only
   a clearly labelled CPU backward reference into the wrapper to check its
   boundary; require exact forward/gradient identity for that plumbing test.
   Two distinct nonzero upstreams and changed inputs reuse one callable.
   Test a single microstep and two accumulated microsteps with the SAME
   differentiable BF16 cache objects, then three optimizer steps/cache refreshes
   against control. This is not a native or real-kernel gradient result.
3. **Actual-kernel simulator.** Begin `(256,256,256)`, then
   `(512,256,1024)` with both LNC2 owners and T128. Test all dX/dU/dD against
   independent quantized-input reference, plus zero/near-zero ReLU, signed
   asymmetric inputs, cancellation and changed upstream. Explicitly record
   installed low-precision simulation mode. Check finite/shape/dtype, RMS,
   absolute maximum/mean error and relative RMS per tensor, not only loss.
   Require expected-zero outputs exactly zero; general nonzero cases must
   actually be nonzero. Missing factor2, zero-gradient, swapped-layout and
   stale-upstream negative controls must fail. The existing DPRE numerical
   ceiling is relative RMS<=0.01; do not silently relax it or claim bitwise
   native equivalence. Any stricter Root-approved rule takes precedence.
4. **Native gradients/integration.** Reuse the existing cached-gradient
   diagnostic coverage, adapted and pinned to the new Function without
   changing its reference/0.01 ceiling: `(2048,256,1024)` and
   `(1024,1024,4096)`, two different upstreams, actual BF16 cache reuse and
   FP32 master-gradient accumulation. Four ranks/LNC2, actual wrapped kernel
   dispatch, retained per-rank receipts and no fallback. Preserve a diagnostic
   poison/dispatch negative control. A small component pass does not establish
   the complete production graph; require production N32768 execution in
   native60 and stop if NCC_IESL012 or other native failure recurs. No SDK,
   graph-registration or allocation-policy workaround inside this timebox.
5. **Matched native60 and admission.** Separate control/candidate research
   derivatives use the existing fixed60 loop-tail cap, preserving checkpoint
   handling, production schedule and all accounting. No `--num-steps` or
   environment override. Same host, seed42, four LNC2 ranks, equally cold
   private caches and unchanged compiler defaults. Base launcher shape:
   `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`.
   Compare actual phase1 printed steps20/30/40/50, harmonic token rate;
   require actual logs/tokens prove matching phases, not a forced clock.
   Report startup, full/charged wall, net tokens, loss, memory/spills and
   checkpoint/cleanup. Retain the custom-kernel threshold>=3%, with cold
   startup<=900s and existing failure rules. The later pure-PyTorch1.5% and
   LEAN2% thresholds do not silently lower this NKI kernel's requirement.
   No win on kernel microseconds alone. A timing pass permits only a separately
   selected, independently reviewed scored-source full run and fresh public20M
   decision; no BPB/adoption/submission claim from this plan.

CPU/simulator envelopes must remain bounded and instance-local, with Root's
existing one-CPU/nice19/resource caps and scratch-local cleanup. A cap failure
is recorded, not retried on the laptop or hidden by reducing the declared
coverage. The old [native diagnostic README](../verification/20260927-dpre-native-gradient-check/README.md)
and [cache addendum](../verification/20260927-dpre-native-gradient-check/CACHED_VARIANT.md)
document historical coverage, not current pass receipts or reusable old hashes.
Simulated Python execution cannot prove compiler acceptance, physical SBUF/PSUM
allocation, scheduling or speed.
[AWS simulator limitations](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/nki_simulator.html#simulation-limitations)

## 7. Bounded schedule and stop points

| Elapsed engineering time | Expected work, if Root confirms DMA and selects this one design |
| --- | --- |
| 0–1h | Freeze proposal/math/ownership map, exact installed API checks, wrapper and reference specification. Stop if ownership or supported APIs are unresolved. |
| 1–5h | Implement fixedT128 C-owned full backward and CE-style binding; source/AST checks and CPU wrapper/cache tests. No alternate ownership design. |
| 5–7h | Simulator gradients and negative controls; correct bounded indexing/rounding defects only. Stop on unexplained error or capacity design failure. |
| 7–9h | Root review, installed native numerical boundary and production composite check in the original guarded path. Stop on compiler/liveness failure; do not debug an SDK. |
| 9–11h | One same-host CLEAN/control and candidate native60 pair, archival and matched-phase analysis if prior checks pass and the host is actually free. |
| 11–12h | Freeze artifacts and verdict. No gain, missing evidence or exhausted time means no advancement. |

This is an expected **10–12h bounded attempt**, not twelve hours of guaranteed
productive device time. Waiting for safe host availability is additional and
never justifies blocking/interfering with CLEAN. A production cold compile
over900s fails the existing gate even if engineering time remains. If a
credible pre-build review raises expected work above12h, Round29 says no build.
The main uncertainty is whether this runtime loop plus16 MiB persistent SBUF
accumulators survives the frozen compiler and enclosing autograd graph without
spills or shared-tensor liveness failure. That uncertainty is explicitly not
resolved by the prior component NEFF, CE precedent, simulator, or this plan.
