# P runtime interpretation: exact Recursive6, profiled step 20

This is the assigned Analyst's completed local interpretation, **not a new fresh
source review, native acceptance or performance gate**. Only four phase reports
and four runtime JSON traces were inspected. No host, training, candidate source,
ledger or launcher was changed. The 4.13 GB device NTFF was not opened or decoded.
This packet persists the completed analysis; no further tracing was performed.

## Conclusion

P permits coarse runtime graph attribution and exposes a blocking-copy wait. It
does **not** establish a data-loader I/O bottleneck, embedding-backward device
cost, every-matmul/SDPA BF16 coverage, an MFU estimate, or a K1/K2 speed win.

All four runtime sizes/SHA256 values were independently checked against the
[retrieval manifest](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20260926-trn21-profile-recursive6ef7/completed/runtime-trace-retrieval.json).
Exact inputs, fields and observed numbers are pinned in `findings.json`; the
completed read-only probes are recorded in `commands.json`. Phase reports are
under the same completed packet at `profile/rank{0,1,2,3}/profile.json`. Each
reports step20, world4, accumulation1 and131072 global tokens/update, exact source
`6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.

## Three distinct timing layers

| Evidence | What it supports | What it does not support |
| --- | --- | --- |
| `user_annotation` / `phase_cpu_annotations` | Inclusive host dispatch/wait ranges | Exclusive device phase percentages |
| `privateuse1_runtime` | Enqueue and runtime dispatch intervals, including queued waits | Per-instruction engine-active time |
| `privateuse1_driver`, process label `NeuronDevice N`, `xu_type=Compute` | Runtime-observed device-labeled compiled-model intervals, correlated by external/sequence IDs and NEFF keys | Decoded device-instruction coverage, internal operator costs, hardware FLOPs |

Every phase `device_ms_if_profiler_attributed` is null. Negative `cpu_self_ms`
occurs in11/10/11/8 rows for ranks0/1/2/3. The reports explicitly warn that nested
CPU dispatch ranges overlap. The annotation durations, their nested children,
runtime dispatches and driver spans must not be summed as independent costs.

## The long data-fetch annotation is mostly waiting

In [rank0 runtime trace](/home/poojith-devan/Desktop/Trainium-trn22-queue-operator/out/p-runtime-traces-20260926-r1/rank0-trace.json):

- `traceEvents[253]`, `P/data_fetch`:275.838232 ms.
- `traceEvents[334]`, `aten::copy_`, `External id=332`:268.582190 ms. Input dimensions
  are `[1,32768]` integer tokens; `Concrete Inputs` contains `False` for nonblocking.
- Matching `neuron::copy::cpu_to_neuron`, `nrt_sequence_id=1846`: enqueue0.000946 ms,
  then267.888589 ms until TensorOp dispatch begins. Runtime dispatch0.488441 ms;
  device-labeled driver transfer0.193672 ms.
- `traceEvents[32388]`, `model_default`, external571, correlated with
  `CompiledFunctionBackward` at `[1520]`:273.642050 ms. It overlaps the data-fetch
  annotation by271.682842 ms.

The long copy returns just after the pending backward graph and short transfer.
This strongly supports synchronization/queue waiting surfaced inside the loader's
blocking H2D copy, **not275 ms of loader CPU or disk work**. The same pattern appears
on other ranks: backward/fetch overlap270.14–278.62 ms and long host copies213.89–
268.58 ms. This does not rule out smaller loader overhead, nor measure the gain
from any change to prefetch/stream scheduling.

## Coarse graphs and K1/K2 implications

External8 maps the forward compiled function to the first `model_default` graph;
external571 maps the backward compiled function. Three later graph launches per
rank correlate to `Torch-Compiled Region: 1/0`, `1/1`, `1/2` inside
`P/optimizer_muon_group_inclusive`. These are runtime graph durations, not exclusive
hardware instruction times or accepted steady-step benchmarks.

| Rank | Forward graph ms | Backward graph ms | Sum of three Muon graph intervals ms |
| --- | ---: | ---: | ---: |
| 0 | 138.108883 | 273.642050 | 7.574627 |
| 1 | 137.073180 | 272.255578 | 10.988568 |
| 2 | 141.857117 | 281.910434 | 11.611239 |
| 3 | 141.687720 | 281.273465 | 21.297725 |

K1: the observed path has three compiled Muon group calls per rank. This supports
examining residual dispatch/scalar-upload work, not a premise that many unbatched
Muon launches dominate the step. The profile does not demonstrate>=5% attainable
whole-step savings. Full optimizer host ranges also include AdamW and other work.

K2: the coarse backward graph contains no separately exposed embedding-backward
event. The profile cannot establish what fraction is embedding scatter, whether
it dominates, or its priority relative to K1. Empty embedding event arrays mean
unresolved visibility, not zero work. K2 remains a hypothesis requiring its own
correctness and measured whole-step evidence.

## Dtype, skew and remaining unknowns

Every rank's forward `CompiledFunction` input metadata lists42 BF16 tensors,
22 `float` tensors and two integer inputs. The runtime exports do not expose any
individual matmul, SDPA, embedding or scatter event names outside Python frames.
Graph-boundary mixed dtype metadata cannot establish internal matmul/SDPA compute
or accumulation dtype; it is neither proof of all-BF16 compute nor proof of an
FP32 matmul. The report fields `every_matmul_sdpa_bf16=UNKNOWN`,
`embedding_backward_device_ms=null` and `mfu=null` remain appropriate.

All trace timestamps share `baseTimeNanoseconds=1782967788000000000`. Rank0's
forward annotation starts1.242–1.245 seconds after the others. Host clip/reduce is
22.429 ms on rank0 versus1251.301–1268.148 ms on ranks1–3. Their loop walls are
0.502097 versus1.744045–1.747249 seconds. This is consistent with collective waiting
from capture-start skew; its exact cause is not proven. Rank0 alone enabled raw
device capture. These windows cannot be treated as clean, matched whole-step
throughput measurements.

Decoded-device engine coverage, event-cap completeness, internal operator/dtype
attribution, hardware FLOP counts and any optimization's causal speedup remain
unknown. No MFU was estimated from host time or the source FLOP formula.
