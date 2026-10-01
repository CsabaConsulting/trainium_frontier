# Round29 — DMA hypothesis and fused MLP backward

## 1. Old dpre kernel: built, CPU-tested, native-failed, parked

The NKI epilogue computes `(down.T @ grad.T) * (2 * relu_t)`;
it is not a completed fused backward for all three gradients.

Last full native60 test: `trn22-nki-dpre-gatefixed-native60-r5`, source
`0bc973832a707893c73304449e219759f5431c672a919103e1247ffabc91c869`.
Actual archive `322aaab9e1ea14156d7b8e46fe315975340ae27b0db6f030fed2be8826526d4c`:
0 training steps, 374.2953488 s, no valid tok/s or BPB. Exact compiler result:

> [INTERNAL_ERROR] [NCC_IESL012] Did not expect any shared internal tensors to be live on input to function sg0000 in subgraph 0, but found custom_call.10, custom_call.13, custom_call.14, custom_call.17, custom_call.18, custom_call.3, custom_call.6, custom_call.9 live.

Subsequent owned-input repair source
`b8d5315aab86db010a1bfba45828c8f27d318d5c3f3da6595fb495f7bcec5bae`
passed its CPU boundary checks, but the last Round14 native triage did **not**
establish whether the repair compiles. Its final archived result was:

> RuntimeError: TRIAGE_PAYLOAD_FAILED_124

The worker printed `OUTPUT_SYMLINK`; no rank-success receipts were generated.
Driver duration 23.586864 s. This was a harness guard stop, not another NKI
compiler verdict. The transient symlink target was not recovered. Original
Guardian safe release was authenticated. There was no successful native
gradient or whole-step timing result, so the lane was parked after its timebox.

Evidence: [last triage outcome](operator/20260927-round14-kernel-triage-preparation/dpre-dispatch-after-qk22r4/ROOT_OUTCOME.json),
[original wall](walls/DPRE_OWNED_INPUTS_20260927.md).

## 2. DMA evidence — decode complete, hypothesis unconfirmed

Original small capture: exact gate-fix `500f80fe`, one rank, batch 1, context
1024, one uncaptured warmup FB followed by **one captured FB**, no optimizer.
It cannot directly supply transfers per production optimizer step or justify
extrapolating its throughput to batch 32.

Capture profile bytes: 128,031,925. Pinned relay including source/manifest:
128,128,638 bytes from trn22 to trn21, SHA checks passed. Original files kept.
CPU-only installed Neuron Explorer 2.32.0.498 ingestion started on trn21 at
02:27:51.890969 IST, nice19, one CPU, no device lock or device execution.
Live CLEAN full run on trn22 was not stopped or changed.

Decode completed in60.088328s at02:28:52 IST. CPU summary SHA256:
`63ada7e81895f9a913821ae23bde7b9351dbb609f5416c3a1f496cc2e4c197a7`.
All measurements below refer only to this recorded batch1 FB. **The trace is
incomplete; these are partial counters, not reliable production estimates.**

| Requested measurement | Recorded result | Limitation |
| --- | ---: | --- |
| DMA transfers per captured FB | 1,485 | Not production optimizer-step count; dynamic metadata missing. |
| Individual DMA packets | 1,681,128 | Different unit from transfers;716,765 software-dynamic,822,370 hardware-dynamic. |
| Aggregate transfer median / p10 | 131,072 / 2,048 bytes | Unweighted quantiles of1,485 recorded transfer rows. |
| Recorded transfer bytes | 440,398,208 | Incomplete traffic coverage. |
| DMA-transpose flags | 600 true;885 unknown | Unknown is not false; does not include TensorE transposes. |
| MBU / DMA-active raw SDK fields | 0.006629726 / 0.013783347 | Schema declares percent; invalid/incomplete capture, not a usable utilization comparison. |
| Top10 framework operations by DMA time | Not found | All251,542 instruction rows have empty framework/HLO names and null framework IDs. |
| Positive / nonpositive aggregate durations | 1,485 /0 | Decoder nevertheless reports18 invalid-duration warnings; packet-level scope needs checking. |

Partition-normalized write descriptor estimate: median1,024B/p10512B,
789 of1,461 known rows below4KiB. Read estimate: median524,288B/p101,024B,
524 of1,196 below4KiB. Read/write sets overlap and must not be summed.
These partial observations are compatible with small transfers but cannot
establish that they dominate full-step time or justify an MLP rewrite.

Actual decoder Warning table, quoted:

> Notification from block DMA of type DMA were dropped. This can cause corresponding data to be incorrect.

> Missing additional dynamic DMA metadata, reported DMA values may be inaccurate. To enable more accurate DMA data, re-capture with --enable-dge-notifs flag or set NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 before running. Note: DGE notifications are disabled by default as they can cause execution timeouts on some models.

The other warnings say compiler metrics and HLO stats are missing. No SDK
upgrade is authorized or proposed. **Hypothesis verdict: inconclusive, neither
confirmed nor disproved.** A useful next measurement would require a bounded
fresh capture with DGE notifications and proven source attribution; changing
the decoder cannot recover events/debug names never recorded. No such capture
or full kernel build has been launched under this report.

Supplementary query SHA256
`3b29698245abd1d77fb1364e701bcd98e614968edee082a3279b47b0a81111e0`
completed in5.073457s. It inspected every instruction name, actual Flow
labels and packet bytes, not a sample. Flow has1,481 Instruction→DPA edges,
but no join can recover absent framework names. Attribution attempts stop.
Recorded non-instruction packets larger than4B total4,079,309,952B; the
440,398,208 aggregate-transfer bytes are only10.80% of that packet-byte
total. This is a coverage comparison, not a complete traffic census.
Nine packet rows have zero duration; none of the filtered>4B data packets
do. The18 decoder warning lines remain preserved, not silently dismissed.

There are78 NEFF executions in the captured session. SDK Summary reports
DMA-active0.013491236s and total0.024759460s; their ratio is54.49%, which
does not reconcile with its raw `dma_active_time_percent=0.013783347`.
Do not choose one as a corrected measurement or infer a bandwidth verdict.
Together with missing/dropped metadata this makes the requested MBU versus
busy-time comparison unusable, not evidence that production DMA is efficient.

The analysis follows AWS's
[profile-querying guidance](https://github.com/aws-neuron/neuron-agentic-development/blob/main/skills/neuron-nki-profile-querying/SKILL.md)
and the installed-version
[schema](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/tools/neuron-explorer/profile-schema-reference.html).
DMA busy time and HBM bandwidth utilization measure different things; a high
busy percentage alone does not confirm small-transfer inefficiency. Name
attribution and DMA coverage will be checked before selecting a kernel target.

## 3. Build decision

No full backward kernel has been started: the DMA condition did not pass.
The [conditional plan](walls/ROUND29_MLP_BACKWARD_PLAN.md) is complete:
10–12 focused hours for one design, CPU/simulator gradients and one native60
control/candidate pair, excluding host wait. This is a bounded attempt, not a
success guarantee. It uses disjoint channel ownership and on-chip hidden tiles;
duplicating dpre adds approximately25% backward GEMM work, a material risk.
Compiler/liveness repairs extending beyond the timebox stop the attempt.
The original custom-NKI>=3% whole-step timing threshold remains; later
pure-PyTorch1.5% tests do not silently lower it. Original correctness,
cold-start, fixed-budget and submission-audit rules stay.
