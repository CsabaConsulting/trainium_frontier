# P3 local profile assessment — 26 September 2026

Verdict: **insufficient measured evidence to choose flash attention or ReLU².** The retrieved native runtime traces establish whole compiled forward/backward intervals, not exclusive attention or MLP-activation/backward costs. No P3 kernel implementation is selected by this report.

Scope: local Analyst work only, under the Final Sprint directive. No host access, downloads, decoder execution, training, source patch, state/queue change, AWS action or new review. The only new artifact is this report. Paths below are relative to `/home/poojith-devan/Desktop/Trainium` unless absolute.

## What the saved evidence actually contains

The P capture is **six-layer Recursive source `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`**, step 20, width 1024, context 1024, four LNC2 ranks, pack 32, accumulation 1, 131,072 global tokens/update. It is not the audited eight-layer e1e2 model. Binding: `research/v2/lanes/p-profile/BINDING.json`. Per-rank reports: `research/v2/operator/20260926-trn21-profile-recursive6ef7/completed/profile/rank{0,1,2,3}/profile.json`.

I rehashed and inspected the four already-local runtime JSON files, one at a time, using Node with a 128 MiB old-generation heap limit. This limit is not an RSS guarantee. No scratch copy, tensor or expanded trace was created.

Local directory: `/home/poojith-devan/Desktop/Trainium-trn22-queue-operator/out/p-runtime-traces-20260926-r1/`.

| File | Bytes | Events | SHA-256, matching the retrieval manifest |
|---|---:|---:|---|
| rank0-trace.json | 10,181,471 | 36,035 | `2c4b3aaed7a1d810937eb0c3d3c44f7f853f3d53cc335237c8d7ae611dd445e9` |
| rank1-trace.json | 16,029,891 | 59,598 | `7f0cd37e094d0b01e8f9d00b7e4f6ecfb33e2b8977e386ea75d5f4da8a271b77` |
| rank2-trace.json | 17,269,103 | 64,010 | `4ba95d95c90a2254078a78918caf4acb4c324d0a551a716809d344b58a730256` |
| rank3-trace.json | 13,547,941 | 50,050 | `14553ef6f1c113dda5d405d6482f37ad99089a9a49fadc6b913577e87ca1db04` |

Manifest: `research/v2/operator/20260926-trn21-profile-recursive6ef7/completed/runtime-trace-retrieval.json`. Across all 209,693 event names, a case-insensitive search for `relu|scaled_dot|flash|attention|fusedmlp|fused_mlp|embedding|scatter|matmul|bmm` found **zero**. This is a visibility result, not zero execution cost. Python call stacks and aggregate graph metadata are not an operator timing decomposition.

Rank 0's `privateuse1_driver/model_default` spans confirm forward **138.108883 ms** (external 8, sequence 281474976714971) and backward **273.642050 ms** (external 571, sequence 281474976714977). Other ranks span forward 137.073180–141.857117 ms and backward 272.255578–281.910434 ms. The backward graph includes many operators; its duration cannot be assigned to ReLU², MLP matmuls or embedding scatter.

The four phase reports have no device-attributed phase durations. Their observed attention/matmul/embedding event arrays are empty. Host annotations overlap and the capture has approximately 1.24 s rank-start skew. Existing detailed interpretation remains valid: `research/v2/analysis/20260926-p-runtime-interpretation/ANALYSIS.md` and `findings.json`. Do not add host annotations to graph times or turn these numbers into exclusive phase percentages.

## Why a smaller local decode cannot recover the answer now

The retrieval manifest explicitly leaves the **4,130,966,280-byte NTFF on trn21**. Its remote session also has a 1,529,888-byte `ntrace.pb`, a 5,929-byte `trace_info.pb`, and 80 NEFFs totaling 167,863,163 bytes. The small per-rank JSON reports are inventories/summaries; they do not contain decoded device instructions. No usable instruction/source summary was retrieved.

The r2 decoder's raw `decoder-process.json`, decoded only in memory from `research/v2/operator/20260926-trn21-p-offline-decoder-r2/completed/results-bundle.json`, records:

- `summary-json --ignore-nc-buf-usage --ignore-instruction-hierarchy`;
- generated scratch **271,389,612 bytes**, exceeding 256 MiB by 2,954,156 bytes;
- `GENERATED_SCRATCH_CAP`, TERM, return code −15 after 1.445742 s;
- **zero stdout/stderr bytes and no usable summary**; mapping coverage remains unknown.

The installed help is preserved at `research/v2/operator/20260926-trn21-profile-recursive6ef7/offline-cli-capability/neuron-explorer-view-help-complete-normalized.json`. It offers system-trace filters and options to omit device instruction/event/DMA traces, but no documented device time-window/model-prefix streaming export with a guaranteed scratch bound. `summary-json` avoids database output; it demonstrably does not avoid temporary expansion. Omitting instruction traces would remove the evidence needed for this question. A guessed NTFF byte prefix or arbitrary NEFF pairing would not establish completeness or attribution.

Therefore no further full decode, cap increase, raw transfer or bespoke binary parser is justified by this local assessment. Streaming the existing runtime JSON is feasible within the stated resource envelope, but cannot create missing operator timings. The local inspection used zero generated scratch beyond this small report, well below 256 MiB.

Older decoded evidence also does not settle current P3. `research/agentic/SOURCE197_OFFLINE_ATTRIBUTION.md` is a different six-layer forward-only library-placement comparison; its source/engine interval unions overlap and exclude backward. `research/trn2-experiment-results.md:483` reports historical single-rank/LNC1 attention costs, explicitly including a previously missed backward kernel. Neither supplies the current eight-layer exclusive split. The old example ReLU² ON/OFF result at line497 is a historical negative, not a current-base comparison.

## Exact audited e1e2 path: live and absent switches

Canonical source: `workflow/candidates/confirmed-clock36-116188/train.py`, independently rehashed as **`e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43`**. The audit copy at `/home/poojith-devan/Downloads/Trainium-clock36-CLAUDE-AUDIT-e1e2f686/train.py` has the same hash. This is a source inspection, not new native execution proof.

| Path | Exact source facts |
|---|---|
| Attention | Lines 74 and 760 instantiate attention only in layers 0,2,4,6,7: five calls per microstep. Lines 681–700 project Q/K/V, apply fused-capable RoPE/QK norm, then call `F.scaled_dot_product_attention(..., is_causal=True)` directly at line 698. There is no custom NKI flash-attention dispatch in this file. Source alone does not reveal the SDK's selected native SDPA kernel or its speed. |
| Attention flags | `--compile-sdpa-direct` exists at line1342 and must be true at lines1378–1379. It is a fixed-contract check, not an A/B path. `--nki-attn` and `USE_NKI_ATTN` are absent. |
| Live MLP training path | `norm` at lines162–163 returns BF16. In `MLP.forward` lines743–753, training/BF16 with no precomputed `fc` and eligible token tiling selects `_FusedMLP.apply`. The audited launch's pack32/context1024 gives32768 tokens/rank, tile512 and an exact multiple of1024, satisfying that branch. All eight blocks contain an MLP. |
| MLP forward | Lines702–708 perform ordinary Torch `linear → relu → square → linear` and save the ReLU output as `relu.T.contiguous()`. `_fused_mlp_kernel` is a function name; it is not an NKI kernel. The model is compiled at line1403. |
| MLP backward | Lines725–733 recompute `relu.square()`, calculate the down-projection input gradient, multiply by `2*relu`, then form dX/dUp/dDown with matrix products. ReLU² elementwise/materialization work must be distinguished from those matrix products before claiming an activation hotspot. |
| ReLU² flags/fallback | No `--nki-relu2`, `USE_NKI_RELU2`, `relu_squared` or custom NKI ReLU² dispatch exists in e1e2. Its fallback at lines746–748 is ordinary Torch ReLU² and is used for eval/noneligible inputs. A historical flag on another file cannot alter this source. |

I also read and rehashed the exact historical `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b:workflow/proposals/research-v2/train.py` Git blob: it matches `6ef7d090…`. Its example `relu_squared`/`--nki-relu2` definitions are at lines 1430/3480, but `_install_submission_training` at lines 4858–4871 installs the eligible BF16 MLP adapter with `forward_backend='native'`. That adapter bypasses the ordinary MLP forward at lines 4488–4503. Hence a printed historical NKI-ReLU flag is not evidence that the active dense training path used it.

## Ranked next proposal

**Rank1: one short, no-trace native component timing on the current e1e2-derived stack; kernel choice remains pending its result.** Use the sole Operator's next eligible slot after higher-priority P0/K2 work. This is a diagnostic suggestion, not a launch assignment or a new gate.

The cheapest actual measurement is paired, synchronized timing of (a) the exact SDPA core forward and backward, and (b) the exact saved-ReLU/square/transposition and `2*relu` derivative boundary used by `_FusedMLP`, with the production BF16 shapes/strides and four-rank contention. Compile each measured callable with the production compiler settings, exclude first compilation from steady-state timing while reporting it separately, warm up, and use repeated paired blocks with synchronization at their boundaries. Retain only small timings/config/finite-gradient records; no NTFF or decoder scratch. Separately identify MLP matrix-product time if the question is about all MLP backward rather than activation alone.

This yields **component execution times**, not exclusive costs inside the fused training graph or predicted whole-step savings. Preserve that limitation: multiplying by five attention calls/eight MLP calls gives only a component estimate, and compiler fusion/layout can change the integrated result. Do not select flash/ReLU² on a noise-level difference or label either a win. A subsequently selected patch uses the Final Sprint's existing CPU exactness, three-step OFF identity and one cold-start/whole-step timing check with one reviewer; this report adds no review ceremony. If exclusive in-graph attribution is required before even nominating a patch, it remains unavailable and needs a genuinely smaller current-base device capture with source mapping and demonstrated complete coverage—not another run of the failed4.13GB decoder.
