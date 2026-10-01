# R151 independent LaneB / CE2 scoring-path audit

Disposition: source facts established; actual organiser runtime failure remains UNKNOWN. No source edits, host/device calls, Torch imports or numerical execution. R151/R152 read directly from LIVE `CLAUDE_ROUNDS.md` snapshot138b5f0b7392868967001ad8e531e69bc88b875d15223092b7ed8187060f3535. No submission or launch authority is created.

## Conclusions first

1. **Participant CE2 is unreachable in the documented official BPB evaluator.** `prepare.py:624` calls `model(x)` only; `prepare.py:628` computes trusted `F.cross_entropy(..., ignore_index=-1, reduction='none')`. LaneB `train.py:1411–1412` returns softcapped logits when `targets is None`, before the MTP/CE2 branch. Therefore the scorer's `reduction='none'` is not a request to the participant CE2 wrapper. README:241–247 expressly identifies this function as official scoring. A hidden scorer departure from that contract is not evidenced here.
2. **LaneB always binds attention grid[2]; its Neuron path has no LNC adaptation or missing-library SDPA fallback.** Lines981–982 bind both wrapped kernels to[2]. Lines1095–1098 select native for saved `lane_b=True` on Neuron, otherwise SDPA. ImportError merely sets handlesNone(983–985); helper1016–1017 then raises `LANEB_NATIVE_LIBRARY_UNAVAILABLE` if either handle is missing—even for no-grad logits evaluation. This contradicts the older R143 fallback statement; it is consistent with the frozen R113 SPEC:22–24. Unsupported shape/dtype likewise refuses, not falls back.
3. **No world4/process-group requirement occurs in LaneB/CE2 wrappers or `load_for_eval`.** AST inspection finds no `dist` references there. World4 assertion2034, batch-ramp broadcast1975–1983, optimizer collectives and table updates belong to training `main`, which the evaluation hook does not call. A single process is not by itself proven to break these wrappers; grid/runtime compatibility needs actual diagnosis.

## Exact assumptions and reachability

All `train.py` line numbers below refer to exact LaneBdcdef066.

| Component | Source assumption | Scoring relevance |
| --- | --- | --- |
| LaneB import/launch981–985 | `wrap_nki(attention_cte)[2]` and `wrap_nki(attention_bwd)[2]`; both imports/handles required; only ImportError caught | Forward handle is used; backward handle must exist but backward does not execute under trusted no-grad scoring. No runtime LNC check. |
| LaneB geometry1018–1026 | Equal Q/K/V `[B,H,T,D]`; positive B/H; T>=512 and divisible512; 1<=D<=128; BF16; same device | Default public B1/T1024 with saved H8/D128 satisfies the wrapper checks. There is no B32 or even-B requirement in this wrapper. Installed-library hidden constraints and actual private scorer shapes are not established by this inspection. |
| Layout991–1011,1028–1032 | Q/K contiguous `[B*H,D,T]`, V converted to `[B*H,T,D]`; cached statistics reshaped `[B,H,128,T/128]`; backward expects matching channel-major outputs | Layout materializations and cached statistics also occur for no-grad forward. Private library/runtime compatibility remains unproved. |
| CE2 guard892–898, kernel518–524 | Neuron, NKI available, contiguous FP32/BF16 logits; rows multiple128; kernel rows divisible128*lanes and vocab divisible512; FP32 targets `[rows,1]` | Training/direct target-bearing calls only, not documented BPB scoring. Actual vocabulary8192 meets the kernel requirement. Unsupported outer conditions use eager CE; outer guard does not separately validate vocab divisibility. |
| CE2 grid699–711,2625–2628 | Grid2 if rows>=512 and divisible256; otherwise1 for positive rows divisible128. Depends on shape, not LNC environment or process group | Training B32/T1024 selects2. Hypothetical target-bearing B1/T1024 also selects2, but evaluator never makes that call. |
| MTP/defaults1357,1413–1417,910–934 | Targets provided + mtp enabled uses CE2. `mtp_weights=None` creates[1,0]; zero auxiliary weight does not skip the second-target call. `reduction='none'` returns weighted per-token loss | Relevant only to non-scoring `model(x,y,...)` calls. With targetsNone the earlier logits return ignores MTP weights/reduction entirely. |

## Evaluator conditions actually evidenced

- Public CLI defaults are B1/T1024(`prepare.py:861–862`), not proof of the private runner's exact argv/environment. Loader emits fixed `[B,T]` and pads final rows/batches(`prepare.py:544–568,580–585`); it does not shrink T on the final batch. Trusted CE alone handles target-1 padding.
- `load_for_eval:1924–1936` rebuilds the saved config, extends rotary capacity to8192, strict-loads full weights, moves to device, disables gradients. It does not invoke `init_runtime`, optimizer setup, FlatWeights or `torch.compile`. Capacity8192 is not a forced evaluation context. The core uses actual input T(`1376–1379`). Neuron compute dtype is BF16(`164–168`).
- Training compiles the core(`2008–2012`) and uses B32/T1024, world4, explicit targets/MTP weights(`2014–2034,2075–2084`). Evaluation hook leaves `_d1_compiled_core=None`(`1224,1370–1373`). Thus compiled-training versus fresh eager-wrapper dispatch is another real condition difference, not proof of a defect.
- `prepare.get_dist_info:90–93` defaults to one rank when distributed env keys are absent. Eval process-group setup is conditional(`827–837`); BPB reductions are conditional on an initialized group(`642–645`). No forward collective is requested. README:219–223 requires a full, rank/world/batch-independent model.
- Other reachable native code matters to the bisect: RoPE uses shape-selected grid2 at B1/T1024(`331–349,2625–2628`); convolution selects grid1 for odd B, grid2 for even B(`836–841`). These helpers and CE2/load hook are AST-identical in exacte1bd, so an e1bd LNC1 failure would not implicate CE2 evaluation. No LNC environment is set by the source; only the training log prints it. Import-time compiler defaults also differ from the main-entry defaults(`36–39`), another condition to record, not alter silently.

## Pins and remaining proof

LaneB `research/v2/proposals/20260930-r113-laneb-wrapped-ce2-flat-z/train.py`: dcdef066d8d0827f4ecdf0a80f8517e5348af28eea7acc06e03df4db49d4e90d.
Exacte1bd `research/v2/proposals/20260929-r109-ce2-flat-zbundle/train.py`: e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310.
Trusted `prepare.py`:5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb; README:4e4d39d5ef9d4bd69bc2048821292ce74f2ea99b684a823f14f3b01de2b11316.

Companion `20260930-r151-independent-laneb-eval-wrapper-audit.py` performs only pinned-source AST assertions; JSON records exact source evidence. The actual failed scorer traceback, its LNC/core visibility/environment/SDK/import paths and shapes are unknown. No authenticated installed attention_cte/attention_bwd source was used to infer hidden constraints. Root's explicitly authorised fresh single-process diagnostic must supply runtime localization; changing LaneB or claiming grid2 as the established cause is premature. CS16r3 and CS18 fixtures remain frozen/pending behind R151.
