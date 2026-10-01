# LaneB two-receipt audit brief — NOT ACCEPTED

Exact model source dcdef066d8d0827f4ecdf0a80f8517e5348af28eea7acc06e03df4db49d4e90d,
unchanged command/seed42. The fixed command is in launch-command.txt, SHA256
223d4291ca69d7fc80b83f82d03e3733ddd2e99cd0dfd2659cc0e883258340bd.
Source line numbers below refer to this byte-identical train.py.

The mandatory eight-item checklist is
`/home/poojith-devan/Desktop/Trainium/workflow/REJECTION_CHECKLIST_2026-09-28_0315.md`,
SHA71717878b1c070bb7bce459579ae31f8b26b6339db9c7661859db1f19b9b9612.
UNKNOWN means FAIL pending evidence, never an assumed PASS. These are preparation
findings, not a fresh independent audit or Claude decision.

## Measured research receipts

| Host / exact archive SHA256 | Public-20M BPB | Raw cold through fifth boundary (s) | Charged through exit (s) | Steps / reported tokens |
| --- | --- | --- | --- | --- |
| trn21 / 05a764083999229bb92dc7a9c3b177480dcdb3d0f526cdc78bdb884c73645bac | 0.9663196934983302 | 663.9014979940839 | 1793.414417522028 | 2886 / 571.7M |
| trn22 / defb95ea5df8ac81778e4f0c637ab44846b779ceda006b3b6b16de748e962b38 | 0.9659050462155375 | 645.364225326106 | 1792.7041728659533 | 2895 / 573.6M |

Mean0.9661123698569338; both seed42 on different hosts. Token totals above are
rounded log reports, not exact tensor/checkpoint reads. These are public research
scores, NOT private/official leaderboard scores or independent-seed evidence.
The raw archives, exact released packet identities and member hashes are in
cold-evidence.json and REFERENCES.json. No checkpoint or archive copy was made.

## Eight required items

| # | Required check | Status | Exact evidence and remaining gap |
| --- | --- | --- | --- |
| 1 | Only train.py; exact command; no extra files, installs or missing clean-box imports | **FAIL — UNKNOWN clean-box sufficiency** | Byte-identical source/command appear in both actual FULL packets and archives. Static import inventory is in SOURCE_INVENTORY.json; lines40–45 require provided Torch/prepare,195–197 NKI,974–982 installed torch_neuronx and bundled nkilib (two installed import spellings). No extra custom source module or pip-install call was found in the inventory. Successful research hosts do not prove a pristine organiser image has the same libraries/API/source bytes. No clean-box dependency/import witness is supplied. |
| 2 | load_for_eval returns trained model; same train/eval model/kernels; stateless causal targets=None | **FAIL — UNKNOWN end-to-end native identity / pending audit** | load_for_eval1920–1936 reads only supplied checkpoint, builds GPT, strict-loads saved state, freezes and returns eval model;1914–1917 saves full model_state/config. GPT1345–1417 uses shared causal hash/core; targets=None1411–1412 returns softcapped logits after the shared model. LaneB1014–1032 and attention1095–1098 have no mode/targets-dependent kernel switch; MLP1153–1163 has dtype/tiling fallback, no `.training` attribute exists in AST. Existing author10/own8 CPU-surrogate checks and isolated native pair are authenticated in source-cpu-evidence.json; both FULLs actually completed B1/context1024 fresh eval. This does not establish exhaustive whole-model native train/eval rounding, batch/shape independence or capture parity. Training compiles forward_core1351–1355 while eval calls its uncompiled Python entrypoint; main/import compiler defaults36–39 differ (recorded envelopes show this). That block is byte-identical to A3, not proof of compiler-policy equivalence. Rotary capacity8192 in loader is not the measured evalcontext1024. |
| 3 | No validation/network/external data or weights/hardcoded local paths/cache dependence | **FAIL — UNKNOWN runtime isolation and cache independence** | Training uses provided tokenizer build_if_missing=False1998 and make_dataloader(...,'train',...)2026; checkpoint I/O1914–1936 is explicit. Static import/I/O inventory finds no direct network API or validation loader call in candidate. Lines38/1988 provide `/tmp/neuron_cache` and optional output-dir defaults, not proof of dependency on pre-existing NEFF. Six training-cache roots were empty in both runs; eval cache is separately recorded and reused, NOT empty. Complete clean-box/network/file-access tracing and eval-cache independence are unmeasured. References to provided assets/libraries are not audited here as fresh organiser-environment proof. |
| 4 | Measured empty-cache cold<=900s and charged training<=1800s | **PASS — exact two research receipts only** | cold-evidence.json authenticates train/rehearsal.json fifth-boundary observations, charged-through-process-exit booleans and exact numeric values above; six unique absent-then-created-empty private training caches per run; terminal/Guardian release; stable retained checkpoint; fresh20971520/context1024 successful eval. Preserved pinned R73 gates plus ace0b4db config adapter and exact legacy-member lookup check raw archives, not a summary assertion. This is not a new clean scoring-box execution or future budget guarantee. |
| 5 | TOTAL checkpoint/log/cache bytes written; no large leftovers | **FAIL — UNKNOWN cumulative writes / current inventory** | Each recorded final checkpoint is587788862B, stable through collection. trn21 SHA d1fb7a623a3a537fe4d559d723831626a0989a7dbe2d1c1cb78fa579dbc6253b; trn22 SHA8334b60bdb693d03b72bbb259892ee3af24a16ab5f99322d9acdd81c80cb4c53. Raw collected-text bytes1109960/973998 and free bytes34400452608/36394434560 are snapshots, NOT cumulative I/O or cache totals. No complete write counters or current leftover census; JSON totals remain null. No present retention or backup claim; no checkpoint bytes retrieved. |
| 6 | No required extra env vars; features default ON | **FAIL — UNKNOWN minimal-environment sufficiency** | Main CLI1989 defaults tbl_freeze_tokens=0 and passes it to build_config2000; exact command also explicitly selects0. LaneB52, FLAT72 and Z91–97 defaultON; build_config1419 defaults D1/D2 on but retains historical function-only freeze286000000. GPTConfig.lane_b187 defaultsFalse for old checkpoint compatibility, while new build_config1421 explicitly uses LANE_B=True. Compiler/cache/output environment reads are fully listed in SOURCE_INVENTORY.json; they use defaults. Recorded research harness env adds SDK paths, cache roots, seed/budget and compiler flags. This is not evidence those extra settings are unnecessary on a clean box. No default was changed for this package. |
| 7 | Fixed seed; no timing-dependent behavior beyond time-based schedule | **FAIL — fixed seed proved; broader literal condition pending audit** | init_runtime134–136 and main1992 hardcode42; both packet train environments bind42. Time-based LR/momentum/batch ramp1938–1983 and budget accounting are explicit. Table force/deadline2105–2112 uses remaining time and recent-step estimates; stop2131–2136 and checkpoint reserve58/2169–2175 are timing-dependent safety behavior. No model branch based on wall time was introduced by LaneB, but it would be inaccurate to call the whole program timing-independent or the two trajectories bit-identical. Whether all inherited schedule/stop/table-force behavior satisfies the literal checklist needs independent/Claude adjudication; it is not silently exempted. |
| 8 | Reviewer-readable CLEAN source, no dead/disabled experiment paths; short kernel/shared-path docstrings | **FAIL — concrete retained paths/documentation gaps** | Eight inline NKI kernels have docstrings and LaneB wrappers988/1014 explain layouts/shared path. `_fused_mlp_kernel`1116 and `_FusedMLP`1126 lack docstrings; header1–25 retains CLEAN-GATEFIX framing and omits current LaneB/table/flat details. Z off branches, old D1/D2/freeze compatibility, alternate injection branches1035–1110 despite pinned-site rejection1214–1215, and non-FLAT path2036–2042 remain. Static inventory is not an exhaustive dead-code proof. No cleanup is authorized in an exact-source package; these findings are preserved for audit, not fixed under the same SHA. |

## Scope and review boundary

Existing reviewc5ebb332 authenticates LaneB SOURCE_CPU for research FULL only:
authorfbd77a18 (10 groups), own95cc6f1b (8 groups /125 metrics), nativebenchc81b88f3
(isolated wrapped forward/backward vs SDPA at B32/H8/T1024/D128). CPU runs use eager
kernel surrogates. Neither those checks nor a successful BPB result grants blanket
native numerical-equivalence, submission-readiness or Claude approval.

Both FULL receipts use exact source and command, seed42/world4, finite printed
training losses, checkpoint stability, table pending0 and exact zero four-rank
table difference. Sparse finite printed losses do not prove every unlogged raw
loss; the raw gate scope is preserved. Layout materializations in LaneB are
explicit charged model operations, not a copy-free performance claim.

Fresh independent package review: UNKNOWN / pending. Claude audit: UNKNOWN /
pending. Last Claude-audited ancestor: UNKNOWN. Root must resolve the seven
non-passing rows and separate human gates; this package makes no ACCEPT,
official-score, export, upload, adoption or launch claim.
