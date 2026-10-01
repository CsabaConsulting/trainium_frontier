# R163 source audit handoff — FULL pending

READY FOR SOURCE REVIEW ONLY, under Round 163. This copy is intentionally
exported before the running FULL completes. Source, exact diff, independent
CPU evidence, both exact-source public2M diagnostics, matched60 evidence,
and unchanged launch command are included. No upload authorization.

The older draft status/checklist below describes missing final evidence;
it does not block this source review. Actual FULL score, cold/budget receipt
and checkpoint evidence will be added as separate files when collected.
Claude may review now and append the FULL check later. Unmeasured telemetry
is explicitly unknown, not an additional instrumentation requirement.

Only train.py is the proposed submitted payload. All other files are audit
evidence. Parent receipts remain historical under the R158 rationale.

## Preserved draft notes

# e1bd-F1 LNCFIX — EXPORT DRAFT, NOT READY

This is factual audit preparation, not a Claude audit, package acceptance, upload authorization or a completed FULL receipt. Only `train.py` is the proposed model payload. No existing export, source, evaluator, log, checkpoint or state is modified. No host call or local tensor/model computation is performed. The F1 FULL was launched at **2026-09-30 14:42:40 IST**; its completion, fresh public20M score, cold/budget proof and final disk accounting are **PENDING/UNKNOWN** in this draft. Unknown checklist outcomes are FAIL pending evidence, not a source rejection or an invented PASS.

Exact source: `25c5aec53ad7cc20f4dadf437d3ebe6dfa4f4be0214ab39d638a45df5ec7f936`. Parent: `e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310`. SOURCE.diff: `7a8f733effaf815a25c1e001fc51aaaeaaba486749b0d86814ac08b02cbd2b80`. Exact unchanged training command (do not execute as part of this audit):

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0
```

F1 adds the documented import-time logical-core cap and applies it to existing RoPE, Conv2 and selected-lanes selectors. The header sentence plus four code edits reverse exactly to e1bd. Kernel bodies are unchanged. Under literal LNC=2 the source-level grid choices remain the parent's choices; unset/other values cap the same kernels to one lane. Source lines 37, 331–335, 833–838 and 2557–2560 identify the shared rules. This is not a process override or an evaluation-only switch.

## Actual evidence available now

| Evidence | Exact measured scope | Raw SHA256 |
| --- | --- | --- |
| Independent source/CPU review | `ACCEPT_SOURCE_CPU_ONLY`; selectors, default cap and LNC2 parent dispatch identity; no model/native/package claim | `e18a626152f3ae41d4fe5cbab105715634e548f831b30e10c050451e1402d29c` |
| Exact-source LNC1 eval | protected e1bd checkpoint 3a0004; B1/T1024/public2M; BPB **0.9617131518505532**, exact delta 0 | `7c2ffe78f6bde5cb44e607d2f896ece59144bd973ac8a3e3ad178ac8e9e7468f` |
| Exact-source unset-LNC eval | same source/checkpoint/B1/T1024/public2M; BPB **0.9617131518505532**, exact delta 0 | `02c7eff48788894e7ec5d11bf7da3cd34a497c30180860dd49ed5fddc518a8c6` |
| Matched native60 timing | four ranks, steps20–59, same six instrumentation edits as historical same-host e1bd control | `6178c1cfa337bd13d786cee6f7c4c1747c9bce523c90d397a4db1212decc4c02` |
| FULL launch only | packet `04d5cc3170b13a6ebba761a25f93d08fff0ee5759ce4ea2c849e85e0e2588a6e`; original Guardian21, six created-empty training caches, dispatch started | launch receipt `96aebe4cbef7e6591aff1d4f38ab709765fa54401d89ada46363b64e2dab2c74` |

The two exact-source diagnostics use the unchanged documented logits evaluator and no monkeypatch/process override. They are research public2M scorer-condition tests, not organiser results or fresh F1-trained checkpoint scores. They use an existing research environment; they are not pristine-image or empty-eval-cache proofs. The trusted evaluator calls `model(x)`, reaching the logits return at source line1343 before participant CE/CE2, then computes its own CE. This establishes that CE2 is not on that documented evaluation call path, not that every possible organiser failure has a known cause.

Native60 executed the timing derivative **65c5e37b3e7074000f0a0cc0239d8aaf7445a773cba1c1600dc3180113c1ae2e**, not exact25c5. Its six reversible instrumentation edits and forced accumulation2 match control. Actual322278.9230573374 tok/s versus control321708.30637139425 tok/s; +0.17737082774742507%, −1.4427471032831818 ms/step. Cold678.8769907839596 s; charged-through-exit53.71356410509907 s for this short run only. Comparison SHA589ecf2e587aeb352e082a04968b0a7cd923955bed4437621e36f0c6824f35d6 leaves `within_noise=null`; Root's separate disposition SHAa960eaabceec4867b8fc59190f4205660fd0ac56342b7848cb32294a1d1d5e3b judges it within noise. It is a descriptive single-historical-control decision, not a numerical threshold, repeated-run confidence interval, speed win or score claim. Raw rank intervals remain in the copied comparison.

## Historical e1bd second-receipt rationale — not a second F1 run

R158 explicitly directs the eventual brief to state that the e1bd receipts cover the second receipt because training under the fixed explicit-LNC2 command is unchanged. This draft records that **human-authorized rationale**, supported by exact source-level dispatch identity and matched timing; it does not turn an old e1bd run into an actual F1 run. F1 has **zero completed FULL receipts authenticated in this draft**, one launch, and no second F1 run. The required one F1 FULL and its actual checkpoint/score remain pending.

The two historical e1bd public20M/T1024 receipts are trn21 archive088637f6f4db8ce993f5cc52e80ccc95f5e3441aacb5cc06ce88e59af2be1a0f (BPB0.9672630039667548; cold678.5742937349714 s; charged1793.4091268959455 s) and trn22 archive7f45233b93aed4b120afe7d2149a02b24efa31cb0c14221878ee811c9c68294e (BPB0.9669715685328759; cold672.9102984368801 s; charged1793.388613705989 s). Historical mean0.9671172862498154 is **not an F1 mean**. Both use seed42, not independent seeds. Preserved historical evidence and exact canonical archive references are included; prior package/compliance approval is not inherited.

R151 controller reporting of LaneB's official `INFRA_NEURON_CRASH` remains an observed failure report, not raw organiser diagnostics. The local LNC findings explain the tested local failure mode. They do not prove the official crash cause, pristine organiser compatibility, or that all LaneB behavior is repaired by this e1bd source. No LaneB SDPA fallback claim is copied.

## Required eight-item rejection checklist

The unchanged human wording is in `REJECTION_CHECKLIST.md`. All eight are **FAIL — pending complete evidence/audit** in this draft; the narrower actual successes above are retained without being promoted to package PASS.

1. **Only train.py; exact command; clean-box imports — FAIL pending.** Source25c5 and launch-command bytes are pinned. Proposed payload is only train.py; review files are not runtime dependencies. Exact diagnostics and FULL launch succeed in the existing research environment. Pristine organiser imports/minimal-launch sufficiency, no-extra-install proof and full outcome remain unknown. See source inventory and raw eval/launch evidence.
2. **Trained checkpoint; shared causal train/eval path — FAIL pending.** `load_for_eval` line1852 reloads supplied model state; attention line971, MLP line1077 and `forward_core` line1307 are shared. The cap applies at import to both paths; `targets=None` line1343 returns logits before CE2. Two actual exact-source evals pass on the protected *parent-trained* checkpoint. A new F1-trained checkpoint and fresh public20M evaluation are still pending. Pure-causal/full same-kernel compliance remains for the separate audit; selector CPU checks are not tensor equivalence proof.
3. **No prohibited data/network/weights/local-cache dependence — FAIL pending.** The source I/O and env calls are inventoried; no new data path is introduced by F1. Original evaluator/tokenizer are unchanged. `ensure_tokenizer(build_if_missing=False)` at line1930 is not evidence that the supplied helper has no mutation/download path. Existing-cache diagnostics do not establish cold clean-box portability or a dynamic network/file-access trace. No forbidden-dependence conclusion is inferred merely from optional cache defaults.
4. **Measured empty-cache cold<=900 s and charged<=1800 s — FAIL pending exact F1 FULL.** Native60's measured cold678.8769907839596 s and short charged53.71356410509907 s are actual, but not a FULL budget proof. The FULL launch shows six absent-then-created-empty training cache paths. Its eventual raw step4 boundary, cold duration, charged-through-exit duration, successful fresh public20M/T1024 result and Guardian release must be collected. Historical e1bd budgets are labelled historical only.
5. **Total checkpoint/log/cache bytes and leftovers — FAIL pending.** Diagnostic parent checkpoint is587788862 bytes, SHA3a0004d915883002980c884fa16a07aec76eed812cb70c683d96cec807ce683f. Exact F1 FULL checkpoint identity is unknown. Launch free space28350431232 B is an entry snapshot, not final occupancy or total writes. Cumulative/transient/overwritten bytes, final retained checkpoint/log/cache inventory and backup state remain unknown; no cleanup scope is created.
6. **No required env beyond launch; defaults on — FAIL pending audit.** Exact launch includes LNC2 and `--tbl_freeze_tokens 0`. The new default cap1 is source-defined and actual unset/LNC1 evals pass. Optional compiler/cache/runtime defaults and genuine requirements are listed separately by source occurrence, not assumed unnecessary. Research environment success is not minimal clean-box proof.
7. **Fixed seed; timing behavior limited as allowed — FAIL pending audit.** Fixed seed42 and unchanged scheduling/time calls are retained. `startup_adjusted_schedule_elapsed` line1870 preserves inherited startup accounting. End-of-budget stop/save/table-update behavior is inherited and must not be misdescribed as only an LR schedule. No bitwise replay claim or new exception is invented.
8. **Readable clean path; no dead experiments; kernel docs/shared eval — FAIL pending audit.** F1 adds one requested header sentence and leaves all other code unchanged. The inherited source still contains disabled/compatibility paths. Fresh AST inventory lists definitions/docstrings, but is not a cleanliness verdict. No source cleanup is performed to hide retained paths; same-kernel and per-kernel documentation adequacy await the actual audit.

## Required completion before any READY export

`FUTURE_FULL_EVIDENCE.template.json` enumerates the missing actual fields: exact packet/archive/member hashes; source/command/environment/host/seed; cold cache identities and step boundary; cold/charged time; steps/tokens; finite/rank consistency; checkpoint hash/bytes/stability; fresh20971520-token/context1024 process and score; terminal/Guardian/reaped processes; final and cumulative I/O accounting with explicit unknowns; current authority; independent package integrity/compliance review; and exact Claude verdict. A launch receipt must never populate these with guessed success. Parent historical rationale must remain distinctly labelled. No final Downloads package or submission action is made here.
