# Recursive schedule on scored architecture: research-only port

Frozen source75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f at fe0969fd, based on exact3e79/b4d83b5d. This branch does not change the clean old-schedule artifact3f5035, any submission, evaluator, data path, cache/timer code or launch machinery.

The35-insertion/3-deletion production diff is byte-for-byte the added/removed patch payload from adopted6ef7/a4aa: default-OFF --recursive-schedule; Muon/Adam LR warmdowns0.95/0.65 and floor0.05; exact quadratic Muon momentum to0.79; NorMuon beta2 ramp0.95->0.97 carried through its existing scalar slot. No other part of the external recipe is ported. Existing Apache2.0 attribution is retained. Flat-table LR precedence is unchanged although this dense profile has no ngram tables.

All8 own stdlib test groups pass normal/-O: exact source pins, existing patch payload, all3 helpers AST-identical to6ef7, Boolean/default CLI, dense progress grid/boundary-neighbour values, actual group-update block including flat priority, beta2 tensor dispatch, and every unchanged architecture/loader/helper. Default-OFF schedules/optimizer semantics equal3e79; source file bytes necessarily differ. OFF checkpoint metadata additionally records RECURSIVE_SCHEDULE=False, so serialized checkpoint bytes are not promised identical.

The existing adoption showed mean delta-0.0013265 on6x; this is not evidence of additive benefit on8x. Scored3e79 trn22 public20M0.9863403 is the control source result, not this candidate's result. Old research loader remains exact3e79, including its row-wise evaluation subclass; the clean-source structural REFUSE is a separate unresolved artifact question.

## Pending local and hardware gates

Tiny local live-loop reference/default/OFF/ON G2 passed:8x128 architecture, synthetic micro1x16, four gloo ranks,3steps, one thread/rank, source bytes unedited. Default and explicitOFF match exact3e79 all12 losses, all65 final parameter digests and ownership plan. ON passed finiteness, rank consistency and strict checkpoint reload, with intentional differences in all65 final parameter tensors. Runtime6.2/6.6/5.5/5.7seconds respectively is CPU diagnostic duration, not performance evidence. Test-only PyArrow import support rejects actual Parquet reads; existing G2 stub-prepare, BF16 override, no NKI CE, deterministic1ms clock and micro slicing are recorded. This is4,523,316 parameters, **not** full113,256,724-parameter validation, native execution or timing evidence. No tables/full model on laptop. After the sequential window MemAvailable was6920816KiB and SwapFree999780KiB; no further tensor work was started and the shared slot was released.

Production profile keeps8x1024, UNet skips, attention0,2,4,6,7, initial cache refresh, LR0.015,batch131072,pack32,context1024,seed42; only --recursive-schedule added to exact scored research argv. Complete pending proposal binds those flags. Do not apply --recursive-schedule to unmodified3e79, which does not define it.

Independent source review, full-size host G2, native correctness/cold qualification and separate Operator assignment remain required before any hardware experiment. The ON optimizer graph's existing tensor-slot route may change compilation behaviour; no cache hit, no recompilation absence and no throughput gain are assumed. Wall-clock-dependent scheduling means different actual startup can change updates even if fixed-clock OFF tests match.

2026-09-25 21:40UTC: resumed the previously authorized research port only after freezing the old-schedule clean diagnostic; reversible by leaving this isolated branch unselected. No host, queue, ledger, submission or Downloads change.
