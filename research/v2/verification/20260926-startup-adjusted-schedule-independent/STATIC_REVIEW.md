# Preliminary source/fake-clock review

Exact7419927c/source36a86274 against baselineb4d83b5d/source3e79 has no blocking
finding in this bounded static review. **PASS_STATIC_AND_FAKE_CLOCK_ONLY** is
not fresh source approval or numerical model, full-size CPU/native acceptance.
Raw tiny evidence and a genuinely fresh independent numerical review are pending.

Own stdlib714 checks pass normal/-O:576 physical-time reference cases,7 exact
fifth-reset transitions and32 extracted actual LR/WD progress-input cases.
Removing only the added helper/CLI/local and reverting the two intended input
substitutions restores the entire baseline AST. All other functions, model,
load/save, budget/stop/logging/checkpoint and clock-sampling code are unchanged.
The actual parser is defaultFalse and supports explicit ON/OFF. OFF never
calls the helper; LR and WD use the same schedule elapsed.

The new helper uses step-start time, not a post-forward/compile clock read.
For steps0–4, progress is zero only while process age plus30-second source
margin fits within900: the onset is870seconds. Excess is retained. After the
fifth boundary, the original budget reset remains exact and step5+ forwards
the existing elapsed unchanged. Any wording that calls all process ages<900
uncharged would be incorrect. This intentional optimizer schedule change
can alter updates; it neither proves ON equivalence nor explains BPB spread.

The selected profile excludes batch-ramp/EMA/dynamic-loss-scale stacks; their
existing clocks are untouched. Downstream momentum/beta2/DEMON consumers
retain the current shared wd_progress formulas. The CLI flag is not added to
saved architecture toggles; the inference/load code is unchanged and this is
not a resumed-training schedule-state claim.

No tensors, model imports, source edits or host actions occurred. This own
technical review and its tests/results must not be included in the fresh
reviewer's packet. The independently prepared neutral packet supplies only
complete diff, lane specification and raw gate evidence, once frozen complete.
