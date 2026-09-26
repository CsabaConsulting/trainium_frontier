# Human-selected Muon momentum warmup arm

Authority: Poojith's 26 September instruction, after the running warmdowns.
Parent: `research/v2/proposals/20260926-sprint-p0-clean-batch/train.py`,
SHA256 `8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee`.

Change exactly one literal: `MUON_MOMENTUM_WARMUP_STEPS = 300` to `150`.
Use the confirmed batch262144 / matrix LR0.021 / ordinary warmdown0.75,
seed42, eight-layer1024 base. Do not stack Recursive, warmdown winners or K2.
Hypothesis: restore approximately the old token-length of the momentum ramp
after doubling the global batch. Improvement is unknown, not predicted fact.

Before release: byte reversal and AST proof of this sole literal edit; execute
the real momentum and LR functions on CPU against an independent formula;
verify every other function, optimizer routing, eval loader and timing code
is byte-identical. This is the human-selected scalar arm in existing reviewed
logic, not a new optimizer implementation. No new independent-session review
is claimed and no extra agent is authorized. Existing scientific correctness,
official accounting, source hashes and original lifetime-lock checks remain.

One full1800-second run per host after its current warmdown and attached20M.
Labels: `fs-20260926-batch-lr021-muon150-trn21-r1` and corresponding `trn22`.
Fresh-process public20M/context1024 is the only selection metric; report
steps, startup, startup charge, speed and raw same-host deltas. Use unchanged
decide.py; do not relabel old comparator ledger rows to manufacture controls.
No submission or source adoption follows packaging. Current final-package
candidate remains the confirmed batch/LR base until a measured winner.

Undo: remove unstarted arms from the current queue; preserve all receipts and
leave running jobs alone. K2 rescue remains local and secondary to launches.
