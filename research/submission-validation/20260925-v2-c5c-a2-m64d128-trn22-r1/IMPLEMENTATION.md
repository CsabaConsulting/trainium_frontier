# Implementer handoff: conditional c5c A2 package

This branch records a completed, non-dispatchable A2 proposal and exact gate
contract for trn22. It changes only the proposal, contract, runbook and this
handoff in this directory. It does not edit `train.py`, `prepare.py`, the evaluator,
tokenizer, data, `decide.py`, a launcher or an existing experiment record.
The table profile is already implemented behind existing research-v2 flags;
the candidate source remains SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.

The contract pins the queue's exact A2 argv, four effective science
environment values, four-table geometry, separate G2/smoke/full labels,
trn22's completed G1 calibration receipt, and the missing valid same-host
c5c A0 control. It records the newer human A1/trn21 and A2/trn22 parallel
direction; attributing an incremental layer-1 gain still requires a
same-host A1 result. The A0 r1 zero-step disk failure is not a control.

After independent static review, this repair places the A2 flag list at the
queue's literal `--no-ngram-ve` token position. It also allows the trusted
non-science `NEURON_COMPETITION_R1_STEP_LOG` required by unchanged
`prepare.py` for optimizer-boundary timing. A staged request must hash-bind
that unique append-only path to the external observer; optional
`NEURON_COMPETITION_R1_OUT_DIR` must match the CLI output directory.
Unapproved `NEURON_COMPETITION_R1_*` science overrides remain excluded.

The four BF16 tables imply roughly 1.5 GiB per rank for weights, dense
gradients and RMSProp moments alone. The actual HBM peak, compile/cache and
checkpoint/root-disk footprint remain unknown. The A0 r2 provisional
10-GiB prelaunch threshold is not an A2 bound. Human EBS/guest expansion,
human trn22 breaker acknowledgement/reset, a fresh conservative A2-specific
projection, clean stage and separate one-shot authorization are required.

This Implementer may run only local syntax, hash, argv-equivalence and diff
self-checks. Those checks are not independent verification. The exact-size
four-rank two-step CPU G2 with RMSProp trace, independent rank-record review,
30-step native smoke, full run, fresh public20M evaluation and verdict are
all pending. No SSH, host operation, training, AWS action, deletion, breaker
reset, staging, dispatch or submission was performed in this package.
# 2026-09-25 control-label reconciliation

The pending trn22 c5c A0 control is now assigned and launched as
`fs-20260925-v2-c5c-a0-control-trn22-r10`, not the old unrun r2 placeholder.
The proposal and gate contract name r10 only as a **pending** control; the valid
same-host control label remains null until its fresh public-20M score and
independent postlaunch receipt review pass. No A2 dispatch authority follows.
The older disk/expansion language above is historical; the human has since
expanded both roots and reset the named breaker. Fresh preflight and an
A2-specific growth projection still gate every native table attempt.
