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
