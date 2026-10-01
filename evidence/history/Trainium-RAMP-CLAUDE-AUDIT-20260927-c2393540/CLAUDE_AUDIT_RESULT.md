# Claude compliance audit: ramp submission version (c2393540)

Date: 27 Sep 2026, ~12:55 IST. Auditor: Claude (full diff vs audited BF16 parent 54c835df + evidence).

## Verdict: SOURCE-AUDIT ACCEPT

No blocking issues. Not a guarantee of organiser acceptance.

- train.py sha256 c23935404ac44314e333ddf3608a6c671f4526263a3348eb901bb3f8d4ca05b0
- Delta vs 54c835df = the already-accepted 70773ef4 changes (norm cast, bf16 logits, one-graph MTP
  default ON) plus the batch ramp, default ON:
  - micro-batch shape unchanged (131,072 tokens); only the number of accumulation micro-steps goes
    1 -> 2 at 1/3 of charged training time; rank 0 decides and broadcasts, so all ranks agree.
  - backward seed LOSS_SCALE / grad_accum_now keeps the gradient an average over the micro-steps
    actually taken.
  - guard raises if the configuration is not micro 131072 / total 262144.
- Training-schedule change only. Eval path (targets=None), load_for_eval (weights_only, strict) and the
  evaluator are unchanged. No data or validation access.
- Launch command byte-identical to the accepted parent.
- Evidence: own cold run, startup 641.0 s, excess 0, charged 1789.7 s, 2,687 steps,
  public20M 0.978286 (the research ramp run of the same recipe scored 0.977592; spread ~0.0007).

Upload train.py only with the unchanged command. Expected official roughly 0.9774-0.9780.
