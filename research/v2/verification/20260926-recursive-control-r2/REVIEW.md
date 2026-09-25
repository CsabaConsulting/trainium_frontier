# Independent result authentication: Recursive control r2

Supervisor reviewed the Operator's retrieved raw records without changing the
candidate or operating the host. `audit.py` executed normally and under Python
`-O`: **68 checks PASS** in each. The checks use explicit exceptions, not
optimization-removable assertions. The archived result is `audit-result.json`.

## Accepted evidence

- Exact committed `6ef7d090` science bytes, host source/evaluator hashes,
  prospective control proposal, seed42, four ranks/LNC2 and launch arguments.
- Source-reported 1,800-second budget, training exit0, 3,662 completed steps,
  training1,784.8s, charged estimate1,785.1s, `over_budget=False`.
- Separately launched unchanged public evaluator after training exit,
  context1,024, actual20,971,520 tokens, exit0 and no evaluation timeout.
- Raw result0.9899528600562257 agrees with the terminal receipt and with
  57,270,336 nats / 83,462,186 bytes / ln(2).
- Retrieved artifacts' hashes/sizes agree with the Operator's retrieval
  manifest. The checkpoint SHA in the host terminal is
  `eff319b5235835d37f7008c2631c9eec0734983a2545c3345a7e30ab70fca181`.
  The checkpoint is retained remotely; this audit does not claim a local
  checkpoint rehash or a new evaluation.

## Interpretation and limits

The prior same-host **identical configuration** scored0.9900560897366009
at3,669steps. This repeat differs by-0.00010322968037512759. Record it as a
genuine adopted-base control, not a new architecture/kernel/schedule win.
The frozen confirmed schedule-base mean0.9901674436550496 is unchanged.

The selector file remains byte-identical (`fa4a3d3d...`), and accepts the new
record as a valid quality control. Appending this record does not relabel the
earlier Recursive arm or modify earlier results. No adoption decision is
needed for a repeat of the already adopted configuration.

This was a warm research run. Uncapped startup and independently measured
charged-through-process-exit time were not captured and remain null.
The777.7-second startup allowance is a source estimate, not a measured cold
startup. Do not substitute the separate855.072-second cold diagnostic.
No new causality test is claimed. The historical preflight is reviewed under
the rules at its launch; it is not a retroactive proof of the newer ps-plus-
neuron-ls guard. Current/future launches use the newer common-lock guard.

Disposition: **CONTROL_RECORDED_NO_NEW_TECHNIQUE**. Not a submission package.
