# Independent review: trn22 fresh public20M calibration

Verdict: **PASS for trn22 evaluator settings only**. This is not a new training result, a same-host advancement control, or an authorization for a scored run. trn21 is uncalibrated in this migration.

I independently inspected the preserved Operator receipt and raw log in
`research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn22-r1/` after the Operator committed them at `0c84a48510c603aa320079fa92d45a83308fb0b7`. The receipt SHA-256 is `d78803b0ed21f9f4470560840ae7f78bb2b8ccaf2c82d4d8d9a472c4a6445a57`; the individual artifact hashes match its integrity section. The request SHA-256 is `2efe1b6404ab262e818ef607a4630e82d641800a03813bf0427c3ad82a4d35b6`.

The process record proves a fresh-process `prepare.py eval-public` invocation on the retained checkpoint, with `--device neuron --batch-size 1 --seq-len 1024 --eval-tokens 20971520 --timeout-seconds 1200`. The unchanged `prepare.py` SHA is `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`. Its CLI source sets the default public population to only 2,097,152 tokens, so the explicit 20,971,520-token argument matters. The checkpoint SHA-256 was reauthenticated as `1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b` and preserved.

The evaluator's own JSON reports 20,971,520 tokens, 83,462,186 bytes, 57,444,784 nats, `eval_over_timeout=false`, and BPB `0.992968300659387`. Independent arithmetic `57444784 / (83462186 * ln(2))` reproduces that BPB exactly; it equals the historical known-checkpoint score. Command return code is zero, the guard reports no failure or cleanup, and the postflight has no Neuron holder/process. Two later checks 25 seconds apart found stable hashes and no processes.

The raw kernel log includes a TPB reset request at attach (20:53:23 UTC), its completion at 20:53:30, HBM scrub, and no later reset/error through device release at 21:05:46. Since the measured evaluation and clean postflight succeeded, this observation did not trip the correctness or containment breaker. This is a limited run-specific adjudication, not a general hardware-health claim.

The historical checkpoint's original training run failed its disk gate and cannot serve as a valid advancement control. G1 only establishes trn22's exact fresh-process evaluation settings and known-score reproducibility. Do not insert it as a new candidate or result in `ledger.jsonl`.
