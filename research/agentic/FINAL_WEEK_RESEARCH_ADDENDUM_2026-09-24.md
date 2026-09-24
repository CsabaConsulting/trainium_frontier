# Final-week Researcher addendum — 2026-09-24

This is planning evidence supplied after the 2026-09-24 reconciliation, not launch authority and
not a controller rebase. The active pinned commit, train hash, fixed queue, disk floor, verifier,
and Operator cleanup boundary remain unchanged until the separate canonical-root change is reviewed.
Public BPB below always means context 1024. The selection-grade population is 20,971,520 tokens;
2,097,152-token values are screens and are not interchangeable with it.

## Reconciled base and metric

- The best measured one-factor file is the MATRIX_LR 0.015 derivative, SHA-256
  `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.
- Its two same-host `trn21` full20M values are `0.9920722145` and `0.9918353324459732`.
  Their mean is `0.9919537735`; relative to the c595 same-host control `0.9932319408`, the
  individual deltas are about `-0.0011597263` and `-0.0013966084`, mean `-0.0012781674`.
- These are public-development measurements. They do not establish an official-score improvement,
  and the original manual 0.0015 KEEP bar did not classify the first arm as a win.

## Measured local evidence

- MATRIX_LR `0.02 -> 0.025` measured `0.9946719707`, delta `+0.0014400298` versus its same-host
  c595 control: discard that direction.
- WARMDOWN_RATIO `0.75 -> 0.60` measured `0.9938121843`, delta `+0.0002191816` versus its same-host
  c595 control: no improvement. Warmdown 0.85 remains unmeasured, not implicitly promising.
- Capacity did not transfer monotonically: the archived 8x1024 short evaluation was worse than the
  nearby 6x1024 path, and 6x1152 full20M (`0.9938092803`) was worse than 6x1024 full20M
  (`0.9928513116`). Do not spend a final-week run on width/depth merely because it is larger.
- The parked local-conv R3 candidate targets the old width-640 base. Its patch or effect must not be
  transferred to the dense 6x1024 / c595 / 116188 frontier without a fresh source audit, port,
  proposal, and independent verification.

## Research order

1. Audit source wiring first. The highest-information bounded code lead remains the same-block
   local-convolution ordering issue: Q/K/V are captured before the local-conv residual. Any dense
   port must be re-derived against the eventual canonical root and change only that causal wiring.
2. Prefer constant-only tuning around the measured MATRIX_LR 0.015 basin (for example separately
   proposed 0.0125 and 0.0175 arms) before broad architecture or optimizer stacks. Each constant
   value is one causal experiment with a same-host current-base control.
3. Use a 2,097,152-token context-1024 screen only to nominate a 20,971,520-token confirmation.
   A screen can never promote or adopt a candidate. Adoption requires two negative full20M runs,
   same-host current controls, and mean delta at most `-0.0008`.

Evidence identities were supplied with the final-week status reconciliation. Retain the exact run
receipts and source hashes when the canonical root is selected; this addendum does not substitute
for them.
