# L2 startup predictor port

Per the Supervisor's explicit follow-up, this ports exactly the reviewed timer
correction from `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b` into the frozen L2 source.
The model, architecture, optimizer and schedule have no other changes.

- Base: `99261e750ec8712f6ccb4a2d961d6134d3f87b12`, SHA-256
  `52e234e90f33fc2ccfcc22edb7afe101be25089020a58c352afda4470ac5a7c1`.
- Corrected `workflow/proposals/research-v2/train.py` SHA-256:
  `93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5`.

At the existing fifth-step startup-exclusion boundary, `recent_step_times.clear()`
discards compile-heavy predictor samples. The existing next-step maximum gains
`default=0.0` for the empty history. These are the only two executable edits.
The 900-second startup cap, 30-second launch margin, all charged excess, 1800-second
training budget, 15-second checkpoint reserve and step-cap condition are unchanged.

`test_startup_predictor.py` and `prospective-116188.patch` are byte-identical copies
from the reviewed correction; the latter is exercised in memory only. The scored
116188 vehicle, pinned c5c source and prepare.py remain untouched.
`test_l2_port.py` runs the nine original tests plus three L2-specific tests:

1. Removing the two changes restores the exact original 52e234e9 bytes and entire AST.
2. The real corrected timer boundary AST exactly matches the reviewed a4aa4351 timer.
3. The old L2 timer prematurely stops after the five recorded synthetic A2 durations;
   the port continues with exactly the same startup allowance, excess and charged time.

All 12 tests passed on the existing system Python in 1.236 seconds. The A2 trace
charges 1092.09 seconds in both versions; the fix changes only the stop prediction.
The raw receipt is `timer-port-self-check.json`. Reproduce with:

```text
python3 research/v2/l2-unet8x1024/startup-predictor-reset/test_l2_port.py
```

These are stdlib-only AST tests, with no train-module import, tensor allocation,
training, host contact or model API calls. `git diff --check` passes. They are
Implementer self-checks, not independent approval or a replacement for remaining
exact-size G2, native and full-budget gates. The earlier architecture model receipts
remain bound to 52e234e9; the exact inverse proof supports review of this narrow port.
