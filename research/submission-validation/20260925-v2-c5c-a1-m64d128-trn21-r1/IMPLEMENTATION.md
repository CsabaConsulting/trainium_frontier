# Implementer handoff

This branch adds only a pending A1 gate package. It does **not** change `train.py`,
`prepare.py`, the evaluator, `decide.py`, data, or any launcher, and it has not accessed a
Trn2 host. A1's exact source is the already-installed c5c research-v2 file; only its
existing command-line table profile differs from the c5c A0 control.

The package contains the inherited A1 proposal, an immutable source/queue fingerprint
contract with `dispatch_authorized=false`, exact full-size four-rank CPU/G2 trace command,
native 30-step smoke gate, and the later same-host full-run decision rule. The local
non-Torch static check passed. It is a **self-check, not independent verification**.

Limits: the full-size c5c/m64 G2 trace, Neuron smoke, same-host A0 control, full A1 run,
fresh public20M eval and cross-host replicate have **not** passed. Both hosts are below
the disk floor, trn22 has a tripped breaker, and the table-run disk projection is
unmeasured after the A0 compile failure. No old m8/stub or different-source native
receipt may be relabeled to fill these gaps. The next action is independent Verifier
review, then human disk/breaker resolution before any Operator host work.
