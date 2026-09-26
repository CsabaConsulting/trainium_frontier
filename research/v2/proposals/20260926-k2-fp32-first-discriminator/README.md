# Disabled FP32-first K2 discriminator

Proposal-first Supervisor selection: `cedc0af1`. Implementation only; no self-review,
host action or native qualification. Original kernel/reference bytes are copied
unchanged. `original_check_native.py` is retained only to authenticate custom
wrapper AST identity; it is not the executed checker.

`check_native.py` defaults to DISABLED before Torch/NKI import. It records all four
routes even after mismatch or route exception. Pre/post tensor hashes are declared
diagnostic observations. Zero exit means only the one observed fixture is exact;
the original full native gate remains failed and no timing/quality follows.

Own CPU tests use Torch2.13 CPU and a unique-row identity substitute kernel with
`aot_eager`, not installed native Torch2.11/NKI0.5. They test exact fixture and
wrapper identity, ordinary-control continuation after an injected compiled-zero
gradient, nonzero failure return, input mutation including signed zero, missing
gradient, nonfinite JSON and route exception retention. They do not establish NKI
correctness. The initial test loader omitted `sys.modules` registration; both the
generic KeyError and underlying Dynamo `ModuleNotFoundError: discriminator` are
preserved in `failed-cpu-attempt{1,2}.txt`. Only the test loader was repaired.

Native packet and all deployment permissions remain separate and disabled. The
Implementer will not independently approve this checker.
