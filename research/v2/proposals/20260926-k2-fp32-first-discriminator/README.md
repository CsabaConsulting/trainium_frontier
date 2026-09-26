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

The bound trn22 packet remains DISABLED. `host_driver.py` and `rank_entry.py`
retain their predecessor 710ea677 control-flow ASTs; changes are host/root/label,
checker/validator/entry pins and removal of the old component-timing arguments.
Shared guard/dispatch bytes are unchanged. `validate_receipts.py` is deliberately
new: it requires four complete one-fixture diagnostic observations, exact
pre/post inputs and cross-rank equality, and explicitly cannot grant native
qualification. A child mismatch stays nonzero with raw receipts; the unchanged
driver preserves that failure without calling its zero-exit completion validator.

Operator obligations remain: after the current trn22 full/eval and selected W1
observation are terminal/retrieved/quiescent, authenticate the original lock
66305:832589, acquire the same lifetime flock, fresh ps and neuron-ls under that
lock, >=10.5GiB disk, clear STOP/breakers, exact staged pins and unique absent
paths. One four-rank/LNC2 child, 600s plus 60s TERM grace, six new private caches;
never install packages or auto-launch another job. Explicit actual process and
compiler identities/descendants, device-idle and original-lock postflight are
mandatory after every outcome; parent/lock exit alone is not proof of quiescence.

Author `test_packet.py` performs 67 stdlib checks normal/-O, including twenty
malformed/failing observation variants and nine fully mocked driver scenarios.
It does not run any host process or replace independent review. Packet support
is staged from the files pinned by `MANIFEST.sha256.json` and binding; do not
stage author tests, CPU output or `original_check_native.py` as executed source.

Native packet and all deployment permissions remain disabled. The Implementer
will not independently approve this checker or mechanical binding.
