# Root replay of supplied production-optimizer tiny comparison

2026-09-26 approximately02:59UTC. Root read test_port.py and its reused
independent_test.py in full, then ran the unchanged supplied test using the
installed /home/poojith-devan/venv/bin/python, one thread and pipefail.
Exit0, final Eligible-context CPU B1/B2/B3, train/load and causality PASS.
Full generated records are copied verbatim from
/tmp/confirmed36-tiny-pvyojj2u/test_artifacts/results.json. Checkpoint/tensor
artifacts stay local and are not committed. This is a replay of Implementer
tests, not a new independent oracle or full-size/native qualification.

The existing test explicitly compares exact research36 to clean e1e2f686,
initial weights, all three per-parameter loss/gradients/updates, production
Muon/AdamW optimizer and differentiable cache trees, synthetic changed-suffix
causality and row/fullbatch consistency in tiny D128/V256 CPU models. Prescribed
fake clocks cover optimizer helper values. Native kernels/collectives/full113M
are not executed. CPU prepare/pyarrow stubs and altered nonpersistent eval
floor are disclosed; this does not substitute for ordinary full-size loading.

Separate fresh functional-reference reportc4464dca remains authoritative only
within its explicit scope. The full-size and native gaps are still pending.
