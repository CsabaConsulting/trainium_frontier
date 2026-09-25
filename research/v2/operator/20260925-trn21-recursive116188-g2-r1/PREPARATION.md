# Exact full-model clean Recursive port CPU check

Staged and host-SHA-verified at18:13:24.502734UTC; `staged/staging.json`
is the raw receipt. No CPU execution, native launch or score claimed.
The parent assigned this CPU-only check after the already staged FP8 diagnostic
and before the exact6ef7 new-base native control on trn21.

Port4083 uses empty science argv; exact6ef7 uses the adopted Recursive-ON argv.
Both use the same unchanged preflight_dryrun.py, real trusted prepare.py,
full6x1024 model, four CPU ranks, three steps and synthetic2x256 inputs.
The reused driver differs only in host/root bindings. CPU shims are recorded
by the existing harness; no stub prepare, package installation or native compile.

Independent review must inspect both complete receipts and the unchanged
harness comparator. PASS alone does not certify native kernels, cold routing,
budget or BPB. No port native/cold launch is authorized here.
