# Author implementation receipt — disabled

Only proposal-local helper, tiny tests and disabled binding were added. No host,
SSH, scientific source, checkpoint, evaluator or state-controller action occurred.
No native or full-model execution occurred. This is not independent acceptance.

The helper imports the exact accepted CPU foundation at SHA256
65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8.
That file is an explicit runtime dependency: staging must preserve its sibling
proposal-relative path and exact bytes. No functions or globals in it are changed.
Reused predicates cover manifest paths/pins, actual retained G2 pairing, saved
full-state qualification, masks, byte comparisons and ordinary source loading.
The old d21d35c probe contributed only initialization/synchronization, serial model
teardown, exclusive rank receipt and process-group cleanup patterns. Neither old
probe source pins nor unsafe lifetime behavior are used. Git diff against the
parent is the exact additive derivation; both cited helpers remain unchanged.

Added native observations synchronize then transfer tensors to CPU, preserve
requires_grad, check all registered state plus direct tensor attributes before
and after every call, validate complete masks once then their bytes every call,
and clone outputs before any later forward. The matrix is six shapes and thirty
calls per source per rank. Cross-source original tensors require full byte
equality; every call also requires equal hashes. Offline --validate-receipts
rejects incomplete/duplicate shape/call/rank sets and cross-rank differences.
No collective is added to model forward. Four workers write separate receipts;
the offline validator runs only after their process groups have been destroyed.

Route evidence includes actual restored native device, NKI availability, dtype,
layers, convolution dimensions, flags and exact source text for convolution and
lane dispatch. Reported branch coverage is explicitly predicate evidence, not
instrumented proof of every compiled kernel. Native compilation/graph/resource
observations must be captured by the separately reviewed job mechanics; this
helper does not manufacture those measurements or claim they were obtained.

Author tests use a seventeen-output tiny model and CPU-only synthetic tokens.
They cover all six shapes, state/direct-attribute/input/mode/future/batch/NaN
mutations, signed zero/one-byte mismatches, alias corruption, mask formula
failures, degenerate suffixes, changed and symlink files, hooks, gradients,
synchronization failures, early full-payload rejection without full allocation,
and missing/duplicate/hash-disagreeing cases and ranks. Full-loader native init,
native cleanup and the real65-tensor route remain unexecuted locally; independent
review must supply its own tiny boundary tests, not interpret these as coverage.

Local runtime: /home/poojith-devan/venv/bin/python, torch2.13.0+cpu.
Commands: python -B test_probe.py and python -O -B test_probe.py (from this
directory); both ten-test suites passed. The default probe invocation prints
DISABLED and does not import torch or scientific models. git diff --check passed.

Fresh source/CPU review, exact host/environment/binding/guardian release, and
all actual native results remain pending. The initial13.5GiB disk reserve is
unchanged. No new runner, disk cleanup, tolerance change, training or BPB claim.
