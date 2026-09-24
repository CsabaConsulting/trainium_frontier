# Independent review: A0 boundary arrival observer

Reviewer: root, separate from the Implementer. Reviewed source commit
`2d091d82b6eb1e08b306300a62c1c79f0a99c010`, integrated on the control branch.

The observer reads only the exact guard process receipt, step-boundary file,
postflight JSON and live process identity. It records arrival times without launching,
signalling, stopping or changing the training job. Source review found no write operation
in the observer. It must run outside the pinned A0 checkout and write its one stdout
JSON receipt to a unique external file.

Two concrete fail-open paths in the earlier version were corrected: a step line already
present when monitoring starts, and a PID-compatible but wrong guard/step-log identity.
The final version requires an empty or absent initial boundary file, checks the exact
guard script and manifest step-log path, and timestamps after each completed read.

I independently reran both test suites on the frozen clean source: 12/12 fake-process
fixtures passed, including clean sequencing, late start, wrong guard, gaps, duplicates,
truncation, partial terminal line, and timeout. `py_compile` and `git diff --check`
also passed. The old five-test fixture initially failed with the new required
`--guard-script`; the Implementer corrected it and I reran both suites successfully.

Verdict: **PASS for sidecar timing evidence only.** It does not establish training
correctness, an official budget result, a score, or host authorization. Operator must
start the exact script before the first boundary, capture its exit status and JSON,
and preserve invalid timing receipts as failed gates.
