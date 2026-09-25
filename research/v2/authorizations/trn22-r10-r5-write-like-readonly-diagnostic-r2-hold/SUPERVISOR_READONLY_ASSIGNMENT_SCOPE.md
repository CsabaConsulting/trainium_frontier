# Supervisor assignment — trn22 r10 r5 read-only diagnostic r2

This assigns **at most one** execution of the exact read-only diagnostic
label `fs-20260925-trn22-c5c-a0-r10-readonly-write-like-diagnostic-r2`
after a separate independent assignment check and fresh full local/host
preflight. A completed or failed call consumes this operational assignment;
do not rerun the label. This is not source staging, training, evaluation,
cleanup, marker reset, AWS action, or permission to change a breaker.

The exact execution checkout is
`/home/poojith-devan/Desktop/Trainium-trn22-r10-diag-r2-supervisor-assignment`
at the committed assignment HEAD. Its `train.py` SHA-256 is
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and its `prepare.py` SHA-256 is
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The older command path in `RUNBOOK.md` points to a disabled Implementer
checkout and must not be used.

The designated Operator's only possible command, after those gates pass, is
`/usr/bin/python3 -B /home/poojith-devan/Desktop/Trainium-trn22-r10-diag-r2-supervisor-assignment/research/v2/authorizations/trn22-r10-r5-write-like-readonly-diagnostic-r2-hold/operator_readonly.py`.
Record the bounded result and a distinct-process full postflight; preserve
the consumed r5 marker and every protected artifact. If any gate fails,
HOLD and report the evidence without host write or retry.
