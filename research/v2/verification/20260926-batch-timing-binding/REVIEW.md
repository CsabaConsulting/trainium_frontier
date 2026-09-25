# Supervisor mechanical review — batch timing packet

Exact packet `8ad66a83` plus estimator correction `d70a2f2e`, now canonical
`9c907f56`/`a4b4838f`. Tensor-free `audit.py` executed normally and under `-O`:
**38 checks PASS** each. It does not execute the candidate or host driver.

Verified committed a489f7 source, exact proposal/package/driver argv and
environment, immutable prepare/harness pins, full-size CPU receipt hashes and
default identity, unchanged common guard/dispatcher, exclusive host lock and
the inherited descriptor passed into the synchronous timeout/harness child.
No automatic full run or evaluation is present. Command uses isolated Python
without `-O`; the inherited environment cannot silently disable its assertions.

Independently recomputed same-host control median268831tokens/s at steps
10,20,...100 and the fixed5% threshold282272.55. Own fixtures cover passing,
below-threshold, divergent aggregate timing and rejected exit/checkpoint/budget
failures. Candidate observed100-step aggregate remains diagnostic because no
same-definition control aggregate exists; it does not veto against a median.

The source's previous full117M four-rank CPU/default-identity receipts are
authenticated, not rerun here. The new fresh Astra source review is separately
running; this binding review cannot stand in for its own executed reference.
Native accumulation, native speed and startup remain unmeasured. The same
guard's deeper-descendant limitations remain: any failure requires fresh
compiler/process/device quiescence before reusing a host.

Disposition: **PASS_BINDING_ONLY_SOURCE_REVIEW_PENDING**. No host launch
assignment yet; current P is preserved. Following source review, Supervisor
must issue the one-shot assignment and Operator must repeat real preflight.
