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

## Source review accepted and one-shot assignment — 2026-09-26 01:35 IST

The fresh isolated Astra session `01a0da21-3da3-7f12-a35a-25c0be650a72`
completed its own 352 checks across 11 bounded CPU reference cases. Its
source/CPU-only PASS is archived in `20260926-fresh-astra-batch262144`.
Supervisor read the complete report, reran its 114 archive checks and the
38 binding checks (normal and optimized Python), and verified the exact
packet hashes below. This supersedes only the pending-source disposition
above. It is not a native performance, quality or adoption verdict.

Assign the sole Operator one trn21 diagnostic, after fresh source, STOP,
disk, process and device checks under the common exclusive host lock:
`fs-20260926-batch262144-timing-trn21-r1`.

- Source SHA256: `a489f7f9e935e4fe9ecd2ad9df5f0f327afcbe9475b71e7fbe0f4e71a48925d9`.
- Source commit: `4e361d15baef94f041501f558620422c55ae8c88`.
- Binding SHA256: `0623920ad14cfdfabdcc9650479e35db58d26d7401a09e3e4282dfe0342c4746`.
- Proposal SHA256: `1528cca9c9f43233abd2e0c91412abe2b8993aae74072cd644ebb6abfa947098`.
- Driver SHA256: `5473a169c991e0666f851c1986b42293d1571c35b2ebdc3704f98398920a9dd8`.

Use the exact pinned argv: Recursive schedule, total batch262144, accumulation2,
seed42, LNC2/four ranks, 105 steps, MAX_TRAIN_SECONDS1800, no LR scaling.
This is a warm-intended research timing screen, not a cold submission test.
Report actual startup and the 100 post-startup steps. Compare the same printed
step10/20/.../100 median against268831tokens/s; pass threshold282272.55.
The different aggregate timing estimator remains diagnostic only.

Keep the same lock for the entire diagnostic. The old queue stays stopped.
Any failed/ambiguous idle check or contention means no launch. Preserve any
error, then require descendant/compiler/device quiescence before another job.
No automatic full run, eval, adoption, cleanup, AWS action or submission.
