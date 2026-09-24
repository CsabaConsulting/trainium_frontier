# Independent static review: warm c5c A0 on trn22

Reviewer: root, separate from package Implementer. Time: 2026-09-24 20:55 UTC.
Result: PASS for the exact pending package; **not a dispatch authorization or native result**.

- Core commit: `1ada1953271bccbb1fd2c35d2ac752df62b928a4`.
- Request commit: `6f63f5982c2023d7dce57fbacb7bdb823f6ca9cf`.
- Request SHA: `9f129cb534ba064c4eddd5c2740bad80ae455023d9438904479fb4d17db4fadd`.
- Source is exact research-v2 c5c; prepare.py and legacy launcher family match frozen identities.
- Independently ran `python3 research/submission-validation/20260925-v2-c5c-a0-control-trn22-r1/test_static.py`: PASS.
- Read the exact command/environment, proposal, runbook and corrections. Source defaults are
  seed 42, 1,000,000 step cap, and 1800 seconds; sanitized workload environment leaves no other
  seed/time override. Explicit flags exactly match the completed CPU/G2 A0 probe.
- Historical 499f753c has only an authenticated 2M score and step-0/1 loss anchor, not a 20M
  control. Proposal now records null rather than inventing a 20M score. The 0.99195 +/- 0.001
  reference is a separate workflow equivalence band, not a claim of improvement.
- prepare.record_step_boundary writes fsynced integer lines, not losses/timestamps. Runbook now
  separates stdout loss identity from externally observed boundary-arrival timing and budget summary.
- Warm-cache reuse only; prelaunch disk 7,623,566,951 bytes, operational floor 6,979,321,856 bytes.
  The roughly 0.6-GiB growth estimate is not a guarantee; Operator must monitor and fail safely.
- Preserve checkpoint, logs, source and caches. No deletion or new cloud resource authority.

Before launch: G1 calibration PASS, exact staging/fingerprint and full cleanliness checks, shared
host lock, no other job, cache/disk recheck, and boundary observer in place. Separate Operator
performs dispatch only after controller's explicit request-bound authorization. Full training's
native first-step equivalence, budget, checkpoint, causality and fresh 20M gates remain pending.

## Corrected retained-cache review (2026-09-24 21:04 UTC)

The prior request named `/tmp/neuron_cache`, which was absent on trn22. It was never launched.
Implementer corrected only the cache-path metadata and environment in source commit
`e65a3097a5b00f6178188a935bb2acb9a5042c6c` and externally bound child request
`bb2045e1dbcc87d02541d10183cce54f4762c80e`, request SHA-256
`2948fff0018628a9da5ba406b9c6630dfd6525387b6bb97ef907357464b7f3cf`.
The six paths are the retained same-host 35c9 control's training caches. The old
`track_a_job.py` `fresh_caches(..., "train")` mapping independently confirms the keys
to `neff`, `local`, `hlo`, `inductor`, `compiler`, and `debug`. Read-only inspection of the
exact old run root confirmed all six directories, with 176,381,952 bytes in their parent.

I read the full correction diff and independently reran the exact static contract: PASS.
The reviewed c5c model SHA, prepare.py SHA, complete training argv, seed/time defaults,
compiler flags, label, operational floor, and checkpoint preservation are unchanged.
The source commit includes the corrected proposal/runbook/test; the request-binding child
contains the final request. Operator must dispatch from that authenticated external child
request while checking out the source parent, not use the older pending in-tree template.
Status: **PASS for source and metadata; dispatch still conditional on G1, live six-cache
identity, fresh disk projection, reviewed observer, source cleanliness and host/device lock.**
