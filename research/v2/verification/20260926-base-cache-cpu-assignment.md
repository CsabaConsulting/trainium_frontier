# Supervisor assignment — bcbd full-parameter CPU qualification only

Issued 2026-09-25 20:48 UTC / 26 September 02:18 IST. Sole Operator:
`/root/trn22_queue_operator`. This is the separate exact assignment required
by the frozen disabled packet; it does not edit that packet or adopt its source.

## Accepted prerequisites

- Frozen CPU packet commit2cf504d9b763e98e7130447d52fc93799e26db50,
  `research/v2/proposals/20260926-base-cache-exact-cpu/`.
- CPU plan SHA256
  `b0c2a4abbce771a344e886d8a6da0723d42314997db245bdccb6dcd5b39202bc`;
  manifest `00fe8964313a08558d30b039b55142d092e60613e7b49f5ed7891d922e4ab17c`.
- Exact repaired mechanical review1b8267dd/26fe571f:74 own checks normal/-O.
  Source/receipt prerequisite313ce755:784 tiny checks; Supervisor separately
  replayed its354 non-tensor authentication checks normal/-O.
- Fresh isolated source reviewfb096c6f, canonical25317b2a. New session
  `01a0da4a-5cd6-78e2-b0dc-33267b8ab74b` ran20:38:21–20:44:20UTC, exit0.
  Input-only packet manifest56172b63b560d7a437a2fbf601b175be7b872e2e2e265adce7302daa03956fae;
  no inherited context, branch or notes. Its own tiny reference passed3183
  bitwise comparisons in8 cases, with96 actual custom-backward executions.
  Supervisor read its full report/provenance, authenticated all50 archived
  files and checked every comparison's shape/dtype/hash/exact/zero-error fields.
  Verdict is only PASS_LIMITED_CPU_QUALIFICATION_SUITABILITY for BF16 compute
  with FP32 masters. Retain its pre-existing FP32-alias/hot-load limitations.

## Exact permitted sequence

1. Preserve the current architecture run's attached fresh20M evaluation to
   terminal exit. Retrieve its result/error and ensure all descendants/device
   holders are quiescent. Do not interrupt or overlap it. No old queue restart.
2. On trn22 only (`ip-172-31-32-253`), stage the exact packet/source blobs to
   its absent root
   `/home/ubuntu/trainium_base_initial_cache_bcbd_cpu_20260926_trn22_r1`.
   Follow the completely read CPU_PLAN and OPERATOR_RUNBOOK unchanged; verify
   every source/request/driver/support hash and existing real prepare hash.
3. Perform four serial CPU-only probes in order: reference6ef7, candidate
   default, explicit OFF, ON. Each exact label/argv is in the pinned plan.
   Acquire the original exclusive common remote flock for EACH probe; fresh
   ps AND neuron-ls under that lock must PASS, alongside STOP/breaker, disk
   >=6.5GiB plus retained checkpoint headroom and MemAvailable>=64GiB.
   Preserve the lock through each synchronous job. Busy/unknown/refused means
   no launch and retain the actual error. No changed command, host or shape.
4. Before each subsequent probe require prior child/torchrun exit0, no timeout,
   expected4ranks/3updates/full117453580 parameters and successful unchanged
   G2 comparison where applicable. Require actual descendant/device quiescence;
   outer exit alone is insufficient. Stop on the first mismatch/error, with
   all evidence retained. No automatic retry or tolerance/shape reduction.
5. Retrieve raw logs, rank reports, requests, environment, under-lock/outer/
   terminal receipts, exact comparisons and checkpoint identities. Retain
   checkpoint files on host; do not download/commit/delete them. Run the
   existing result auditor and send artifacts for independent numeric review.

This uses small synthetic CPU activations and a fixed clock with the full
parameter model; it is NOT native/full-production workload or budget evidence.
Full-size standard G2 does not expose intermediate gradients/optimizer state;
do not claim them checked. All four whole-checkpoint hashes must agree.

No native, cold-cache, full training, eval, source adoption or submission is
authorized by this assignment. The separately reviewed cold-five binding
98751c5e/79402cfd remains disabled pending numeric acceptance and its own exact
assignment. An actually ready architecture decision-tree follow-up can preempt
this sequence before its first launch; there is no such released job now.

Reason: the independently accepted cache-refresh source needs exact-size CPU
qualification before its selected CT test. This bounded existing-host sequence
can run while other host training continues, without changing any AWS resource.
Undo: revoke before first launch; after a launch preserve its evidence and
complete/stop under the defined failure rules, never kill unrelated work.
