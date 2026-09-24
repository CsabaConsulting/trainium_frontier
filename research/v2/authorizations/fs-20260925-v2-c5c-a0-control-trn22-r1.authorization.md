# One-run controller authorization: trn22 warm c5c A0

Decision: **conditionally authorize one guard-dispatched training run** for label
`fs-20260925-v2-c5c-a0-control-trn22-r1` only, using the adjacent finalized request
`fs-20260925-v2-c5c-a0-control-trn22-r1.request.json` (SHA-256
`4f1495c86ac236789a085d5f235a3d2c3550653c9690f49ff66bb596e3d17c8e`).
This is a baseline/equivalence run, not a submission or quality-win declaration.

## Passed independent gates

- Completed human-approved proposal and source/static review:
  `research/submission-validation/20260925-v2-c5c-a0-control-trn22-r1/`
  at source commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c`.
  The reviewed child request template SHA is
  `2948fff0018628a9da5ba406b9c6630dfd6525387b6bb97ef907357464b7f3cf`.
  The finalized request changes only `state` and `dispatch_authorized` and adds this
  controller binding; all core fields and argv are byte-for-byte-equivalent JSON values.
- G1 trn22 fresh-process evaluator calibration PASS at
  `research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn22-r1/operator-receipt.json`,
  SHA `d78803b0ed21f9f4470560840ae7f78bb2b8ccaf2c82d4d8d9a472c4a6445a57`.
  The `0.992968300659387` score merely reproduces an old checkpoint; it is not a new candidate.
- Exact c5c CPU/G2 set: four probes and three exact comparisons PASS, receipt
  `research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/operator-retrieval-receipt.json`,
  SHA `0ac6424a52d03ccfe4a32840462282cf937ac7d6642d4df59fc0f22beec1834f`.
- Operator source-stage receipt `research/v2/fs-20260925-v2-c5c-a0-control-trn22-r1.operator-stage.json`,
  SHA `85329a6d4d287de6d7d9de347e19bd32714c5596fe998ddb09e158c4c445247c`:
  unique checkout clean at exact parent commit; 11/11 request fingerprints match; six
  existing warm train-cache roots retain their identities; output/log label absent.
  Independent read-only SSH rechecked clean HEAD and all 11 exact hashes.
- External timing observer code is separately staged outside the pinned A0 checkout,
  remote SHA `a6f437fe8a9d0a14ec3025bc822f1afa12e3f355fca730e4ae0db7ae00263592`.
  Independent tests passed 12/12; it is read-only and has not yet run.
- Stage free space was 7,950,118,912 bytes; a later independent read-only check saw
  7,950,094,336 bytes, above the request's 7,623,566,951-byte pre-dispatch requirement.
  These snapshots do not waive the fresh pre-launch recheck.

## Mandatory final live conditions and execution

The Operator must, immediately before dispatch, recheck hostname, source clean HEAD and
all 11 hashes, retained cache path identities, exact label/output/log absence, shared lock
availability, no other train/eval/guard process or Neuron holder, and free space at least
7,623,566,951 bytes. The unchanged generic launcher performs its own preflight and acquires
the shared device lock; failure of either check ends this authorization without a run.

Start the separately staged read-only step-boundary observer **before** invoking the launcher.
Verify it is alive, its receipt path is unique, and the boundary file is absent before launch.
Use only the known legacy launcher with a transport SSH-key-path override in memory; do not
edit pinned model/prepare/launcher code. The finalized request already contains the exact
full-chip LNC2, four-rank, seed-42, max-1800-second workload and 3600-second outer timeout.
One label is one attempt. A failed launch or run never reuses this label.

Watch disk during the run. If the operational 6.5-GiB floor would be crossed, stop **only**
the verified owned process tree through the guard and record a failed gate. Do not prune
artifacts or caches. Preserve and retrieve all logs, manifests, boundary/timing receipt,
checkpoint and postflight; require clean quiescence, five consecutive observed optimizer
boundaries, sensible budget, finite checkpoint and later independent causality/public20M
jobs. No automatic adoption or submission follows this authorization.
