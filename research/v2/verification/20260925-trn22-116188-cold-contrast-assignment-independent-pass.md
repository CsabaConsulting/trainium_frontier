# Independent final assignment review: trn22 116188 true-cold contrast

**PASS for the exact prepared assignment at controller `4a7fc232`; no host
preflight or launch is performed or implied by this receipt.** I did not
author the Researcher proposal or Supervisor assignment, and did not SSH to
trn22. Reviewed file:
`research/v2/assignments/20260925-trn22-116188-cold-contrast-r1.json`,
SHA-256 `7e80e3b3cd027c5a86b55c8863ff4789cf01cadc4419b16cfb29f7c603444571`.
This review clears the two assignment-level conditions in the earlier
proposal HOLD at `3bbd5c68`; it does **not** alter the proposal or approve a
submission.

## Independent bindings checked

- One-use label `fs-20260925-116188-cold-contrast-trn22-r1`, host trn22.
  The source path exists in the controller checkout and hashes to
  `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.
  The assignment's ordered bare training argv matches both the authentic
  four-rank/three-step real-`prepare.py` CPU G2 `PASS` receipt at
  `research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/remote-run/receipts/fs-20260925-g2-s0-116188-trn22-r1.json`
  and the completed September 24 116188 launch proposal. The file bakes
  matrix LR 0.015, seed 42 and 1800 seconds unless environment overrides
  them; the assignment forbids such inherited overrides.
- The assignment says **copy only the exact `train.py`** into the unique new
  sandbox. It binds `PYTHONPATH` to the existing trusted trn22 checkout's
  `prepare.py`, SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
  and requires both a host-file hash and a fresh-process imported
  `prepare.__file__`/SHA check before training and evaluation. It forbids an
  extra mutable evaluator copy. The Operator must recheck immediately before
  use; this offline review cannot establish the host's future bytes.
- A unique absolute, job-owned root and seven distinct cache directories are
  fully bound; all cache paths sit under that root. The assignment requires
  root absence before staging, private precreation of every directory,
  verified emptiness, and an explicit local
  `NEURON_COMPILE_CACHE_URL` at
  `/home/ubuntu/trainium_116188_cold_20260925_trn22_r1/cache/persistent`.
  The other six native/inductor cache paths are distinct. `NEURON_CC_FLAGS`
  must be externally unset, with no `--cache_dir` or other compiler override;
  the training file's baked optlevel-1/BF16 flags therefore apply. The
  assignment requires effective cache-path evidence and new files under the
  persistent-cache URL after first compile, not an unsupported assumption
  that an empty `NEURON_COMPILE_CACHE_DIR` proves coldness.
- The previous trn22 c5c A0 run **and** its fresh public-20M evaluation must
  be terminal before this slot. Fresh Operator checks must prove label/root
  absence, ≥6.5 GiB free, no job, Neuron device or lock holder, no STOP,
  exact source/prepare/data/tokenizer, seed/budget/topology and one job per
  host. A separate fresh-process context-1024, 20,971,520-token eval,
  checkpoint/log retrieval and quiescent postflight are required. No failed
  label may be retried.

## Permitted state transition and limits

For this **exact assignment file and no other**, the Supervisor may change
only `assignment_status` from `PREPARED_PENDING_INDEPENDENT_FINAL_REVIEW`
to `ASSIGNED` and `launch_authorized` from `false` to `true`, then hand the
unchanged run identity/conditions to a distinct Operator. This is not a
waiver of the Operator's fresh preflight. If the cache root is not absent,
any cache is populated before launch, source/import/env mismatch occurs,
disk/idle/STOP fails, or the prior A0 eval is still active, the Operator must
refuse the launch and report the exact failed check. If compiler writes under
the pinned persistent URL are not observed, any run is **inconclusive as a
true-cold contrast** even if training completes.

No possible result of this 116188 contrast makes S0 submittable. S0 still
requires its own new-label, proven cold-cache, full-budget clean replay and
fresh-process public-20M evaluation.
