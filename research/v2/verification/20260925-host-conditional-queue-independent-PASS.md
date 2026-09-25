# Independent operational review: conditional host queue

Verdict: **PASS for deployment of the waiting bootstrap and controller-resolved queue.**

Reviewed 2026-09-25 21:22 IST. Candidate:
`600dfd7593ea0eb7228762e786361d2c943e35fd`, with the complete delta from the
previously reviewed `2630351211e477a7028a9b34ff7f5b489fcb77fd` inspected.

Exact `workflow/tools/host_queue.py` SHA-256:
`0ce40ad8aea85aa556b3b87e4753c25049751024a137c69947b8e688e482b50d`.
Exact `workflow/tools/test_host_queue.py` SHA-256:
`b4a39769eba6749fff4b503054d8329a369cd98c2810b35548b9dcd369db2995`.

Independent local execution of
`python3 -m unittest discover -s workflow/tools -p test_host_queue.py -v`
passed **12/12 tests** in 22.830 seconds.
`git diff --check 26303512 600dfd75` also passed. No reproduced defect remains
in the reviewed delta. This Verifier did not alter the runner or contact hosts.

The new behavior is consistent with the controller's conditional-queue request:

- An armed empty bootstrap with a resolution inbox waits without acquiring the
  Neuron device lock. It has no old flag-bank fallback.
- The controller publishes a content-hashed manifest pointer. Resolution cannot
  change the bootstrap host, evaluator, executables, cache paths, run root,
  lock, STOP paths, freeze, default source or inbox. The pointer is explicitly
  an authorized-controller content pin, not a cryptographic signature.
- The queue rereads the resolution before a new job and after acquiring the
  common lock. The regression proves a replaced resolution suppresses an old
  unstarted job while the prior holder retains the lock. In-flight jobs retain
  the earlier train/eval/archive, freeze-completion and inherited-lock behavior.
- Consumed labels remain bound to their exact job fingerprints across queue
  replacements; failed/partial labels cannot be silently retried.
- A job's actual source SHA is separate from its comparison/base identity.
  A supplied same-host-control base identity must match. Proposal binding and
  receipts retain the actual source and argv; the runner computes no verdict.
- The f9b6a9cb schedule source is accepted only through a hash-checked registry
  entry with source-bound Supervisor declarations for independent diff review,
  flag-off identity and a CPU G2 receipt matching the exact job argv. A wrong
  profile, unknown source or changed bytes is rejected. The Supervisor remains
  responsible for declaring PASS only from actual reviewed evidence.
- Optional table throughput/budget metadata is carried into the result receipt,
  without changing the legal launch, fresh public-20M/context-1024 evaluation,
  decision thresholds, or source/evaluator protections.

This PASS permits the Operator to install the waiting bootstrap after normal
deployment preflight. It does not select the conditional table branch, approve
the schedule experiment's yet-to-be-supplied evidence, or assert a live launch.
In particular the reported five-step A2 startup/lookahead termination is not a
valid table-quality result and must not select a table-failure branch. That is
an experiment eligibility issue, not a defect in this runner.
