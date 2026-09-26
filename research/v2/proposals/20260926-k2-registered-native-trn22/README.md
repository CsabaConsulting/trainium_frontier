# Disabled registered K2 native packet — trn22 r1

Proposal-first commit: d9f241f307818ad837c688a5ff3ff99daf552ac8.
No native work or staging has occurred. Checker/mechanics review and a separate
one-use Operator assignment remain pending. Parent reported the adapter's limited
fresh source/CPU review; that receipt must accompany release and does not cover
this new native checker. Current jobs are protected.

The exact 44-line adapter from aa3d4762 is unchanged. The registered candidate
alone is the integration experiment; no kernel, reference, dtype, threshold,
production training, routing, data, evaluator or compiler flag was changed.
Original full-case and FP32-first failures remain historical failures.

`check_native.py` uses FP32 unique first; candidate/control success unlocks all
20 original fixture/dtype pairs, then two changed-tangent observations interleaved
after each dtype's suite. Original custom Function observations stay separate.
The original 150 checks per rank remain required, plus ordinary compiled controls,
sentinel and replays. Empty metadata preserves the original direct-only scope;
it is not silently promoted to an empty-autograd test. The ordinary control here
is compiled, as required by the native proposal.

`validate_receipts.py` reconstructs expected gradients and input hashes using
the exact protected stdlib reference. Candidate success cannot be asserted by
an all-zero receipt, missing rank/case/replay, stale source/argv or contradictory
status/exit. Old Function mismatch is not counted as candidate PASS and does not
mask candidate comparisons. `native_qualification` stays false in receipts:
complete data still needs independent result review.

Own tests: stdlib synthetic receipts and mocked effects only, normal and `-O`.
They replay nine old-driver scenarios (success, wrong host, STOP, low disk,
wrong hash, existing cache, guard refusal, child nonzero and incomplete result),
verify exact driver/entry-body and shared-helper identity, unchanged old wrapper
AST, source pins/default disabled, complete synthetic data and 25 refusal cases.
There is no new native receipt and no new Torch job. The adapter's prior tiny
CPU evidence stays in the immutable aa3d4762 lane; it is not native qualification.
Any test failure is retained rather than converted into an observed native result.

## Operator recipe — DISABLED, not an assignment

Only after separate gate acceptance/release: stage the exact manifest files in
the new absent root named in `NATIVE_BINDING.disabled.json`. Authenticate the
original existing shared lock inode/device; never replace it. Execute only the
bound dispatcher under the existing sole-Operator route. It holds the original
lifetime flock, authenticates the inherited FD and performs fresh ps/neuron-ls
inside the lock. Driver then runs one exact timeout-wrapped child. The binding
contains the exact commands; `bound-command.txt` is the child argv, **not** an
unguarded command to run by itself.

Before release, current control/eval must be terminal, retrieved and fully
quiescent. All original STOP/resource/hash checks apply. The six diagnostic cache
roots must be absent and are created empty before the child; no existing cache
or checkpoint is removed. Worker env receipts are not proof the SDK honored the
variables; preserve actual compiler/debug/cache evidence and all4 closed CSVs.

Retrieve all logs, rank env/exit/result receipts, terminal and completion records,
cache manifest, all graph/compiler evidence and explicit process postflight.
After **every** exit, the Operator must account for observed timeout/torchrun/
rank/worker/compiler descendants and verify identity absence plus fresh ps and
neuron-ls and unchanged original lock. Any ambiguity parks the route. The existing
driver does not itself claim complete descendant cleanup; there is no automatic
retry, full run, evaluator or next job. 600+60s is a diagnostic watchdog, not an
official 1800s quality budget or permission to exceed it in a quality run.

Runtime cost/compile coverage within 600s is unmeasured; an incomplete/timeout
result remains incomplete even if some individual cases match. Strict host
readbacks can perturb execution and do not reveal compiled VJP operand values.
No cache hit, throughput, quality improvement or production adoption is inferred.
