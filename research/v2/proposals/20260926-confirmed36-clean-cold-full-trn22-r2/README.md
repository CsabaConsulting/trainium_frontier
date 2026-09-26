# Disabled confirmed36 clean cold full replay: lifetime repair

The original Supervisor proposal remains byte-identical to e80692e2469cc7620e75132861462157ffc065bf. All stage, launch and submission authorization stays false. This is one fixed-job control-plane repair, pending fresh independent review and the separate actual cold-five gate/release.

## Contract and exact scope

The baseline at 3378ac01 used outer flock → driver → foreground timeout → protected harness. The unchanged harness's Popen closes the inherited lock descriptor in its child. A timeout could therefore release the outer lock while a child remained alive. The baseline's four-literal driver derivation remains recorded in `driver_derivation`; it describes the historical baseline, not the repaired driver.

The repaired driver stays the exact existing flock child and authenticates that original inherited open-file description using the unchanged idle guard. It additionally pins device 66305/inode 832589 at the final runtime acquisition, immediately before each phase, on every guardian event, and through final postflight. It requires 13.5 GiB immediately before training and 10.5 GiB immediately before evaluation. The earlier staging checks remain supplemental.

`lifetime_guard.py` activates Linux child-subreaper mode before the first helper or job subprocess. Kernel parent/child links, including every task's child list, define ownership. PID/start-time pairs identify observed processes and pidfds prevent PID-reuse signalling. Sessions, process groups, command names and environment labels never grant ownership. An orphan from a new session or double fork is adopted by the guardian. Kernel `waitpid(-1, WNOHANG)` returning ECHILD is required before evaluation and after postflight; an empty name scan cannot qualify.

On phase exit with survivors, driver exception, timeout or caught signal, only owned pidfds receive TERM, then KILL after the unchanged 60-second grace. Reaping continues for at most another 10 seconds. Failure to establish quiescence parks the driver with the same lock open, without a retry or successor. A successful phase which leaked descendants fails with 125. Nonzero phase exits remain their original code; signals retain 128+signal. Training failure prevents evaluation.

Receipt writes cannot interrupt discovery, signalling or reaping. Audit failure first contains the owned lineage and then parks locked. Final process/device/fuser postflight, checkpoint rehash and child quiescence all occur before release. A failed postflight also contains its helper descendants and parks locked. The idle guard still excludes only the exact driver and its authenticated outer flock, so no extra guardian process creates a false busy match.

An initial refusal before any phase starts releases after owned helper quiescence without claiming external device idleness; it never waits for or signals another job. A started phase requires the full locked postflight.

SIGTERM, SIGINT, SIGHUP and SIGQUIT are handled. SIGKILL/fatal native failure of the guardian itself or host death cannot be caught by userspace and is outside this protocol. The guardian has no enclosing timeout and must not be killed while descendants exist. This is not a cgroup or kernel-enforced survival guarantee.

The scientific e1e2 bytes, proven long submission argv, both exact foreground-timeout commands, private cold-cache routes, protected prepare/evaluator, protected harness SHA 7a55cb87…, source1800/startup900 accounting and public20M/context1024/B1 evaluation remain unchanged. No host contact, staging, training, cleanup, AWS action or submission occurred during implementation.

## Local author evidence and required independent gate

`test_lifetime.py` executes the exact immutable harness Popen AST statement against harmless bounded CPU fixtures under real flock. It first reproduces the old defect, then checks normal train/eval, timeout, harness SIGKILL, driver exception, caught signal, train/eval failure, new sessions/grandchildren, unrelated-process preservation, wrong inode, postflight helper failure, unknown quiescence and failed audit writing. Competing acquisitions must fail while descendants, the phase gap or postflight are active. Local fail-locked test guardians are disposed only after every fixture descendant is proven gone; that disposal is not production behavior.

The test program runs in normal Python and -O with unconditional unittest checks; the actual fixture driver runs normal `-I -B`, like production. `test_binding.py` checks the immutable input pins, unchanged numerical argv, runtime disk floors, cold-cache construction, actual harness containment and official accounting. These are author checks, not independent acceptance or native evidence. The next reviewer must receive exact diff/spec/receipts only and execute an independent CPU fault test. Default review verdict remains REFUSE until that is complete.

All actual native cold timing, charge, finite ranks, checkpoint integrity, final fresh20M, explicit remote process/device/original-lock postflight and Claude-via-human submission audit remain separate required evidence. There is no automatic next job.
