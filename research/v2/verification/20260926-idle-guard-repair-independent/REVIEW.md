# Independent shared-guard repair review — scoped PASS

Reviewed `65f848af21169dff1978ba9009c217473e3ef885`. The specific driver-death
lock gap reproduced against `bdfa06a3` is fixed at the immediate timeout/job
boundary. Original HOLD and reproduction remain in review commit `ab1ac66d`.
No candidate edits, hosts, tensors, native commands or model sessions were used.

The guard now returns the authenticated inherited descriptor; both drivers pass
that exact descriptor into the existing synchronous job call. Authentication
still requires the exact shared lock path, owner/regular-file inode identity,
already-exclusive FLOCK state, and exact immediate outer-flock argv. There is no
unlock/reacquire operation or additional locking path. Fresh `ps` and `neuron-ls`
checks remain inside that live lock and before launch. Direct invocation refuses.

## Independently executed evidence

My own test executes each candidate driver's **actual `subprocess.call` AST**,
unchanged, with harmless local timeout/child arguments. The real guard authenticates
the inherited util-linux lock. Results for both CT and P, in normal and `-O` runs:

| Boundary | Contender result |
| --- | --- |
| Driver waiting, child alive | 1: refused |
| Fixture driver killed, job child still alive | 1: refused |
| Child exited | 0: acquired |

This reverses the old reproduced result (contender acquired immediately after
driver death). Only identified local fixture processes were signalled; temporary
directories were private. The child command has no tensor or device operations.

Own **13 tests passed** normally (0.659 s) and under `-O` (0.691 s), including
both actual driver call expressions, authenticated-FD return/order, direct-call
refusal, a separate identically named driver, busy torchrun, malformed/unknown
Neuron JSON, invalid process PID, command stderr, preserved timeout details and
dispatch contention evidence. The Operator's **15 tests were independently run**
and passed (1.826 s). A test-suite PASS here is limited control-plane evidence.

The 35-line dispatch wrapper makes exactly one fixed flock invocation; it has no
retry, alternative lock, detached child inside its implementation or queue. The
wrapper waits for the outer flock, captures stdout/stderr, and writes an exclusive
terminal receipt in `finally`, including when lock contention prevents driver
entry. My own mocked-contention test confirms one invocation, exit 1 and a failed
outer receipt with no fabricated driver/under-lock receipt. Real lock contention
is separately exercised by the live-process fixture. Exact binding/host staging
and Operator interpretation of that receipt remain the supervisor's review scope.

## Bounds

These bindings have no attached evaluation or automatic next job. The tested
descriptor protects the immediate timeout/job child and the normal synchronous
driver chain. **This is not a claim that every deeper grandchild inherits the
descriptor through arbitrary multi-level process kills.** The unchanged CT
harness's inner torchrun boundary closes descriptors by default, as its packet
documents. Any failure still requires retained process/compiler/device quiescence
evidence before reuse; the next launch must independently pass both fresh idle
checks under the common lock. No orphan, native startup, timing, quality or full
transitive process-tree survival result is claimed. Existing disabled historical
queues must not be re-armed through this review.

Disposition: **PASS for the reviewed guard/dispatch source and the repaired
immediate-child lifetime contract**, with the above explicit scope. This is not
an independent host launch assignment or waiver of exact bindings, fresh machine
preflight, fixed budgets or post-failure quiescence.

## Exact hashes

| Artifact | SHA256 |
| --- | --- |
| Guard | `adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737` |
| Dispatch | `c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a` |
| CT driver | `03ba9c6b200a0c5ff34b5ac3a32fc90940d3f24fc2776671f304a8e057904d86` |
| P driver | `cd7a0f2835c6f8e7d41f7b77ea71c0ee035b0d6168949e975e98f69ba52ff2f3` |
| Own test | `6eae844ca24a03e36471121f3c319f9207d8f07fee16e09df30cdafc33c71015` |
