# A0 trn22 r3 paired disk monitor: disabled implementer package

Status: **HOLD — no host staging, no native dispatch, no breaker reset, no AWS action.**
The prior r2 label and its single-monitor binding remain HOLD. This file is an
Implementer handoff for an independent Verifier and then a separate Operator
decision. It grants no launch authority.

## Exact one-label package

- Label: `fs-20260925-v2-c5c-a0-control-trn22-r3` (new; do not reuse r1/r2).
- Source commit to review/deploy later: `e4272b4dbd8e7b676694afec14b248fef588f8a8`.
- Model `train.py` SHA-256: `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
- `prepare.py` SHA-256: `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Disabled r3 request SHA-256: `9202999d0ffc315e9d9c7ef2d4030a38fc2ca1fbe92ec37888c47a49cebea575`.
- r3 base binding SHA-256: `44b2395470ed17d6569328c6293fb952601904f918af265a86142d46ec767eb5`.
- r3 pair binding SHA-256: `cf35131c0daa829e56a68be756805deefa476b6648152a0857656f9841196817`.
- Base monitor script SHA-256: `ae7538e01c5bff1338bc3c469a7d1e0980e12ade30daff99ae29c5c56ee54511`.
- Paired wrapper SHA-256: `32569f15366c16a8efef651b3db981386e6db6b63e8ed968d1c469e29e73c87c`.
- Launch script SHA-256: `b2527f56fa6f5f7043b721333a8d1209f45ef63a236e52338bb639f6c24c890a`.
- Interlock module SHA-256: `f2fcecd427ea0c4fd5396799fe845d327c2e1384b05e1ef9445b8fb3cf9d2b4d`.

The disabled request's command argv is byte-for-byte the r2 science command
after only the r2-to-r3 output label/path substitutions. The six warm-cache
paths and environment map are unchanged. It retains LNC2/world4, seed 42,
1,800 training seconds, 10 GiB prelaunch root free, and a 6.5 GiB operational
floor. The monitor's stop threshold is 9,126,805,504 bytes (floor + 2 GiB).

## Exact-env CPU G2 source-identity decision — still HOLD

The archived trn22 CPU G2 r3 is **valid evidence for the c5c numeric recipe,
not an automatically transferable launch gate for this r3 source commit**.
Its contract (`research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn22-r3/GATE_CONTRACT.json`,
SHA-256 `164ac39016622ac3918341835b43a53cf3bd94f03bc29596b574d11d1e8ed5bd`)
pins the *entire* checkout at `e65a3097a5b00f6178188a935bb2acb9a5042c6c`.
Its Operator receipt (SHA-256
`c8d73cc447a4fb99c9aba1869f3df47accee550ea2e04d704b9a43cd1bd8e116`)
and independent numeric review (SHA-256
`e76654c7882c2ede9e382914aebb672b10841a083fc0eb2271e15b39965a1de4`)
say PASS **CPU G2 only**, with native A0 r2 still HOLD. The proposed r3
source commit is `e4272b4d...`, not the pinned G2 commit. The diff from
`e65a3097...` is limited to launcher/monitor files, tests, and documentation;
the c5c `train.py` and `prepare.py` SHA-256s, training argv, and four
science-effective environment values are unchanged. G2 invokes the training
script directly, not this native launcher, so the old numeric result is
scientifically relevant, but it does **not** exercise the new interlock.

**Implementer assessment:** under the existing exact-commit G2 contract,
do not rebind or mark G2 PASS for `e4272b4d...` by inference. The default
safe route is a newly labelled CPU G2 against the exact r3 source commit,
with its own independently reviewed contract and receipt. A separate
independent Verifier/Supervisor may instead decide that an explicit,
hash-bound launcher-only equivalence addendum is sufficient under the team
protocol; that decision and its scope must be archived *before* changing
the r3 G2 gate. This Implementer package makes neither decision on their
behalf and has `g2_source_identity_reconciled: false` in its proposal.
Regardless of that disposition, a CPU G2 cannot validate the paired monitors
or authorize native training. The disabled r3 request remains HOLD.

## What changed and why

The e65/r2 launcher created `.job.json` *after* preflight and the guard wrote
`.process.json` only *after* spawning the training child. A monitor dying in
that interval had no authenticated guard PID to stop. The r3 launcher requires
both paired monitors' exact UID, PID birth, argv, script hashes, mutually
acknowledged readiness, and root samples younger than one second before and
after preflight and again immediately before guard spawn. It also rechecks
10 GiB root free. Any failure before entering the spawn-call critical section
refuses the guard and attempts a unique, fsynced no-spawn receipt before
propagating failure. If the receipt
cannot be written, `NO_SPAWN_RECEIPT_FAILURE` is emitted and the run is HOLD;
the failed path still does not spawn a guard. Once the critical section
begins (just before `Popen`), an interrupted or ambiguous spawn emits
`GUARD_SPAWN_STATUS_UNKNOWN_HOLD`,
never a false no-spawn receipt. It never uses a numeric-PID fallback or
signals a process group.

The two monitors start before the launcher and each checks root free every
0.25 seconds while watching its peer. Each independently binds the exact
guard/child receipt, UID, PID birth ticks, argv, source, and pidfd. At or below
9,126,805,504 bytes, or on peer death/staleness after binding, the survivor
sends SIGTERM to **only that authenticated guard pidfd**. The guard's existing
handler owns training-child cleanup. If a peer dies during preflight, the
survivor stays resident, waits for either the launcher's exact no-spawn receipt
or a late exact process receipt, and then stops the guard. If identity cannot
be proven, signaling is refused and the run is HOLD.

## Future gates — not performed by this package

1. Independent Verifier reviews the exact diff, all source/request/binding
   hashes, 36 synthetic tests, raw remote-launch string syntax, no model/
   evaluator changes, single-label paths, and failure boundaries.
2. Supervisor selects r3 and obtains an explicit *one-label, one-attempt*
   authorization. The authorization file does not exist in this package. It
   must be privately owned (`0600`) and bind the exact request, base binding,
   and pair binding SHA-256s with `authorized`, `paired_authorized`, and
   `dispatch_authorized` all true. The disabled request must then be
   re-materialized with top-level `dispatch_authorized: true` and **all
   dependent hashes and bindings re-reviewed**; never edit the disabled file
   in place or treat this package as authorization.
3. A separate Operator checks the host-specific protocol, unique paths,
   exact source commit, 10 GiB root free, idle/clean host, G1/G2 evidence
   **including the source-identity disposition above**,
   no Neuron holder, and no r1/r2 artifacts reused. Any missing proof is HOLD.
4. Only after the prior gates: stage scripts and JSON in a privately owned
   external directory on the host (not in source checkout), start `peer` then
   `primary`, wait for both `PEER_BOUND` and fresh `ROOT_SAMPLE` evidence, and
   recheck liveness. The r3 launcher itself checks the pair repeatedly and
   will refuse a disabled/absent/invalid authorization.
5. Preserve both monitor journals, readiness, interlock/no-spawn receipts,
   launcher, guard, postflight, checkpoint, and logs. No cleanup or deletion is
   authorized by this package. A native run, if separately authorized, still
   requires independent numerical equivalence, causality, checkpoint, and
   public20M validation before being called a control.

## Residual boundary and response

Two independent user-space monitors protect against **one** monitor's death or
stall. They cannot promise safety if both die simultaneously, the host/kernel
is unscheduled or fails, the root filesystem cannot complete writes, or an
unrelated process consumes over the 2 GiB reserve faster than the polling and
guard cleanup path. A guard crash between spawning the training child and
fsyncing its `.process.json` can leave a child whose identity cannot be
proved to the monitors; neither monitor may guess a PID, and that is HOLD.
A valid guard identity mismatch also forbids a signal;
it is an immediate manual safety HOLD, not a success. If this residual is not
acceptable, do not dispatch; kernel/service-level containment needs separate
design and review. The system must never infer a BPB result from a disk stop.
