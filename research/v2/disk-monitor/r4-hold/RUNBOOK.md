# trn22 c5c A0 r4 safety package — Implementer HOLD draft

**No host action or native dispatch is authorized.** This package is an
Implementer draft for a separate Verifier and Supervisor. Its request has
`dispatch_authorized: false`; no one-shot authorization is supplied. The
frozen source is `446b657695f086f9b484fa601c9a3b61c2495912`, while this
package is a later commit. A host source checkout, if separately approved,
must have HEAD at the frozen source commit, not at this packaging commit.

The prior r4 independent receipts for `999b0493`, `f8fe611a`, and `11379083`
are FAIL/HOLD evidence. In particular, the `11379083` review found a
markerless changed-session orphan falsely certified clean and an early
launcher refusal without a durable no-spawn receipt. The `446b6576` repair
has local implementation tests, but **no independent source-safety PASS is
claimed here**. The historical r3 request/bindings pin `e4272b4d` and older
script hashes; they must not be staged or reused as r4 evidence.

## Frozen draft identities

The exact source-file hashes and the request → base-binding → pair-binding
hash chain are in `GATE_CONTRACT.json`. The r4 label is
`fs-20260925-v2-c5c-a0-control-trn22-r4`. The draft preserves the r3
science argv after only the r3-to-r4 label and output-path substitution,
the six existing warm-cache paths, seed 42, four ranks/LNC2, 1,800 training
seconds, the 10-GiB prelaunch root-free gate, the 6.5-GiB operational floor,
and the floor-plus-2-GiB stop threshold. It does not modify `train.py`,
`prepare.py`, evaluator, tokenizer, scoring, or protected caches/checkpoints.

The proposed new source checkout is
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r4`;
the proposed external monitor directory is
`/home/ubuntu/trainium_a0_disk_monitor_20260925_r4`. These are **names in a
draft contract**, not a staging instruction or evidence that either path is
free. Existing r1/r2/r3 paths and protected assets must remain untouched.

## Source transfer plan — separate contract required

1. A distinct Operator first performs a *read-only* trn22 feasibility audit:
   exact host/UID, existing checkout and proposed-target inventory, root
   free/inodes, six warm caches, protected checkpoint identities, host lock,
   matched processes, Neuron holders, breaker state, and label uniqueness.
   A failed or ambiguous check is HOLD.
2. A separate source-stage Implementer freezes an immutable transfer artifact
   for exactly `446b6576` with byte size/SHA-256, complete member manifest,
   and clean-Git reconstruction checks. A separate Verifier reviews the
   artifact and extraction/clone procedure. This draft has **no transfer
   artifact** and grants no permission to write the host.
3. After a new transfer-only Supervisor assignment, a distinct Operator may
   stage only to a previously absent, privately owned new target. No
   overwrite, in-place patch, deletion, cache cleanup, or fallback to an old
   checkout is allowed. Preserve partial output on failure. A fresh-session
   readback must verify artifact bytes before any checkout is materialized.
4. A separately assigned checkout step must prove HEAD exactly `446b6576`,
   a clean tree including all untracked paths, exact hashes for every source
   file in the gate contract, no linked/special/traversal archive members if
   an archive is used, and unchanged protected assets. A separate Verifier
   reviews its raw receipt. None of these checks has happened here.

## Review and one-shot gates

The next independent Verifier must review the entire r4 source diff and
adversarial failures against `446b6576`, this package's hashes and science
equivalence, and the disabled/absent-authorization boundary. The older
trn22 CPU G2 r3 passed numerically on `e65a3097`, **not** on `446b6576`;
it cannot be silently rebound. A new exact-source G2 or a separately
documented, independently approved source-identity disposition is required.

Even after static review and clean staging, a new full host preflight, live
paired-monitor readiness, external optimizer-boundary observer, source and
protected-asset hashes, label uniqueness, and a Supervisor's exact one-label,
one-attempt authorization are separate gates. The disabled draft request
cannot be flipped in place: a dispatchable request would change its SHA,
requiring new base/pair bindings, independent package review, and a fresh
one-shot authority. `fresh_start_launch.py` refuses this draft's false
dispatch flag, and the paired monitors have no authorization file. No native
run, evaluation, score, or control result exists from this package.

Local Implementer checks only (not independent safety acceptance):

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/v2/disk-monitor/r4-hold/test_hold_package.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/test_fresh_start_monitor_interlock.py research/v2/disk-monitor/test_paired_root_free_guard_monitor.py research/v2/disk-monitor/test_root_free_guard_monitor.py research/v2/disk-monitor/test_r4_interlock_implementation.py research/v2/disk-monitor/test_r4_host_cleanup_implementation.py research/v2/disk-monitor/test_r4_early_refusal_implementation.py -q
```

Do not use broad discovery as an r4 acceptance test: historical r3 and prior
r4 negative-proof tests intentionally assert earlier unsafe behavior.
