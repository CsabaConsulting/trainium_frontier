# Independent trn22 r4 private-cache-lock review — HOLD

Reviewed 2026-09-25 UTC by a Verifier separate from the Implementer. The
subject is **only** the disabled successor at canonical commit
`729775df49c752c1b3b666dc3b49c1e8ce0bf69e`. I read `AGENTS.md`,
`README.md`, `research/agentic/CURRENT_STATE.md`,
`research/experiment-protocol.md`, and `research/agentic/DECISION_LOG.md` in
that order, then the fixed package and cited evidence. This was local static
review and disposable-fixture testing; no host, SSH, AWS, marker, transfer,
checkout, training, evaluation, or protected-artifact write occurred.

**Verdict: HOLD for these exact bytes; DISPATCH HOLD.** This is not
`PASS_DISABLED_STATIC_ONLY`, a fresh host-preflight result, an enabled request,
or a one-shot Supervisor assignment. Neither the old r3 PASS (superseded by
the late-label HOLD), the prior r4 HOLD, nor this review authorizes dispatch.

## Exact bytes and evidence

| Artifact | SHA-256 |
| --- | --- |
| `GATE_CONTRACT.HOLD.json` | `7dd0601a964fd07ec913b0a6bbb4b7e8a056b7e3d4be6332be3404108c5f14f9` |
| `RUNBOOK.md` | `29f5d932dc5054d8833a2a6087674de67ac0d8207c3c7deef1e130488512a8b2` |
| `stage_r4_remote_lock_hold.py` | `1ad01ed1e141d283c620bf39080d69c6d1adc184b1160d9c4edfea5507dffab1` |
| `stage_r4_transport_lock_hold.py` | `cf8ac477e7001f7c992fd063cf72585948d1a63e85b5e5bcde9c929f7c696837` |
| Supplied `test_lock_final_scan_hold.py` | `2eae8ac26fb30258cb05989a9bdeedd5920520a628f2a49e4d57831437061411` |
| Verifier `test_independent_disabled_static_review.py` | `bcea2b595b9b56575010f2cbd241741f5863debfc96648cb68d6a3f6389aa057` |
| Prior independent r4 lock/final-scan HOLD | `e34f289bb253b7c9d7bf0ab86a07fbd5962401519ff7a75a6f7c62a4b7b0ca9a` |
| Prior controlling r3 late-label HOLD addendum | `94de8bc3d12a4c36cdad6d2160c1db081d4ea0c0035b6024bf7471e389ffaa0b` |
| Historical trn22 group/ACL Operator raw receipt | `753cd90daee7ab595383cff8b1f02b79766986f452bb6a3521dde5065b88e112` |

The contract and runbook agree on the actual package hashes. The sole local
source capsule was rehashed as 104,154,875 bytes, SHA-256
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
The frozen source commit is `446b657695f086f9b484fa601c9a3b61c2495912`;
its launcher, host, and guard source hashes independently matched the three
contract pins. The frozen launcher takes `LOCK_EX|LOCK_NB` on the same
`/home/ubuntu/.cache/trainium_freshstart/device0.lock` and passes the FD to
its guard and child. No source or host byte was changed in this review.

## Blocking finding

`stage_r4_remote_lock_hold.py:306-317` opens the existing lock name with
`O_RDONLY|O_NOFOLLOW|O_CLOEXEC` **without `O_NONBLOCK` and without a prior
type classification**. It calls `fstat`/regular-file validation only after
that open returns. A local synthetic pre-existing FIFO at the lock name
therefore caused the read-only `_open_existing_lock` call to remain blocked
past a bounded 0.5-second child timeout, instead of returning a prompt HOLD.
The child was killed and reaped by the test timeout. Both `probe()` and the
writer's prechecks use this helper. The runbook's claim that an absent or
unsafe existing lock is HOLD is not met for this nonregular path; a future
Operator could lack a completed preflight/failure postflight while the remote
helper waits. This is a static negative fixture, **not** an observation that
the real trn22 lock is a FIFO. Historical read-only host evidence instead
reported a regular, single-link lock file.

This is distinct from the inherited protected-checkpoint path:
`_protected()` classifies the pre-existing checkpoint as regular with
`_safe_path()` before `_safe_open_regular()`; an independent local FIFO
checkpoint fixture was rejected before that open. A replacement race by a
malicious same-UID actor remains within the package's explicitly bounded
residual threat model, but it does not explain away the unconditional
pre-existing lock-file blocking case.

## Other reviewed gates and limits

The prior r4 HOLD's observed-mode contradiction and wrong runbook transport
pin are resolved in these bytes. Historical Operator evidence found `.cache`
`1000:1000` mode `0700`, inner lock directory `1000:1000` mode `0775`, lock
file `1000:1000` mode `0664`, and no relevant POSIX ACL xattrs. The new
no-follow opened/named identity check accepts only that narrow private-parent
exception, rejects mode/ACL uncertainty and inode substitution, and binds
the identity through the probe, write, and receipt comparator. This relies
on ordinary DAC path traversal; it does **not** prove exclusion of same-UID,
privileged, alternate-namespace, indirect-alias, or noncooperating actors.
The historical observation is not a fresh host preflight.

The writer's held cooperative flock, repeated live STOP/process,
checkpoint-holder/lock, Neuron and label checks, final label scan after the
potentially slow checks, one-second scan-to-exclusive-open bound, and
`O_CREAT|O_EXCL|O_NOFOLLOW` target creation are present. Local negatives
confirmed late-label rejection, lock contention/inode/ACL drift rejection,
and a target collision preserving the existing bytes. Pre/postflight
comparators rejected altered boot identity, protected checkpoint and six-cache
pins, disk floor, target type, lock identity, final-label timing, STOP,
process, and Neuron evidence. The frozen exact source, old-path inventory
including file-backed `.git`, and distinct-session hash/postflight comparisons
remain required. This static package has no enabled wrapper, marker, raw
journal, or actual transfer result; those future obligations are untested
and unauthorized here. `main()` and `build_ssh_argv()` both raise
`DISPATCH HOLD` before host or key access.

Local-only reproducible command, run from this directory:
`python3 -B -m unittest -v test_lock_final_scan_hold test_independent_disabled_static_review`.
Result: **22/22 tests completed** (13 supplied, 9 independent). One of the
independent passing tests intentionally *witnesses the blocking defect*; the
green suite is not a package PASS. The independent negatives also cover
observed private modes, ACL uncertainty, named/opened inode drift, receipt
mutations, target collision, and disabled entry points. `git diff --check`
passed for this review branch.

**Next boundary:** keep dispatch disabled. A separate Implementer must make
the lock-file type check nonblocking and preserve no-follow/no-write and
fail-closed behavior; a different Verifier must review those new exact bytes.
Any later enabled transfer-only request still needs its own independent
final review, fresh host preflight, and distinct Supervisor one-shot
assignment. No host permission change, lock replacement, cleanup, same-path
retry, transfer, checkout, G2, native run, evaluation, or submission follows
from this receipt.
