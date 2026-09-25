# Independent Verifier receipt: trn22 c5c A0 r3

**FAIL safety review; native dispatch HOLD.** Reviewed package `e90955fd0dfd49fae39a85874ce780ab0dea3cc6` and source `e4272b4dbd8e7b676694afec14b248fef588f8a8` using local static/synthetic checks only. No SSH, AWS, staging, training, deletion, breaker action, or score claim.

## Scope-limited checks that passed

- The diff from pinned source `e65a3097a5b00f6178188a935bb2acb9a5042c6c` to e427 touches launcher/monitor/tests/docs only; `train.py` and `prepare.py` are unchanged, with SHA-256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`. All 11 expected source hashes in the disabled request match e427.
- Disabled request SHA-256 `9202999d0ffc315e9d9c7ef2d4030a38fc2ca1fbe92ec37888c47a49cebea575`, base binding SHA-256 `44b2395470ed17d6569328c6293fb952601904f918af265a86142d46ec767eb5`, and pair binding SHA-256 `cf35131c0daa829e56a68be756805deefa476b6648152a0857656f9841196817` match the runbook and cross-references. The r3 science argv equals the r2 pending argv after only r2-to-r3 output/label path substitution. Environment and six warm-cache paths are identical: LNC2, four ranks, seed 42, 1800 training seconds.
- The unique r3 label does not reuse r1/r2. Static launch gate is 10,737,418,240 bytes (10 GiB), stop threshold 9,126,805,504 bytes (8.5 GiB), operational floor 6,979,321,856 bytes (6.5 GiB). Stop-to-floor reserve is 2 GiB; launch-to-stop margin is 1.5 GiB. These are package constants, **not a fresh host-space measurement**.
- `dispatch_authorized` is false. No private one-shot authorization exists. Changing the request to true invalidates all current request/base/pair hash bindings, which must be rematerialized and independently reviewed. This package provides no launch authority.
- Independently reran all three implementer suites: 36/36 passed. Python AST parse and `git diff --check` passed. Those tests use instantaneous mocks, not a host disk-pressure/process-signal test.

## Blocking finding 1: source drift vetoes an already-owned emergency stop

`root_free_guard_monitor.py:313-339` authenticates the exact guard and child and opens the guard pidfd. At low root free, `_stop_owned_guard` at lines 345-365 rechecks mutable checkout source bytes *before* signaling that pidfd (`_source_is_exact`, line 348). If any source file changes or becomes unreadable after binding, both the normal and `fail_closed` paths raise; `fail_closed` logs `SIGNAL_REFUSED` and exits at lines 423-465. The paired wrapper inherits this path at `paired_root_free_guard_monitor.py:262-265`.

The independent adversarial tests change **only** the `train.py` hash after successful guard/child/pidfd binding, then cross the 8.5-GiB stop threshold. Both base and paired monitors signal **nothing** while the same authenticated guard and child remain live. Source drift should reject a *new spawn* and trigger an owned stop; once pidfd ownership is proven, mutable source bytes cannot be a prerequisite for emergency termination. Repair by separating pre-spawn source integrity from post-bind stop authority, using the pinned in-memory pidfd/UID/birth/argv identity for contained termination. Test source drift and source-read failure during low-free conditions.

## Blocking finding 2: TERM-ignoring guard outlives the monitor

`_stop_owned_guard` sends one TERM and enters `STOPPING` (`root_free_guard_monitor.py:365-368`). If the authenticated guard remains live after the configured 120-second timeout, `tick` raises at lines 415-417. `fail_closed` only attempts another stop in `BOUND`, not `STOPPING`, and `run` exits/closes the pidfd at lines 441-465. A third adversarial test keeps the authenticated guard and child live after TERM: the monitor exits after the timeout, with one TERM, no escalation, and both processes still running. Both paired monitors share this path. The outer 3600-second `timeout` is not prompt disk containment.

Repair requires a bounded authenticated escalation to KILL of the exact owned process tree after TERM grace, with continued monitoring until quiescence. Do not replace pidfd ownership with an unverified numeric PID or broad process-group kill.

## Other boundaries and required gates

- The launcher checks both mutual heartbeats and root free before/after preflight and before guard spawn. Before its `Popen` critical section it writes an exclusive fsynced no-spawn receipt on failure; after spawn begins it correctly treats interruption as ambiguous HOLD, not a false no-spawn proof. However, initial remote checks at `fresh_start_launch.py:25-56` occur before the receipt `try`; an early failure can leave monitors without no-spawn proof. This is an operational HOLD, not evidence that a guard started.
- Simultaneous monitor/host failure, filesystem stall, and guard crash between child spawn and `.process.json` remain admitted limits. The `>1s` poll-gap check cannot ensure sub-second wall-clock polling during a scheduler or I/O stall; it only detects a delay after execution resumes. These must remain explicit after a repair.
- Prior exact-env CPU G2 pinned the whole e65 checkout. Its c5c numeric result is useful context, but it neither validates the e427 interlock nor satisfies the strict exact-source gate for e427 or a later repair commit. **Do not silently hash-rebind it.** Require a new, uniquely labelled, independently reviewed CPU G2 after the safety source is fixed/frozen, or a separately ratified hash-bound exception before any gate is marked PASS.
- A future Operator must still prove fresh root free >=10 GiB, clean/idle host and device, breaker state, exact source and monitor identities, live paired readiness, and a human one-label/one-attempt authorization. None was checked on either host in this review.

## Reproduction

From this branch, with no host access or real signals:

```text
python3 -m unittest research/test_fresh_start_monitor_interlock.py research/v2/disk-monitor/test_paired_root_free_guard_monitor.py research/v2/disk-monitor/test_root_free_guard_monitor.py -v
# 36 tests, OK
python3 -m unittest research/v2/disk-monitor/test_r3_adversarial_verifier.py -v
# 3 tests, OK; each asserts an observed unsafe outcome
```

The adversarial test file is review evidence only. This branch does not alter the r3 implementation or release any gate. A separate Implementer should repair the two failures; a different Verifier must review the repair before dispatch is reconsidered.
