# Independent Verifier receipt: trn22 A0 r4 frozen source 999b0493

**FAIL safety review; host dispatch HOLD.** This receipt covers frozen source
`999b0493c345443306cef5196907c36063b9992a`, not a future repair or a host
package. I used only local static inspection and a synthetic fake system. I did
not connect to hosts, change AWS state, stage files, train, evaluate, or send a
real process signal.

## Blocking finding: run-marker spoof permits an unrelated pidfd KILL

`RealSystem.owned_processes()` at `root_free_guard_monitor.py:165-182` inventories
every same-UID process with the exact, ordinary environment run marker. The
subsequent `_owned_descendants()` at lines 449-465 accepts every such process
whose birth tick is after the guard's; it does not prove ancestry or membership
in the bound child's session. `_kill_live_targets()` at lines 502-528 opens a
pidfd, rechecks identity and the marker, and KILLs it. A pidfd pins identity but
does not prove that the process belongs to this run. The marker is copyable by
another same-UID process.

The independent synthetic test in
`research/v2/disk-monitor/test_r4_marker_spoof_independent.py` creates an
unrelated same-UID process with PPID 1, its own process group and session, a
birth tick after the bound child, and the same marker. After the synthetic
guard and child ignore TERM, the r4 monitor sends a KILL to that unrelated
pidfd. The safety assertion fails:

```text
AssertionError: ('KILL', 145679) unexpectedly found in
[112345, ('KILL', 145679), ('KILL', 112346), ('KILL', 112345)]
```

This test's expected safe outcome is **no signal to the unrelated process**.
It is intentionally red on 999b and should become green on the repaired source.
No real PID is used; `FakeSystem` only records would-be signals.

## Scope-limited checks

- The standard four local synthetic suites run 49/49 green. They do not cover
  same-UID marker spoofing; their unrelated-process case lacks the marker.
- The diff from r3 verifier base `068271a3` to `999b0493` changes monitor and
  test files, not the scored `train.py` or `prepare.py`. Their SHA-256 values
  remain `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Frozen r4 monitor SHA-256 is
  `db0066d529e1c778373a3162689d6a7f25fa022376b063cce560d3be9b551b03`;
  paired monitor SHA-256 is
  `1de80b9ac9eb754b48387b1bc06c9eb4ad15057cc04459fc19823918c42eed0f`.
  The independent test SHA-256 is
  `a0aab190e0f5133b9a61da0d522b052ea55f8a879dd0702471f769bab8036c37`.
- The old r3 adversarial tests deliberately assert unsafe r3 outcomes and are
  historical negative proofs, not acceptance gates for r4.

## Required repair and limits

Require both the inherited marker **and** a provenance predicate tied to the
bound child (such as the authenticated child session), rechecked after pidfd
open before any descendant KILL. An ambiguous or potentially reused session,
unreadable identity, or live process that cannot be safely attributed must
produce an explicit HOLD rather than a broad signal or a false clean pass.
The bound guard and bound child still require their own pinned identity checks.

This is a frozen-source FAIL receipt. It does not release a new source commit,
disabled request, G2 package, or native dispatch. Those require a separate
independent review against their exact hashes and host gates.

## Reproduction (safe local fake only)

```text
python3 -m unittest research/test_fresh_start_monitor_interlock.py research/v2/disk-monitor/test_root_free_guard_monitor.py research/v2/disk-monitor/test_paired_root_free_guard_monitor.py research/v2/disk-monitor/test_r4_interlock_implementation.py -q
# 49 tests, OK
python3 -m unittest discover -s research/v2/disk-monitor -p 'test_r4_marker_spoof_independent.py' -v
# 1 test, FAIL at the no-unrelated-KILL assertion
```

Canonical integration keeps the original verifier branch and command above as
the historical evidence. Its local copy is named
`r4_marker_spoof_verifier_evidence.py` so an unavailable frozen r4 source is
not accidentally collected as a current acceptance test.
