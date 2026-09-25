# Independent Verifier receipt: trn22 c5c A0 r4 source 11379083

**FAIL safety review; native dispatch HOLD.** This review covers frozen source
and test commit `113790835b1c0de237b64c45e9303b36eee49266` on
`codex/20260925-a0-trn22-r4-interlock`. The independent evidence is on
`verify/20260925-a0-trn22-r4-11379083`. I used local static inspection and
synthetic fakes only. No SSH, AWS, host staging, training, leaderboard action,
or real process signal occurred. The reviewer made no implementation fix.

## Frozen identities and scope

| Item | SHA-256 |
| --- | --- |
| scored `train.py` | `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` |
| `prepare.py` | `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` |
| `research/fresh_start_launch.py` | `b2527f56fa6f5f7043b721333a8d1209f45ef63a236e52338bb639f6c24c890a` |
| `research/fresh_start_monitor_interlock.py` | `f2fcecd427ea0c4fd5396799fe845d327c2e1384b05e1ef9445b8fb3cf9d2b4d` |
| `research/fresh_start_host.py` | `9822edf759ad9a88b399b4734e6e9e58998ebc0d9b21c3140dc263f416b2fa24` |
| `research/fresh_start_guard.py` | `5650558c6f4ebdcfd10c54506c3cb2d42a2707d8e750db5f19a9b902558d9b7c` |
| base monitor | `f60fc970c717778e943b1da8c4ffafe6a756cb5e465cf92267f901397b0c529b` |
| paired monitor | `1de80b9ac9eb754b48387b1bc06c9eb4ad15057cc04459fc19823918c42eed0f` |

The scored `train.py` and `prepare.py` hashes match the r3 receipt and have no
diff from r3 source `e4272b4dbd8e7b676694afec14b248fef588f8a8`.
`git diff --check` passed for the frozen source diff. I read the r3 failure
receipt and the two prior r4 independent failure receipts at `971e9fdb` and
`17886a5f` before testing this revision.

The five intended r4 synthetic suites passed **63/63**:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/test_fresh_start_monitor_interlock.py research/v2/disk-monitor/test_paired_root_free_guard_monitor.py research/v2/disk-monitor/test_root_free_guard_monitor.py research/v2/disk-monitor/test_r4_interlock_implementation.py research/v2/disk-monitor/test_r4_host_cleanup_implementation.py -q
# Ran 63 tests in 0.293s; OK
```

These fakes support the narrower conclusions that a marker-spoof outsider is
not signaled; guard shutdown and prior-run preflight inventory it and HOLD;
Popen success followed by pidfd-open or process-receipt failure attempts a
bounded direct-child stop; reused guard, child, worker, and peer PIDs are
refused; a markerless worker still in the live child session is included; a
changed-session orphan **with** a marker causes HOLD; zombies are skipped;
post-bind source drift still stops an authenticated guard; and TERM-ignoring
guard/child escalate to pidfd KILL or unresolved HOLD. The historical r3
negative tests assert old unsafe behavior and are not r4 acceptance gates.

## Blocking finding 1: a markerless changed-session orphan can be certified clean

The guard starts the direct child in a new session. A descendant can later
start another session and exec with a sanitized environment. If the bound
guard and direct child exit, the worker can retain neither the run marker nor
the original child session. `fresh_start_host.cleanup_owned()` at lines
120-134 inventories only those two sets; the guard uses that inventory for
its postflight at `fresh_start_guard.py:128-150`. The host snapshot's
process-name filter at `fresh_start_host.py:173-184` can also miss a worker
with an innocuous command name and no Neuron device handle.

The base monitor uses the same marker/session union at
`root_free_guard_monitor.py:480-509`. With no candidates, `_quiescent()`
returns true and `_finish_quiescent()` accepts a `host_clean=true` postflight.
The paired wrapper inherits this result. The independent fake leaves one
same-UID worker alive in another session, without the marker, after guard and
child exit. It sets root free **one byte below the 8.5-GiB stop threshold**,
still above the host's 6.5-GiB health floor, and supplies the clean postflight
the host code can produce in this case. Both base and paired monitors return
`0` despite the live worker and low free space. Two safe-outcome assertions
fail with `AssertionError: 0 == 0`. No process is spawned or signaled by the
test.

This is a false clean result, not merely a refusal to signal an unproven
process. A repair must either close this provenance gap or explicitly prevent
the monitor and postflight from declaring such a state clean. The current
source does neither.

## Blocking finding 2: early launcher refusal has no no-spawn proof

The remote launch script checks checkout existence, lock acquisition, disk,
source cleanliness, commit, and label at `fresh_start_launch.py:25-56` before
the `try` begins at line 60. Its `write_no_spawn()` call is only in the
handler at lines 109-117. The independent test executes this string locally
with a missing temporary checkout and `Popen` mocked to forbid spawning.
The expected `checkout missing` assertion occurs, but the no-spawn receipt
does not exist. The safe-outcome assertion fails. This reproduces the r3
review's unresolved operational gap; it is **not evidence that a guard did
spawn**. A surviving paired monitor may wait without terminal proof.

The script also sets `spawn_call_started=True` immediately before opening the
log, at lines 99-100. A log-open failure is therefore reported as ambiguous
even though `Popen` has not started. This is another pre-Popen proof gap,
established by static control flow; the independent test exercises the earlier
missing-checkout path.

## Independent reproduction

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/v2/disk-monitor/test_r4_11379083_orphan_independent.py -v
# 2 tests, both FAIL: base and paired monitor return 0 with a live orphan
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/v2/disk-monitor/test_r4_11379083_early_launch_independent.py -v
# 1 test, FAIL: no durable no-spawn receipt after early refusal
```

These three tests intentionally assert the required safe outcomes and are red
on frozen `11379083`. They are reviewer evidence, not implementation changes.

## Package and dispatch gate

The checked-in r3 request remains `dispatch_authorized: false` and pins source
commit `e4272b4d`, while this source is `11379083`. The request and base
binding retain the old guard, host, and base-monitor hashes; the pair binding
retains the old wrapper hash. Their frozen SHA-256s are respectively
`9202999d0ffc315e9d9c7ef2d4030a38fc2ca1fbe92ec37888c47a49cebea575`,
`44b2395470ed17d6569328c6293fb952601904f918af265a86142d46ec767eb5`,
and `cf35131c0daa829e56a68be756805deefa476b6648152a0857656f9841196817`.
No r4 request, binding, one-shot authorization, exact-source G2 receipt, fresh
host preflight, or live paired readiness was created or verified here. These
stale r3 bindings remain HOLD even if a later source repair passes its own
review. Any uncommitted r4 rebinding is likewise HOLD until frozen and
independently reviewed. This receipt grants no dispatch authority.

Canonical integration retains the original verifier branch and reproduction
commands above as historical evidence. The local copies are named
`r4_11379083_orphan_verifier_evidence.py` and
`r4_11379083_early_launch_verifier_evidence.py` to avoid auto-collecting
negative tests against an unavailable frozen source.
