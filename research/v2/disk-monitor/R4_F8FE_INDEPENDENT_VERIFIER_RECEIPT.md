# Independent Verifier receipt: trn22 A0 r4 repair source f8fe611a

**FAIL package safety review; host dispatch HOLD.** I reviewed frozen source
`f8fe611ac127a7e00f896ff26f2620896d096946` locally. No SSH, AWS action,
host staging, training, evaluation, or real process signal occurred. This is
not an assessment of a later repair, disabled request, G2 package, or native
launch readiness.

## Corrected monitor behavior confirmed

The base monitor now requires a live, pidfd-pinned child session plus the
marker before sending a descendant KILL. A same-UID outsider bearing the
marker but in another session remains unsignaled and causes HOLD. `setsid(2)`
creates a session with the new leader as its initial sole member; `fork(2)`
passes session membership to children, and `setpgid(2)` cannot move an outside
process into a different session. Thus the live child pidfd is a meaningful
kernel provenance anchor. These Linux semantics are documented by
[setsid(2)](https://www.man7.org/linux/man-pages/man2/setsid.2.html) and
[setpgid(2)](https://man7.org/linux/man-pages/man2/getpgrp.2.html).

The revised monitor's 50 standard synthetic tests pass, including post-bind
source-hash/read failure, TERM-ignoring guard/child escalation, late marked
worker KILL while the bound child remains live, marker-spoof outsider HOLD,
and reused child PID refusal. The paired wrapper inherits the base stop path.
The r3 tests asserting historical unsafe behavior are not acceptance tests.

## Blocking finding 1: guard cleanup still signals marker-only outsiders

`research/fresh_start_host.py:85-127` selects every same-UID process with the
ordinary run marker, rechecks numeric PID/birth and marker, then calls
`os.kill(pid, SIGTERM/SIGKILL)`. It never proves membership in the bound child
session or ancestry. `research/fresh_start_guard.py:53` calls this cleanup
unconditionally in `finally`; the preflight also invokes cleanup for a prior
label. The repaired monitor cannot prevent this separate signal path from
acting on an unrelated same-UID marker-spoof process. The numeric signal also
has a PID-reuse window between the final identity read and `os.kill`.

The independent monkeypatched fake in
`test_r4_repair_adversarial_independent.py` records a would-be SIGTERM to an
unrelated PID 45679, and its no-unrelated-signal assertion fails. It patches
`os.kill`; **no real PID was signaled**. This is package-wide FAIL even though
the revised monitor itself excludes the outsider.

## Blocking finding 2: markerless live session worker yields a clean result

`root_free_guard_monitor.py:449-483` inventories only
`RealSystem.owned_processes(label)`, which selects processes bearing the
marker. A descendant may `execve` with a sanitized environment and lose the
marker while retaining its kernel session. If the guard and bound child exit,
an unmarked worker still in that session is invisible to `_quiescent()`.
With a clean postflight receipt, the monitor returns result 0 while the
worker remains live. The independent fake reproduces this exact false-clean
path; its no-clean-with-live-worker assertion fails. A worker with innocuous
argv (e.g. `sleep`) is not necessarily caught by postflight's limited process
name heuristic. A complete inventory must detect such an unmarked live
child-session member and HOLD rather than certify quiescence; it need not
signal the unmarked process.

## Frozen identities and limits

| Item | SHA-256 |
| --- | --- |
| scored `train.py` | `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` |
| `prepare.py` | `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` |
| base monitor | `f9bd3b0646a6b865d15fbf733d3779e1079ab5fcdaf13d97cf68b1926b7ab773` |
| paired monitor | `1de80b9ac9eb754b48387b1bc06c9eb4ad15057cc04459fc19823918c42eed0f` |
| guard cleanup module | `c4442b917084239099864f8dd53e4a5467ec8440b8eec9057bbdad5b19a84065` |
| guard | `ac6f7d23dbe998b1d9c8ae14e209c6ffc21fac5d927895b1b92b15dc493f6de6` |
| independent tests | `e3a8bbf9b9b0e7ea5c92110b4d6b6f30ab6f8cd7197486270944d7989a62b537` |

No new request, bindings, G2 artifacts, one-shot authorization, host root-free
measurement, idle-host proof, or live monitor readiness were reviewed here.
The previous exact-source G2 cannot be silently rebound to this source.

## Reproduction (safe local fakes only)

```text
python3 -m unittest research/test_fresh_start_monitor_interlock.py research/v2/disk-monitor/test_root_free_guard_monitor.py research/v2/disk-monitor/test_paired_root_free_guard_monitor.py research/v2/disk-monitor/test_r4_interlock_implementation.py -q
# 50 tests, OK
python3 -m unittest discover -s research/v2/disk-monitor -p 'test_r4_repair_adversarial_independent.py' -v
# 3 tests: repaired monitor outsider exclusion OK; guard-cleanup and
# markerless-live-worker safety assertions FAIL
```

The repair must close both failures, then freeze new exact source. A fresh
independent review must cover the whole package and its host-side gates before
any dispatch can be reconsidered.

Canonical integration keeps the original verifier branch and command above as
the historical evidence. Its local copy is named
`r4_repair_adversarial_verifier_evidence.py` so an unavailable frozen r4 source
is not accidentally collected as a current acceptance test.
