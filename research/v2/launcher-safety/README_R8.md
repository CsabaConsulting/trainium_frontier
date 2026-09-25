# r8 launcher-source successor: bounded timeout and exact source fingerprints

**DISABLED SOURCE CANDIDATE / NO LAUNCH AUTHORITY.** This is a distinct
Implementer repair directly atop r7 source commit
`e74f99958067a10de1467c8801ad601fadd46684`, responding to the
independent r7 source HOLD at
`339c3009b544987fd25d4024579cfa3b75c158ce`. The r7 branch, its
Verifier receipt, r5/r6 disabled packages, frozen 446b source capsule and
host state are not edited. No SSH, AWS, Trn2, transfer, checkout, training,
evaluation, monitor arm, or assignment occurred for this candidate.

The only production edit is `research/fresh_start_launch.py`, whose r8
SHA-256 is `dcf9fdc0e1d8ecc459e6687b87511f00a5747d3fff19dfdfeae0a6b57d818c07`
(r7 was `ece127ad02f72871341369736732f9b324bad9ffa865bc503def7732bb8ad3ab`).
The remaining changes are offline tests and this runbook. The scored model
`train.py` remains
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
`prepare.py`, including the public evaluator/scorer, remains
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
No tokenizer, data, model, optimizer, monitor, guard, host-preflight, or
scoring source is changed. The Git parent-to-candidate diff is the exact
review surface; this document does not claim an independent PASS.

## Fail-closed entry gates

Local validation now requires the exact eleven A0 request fingerprint
paths, once each, before creating `.dispatch.json` or reaching SSH. The
embedded remote launcher independently checks that same complete list
before checkout, host lock, no-spawn receipt, preflight, manifest, or guard.
`research/scratch` remains a dirty-tree scan target, not a fingerprintable
file. Unknown, private, missing, duplicate, directory, absolute and
traversal paths are refused at both entry points.

Both paths also require a bounded GNU `timeout` argv: optional
`--foreground`, optional exact `--signal=TERM`, one mandatory positive
`--kill-after=...`, a positive main duration and a nonempty child command.
Only integer durations with optional `s`, `m`, `h`, or `d` suffix are
accepted, capped at 86,400 seconds for each duration. Zero (which disables
GNU `timeout`), negative, infinite, overflow-sized, unknown/duplicate
options, missing duration/child and child optimization assignments refuse
before local receipt/SSH or remote host lock/guard. The reviewed A0 command
prefix `timeout --foreground --signal=TERM --kill-after=60s 3600s env -i`
is accepted. A malformed direct remote request fails without a bootstrap
no-spawn receipt; that early refusal is a manual HOLD, not a clean finish.

The r7 explicit non-`assert` authority, optimization, host, source,
disk, label and monitor-interlock gates remain. Normal valid requests retain
the existing post-bootstrap private no-spawn receipt and paired-monitor
ordering. No fallback launcher, process signaling, cleanup, or request
enablement is added.

## Offline Implementer checks

All SSH, Git, preflight, monitor and guard calls in the tests are fakes or
isolated temporary files. The new witness matrix directly reverses the
r7 Verifier's `timeout 0` and private-path fake-contact counterexamples,
adds zero kill-after, absent kill-after, malformed duration/option/child,
and incomplete/extra/duplicate fingerprint cases, and checks a valid A0
shape reaches only fake contact. The existing launcher suite also checks
real child interpreters for `-O`, `-OO` and inherited `PYTHONOPTIMIZE`.

```text
PYTHONDONTWRITEBYTECODE=1 python3 research/test_fresh_start_launch_nonassert.py
PYTHONDONTWRITEBYTECODE=1 python3 -O research/test_fresh_start_launch_nonassert.py
PYTHONOPTIMIZE=1 PYTHONDONTWRITEBYTECODE=1 python3 research/test_fresh_start_launch_nonassert.py
PYTHONDONTWRITEBYTECODE=1 python3 research/test_fresh_start_launch_bounded_r8.py
PYTHONDONTWRITEBYTECODE=1 python3 -O research/test_fresh_start_launch_bounded_r8.py
PYTHONOPTIMIZE=1 PYTHONDONTWRITEBYTECODE=1 python3 research/test_fresh_start_launch_bounded_r8.py
```

The six retained r4 monitor/interlock suites must still pass separately.
The historical r3 adversarial Verifier suite encodes old unsafe outcomes;
its three intentionally failing expectations are not an r8 acceptance
criterion. A new independent Verifier must inspect this exact source and
test diff, run its own adversarial checks, and issue a separate source
verdict. Implementer tests alone do not clear the r7 HOLD.

## Exact-source rebind, not authorization

After an independent source PASS, a separate implementer must build a new
capsule/manifest pinned to the r8 commit and launcher SHA above. The old
446b/r7 source hashes and transfer/request bindings cannot be silently
reused. A distinct transfer-only request, independent final review, fresh
full host preflight, one-shot Supervisor assignment, Operator checkout and
independent result review remain prerequisites. The trn22 A0 native request
must then be newly materialized and reviewed, not flipped in place from
r5/r6: rebind expected commit, all eleven source fingerprints, request →
base → pair hashes, monitor identity and one-shot authority to the actual
clean checkout. Reassess G2 numeric carry-forward against new source bytes;
host-specific G1 public-20M calibration remains evidence, not native safety.

Before any later scored run, redo both-host protocol checks and prove clean
matching source, idle process/Neuron/lock state, protected assets, absent
unique label/output, seed 42, full-chip LNC2/four-rank legal command,
matrix LR 0.015, 1,800-second training budget, at least 10 GiB prelaunch
root-free and the immutable 6.5-GiB floor, live paired ≤1-second disk
monitors/observer and one run per host. Fresh public-20M/context-1024
evaluation is separate and follows only a valid retrieved checkpoint.
None of those later gates is satisfied by this disabled source branch.
