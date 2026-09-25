# r7 launcher-source successor: optimization fail-closed implementation

**Source-only Implementer candidate. No host contact, transfer, checkout,
monitor arm, training, evaluation, enabled request, or launch authority.**
This is a distinct safety successor to frozen source commit
`446b657695f086f9b484fa601c9a3b61c2495912`, responding to the
independent r5 HOLD at `919b6f99688d5bd8a23ec35e8cd133faa717b0b6`
and the disabled r6 diagnostic packet at
`2b2bcaf2b44ab6c5477ba01f46e8a57af26e8c00`. Neither earlier
package is changed or made executable by this source commit.

The exact production change is confined to `research/fresh_start_launch.py`.
Its candidate SHA-256 is
`ece127ad02f72871341369736732f9b324bad9ffa865bc503def7732bb8ad3ab`;
the frozen 446b launcher was
`9488efa918119439a33cbe68cb5638bd1af3b15f1f8f61ec790a216e2bc1e3f4`.
The remaining diff is one new offline test file, the prior r4 synthetic
early-refusal fixture updated for the stronger request gate, and this
runbook. Root `train.py` remains
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
`prepare.py` remains
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
No evaluator, tokenizer, scoring, data, model, optimizer, or control-recipe
byte is edited. The final source commit and whole-tree diff are recorded
by the Git branch, not by this pre-commit file hash alone.

## Safety behavior to review

- Local `main()` refuses nonzero `sys.flags.optimize` or any inherited
  `PYTHONOPTIMIZE` before reading a request, creating a dispatch receipt,
  or invoking SSH. Its request/host/label/commit/path/interlock/timeout/
  Neuron/disk checks are explicit conditional exceptions, never `assert`.
  It also refuses a child `PYTHONOPTIMIZE` assignment or environment key.
- The embedded remote `python3 -c` script independently refuses the same
  interpreter conditions before checkout, host-lock, receipt, manifest,
  preflight, or guard side effects. It independently checks one-label
  dispatch authority, host identity, interlock paths, bounded command,
  child environment, 6.5-GiB floor and 10-GiB prelaunch threshold.
  The frozen remote assertions for checkout, disk, source dirt, exact
  commit and label absence are now explicit exceptions. A dangling
  symlink at a label path counts as occupied.
- On a valid, unoptimized request, the existing host lock, paired live
  monitor checks, no-spawn receipts, preflight, manifest, job and guard
  ordering remain. An early optimization or malformed-request refusal
  occurs before the bootstrap no-spawn receipt; it is a manual HOLD, not a
  claim that a guard was spawned or that the paired monitors finished.
  A post-bootstrap failure still attempts the existing private durable
  no-spawn receipt. No process-signaling or cleanup authority is added.

Offline Implementer tests use fake SSH, fake Git, fake guard `Popen`,
temporary local directories, and no host calls. The new suite exercises
hostile requests, optimized interpreter/environment refusals, dirty or
untracked source, wrong commit, below-floor disk, occupied and dangling
label paths, failed interlock, and a valid path ending at a fake guard.
It explicitly compiles both launcher paths with optimization 0 and 1, and
uses child Python processes for real `-O`, `-OO`, and `PYTHONOPTIMIZE`
environment modes. The six retained implementation suites are run as
separate files; the historical r3 adversarial verifier is a negative
record for older source and is not an r7 implementation test.

```text
PYTHONDONTWRITEBYTECODE=1 python3 research/test_fresh_start_launch_nonassert.py
PYTHONDONTWRITEBYTECODE=1 python3 -O research/test_fresh_start_launch_nonassert.py
PYTHONOPTIMIZE=1 PYTHONDONTWRITEBYTECODE=1 python3 research/test_fresh_start_launch_nonassert.py
```

## Rebinding sequence, not authorization

An independent Verifier must inspect this exact diff, rerun offline
negative/positive tests, and separately accept the launcher source.
Only then may a separate Implementer make a new exact-source capsule and
manifest for the new commit/hash; the old 446b capsule and source-safety
PASS cannot be silently reused. A separate transfer-only request,
independent review, fresh full host preflight, narrow Supervisor assignment,
Operator transfer/checkout, and independent result reviews remain needed.
Any eventual trn22 A0 r6 request must be newly materialized after the
actual clean checkout: rebind `expected_commit`, all source hashes,
request → base → pair hashes, and the one-shot monitor authorization.
Do not flip the r5 or r6 DISABLED JSON in place. Reassess G2 numeric-only
carry-forward against the exact new source/argv/environment and retain
host-specific G1 public-20M evidence; neither implies native safety.

Before any separate one-shot native assignment, redo every protocol pre-run
check on both boxes, prove idle Neuron/process/lock and clean synchronized
source, protected checkpoint/cache identities, absent label/output,
seed 42, full-chip LNC2/four-rank legal command, 1,800-second training
budget, matrix LR 0.015, at least 10,737,418,240 root-free bytes, the
immutable 6,979,321,856-byte floor, independently live paired monitors
stopping at 9,126,805,504 bytes with at most one-second polling, and the
read-only boundary observer before guard spawn. One run per host. A fresh
separately authorized public-20M/context-1024 evaluation follows only a
valid retrieved training terminal and actual checkpoint hash. No training
or evaluation step is authorized by this source branch.
