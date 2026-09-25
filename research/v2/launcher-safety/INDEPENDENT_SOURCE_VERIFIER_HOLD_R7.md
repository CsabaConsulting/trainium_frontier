# Independent Verifier HOLD: trn22 A0 r7 launcher source

**Verdict: HOLD for the operational launcher source gate.** This is an offline
source review, not an enabled request, host preflight, transfer, monitor arm,
native launch, evaluation, or submission. I made no SSH/AWS/Trn2 contact and
did not edit the subject source. The exact subject is
`e74f99958067a10de1467c8801ad601fadd46684` on
`codex/20260925-trn22-a0-launcher-nonassert-r7`, a direct child of frozen
`446b657695f086f9b484fa601c9a3b61c2495912`. Its clean Implementer
checkout was inspected read-only. Independent fake-contact tests and this
receipt are on a distinct Verifier branch.

## Blocking source-gate counterexamples

1. **A zero-duration `timeout` passes as a bounded command.** Local
   `validate_request()` checks only `command[0] == "timeout"` at
   `research/fresh_start_launch.py:281-286`; the embedded remote checks the
   same at lines 59-64. GNU `timeout --help` states that duration `0`
   disables the timeout. An enabled synthetic request with
   `command=["timeout", "0", "true"]` therefore passed normal-mode local
   validation, created a `.dispatch.json` receipt, and reached the fake SSH
   seam. In an isolated fake remote with fake Git, preflight, interlock, and
   `Popen`, it created the host-lock directory, launch/job/log files, and
   reached one **fake** guard spawn. No real guard or command ran. The prefix
   check was already insufficient at 446b (its local line 183); e74 retains
   this inherited gap while claiming a bounded-command gate. An actual
   reviewed pair binding would also authenticate an exact request, but that
   later layer does not make this launcher source gate correct.

2. **A non-source fingerprint path reaches contact and a remote target write
   before refusal.** Local lines 303-308 accept any relative path without a
   dot-prefixed component. The synthetic
   `fingerprint_paths=["research/agentic/private.json"]` consequently created
   a local dispatch receipt and reached fake SSH. The remote's new exact
   whitelist at lines 143-147 rejects it, but only after
   `HOST_DIRECTORY.mkdir()` and `HOST_LOCK.open('a')` at lines 125-127; its
   early-error path then writes a no-spawn receipt. The fake remote recorded
   the lock directory and receipt, with no manifest/job/guard. This path was
   broadly unvalidated at 446b; e74 adds a useful remote whitelist but leaves
   local validation inconsistent and checks the remote field after writes.
   The r6 source-successor brief required malformed or unsafe enabled
   requests to stop before local receipt/SSH and remote state creation.

The exact counterexamples and controls are archived in
`research/v2/launcher-safety/test_independent_r7_adversarial.py`, SHA-256
`9d3e27c1e5b39b8bea37c5a41f9f384ce069fe81b0faddb9bab94e241c46df27`.
All SSH, guard, process, Git, preflight, and monitor contacts in that test are
fakes or existing synthetic fixtures. The two counterexamples require a
normal, unoptimized interpreter; optimization is separately rejected.

## Verified portions and exact bytes

The four-file diff from 446b to e74 is exactly:

| Changed path | SHA-256 at e74 |
| --- | --- |
| `research/fresh_start_launch.py` | `ece127ad02f72871341369736732f9b324bad9ffa865bc503def7732bb8ad3ab` |
| `research/test_fresh_start_launch_nonassert.py` | `3801527195f73744c63caf29e3c8e73a2c58bf8481c87a7f2e46d28ce60ef404` |
| `research/v2/disk-monitor/test_r4_early_refusal_implementation.py` | `f0294a49c7338f53233f8031bf89157da0b596cd4cf358b6cce6f3b6e5564fb3` |
| `research/v2/launcher-safety/README_R7.md` | `0bf81bb4d9984c2e04ec2dde97bc8fef6e700e6365a647f8d7ae9543c41b9016` |

`git diff --check 446b..e74` passed. Direct blob comparison proves
`train.py`, `prepare.py`, `artifacts/tokenizer_8192.zip`,
`research/run_experiment.py`, `fresh_start_guard.py`, `fresh_start_host.py`,
`fresh_start_monitor_interlock.py`, `fresh_start_sync.py`, and both root-free
monitor scripts byte-identical to 446b. Relevant SHA-256 values at e74 are:

| Protected or interlock source | SHA-256 |
| --- | --- |
| `train.py` | `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` |
| `prepare.py` (public evaluator/scorer) | `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` |
| `artifacts/tokenizer_8192.zip` | `12fa2c4a56d9f26d46e8ea0a57f69d78145930eecb0a765c9bfc085176af8a54` |
| `research/run_experiment.py` | `88243a74f0f080929f93d0b88ddedbe6f5351ad37bf272229d9ae6d26efd5d2d` |
| `research/fresh_start_guard.py` | `df902becb2faf009d04d7ecfaee3791df67e822e90ec6112adbc39be81599ecc` |
| `research/fresh_start_monitor_interlock.py` | `f2fcecd427ea0c4fd5396799fe845d327c2e1384b05e1ef9445b8fb3cf9d2b4d` |
| base root-free monitor | `91dae71468d5d029857371ccd2c960769b71874e855bb058130e738b91bb2c60` |
| paired root-free monitor | `9e2d6ed26d918131c3e12b9937093e465b39aa895a9bf2be7339de8ae5094344` |

The scored model, tokenizer, and scoring functions therefore have unchanged
source bytes; this static fact is not a new native/model-result claim. Neither
the prior 446b capsule nor its launcher hash can be reused for e74.

The e74 launcher has no AST `assert` in its local source or embedded remote
script. Its local optimization check precedes request reading, dispatch
receipt creation, and SSH; the independent remote check precedes checkout,
lock, receipt, manifest, and guard operations. The added remote
`dispatch_authorized is True` gate and local strict-boolean gate refuse
disabled/nonboolean requests before fake contact or writes. Inherited or
child `PYTHONOPTIMIZE` and runtime `-O`/`-OO` also refuse. For valid
unoptimized requests, the pre-existing paired-interlock sequence and private
no-spawn receipt behavior remain; the frozen r4 fixture still proves a
missing checkout or early log-open failure yields a no-spawn receipt.

## Offline test record and historical r3 failures

| Check | Result |
| --- | --- |
| New `test_fresh_start_launch_nonassert.py`, normal Python | 10/10 PASS |
| Same new test file under `python3 -O` | 10/10 PASS |
| Same new test file under `PYTHONOPTIMIZE=1` | 10/10 PASS |
| Six frozen r4 monitor/interlock/early-refusal suites | 72/72 PASS |
| Independent adversarial fake-contact and r3-outcome tests | 7/7 PASS, including both HOLD counterexamples |
| Historical `test_r3_adversarial_verifier.py` on e74 | 3/3 FAIL because it asserts the old unsafe behavior; investigated below |

The three r3 tests are negative-history assertions, not current acceptance
tests. Their first two expect source drift after pidfd binding to prevent a
low-disk stop. On e74, both base and paired synthetic monitors returned from
`tick()` in `STOPPING` with the authenticated guard pidfd TERM recorded,
so the expected exceptions were not raised. The third expects an
unresponsive authenticated guard to outlive a TERM timeout with no
escalation. The current synthetic `run()` returned `3` in explicit `HOLD`
after TERM plus pidfd KILL to the authenticated guard and child. These
outcomes are also asserted in the frozen r4 implementation suite; the
monitor source blobs are unchanged from the independently reviewed 446b
repair. Thus the three red historical tests do not indicate a new monitor
regression, while the two launcher counterexamples above still block PASS.

**Disposition:** keep r5/r6 packages disabled and trn22 native dispatch on
HOLD. A separate Implementer should close both source-gate counterexamples,
then obtain a fresh independent source review. Any later capsule, request,
pair bindings, G2 identity disposition, host preflight, Operator transfer,
and one-shot native assignment must be exact to that new source; none is
approved by this receipt.
