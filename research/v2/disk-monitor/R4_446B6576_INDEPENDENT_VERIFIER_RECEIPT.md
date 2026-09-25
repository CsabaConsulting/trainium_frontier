# Independent Verifier receipt: trn22 c5c A0 r4 source 446b6576

**PASS for the frozen local source repair only; native dispatch remains HOLD.**
I independently reviewed commit `446b657695f086f9b484fa601c9a3b61c2495912`
against its parent `113790835b1c0de237b64c45e9303b36eee49266` in an
isolated clean checkout. This review neither implements nor operates the trn22
run. It uses source inspection and synthetic local tests only: no SSH, AWS,
Neuron device, host staging, training, evaluation, submission, or real process
signal. This PASS cannot be used as a one-label dispatch authorization.

## Frozen source identity

| File | SHA-256 at 446b6576 |
| --- | --- |
| `train.py` | `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` |
| `prepare.py` | `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` |
| `research/fresh_start_guard.py` | `df902becb2faf009d04d7ecfaee3791df67e822e90ec6112adbc39be81599ecc` |
| `research/fresh_start_host.py` | `b34c368c12175615514e63daebc296ec09332ef801e176e0d19cc3edc045fed3` |
| `research/fresh_start_launch.py` | `9488efa918119439a33cbe68cb5638bd1af3b15f1f8f61ec790a216e2bc1e3f4` |
| `research/fresh_start_monitor_interlock.py` | `f2fcecd427ea0c4fd5396799fe845d327c2e1384b05e1ef9445b8fb3cf9d2b4d` |
| base root-free monitor | `91dae71468d5d029857371ccd2c960769b71874e855bb058130e738b91bb2c60` |
| paired root-free monitor | `9e2d6ed26d918131c3e12b9937093e465b39aa895a9bf2be7339de8ae5094344` |

`git diff --check 11379083..446b6576` passed. `git diff --exit-code
e4272b4d..446b6576 -- train.py prepare.py` passed: neither scored training
code nor evaluator changed. I read the active `AGENTS.md`, `README.md`,
`CURRENT_STATE.md`, experiment protocol, decision log, workflow role contract,
r3/r4 runbooks, and the prior independent failure receipts before review.

## Independent reproduction and local tests

The **unchanged** prior independent failure repros were copied byte-for-byte
from the controller branch into this isolated verifier branch, not modified
to make them pass. SHA-256: markerless changed-session orphan test
`6c89b1e456017a09771a68ee8c72b30387de260e8012bb7ec7eaf1ee5d7677e5`;
early-launch no-spawn test
`ca9f79e96a0b3ef4622a6ca04e395c807211f97f085af0a588cfc6255b1fae08`.
The earlier independent marker-spoof test was also copied unchanged,
SHA-256 `5209672111d64773c664862785a6f7fb2697f28f49d103ccf2b38676d868a833`.
These tests import the frozen 446b6576 modules from this checkout.

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/test_fresh_start_monitor_interlock.py research/v2/disk-monitor/test_paired_root_free_guard_monitor.py research/v2/disk-monitor/test_root_free_guard_monitor.py research/v2/disk-monitor/test_r4_interlock_implementation.py research/v2/disk-monitor/test_r4_host_cleanup_implementation.py research/v2/disk-monitor/test_r4_early_refusal_implementation.py -q
# Ran 72 tests in 0.333s; OK
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/v2/disk-monitor/r4_11379083_orphan_verifier_evidence.py research/v2/disk-monitor/r4_11379083_early_launch_verifier_evidence.py -v
# Ran 3 tests in 0.010s; OK
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/v2/disk-monitor/r4_marker_spoof_verifier_evidence.py -v
# Ran 1 test in 0.009s; OK
```

The three unchanged 11379083 repro assertions were red on the previous
frozen source and green here. The two orphan cases now return a non-clean
outcome with a live markerless worker in a changed session at one byte below
the stop threshold. The missing-checkout case writes an exclusive private
newline-terminated no-spawn receipt without reaching `Popen`. The independent
marker-spoof case records no would-be signal to the unrelated process.

## Safety analysis

- **Provenance and no unproven signal:** guard/child binding checks UID, PID
  birth, exact executable/argv, parent at first bind, child session and process
  group, then opens pidfds and revalidates. The monitor TERM/KILL routes use
  pidfds only. A discovered descendant can be KILLed only while the exact
  child remains live and reauthenticated in its kernel session; marker-only,
  changed-session, reused-PID and unreadable identities become HOLD, not
  signal authority. The guard's no-pidfd fallback is limited to its own
  unreaped `Popen` direct child. Static search found no `os.kill` or `killpg`
  route in the host/guard/monitor/launcher production paths.
- **Markerless changed-session/false clean:** the base monitor now inventories
  every live same-UID process born at or after the bound child's kernel birth
  tick as a HOLD-only hint, in addition to marker and session candidates.
  The guard's postflight and next preflight use the corresponding host-side
  birth inventory; a prior state missing the child's birth tick fails. This
  closes the reproduced false-clean path. It is deliberately overinclusive:
  unrelated later same-UID processes can cause a conservative HOLD.
- **Emergency-stop ownership:** on low root free, the bound monitor tries
  authenticated guard pidfd TERM before source/hash revalidation; source
  drift after binding invokes the same scoped stop through `fail_closed`.
  Bounded escalation rechecks pidfd-bound guard/child and live-session worker
  identities before KILL. The paired wrapper's source/pair drift reaches that
  fail-closed path rather than vetoing the already-owned stop.
- **Early no-spawn:** after a valid serialized request is parsed, all remote
  checkout/import/lock/disk/source/label/interlock checks are inside the
  early `try`. Failure before `Popen` attempts the standalone exclusive,
  `0600`, file-plus-parent-fsynced receipt even when the checkout cannot be
  imported. The later preflight/interlock/log-open path uses the imported
  equivalent. `spawn_call_started` is now set *after* log creation and
  immediately before `Popen`; once it is set, an exception is ambiguous HOLD,
  never a false no-spawn claim. The paired rescue accepts only a complete
  same-label/commit/pair-path receipt, and if neither receipt nor authenticated
  guard appears by the deadline it announces HOLD and remains resident.

The preceding no-spawn claim requires the reviewed valid request and usable
private receipt directory. A local launcher rejection before SSH, malformed or
truncated remote stdin, transport failure, or failed receipt write provides no
remote no-spawn proof; these cases must remain **HOLD**, not be inferred safe.
Two simultaneous monitor failures and privileged/non-same-UID descendants
also remain outside this user-space monitor's proven containment envelope.

## Package and launch gate: still HOLD

The checked-in r3 request SHA-256
`9202999d0ffc315e9d9c7ef2d4030a38fc2ca1fbe92ec37888c47a49cebea575`
has `dispatch_authorized=false` and pins source `e4272b4d...`, not 446b6576.
Its base binding SHA-256
`44b2395470ed17d6569328c6293fb952601904f918af265a86142d46ec767eb5`
and pair binding SHA-256
`cf35131c0daa829e56a68be756805deefa476b6648152a0857656f9841196817`
also pin old monitor/wrapper hashes. I found no exact 446b6576 one-label
request, refreshed binding, independent package review, source-matched G2
disposition, live host preflight, or one-shot native authorization in this
frozen source. Do not stage or dispatch merely because the **source repair**
passed. Those are separate Supervisor/Operator/Verifier gates.
