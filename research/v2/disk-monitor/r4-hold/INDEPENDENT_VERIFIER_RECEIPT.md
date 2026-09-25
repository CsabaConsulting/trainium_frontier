# Independent Verifier receipt: trn22 c5c A0 r4 HOLD-only package

**PASS for static consistency of the frozen non-dispatchable package; all host
transfer, G2, monitor, and native-run gates remain HOLD.** This review binds
Implementer package commit `3d65cbe00f6cf34725b681427597f282c7df6646`
to frozen source commit `446b657695f086f9b484fa601c9a3b61c2495912`.
I reviewed the six new package files in an isolated checkout. I did not
implement this package, touch either host or AWS, stage source, operate a
monitor, run training/evaluation, or submit. This receipt is not a Supervisor
authorization or an Operator instruction.

## Immutable package identities

| New file under `research/v2/disk-monitor/r4-hold/` | SHA-256 |
| --- | --- |
| `GATE_CONTRACT.json` | `b79ab8e1ddbceac7aa9dac55b2d6c4fa275cbf432cf84f61d02ce9aaa7bade84` |
| `RUNBOOK.md` | `3afa613d97d566c96334496cf9889e2c47fe4428ef4fbd0b4c5501e1d33eba5c` |
| `fs-20260925-v2-c5c-a0-control-trn22-r4.request.DISABLED.json` | `ece18831f81ff8460f923165bb758c6aa02435ffe01e08398ccda891dd8baac0` |
| `fs-20260925-v2-c5c-a0-control-trn22-r4.base-binding.STATIC-ONLY.json` | `f467197d2f85ecffaf06af56bef171a999491c5ef91896435f6748c6889e59be` |
| `fs-20260925-v2-c5c-a0-control-trn22-r4.pair-binding.STATIC-ONLY.json` | `617c3c66770d15047a1133f48404fdf15cb454e2d2bb30ab713b1356831f4474` |
| `test_hold_package.py` | `78e84fd474cb51b5a07f518b53f1580a9493f1aa113576e30d879a7ca7b89bd7` |

The package commit has parent exactly `446b6576`, and its diff adds only
these six files. `git diff --check 446b6576..3d65cbe0` passed. The contract's
request/base/pair SHA chain matches their bytes. Both bindings name the same
new r4 label, checkout, host `ip-172-31-32-253`, UID `1000`, and frozen
source. The base points to the exact disabled-request hash and exact base
monitor SHA; the pair points to the exact base-binding hash and paired
wrapper SHA. All eleven source file hashes in the request, base binding, and
contract match the actual files at frozen commit `446b6576`. Important
examples: scored `train.py`
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
`prepare.py`
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
launcher `9488efa918119439a33cbe68cb5638bd1af3b15f1f8f61ec790a216e2bc1e3f4`,
guard `df902becb2faf009d04d7ecfaee3791df67e822e90ec6112adbc39be81599ecc`,
host `b34c368c12175615514e63daebc296ec09332ef801e176e0d19cc3edc045fed3`,
base monitor `91dae71468d5d029857371ccd2c960769b71874e855bb058130e738b91bb2c60`,
and paired wrapper `9e2d6ed26d918131c3e12b9937093e465b39aa895a9bf2be7339de8ae5094344`.

## Science and safety boundary

The draft retains the r3 science command exactly after r3-to-r4 label/output
path substitution: seed 42, four ranks/LNC2, c5c dense 1024 with matrix LR
0.015, 1,800 training seconds, 10-GiB prelaunch root free, 6.5-GiB
operational floor, and the floor-plus-2-GiB monitor stop threshold
`9,126,805,504` bytes. The six warm-cache paths, environment map, boundary
observer script hash, and observer command likewise match r3 after the label
change. A normalized JSON diff against r3 showed only frozen commit/source
hash refreshes and draft `purpose`/`state` text beyond those label changes;
there is no new training flag or data/evaluation path. The package does not
change scored source or protected assets.

The request is explicitly `dispatch_authorized: false`, both bindings are
`STATIC_ONLY`, all `required_gates` are false, and no one-shot authorization
or source-transfer artifact is supplied. The launcher rejects this exact
disabled draft at its top-level authorization assertion, before any SSH or
local dispatch receipt. A local refusal check returned exit 1 with
`one-label dispatch is not authorized`; the r4 local dispatch path remained
absent. The paired monitors separately require an exact one-shot authorization
with `authorized`, `paired_authorized`, `dispatch_authorized`, and
`one_attempt_only` all true, plus the new request/base/pair hashes; they
cannot be armed from this package alone. The old r3 hashes and source
commit `e4272b4d...` are recorded as stale, not reused.

The runbook's proposed source transfer is a future, separate workflow:
read-only host feasibility, an immutable byte-size/SHA/member-manifest source
artifact for exactly `446b6576`, independent review, a transfer-only
Supervisor assignment, an absent private target with no overwrite/deletion,
fresh-session byte readback, then a separately assigned clean checkout and
independent review. Its proposed `.request.pending.json` and `.base-binding.json`
host paths are *not* the names of the disabled/static draft files; any later
staging or re-materialization needs an exact mapping and fresh hash-bound
review. Flipping `dispatch_authorized` changes the request SHA and therefore
requires newly regenerated and reviewed base/pair bindings and new one-shot
authority. The frozen package itself grants none of these steps.

## Local checks

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/v2/disk-monitor/r4-hold/test_hold_package.py -v
# 6 tests; OK
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/test_fresh_start_monitor_interlock.py research/v2/disk-monitor/test_paired_root_free_guard_monitor.py research/v2/disk-monitor/test_root_free_guard_monitor.py research/v2/disk-monitor/test_r4_interlock_implementation.py research/v2/disk-monitor/test_r4_host_cleanup_implementation.py research/v2/disk-monitor/test_r4_early_refusal_implementation.py -q
# 72 synthetic tests; OK
PYTHONDONTWRITEBYTECODE=1 timeout 5s python3 research/fresh_start_launch.py research/v2/disk-monitor/r4-hold/fs-20260925-v2-c5c-a0-control-trn22-r4.request.DISABLED.json
# exit 1; AssertionError: one-label dispatch is not authorized
```

The source-safety review at `R4_446B6576_INDEPENDENT_VERIFIER_RECEIPT.md`
was a separate local PASS, archived on canonical branch commit
`c7775e0d5b9c75f301d6762433f121901f968940` with receipt SHA-256
`2a5f2b0c51a852a154f7fef44de092ddbbaaf45cc72aa36ca4fd90921f6b17b7`,
and explicitly left native dispatch on HOLD. This
static package PASS does not silently change the draft contract's
`source_safety_pass_claimed: false` or any required-gate flag. Before even
**source transfer**, a fresh exact-source artifact/manifest, separate
transfer/extraction contract and independent review, read-only trn22
feasibility, and explicit Supervisor assignment are still required. CPU G2
source-identity disposition, new independently reviewed dispatchable
request/bindings and one-shot authorization, complete fresh host checks,
live paired monitors plus boundary observer, and an assigned Operator remain
further gates before a native A0 control. No BPB result is established here.
