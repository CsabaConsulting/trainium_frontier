# Independent common-host idle-guard review — HOLD

Reviewed commit `bdfa06a359fd91c5e559584ee18eb04eb0d87818` without modifying its
guard, drivers or bindings. This is control-plane review, not a model gate.
No hosts, devices, tensors, model sessions or external services were used.

## Blocking: lock descriptor is dropped at the child boundary

Both changed drivers authenticate their inherited common lock, then call
`subprocess.call(COMMAND, ...)` without `pass_fds`:

- `research/v2/operator/20260926-ct-architecture-startup-only/host_driver.py:42`
- `research/v2/operator/20260926-trn21-profile-recursive6ef7/host_driver.py:47`

Python's default descriptor closing means the timeout/job child does not retain
the authenticated lock description. The outer util-linux flock and driver hold
it through normal synchronous execution, but if the driver dies, the outer flock
exits and releases the lock even while the native child can remain alive.

My independently authored real-process fixture used the exact outer argv shape,
the candidate's real `inherited_lock`, and the drivers' same synchronous
`subprocess.call` boundary. With a harmless local sleeping child:

1. While the driver waited, a separate nonblocking lock contender returned 1.
2. I killed only the fixture driver, leaving its child alive.
3. The outer flock exited; a new contender returned 0 while that child was still
   alive. The test then terminated only its own known child.

This reproduced identically under normal Python and `-O`. It is a demonstrated
lifetime gap, not an observed native orphan or simultaneous host launch. Fresh
idle checks may catch a surviving native job, but do not satisfy the separate
requirement to retain the lock through the complete child lifetime.

**Narrow required revision:** carry the already-authenticated descriptor into
the immediate timeout/job child, and verify with a driver-death contention test
that the lock remains held while that child lives. Do not release/reacquire the
lock, change its path/inode, or weaken containment. Any additional descendant
boundary used by the actual binding must preserve or otherwise prove the claimed
lifetime; this review does not certify escaped descendants merely from parent
exit. Re-review the changed guard/driver hashes before launch.

## Passing observations

The common path is exact and existing; the guard rejects symlinks/nonregular or
wrong-owner files, authenticates the exact immediate flock-parent argv, matches
the inherited descriptor by device/inode, and requires existing exclusive FLOCK
state before checking idle. Reasserting that same open-file-description lock
creates no release/reacquisition gap. Direct unwrapped invocation was refused.

Fresh `ps` and `neuron-ls` calls occur after authentication. Successful clean
fixtures exercised both calls. Busy torchrun, another identically named driver,
malformed Neuron JSON, unknown device shape, invalid device PID and command
stderr were refused. Self exclusions are only the authenticated driver and its
immediate parent, not an arbitrary ancestry or name ignore list. No successful
launch was inferred from Operator PASS labels.

The guard records refusal/success in a `finally` block. Its hashes of command
outputs are not raw-output preservation; the Operator's outer-lock-contention
receipt remains a separate operational responsibility. Neither disabled packet
is assigned launch authority by this review. There is no attached eval in these
two diagnostic bindings; their sequential waits provide normal-lifetime coverage
but do not cure the reproduced driver-death gap.

## Executed independent evidence

`test_guard_independent.py` SHA256:
`d94fccb50e2548b8816e86d95badc407c96dc3d4cdcf97f2491026151ef0baf9`.

- Own 10 stdlib tests, normal: exit 0, 0.314 s.
- Own same 10 tests, `-O`: exit 0, 0.305 s.
- Operator's 11 supplied tests independently executed: exit 0, 0.115 s.

The own test suite deliberately expects and exposes the lifetime defect; its
unittest `OK` means the reproduction was successful, **not guard acceptance**.
The source and exact observed return codes are retained in `RESULT.json` and
the runnable independent test. All real child commands were local harmless
fixtures using private temporary directories; no host path was opened.

Exact reviewed support SHA256:

| File | SHA256 |
| --- | --- |
| `idle_guard.py` | `5d735b4baf1b68bab2e9528adc4b1d09a930cb2beafdd069f1a6a513f65c34fb` |
| CT architecture driver | `4be8925a065e537de745447810c4596b0a7396d900d95589e8eb4c6e96700919` |
| P driver | `206f4e34d2397dfe7fa42e540e1905d78408cdc49d5b1baa5ade5a5a871d38f9` |

Disposition: **HOLD for the full-child-lifetime requirement**. No candidate edits,
host launches, threshold changes or new launch framework were performed.
