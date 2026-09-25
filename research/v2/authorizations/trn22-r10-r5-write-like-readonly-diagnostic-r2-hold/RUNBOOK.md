# trn22 r10 r5 write-like diagnostic r2 — disabled, read-only

**HOLD. Do not invoke the Operator or contact trn22 from this package yet.**
`REQUEST.DISABLED.json` has `dispatch_authorized=false`. The only path
accepted by `operator_readonly.py` is the distinct, currently absent
`REQUEST.ENABLED_AFTER_REVIEW.json`, bound to a distinct independent final
read-only PASS and Supervisor read-only assignment. This Implementer created
none of those authorities and performed no SSH, native run, evaluation,
cleanup, marker reset, or AWS action.

This r2 package is a narrow repair of the independent static HOLD at
`d8e70c2b848c88573d09432953d73b3090bb6d0a`. The r1 package at
`5b60d9ec3590f20ea4d4cc29dfecd57062e24124` is retained unchanged as
historical evidence. Its only blocking defect was that it checked the
remote GNU timeout *inside* a process already started by that timeout.
The r2 Operator restores the reviewed r5 stage operator's separate
read-only identity-probe SSH **before** each timed phase. This new request
uses a distinct label, source hash and authority schema; no r1 authority
could enable it.

## Why this diagnostic exists

The r5 source-stage one-shot was consumed without an accepted stage. The
independent result HOLD at `b1387247faf1451dbfd04aadf72a821658b92063`
binds the archived one-attempt journal SHA-256
`229a331b1aa8b302c607933273a7b09d6b35df9d3892e6ddbad41fb894bb1da8`.
It records a generic write-phase SSH failure; preflight and distinct-process
postflight found the original e65 source and protected checkpoint/caches
intact, with both remote bundle and target absent. The r5 wrapper grouped
nonzero SSH return code, any stderr, and oversized stdout into that generic
error and did not archive raw stream metadata. The historical cause is
**unknown**. A previous 20/20 read-only timing probe had a worst 19.642 ms
margin under the one-second scan gate, but it did not exercise the write
phase and does not prove timing caused the failure.

This diagnostic samples the same pre-open checks with a **new label**
`fs-20260925-trn22-c5c-a0-r10-readonly-write-like-diagnostic-r2`. It does
not reuse, remove, or reset the consumed r5 marker. The r5 `remote_incremental.py`
still scans its original stage label because the point is to model that
exact path; the new diagnostic label exists only in the request and receipt,
and creates no host file.

## Exact no-write sequence

Each primary phase is preceded by a **separate pinned read-only SSH** of
`/usr/bin/env -i LC_ALL=C /usr/bin/timeout --version`, bounded to 25 seconds.
The Operator rejects nonzero return, any stderr, stdout above 4096 bytes,
or a malformed/non-GNU version line **before** starting the timed phase.
Active Desktop/former/package STOP and breakers are checked before that
probe, after it, and immediately before the phase SSH. Each SSH uses the
same pinned trn22 endpoint, known-hosts digest and ED25519 fingerprint and
key metadata check. The phase then uses the fixed remote 320-second
absolute timeout and local 350-second wait. The older in-child timeout
check remains defense-in-depth, but is no longer the first authentication
of the binary. The strict stage-source one-second final-check bound is not
relaxed. There are up to three phase SSH sessions and three preceding
read-only identity-probe SSH sessions, not three total SSH sessions.

1. After its separate timeout-identity probe, fresh phase SSH process:
   call the **unchanged** r5
   `remote_incremental.perform("preflight")`. Keep the exact snapshot in
   memory to bind the next process; the published receipt contains only a
   snapshot hash and finite whitelisted fields.
2. After its own separate timeout-identity probe, distinct phase SSH process:
   read the exact 109,800-byte pinned bundle from
   stdin **without writing it**. Under the existing `_host_lock` flock,
   run `_probe(PRE_FREE)`, `_stable(preflight, probe)`, target/bundle absence,
   bundle length/SHA-256 validation, and an
   `O_RDONLY|O_DIRECTORY|O_NOFOLLOW` open of `/home/ubuntu`. While holding
   that read-only directory fd and the flock, call the unchanged
   `_final_no_write_continuity`, take a monotonic would-open sample, and
   call `_freshness_evidence`. There is **no** `O_CREAT`, `O_EXCL`, payload
   write, `git init`, `git fetch`, checkout, rename, or unlink. The remote
   receipt gives each passed gate, current failure gate, finite gate timing,
   finite failure class, and the one-second freshness ages. A refusal never
   becomes a pass.
3. Following another separate timeout-identity probe, a third phase SSH
   process attempts the unchanged
   `remote_incremental.perform("postflight")` after every dispatched
   preflight/second-phase outcome. Local STOP/breaker and identity checks
   still precede this phase, so a new STOP can block even postflight.

Every phase records return code, stdout/stderr byte counts and SHA-256 only.
No raw SSH stream, exception text, command line, key, credential, or host
environment is emitted by the Operator. `READONLY_DIAGNOSTIC_COMPLETE`
would mean these *current* no-write checks passed; it would **not** diagnose
the historical r5 write failure, prove an actual `O_EXCL` succeeds, create
an r10 stage, authorize a native run/eval, or permit cleanup. A failed gate
is a HOLD, not a retry authorization.

## Source and authority pins

| Item | Exact pin |
| --- | --- |
| r5 reviewed/assigned source | `ed2ca7fc43d29f6909c001ee97e2875ef8ae4ee7` |
| independent consumed-stage result | `b1387247faf1451dbfd04aadf72a821658b92063` |
| independent consumed-stage receipt | `58461a36b0f6c0a9901be40139cf3da75d65f5c08727f55f16e2fd1ad20d9dc6` |
| independent r1 static HOLD | `d8e70c2b848c88573d09432953d73b3090bb6d0a`; receipt `af64f1bd289ad757f455ce927251944637efecdaf3195a0343bfe219f61e9968` |
| unchanged r5 remote source | `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f` |
| unchanged incremental bundle | 109,800 bytes, `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e` |
| original e65 source / intended r10 science | `e65a3097a5b00f6178188a935bb2acb9a5042c6c` / `22bfbab5a77d6feb670d2d737504f8b70b08ddc1` |
| root `train.py` / `prepare.py` bytes | `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` / `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` |
| new read-only remote source | `e9cd37be06d644d99b6be1cc4ee73541a66d028998fd172fe03813fb51ac4ada` |
| new disabled Operator source | `c1c740c22d6127b032b10763c5e5c3202abb199d99be5c0cdc3c846d663e6e18` |
| canonical disabled request | `9a7bfa79a5d8ab18fcfe8e50b99023476c46670f768231abfebf24ad02090e92` |

The package references the **existing** r5 source and bundle by relative
path; it does not edit or duplicate them. It never edits `train.py` or
`prepare.py`. A clean local science checkout is required before an enabled
diagnostic could dispatch. No same-HEAD native-training claim is made here.

## Independent steps still required

1. A separate Verifier reviews the exact new Operator/remote/request bytes,
   the separate pinned GNU-timeout probe before every timed phase,
   pre/post-probe and immediate pre-phase STOP checks, cross-process
   snapshot binding, all whitelisted output fields and normal plus `-O`
   adversarial tests. It must explicitly check that the
   generated `write_like` branch cannot invoke stage `perform("write")`.
2. Only after that review may an authorized actor create a canonical
   `REQUEST.ENABLED_AFTER_REVIEW.json` with exactly the fields in
   `operator_readonly._authority` (the disabled request differs by schema
   and `dispatch_authorized`). The independent final review must bind its
   SHA-256 and source hashes. A separate Supervisor assignment must bind
   that review. Neither file can be inferred from the old r5 stage PASS or
   assignment; those governed a consumed, materially different attempt.
3. Before any possible Operator command, recheck the clean worktree,
   science/source/bundle hashes, Desktop/former/package STOPs and breakers,
   the retained r5 marker hash, SSH identity, host lock, source/protected
   assets, process/Neuron quiescence, target/bundle absence, and the pinned
   byte/inode floors. A failed preflight is HOLD. No cleanup or marker
   change is authorized.

The **proposed command, not to be run until those gates pass**, is:

```text
/usr/bin/python3 -B /home/poojith-devan/Documents/ChatGPT/Trainium-trn22-r10-write-like-diagnostic-r2-disabled/research/v2/authorizations/trn22-r10-r5-write-like-readonly-diagnostic-r2-hold/operator_readonly.py
```

Offline Implementer checks (safe; no host contact):

```text
cd /home/poojith-devan/Documents/ChatGPT/Trainium-trn22-r10-write-like-diagnostic-r2-disabled/research/v2/authorizations/trn22-r10-r5-write-like-readonly-diagnostic-r2-hold
sha256sum -c SHA256SUMS
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s . -p 'test_*.py' -q
PYTHONDONTWRITEBYTECODE=1 python3 -O -B -m unittest discover -s . -p 'test_*.py' -q
```

The source package is a diagnostic proposal only. No result or improved BPB
is claimed.
