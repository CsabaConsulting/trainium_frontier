# Independent static HOLD — trn21 r10 read-only timing r1 (disabled)

Verdict: **HOLD for this exact disabled package; do not enable host
contact.** Subject `abb4ef4d0ee048aedc499fe5406190d65a2cfb62` descends
from independently static-PASSed stage r3
`7205abf5469edc616c4e4b58caee27bc1957bbd4` and its independent
receipt `79bac32c0ce8dbd6e343e27fccf9def272234deb`. No SSH/EC2/host
contact, marker, bundle or checkout creation, host write, training,
evaluation, breaker reset, cleanup, AWS action or submission occurred in
this review.

Only six files under
`research/v2/authorizations/trn21-r10-stage-readonly-timing-r1-hold/`
were added/changed by this commit: the disabled request, runbook, manifest,
collector, remote timing source and test. No `train.py`, `prepare.py`, data,
evaluator, stage writer or scorer was changed. `sha256sum -c SHA256SUMS`
passed **5/5**; the collector's `--check` also verified these external
exact-byte pins:

| Pin | Verified SHA-256 |
| --- | --- |
| stage r3 `remote_incremental.py` from `7205abf5` | `47e2c28bd6fd735904398a48d6c91ef95b93454cef9558b121c26b0ee5b66a0b` |
| stage r3 `operator_once.py` from `7205abf5` | `6ecad35400199b7c1cba41c78d5d4f14363d37180c40e8aab5e9b485846db74f` |
| r3 independent static-PASS receipt from `79bac32c` | `ab0a63e47ece6d9055167b62679b40bee949b1d8e1e948195967960a77bf859e` |
| new `remote_timing.py` | `5a1298febfb67f6a61eb1f6fa02f974c7cbe15c6eb0bc221338b85d892be0db7` |
| `REQUEST.HOLD.json` | `9f2ed81e1c04decc16e0747d6e71f52037da759241ea1c8774dbbf790604d897` |

The timing source's 13 hostname, UID, device, source, target, bundle,
stage label, existing lock, base/target commit, free-byte/inode floor and
one-second freshness constants match the pinned r3 stage source exactly.
The collector pins the trn21 SSH hostname and ED25519 fingerprint;
its local-only known-hosts/identity metadata check passed with known-hosts
SHA-256 `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`
and fingerprint `SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc`.
It did not open a network connection.

The checked-in request denies host contact, host write, marker, training,
evaluation and AWS authority. The three exact future enablement/review/
assignment files are absent. `--check` returns
`DISABLED_NO_HOST_CONTACT`; direct `--capture` and default invocation
return nonzero `HOLD` before SSH. The bootstrap executes the pinned stage
source with a non-`__main__` name, then calls only the pinned timing `run()`.
That function acquires the existing lock through the stage's read-only-opened
nonblocking lock helper, calls the full read-only `_probe()`, and makes at
most three calls to the exact `_final_no_write_continuity()` and
`_freshness_evidence()` helpers. It records an immediate **clock-only**
hypothetical open point. It never calls `perform`, `_write_bundle`,
`_reconstruct`, or `O_CREAT`/`O_EXCL` on the host; no one-shot marker is
created. A failed helper stops the remaining trials and records HOLD.

The collector checks STOP at active Desktop, former Documents and its
package root. It reads breaker and launch HOLD only from active Desktop
state. There is one future SSH phase: it checks state before identity work,
immediately before that SSH call, and after return; a changed state hash or
STOP makes the result HOLD. In separate offline adversarial tests, pre-SSH
state drift prevented the mocked SSH call, post-SSH drift archived a HOLD,
and former Documents STOP blocked before controller-state read. Remote
stdout and stderr are byte-bounded in the local receipt; sensitive raw text
is suppressed there. These checks pass, but they do not fix the acceptance
defect below.

Offline tests passed **5/5 normal and 5/5 under `-O`**; `git diff --check`
passed. **Independent adversarial failure:** with a mocked, zero-exit SSH
response and stable controller, the collector accepted ordered start/three
trial/end JSONL with every reported activity, label and space age set to
`9999.0` seconds, a missing start identity and `stderr=b"remote warning"`.
It wrote status `READONLY_TIMING_OBSERVED_NOT_STAGE_AUTHORITY`, rather than
HOLD. `_parse_jsonl()` checks record order and status strings but does not
require the numeric timing/age fields or their finite one-second bounds, the
preflight identity and free-space fields, or empty stderr. `capture()`
ignores nonempty stderr when deciding observed-success status. A malformed
or anomalous timing response can therefore be promoted into a positive
feasibility observation. This is a real collector-result gate failure even
though the pinned remote source normally emits richer records and a later
independent result review is required.

Minimum repair: fail closed on nonempty stderr; validate an exact start
identity/host/source/target/label/lock/commit and disk/inode floor, exactly
three trials with finite nonnegative elapsed/scan values and finite
activity/label/space ages in `[0, 1.0]`, and a matching terminal record.
Add a regression for the adversarial record and rerun normal/`-O` tests.
Submit a new disabled package to a different Verifier. Do not reinterpret
this HOLD as permission for an Operator timing attempt.

The timing probe has **not** run on trn21, so three-trial elapsed times and
actual one-second feasibility remain unknown. A clock-only observation
does not include actual `O_EXCL` latency or the writer's post-open checks;
even a repaired clean observation will not authorize a stage attempt.
Separate read-only enablement, independent final review, Supervisor
assignment and fresh local/host gates must precede any Operator contact.
The stage writer retains its own final review, one-shot and native gates.
