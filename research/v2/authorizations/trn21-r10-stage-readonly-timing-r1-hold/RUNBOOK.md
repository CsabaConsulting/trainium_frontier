# trn21 r10 stage r3 pre-open timing — ENABLED READ-ONLY REQUEST / DISPATCH HOLD

This is an Implementer-only request package for a future feasibility probe,
**not** a source-stage attempt. `READONLY_ENABLEMENT.FINAL.json` requests one
bounded read-only timing contact, but is not dispatch authority by itself.
No trn21 contact, host file write, bundle/checkout creation, one-shot marker,
training, evaluation, AWS action, breaker change or submission occurred.
The original `REQUEST.HOLD.json` remains disabled with every write/launch
authority false. The distinct independent final read-only review and
Supervisor one-attempt assignment are absent; the invocation therefore exits
before SSH or a local result. Neither the source-stage r3 static PASS nor this
request authorizes the stage writer or a native run.

The r2 collector addressed the independent r1 static HOLD in
`research/v2/verification/20260925-trn21-r10-stage-readonly-timing-r1-independent-static-HOLD.md`
(commit `252d5610`; receipt SHA-256
`1ea3a6967691574ebc57966462ec27a1adf0bacf68e7092deafa642f571287b2`).
The independent r2 static HOLD at `ca3578e7` found that the individually
bounded fields still allowed an impossible positive trial: zero seconds of
pre-open time with a 0.4-second label scan and 0.3-second label age. This r3
collector keeps r2's empty-stderr, exact-record, identity, finite-range and
disk/inode checks and additionally enforces remote monotonic chronology:
`preopen_seconds >= label_scan_duration_seconds + label_age_seconds` and
`label_age_seconds >= activity_age_seconds >= space_age_seconds`. A one-
microsecond allowance covers floating-point monotonic subtraction roundoff;
it is negligible relative to the one-second freshness gate. The stage source,
local stage operator, independent stage review, disabled request and remote
timing source pins remain unchanged. The source remained disabled until its
independent static review. That review is now
[STATIC PASS, dispatch HOLD](../../verification/20260925-trn21-r10-stage-readonly-timing-r3-independent-static-PASS.md)
at `adee3b47cbeb8d82b5b716ce6f5aafcd80a4aee3` (receipt SHA-256
`78aab14b15c88431974322ad0bfe6e5ee96f1873dfb53266290c19db71f8de2c`).
The collector's new authority, root and fixed-result gates need a separate
final review of this exact committed request.

## Exact subject

- Independently static-PASSed disabled stage source: commit
  `7205abf5469edc616c4e4b58caee27bc1957bbd4`, reviewed at
  `79bac32c0ce8dbd6e343e27fccf9def272234deb` in
  `research/v2/verification/20260925-trn21-r10-incremental-stage-r3-active-root-independent-static-PASS.md`
  (receipt SHA-256 `ab0a63e47ece6d9055167b62679b40bee949b1d8e1e948195967960a77bf859e`).
- Exact stage remote source SHA-256:
  `47e2c28bd6fd735904398a48d6c91ef95b93454cef9558b121c26b0ee5b66a0b`.
  Exact local r3 operator SHA-256:
  `6ecad35400199b7c1cba41c78d5d4f14363d37180c40e8aab5e9b485846db74f`.
  The collector checks these bytes and both static review receipts before any
  future contact; it never imports or calls the local stage operator.
- The remote timing source separately pins trn21 hostname/UID/device, private
  e65 source, r10 target, bundle, stage label, existing host lock, e65/r10
  commits, 13,177,973,911-byte and 10,000-inode floors, and the one-second
  final freshness gate. The future collector pins the trn21 SSH host key.

## Measurement, not a write

A future separately reviewed capture would send only the pinned stage and
timing source bytes over one strictly host-key-checked SSH stdin. On trn21,
the timing code would acquire the **existing read-only opened** launcher lock,
call the exact r3 full `_probe()` under that lock, then call the exact r3
`_final_no_write_continuity()` at most three times. That helper performs the
actual pre-open sequence: activity, two broad label scans, fresh activity,
root space, direct STOP/target checks and the one-second age gate. The probe
records each helper's elapsed time, scan duration, disk/inode observation and
sample ages at an immediate **clock-only** hypothetical open point. It never
calls `_write_bundle`, `perform`, `os.open(...O_CREAT/O_EXCL)`, or a stage
reconstruction. A failure stops further trials and is HOLD. A successful
timing result is only a dated feasibility observation; scheduler delay and
actual `O_EXCL`/post-open behavior remain unmeasured and still subject to the
r3 writer's live gates.

The local collector checks STOP in active Desktop, legacy Documents and its
own package checkout. The execution-root gate separately checks STOP in the
clean trn21 stage-r3 worktree at
`/home/poojith-devan/Desktop/Trainium-trn21-r10-stage-r3-active-root`, exact
HEAD `7205abf5469edc616c4e4b58caee27bc1957bbd4`, clean Git status,
science `train.py`/`prepare.py` and the three exact stage files. It rejects a
wrong configured path, symlinked root, dirty root, changed HEAD/source or
optimized Python. The collector repeats this gate before SSH and after contact, including
when called directly. It reads breaker and launch HOLD from **only** active
`/home/poojith-devan/Desktop/Trainium/research/v2/state.json` before SSH,
again immediately before contact, and after contact. Any STOP, tripped
global/trn21 breaker, launch-state change or state-hash drift is HOLD. A stale
Documents state cannot enable contact; a Documents STOP can block it. Future
raw output is bounded, credential-screened and hashed. One fixed local result,
`OPERATOR_RAW_TRN21_R10_TIMING_RESULT.json` in the timing package worktree,
is created with `O_EXCL` and a fsynced `HOLD_IN_PROGRESS` record before SSH.
A preexisting or partial result consumes the attempt. Timeout, malformed or
partial output, local-gate drift or root drift is HOLD, never timing PASS.

## Exact future invocation — still HOLD

The proposed Operator command is from the clean trn21 stage-r3 execution
worktree, with the timing package in its separate clean checkout. It is
documented for independent final review, not for execution now:

```text
cd /home/poojith-devan/Desktop/Trainium-trn21-r10-stage-r3-active-root
/usr/bin/python3 -B /home/poojith-devan/Desktop/Trainium-trn21-r10-timing-enabled-r3-r1/research/v2/authorizations/trn21-r10-stage-readonly-timing-r1-hold/invoke_timing_readonly.py
```

The enabled request binds the exact trn21 endpoint, ED25519 fingerprint,
known-hosts bytes, stage source/operator, reviewed timing source, clean local
execution root and c5c science bytes. It permits only this read-only timing
observation: host writes, process signals, a stage marker, training,
evaluation and AWS actions remain false. The independent Verifier must review
this final committed package and provide the exact
`INDEPENDENT_READONLY_ENABLEMENT.PASS.json`; only a Supervisor may then provide
the separate `SUPERVISOR_READONLY_ASSIGNMENT.json`. Both are absent. An
Operator would still need fresh checks and a one-attempt assignment. Timing
alone cannot prove actual `O_EXCL` latency or authorize stage/native work.

## Offline Implementer checks

```text
python3 -B research/v2/authorizations/trn21-r10-stage-readonly-timing-r1-hold/collect_readonly.py --check
python3 -B -m unittest discover -s research/v2/authorizations/trn21-r10-stage-readonly-timing-r1-hold -p test_readonly.py -q
python3 -O -B -m unittest discover -s research/v2/authorizations/trn21-r10-stage-readonly-timing-r1-hold -p test_readonly.py -q
```

`--check` reports `DISABLED_NO_HOST_CONTACT`; 19/19 normal and 19/19 `-O`
offline tests pass. Synthetic tests include the contradictory-chronology
parser and capture witness, exact request/refusal, wrong-root and optimized
Python rejection, and the fixed pre-SSH result. These Implementer checks are
not independent verification. No source-stage or native run is implied by a
timing receipt.

`sha256sum -c SHA256SUMS` is run from this package directory.
