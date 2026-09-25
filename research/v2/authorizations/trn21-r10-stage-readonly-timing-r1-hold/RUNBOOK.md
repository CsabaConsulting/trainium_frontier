# trn21 r10 stage r3 pre-open timing — DISABLED / READ-ONLY

This is an Implementer-only feasibility probe, **not** a source-stage attempt.
No trn21 contact, host file write, bundle/checkout creation, one-shot marker,
training, evaluation, AWS action, breaker change or submission occurred.
`REQUEST.HOLD.json` has host contact and every write/launch authority false.
The three distinct future enablement, independent review and Supervisor
assignment files are absent; `collect_readonly.py --capture` therefore exits
before SSH. The previous source-stage r3 static PASS does **not** enable this
probe or authorize the stage writer.

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
  The collector checks these bytes and the review receipt before any future
  contact; it never imports or calls the local stage operator.
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
own package checkout. It reads breaker and launch HOLD from **only** active
`/home/poojith-devan/Desktop/Trainium/research/v2/state.json` before SSH,
again immediately before contact, and after contact. Any STOP, tripped
global/trn21 breaker, launch-state change or state-hash drift is HOLD. A stale
Documents state cannot enable contact; a Documents STOP can block it. Future
raw output is bounded, credential-screened, hashed and saved only in a new
exclusive local `OPERATOR_RAW_TRN21_R10_TIMING_<UTC>.json` receipt. A timeout
or malformed/partial output is HOLD, not a timing PASS.

## Offline Implementer checks

```text
python3 -B research/v2/authorizations/trn21-r10-stage-readonly-timing-r1-hold/collect_readonly.py --check
python3 -B -m unittest discover -s research/v2/authorizations/trn21-r10-stage-readonly-timing-r1-hold -p test_readonly.py -q
python3 -O -B -m unittest discover -s research/v2/authorizations/trn21-r10-stage-readonly-timing-r1-hold -p test_readonly.py -q
```

`--check` reports `DISABLED_NO_HOST_CONTACT`; 5/5 normal and 5/5 `-O`
offline tests pass. These checks are not independent verification. A separate
Verifier must review the exact new package before anyone prepares the absent
read-only enablement, and an Operator may only act after a separate Supervisor
assignment. No source-stage or native run is implied by a timing receipt.
