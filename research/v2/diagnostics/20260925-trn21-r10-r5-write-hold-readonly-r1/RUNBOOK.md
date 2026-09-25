# trn21 r10 r5 write-HOLD diagnostic — disabled, read-only source

This package is an **Implementer-only, offline candidate**. It does not authorize or
perform SSH, source staging, native training, evaluation, cleanup, marker reset,
or AWS action. The checked-in `REQUEST.HOLD.json` has both dispatch and
read-only-diagnostic authority set to `false`; the enabled request, independent
final review, and Supervisor assignment are deliberately absent. Invoking
`operator_readonly.py` exits `HOLD` before host contact. Do not substitute an
older source-stage assignment or run this manually.

## Why this exists

The consumed trn21 r5 stage (`1b1f33bb`, independently held at `97ac177b`)
and trn22 r5 stage both passed full preflight, then returned the same generic
`SSH phase failed or emitted stderr: write`. Neither archived remote stderr or
return code. Distinct-process postflights found no new bundle or checkout and
intact e65/protected assets. The underlying causes remain **unknown**. This
probe tests whether the current trn21 pre-open path can pass without any
exclusive open; it cannot reconstruct the discarded error or prove a future
write will succeed.

## Exact frozen references

| Subject | SHA-256 / identity |
|---|---|
| Archived r5 local Operator | `f48a35c460498a81a56cc73f68f5130a4d91f319a88be7c0ad6a1668f80de70b` |
| Archived r5 remote source | `6f0da61a76595a6fea18662d07d7311c504ab3a4b641ac972c440438bda08bf6` |
| Exact 109,800-byte Git bundle | `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e` |
| Consumed r5 journal copy | `54c65164271d895392c2e170b47dbdf07f62e0aab2cfa378e529afdbbeca3a4c` |
| Independent r5 result HOLD | `786f3f26179db7a31adb7b4440460e662427494f7cbfe192c76878803cea14bf` |
| Existing accepted e65 source | `e65a3097a5b00f6178188a935bb2acb9a5042c6c`, host `ip-172-31-34-85` |
| Unstaged r10 target | `22bfbab5a77d6feb670d2d737504f8b70b08ddc1` |
| This remote diagnostic | `dbadee0421c1b4e5fde63976f4526c8083fe276a134a6151c2294330e020ab09` |
| This local diagnostic Operator | `7c2832eccf1b9e989639ebed40a88771df10a5b417343c2f5e312be5d2a00501` |

The diagnostic label is
`fs-20260925-trn21-r10-r5-write-hold-readonly-d1`. It is distinct from the
consumed r5 source-stage label and marker. It never reads or resets that
local marker. The archived Git bundle is sent only as input bytes to validate
the r5 hash/length check; it is never written on the host. No dataset,
validation data, credential file contents, or `.env` is accessed.

## Planned read-only sequence, if separately authorized later

For **each** of three distinct SSH phases, the local helper first checks the
active Desktop controller STOP/breakers, verifies pinned host-key/key-file
**metadata** without reading private-key bytes, repeats local STOP after
identity, performs a *separate read-only pinned SSH* to verify remote
`/usr/bin/timeout --version` is GNU coreutils, then repeats local STOP before
the phase SSH. This is not replaced by a check *inside* the timeout wrapper.

1. `preflight`: invoke only archived r5 `perform("preflight")`, which takes the
   same existing nonblocking host lock and full safe snapshot. It requires
   absent r5 target/bundle and produces the exact prior snapshot.
2. `preopen`: in a **different SSH process**, repeat the archived r5 full
   `_probe`, `_stable` distinct-session and age binding, input bundle hash,
   target absence, an `O_RDONLY|O_DIRECTORY|O_NOFOLLOW` root open, and the
   original `_final_no_write_continuity` with its **unchanged ≤1-second**
   final label/activity/space freshness gates. Stop at a `would_open` sample;
   **never call the archived `_write_bundle`, `os.open` with create/write
   flags, `_reconstruct`, or Git fetch**. Timed subchecks and failures emit
   bounded enum/size/return-code evidence without raw stderr or exception
   text. A PASS means only that this *read-only* sample passed.
3. `postflight`: in another process, attempt the archived r5 full read-only
   postflight even if an earlier diagnostic phase holds. Both target and
   bundle must remain absent for a clean result.

The probe inherits the existing full source/Git/host/mount/lock/ACL,
checkpoint/cache, STOP, label, process/Neuron, root byte/inode and
distinct-process checks. It does not weaken the original safety gate or
convert its source-only PASS into native authority. A later enabled request
must bind the exact local/remote code hashes, archived consumed evidence and
new label; an independent Verifier must review it, then the Supervisor must
issue a separate read-only assignment. A designated Operator needs fresh
local and live host preflight. None exists now.

Offline Implementer checks (not independent verification):

`python3 -B -m unittest discover -s research/v2/diagnostics/20260925-trn21-r10-r5-write-hold-readonly-r1 -p 'test_*.py' -q`

`python3 -O -B -m unittest discover -s research/v2/diagnostics/20260925-trn21-r10-r5-write-hold-readonly-r1 -p 'test_*.py' -q`
