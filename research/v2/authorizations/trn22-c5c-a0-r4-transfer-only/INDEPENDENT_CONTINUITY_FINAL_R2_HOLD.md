# Independent final review: trn22 r4 continuity transfer-only request

**Verdict: HOLD.** Do not create the fixed-path
`INDEPENDENT_CONTINUITY_FINAL_R2_REVIEW.json` PASS file, issue a Supervisor
assignment, or dispatch this transfer. This is a review of the exact new
request/wrapper at Implementer commit
`60592d24970c536f81269934a309288acf91fefa` on
`codex/20260925-trn22-r4-continuity-final-request`, based on the separately
reviewed disabled repair `d682a90b1e75bdac128938feda884484ea87a8d5`.
I did not implement the candidate or operate trn22. I read the repository
contract in the required order and inspected the actual new files, the
repaired disabled core/transport/remote, source-stage contract, and earlier
independent static PASS and final HOLD receipts. No SSH, host/AWS contact,
transfer, checkout, G2, training, evaluation, cleanup, key-content read,
breaker reset, or submission occurred in this review.

## Exact subject and local checks

| Artifact | SHA-256 |
| --- | --- |
| New `TRANSFER_REQUEST.CONTINUITY_FINAL_R2.json` | `99f46c558c49ba4a404daf8908e387b7bc31deaff8ab72e26142596efddaafd5` |
| New `operator_transfer_continuity_final_r2.py` | `14dac3c538d3d08e2876c1327e177f2a4c4696cfadb9884004725305383ccf23` |
| Disabled continuity static review | `669d99837e25ee634e4db418125b1b8bf13862a1907baad10ac213b5e89c7c57` |
| Reused disabled core / SSH transport / remote writer | `7590e275f94f79ffac9098bfd3c0345d4c9cb5ce10a26624065b0e04d6189736` / `ea14fd19f6992400ad56e59c8dcbec620c784f616c4ff60f6d0568aa721a4ec0` / `023e578693c3fb826b81cfc645190d79e409757dba0bd8df29adad6e3f90a69d` |
| Independent local adversarial tests | `23357bc111565f7da7204546cc7e441c9ff2bacda977fc9888018aee045f8ab3` |

The frozen outside-Git capsule independently rehashed to
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`
at 104,154,875 bytes, mode `0600`, one link. The private key was checked by
metadata only. The request/code/source-stage hashes match their pinned files.
The actual final PASS and Supervisor assignment files, fixed marker and fixed
journal were absent before this review. The request's
`dispatch_authorized: true` is intent only and cannot reach SSH without the
two actual fixed-path authority files.

The four maintained and prior independent offline suites passed **45/45**.
Their tests and my source review cover actual-file review-to-assignment hash
binding, local STOP/breakers, fixed `O_EXCL` marker/journal, pinned ED25519
SSH identity, preflight boot/mount/parent/source binding, STOP/process gates,
old-path and protected-asset checks, distinct readback, partial preservation,
and a separate full read-only postflight after an attempted write failure.
My additional offline suite passed **4/6** and exposed the two failures below.
Its passing cases independently exercised missing/tampered actual authority
files before marker/SSH, marker/journal mode and `fsync` plus replay refusal,
the literal pinned SSH argv, and preflight source/mount/old-path/protected-cache
mutations. All synthetic remote writes were confined to temporary local
directories. `git diff --check` passed.

## Blocking findings

1. **Late Neuron activity can precede target creation.**
   `stage_r4_remote.write()` performs `_live_prewrite_safety()` once, including
   `fuser /dev/neuron0` and `neuron-ls`, at lines 575 and 535–563. It then
   rechecks identity and runs a final STOP/process-only check at lines
   576–585 before `O_EXCL` at line 587. In my deterministic local fixture,
   Neuron becomes busy during that second identity check; the last
   STOP/process check remains clear, and the actual writer creates the exact
   target. The negative test fails with `Neuron became busy before O_EXCL,
   but the exact target was created`. Thus this enabled request does not
   satisfy the required late Neuron/holder gate at the write boundary. The
   writer must recheck both Neuron holder signals after intervening work and
   fail before creation on busy or uncertain results. A later postflight
   cannot undo an unauthorized write.

2. **A file-backed old `.git` aborts the full read-only preflight.**
   `_old_paths()` includes each old checkout's `.git/HEAD`. `_safe_path()`
   opens every ancestor with `O_DIRECTORY|O_NOFOLLOW` but catches only
   `FileNotFoundError`. A normal Git worktree has a regular `.git` file;
   inspecting its nested `HEAD` raises `NotADirectoryError`. My local fixture
   reproduces that exact exception. Because `_old_inventory()` calls
   `_safe_path()` during `probe()`, the full preflight exits before emitting
   its JSON receipt. This failure is safe against a write, but the candidate
   cannot complete the required inventory on a host with such an old
   checkout. The repair must classify this known file-backed form without
   following links or weakening old-path/protected-asset checks.

The failed suite command is
`PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v
research/v2/authorizations/trn22-c5c-a0-r4-transfer-only/independent_continuity_final_r2_negative_tests.py`:
**6 tests, 4 pass, 2 expected safety failures reproduced**. The 45 passing
tests do not override these findings. A separate Implementer must repair the
exact code and request bindings; the changed bytes require a new independent
final review. Fresh host feasibility and a distinct Supervisor one-shot
transfer-only assignment remain later gates. No final PASS file was issued.
