# Independent trn22 e65→r10 incremental source-stage r1 review — static HOLD

Reviewed 2026-09-25 09:55 UTC by a Verifier separate from the Implementer.
Exact disabled subject: `3d26e963df349f444cd560114400788bf572b062`,
on `origin/codex/20260925-trn22-r10-incremental-stage-disabled`; its parent is
the previously static-reviewed trn22 r4 package at
`19fbfdc88763b6b5762e219339dcd156e7977a30`.

**Verdict: HOLD for this exact disabled package.** The package incorrectly pins
the SHA-256 of its claimed r4 independent static PASS receipt. The constant
`R4_STATIC_PASS_SHA256` in `operator_once.py:50`, the matching field in
`REQUEST.HOLD.json:35`, and `RUNBOOK.md:19` are all
`5f1b18e45fda3e726ccde6063523ee2679f3a8dbc42480cc1e84936d1b1d`
(only 60 hexadecimal characters). The actual bytes of
`research/v2/verification/20260925-trn22-r9-incremental-stage-r4-independent-static-PASS.md`
at claimed review commit `efd5f2396b8bb2c37865d3cb751f45ed2c07bb81`
hash to
`5f1b18e45fda3e726ccde6063523ee2679f3a8dbc42480cc1e84936d1b78bb1d`
(64 characters). The supplied tests compare mutually consistent constants,
not those constants against the claimed prior receipt's actual bytes. Thus
they do not detect the false evidence pin. A later enabled request bound to
the current value would misidentify its supporting independent review.
Correct these three references, refresh `SHA256SUMS`, add an actual-byte
regression assertion, and seek a new independent review of the distinct
repair. No host action may use this package.

## Independently checked boundaries

- The checked-in direct bundle is 109,800 bytes with SHA-256
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
  `git bundle verify` reports exactly one prerequisite,
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c`, and HEAD
  `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`; the delta is 15 commits.
  Both commits keep `train.py` SHA-256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `prepare.py` SHA-256 `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The r10 launcher blob matches the separately source-PASSed
  `1be19bc31d2bdcde8abe9fe90cc1ac884d6efd5f7be197d7bdbfe64c257dcff7`.
- The r4→r10 wrapper/remote diff rebinds the distinct trn22 target, label,
  one-attempt marker and bundle, adds r4 to prior-marker checks, and extends
  the target file-hash inventory to r10. It does not change the r4 no-unlink
  write algorithm. `operator_once.py` has no checked-in enabled request,
  independent final PASS or Supervisor assignment; normal and optimized
  entrypoints both exit `HOLD` before a marker or SSH. The request explicitly
  disables source staging, native training and evaluation.
- Source inspection and synthetic boundary tests show the inherited two
  broad pre-open label scans, subsequent STOP/process/Neuron/root-space
  checks, finite ≤1-second freshness at actual exclusive-open return, and
  the third broad scan plus activity/STOP/root-space checks before payload.
  Late persistent STOP, label, process, Neuron-holder and disk losses in the
  supplied intervals refuse. An empty or nonempty provisional bundle is
  retained after post-open failure; no automatic unlink, rename, retry or
  checkout follows that failure. Successful staging requires a distinct
  readback and **completed** postflight; a failed postflight is recorded as
  `POSTFLIGHT_HOLD`, not represented as success. The target and marker names
  are distinct from r9 r1–r4 and from the trn21 r10 stage path.
- In an isolated local worktree at the exact subject commit, `sha256sum -c`
  passed for all eight listed package files, `git diff --check` passed,
  supplied tests passed **36/36** under normal Python and **36/36** under
  `python3 -O`, and the two direct entrypoint checks each returned `HOLD`
  with exit code 1. The r3 HOLD and r10 launcher PASS receipt hashes match
  their actual claimed bytes; only the r4 PASS digest above is wrong.

## Limits and prohibited inference

This is a **static HOLD**, not a table or BPB verdict. There was no EC2/SSH
contact, marker, source transfer, checkout, cleanup, native training,
evaluation, submission or AWS action. The inherited r4 residuals remain:
live trn22 timing of the broad scans is unknown; userspace scans are not
atomic against same-UID/root or alias-path interference; a label introduced
after the last scan or a delayed payload can escape a sampled boundary.
Independent read-only timing, an exact enabled request and final review,
fresh host checks, a Supervisor one-attempt assignment, and later stage-result,
G2 and native gates would still be required even after the digest repair.
