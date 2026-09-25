# Independent trn22 e65→r9 incremental source-stage r3 review — HOLD

Reviewed 2026-09-25 08:46 UTC by a Verifier separate from the Implementer.
Exact subject: `80c96851edbb6cbc707c82c68773a95ea1231f91`, direct child of
held r2 `a7643f05c5468da6537c2d7b403a05ac048679f0`, on
`origin/codex/20260925-trn22-r9-incremental-source-stage-r3-hold`.
The subject adds only seven files under
`research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-r3-hold/`;
the r1/r2 packages and their independent HOLDs are unchanged. This is an
**offline static review**. No SSH/EC2/AWS host contact, stage, marker,
checkout, native training, evaluation, cleanup, breaker reset, or submission
occurred.

**Verdict: HOLD for source-stage enablement.** R3 closes the two concrete r2
pre-open defects under the stated cooperative-flock/ordinary-DAC model, but
introduces a zero-byte provisional-file unlink as part of its successful
failure handling. The current human controller brief reserves cleanup
decisions to the human; neither this disabled package nor any documented
human acknowledgement approves this deletion. Its runbook instead says a
future enabled request and independent review may accept it. That is not the
required authority. Do not issue a final source-stage PASS, enabled request,
one-attempt assignment, host write, or native approval from these bytes.

## Blocking authority and cleanup finding

`remote_incremental.py:591-593` calls `_rollback_unpopulated_bundle()` when a
post-`O_EXCL`, pre-payload check fails. At lines 448-462 the function verifies
that the opened and named objects are the same empty, single-link inode, then
uses `os.unlink(BUNDLE.name, dir_fd=parent)` and fsyncs the parent. This is a
real deletion of a newly created host file. The latest
`research/v2/CONTROLLER_BRIEF_2026-09-25.md` says cleanup decisions are
human-reserved; `workflow/CONTROLLER_PROMPT.md` independently reserves
protected deletion and prohibits weakening gates. The older standing
authorization allowed only narrowly identified controller scratch-checkpoint
cleanup, not stage-bundle deletion. The r3 `RUNBOOK.md:77-85` correctly says
this is not general cleanup authority, but incorrectly treats a future
enabled request plus independent review as enough to accept it. A Verifier
cannot supply the missing human decision.

There is also a precise **out-of-model limitation** in the rollback's
unqualified claim that it "never deletes a ... replaced ... file". The
independent disposable-directory witness inserts a *same-UID path-swap*
between the candidate's `os.stat` and `os.unlink`: the original empty inode
is renamed aside and a different nonempty inode takes its name. The
candidate deletes that replacement, returns from rollback, and leaves the
original inode under the other name. This requires a concurrent same-UID or
privileged actor (or our synthetic hook), which `RUNBOOK.md:88-94` explicitly
excludes from its bounded cooperative threat model. It is **not** evidence
that ordinary persistent host activity bypasses r3's pre-open checks. It does
show that the unconditional replacement-preservation sentence and any
general-purpose cleanup claim are false. The no-cleanup authority gap above
is blocking even without an adversarial actor.

**Narrowest repair:** remove this unlink path. If a post-open freshness or
activity gate fails, retain the exact provisional empty bundle and consumed
one-attempt marker, return HOLD, and require a separate human decision before
any later cleanup or retry. The actual-open freshness test and no-false-
success result can remain unchanged; this does not relax the inherited
one-second bound. Alternatively, obtain explicit human approval for the
one-inode rollback and narrow the runbook's replacement claim to its stated
same-UID-excluding threat model; the same-UID race would still exist. A
distinct repair needs a fresh independent review.

## What r3 *does* prove, and remaining bounded residuals

- A full activity check and root byte/inode check now follow both final broad
  label scans before `O_EXCL` (`remote_incremental.py:403-427`). Independent
  ordinary persistent STOP, process, Neuron-holder and disk-loss injections
  during the scans all refuse before either stage target exists. This closes
  r2 finding 1 in the synthetic bounded model.
- `_freshness_evidence()` is repeated at the **return of the actual**
  `os.open(O_EXCL)` (`remote_incremental.py:555-567`), not only before the
  call. An ordinary 1.1-second delayed open returns HOLD and, with no
  conflicting actor, rolls back the empty provisional name. This closes r2
  finding 2's false successful write receipt, subject to the cleanup HOLD.
- Another label scan, full activity check and root byte/inode check occur
  after open but before payload (`remote_incremental.py:568-590`). Ordinary
  persistent STOP/process/Neuron/disk changes injected at that point refuse,
  and the tested empty bundle is removed; an injected partial payload is
  instead retained without a checkout. The wrapper validates finite ages
  and both space records before readback.
- The independent witness delays the first **payload** `os.write` by 1.1
  seconds after those post-open age samples. R3 still creates the checkout
  and reports the earlier sampled ages under one second. This needs no
  adversarial actor, but the inherited numeric criterion and r2 HOLD were
  expressly tied to first `O_EXCL`, not first payload-byte completion.
  Therefore it is an **audit limitation, not a separate r2-regression
  verdict**. Do not describe the receipt as measuring freshness at actual
  payload completion. If a future authority claims that broader bound, it
  needs an after-write timestamp and a terminal HOLD retaining any nonempty
  partial; a pre-write check alone cannot remove scheduling races.
- A persistent attempt-label file introduced during the post-open `fuser`
  call *after* the third/last broad label scan reaches bundle and checkout
  creation with a write-phase receipt. This is the noncooperating same-UID/
  root race explicitly disclosed in `RUNBOOK.md:88-94`, not a hidden claim
  that r3 makes broad directory scans atomic. A later readback would fail its
  fresh label scan, but the write is already material. Future final review
  must explicitly accept this residual rather than call it impossible.
- `operator_once.py:334-354` attempts a postflight even after failure and
  records `POSTFLIGHT_HOLD` on an error. A newly appearing local STOP can
  prevent `_ssh_phase("postflight")` from contacting the host. Thus the
  wrapper proves an **unconditional attempt**, not an unconditional
  *completed* full remote postflight. Only an actual postflight receipt may
  support a successful stage result. This is truthful fail-closed behavior,
  but the runbook must not treat an attempted phase as a completed audit.

## Exact bytes and offline evidence

| Item | SHA-256 |
| --- | --- |
| `REQUEST.HOLD.json` | `2daf85561b7683ec4ee01a350f4f03fd4887ac66d64c7bafcb5988cbfeca595b` |
| `RUNBOOK.md` | `d7aa4b824d0050c2c3cfcb172b590530f442157e5b2e62ba54619e53c20f4a40` |
| `operator_once.py` | `d7a0b7467a3dca649b3056974b7d5e6e649fa900be0d47a2debf9a7f364d3794` |
| `remote_incremental.py` | `dcfde1310f143b561c24c97efc1cb0a193975cbadff80013fc8c76ef33197b6b` |
| `test_incremental_stage.py` | `b873ccc43021a73a5b7bba8174113fdec024eeeefb42a0382360c9f3f9d7fa14` |
| `test_late_scan_r2.py` | `0ffd411514fa3a7a175789a2f5b4ab26e7f1c5b460f52aa75994bf5cd7aa4546` |
| `test_final_boundary_r3.py` | `7ebd131ca6e215150e8e11c92d7e8c7a90648cc53a1a65c28b9194c82c778469` |
| reused r1 bundle | `f56ba94c4bc8d248583d356bf141b5523688d27c45c68a01765ceb70b5331ee7` |
| independent r3 witness file | `277e0a0f80cf303bdf43fbb918197eb720987e42868dd2f9584196f0420e7d0b` |

The 101,237-byte bundle independently passes `git bundle verify`, with r9
`e046de16826bfc3b954136e3c60804aea0554ecc` HEAD and e65
`e65a3097a5b00f6178188a935bb2acb9a5042c6c` sole prerequisite.
The disabled request binds the exact r3 remote/operator hashes and has
`dispatch_authorized=false`, `source_stage_authorized=false`, and both
native/eval flags false. The enabled request, final PASS and Supervisor
assignment files are absent. The CLI entrypoint refuses before a marker
or contact, including under optimized Python. No `train.py`, `prepare.py`,
data or evaluator bytes are changed by the subject.

Supplied tests: **35/35 PASS** under normal Python and **35/35 PASS** under
`python3 -O`. Independent file
`test_trn22_r9_incremental_stage_r3_independent.py`: **9/9 methods PASS**
normal and **9/9 PASS** under `python3 -O`, including eight additional
parameterized persistent STOP/process/Neuron/disk cases. The same-UID swap,
delayed-payload and late-label tests are witnesses of the bounded limitations
above, so their PASS means the behaviors reproduced, not stage approval.
`git diff --check 80c96851^ 80c96851` passes. Live trn22 timing of r3's
three full activity checks, three broad label scans and disk observations
remains **unknown**; no host timing attempt was part of this review.

No enabled request, independent final stage PASS, Supervisor assignment,
host source-stage attempt, native A0, table run, or BPB result follows from
this HOLD.
