# Independent re-review: trn21 G1 r2 guarded evaluator authorization schema

Reviewed: 2026-09-24 23:30 UTC. Implementer schema-fix commit:
`03dfb3e12437c874598813d2829b91dd3cda0cbe` (controller cherry-pick
`76523ad7c5631884a6931bf62a0b394facf2cdcd`). This review supersedes
the schema HOLD in `20260925-g1-trn21-r2-guard-static-review.md` only.

**Verdict: PASS for the corrected static guard package to proceed to a
separate Supervisor one-label authorization decision. Evaluator dispatch
remains HOLD.** This is not trn21 G1 PASS, A0 training authority, a new model
score, or permission to clean a host, reset a breaker, modify AWS resources,
or submit. I made no host connection or host change.

## Exact package binding

| Reviewed item | SHA-256 |
| --- | --- |
| `request.staged.json` | `aba839eb87b041fbb266e2db3795d49f29afcf76c91ecd5eee96a5b3f978dd2a` |
| `request.guard-pending.json` | `921498a53b010006649a4570f10513d43988a1f0ea770031f1c739158519ff91` |
| `guarded_eval.py` | `5438d644513b0cddb554727e9c50f4b5e09bba43628ce5869109d17c1c598e70` |
| `dispatch_one_label.py` after fix | `7af47499490c6250f3fabb0514c01162f48336eec505ee3ec5f25cb8bcbbfcef` |
| `test_guard_package.py` after fix | `a12d6aa0c1111e90cbabd9dcf2c2d31cbd4628c078a5db5ccab715672b1ebbab` |
| Guard runbook | `db6f70bf4e1cf7bad899b012f02788072a0149e9d2191762ec724ad121bf843a` |
| Independent staged-source review | `bf3c78a7db65ea7829386129969e7bb06cc700f11c2d4b2f251c448f04878a21` |

The fix changes only `dispatch_one_label.py` and its local test file;
`git diff 03dfb3e1^ 03dfb3e1 --check` passed. The pending request and
host-local monitor retain exactly the hashes reviewed in the prior HOLD
receipt. The pending request still has `launch_authorized=false` and null
authorization. Its evaluator suffix equals the staged request's command
after the original five-item outer-timeout prefix. The label, host, trained
commit, historical environment, checkpoint path/size/hash, trusted
`prepare.py` and trained `train.py` hashes, 1,024 context, 20,971,520-token
limit, 1,200-second evaluator timeout, and 6.5-GiB floor plus 2-GiB reserve
all remain bound. The additional hash entries in the pending request are the
proven launch/runtime closure; direct hashes of those exact files at trained
commit `e7f11c38dc3b5e68b373322e10cfef580f1e8d36` matched the manifest.
The six historical cache directories are path/presence-bound, **not**
cache-content-SHA-bound; this static review makes no stronger claim.

## Corrected boundary and negative tests

The adapter now rejects any authorization object whose keys are not exactly
`decision`, `pending_package_sha256`, `stage_review_sha256`,
`guard_package_verifier`, and `supervisor_authorized_at_utc`. The nested
verifier must be an object with exactly `path` and `sha256`. The timestamp
must be a real calendar timestamp with an explicit `Z` or `+00:00` UTC offset;
truthy non-timestamps and non-UTC offsets fail. These checks run before
verifier-path access, SSH-key-path access, launcher import, or remote dispatch.
The adapter still compares every non-authorization request field against the
pending package, checks the expected stage/pending hashes, checks the reviewed
receipt file by hash, and refuses the pending file itself. A separate
Supervisor-authored copy is still mandatory.

The supplied local suite passed **8/8** tests. Its subprocess negatives
confirm extra/missing top-level authorization keys, extra/missing nested
verifier keys, and malformed/non-UTC timestamps fail before transport. My
independent direct schema matrix passed five missing-field, five invalid
verifier-shape, twelve malformed/non-UTC timestamp and two valid-UTC cases.
The corrected UTC parser rejects impossible dates and leap seconds. No
synthetic test contacted a host or opened a real SSH key.

The unchanged monitor was inspected against the prior receipt: it launches
only the exact `prepare.py eval-public` argv in a new child session, verifies
its own child PID birth tick/UID/process group/marker/argv before signalling,
samples root free bytes every one second, and requests a scoped stop below
9,126,805,504 bytes. The inherited proven launcher checks source/runtime
hashes, clean checkout, host lock, unique label, and the 8-GiB floor before
starting its owned-process guard. None of this is a substitute for a fresh
live preflight or an observed run.

## Remaining launch and result gates

The Supervisor must create and hash the exact authorized copy described in
`RUNBOOK_GUARD.md`, binding this receipt's absolute path and SHA-256. A
distinct Operator must then recheck host identity, source/checkpoint/cache
and tokenizer hashes, environment names, no holders or existing label,
closed breakers and current disk space; install only the hash-checked monitor
at its absent external path; dispatch this label once; observe the floor and
guard; and archive two stable quiescent postflight checks with the evaluator
log and monitor events. Any attempted evaluator dispatch consumes r2. A
distinct result Verifier must confirm the fresh public-20M calibration before
any trn21 G1 PASS. No A0 or other scored job follows automatically.
