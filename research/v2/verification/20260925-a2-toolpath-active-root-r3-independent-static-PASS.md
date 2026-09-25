# Independent A2 tool-path r3 review — disabled static PASS only

Subject: `origin/codex/20260925-a2-toolpath-active-root-r3@9baa30250af213ed20451d06d1d1b14e851fef99`.
Verifier was separate from the Implementer. Verdict: **PASS_DISABLED_STATIC_ONLY**.
This accepts the local, disabled repair of the r2 controller-drift defect; it is
not permission to contact trn22, stage source, run CPU G2, train, evaluate,
delete, reset a breaker, or submit. The A2 r4 CPU-G2 HOLD remains controlling.

- The collector reads breaker and launch HOLD solely from
  `/home/poojith-devan/Desktop/Trainium/research/v2/state.json`; it checks STOP
  in that active controller, the legacy Documents checkout, and its package
  worktree before each future SSH read. The new r3 branch compares active-state
  SHA-256 before and after the observations and forces `HOLD` on any drift.
- The fixed e65 Git tree contains neither `workflow/tools` path. Its `train.py`
  and `prepare.py` SHA-256 are exactly `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The expected tool hashes match the controller's corresponding local files;
  host presence remains unknown until a separately authorized read-only probe.
- All five local source pins pass `collect_readonly.py --check`; request SHA is
  `65e63ffa09213ea5831f5a2797378bc67a7b93acd215d7a519a9c976ec446dc1`,
  remote tool-probe SHA is `51d5f6be8b313a51efc56609dd391610ee1b6528e9f118f95176679484ef83b7`,
  and collector SHA is `ad1d62d93ca70e9ccfaae6fccc4e3bf066878cfab9a2aae6fe5cd9248647daac`.
- The checked-in request has all authority bits false. The three future
  enablement/review/assignment files are absent. Default `--capture` returns
  exit 3, `HOLD_NO_SUCCESS_CLAIM`, before SSH. The only write in this package
  is an exclusive **local** receipt after a future separately authorized
  capture; fixed remote scripts open files read-only and do not alter data or
  evaluation code. Their output is bounded and credential-pattern-suppressed.
- Supplied CPU/offline tests pass 10/10 with `python3 -B` and 10/10 with
  `python3 -OB`; `git diff --check e4f523cc 9baa3025` passes. The new
  non-breaker active-state-drift witness ends in HOLD despite a clean stale
  Documents state.

Future enablement requires its own independent review, exact byte chain,
Supervisor assignment and Operator action. The future request says “one
read-only capture,” but this disabled collector does not itself consume a
one-shot marker; that anti-replay/assignment boundary must be checked before
any read-only dispatch. Any later observation proves planning feasibility at
that instant only, not CPU/native authority or a BPB result.
