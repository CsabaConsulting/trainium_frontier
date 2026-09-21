# Current state

Last reconciled: 2026-09-22

## Verified repository state

- Local checkout: `main` at `7f9aaaf5` (older than the latest research handoff)
- Authoritative research branch: `origin/research/fresh-start-20260918`
- Authoritative research commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Recovery-only snapshot: `origin/backup/fresh-start-20260918` at `c7f151a7`; it does not
  supersede or replace the authoritative branch
- Existing runner: `research/run_experiment.py`
- Existing ledger: `research/experiment-log.jsonl`
- Mandatory machine protocol: `research/experiment-protocol.md`
- Phase 1 deadline: 2026-09-30 23:59 PT

## Known score state

- Leaderboard screenshot shows best accepted score: `1.0246`.
- Best preserved comparable public evaluation is `0.9928513116311253` BPB; its independent cold
  replay is `0.9938556748843932`. These are public development measurements, not the private
  leaderboard score.
- The frozen one-file candidate is
  `submissions/dense6x1024_public0992851/train.py` on the authoritative branch.
- The exact commit/configuration that produced official `1.0246` is not yet recorded here.
- A later submission was rejected by review; its reviewer explanation is not yet recorded here.

## Blocking reconciliation tasks

1. Record the submission ID, commit, launch command, and config for official `1.0246`.
2. Record the complete reviewer reason for the rejected submission.
3. Record the current leaderboard rank and tenth-place cutoff.
4. Reconcile and preserve the existing untracked receipts on both Trn2 boxes before selecting a
   clean immutable worktree for any future run. The 2026-09-21 read-only inventory is recorded in
   `research/agentic/HOST_INVENTORY_2026-09-21.md`.
5. Confirm official team membership and replace the emailed shared PEM with per-user access.
6. Use the US source Region/profile pair: `us-west-2` with
   `us.anthropic.claude-fable-5-1` and `us.openai.gpt-6-astra`. The earlier zero quotas and
   access-denied results apply to the incorrect `eu-north-1` plus `global.*` route. Current US calls
   omit unsupported `temperature`. Astra passed the runtime doctor and completed the first bounded
   read-only Researcher/Supervisor cycle; Fable most recently returned a transient
   service-unavailable response.

## Active technical direction

Source197 offline attribution is complete. It identifies a caller-boundary constraint: moving the
same single library MLP from layer 0 to layer 5 changes the compiled forward graph from 135.871 ms
to 329.259 ms and adds 16.344 GB of reported spill reloads. The direct library-family work changes
only modestly; ordinary BF16 MLP/`F.linear` regions, transposes, and spills expand sharply. The
forward delta explains 99.55% of the unprofiled microstep gap. See
`research/agentic/SOURCE197_OFFLINE_ATTRIBUTION.md`.

The complete saved-ROW-FP8 MLP backward remains on hold. The next proposal must be one bounded
caller-boundary/layout experiment that preserves ordinary compiled MLP behavior and defines a
same-host complete-forward/step gate. It must not assume that `contiguous()` is the fix without an
evidence-linked boundary argument.

Bedrock Researcher/Supervisor review narrowed that next experiment to moving only the selected
layer-5 library MLP's two exact BF16 transposed weight mirrors from forward into the already charged
weight-cache refresh boundary. The locally reconciled draft is
`research/agentic/proposals/20260922-source197-refresh-layout-l5.json`. It is not approved for
implementation or execution. Wave42's prior full-model loss with persistent prepacked weights is
recorded as related negative evidence; the new causal distinction is the exact Source197 library
boundary at one selected layer.

The saved ROW-FP8 forward is a real component lead (about 9.4% at one rank and 19.1% under four-rank
contention in source199), but it has no backward, whole-step, or BPB proof. It must achieve a
repeatable complete-step improvement of at least 3% and pass numerical/cache/evaluation gates before
any 30-minute quality run is proposed.

Do not reopen the parked attention redesign, generic full-model autocast FP8, Vector-engine square,
compiler-SBUF placement, width1152, or previously tested library-MLP compositions without a new
profile-backed causal hypothesis.

## Human approvals required

- Any Trn2 training launch
- Any AWS resource mutation
- Any merge to the submission candidate
- Any leaderboard submission

## Agent-loop implementation

- Architecture and safety plan: `research/agentic/AUTONOMOUS_LOOP_PLAN.md`
- Current stage: Milestone B (offline evidence gate complete; next bounded proposal not yet
  approved), before remote execution authority.
