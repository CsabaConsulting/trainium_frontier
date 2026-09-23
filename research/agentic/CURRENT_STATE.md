# Current state

Last reconciled: 2026-09-23 02:34 UTC

## Verified repository state

- Active controller checkout: `agent/control-plane-20260921` at `3b26fee1`, clean and synced to
  `origin/agent/control-plane-20260921`
- Authoritative research branch: `origin/research/fresh-start-20260918`
- Authoritative research commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Recovery-only snapshot: `origin/backup/fresh-start-20260918` at `c7f151a7`; it does not
  supersede or replace the authoritative branch
- Existing runner: `research/run_experiment.py`
- Existing ledger: `research/experiment-log.jsonl`
- Mandatory machine protocol: `research/experiment-protocol.md`
- Autonomous state: 29 retrieved result records (30 launch attempts including the failed initial
  `trn22` control); no open run or proposal at reconciliation; local user service active in cycle
  196 with a closed breaker and automatic Git push enabled
- Evidence durability: commit `691faccb` retrieved and SHA-verified the previously omitted
  per-run launch receipt for all 29 durable results. Future collection retrieves that receipt,
  enumerates every untracked path, and fails the correctness gate on any unexplained post-run dirt.
- Research grounding: after two source-contradicted proposals were independently rejected before
  hardware use, commit `3b26fee1` added a bounded authoritative `train.py` measured/closed evidence
  index to the Researcher prompt. The complete-source Astra verification gate remains unchanged.
- Phase 1 deadline: 2026-09-30 23:59 PT

## Known score state

- Leaderboard screenshot shows best accepted score: `1.0246`.
- Best preserved comparable public evaluation is `0.9928513116311253` BPB; its independent cold
  replay is `0.9938556748843932`. These are public development measurements, not the private
  leaderboard score.
- Best independently verified autonomous-loop full result is
  `20260922-bf16-sdpa-input-only-default-shape`: `1.002028` versus its same-host `trn22` control
  `1.005967`, at 2233 versus 2076 steps (7.56% apart, inside the 10% comparability gate). This is a
  valid KEEP relative to the pinned control, but it does not beat the preserved `0.992851` public
  development record.
- Latest completed full result is `20260922-bf16-norm-output-except-mlp-input`: `1.020309` versus
  its `trn21` control `1.005734`, at 2352 versus 2071 steps. The 13.57% step difference exceeds the
  comparability gate, so its durable verdict is CONFOUNDED, not a win.
- The frozen one-file candidate is
  `submissions/dense6x1024_public0992851/train.py` on the authoritative branch.
- The exact commit/configuration that produced official `1.0246` is not yet recorded here.
- A later submission was rejected by review; its reviewer explanation is not yet recorded here.

## Blocking reconciliation tasks

1. Record the submission ID, commit, launch command, and config for official `1.0246`.
2. Record the complete reviewer reason for the rejected submission.
3. Record the current leaderboard rank and tenth-place cutoff.
4. `trn21` is eligible and idle with 8.12 GiB free as of 2026-09-23 01:27 UTC. `trn22` is idle but
   excluded from launches with 7.37 GiB free, below the configured 7.5 GiB gate. Do not remove
   unknown or teammate-owned data merely to recover the missing space.
5. Confirm official team membership and replace the emailed shared PEM with per-user access.
6. The valid Bedrock route is `us-west-2` with `us.anthropic.claude-fable-5-1` and
   `us.openai.gpt-6-astra`; the old `eu-north-1` plus `global.*` failures are irrelevant. Fable's
   long streamed Researcher calls previously suffered silent read timeouts and premature
   termination. Commit `500615a9` raised the validated socket read timeout to 1,800 seconds while
   retaining the single fresh retry and fail-closed parsing. Multiple production calls have since
   completed beyond the former 600-second boundary, so the transport repair is now live-verified.

## Active technical direction

There is no active proposal. The Researcher must reconcile any new idea against the 29-run ledger
and the durable rejection feedback before proposing one single causal change. The strongest live
autonomous result is BF16 SDPA input casting, but it has not beaten the preserved `0.992851` public
development record. Recent norm-boundary variants increased throughput while worsening BPB or
became step-count-confounded, so throughput alone is not evidence of progress.

Source197 remains useful historical systems evidence: moving the same single library MLP from
layer 0 to layer 5 changed the compiled forward graph from 135.871 ms to 329.259 ms and added
16.344 GB of spill reloads. Any renewed caller-boundary/layout proposal must preserve ordinary MLP
behavior and define a same-host complete-forward/step gate; it must not assume `contiguous()` is a
fix without an evidence-linked boundary argument. The complete saved-ROW-FP8 MLP backward remains
on hold.

The saved ROW-FP8 forward is a real component lead (about 9.4% at one rank and 19.1% under four-rank
contention in source199), but it has no backward, whole-step, or BPB proof. It must achieve a
repeatable complete-step improvement of at least 3% and pass numerical/cache/evaluation gates before
any 30-minute quality run is proposed.

Do not reopen the parked attention redesign, generic full-model autocast FP8, Vector-engine square,
compiler-SBUF placement, width1152, or previously tested library-MLP compositions without a new
profile-backed causal hypothesis.

## Human strategic research priority — 2026-09-22

The human operator wants the unattended Researcher to search for mechanisms capable of closing the
leaderboard gap, not merely accumulate small step-rate changes. Within the standing authorization,
rank new hypotheses by plausible fixed-budget BPB upside and information gain across these lanes:

1. **Targeted FP8 or mixed-precision systems work.** Pursue precision changes only at a measured
   expensive boundary, with numerical, backward, whole-step, checkpoint, causality, and native-eval
   gates. The saved ROW-FP8 lead remains relevant; generic full-model autocast FP8 remains closed.
2. **Profile-directed NKI fusion.** Target a dominant end-to-end cost such as boundary/layout
   conversion, attention/MLP materialization, optimizer work, or cache refresh. An isolated kernel
   microbenchmark is not a win unless the complete step improves and BPB is preserved.
3. **Learning-efficiency re-optimization after a systems gain.** Once a measured throughput or
   memory win changes the feasible frontier, use bounded one-factor tests of capacity allocation,
   depth/width/context, batch or accumulation, learning-rate/warmdown, optimizer, and auxiliary
   memory capacity. Do not substitute an unstructured broad sweep or reopen a closed regime without
   new causal evidence.
4. **Other competition-legal architecture or training ideas.** Prefer mechanisms that improve
   loss per token or useful tokens processed within 1,800 seconds, and reconcile them against the
   full ledger before proposing a run.

Isolate causal factors first. A stack-confirmation experiment may combine only independently
validated KEEP changes and must declare the interaction hypothesis. Public development BPB is the
selection proxy; official private-shard BPB remains unknown until a human-approved submission.

## Human approvals required

- Development Trn2 launches are covered by `STANDING_AUTONOMOUS_AUTHORIZATION.md` after all
  proposal, independent-verification, preflight, and evidence gates pass.
- Any AWS resource mutation
- Any merge to the submission candidate
- Any leaderboard submission

## Agent-loop implementation

- Architecture and safety plan: `research/agentic/AUTONOMOUS_LOOP_PLAN.md`
- Current stage: unattended controls and flag experiments are live. The code-change path uses
  Researcher → proposal Verifier → isolated Implementer → deterministic diff/static gate →
  implementation Verifier → typed Operator → Analyst. Code candidates are limited to `train.py`,
  committed on isolated `autoloop/<proposal-id>` branches, and never merged automatically.
