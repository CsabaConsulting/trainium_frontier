# Current state

Last reconciled: 2026-09-24 09:10 UTC

## Final-week override — LR 0.015 adopted, canonicalization in progress

This section supersedes the older score, base, queue, and host statements below.

- The best confirmed public-development candidate is the review-clean dense6x1024 file with
  `MATRIX_LR = 0.015`, SHA-256
  `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.
  Two seed-42, full-budget, fresh-process 20,971,520-token evaluations on `trn21` measured
  `0.9920722144854677` at 3,635 steps and `0.9918353324459732` at 3,654 steps. Their arithmetic
  mean is `0.99195377346572045`; both beat the prior public best `0.9928513116311253`.
- The evidence for both runs is now imported into this canonical controller checkout under
  `research/submission-validation/20260924-c595-matrixlr015-trn21/`. The confirmation Operator
  receipt records a clean seed-42 run, 1,800-second budget compliance, 20M token/83,462,186-byte
  evaluation, finite checkpoint, and exact candidate hash.
- A canonical root candidate was implemented from reviewed parent `4ed89f36` as commit
  `e12d0deedea806128c1999f4b0a6340b967a7e64`. It changes only root `train.py` and has the exact
  candidate hash above. Independent verification is still required before the controller may pin
  or run it.
- `TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS` is 900. The two LR 0.015 runs observed approximately
  839 and 831 seconds of startup, so cold-compile margin is only 61–69 seconds. Compile-cache
  reuse may accelerate research, but official-cold behavior must remain a separate gate.
- The old fixed queue based on `9173f626…` is retired for science decisions. It remains inert
  until the new base, controls, proposals, and independent receipts are installed. No old 640-stack
  proposal may be launched as a fallback.
- At 09:04 UTC both `trn21` and `trn22` were idle with no Neuron/device or experiment-lock holder.
  Disk was 6.8575 GiB and 6.9148 GiB free respectively. An Operator is checking only archived,
  controller-owned cleanup targets; no unknown worktree, data, log, or unapproved checkpoint may
  be removed. The protocol's approximately 5 GiB safety floor is unchanged.
- Final-week selection uses context-1024 public BPB. A 2,097,152-token fresh-process evaluation may
  screen an arm, but cannot establish a KEEP. Promotion requires a fresh 20,971,520-token evaluation;
  adoption requires two negative 20M deltas with mean at most `-0.0008` against current-base,
  same-host controls. Submission and instance lifecycle actions remain human-only.

## Human-authorized fixed-queue override — pending independent verification

The current implementation branch represents, but does not activate, the 2026-09-23 replacement
plan in `frontier-full-only-queue.json`. It parks
`20260923-local-conv-pre-attn-serial-only-r3`, pins the exact dense6x1024 lean train bytes
(`9173f626…`), restores `trn22`, and disables general proposals and smoke runs. The only eligible
order is exact cold control on `trn22`, MATRIX_LR 0.025 on `trn21`, then LR 0.015, warmdown 0.60,
warmdown 0.85, total batch 262144, and MAX_TRAIN_SECONDS 1080. Public BPB from a fresh context-1024
evaluation over exactly 20,971,520 tokens / 83,462,186 bytes is the only selection metric.

This is not launch approval. Every queue entry remains pending an independent Verifier receipt.
The persisted disk breaker is preserved and the fixed controller additionally requires the exact
pushed R8 cleanup receipt plus an explicit Operator state transition. No breaker was cleared and no
remote or hardware action was performed while implementing this override.

## Operational override — 2026-09-23 16:35 UTC

This section supersedes older dynamic controller, proposal, and host statements below.

- The controller branch is synced to origin at `1493ff2b080a5eb3b2a45fd72742906e43906736`.
  The systemd service exited fail-closed with status 3 and is waiting on its configured five-minute
  restart because the persisted breaker is open; an active service unit is not evidence of active
  research while this breaker remains set.
- Proposal `20260923-local-conv-pre-attn-serial-only-r3` passed proposal verification and independent
  implementation verification. Its isolated candidate is
  `79b0cecb2f655441f4e0898f6140308eea6072f0`. No smoke launched: two inventories measured idle
  `trn21` at 7.3 GiB free, below the immutable 7.4 GiB gate, and opened the breaker.
- Read-only live inventory at 16:34 UTC confirmed that both boxes are idle with no `torchrun`,
  `train.py`, or `run_experiment.py` process. Available bytes were 7,835,312,128 on `trn21` and
  8,628,547,584 on `trn22`. The controller still schedules only `trn21`; `trn22` remains excluded
  pending the separate hash-preserving cleanup/re-enable decision required by
  `operations/TRN22_DISK_GATE_RECOVERY_2026-09-23.md`.
- Read-only attribution found 2,254,217,216 bytes of immutable controller run directories on
  `trn21` and 4,371,390,464 bytes on `trn22`. No remote file was removed. The exact inventory and
  recovery boundary are recorded in
  `operations/TRN2_AUTOLOOP_DISK_BLOCK_2026-09-23_R1.md`.
- The latest full result remains the n-gram-table cooldown DISCARD, `1.007831` / 2072 steps versus
  the same-host control `1.005734` / 2071. No autonomous result beats the preserved public record
  `0.9928513116311253`; there is no new submission candidate.

## Primary-source research override — 2026-09-23 15:26 UTC

- The primary-source and source-history review is durable in
  `research/agentic/PRIMARY_SOURCE_RESEARCH_MAP_2026-09-23.md`. It is a ranked queue, not run
  authorization.
- `20260923-ngram-ve-cooldown-lr-flag-only-default-shape` completed its smoke process successfully
  at `1.168643` / 334 steps versus the same-host smoke control `1.165855` / 334. It preserved
  throughput but was `+0.002788` BPB worse at the short horizon. The proposal defines smoke as a
  correctness/throughput gate, so collection and the independent Analyst verdict still control
  whether a full run is eligible. This is not an improvement.
- After the active proposal closes, the strongest next architecture lead is a source-history
  repair: commit `7518d91e` introduced the Q/K/V snapshot before `local_conv` for a rejected MUDD
  experiment, and that ordering survived. The current trained local convolution therefore does not
  feed same-block default attention, contrary to its comments and its earlier pre-attention wiring.
  It has prior GPU and Trn2 positive evidence through related paths and matches the Canon-A placement
  studied in arXiv:2512.17351. The Researcher must still produce one bounded proposal and an
  independent verifier must approve it before implementation or hardware use.
- Secondary zero-hardware lanes are modern Muon feasibility (RowUpdateFloor/radial control,
  SOAP-Muon, then Newton-Muon) and completing the saved ROW-FP8 backward. Training-time random token
  replacement is parked until a measured stack actually enters a repeated-data regime.
- No new full-run score beats `0.9928513116311253`, and submission validation remains
  STOP/ABANDON_CURRENT_PROPOSAL.

## Reconciliation override — 2026-09-23 15:13 UTC

This section supersedes the stale dynamic run/box statements later in this handoff.

- The autonomous controller is healthy on `agent/control-plane-20260921` at `b933e2ca`, synced
  to origin, with breaker closed after cycle 287 and 37 retrieved runs.
- Latest measured experiment: `20260923-ngram-ve-double-hash-r2` smoke was a clean DISCARD at
  `1.171049` / 325 steps versus the same-host smoke control `1.165855` / 334 steps. It was 2.66%
  slower and `+0.005194` BPB worse. Verdict and checkpoint-prune evidence are pushed in
  `2ed3c130` and `b2cea2e7`; do not promote it to a full run.
- The preceding attention-scale 0.12 full run was also a clean DISCARD: `1.009764` / 2071 steps
  versus `1.005734` / 2071. The next value-residual proposal was rejected by the independent
  verifier before hardware use (`b933e2ca`).
- Both Trn2 boxes were idle with no Neuron holder at 15:04 UTC. Approved exact-target cleanup of
  three already archived/reconstructable checkouts restored trn21 to 8,371,343,360 free bytes;
  see `research/agentic/operations/TRN21_ARCHIVED_WORKTREE_CLEANUP_2026-09-23_R7.md`.
- The best valid autonomous full KEEP remains BF16 SDPA input casting at `1.002028` versus
  `1.005967`; it still does not beat the preserved public-development result `0.9928513116311253`
  (cold replay `0.9938556748843932`). There is no new submit-ready winner.
- A new external review was reconciled in
  `research/agentic/EXTERNAL_REVIEW_RECONCILIATION_2026-09-23_R2.md`. Its strongest duplicate-clear
  control-stack lead is the existing `--no-ngram-ve-flat-lr` flag (late table cooldown). Search of
  `DECISION_LOG.md`, `CURRENT_STATE.md`, `research/experiment-log.jsonl`, and `train.py` found the
  flag but no measured `flat_lr=False` run. Treat that as a hypothesis requiring the normal
  proposal and independent-verification gates, not as a result.
- Submission validation remains STOP/ABANDON_CURRENT_PROPOSAL at `beba1595` in the separate
  `Trainium-submission-validation` checkout. Do not submit the current lean-review file: the
  small compiled-Muon raw-byte discriminator is not a valid gate, and no replacement gate plus
  complete dress rehearsal has passed.

## Verified repository state

- Active controller checkout: `agent/control-plane-20260921`, clean and synced to
  `origin/agent/control-plane-20260921`; deployed controller code includes `3b26fee1`
- Authoritative research branch: `origin/research/fresh-start-20260918`
- Authoritative research commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Recovery-only snapshot: `origin/backup/fresh-start-20260918` at `c7f151a7`; it does not
  supersede or replace the authoritative branch
- Existing runner: `research/run_experiment.py`
- Existing ledger: `research/experiment-log.jsonl`
- Mandatory machine protocol: `research/experiment-protocol.md`
- Autonomous state: 30 retrieved result records (31 launch attempts including the failed initial
  `trn22` control); proposal `20260923-unet-skip-zero-init-r2` is smoke-verified and awaiting its
  full run; automatic Git push is enabled, but the controller breaker is open on the `trn21` disk
  gate described below
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
- Latest completed smoke is `20260923-unet-skip-zero-init-r2`: `1.167766` at 330 steps on `trn21`,
  versus the same-host control smoke `1.165855` at 334 steps. All correctness gates passed and the
  independent Analyst returned KEEP/eligible-for-full. A smoke is not a comparable full-result win.
- The frozen one-file candidate is
  `submissions/dense6x1024_public0992851/train.py` on the authoritative branch.
- The exact commit/configuration that produced official `1.0246` is not yet recorded here.
- A later submission was rejected by review; its reviewer explanation is not yet recorded here.

## Blocking reconciliation tasks

1. Record the submission ID, commit, launch command, and config for official `1.0246`.
2. Record the complete reviewer reason for the rejected submission.
3. Record the current leaderboard rank and tenth-place cutoff.
4. `trn21` is idle but blocked at 7.26 GiB free, below the configured 7.40 GiB gate; the second
   failed full-run preflight opened the breaker at 2026-09-23 03:11 UTC. `trn22` remains excluded
   at about 7.37 GiB. Read-only attribution found 3.89 GB in old controller-created isolated run
   worktrees, but current cleanup authority covers scratch checkpoints only. See
   `research/agentic/operations/TRN21_UNET_FULL_DISK_BLOCK_2026-09-23.md`; do not lower the gate or
   remove worktrees/caches without explicit human coordination.
5. Confirm official team membership and replace the emailed shared PEM with per-user access.
6. The valid Bedrock route is `us-west-2` with `us.anthropic.claude-fable-5-1` and
   `us.openai.gpt-6-astra`; the old `eu-north-1` plus `global.*` failures are irrelevant. Fable's
   long streamed Researcher calls previously suffered silent read timeouts and premature
   termination. Commit `500615a9` raised the validated socket read timeout to 1,800 seconds while
   retaining the single fresh retry and fail-closed parsing. Multiple production calls have since
   completed beyond the former 600-second boundary, so the transport repair is now live-verified.

## Active technical direction

The active proposal is `20260923-unet-skip-zero-init-r2`, candidate commit `7e500778`. Its
independently verified smoke passed every correctness gate and earned KEEP, but its full promotion
has not started because the `trn21` disk breaker is open. It cannot be called an improvement until
a retrieved, comparable full run beats its same-host control. The strongest completed autonomous
full result remains BF16 SDPA input casting, which has not beaten the preserved `0.992851` public
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
