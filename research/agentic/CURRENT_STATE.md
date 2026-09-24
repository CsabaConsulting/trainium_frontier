# Current state

Last reconciled: 2026-09-24 13:24 UTC

## Track-A controls valid; A1 blocked before step 1 by ownership adapter

This section supersedes earlier Track-A readiness and host-control statements below.

- Exact `35c9ef79...` plus MATRIX_LR 0.015 controls are valid on both hosts. The `trn21`
  control completed 3,636 steps and measured fresh-process public2M BPB
  `0.9866973734526626`; its evidence is pushed at `b0b23cbc`. The `trn22` control completed
  3,634 steps and measured `0.9873495157460374`; its evidence is pushed at `499f753c`.
  Their descriptive difference is `+0.0006521422933748` for trn22, so candidates must use the
  control from the same host. The screen nomination thresholds are `<=0.9856973734526626` on
  trn21 and `<=0.9863495157460374` on trn22. This screen cannot establish a KEEP.
- A1 (last-layer bigram+trigram tables, multiplier 64, dimension 128, RMSProp) has no BPB result.
  Retry r2 passed staging and source-integrity checks, then failed on all four ranks before step 1
  at the immutable ownership adapter with
  `ValueError: ownership probe is restricted to AdamW and Muon`. There was no checkpoint or
  evaluation. The exact failure, traceback, empty step ledger, and clean postflight are pushed on
  `evidence/20260924-v2-track-a-a1-trn21-r2` at `9d92ff82`; trn21 ended idle with
  9,867,505,664 free bytes. Do not interpret this as evidence for or against tables.
- Read-only source analysis found that the adapter wraps all optimizer groups, although its plan
  supports only FP32 AdamW and Muon. A1 adds a BF16 RMSProp group. The selected repair is to retain
  owner partitioning for FP32 AdamW/Muon while excluding the known RMSProp group from owner
  planning and stepping that original full group identically on every rank. Unknown kinds still
  fail. This is an infrastructure repair proposal, not a model change or run authorization; it is
  now in separate implementation and independent-verification gates.
- Hybrid-repair implementation `bf46f25c` passed its initial six CPU/fake-world contracts but
  failed independent adversarial verification at `0569cf3e`: checkpoint group-kind tampering and
  post-plan live group-kind drift were not authenticated before state mutation/collectives. That
  could silently stop RMS updates or create rank-asymmetric failure. The FAIL receipt and tests are
  pushed on `experiment/20260924-v2-a1-hybrid-rmsprop-ownership-repair`; native packaging and
  hardware remain forbidden while an Implementer adds exact ordered group-schema validation and a
  fresh Verifier repeats the gate.
- A flags-only AdamW substitution is rejected as a repair: it changes the table optimizer, still
  conflicts with the adapter's FP32 requirement for BF16 tables, and historical paired evidence
  found worse per-step quality and throughput. Disabling ownership globally is also not selected
  because ownership previously improved dense whole-step time by about 5.06%.
- Track-A A4 (`--mlp-sandwich-norm`) was not scheduled because it is a prior dense duplicate.
  The full 1,800-second `t39-sandwich-1800s` run regressed context-1024 BPB by `+0.006100`
  (`1.040276` versus `1.034176`) and was already discarded. No new branch, edit, or run was made.
- Track-A A5 (`--relu2-tau 0.5`) was also stopped before packaging. The same shifted-ReLU
  mechanism already lost about `+0.002903` observed BPB in paired 900-second dense runs, and exact
  `35c9...` cannot run it as a flags-only arm because its fused training MLP hard-requires
  `RELU2_TAU == 0.0`. A valid retest would be a separately researched kernel implementation, not
  the requested one-factor flag experiment.
- Track-A A6 (`--head-gate`, whose default includes per-head normalization) was stopped as another
  prior dense negative. `t39-hgate-900s` ran about 25% slower per step, completed about 19.8% fewer
  steps, and lost roughly `+0.0135` BPB against paired controls. Exact `35c9...` has no fused NKI
  implementation to change that wall-clock tradeoff, and the flag bundles two explanatory factors.
- Track-A A8's embedding-LR 0.45 sub-arm was stopped as a closed axis. The exact prior
  `t04-embedlr-045-900s` change from 0.30 to 0.45 measured only `-0.000057` BPB, below noise; the
  durable study concluded that 0.30 is a flat optimum. No full run was repeated.
- A8 embedding LR 0.60 was also stopped on final-week value-of-information. It has no isolated
  positive evidence, and width scaling makes dense1024 at 0.60 only about 5.4% more aggressive
  than the old dense640 0.45 null point in effective optimizer LR. Matrix LR 0.015 provides no
  mechanism for moving this closed embedding optimum upward.
- A8 warmdown ratio 0.90 was stopped as an exact prior duplicate. The one-factor
  `t04-warmdown-90-900s` run was `+0.000254` BPB worse than its same-code control mean; ratio 0.60
  was also weakly negative, bracketing the 0.75 default. The durable schedule study closed this
  axis, so no 1,800-second repeat was launched.
- Track-A A9's x0-gate sub-arm was stopped as a decisive prior wall-clock negative. Its paired
  dense run was about 13% slower, completed 11.2% fewer steps, and lost `+0.008694` BPB; the
  estimated per-step effect was within noise. The same extra per-block reduction remains outside
  exact `35c9...` optimized adapters, so no repeat was packaged.
- A9 out-pool was likewise stopped as a prior dense negative/null. The exact one-factor
  `t39-outpool-900s` run lost about `+0.0013` BPB at both 1024 and 2048 context while completing
  1.4% fewer steps; its throughput-adjusted residual was also non-improving. No repeat was made.
- A10 (depth 8 x width 768 plus winning tables) is ineligible: there is no adopted table winner,
  and the proposed arm bundles depth, width, and table changes. Prior depth-8 evidence is also
  strongly wall-clock negative. The duplicate-cleared Track-A queue therefore has no immediately
  runnable arm: A1 awaits the hybrid-ownership repair gates; A2 shares that blocker; A3/A7 flags
  are absent; A4/A5/A6/A8/A9 are evidence-stopped; A10 is dependency-blocked. Unblocking and
  verifying A1 is the next useful Track-A action.
- The next researched code-change lane after A1 is dense local-conv ordering. Exact `35c9...`
  still snapshots Q/K/V before applying the trained causal-convolution residual, so same-block
  serial attention does not see local-conv output. Moving the existing update before those
  snapshots on the serial/non-sandwich path is one parameter-neutral, operator-count-neutral
  causal change with related positive evidence, but it has never run on exact dense35c9+LR0.015.
  It requires a new proposal, independent implementation/verification, equivalence and causality
  gates, native NKI/eager equality, and same-host throughput/BPB comparison; it is not currently
  hardware-authorized.
- Track B1 (`LNC1`, world size 8) is blocked and is not a one-factor topology arm on exact 35c9.
  Pack32 makes one world microbatch 262,144 tokens, exceeding and failing divisibility against the
  131,072-token total batch, while the ownership adapter explicitly requires world size 4.
  Changing pack or total batch would confound topology; all exact LNC1 kernel, ownership,
  collective, HBM, checkpoint, and fresh-cache gates also remain unauthenticated. Historical LNC1
  evidence is unfavorable. Do not propose B1 until A1 works and a separately matched pack16
  topology contract clears the documented component gates.
- A2 (bigram+trigram tables at layers `1,-1`, mult64, dim128, BF16 RMSProp) is
  `GO-CONDITIONAL` only after A1 passes and produces a valid archived same-host result. Historical
  evidence found adding layer 1 improved about `-0.0118` BPB at fixed steps, with a roughly 5.86%
  throughput cost; the current-stack prediction versus A1 is about `-0.010` fresh2M BPB with a
  wide `[-0.025,+0.005]` interval. The exact profile creates four 128-MiB tables plus roughly
  512 MiB each of replicated RMSProp moments and dense gradients, so ownership correctness, HBM,
  collectives, checkpoint/load, causality, throughput, and same-host A1 comparison must gate it.
  No executable A2 proposal may exist before A1 clears those dependencies.
- The paused A2 scaffold is unauthorized and must not run: it has the same RMSProp/ownership
  blocker. Track B2 sparse touched-row work also remains rejected before hardware. The best
  confirmed submission candidate therefore remains the review-clean MATRIX_LR 0.015 dense file,
  best single public20M BPB `0.9918353324459732`, two-run mean `0.99195377346572045`.

## Standing mission v2 — table-first first-place campaign

At 2026-09-24 15:30 IST the human replaced the small-tuning critical path with the table-first
campaign in `STANDING_MISSION_V2_2026-09-24.md`. The exact Recursive primary source and the exact
`35c9...` flag surface have been audited. Track A now starts with per-host `35c9...` LR-0.015
controls followed by last-layer and layers-`1,-1` hashed-table arms. Track B develops a
touched-row-only synchronized table optimizer. Track C keeps the review-clean candidate behind
adopted research changes by at most 12 hours. Window-pattern and RoPE-base experiments are not
launchable as flag-only arms because those flags do not exist in `35c9...`.

Track B2's CPU feasibility contract and blocked native-probe harness independently passed on branch
`experiment/20260924-track-b2-ngram-touched-row` at verifier receipts `b58ce84f` and `a2a44065`.
This is not hardware authorization: the probe remains non-dispatchable until it is bound to an
authenticated successful `35c9...` Track-A receipt, retained checkpoint, independently reviewed
candidate, host and immutable request. One separately human-approved component probe must then
prove key/table/logit identity and native distributed-update semantics; even a passing probe does
not authorize sparse-update training.

The Track-A launcher repair independently passed 21/21 tests and hostile descendant-process
fixtures. Its exact dual-host controls are verifier-finalized and pushed on
`experiment/20260924-v2-track-a-35c9-controls-a1` at `65155c66`; A1 itself remains locked until an
authenticated archived trn21 control is bound and re-reviewed. The exact `35c9...` source and
evaluation files remain unchanged.

The completed review-clean trn22 control produced an authentic observational 20M measurement of
`0.992968300659387` at 3,631 steps, but it is `INVALID_CONTROL_FOR_ADVANCEMENT`: postflight free
space was 5,928,337,408 bytes, below the 6.5-GiB operational floor. Its evidence is pushed at
`68ecfeb1`, with Analyst verdict `59f4d8ab`. No trn22 candidate may compare against that run. A new
unique control is required after separately authorized cleanup, and the observed footprint implies
a conservative pre-staging floor of 8,833,482,752 bytes if the host must finish above 6.5 GiB.

The previously launched Trn22 `116188cf...` current-base control remains valid submission-safety
work and is allowed to finish; it is not mislabeled as a `35c9...` Track-A control. The old LR0125
and embedding-only harness work is parked unlaunched. The legacy autonomous fixed queue remains
inactive until the v2 proposals, independent receipts, and exact host controls replace it.

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
- At 09:07 UTC both hosts remained idle and unheld. The Operator reclaimed 614,690,816 allocated
  bytes on `trn22` by deleting three exact, archived controller worktrees; it now has 7.4873 GiB
  free. `trn21` remains untouched at 6.8574 GiB because its LR0.015 checkpoint explicitly forbids
  pruning. The final-week operational gate is 6.5 GiB, still 1.5 GiB above the unchanged hard
  approximately-5 GiB protocol floor. See
  `operations/TRN2_FINAL_WEEK_GATE_CLEANUP_2026-09-24T090757Z.json`.
- Final-week selection uses context-1024 public BPB. A 2,097,152-token fresh-process evaluation may
  screen an arm, but cannot establish a KEEP. Promotion requires a fresh 20,971,520-token evaluation;
  adoption requires two negative 20M deltas with mean at most `-0.0008` against current-base,
  same-host controls. Submission and instance lifecycle actions remain human-only.
- The next host work is deliberately asymmetric: `trn22` needs a fresh current-base 2M+20M control;
  `trn21` first receives evaluator-only 2M calibration of its two retained exact-base checkpoints,
  then MATRIX_LR 0.0125. This replaces a redundant third `trn21` control training run. The next
  bracket is MATRIX_LR 0.0175; embedding 0.45 ranks ahead of 0.20. The 1080-second diagnostic,
  unembedding, scalar LR, and softcap are outside the immediate critical path.

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
