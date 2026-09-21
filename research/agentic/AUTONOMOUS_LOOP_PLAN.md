# Autonomous Bedrock research loop — implementation plan

Status: design approved for implementation planning; no hardware authority granted.

## 1. Goal and boundary

Continuously find and test legal changes that minimize official Phase 1 `val_bpb` after a
30-minute full-chip Trn2 training run. The system may autonomously research, deduplicate ideas,
prepare proposals, implement isolated patches, verify them, retrieve artifacts, and analyze
results. Under the repository contract, it may launch only an exact experiment or conditional
decision tree that a human has approved in advance. It may never submit to the leaderboard.

The optimizer target is final comparable BPB, not steps/s. Throughput is an intermediate metric.

## 2. Deployment shape

Run the control plane on a machine that stays awake and is not a Trainium worker. For the current
deadline, this can be the laptop; for true 24/7 operation, use a small dedicated controller after a
human provisions it. Do not run the orchestrator on either Trn2 box because agent activity can
contaminate timing, memory, disk, and process state.

Components:

- **Bedrock Runtime**: model inference only, through configurable inference-profile IDs.
- **Controller**: deterministic Python state machine; models cannot bypass its gates.
- **Repository**: durable source of truth, proposals, decisions, summaries, and reviewed patches.
- **Two Trn2 workers**: execute allowlisted experiment manifests, one process per box.
- **GitHub private repository**: collaboration and durable history.
- **Optional Tavily**: primary-source discovery for the Researcher only; never supplies training
  data and never becomes available to the Operator.

Prefer direct Bedrock Runtime + a small local orchestrator for Phase 1. AgentCore/Step Functions
can be a Phase 2 hardening step; building production cloud orchestration now would consume the
competition window.

## 3. Model assignments

Model IDs are configuration, not repository constants. Catalog visibility is not proof that an
account can invoke a model; `doctor` must successfully call every configured model.

| Role | Preferred class | Job | Tools |
|---|---|---|---|
| Supervisor | strongest reasoning model (target: GPT-6 Astra) | portfolio, gate selection, delegation | read state; call specialists |
| Researcher | strongest long-horizon model (target: Fable 5/5.1) | repository gap analysis and primary-source research | repository search; Tavily; no writes |
| Implementer | strong coding model | one approved patch in one worktree | scoped file read/write; tests; no SSH |
| Verifier | different model family from Implementer | rules, diff, numerical and causal checks | read-only diff/tests; no candidate edits |
| Operator | deterministic code, not an LLM | inventory, sync, launch, collect | allowlisted remote wrapper only |
| Analyst | reasoning model, preferably different from Implementer | matched-control verdict | logs/ledger read; decision draft |
| Summarizer | low-cost fast model | compress logs and research | read-only |

Do not make a frontier model perform mechanical polling or log copying. Deterministic code does
that more reliably and preserves Bedrock quota.

## 4. State machine

Every experiment has one immutable `proposal_id` and moves only forward:

```text
idea
  -> deduplicated
  -> proposal_ready
  -> human_implementation_approved
  -> implemented
  -> independently_verified
  -> human_run_approved
  -> smoke_running -> smoke_failed | smoke_passed
  -> human_full_run_approved
  -> full_running -> failed | collected
  -> analyzed
  -> kept | discarded | repeat | confounded
  -> merged_or_closed
```

No LLM writes state directly. It returns structured output; the controller validates the schema
and performs the transition. Invalid or missing evidence leaves the state unchanged.

## 5. Durable experiment record

Create one directory per proposal:

```text
research/agentic/experiments/<proposal_id>/
  proposal.json
  approvals.json
  implementation.json
  verification.json
  run-manifest.json
  result.json
  verdict.md
```

`run-manifest.json` freezes:

- proposal and control IDs
- full Git commit hashes
- permitted changed files
- unique run label
- exact launch command as an argument vector, never a shell string
- box assignment
- seed 42
- `NEURON_LOGICAL_NC_CONFIG`, `NEURON_CC_FLAGS`, and relevant environment
- time limit, expected output directory, and success/abort gates
- approval identity and timestamp

Raw model transcripts go to ignored `research/agentic/runs/`. Essential experiment evidence goes
to git. Checkpoints remain untracked; record hashes and paths only.

## 6. Duplicate prevention and research memory

Before proposing anything, the Researcher must query:

1. `research/experiment-log.jsonl`
2. `research/trn2-experiment-results.md`
3. `research/agentic/DECISION_LOG.md`
4. active proposal directories
5. aliases/flags in `train.py`

Build a generated idea registry with normalized fields:

```text
technique | aliases | flags | commits | run labels | hardware | budget | result | verdict | reason
```

The generated registry is searchable but never overrides raw evidence. An idea can be reopened
only when the proposal names the changed condition or new evidence that makes the old result
non-comparable.

## 7. Remote execution boundary

The Bedrock models never receive a general shell or the SSH private key. The controller exposes
only these typed operations:

- `inventory(box)` — process, disk, device-holder, commit, dirty tree, and environment checks
- `stage_candidate(box, commit, allowed_files)` — copies a reviewed candidate and verifies hashes
- `launch_approved(proposal_id)` — loads the frozen manifest and refuses any unapproved value
- `poll(proposal_id)` — reads PID/status and the end of the designated log
- `collect(proposal_id)` — copies only declared logs/results and verifies hashes

The wrapper must reject arbitrary commands, unknown boxes, dirty candidates, reused labels,
unapproved manifests, missing `--no-commit`, wrong seed, unsafe paths, occupied devices, low disk,
and commands outside competition launch shapes. Use a per-box lock so two controllers cannot run
at once.

Do not give the runtime permission to start, stop, resize, terminate, or reprovision EC2.

## 8. Experiment ladder

An experiment earns expensive compute in stages:

1. **Static/CPU check** — import, parser, unit and shape checks where meaningful.
2. **Numerical check** — candidate versus eager/control implementation.
3. **Causality/stateless check** — required for any forward-path change.
4. **Compile/smoke** — cheapest full-chip duration that answers correctness and gross throughput.
5. **Paired intermediate run** — same box/day control when a proxy-quality comparison is needed.
6. **Full 1,800-second confirmation** — only for a promising candidate.
7. **Independent reproduction** — required before consuming a leaderboard submission on a fragile
   or noisy improvement.

Predeclare all thresholds. A short-run BPB win alone does not establish a full-budget win.

## 9. Scheduler and all-night operation

The controller runs a small event loop:

1. Reconcile repository and worker inventory.
2. Stop if any global circuit breaker is open.
3. Keep Researcher/Supervisor work ahead of hardware by maintaining a small ranked queue.
4. Assign at most one approved manifest to each idle box.
5. Poll deterministically; do not spend LLM tokens on waiting.
6. Collect outputs immediately when a run ends.
7. Invoke Analyst once per completed pair, update proposed verdict, and wait for merge approval.
8. Launch the next manifest only if its pre-approved dependency conditions are satisfied.

An overnight batch approval must enumerate exact manifests and conditions, for example: “Run smoke
A; if it passes all correctness gates and step rate is >= control - 2%, run full A; otherwise stop
that branch and run smoke B.” It is not blanket permission for an agent to invent more runs.

## 10. Circuit breakers

Stop new work automatically when any condition is true:

- Bedrock daily token or spending cap reached
- Git tree/commit mismatch or merge conflict
- unknown process holds a Neuron device
- disk below 5 GiB
- duplicate label or proposal
- crash, NaN, causality failure, missing checkpoint, or over-budget run
- controller cannot retrieve the complete log
- box has two consecutive infrastructure failures
- repository state cannot be pushed safely
- configured competition cutoff time reached
- human creates `research/agentic/STOP`

Already-running training is observed and allowed to finish unless a human explicitly intervenes.

## 11. Repository and Git workflow

- One worktree and branch per proposal: `experiment/<proposal_id>`.
- Implementer may touch only the proposal's allowlist.
- Verifier operates from the diff and control commit, never from prose alone.
- Operator runs an immutable verified commit, not a dirty directory.
- Result collection uses a separate result commit containing only declared evidence.
- Supervisor proposes a merge; a human approves it under the current contract.
- Never auto-force-push, rewrite history, delete branches, or clean unknown files.

## 12. Bedrock access and quota gates

The current AWS browser login and catalog read work. Actual model invocation is not yet ready:

- available serverless models returned a daily-token throttling error;
- GPT-5.6 Luna and Fable 5.1 catalog IDs were visible but not enabled for direct invocation;
- inference-profile IDs must be tested, not assumed.

Before daemon mode:

1. Confirm promotional-credit balance and Bedrock billing owner.
2. Enable or request access to the exact inference profiles.
3. Request appropriate daily token/service quotas.
4. Prove one 16-token call per configured model with `bedrock_team.py doctor`.
5. Set per-role max tokens, calls/hour, calls/day, and daily budget.
6. Configure fallbacks so quota exhaustion pauses safely instead of silently switching to an
   inappropriate model.

## 13. Implementation order

### Milestone A — reconcile truth

- Resolve the exact `1.0246` commit/config and rejected-review reason.
- Read-only inventory both Trn2 boxes.
- Merge all current experiment logs and produce the idea registry.
- Do not queue technical experiments before these are complete.

### Milestone B — read-only autonomous research

- Add structured model output and schema validation to `bedrock_team.py`.
- Add repository search and primary-source/Tavily tools to Researcher only.
- Add Supervisor delegation and archive every call with token usage.
- Run for several cycles without code or hardware access; audit duplicate rate and usefulness.

### Milestone C — isolated implementation and verification

- Add worktree manager and scoped patch tool.
- Add deterministic CPU/static test runner.
- Enforce separate model families for implementation and review where available.
- Require human approval before merging or remote staging.

### Milestone D — remote smoke execution

- Add typed inventory/stage/launch/poll/collect wrappers.
- Test on a no-training dry run, then one human-approved smoke.
- Simulate all circuit breakers and recovery paths.

### Milestone E — overnight approved queue

- Prepare a ranked batch of exact proposals/manifests.
- Human approves the manifests and conditional decision tree.
- Run one supervised evening rehearsal.
- Enable the controller overnight with a hard stop time and morning report.

## 14. Definition of production-ready

The loop is ready for unattended approved work only after it demonstrates:

- zero duplicate launches across a simulated restart;
- recovery from model throttling without losing state;
- refusal of unapproved and malformed manifests;
- per-box mutual exclusion;
- exact commit/environment/log provenance;
- complete result retrieval after a controller restart;
- independent verification cannot be bypassed;
- STOP file halts new launches;
- no AWS lifecycle or leaderboard capability exists in the agent runtime.
