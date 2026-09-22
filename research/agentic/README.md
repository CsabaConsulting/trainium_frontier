# Bedrock agent starting line

This folder is the coordination surface for AI agents. It deliberately sits on top of the existing
`research/run_experiment.py` and `research/experiment-log.jsonl` instead of replacing them.

## Recommended short-deadline architecture

Use a supervisor prompt with specialist calls through the Amazon Bedrock Runtime Converse API.
Converse keeps the invocation interface model-independent. Store durable truth here and in git;
do not rely on model conversation memory as the experiment database.

Roles:

1. **Supervisor** — reads `CURRENT_STATE.md`, chooses the next gate, and delegates.
2. **Researcher** — mines the existing 350+ experiments and primary sources for one new hypothesis.
3. **Implementer** — creates a narrow patch in a dedicated branch/worktree.
4. **Verifier** — checks rules, numerical correctness, causality, and the diff.
5. **Operator** — launches only a human-approved proposal on an assigned Trn2 box.
6. **Analyst** — compares the result with a matched control and writes the verdict.

For the remaining Phase 1 window, avoid building Lambda action groups or giving a model broad AWS
permissions. Agents should produce proposals and patches. A human should approve machine launches
and leaderboard submissions.

## Workflow

```text
CURRENT_STATE -> research proposal -> human approval -> isolated implementation
              -> independent verification -> human run approval -> operator
              -> artifact retrieval -> analyst verdict -> DECISION_LOG/CURRENT_STATE
```

## Files

- `CURRENT_STATE.md`: compact, authoritative handoff; update after every accepted result.
- `DECISION_LOG.md`: append-only index of tested/closed directions.
- `EXPERIMENT_PROPOSAL.json`: template and machine-readable run gate.
- `prompts/`: reusable role prompts for Bedrock models.

## AWS permissions

The research/coding identity should normally need only Bedrock inference permissions and repository
access. Do not give it EC2 mutation, IAM, billing, Secrets Manager read, or leaderboard credentials.
The machine operator should use a separate identity and the minimum access needed for the existing
instances. SSH should use each teammate's own key.

## Model policy

Keep model IDs in deployment configuration, not in prompts or source files. Use the strongest
available reasoning model for supervisor/research/review, and a faster model for summarization and
log extraction. Evaluate roles on a small fixed task set before switching models.

## First run: read-only Bedrock agent

Create the isolated local environment:

```bash
uv venv .agent-venv
uv pip install --python .agent-venv/bin/python -r research/agentic/requirements.txt
```

After AWS authentication is configured, verify access and discover model IDs:

```bash
.agent-venv/bin/python research/agentic/bedrock_team.py doctor
.agent-venv/bin/python research/agentic/bedrock_team.py models
```

Select a visible model or inference-profile ID, then run the first reconciliation task:

```bash
export AWS_PROFILE=trainium-agent
export AWS_REGION=us-west-2
export BEDROCK_MODEL_ID='<model-or-inference-profile-id>'
.agent-venv/bin/python research/agentic/bedrock_team.py ask \
  --role supervisor \
  --task 'Reconcile the current state and specify the single next evidence-gathering action.' \
  --include research/audit-sub1-2026-09-16.md
```

The runner archives responses under the ignored `research/agentic/runs/` directory. It exposes no
tools that can execute code, edit files, use SSH, mutate AWS, or submit to the leaderboard.

## Unattended flag-only experiment loop

`autoloop.py` (package `autoloop/`) is the restart-safe controller authorized by
`STANDING_AUTONOMOUS_AUTHORIZATION.md`: typed SSH operator, immutable per-run clones of the
authoritative commit, independent verifier gate, matched controls, artifact retrieval, and
git archiving before the next launch. Read `AUTOLOOP.md` first; it lists the human checklist
and the known limitations. Mocked tests: `python -m unittest discover -s research/agentic/tests`.

## One-command controller

The deterministic controller packages the latest fresh-start handoff for a Researcher and then has
a Supervisor review that report. It cannot edit code or reach the Trn2 boxes.

```bash
export AWS_PROFILE=trainium-agent
export AWS_REGION=us-west-2
export BEDROCK_RESEARCHER_MODEL_ID='us.anthropic.claude-fable-5-1'
export BEDROCK_SUPERVISOR_MODEL_ID='us.openai.gpt-6-astra'

.agent-venv/bin/python research/agentic/agentctl.py bootstrap
.agent-venv/bin/python research/agentic/agentctl.py cycle --dry-run
.agent-venv/bin/python research/agentic/agentctl.py cycle
```

Only after repeated cycles are useful should unattended read-only mode be used:

```bash
.agent-venv/bin/python research/agentic/agentctl.py daemon --interval-minutes 30
```

Create `research/agentic/STOP` to prevent the next cycle. The daemon stops on any model, state, or
readiness failure; it does not silently switch models. Runtime state and full model transcripts are
ignored locally. Human-approved experiment evidence remains durable in the repository.
