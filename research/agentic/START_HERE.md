# Start here — shortest path to an operating research loop

Status at 2026-09-21: the local controller, clean latest-research worktree, prompts, state, logging,
and safety boundaries are installed. The dry run passes. A live cycle stops at Bedrock invocation
because the AWS account cannot currently invoke the requested models.

## Two external blockers

### 1. Make at least one Bedrock text model callable

The account can authenticate and list 43 models. Live tests returned:

- `global.openai.gpt-6-astra`: account access denied;
- `global.anthropic.claude-fable-5-1`: account access denied;
- Nova Micro, Nova 2 Lite, and Mistral Pixtral fallback profiles: daily-token throttle.

The AWS account owner must resolve model entitlement and/or Bedrock daily token quota. Catalog or
inference-profile status `ACTIVE` is not sufficient; this exact command must print
`Model invocation: OK`:

```bash
AWS_PROFILE=trainium-agent AWS_REGION=eu-north-1 \
  .agent-venv/bin/python research/agentic/bedrock_team.py doctor \
  --model-id '<inference-profile-id>'
```

One callable strong model is enough to start. Two different callable families are preferred for
independent research and review. Do not start daemon mode until both configured IDs pass `doctor`.

### 2. Reconcile the official submissions

For the accepted `1.0246` row and the rejected row, capture the submission Details page or its full
text, including submission ID, timestamp, file hash/name, status, score, and reviewer explanation.
Do not share passwords, AWS credentials, private keys, session cookies, or tokens.

This is necessary because repository public BPB and official leaderboard BPB are different evidence.

## Start commands after Bedrock is fixed

```bash
cd /home/poojith-devan/Desktop/Trainium
export AWS_PROFILE=trainium-agent
export AWS_REGION=eu-north-1
export BEDROCK_RESEARCHER_MODEL_ID='<callable-profile-id>'
export BEDROCK_SUPERVISOR_MODEL_ID='<callable-profile-id>'

.agent-venv/bin/python research/agentic/agentctl.py status
.agent-venv/bin/python research/agentic/agentctl.py cycle
```

The first cycle performs source197 offline attribution. It cannot edit files, SSH, start training,
mutate AWS, merge, or submit. Its transcripts and state are logged locally.

## What becomes autonomous next

After the first report is reviewed:

1. Researcher and Supervisor finish the source197 attribution plan.
2. A proposal is written with one causal change and explicit gates.
3. After human implementation approval, an Implementer works in an isolated experiment worktree.
4. A different Verifier checks the diff and numerical/casual contracts.
5. After human run approval, a deterministic Operator—not an LLM—may execute the exact manifest.
6. Logs, hashes, metrics, comparison, failure state, and verdict are written back to the repository.
7. Only a candidate with comparable lower BPB becomes a frozen one-file `train.py` submission.
8. Leaderboard submission remains a human action.

Unattended research mode is deliberately later:

```bash
.agent-venv/bin/python research/agentic/agentctl.py daemon --interval-minutes 30
```

Do not use it until model access is stable and several manual cycles produce useful, non-duplicate
reports. Create `research/agentic/STOP` at any time to block the next cycle.

## First technical queue

1. Offline source197 dispatch/NEFF attribution.
2. Conditional complete saved ROW-FP8 backward implementation.
3. Numerical/cache/evaluation and complete-step gates.
4. Require a repeatable whole-step gain of at least 3%.
5. Human-approved 30-minute seed-42 quality run.
6. Matched public evaluation and independent replay.
7. Freeze and verify one-file `train.py`.
8. Human leaderboard submission and result capture.

Broad parameter sweeps, generic autocast FP8, and unprofiled NKI work are not in this queue because
the repository already contains measured failures in those directions.
