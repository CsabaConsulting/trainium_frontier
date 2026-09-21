# Start here — shortest path to an operating research loop

Status at 2026-09-21: the local controller, clean latest-research worktree, prompts, state, logging,
and safety boundaries are installed. The dry run passes. The intended Bedrock route is now known:
source Region `us-west-2` with `us.*` inference profiles. Calls reach model validation after removing
unsupported sampling fields, but a successful health response is still required before a cycle.

## Current external gates

### 1. Prove one US Bedrock profile call

Do not use `eu-north-1` with `global.*` for this controller. That route exposed zero applied quotas
and returned account-access errors. Csaba's existing Claude Code configuration uses the US route:

- source Region: `us-west-2`
- Fable: `us.anthropic.claude-fable-5-1`
- Astra: `us.openai.gpt-6-astra`

Both US profiles are listed `ACTIVE`, and requests reach the models. Fable/Astra reject the generic
`temperature` field, so the runner omits it for those families. This exact command must still print
`Model invocation: OK` before daemon mode:

```bash
AWS_PROFILE=trainium-agent AWS_REGION=us-west-2 \
  .agent-venv/bin/python research/agentic/bedrock_team.py doctor \
  --model-id '<inference-profile-id>'
```

Historical clarification: read-only Service Quotas inspection found these values on the discarded
Stockholm/global route:

| Required quota | Current value | Adjustable | Quota code |
|---|---:|---|---|
| Global cross-region tokens/minute — Claude Fable 5.1 | 0 | Yes | `L-50CC5674` |
| Global cross-region tokens/minute — GPT-6 Astra | 0 | Yes | `L-B38B530A` |
| Global cross-region tokens/minute — Amazon Nova 2 Lite | 0 | Yes | `L-71C69B70` |
| Cross-region tokens/minute — Amazon Nova Micro | 0 | Yes | `L-DC7FF66C` |
| Model tokens/day — Amazon Nova 2 Lite | 0 | No | `L-210172B5` |
| Model tokens/day — Amazon Nova Micro | 0 | No | `L-D2912E70` |

Those zero values do not establish that the US profiles are blocked. Do not submit more quota
requests until the US health call is tested again after any transient service error. The Anthropic
first-time-use form already exists for this account; do not resubmit or disclose its contents.

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
export AWS_REGION=us-west-2
export BEDROCK_RESEARCHER_MODEL_ID='us.anthropic.claude-fable-5-1'
export BEDROCK_SUPERVISOR_MODEL_ID='us.openai.gpt-6-astra'

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
