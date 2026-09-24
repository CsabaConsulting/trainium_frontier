# Start here: running the autonomous workflow with Codex

**Fastest start:** paste `workflow/CODEX_PROMPT.md` (everything below its line) into the first Codex
session. It walks Codex through the setup, gates, run loop and submission track below.

## One-time setup (≈30 min, human)
1. Copy `workflow/` into the research repo root, and copy `workflow/AGENTS.md` to `AGENTS.md` at the repo
   root.
2. Create `research/v2/`. Seed `research/v2/queue.json` from `reports/2026-09-24-queue-v2.json`, and
   create an empty `research/v2/ledger.jsonl` and `research/v2/state.json`.
3. On the controller workstation, run `pip install torch pyarrow numpy` (CPU is fine) for the G2 preflight.
   Point G2 at the **real** `prepare.py` (default: the directory of `train.py`). `--stub-prepare` is only for
   machines without it.
4. Give only the Operator session SSH access to trn21 and trn22 (per-user keys, not the shared PEM). Every
   other session has repo access only.
5. Decide the alert channel (email, Slack or phone) and put its command in `research/v2/state.json`
   (`alert_cmd`). Never commit a secret. Reference an environment variable instead.

## Sessions (start each as a separate Codex task in the repo; paste the prompt)
| Session | Prompt |
|---|---|
| **sup** | "You are the Supervisor. Follow AGENTS.md and workflow/roles/supervisor.md. Run the cycle continuously; keep trn21 and trn22 busy; never override decide.py; alert me on AGENTS §9 events." |
| **op** | "You are the Operator. Follow AGENTS.md and workflow/roles/operator.md. Execute only jobs the Supervisor hands you, with full preflight and contained.py; write receipts; never edit code." |
| **res** | "You are the Researcher. Follow AGENTS.md §7 and workflow/roles/researcher.md. Keep research/v2/backlog.md ranked by EV per host-hour and write the next proposals." |
| **ver** | "You are the Verifier. Follow workflow/roles/verifier.md. Re-run every check yourself. Default to refuse until the evidence is complete." |
| **impl-<label>** | "You are the Implementer for proposal <label>. Follow workflow/roles/implementer.md. One causal change behind a flag; self-check with preflight_dryrun.py; do not launch." |
| **ana** | "You are the Analyst. Follow workflow/roles/analyst.md. Annotate verdicts, run evaluator-only diagnostics, write the six-hour summaries." |
| **port** | "You are the Porter. Follow workflow/roles/porter.md. First job: S0 (numerics-identical cleanup of 116188)." |

## Files the sessions use
- Research file for every Track-A arm: `workflow/proposals/research-v2/train.py` (SHA-256 `c5c410af…`).
  Deploy it at `submissions/research_v2/train.py` next to the organizer `prepare.py`.
- Submission candidate S0: `workflow/candidates/S0/` (bare launch command; see its `EVIDENCE.md`).
- Host runner: `workflow/tools/run_job.py`. Copy `workflow/templates/host.json` to
  `research/v2/host.json` on each host, set `host`, `archive_dir` and the G1-verified `eval_args`, then
  set `eval_args_verified: true`.

## First 6 hours (what should happen without you)
1. **op:** G1 receipt (prepare.py facts, shard counts). Then A0 controls on both hosts.
2. **ver:** verify IMPL-1 (`workflow/proposals/IMPL-1/`). **op:** Neuron smoke on the IMPL-1 bytes.
3. **sup:** D1 epoch-value run, then flag-bank arms while IMPL-1 is gated. After that, A2 on trn22 and A1
   on trn21.
4. **port:** S0 cleanup, then `--compare` identical, then the Verifier audit, then two cold replays queued.
5. **res:** proposals for IMPL-2 (epoch-2 table freeze), IMPL-3 (fresh tail) and A-eps, ready for the table
   verdict.

## You (human) are needed for
- ADOPT acknowledgements;
- submissions (S0 around Sep 25 midday IST if it passes);
- breaker resets;
- the EBS resize (strongly recommended);
- confirming the quota rules with the organizers.
