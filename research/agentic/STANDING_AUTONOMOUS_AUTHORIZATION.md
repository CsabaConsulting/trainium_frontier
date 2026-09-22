# Standing autonomous experiment authorization

Recorded 2026-09-22 from the human operator's active conversation. The operator explicitly
requested an unattended Bedrock-driven research loop that may test its proposals on the two
existing Trn2 EC2 instances and must return all work and results to this repository. This
supersedes the earlier per-run approval requirement for development experiments only.

Scope:

- Bedrock may research and propose any competition-legal single-factor development experiment.
- A deterministic Operator may launch a run on an existing Trn2 instance after an independently
  verified, complete proposal and a fresh pre-run protocol check. The proposal ID, exact commit,
  label, box, launch argument vector, environment, and abort criteria must be frozen before launch.
- The Supervisor/Researcher, Implementer, Verifier, Operator, and Analyst boundaries in `AGENTS.md`
  still apply; an LLM must never receive an arbitrary SSH/shell tool or the private key.
- The Operator must retrieve full logs and result rows, record failures as well as successes, and
  commit durable experiment evidence before the next run on that box.
- One training process per box; no lifecycle changes, deletion of existing data, forced Git
  operations, private validation access, or leaderboard submission.
- `research/agentic/STOP` immediately prevents new launches. Running training finishes normally.

This standing authorization does not mark any unverified proposal's gates as passed. It does not
authorize changing competition rules, modifying `prepare.py`, or merging the submission candidate.
