"""Unattended, restart-safe, flag-only experiment controller for the Trn2 boxes.

Package layout:

- config.py     pinned hosts, key path, commit, shapes, caps (deterministic constants + JSON overrides)
- ledger.py     tolerant parsing of research/experiment-log.jsonl (local and retrieved remote copies)
- manifest.py   Proposal / RunManifest schemas and every launch-shape validation rule
- models.py     minimal Bedrock Converse client behind a Protocol so tests can inject fakes
- verifier.py   deterministic gate + independent LLM verifier (different model family)
- remote.py     typed SSH/scp operator: inventory, stage, launch, poll, collect; argv guard
- state.py      atomic state file, per-proposal experiment directories, restart reconciliation
- analyst.py    deterministic matched-control comparison (+ optional LLM prose)
- archive.py    git add/commit of declared evidence only; never force, never rewrite
- controller.py the event loop that wires the above and enforces the circuit breakers

Nothing in this package can submit to the leaderboard, call the EC2 API, read the SSH private key
contents, or hand a model a shell.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
