#!/usr/bin/env python3
"""Zero-model-token alert for an inactive or circuit-broken autonomous loop."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = ROOT / "research" / "agentic" / "runtime" / "autoloop_state.json"
STOP_FILE = ROOT / "research" / "agentic" / "STOP"
SERVICE = "trainium-autoloop.service"
TITLE = "[Trainium blocker] autonomous loop requires attention"


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, check=False)


def blocker_reason() -> str | None:
    active = run(["systemctl", "--user", "is-active", SERVICE])
    if active.stdout.strip() != "active":
        if STOP_FILE.exists():
            return None
        return f"{SERVICE} is {active.stdout.strip() or 'inactive'}"

    try:
        state = json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"cannot read autonomous state: {exc}"
    breaker = state.get("breaker")
    if breaker:
        return "controller circuit breaker is open: " + json.dumps(breaker, sort_keys=True)[:1500]
    return None


def notify(reason: str) -> None:
    repo = os.environ.get("TRAINIUM_ALERT_REPOSITORY", "CsabaConsulting/trainium_frontier")
    assignee = os.environ.get("TRAINIUM_ALERT_ASSIGNEE", "poojithdevan4D")
    existing = run([
        "gh", "issue", "list", "--repo", repo, "--state", "open",
        "--search", f'"{TITLE}" in:title', "--json", "number", "--jq", ".[0].number // empty",
    ])
    if existing.returncode:
        raise RuntimeError(existing.stderr.strip() or "GitHub blocker lookup failed")
    if existing.stdout.strip():
        return
    body = (
        "The local zero-token watchdog found a condition requiring human attention.\n\n"
        f"- Reason: `{reason}`\n"
        f"- Service: `{SERVICE}`\n"
        "- State file: `research/agentic/runtime/autoloop_state.json` (local runtime; not committed)\n\n"
        "No AWS lifecycle action or leaderboard submission was attempted. Review the local service "
        "and durable experiment evidence before restarting or changing worker state."
    )
    created = run([
        "gh", "issue", "create", "--repo", repo, "--title", TITLE,
        "--body", body, "--assignee", assignee,
    ])
    if created.returncode:
        raise RuntimeError(created.stderr.strip() or "GitHub blocker issue creation failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="report health without notifying")
    args = parser.parse_args()
    reason = blocker_reason()
    if reason is None:
        print("OK: autonomous loop active; breaker closed")
        return 0
    print(f"BLOCKED: {reason}")
    if args.check_only:
        return 2
    notify(reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
