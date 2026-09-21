#!/usr/bin/env python3
"""Deterministic control plane for read-only Bedrock research cycles.

This controller intentionally has no SSH, EC2 mutation, training, git-write, merge, or leaderboard
capability. It packages the latest handoff for a Researcher and asks a Supervisor to select the next
evidence gate. Every model call is archived by bedrock_team.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "research" / "agentic"
RUNTIME_DIR = AGENT_DIR / "runtime"
STATE_FILE = RUNTIME_DIR / "state.json"
STOP_FILE = AGENT_DIR / "STOP"
PYTHON = ROOT / ".agent-venv" / "bin" / "python"
TEAM = AGENT_DIR / "bedrock_team.py"
FRESH = ROOT / ".worktrees" / "fresh-start"
EXPECTED_FRESH_COMMIT = "9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a"

RESEARCH_CONTEXT = [
    "research/agentic/RESEARCH_HANDOFF_AUDIT_2026-09-21.md",
    ".worktrees/fresh-start/research/RESUME.md",
    ".worktrees/fresh-start/research/HANDOFF-CLAUDE-2026-09-20.md",
    ".worktrees/fresh-start/research/fresh-start-library-followup-2026-09-20.md",
    ".worktrees/fresh-start/research/fresh-start-library-fusion-next-2026-09-20.md",
]

FIRST_TASK = (
    "Perform the next read-only research gate: analyze the recorded source197 first-layer versus "
    "last-layer evidence and specify exactly what offline dispatch/NEFF attribution must be completed "
    "before deciding whether to implement saved ROW-FP8 backward. Do not propose a training run, "
    "edit, or AWS action. Reconcile every suggestion against the closed directions and cite exact "
    "repository evidence. Return at most three ranked findings and one finite deliverable."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_git(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def checks() -> list[dict]:
    rows: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        rows.append({"name": name, "ok": ok, "detail": detail})

    add("python", PYTHON.is_file(), str(PYTHON))
    add("bedrock_runner", TEAM.is_file(), str(TEAM))
    add("stop_file_absent", not STOP_FILE.exists(), str(STOP_FILE))
    add("fresh_worktree", FRESH.is_dir(), str(FRESH))
    if FRESH.is_dir():
        try:
            head = run_git("rev-parse", "HEAD", cwd=FRESH)
            add("fresh_commit", head == EXPECTED_FRESH_COMMIT, head)
            dirty = run_git("status", "--porcelain", cwd=FRESH)
            add("fresh_clean", not dirty, dirty or "clean")
        except RuntimeError as exc:
            add("fresh_git", False, str(exc))
    for relative in RESEARCH_CONTEXT:
        add(f"context:{relative}", (ROOT / relative).is_file(), relative)
    add("aws_profile", bool(os.getenv("AWS_PROFILE")), os.getenv("AWS_PROFILE", "not set"))
    add(
        "researcher_model",
        bool(os.getenv("BEDROCK_RESEARCHER_MODEL_ID") or os.getenv("BEDROCK_MODEL_ID")),
        os.getenv("BEDROCK_RESEARCHER_MODEL_ID") or os.getenv("BEDROCK_MODEL_ID", "not set"),
    )
    add(
        "supervisor_model",
        bool(os.getenv("BEDROCK_SUPERVISOR_MODEL_ID") or os.getenv("BEDROCK_MODEL_ID")),
        os.getenv("BEDROCK_SUPERVISOR_MODEL_ID") or os.getenv("BEDROCK_MODEL_ID", "not set"),
    )
    return rows


def save_state(state: dict) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE_FILE)


def load_state() -> dict:
    if STATE_FILE.is_file():
        return json.loads(STATE_FILE.read_text())
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "phase": "source197_offline_attribution",
        "cycle_count": 0,
        "last_cycle": None,
        "last_status": "initialized",
    }


def bootstrap(_args) -> int:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    (AGENT_DIR / "runs").mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        save_state(load_state())
    print(f"Runtime initialized: {STATE_FILE.relative_to(ROOT)}")
    return status(_args)


def status(_args) -> int:
    rows = checks()
    for row in rows:
        marker = "OK" if row["ok"] else "BLOCKED"
        print(f"{marker:7} {row['name']}: {row['detail']}")
    if STATE_FILE.is_file():
        state = load_state()
        print(
            f"STATE   phase={state['phase']} cycles={state['cycle_count']} "
            f"last_status={state['last_status']}"
        )
    return 0 if all(row["ok"] for row in rows) else 2


def invoke(role: str, model_id: str, task: str, includes: list[str]) -> subprocess.CompletedProcess:
    command = [
        str(PYTHON),
        str(TEAM),
        "ask",
        "--role",
        role,
        "--model-id",
        model_id,
        "--task",
        task,
    ]
    for relative in includes:
        command.extend(["--include", relative])
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def latest_role_record(role: str) -> Path | None:
    candidates = sorted((AGENT_DIR / "runs").glob(f"*-{role}.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_cycle(dry_run: bool = False) -> int:
    if STOP_FILE.exists():
        print(f"STOP is present: {STOP_FILE}", file=sys.stderr)
        return 3
    failed = [row for row in checks() if not row["ok"]]
    if failed:
        for row in failed:
            print(f"BLOCKED {row['name']}: {row['detail']}", file=sys.stderr)
        return 2

    researcher_model = os.getenv("BEDROCK_RESEARCHER_MODEL_ID") or os.environ["BEDROCK_MODEL_ID"]
    supervisor_model = os.getenv("BEDROCK_SUPERVISOR_MODEL_ID") or os.environ["BEDROCK_MODEL_ID"]
    if dry_run:
        print("Dry run passed. Planned cycle:")
        print(f"1. Researcher model: {researcher_model}")
        print(f"2. Task: {FIRST_TASK}")
        print("3. Supervisor reviews the archived Researcher response and selects one evidence gate.")
        print("4. No code edit, SSH, EC2, training, merge, or submission capability is exposed.")
        return 0

    state = load_state()
    state["last_cycle"] = {"started_at": utc_now(), "status": "researcher_running"}
    state["last_status"] = "running"
    save_state(state)

    researcher = invoke("researcher", researcher_model, FIRST_TASK, RESEARCH_CONTEXT)
    if researcher.returncode:
        state["last_cycle"].update(
            {"finished_at": utc_now(), "status": "researcher_failed", "error": researcher.stderr[-4000:]}
        )
        state["last_status"] = "model_blocked"
        save_state(state)
        print(researcher.stdout, end="")
        print(researcher.stderr, file=sys.stderr, end="")
        return researcher.returncode

    researcher_record = latest_role_record("researcher")
    if researcher_record is None:
        raise RuntimeError("Researcher succeeded but no archived record was found")
    relative_record = str(researcher_record.relative_to(ROOT))
    supervisor_task = (
        "Review the archived Researcher report for the source197 offline-attribution gate. Select "
        "one finite next evidence deliverable, reject unsupported or duplicate ideas, and state "
        "whether any human approval is needed. Do not authorize an experiment, edit, or AWS action."
    )
    supervisor = invoke(
        "supervisor",
        supervisor_model,
        supervisor_task,
        ["research/agentic/RESEARCH_HANDOFF_AUDIT_2026-09-21.md", relative_record],
    )
    if supervisor.returncode:
        state["last_cycle"].update(
            {"finished_at": utc_now(), "status": "supervisor_failed", "error": supervisor.stderr[-4000:]}
        )
        state["last_status"] = "model_blocked"
        save_state(state)
        print(supervisor.stdout, end="")
        print(supervisor.stderr, file=sys.stderr, end="")
        return supervisor.returncode

    supervisor_record = latest_role_record("supervisor")
    state["cycle_count"] += 1
    state["last_cycle"].update(
        {
            "finished_at": utc_now(),
            "status": "completed",
            "researcher_record": relative_record,
            "supervisor_record": str(supervisor_record.relative_to(ROOT)) if supervisor_record else None,
        }
    )
    state["last_status"] = "awaiting_human_review"
    save_state(state)
    print(researcher.stdout)
    print("\n--- SUPERVISOR ---\n")
    print(supervisor.stdout)
    print(f"\nCycle state: {STATE_FILE.relative_to(ROOT)}")
    return 0


def cycle(args) -> int:
    return run_cycle(dry_run=args.dry_run)


def daemon(args) -> int:
    completed = 0
    while args.max_cycles == 0 or completed < args.max_cycles:
        result = run_cycle(dry_run=False)
        if result:
            print("Cycle failed; daemon stops without retrying or changing models.", file=sys.stderr)
            return result
        completed += 1
        if args.max_cycles and completed >= args.max_cycles:
            break
        time.sleep(args.interval_minutes * 60)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("bootstrap", help="create ignored runtime directories and initial state")
    init.set_defaults(func=bootstrap)
    show = sub.add_parser("status", help="show every readiness gate")
    show.set_defaults(func=status)
    once = sub.add_parser("cycle", help="run one Researcher -> Supervisor cycle")
    once.add_argument("--dry-run", action="store_true")
    once.set_defaults(func=cycle)
    run = sub.add_parser("daemon", help="repeat read-only cycles; stop on any failure")
    run.add_argument("--interval-minutes", type=int, default=30)
    run.add_argument("--max-cycles", type=int, default=0, help="0 means unlimited")
    run.set_defaults(func=daemon)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        raise SystemExit(args.func(args))
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
