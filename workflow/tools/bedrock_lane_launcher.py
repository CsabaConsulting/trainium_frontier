#!/usr/bin/env python3
"""Launch isolated, budget-bounded Claude Code builder lanes on the local CPU box.

The Bedrock bearer token is parsed from a private dotenv file, never printed,
passed only in the child environment, and redacted from captured output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone


BASE = Path("/home/poojith-devan/Documents/ChatGPT/trainium-bedrock-lanes-20260925")
CONTROLLER = Path("/home/poojith-devan/Desktop/Trainium")
ENV_FILE = CONTROLLER / ".env"
PROMPT_FILE = CONTROLLER / "workflow/parallel_ai_lanes.md"
PROMPT_SHA256 = "a2dc88d4531f7fe9145da799930753664bc95a7587c681fe1673c1111396403c"
SOURCE_HEAD = "1f9e684eccd41638ac4bf0facf84d00fbe13705f"
CLAUDE_BINARY = Path("/home/poojith-devan/.local/share/claude/versions/2.1.282")
TORCH_VENV = Path("/home/poojith-devan/trtllm-vs-vllm-benchmark/.venv-vllm")
CREDENTIAL_FREE_SHELL = BASE / "credential_free_shell"
LANES = {
    "l1": {
        "dir": BASE / "l1-fp8",
        "home": BASE / "l1-home",
        "branch": "codex/bedrock-l1-fp8-20260925",
        "heading": "## Lane 1: Throughput (FP8 projections). The biggest proven lever on Trn2.",
        "budget_usd": 30.0,
    },
    "l2": {
        "dir": BASE / "l2-unet",
        "home": BASE / "l2-home",
        "branch": "codex/bedrock-l2-unet-20260925",
        "heading": "## Lane 2: The #3 team's architecture",
        "budget_usd": 30.0,
    },
}
LOG_DIR = BASE / "logs"
MAX_WALL_SECONDS = 6 * 3600


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_bedrock_env() -> tuple[dict[str, str], str, str]:
    info = ENV_FILE.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("private dotenv owner/mode check failed")
    selected: dict[str, str] = {}
    wanted = {"AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "BEDROCK_RESEARCHER_MODEL_ID"}
    for line_no, raw in enumerate(ENV_FILE.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        matched = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", raw)
        if not matched:
            raise RuntimeError(f"dotenv assignment syntax invalid at line {line_no}")
        key, value = matched.groups()
        if key not in wanted:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not value or "\x00" in value:
            raise RuntimeError(f"dotenv value invalid for {key}")
        selected[key] = value
    if set(selected) != wanted:
        raise RuntimeError("required Bedrock dotenv keys are missing")
    if selected["AWS_REGION"] != "us-west-2":
        raise RuntimeError("Bedrock region does not match authorized us-west-2")
    model = selected["BEDROCK_RESEARCHER_MODEL_ID"]
    if not model.startswith("us.anthropic.claude-"):
        raise RuntimeError("researcher model is not the expected Bedrock route")
    token = selected["AWS_BEARER_TOKEN_BEDROCK"]
    if len(token) < 50:
        raise RuntimeError("Bedrock bearer token is unexpectedly short")
    env = {
        "HOME": "/home/poojith-devan",
        "USER": "poojith-devan",
        "LOGNAME": "poojith-devan",
        "LANG": "C.UTF-8",
        "PATH": "/opt/torch/bin:/usr/local/bin:/usr/bin:/bin",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "AWS_BEARER_TOKEN_BEDROCK": token,
        "AWS_REGION": selected["AWS_REGION"],
        "AWS_DEFAULT_REGION": selected["AWS_REGION"],
        "AWS_EC2_METADATA_DISABLED": "true",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_AWS_CHAIN_RESOLVE_TIMEOUT_MS": "5000",
    }
    return env, token, model


def fenced_block_after(document: str, marker: str) -> str:
    marker_at = document.index(marker)
    open_at = document.index("\n```\n", marker_at) + len("\n```\n")
    close_at = document.index("\n```", open_at)
    return document[open_at : close_at + 1]


def lane_prompt(lane: str) -> str:
    raw = PROMPT_FILE.read_bytes()
    if sha256(raw) != PROMPT_SHA256:
        raise RuntimeError("committed prompt bytes changed; refusing to launch")
    document = raw.decode("utf-8")
    shared = fenced_block_after(document, "**Shared context (paste at the top of every lane prompt):**")
    specific = fenced_block_after(document, LANES[lane]["heading"])
    return shared + "\n" + specific


def verify_clone(lane: str) -> None:
    cfg = LANES[lane]
    directory: Path = cfg["dir"]
    if not directory.is_dir() or not cfg["home"].is_dir():
        raise RuntimeError("lane clone or isolated home is missing")
    for relative in (
        "workflow/proposals/research-v2/train.py",
        "05-train-submission-clean-116188.py",
        "workflow/tools/preflight_dryrun.py",
    ):
        if not (directory / relative).is_file():
            raise RuntimeError(f"verbatim prompt path missing: {relative}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=directory, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=directory, text=True).strip()
    if (head, branch, dirty) != (SOURCE_HEAD, cfg["branch"], ""):
        raise RuntimeError("lane clone is not the expected clean, pinned branch")
    if not CLAUDE_BINARY.is_file() or not TORCH_VENV.is_dir() or not CREDENTIAL_FREE_SHELL.is_file():
        raise RuntimeError("Claude Code binary, CPU torch environment, or credential-free shell is missing")


def sandbox_command(lane: str, model: str, prompt: str, *, probe: bool = False) -> list[str]:
    cfg = LANES[lane]
    command = [
        "/usr/bin/bwrap", "--unshare-all", "--share-net",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/etc", "/etc",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--dir", "/run",
        "--dir", "/run/systemd",
        "--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve",
        "--dev-bind", "/dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--dir", "/home",
        "--bind", str(cfg["home"]), "/home/poojith-devan",
        "--dir", "/work",
        "--bind", str(cfg["dir"]), "/work",
        "--dir", "/opt",
        "--ro-bind", str(TORCH_VENV), "/opt/torch",
        "--ro-bind", "/usr/bin/bash", "/opt/bash_real",
        "--ro-bind", str(CREDENTIAL_FREE_SHELL), "/usr/bin/bash",
        "--ro-bind", str(CREDENTIAL_FREE_SHELL), "/usr/bin/sh",
        "--ro-bind", str(CREDENTIAL_FREE_SHELL), "/usr/bin/dash",
        "--ro-bind", "/usr/bin/false", "/usr/bin/ssh",
        "--ro-bind", "/usr/bin/false", "/usr/bin/scp",
        "--ro-bind", "/usr/bin/false", "/usr/bin/sftp",
        "--ro-bind", str(CLAUDE_BINARY), "/usr/local/bin/claude",
        "--chdir", "/work",
        "/usr/local/bin/claude",
        "--bare", "--restricted", "--strict-mcp-config", "--no-chrome",
        "--permission-mode", "dontAsk" if probe else "auto",
        "--permission-prompts", "none",
        "--model", model,
        "--max-budget-usd", "0.20" if probe else str(cfg["budget_usd"]),
        "--max-turns", "1" if probe else "100",
        "--tools", "" if probe else "Read,Edit,Write,Glob,Grep,WebFetch,Bash",
        "--allowedTools", "Read,Edit,Write,Glob,Grep,WebFetch,Bash",
        "--append-system-prompt", "This is an isolated CPU builder. Never enumerate provider credentials or environment values, inspect process environments, access SSH hosts, or launch training. Work only in /work; /opt/torch/bin/python provides CPU PyTorch. Preserve the user prompt exactly as the task. Deliver the code patch and verification evidence in /work; do not push or mutate external services.",
        "--verbose",
        "-p", "--output-format", "json" if probe else "stream-json",
        prompt,
    ]
    return command


def probe() -> int:
    verify_clone("l1")
    env, token, model = load_bedrock_env()
    result = subprocess.run(
        sandbox_command("l1", model, "Reply with exactly: BEDROCK_BUILDER_READY", probe=True),
        cwd=LANES["l1"]["dir"], env=env, text=True, capture_output=True, timeout=150,
    )
    stdout = result.stdout.replace(token, "[REDACTED_BEDROCK_TOKEN]")
    stderr = result.stderr.replace(token, "[REDACTED_BEDROCK_TOKEN]")
    if result.returncode == 0:
        try:
            parsed = json.loads(stdout)
            print(json.dumps({
                "status": "PASS", "route": "Bedrock", "region": env["AWS_REGION"],
                "model": model, "result": parsed.get("result"),
                "cost_usd": parsed.get("total_cost_usd"),
            }))
        except json.JSONDecodeError:
            print(json.dumps({"status": "PASS_NON_JSON", "stdout_tail": stdout[-500:]}))
    else:
        print(json.dumps({
            "status": "FAIL", "exit_code": result.returncode,
            "stdout_tail": stdout[-1200:], "stderr_tail": stderr[-1200:],
        }))
    return result.returncode


def write_status(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def worker(lane: str) -> int:
    verify_clone(lane)
    prompt = lane_prompt(lane)
    env, token, model = load_bedrock_env()
    LOG_DIR.mkdir(mode=0o700, exist_ok=True)
    status_path = LOG_DIR / f"{lane}-status.json"
    log_path = LOG_DIR / f"{lane}-stream.jsonl"
    status = {
        "lane": lane,
        "state": "STARTING",
        "started_utc": utc_now(),
        "branch": LANES[lane]["branch"],
        "source_head": SOURCE_HEAD,
        "prompt_file_sha256": PROMPT_SHA256,
        "prompt_sha256": sha256(prompt.encode()),
        "budget_usd": LANES[lane]["budget_usd"],
        "max_turns": 100,
        "model": model,
        "sandbox": "bubblewrap; lane-only home and clone; no SSH/AWS credential files or controller mounted; allowed shell/edit tools; provider env stripped from shell children",
        "log_path": str(log_path),
        "worker_pid": os.getpid(),
    }
    write_status(status_path, status)
    command = sandbox_command(lane, model, prompt)
    with log_path.open("w", encoding="utf-8") as log:
        os.chmod(log_path, 0o600)
        process = subprocess.Popen(
            command, cwd=LANES[lane]["dir"], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        status.update(state="RUNNING", child_pid=process.pid, launch_utc=utc_now())
        write_status(status_path, status)
        timer = threading.Timer(MAX_WALL_SECONDS, process.terminate)
        timer.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                safe = line.replace(token, "[REDACTED_BEDROCK_TOKEN]")
                log.write(safe)
                log.flush()
                try:
                    event = json.loads(safe)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "system" and event.get("session_id"):
                    status["session_id"] = event["session_id"]
                    write_status(status_path, status)
                if event.get("type") == "assistant" and "first_response_utc" not in status:
                    status["first_response_utc"] = utc_now()
                    status["first_response_kind"] = [part.get("type") for part in event.get("message", {}).get("content", [])]
                    write_status(status_path, status)
                if event.get("type") == "result":
                    status["result_is_error"] = event.get("is_error")
                    status["reported_cost_usd"] = event.get("total_cost_usd")
                    status["result_preview"] = str(event.get("result", ""))[:600]
                    write_status(status_path, status)
            code = process.wait()
        finally:
            timer.cancel()
        status.update(state="COMPLETED" if code == 0 else "FAILED", exit_code=code, ended_utc=utc_now())
        write_status(status_path, status)
    return code


def start(lane: str) -> int:
    verify_clone(lane)
    lane_prompt(lane)
    load_bedrock_env()
    LOG_DIR.mkdir(mode=0o700, exist_ok=True)
    status_path = LOG_DIR / f"{lane}-status.json"
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("state") in {"STARTING", "RUNNING"}:
            raise RuntimeError(f"{lane} already has an active worker")
        raise RuntimeError(f"{lane} has a prior terminal record; explicit rotation is required")
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "worker", lane],
        cwd=LANES[lane]["dir"], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
        env={"HOME": "/home/poojith-devan", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    )
    for _ in range(30):
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") == "RUNNING":
                print(json.dumps({
                    "lane": lane, "state": "RUNNING", "worker_pid": process.pid,
                    "child_pid": status["child_pid"], "launch_utc": status["launch_utc"],
                    "branch": status["branch"], "log_path": status["log_path"],
                    "budget_usd": status["budget_usd"],
                }))
                return 0
        if process.poll() is not None:
            raise RuntimeError(f"{lane} worker exited during startup with code {process.returncode}")
        time.sleep(0.2)
    raise RuntimeError(f"{lane} worker did not report launch within 6 seconds")


def show_status(lane: str) -> int:
    path = LOG_DIR / f"{lane}-status.json"
    if not path.is_file():
        raise RuntimeError(f"no {lane} status yet")
    print(path.read_text(encoding="utf-8"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("probe", "start", "worker", "status"))
    parser.add_argument("lane", choices=tuple(LANES), nargs="?")
    args = parser.parse_args()
    try:
        if args.action == "probe":
            return probe()
        if args.lane is None:
            parser.error("lane is required for this action")
        if args.action == "start":
            return start(args.lane)
        if args.action == "worker":
            return worker(args.lane)
        return show_status(args.lane)
    except Exception as exc:
        print(f"Bedrock lane launcher: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
