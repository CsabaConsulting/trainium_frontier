#!/usr/bin/env python3
"""GPU-or-Trn2 experiment-runner wrapper for train.py.

Device-agnostic: `detect_device_type` (train.py:73) already picks CUDA before
Neuron, so this same script and the same train.py run unmodified on whichever
hardware is currently allocated — a rented GPU box today, a real Trn2
instance later. Only the launch *shape* differs (see --nproc-per-node/--lnc
below); the code path is identical either way.

Launches `train.py` via `torchrun` (matching the actual Phase 1 submission
launch-command format: `NEURON_LOGICAL_NC_CONFIG=<lnc> torchrun --standalone
--nproc_per_node=<n> train.py ...`), with a fixed-step or fixed-wall-clock
ceiling. Captures the full log, parses train.py's own final `key: value`
summary block, and appends one row to a persistent, git-tracked results log —
same convention as autoresearch's results.tsv (keep/discard/crash, one row
per experiment).

Default shape (--nproc-per-node 1 --lnc 1) is a single-device run — right for
a single-GPU rental box. `NEURON_LOGICAL_NC_CONFIG` is set regardless; it's a
no-op on CUDA, read only by Neuron.

Usage (single-GPU box, e.g. a rented RTX 6000 Pro):
    python research/run_experiment.py --label sol1-warmdown75 \
        --solution 1 --description "warmdown=0.75 final_lr_frac=0.05" \
        --num-steps 800 --max-train-seconds 900

Full-chip Trn2 shape, once allocated (matches README's default Phase 1
submission command — use this shape for any keep/discard decision meant to
carry over to the real submission, not the single-device default above):
    python research/run_experiment.py --label sol1-warmdown75 \
        --solution 1 --description "warmdown=0.75 final_lr_frac=0.05" \
        --num-steps 800 --max-train-seconds 900 --nproc-per-node 8 --lnc 1

Run from the repo root (same directory as train.py).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = Path(__file__).resolve().parent / "experiment-log.jsonl"

SUMMARY_KEYS = {
    "checkpoint_path", "training_seconds", "budget_seconds", "target_steps",
    "max_train_seconds", "startup_allowance_est", "charged_seconds_est",
    "over_budget", "total_seconds", "compiled",
    "nki_relu2", "num_steps", "total_tokens_M", "world_size", "lnc",
    "seq_len", "num_params_M", "causality_passed", "causality_max_abs",
    "public_val_bpb", "eval_seconds", "eval_timeout", "eval_over_timeout",
}
LINE_RE = re.compile(r"^(" + "|".join(SUMMARY_KEYS) + r"):\s*(.+)$")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def parse_summary(log_text: str) -> dict:
    parsed: dict[str, str] = {}
    for line in log_text.splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            parsed[m.group(1)] = m.group(2).strip()
    typed: dict = {}
    for k, v in parsed.items():
        if v in ("True", "False"):
            typed[k] = v == "True"
        else:
            try:
                typed[k] = int(v)
            except ValueError:
                try:
                    typed[k] = float(v)
                except ValueError:
                    typed[k] = v
    return typed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="short slug, e.g. sol1-warmdown75")
    p.add_argument("--solution", required=True, help="experiment-rollout-plan.md Solution number, e.g. '1'")
    p.add_argument("--description", required=True, help="what changed this run, e.g. 'warmdown=0.75'")
    p.add_argument("--num-steps", type=int, required=True)
    p.add_argument("--max-train-seconds", type=int, required=True,
                    help="operational ceiling for quality-axis runs; the actual control for time-axis runs")
    p.add_argument("--nproc-per-node", type=int, default=1, choices=[1, 4, 8],
                    help="matches README's valid submission shapes; default=1 is a single-device run "
                         "(right for a single-GPU box) — pass 8 for the full-chip Trn2 submission shape")
    p.add_argument("--lnc", type=int, default=1, choices=[1, 2],
                    help="NEURON_LOGICAL_NC_CONFIG; no-op on CUDA, read only by Neuron")
    p.add_argument("--eval-tokens", type=int, default=None,
                    help="override train.py's --eval-tokens for faster early iteration")
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                    help="anything else to pass through to train.py")
    p.add_argument("--no-commit", action="store_true", help="skip git-committing the log row")
    args = p.parse_args()

    launch_cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={args.nproc_per_node}", "train.py",
        "--num-steps", str(args.num_steps),
        "--max-train-seconds", str(args.max_train_seconds),
        "--out-dir", f"out/{args.label}",
    ]
    if args.eval_tokens is not None:
        launch_cmd += ["--eval-tokens", str(args.eval_tokens)]
    launch_cmd += args.extra_args

    env = os.environ.copy()
    env["NEURON_LOGICAL_NC_CONFIG"] = str(args.lnc)
    launch_cmd_str = f"NEURON_LOGICAL_NC_CONFIG={args.lnc} " + " ".join(launch_cmd)

    run_log_path = Path(__file__).resolve().parent / f"run-logs/{args.label}.log"
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"$ {launch_cmd_str}", flush=True)
    started = time.monotonic()
    status = "keep"  # default; overwritten below on crash/timeout
    try:
        proc = subprocess.run(launch_cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        log_text = proc.stdout + proc.stderr
        crashed = proc.returncode != 0
    except Exception as e:  # e.g. subprocess-level failure, not train.py's own crash handling
        log_text = f"<run_experiment.py failed to launch train.py: {e}>"
        crashed = True
    wall_elapsed = time.monotonic() - started

    run_log_path.write_text(log_text)
    summary = parse_summary(log_text)

    if crashed or "public_val_bpb" not in summary:
        status = "crash"

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "solution": args.solution,
        "description": args.description,
        "launch_command": launch_cmd_str,
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "status": status,
        "wall_elapsed_seconds": round(wall_elapsed, 1),
        "run_log": str(run_log_path.relative_to(REPO_ROOT)),
        **summary,
    }

    with LOG_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"--- logged: status={status} public_val_bpb={summary.get('public_val_bpb')} "
          f"num_steps={summary.get('num_steps')} total_seconds={summary.get('total_seconds')} "
          f"-> {LOG_PATH.relative_to(REPO_ROOT)}", flush=True)

    if not args.no_commit:
        try:
            git("add", str(LOG_PATH.relative_to(REPO_ROOT)))
            git("commit", "-m", f"experiment log: {args.label} ({status})")
            print("committed log row.", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"WARNING: git commit failed (log row still written locally): {e}", flush=True)

    if status == "crash":
        sys.exit(1)


if __name__ == "__main__":
    main()
