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
import fcntl
import hashlib
import json
import os
import re
import shlex
import socket
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
    # train.py --eval-seq-len: a second eval of the same weights at a shorter context, for the
    # open "what context length does the scorer actually use?" question (2026-09-11).
    "eval_alt_seq_len", "public_val_bpb_alt", "eval_alt_seconds",
    # T3.1 (2026-09-11): one line carrying every n-gram-VE knob that changes the model, so
    # two n-gram arms are distinguishable in the log without re-reading the run log.
    "ngram_ve",
    # T4.2 / T3.9 (2026-09-12): the Aster output-bigram-logit-table knobs and the Recursive
    # one-liner toggles, same reasoning as ngram_ve above.
    "out_bigram", "oneliners",
    # T3.2 (2026-09-13): the Engram-lite knobs, same reasoning as ngram_ve above.
    "engram",
    # T2.1 (2026-09-11): the batch-size ramp's knobs, same reasoning as ngram_ve above.
    "batch_ramp",
    # T2.2 (2026-09-11): the weight-EMA knobs plus its in-run raw-vs-EMA comparison. ema_delta is
    # the whole result of the experiment, so it belongs in the log row, not just the run log.
    "weight_ema", "public_val_bpb_ema", "ema_delta", "eval_ema_seconds",
    "public_val_bpb_ema_alt", "ema_delta_alt", "weight_ema_arms",
    # Plan 11.1 (2026-09-14): Recursive's three schedule tricks (Demon beta1, NorMuon beta2 ramp,
    # per-depth Muon LR/momentum) on one line, same reasoning as ngram_ve above.
    "sched_tricks",
}
LINE_RE = re.compile(r"^(" + "|".join(SUMMARY_KEYS) + r"):\s*(.+)$")
LOSS_RE = re.compile(r"^step\s+(\d+)\s+\|\s+loss\s+([0-9.eE+-]+)\s+\|")

# Human-authorized frontier-control command. Exact mode exists specifically so the protocol-owned
# runner can attach a unique label and durable receipts without adding even one argument to the
# submitted train.py invocation.
EXACT_DENSE1024_TRAIN_ARGV = [
    "torchrun", "--standalone", "--nproc_per_node=4", "train.py",
    "--n-embd", "1024", "--no-ngram-ve", "--no-qk-shift",
    "--compile-sdpa-direct", "--nki-local-conv", "--bf16-norm-output",
    "--pack-factor", "32", "--clip-after-reduce", "--seq-len", "1024",
    "--no-eval-public",
]
EXACT_DENSE1024_CHECKPOINT = Path("out/final.pt")
EXACT_DENSE1024_EVAL_SEQ_LEN = 1024
EXACT_DENSE1024_EVAL_TOKENS = 0
EXACT_CACHE_ENV_KEYS = (
    "TORCH_NEURONX_NEFF_CACHE_DIR",
    "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
    "TORCH_NEURONX_HLO_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "NEURON_COMPILE_CACHE_DIR",
)


def run_to_log(command: list[str], path: Path, *, cwd: Path, env: dict,
               pass_fds: tuple[int, ...] = ()) -> int:
    """Persist child output while it runs, including output preceding a crash."""
    with path.open("x") as log:
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                              stderr=subprocess.STDOUT, pass_fds=pass_fds)
        log.flush()
        os.fsync(log.fileno())
    return proc.returncode


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


def parse_logged_losses(log_text: str) -> list[dict[str, int | float]]:
    """Retain every loss the frozen source prints through step 50.

    The file prints steps 0--4 and then multiples of ten. This is deliberately evidence, not an
    invented "first 50 losses" gate.
    """
    points: list[dict[str, int | float]] = []
    for line in log_text.splitlines():
        match = LOSS_RE.match(line.strip())
        if match and int(match.group(1)) <= 50:
            points.append({"step": int(match.group(1)), "loss": float(match.group(2))})
    return points


def parse_eval_json(text: str) -> dict:
    """Find the one evaluate_bpb object in output that may include import/compiler chatter."""
    decoder = json.JSONDecoder()
    matches: list[dict] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "val_bpb" in value:
            matches.append(value)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one eval JSON object, found {len(matches)}")
    return matches[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="short slug, e.g. sol1-warmdown75")
    p.add_argument("--solution", required=True, help="experiment-rollout-plan.md Solution number, e.g. '1'")
    p.add_argument("--description", required=True, help="what changed this run, e.g. 'warmdown=0.75'")
    p.add_argument("--num-steps", type=int)
    p.add_argument("--max-train-seconds", type=int,
                    help="operational ceiling for quality-axis runs; the actual control for time-axis runs")
    p.add_argument("--nproc-per-node", type=int, default=1, choices=[1, 4, 8],
                    help="matches README's valid submission shapes; default=1 is a single-device run "
                         "(right for a single-GPU box) — pass 8 for the full-chip Trn2 submission shape")
    p.add_argument("--lnc", type=int, default=1, choices=[1, 2],
                    help="NEURON_LOGICAL_NC_CONFIG; no-op on CUDA, read only by Neuron")
    p.add_argument("--eval-tokens", type=int, default=None,
                    help="override train.py's --eval-tokens for faster early iteration")
    p.add_argument("--entrypoint", type=Path, default=Path("train.py"),
                    help="Explicit repository-local training wrapper; default unchanged train.py.")
    p.add_argument("--exact-dense1024", action="store_true",
                    help="invoke the pinned dense1024 submission argv exactly, then cold-evaluate out/final.pt")
    p.add_argument("--expected-train-sha256", default="",
                    help="required exact train.py SHA-256 in --exact-dense1024 mode")
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                    help="anything else to pass through to train.py")
    p.add_argument("--no-commit", action="store_true", help="skip git-committing the log row")
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.label):
        p.error("--label must be a filename-safe slug (letters, digits, _, -, .)")
    if args.exact_dense1024:
        if args.num_steps is not None or args.max_train_seconds is not None or args.eval_tokens is not None:
            p.error("exact mode forbids runner-owned train/eval overrides")
        if args.entrypoint != Path("train.py") or args.extra_args:
            p.error("exact mode forbids entrypoint and extra-argument overrides")
        if not re.fullmatch(r"[0-9a-f]{64}", args.expected_train_sha256):
            p.error("exact mode requires --expected-train-sha256")
        cache_paths = []
        for cache_key in EXACT_CACHE_ENV_KEYS:
            for env_key in (cache_key, f"AUTOLOOP_EVAL_{cache_key}"):
                value = os.environ.get(env_key, "")
                if not value or not Path(value).is_absolute():
                    p.error(f"exact mode requires absolute unique cache env {env_key}")
                cache_paths.append(value)
        if len(cache_paths) != len(set(cache_paths)):
            p.error("exact mode cache paths must be pairwise unique")
        if any(Path(path).exists() for path in cache_paths):
            p.error("exact mode requires every train/eval compile-cache path to be fresh")
    elif args.num_steps is None or args.max_train_seconds is None:
        p.error("standard mode requires --num-steps and --max-train-seconds")
    entrypoint = (REPO_ROOT / args.entrypoint).resolve()
    if (not entrypoint.is_relative_to(REPO_ROOT) or not entrypoint.is_file()
            or entrypoint.suffix != ".py"):
        p.error("--entrypoint must name a Python file inside this repository")
    entrypoint_name = str(entrypoint.relative_to(REPO_ROOT))
    source_paths = list(dict.fromkeys(("train.py", "prepare.py", "research/run_experiment.py",
                                      entrypoint_name)))

    # The lock is inherited by torchrun: killing just this wrapper cannot permit
    # a second run to overlap surviving workers. A new checkout still needs the
    # protocol's host-wide process/device-holder preflight.
    lock_path = REPO_ROOT / "out" / ".experiment.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("another experiment owns this checkout's run lock")

    if args.exact_dense1024:
        actual_train_sha = hashlib.sha256((REPO_ROOT / "train.py").read_bytes()).hexdigest()
        if actual_train_sha != args.expected_train_sha256:
            p.error("train.py SHA-256 differs from --expected-train-sha256")
        launch_cmd = list(EXACT_DENSE1024_TRAIN_ARGV)
    else:
        launch_cmd = [
            "torchrun", "--standalone", f"--nproc_per_node={args.nproc_per_node}", entrypoint_name,
            "--num-steps", str(args.num_steps),
            "--max-train-seconds", str(args.max_train_seconds),
            "--out-dir", f"out/{args.label}",
        ]
        if args.eval_tokens is not None:
            launch_cmd += ["--eval-tokens", str(args.eval_tokens)]
        launch_cmd += args.extra_args

    env = os.environ.copy()
    env["NEURON_LOGICAL_NC_CONFIG"] = str(args.lnc)
    env["PYTHONUNBUFFERED"] = "1"
    launch_cmd_str = f"NEURON_LOGICAL_NC_CONFIG={args.lnc} " + shlex.join(launch_cmd)

    run_log_path = Path(__file__).resolve().parent / f"run-logs/{args.label}.log"
    eval_log_path = Path(__file__).resolve().parent / f"run-logs/{args.label}.eval.log"
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = run_log_path.with_suffix(".launch.json")
    if run_log_path.exists() or manifest_path.exists() or eval_log_path.exists():
        p.error(f"refusing to overwrite existing experiment {args.label}; use a new label")
    # Capture provenance BEFORE creating run artifacts, rather than diagnosing
    # every successful run as dirty because it has just written a new log.
    source_commit = git("rev-parse", "HEAD")
    source_dirty = bool(git("status", "--porcelain"))
    training_source_dirty = bool(git("status", "--porcelain", "--", *source_paths))
    manifest = {
        "label": args.label,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "git_commit": source_commit,
        "git_dirty": source_dirty,
        "source_dirty": training_source_dirty,
        "launch_command": launch_cmd_str,
        "description": args.description,
        "entrypoint": entrypoint_name,
        "mode": "exact-dense1024-cold-eval" if args.exact_dense1024 else "standard",
        "exact_train_argv": launch_cmd if args.exact_dense1024 else None,
        "external_eval": ({
            "program": sys.executable,
            "argv": ["prepare.py", "eval-public", "--train-py", "train.py", "--checkpoint",
                     str(EXACT_DENSE1024_CHECKPOINT), "--seq-len",
                     str(EXACT_DENSE1024_EVAL_SEQ_LEN), "--eval-tokens",
                     str(EXACT_DENSE1024_EVAL_TOKENS)],
            "fresh_process": True,
        } if args.exact_dense1024 else None),
        "environment": {k: v for k, v in env.items()
                        if k.startswith(("NEURON_", "NKI_", "XLA_"))
                        or k in ("OMP_NUM_THREADS", "MALLOC_ARENA_MAX", "PYTHONUNBUFFERED")},
        "source_sha256": {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                          for name in source_paths},
    }
    with manifest_path.open("x") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")
        manifest_file.flush()
        os.fsync(manifest_file.fileno())

    print(f"$ {launch_cmd_str}", flush=True)
    started = time.monotonic()
    status = "keep"  # default; overwritten below on crash/timeout
    try:
        returncode = run_to_log(launch_cmd, run_log_path, cwd=REPO_ROOT, env=env,
                                pass_fds=(lock_file.fileno(),))
        crashed = returncode != 0
    except Exception as e:  # e.g. subprocess-level failure, not train.py's own crash handling
        with run_log_path.open("a") as log:
            log.write(f"\n<run_experiment.py failed to launch train.py: {e}>\n")
        returncode = None
        crashed = True
    wall_elapsed = time.monotonic() - started

    log_text = run_log_path.read_text(errors="replace")
    summary = parse_summary(log_text)
    logged_loss_points = parse_logged_losses(log_text)

    external_eval = None
    if args.exact_dense1024 and not crashed:
        checkpoint = REPO_ROOT / EXACT_DENSE1024_CHECKPOINT
        if summary.get("checkpoint_path") != str(EXACT_DENSE1024_CHECKPOINT) or not checkpoint.is_file():
            crashed = True
        else:
            eval_argv = [
                sys.executable, "prepare.py", "eval-public", "--train-py", "train.py",
                "--checkpoint", str(EXACT_DENSE1024_CHECKPOINT),
                "--seq-len", str(EXACT_DENSE1024_EVAL_SEQ_LEN),
                "--eval-tokens", str(EXACT_DENSE1024_EVAL_TOKENS),
            ]
            eval_env = dict(env)
            for cache_key in EXACT_CACHE_ENV_KEYS:
                cold_path = eval_env.pop(f"AUTOLOOP_EVAL_{cache_key}", "")
                if not cold_path:
                    raise RuntimeError(f"missing cold-eval cache identity for {cache_key}")
                eval_env[cache_key] = cold_path
            eval_started = time.monotonic()
            eval_returncode = run_to_log(eval_argv, eval_log_path, cwd=REPO_ROOT, env=eval_env)
            eval_text = eval_log_path.read_text(errors="replace")
            try:
                eval_result = parse_eval_json(eval_text) if eval_returncode == 0 else {}
            except ValueError:
                eval_result = {}
            external_eval = {
                "argv": eval_argv,
                "fresh_process": True,
                "returncode": eval_returncode,
                "seq_len": EXACT_DENSE1024_EVAL_SEQ_LEN,
                "eval_tokens": EXACT_DENSE1024_EVAL_TOKENS,
                "log": str(eval_log_path.relative_to(REPO_ROOT)),
                "wall_seconds": round(time.monotonic() - eval_started, 3),
                "result": eval_result,
            }
            if eval_returncode != 0 or not isinstance(eval_result.get("val_bpb"), (int, float)):
                crashed = True
            else:
                summary.update({
                    "public_val_bpb": float(eval_result["val_bpb"]),
                    "eval_seconds": eval_result.get("eval_seconds"),
                    "eval_timeout": eval_result.get("eval_timeout"),
                    "eval_over_timeout": bool(eval_result.get("eval_over_timeout", False)),
                    "eval_context": EXACT_DENSE1024_EVAL_SEQ_LEN,
                    "eval_tokens_requested": EXACT_DENSE1024_EVAL_TOKENS,
                    "selection_metric": "public_val_bpb",
                })

    if crashed or "public_val_bpb" not in summary or summary.get("eval_over_timeout") is True:
        status = "crash"

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "solution": args.solution,
        "description": args.description,
        "launch_command": launch_cmd_str,
        "git_commit": source_commit,
        "git_dirty": source_dirty,
        "source_dirty": training_source_dirty,
        "hostname": manifest["hostname"],
        "environment": manifest["environment"],
        "source_sha256": manifest["source_sha256"],
        "returncode": returncode,
        "status": status,
        "wall_elapsed_seconds": round(wall_elapsed, 1),
        "run_log": str(run_log_path.relative_to(REPO_ROOT)),
        "exact_train_argv": launch_cmd if args.exact_dense1024 else None,
        "external_eval": external_eval,
        "logged_loss_points_through_step_50": logged_loss_points,
        "loss_trace_limitation": (
            "frozen source logs steps 0-4 and then multiples of 10; an exact first-50-loss "
            "comparison is unavailable without changing train.py"
            if args.exact_dense1024 else None
        ),
        **summary,
    }

    with LOG_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())

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
