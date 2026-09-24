#!/usr/bin/env python3
"""Run one approved Trn2 job with fail-closed, non-overwriting evidence."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contained  # noqa: E402
import decide  # noqa: E402

EVAL_TOKENS, EVAL_SEQ = decide.EVAL_TOKENS, decide.EVAL_SEQ
HOST_LOCK = Path("/home/ubuntu/.cache/trainium_freshstart/device0.lock")
DECIDE_SHA256 = "a47bedd258697b0791f48dcac3098d9071b5aaf00a7664d9e39987b4052adc66"
LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
DIST_ENV = {
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
}
EVAL_FIELDS = {
    "val_bpb", "eval_seconds", "eval_timeout", "eval_over_timeout",
    "num_tokens", "num_bytes", "total_nats",
}
CAUSAL_FIELDS = {
    "schema", "causality_passed", "max_abs_diff", "tolerance", "seq_len",
    "train_py_sha256", "checkpoint_sha256",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path: Path, obj) -> None:
    data = (json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_dir(path.parent)


class JobError(Exception):
    def __init__(self, code: int, stage: str, msg: str):
        super().__init__(msg)
        self.code, self.stage = code, stage


def git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if p.returncode:
        raise JobError(4, "preflight", f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def ensure_label(value) -> str:
    if not isinstance(value, str) or not LABEL_RE.fullmatch(value):
        raise JobError(4, "preflight", "invalid label; use 1-128 ASCII letters/digits/._-")
    return value


def ensure_plain_dir(path: Path, create=False) -> Path:
    path = Path(os.path.abspath(path))
    cur = Path(path.anchor)
    for part in path.parts[1:]:
        cur /= part
        if cur.exists() or cur.is_symlink():
            if stat.S_ISLNK(os.lstat(cur).st_mode):
                raise JobError(4, "preflight", f"symlinked path component refused: {cur}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise JobError(4, "preflight", f"directory missing: {path}")
    return path


def confined(repo: Path, value, what: str) -> Path:
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise JobError(4, "preflight", f"{what} must be a confined relative path")
    path = Path(os.path.abspath(repo / rel))
    try:
        path.relative_to(repo)
    except ValueError as exc:
        raise JobError(4, "preflight", f"{what} escapes repository") from exc
    cur = repo
    for part in rel.parts:
        cur /= part
        if cur.exists() or cur.is_symlink():
            if stat.S_ISLNK(os.lstat(cur).st_mode):
                raise JobError(4, "preflight", f"symlinked {what} component refused: {cur}")
    return path


def regular_tracked(repo: Path, rel, digest: str, what: str) -> Path:
    path = confined(repo, rel, what)
    try:
        st = os.lstat(path)
    except FileNotFoundError as exc:
        raise JobError(4, "preflight", f"{what} missing") from exc
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise JobError(4, "preflight", f"{what} must be a regular single-link file")
    if subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch",
                       str(path.relative_to(repo))], capture_output=True).returncode:
        raise JobError(4, "preflight", f"{what} is not tracked")
    if sha256(path) != digest:
        raise JobError(4, "preflight", f"{what} SHA-256 differs from pin")
    return path


def read_json(path: Path, stage="preflight"):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise JobError(4 if stage == "preflight" else 7, stage,
                       f"invalid JSON {path}: {exc}") from exc


@contextlib.contextmanager
def host_lease(cfg: dict):
    expected, actual = cfg.get("expected_hostname"), socket.gethostname()
    if expected != actual:
        raise JobError(4, "preflight", f"hostname mismatch: expected={expected!r} actual={actual!r}")
    if cfg.get("test_mode") is not True and Path(cfg.get("host_lock_path", HOST_LOCK)) != HOST_LOCK:
        raise JobError(4, "preflight", f"production host lock must be the legacy lock {HOST_LOCK}")
    lock = Path(cfg.get("host_lock_path", HOST_LOCK)) if cfg.get("test_mode") is True else HOST_LOCK
    if not lock.is_absolute():
        raise JobError(4, "preflight", "host lock path must be absolute")
    ensure_plain_dir(lock.parent)
    try:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as exc:
        raise JobError(4, "preflight", f"cannot open shared host lock: {exc}") from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise JobError(4, "preflight", "shared host/device lease is already held") from exc
        yield
    finally:
        os.close(fd)


def effective_env(cfg: dict) -> dict[str, str]:
    additions = cfg.get("env", {})
    if not isinstance(additions, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                              for k, v in additions.items()):
        raise JobError(4, "preflight", "host env must map strings to strings")
    env = dict(os.environ)
    env.update(additions)
    forbidden = sorted(k for k in env if k.startswith("NEURON_COMPETITION_"))
    if forbidden:
        raise JobError(4, "preflight", f"science-changing environment override refused: {forbidden}")
    if env.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise JobError(4, "preflight", "effective NEURON_LOGICAL_NC_CONFIG must equal 2")
    cc = env.get("NEURON_CC_FLAGS")
    if cc is not None and "--optlevel=1" not in cc.split():
        raise JobError(4, "preflight", "effective NEURON_CC_FLAGS lacks --optlevel=1")
    return env


def option_values(argv: list[str], option: str) -> list[str]:
    out = []
    for i, token in enumerate(argv):
        if token == option:
            if i + 1 == len(argv):
                raise JobError(4, "preflight", f"missing value for {option}")
            out.append(argv[i + 1])
        elif token.startswith(option + "="):
            out.append(token.split("=", 1)[1])
    return out


def require_option(argv, option, expected, stage="preflight") -> None:
    got = option_values(argv, option)
    if got != [str(expected)]:
        raise JobError(4 if stage == "preflight" else 6, stage,
                       f"expected exactly one {option} {expected}; got {got}")


def validate_science_argv(job: dict) -> None:
    argv = job.get("argv")
    if not isinstance(argv, list) or any(not isinstance(x, str) for x in argv):
        raise JobError(4, "preflight", "argv must be a string list")
    require_option(argv, "--seed", 42)
    require_option(argv, "--max-train-seconds", 1800)
    require_option(argv, "--seq-len", EVAL_SEQ)
    if argv.count("--no-eval-public") != 1 or "--eval-public" in argv:
        raise JobError(4, "preflight", "training must have exactly one --no-eval-public")


def validate_g2(g2: dict, job: dict, prepare_sha: str, test_mode: bool) -> None:
    tiny = {"verdict": "PASS", "train_py_sha256": job["train_sha256"], "argv": job["argv"]}
    if test_mode and g2 == tiny:
        return
    required = {"schema", "verdict", "train_py_sha256", "argv", "prepare", "steps", "nproc",
                "torchrun_rc", "timed_out", "ranks", "plan_sha256"}
    if not isinstance(g2, dict) or not required.issubset(g2):
        raise JobError(4, "preflight", f"G2 receipt missing fields: {sorted(required - set(g2))}")
    if (g2["schema"] != "g2-preflight-v1" or g2["verdict"] != "PASS"
            or g2["train_py_sha256"] != job["train_sha256"] or g2["argv"] != job["argv"]):
        raise JobError(4, "preflight", "G2 identity/verdict does not bind exact train bytes and argv")
    prep = g2["prepare"]
    if not isinstance(prep, dict) or prep.get("stub") is not False or prep.get("sha256") != prepare_sha:
        raise JobError(4, "preflight", "G2 must use pinned canonical prepare.py with stub=false")
    if (g2["nproc"] != 4 or isinstance(g2["steps"], bool) or not isinstance(g2["steps"], int)
            or g2["steps"] <= 0 or g2["timed_out"] is not False or g2["torchrun_rc"] != 0):
        raise JobError(4, "preflight", "G2 process/step/exit contract failed")
    plan, ranks = g2["plan_sha256"], g2["ranks"]
    if not isinstance(plan, str) or len(plan) != 64:
        raise JobError(4, "preflight", "G2 plan SHA is missing")
    if not isinstance(ranks, list) or len(ranks) != 4 or {r.get("rank") for r in ranks} != {0, 1, 2, 3}:
        raise JobError(4, "preflight", "G2 needs four distinct rank reports")
    for rank in ranks:
        if (rank.get("status") != "PASS" or rank.get("plan_sha256") != plan
                or rank.get("cross_rank_mismatches") != [] or not isinstance(rank.get("plan_summary"), dict)):
            raise JobError(4, "preflight", "G2 rank PASS/plan/consistency fields failed")
    ck = next(r for r in ranks if r["rank"] == 0).get("checkpoint_check")
    if not isinstance(ck, dict) or ck.get("ok") is not True or ck.get("path_exists") is not True:
        raise JobError(4, "preflight", "G2 checkpoint strict-load check failed")


def other_training_procs():
    pattern = re.compile(r"(torchrun|train\.py|prepare\.py eval-public|run_experiment\.py)")
    hits, me = [], os.getpid()
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            pid = int(path.parent.name)
            cmd = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError):
            continue
        if pid != me and pattern.search(cmd) and "run_job.py" not in cmd and "preflight_dryrun.py" not in cmd:
            hits.append({"pid": pid, "cmd": cmd[:160]})
    return hits


def ledger_labels(path: Path) -> set[str]:
    if not path.exists():
        return set()
    labels = set()
    for no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JobError(4, "preflight", f"invalid ledger line {no}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("label"), str) or row["label"] in labels:
            raise JobError(4, "preflight", f"invalid/duplicate ledger row {no}")
        labels.add(row["label"])
    return labels


def git_status(repo: Path) -> list[str]:
    text = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    return text.splitlines() if text else []


def status_path(line: str) -> str:
    value = line[3:]
    if " -> " in value or value.startswith('"'):
        return ""
    return value


def preflight(cfg, job, d, repo, out_dir, env) -> dict:
    st = shutil.disk_usage(repo)
    needed = int(cfg.get("disk_floor_gib", 6.5) * 2**30 + job.get("footprint_gib", 1.2) * 2**30)
    other = other_training_procs()
    if cfg.get("test_mode") is True and cfg.get("test_ignore_other_training_processes") is True:
        other = []
    checks = {
        "tagged_processes": contained.describe(contained.tagged_pids(job["label"])),
        "other_training_processes": other, "neuron_holders": contained.neuron_holders(),
        "disk_free_bytes": st.free, "disk_needed_bytes": needed,
        "git_head": git(repo, "rev-parse", "HEAD"), "git_status": git_status(repo),
        "actual_hostname": socket.gethostname(),
        "effective_env": {k: env.get(k) for k in ("NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS", "NEURON_COMPILE_CACHE_DIR")},
    }
    problems = []
    if checks["tagged_processes"] or checks["other_training_processes"]:
        problems.append("host not idle")
    if checks["neuron_holders"]:
        problems.append("/dev/neuron held")
    if st.free < needed:
        problems.append("disk below floor + footprint")
    ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                               str(job.get("git_commit", "")), checks["git_head"]], capture_output=True)
    if ancestor.returncode:
        problems.append("pinned science commit is not an ancestor of current HEAD")
    if not cfg.get("test_mode") and job.get("origin_ref"):
        if git(repo, "rev-parse", job["origin_ref"]) != checks["git_head"]:
            problems.append("current HEAD differs from pinned origin ref")
    elif not cfg.get("test_mode"):
        problems.append("production job lacks origin_ref/origin_commit pin")
    if checks["git_status"]:
        problems.append("repository has tracked or untracked changes")
    if out_dir.exists() or out_dir.is_symlink():
        problems.append("fresh output path already exists")
    ledger = repo / "research/v2/ledger.jsonl"
    if job["label"] in ledger_labels(ledger):
        problems.append("label already exists in ledger")
    if any((d / name).exists() or (d / name).is_symlink()
           for name in ("preflight.json", "claim.json", "launch-train.json", "record.json", "verdict.json")):
        problems.append("label was already claimed/launched/recorded")
    if cfg.get("eval_args_verified") is not True:
        problems.append("eval argv is not marked verified against canonical prepare.py")
    checks.update(problems=problems, ok=not problems, utc=now())
    attempt = d / f"preflight-attempt-{time.time_ns()}-{os.getpid()}.json"
    write_new(attempt, checks)
    if problems:
        raise JobError(4, "preflight", "; ".join(problems))
    return checks, attempt


def launch_and_wait(cfg, d, repo, label, cmd, log_name, timeout, env, phase):
    run_env = dict(env)
    run_env[contained.TAG] = label
    argv, scope = list(cmd), None
    if cfg.get("use_cgroup") and shutil.which("systemd-run"):
        scope = f"exp-{label}"
        argv = ["systemd-run", "--user", "--scope", f"--unit={scope}", "--collect", "--quiet"] + argv
    fd = os.open(d / log_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as log:
        proc = subprocess.Popen(argv, env=run_env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, cwd=repo)
    runtime = d / f"runtime-{phase}"
    runtime.mkdir(mode=0o700)
    receipt = {"label": label, "started_utc": now(), "leader_pid": proc.pid, "pgid": proc.pid,
               "cgroup_scope": scope, "argv": cmd, "env_keys": sorted(cfg.get("env", {}))}
    write_new(runtime / "launch.json", receipt)
    write_new(d / f"launch-{phase}.json", receipt)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        rc = None
    return rc, runtime


def quiesce(label, d, runtime, files, window, grace=60, stop_first=False):
    ns = argparse.Namespace(label=label, dir=str(runtime), grace=grace, window=window,
                            files=[str(x) for x in files])
    with contextlib.redirect_stdout(sys.stderr):
        if stop_first:
            contained.cmd_stop(ns)
        rc = contained.cmd_quiesce(ns)
    q = read_json(runtime / "quiescence.json", "quiesce")
    q["intervened"] = stop_first
    if rc and not stop_first:
        return quiesce(label, d, runtime, files, window, grace, True)
    if rc:
        raise JobError(3, "quiesce", f"{label}: survivors/holders after stop; breaker")
    phase = runtime.name.removeprefix("runtime-")
    write_new(d / f"quiescence-{phase}.json", q)
    return q


SUMMARY_TYPES = {
    "training_seconds": float, "budget_seconds": float, "max_train_seconds": float,
    "startup_allowance_est": float, "charged_seconds_est": float, "over_budget": bool,
    "num_steps": int, "total_tokens_M": float, "world_size": int,
    "seq_len": int, "num_params_M": float,
}
STEP_RE = re.compile(r"^step\s+(\d+)\s+\|\s+loss\s+(\S+).*?\|\s+dt\s+(\S+)s.*?\|\s+epoch\s+(\d+)\s*$")


def scalar(raw, typ):
    if typ is bool:
        if raw not in ("True", "False"):
            raise ValueError(raw)
        return raw == "True"
    value = typ(raw)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(raw)
    return value


def parse_training_log(path: Path) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    marks = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(marks) != 1:
        raise JobError(5, "train", f"expected one complete training summary; got {len(marks)}")
    out = {}
    for line in lines[marks[0] + 1:]:
        if ":" not in line:
            continue
        key, raw = (x.strip() for x in line.split(":", 1))
        if key not in SUMMARY_TYPES:
            continue
        if key in out:
            raise JobError(5, "train", f"duplicate summary field {key}")
        try:
            out[key] = scalar(raw, SUMMARY_TYPES[key])
        except ValueError as exc:
            raise JobError(5, "train", f"invalid summary field {key}={raw!r}") from exc
    missing = set(SUMMARY_TYPES) - set(out)
    if missing:
        raise JobError(5, "train", f"summary missing {sorted(missing)}")
    if (out["over_budget"] is not False or out["world_size"] != 4 or out["seq_len"] != EVAL_SEQ
            or out["budget_seconds"] != 1800 or out["max_train_seconds"] != 1800
            or min(out["num_steps"], out["training_seconds"], out["charged_seconds_est"],
                   out["total_tokens_M"], out["num_params_M"]) <= 0):
        raise JobError(5, "train", "summary violates budget/world/shape/positive-count contract")
    dts, losses, epoch2 = [], [], None
    for line in lines[:marks[0]]:
        match = STEP_RE.match(line.strip())
        if not match:
            continue
        try:
            step, loss, dt, epoch = int(match[1]), float(match[2]), float(match[3]), int(match[4])
        except ValueError as exc:
            raise JobError(5, "train", "malformed step metric") from exc
        if not math.isfinite(loss) or not math.isfinite(dt) or loss <= 0 or dt <= 0:
            raise JobError(5, "train", "non-finite/non-positive logged loss or dt")
        losses.append({"step": step, "loss": loss})
        if step >= 20:
            dts.append(dt)
        if epoch >= 2 and epoch2 is None:
            epoch2 = step
    if not losses or "bad loss detected" in "\n".join(lines).lower():
        raise JobError(5, "train", "missing finite losses or bad-loss marker")
    out["logged_losses"] = losses
    out["median_step_s"] = sorted(dts)[len(dts) // 2] if dts else None
    out["epoch_boundary_step"] = epoch2
    return out


def checkpoint_identity(path: Path, out_dir: Path, code=5) -> dict:
    try:
        st = os.lstat(path)
    except FileNotFoundError as exc:
        raise JobError(code, "train" if code == 5 else "prune", "checkpoint missing") from exc
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size <= 0:
        raise JobError(code, "train" if code == 5 else "prune", "checkpoint is not regular/single-link/nonempty")
    try:
        path.resolve(strict=True).relative_to(out_dir.resolve(strict=True))
    except (ValueError, FileNotFoundError) as exc:
        raise JobError(code, "train" if code == 5 else "prune", "checkpoint escapes output root") from exc
    return {"device": st.st_dev, "inode": st.st_ino, "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "sha256": sha256(path)}


def build_eval_cmd(cfg, job, out_dir):
    if cfg.get("eval_cmd_override") is not None:
        if cfg.get("test_mode") is not True:
            raise JobError(4, "preflight", "eval override requires host test_mode")
        cmd = list(cfg["eval_cmd_override"])
    else:
        cmd = [cfg.get("python", "python"), "prepare.py"]
    cmd += ["eval-public", "--train-py", job["train_py"], "--checkpoint", str(out_dir / "final.pt")]
    cmd += list(cfg.get("eval_args", []))
    if cmd.count("eval-public") != 1:
        raise JobError(6, "eval", "eval command needs one literal eval-public")
    require_option(cmd, "--seq-len", EVAL_SEQ, "eval")
    require_option(cmd, "--eval-tokens", EVAL_TOKENS, "eval")
    timeouts = option_values(cmd, "--timeout-seconds")
    if len(timeouts) != 1 or not timeouts[0].isdigit() or int(timeouts[0]) <= 0:
        raise JobError(6, "eval", "eval command needs one positive timeout")
    return cmd


def build_causal_cmd(cfg, job, out_dir):
    if cfg.get("causality_cmd_override") is not None:
        if cfg.get("test_mode") is not True:
            raise JobError(4, "preflight", "causality override requires host test_mode")
        cmd = list(cfg["causality_cmd_override"])
    else:
        cmd = [cfg.get("python", "python"), str(HERE / "check_causality.py")]
    cmd += ["--train-py", job["train_py"], "--checkpoint", str(out_dir / "final.pt"),
            "--seq-len", str(EVAL_SEQ)]
    require_option(cmd, "--seq-len", EVAL_SEQ, "eval")
    return cmd


def json_objects(path: Path) -> list[dict]:
    text, decoder, objects = path.read_text(errors="replace"), json.JSONDecoder(), []
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)
    return objects


def parse_eval(path, expected_timeout):
    objects = json_objects(path)
    if len(objects) != 1:
        raise JobError(6, "eval", f"expected one evaluator JSON object; got {len(objects)}")
    ev = objects[0]
    if set(ev) != EVAL_FIELDS:
        raise JobError(6, "eval", f"evaluator field set differs: {sorted(set(ev) ^ EVAL_FIELDS)}")
    for key in ("val_bpb", "total_nats", "eval_seconds"):
        if isinstance(ev[key], bool) or not isinstance(ev[key], (int, float)) or not math.isfinite(ev[key]) or ev[key] <= 0:
            raise JobError(6, "eval", f"invalid evaluator value {key}")
    if isinstance(ev["num_tokens"], bool) or not isinstance(ev["num_tokens"], int) or ev["num_tokens"] != EVAL_TOKENS:
        raise JobError(6, "eval", "wrong evaluator token count")
    if isinstance(ev["num_bytes"], bool) or not isinstance(ev["num_bytes"], int) or ev["num_bytes"] <= 0:
        raise JobError(6, "eval", "invalid evaluator byte count")
    if ev["eval_over_timeout"] is not False or ev["eval_timeout"] != expected_timeout:
        raise JobError(6, "eval", "evaluator timeout facts violate contract")
    recomputed = ev["total_nats"] / (math.log(2) * ev["num_bytes"])
    if not math.isclose(ev["val_bpb"], recomputed, rel_tol=1e-12, abs_tol=1e-12):
        raise JobError(6, "eval", "val_bpb does not recompute from nats/bytes")
    return ev


def parse_causality(path, job, checkpoint_sha):
    objects = json_objects(path)
    if len(objects) != 1:
        raise JobError(6, "causality", f"expected one causality JSON object; got {len(objects)}")
    row = objects[0]
    if set(row) != CAUSAL_FIELDS:
        raise JobError(6, "causality", "causality result field set differs")
    if (row["schema"] != "checkpoint-causality-v1" or row["causality_passed"] is not True
            or row["seq_len"] != EVAL_SEQ or row["train_py_sha256"] != job["train_sha256"]
            or row["checkpoint_sha256"] != checkpoint_sha):
        raise JobError(6, "causality", "causality PASS is not bound to exact source/checkpoint/context")
    for key in ("max_abs_diff", "tolerance"):
        if isinstance(row[key], bool) or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
            raise JobError(6, "causality", f"invalid causality {key}")
    if row["max_abs_diff"] > row["tolerance"]:
        raise JobError(6, "causality", "causality numerical check failed")
    return row


def archive_checkpoint(cfg, ckpt, ident):
    root = ensure_plain_dir(Path(cfg["archive_dir"]), create=True)
    objects = ensure_plain_dir(root / "sha256", create=True)
    dst = objects / f"{ident['sha256']}.pt"
    reused = dst.exists()
    if reused:
        st = os.lstat(dst)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or sha256(dst) != ident["sha256"]:
            raise JobError(7, "archive", "existing archive object is unsafe/hash-mismatched")
    else:
        tmp = objects / f".{ident['sha256']}.{os.getpid()}.{time.time_ns()}.partial"
        sfd = os.open(ckpt, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        dfd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with os.fdopen(sfd, "rb", closefd=False) as src, os.fdopen(dfd, "wb", closefd=False) as out:
                shutil.copyfileobj(src, out, 1 << 20)
                out.flush()
                os.fsync(out.fileno())
        finally:
            os.close(sfd)
            os.close(dfd)
        if sha256(tmp) != ident["sha256"]:
            tmp.unlink()
            raise JobError(7, "archive", "temporary archive hash mismatch")
        try:
            os.link(tmp, dst, follow_symlinks=False)
        except FileExistsError as exc:
            tmp.unlink()
            raise JobError(7, "archive", "archive object appeared concurrently") from exc
        tmp.unlink()
        fsync_dir(objects)
    if sha256(dst) != ident["sha256"]:
        raise JobError(7, "archive", "archive readback mismatch")
    receipt = {"schema": "checkpoint-archive-v1", "utc": now(), "source": str(ckpt),
               "destination": str(dst), "sha256": ident["sha256"], "size": ident["size"],
               "reused_existing": reused, "readback_verified": True}
    return str(dst), receipt


def append_ledger(path, rec):
    required = {"schema", "label", "status", "checkpoint_sha256", "public_20m", "eval_tokens", "eval_seq_len"}
    if rec.get("schema") != "trn-run-record-v2" or not required.issubset(rec):
        raise JobError(7, "ledger", "record schema invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        labels = set()
        for no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JobError(7, "ledger", f"bad ledger line {no}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("label"), str) or row["label"] in labels:
                raise JobError(7, "ledger", f"invalid/duplicate ledger line {no}")
            labels.add(row["label"])
        if rec["label"] in labels:
            raise JobError(7, "ledger", "duplicate ledger label")
        f.seek(0, os.SEEK_END)
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    fsync_dir(path.parent)


def postflight_status(repo, d, out_dir, ledger):
    raw, allowed, bad = git_status(repo), [], []
    prefixes = (d.relative_to(repo).as_posix() + "/", out_dir.relative_to(repo).as_posix() + "/")
    ledger_rel = ledger.relative_to(repo).as_posix()
    for line in raw:
        rel = status_path(line)
        if rel and (rel.startswith(prefixes) or rel == ledger_rel):
            path = repo / rel
            allowed.append({"status": line[:2], "path": rel,
                            "sha256": sha256(path) if path.is_file() and not path.is_symlink() else None})
        else:
            bad.append(line)
    return raw, allowed, bad


def maybe_inject(cfg, stage):
    if cfg.get("test_mode") is True and cfg.get("inject_failure_stage") == stage:
        raise JobError(7, stage, f"test-only injected {stage} failure")


def prune(ckpt, out_dir, ident):
    current = checkpoint_identity(ckpt, out_dir, 7)
    if any(current[k] != ident[k] for k in ("device", "inode", "size", "sha256")):
        raise JobError(7, "prune", "checkpoint identity changed before prune")
    ckpt.unlink()
    fsync_dir(ckpt.parent)
    return {"schema": "checkpoint-prune-v1", "utc": now(), "path": str(ckpt),
            "sha256": ident["sha256"], "inode": ident["inode"], "pruned": True}


def run(cfg: dict, job: dict, repo: Path, job_path: Path) -> dict:
    if "test_mode" in job:
        raise JobError(4, "preflight", "test_mode may be set only by the host config")
    label = ensure_label(job.get("label"))
    ensure_plain_dir(repo / "research/v2/experiments")
    d = confined(repo, f"research/v2/experiments/{label}", "experiment directory")
    if not d.is_dir() or job_path != (d / "job.json").resolve():
        raise JobError(4, "preflight", "job must be canonical <label>/job.json")
    job_st = os.lstat(d / "job.json")
    if (not stat.S_ISREG(job_st.st_mode) or job_st.st_nlink != 1
            or subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch",
                               str((d / "job.json").relative_to(repo))], capture_output=True).returncode):
        raise JobError(4, "preflight", "canonical job.json must be a tracked regular single-link file")
    if job.get("host") != cfg.get("host"):
        raise JobError(4, "preflight", "job/host mismatch")
    test_mode = cfg.get("test_mode") is True
    if any(cfg.get(k) is not None for k in ("train_cmd_override", "eval_cmd_override", "causality_cmd_override")) and not test_mode:
        raise JobError(4, "preflight", "command override requires host test_mode")
    validate_science_argv(job)
    env = effective_env(cfg)
    out_root = confined(repo, cfg.get("out_root", "out"), "out_root")
    if out_root.exists():
        ensure_plain_dir(out_root)
    out_dir = confined(repo, f"{Path(cfg.get('out_root', 'out')).as_posix()}/{label}", "output path")
    train = regular_tracked(repo, job["train_py"], job["train_sha256"], "train.py")
    prepare = regular_tracked(repo, "prepare.py", job["prepare_py_sha256"], "prepare.py")
    decide_path = None
    causality_tool = None
    if not test_mode:
        decide_path = regular_tracked(repo, "workflow/tools/decide.py", DECIDE_SHA256, "decide.py")
        causality_tool = regular_tracked(repo, "workflow/tools/check_causality.py",
                                         job["causality_tool_sha256"], "causality tool")
        if causality_tool.resolve() != (HERE / "check_causality.py").resolve():
            raise JobError(4, "preflight", "runner and pinned causality tool are from different trees")
    g2_path = regular_tracked(repo, job["g2_receipt"], job["g2_receipt_sha256"], "G2 receipt")
    validate_g2(read_json(g2_path), job, sha256(prepare), test_mode)
    try:
        eval_cmd = build_eval_cmd(cfg, job, out_dir)
        causal_cmd = build_causal_cmd(cfg, job, out_dir)
    except JobError as exc:
        raise JobError(4, "preflight", str(exc)) from exc
    evaluator_timeout = int(option_values(eval_cmd, "--timeout-seconds")[0])
    if job.get("eval_timeout_s", 0) <= evaluator_timeout:
        raise JobError(4, "preflight", "outer eval timeout must exceed evaluator timeout")
    if job.get("timeout_s", 0) <= 1800 or job.get("causality_timeout_s", 0) <= 0:
        raise JobError(4, "preflight", "outer training/causality timeouts are not safe positive bounds")
    ledger = repo / "research/v2/ledger.jsonl"

    with host_lease(cfg):
        checks, attempt = preflight(cfg, job, d, repo, out_dir, env)
        write_new(d / "claim.json", {"schema": "launch-claim-v1", "label": label, "utc": now(),
                                      "job_sha256": sha256(job_path), "preflight_attempt_sha256": sha256(attempt)})
        write_new(d / "preflight.json", checks)
        train_cmd = list(cfg["train_cmd_override"]) if test_mode else [
            "torchrun", "--standalone", "--nproc_per_node=4", job["train_py"]]
        train_cmd += job["argv"] + ["--out-dir", str(out_dir.relative_to(repo))]
        rc, runtime = launch_and_wait(cfg, d, repo, label, train_cmd, "run.log",
                                      job.get("timeout_s", 3600), env, "train")
        ckpt = out_dir / "final.pt"
        if rc is None:
            quiesce(label, d, runtime, [d / "run.log"], cfg.get("quiescence_window_s", 10),
                    cfg.get("stop_grace_s", 60), True)
            raise JobError(5, "train", "training timeout; stopped")
        q = quiesce(label, d, runtime, [d / "run.log"] + ([ckpt] if ckpt.exists() else []),
                    cfg.get("quiescence_window_s", 10), cfg.get("stop_grace_s", 60))
        if rc:
            raise JobError(5, "train", f"training failed rc={rc}")
        summary = parse_training_log(d / "run.log")
        identity = checkpoint_identity(ckpt, out_dir)
        write_new(d / "train-summary.json", {**summary, "exit_code": rc,
                                               "train_py_sha256": job["train_sha256"],
                                               "checkpoint_sha256": identity["sha256"],
                                               "quiescence_sha256": sha256(d / "quiescence-train.json")})

        clean_env = {k: v for k, v in env.items() if k not in DIST_ENV}
        erc, eruntime = launch_and_wait(cfg, d, repo, f"{label}-eval", eval_cmd, "eval.log",
                                        job.get("eval_timeout_s", 2400), clean_env, "eval")
        qe = quiesce(f"{label}-eval", d, eruntime, [d / "eval.log"],
                     cfg.get("quiescence_window_s", 10), cfg.get("stop_grace_s", 60), erc is None)
        if erc:
            raise JobError(6, "eval", f"evaluation failed rc={erc}")
        ev = parse_eval(d / "eval.log", evaluator_timeout)
        if checkpoint_identity(ckpt, out_dir) != identity:
            raise JobError(6, "eval", "checkpoint changed during evaluation")
        write_new(d / "eval.json", {**ev, "eval_seq_len": EVAL_SEQ, "argv": eval_cmd,
                                      "checkpoint_sha256": identity["sha256"],
                                      "quiescence_sha256": sha256(d / "quiescence-eval.json")})

        crc, cruntime = launch_and_wait(cfg, d, repo, f"{label}-causality", causal_cmd, "causality.log",
                                        job.get("causality_timeout_s", 1200), clean_env, "causality")
        qc = quiesce(f"{label}-causality", d, cruntime, [d / "causality.log"],
                     cfg.get("quiescence_window_s", 10), cfg.get("stop_grace_s", 60), crc is None)
        if crc:
            raise JobError(6, "causality", f"causality check failed rc={crc}")
        causal = parse_causality(d / "causality.log", job, identity["sha256"])
        if checkpoint_identity(ckpt, out_dir) != identity:
            raise JobError(6, "causality", "checkpoint changed during causality check")
        write_new(d / "causality.json", {**causal, "argv": causal_cmd,
                                           "quiescence_sha256": sha256(d / "quiescence-causality.json")})

        if (git(repo, "rev-parse", "HEAD") != read_json(d / "preflight.json")["git_head"]
                or sha256(train) != job["train_sha256"]
                or sha256(prepare) != job["prepare_py_sha256"] or sha256(g2_path) != job["g2_receipt_sha256"]
                or (decide_path and sha256(decide_path) != DECIDE_SHA256)
                or (causality_tool and sha256(causality_tool) != job["causality_tool_sha256"])):
            raise JobError(7, "postflight", "HEAD or pinned source/G2 changed during run")
        raw_status, allowed_status, unexpected = postflight_status(repo, d, out_dir, ledger)
        if unexpected:
            raise JobError(7, "postflight", f"unexpected repository changes: {unexpected}")

        maybe_inject(cfg, "archive")
        uri, archive_receipt = archive_checkpoint(cfg, ckpt, identity)
        write_new(d / "archive.json", archive_receipt)
        maybe_inject(cfg, "record")
        anomalies = [f"{phase}: tagged descendant outlived leader and containment stopped it"
                     for phase, receipt in (("train", q), ("eval", qe), ("causality", qc))
                     if receipt.get("intervened")]
        rec = {
            "schema": "trn-run-record-v2", "label": label, "arm_id": job["arm_id"], "kind": job["kind"],
            "host": cfg["host"], "timestamp_utc": now(), "base_sha256": job["base_sha256"],
            "train_py_sha256": job["train_sha256"], "prepare_py_sha256": job["prepare_py_sha256"],
            "decide_py_sha256": DECIDE_SHA256,
            "causality_tool_sha256": sha256(causality_tool) if causality_tool else sha256(Path(cfg["causality_cmd_override"][1])),
            "git_commit": git(repo, "rev-parse", "HEAD"), "git_dirty": bool(unexpected),
            "git_status_raw": raw_status, "git_status_allowed": allowed_status, "argv": job["argv"],
            "env": {k: env.get(k) for k in ("NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS", "NEURON_COMPILE_CACHE_DIR")},
            "g2_receipt_sha256": sha256(g2_path), "preflight_receipt_sha256": sha256(d / "preflight.json"),
            "quiescence_receipt_sha256": sha256(d / "quiescence-train.json"),
            "eval_quiescence_receipt_sha256": sha256(d / "quiescence-eval.json"),
            "causality_quiescence_receipt_sha256": sha256(d / "quiescence-causality.json"),
            "archive_receipt_sha256": sha256(d / "archive.json"),
            "status": "evaluated", "steps": summary["num_steps"], "training_seconds": summary["training_seconds"],
            "startup_s": summary["startup_allowance_est"], "over_budget": summary["over_budget"],
            "epoch_boundary_step": summary["epoch_boundary_step"], "median_step_s": summary["median_step_s"],
            "causality_passed": True, "causality_max_abs": causal["max_abs_diff"],
            "eval_tokens": ev["num_tokens"], "eval_seq_len": EVAL_SEQ, "eval_num_bytes": ev["num_bytes"],
            "eval_total_nats": ev["total_nats"], "public_20m": ev["val_bpb"],
            "checkpoint_sha256": identity["sha256"], "checkpoint_uri": uri, "anomalies": anomalies,
        }
        write_new(d / "record.json", rec)
        maybe_inject(cfg, "ledger")
        append_ledger(ledger, rec)
        maybe_inject(cfg, "verdict")
        rows = decide.load(str(ledger))
        verdict = decide.drift(rows) if job["kind"] == "control" else decide.decide_arm(rows, job["arm_id"])
        if anomalies:
            verdict = {**verdict, "human_alert": "containment intervention recorded; Supervisor disposition required"}
        write_new(d / "verdict.json", verdict)
        protected = job.get("kind") == "control" or label in cfg.get("protected_labels", [])
        if not protected:
            write_new(d / "prune-authorized.json", {"schema": "checkpoint-prune-authorization-v1",
                                                      "utc": now(), "path": str(ckpt),
                                                      "sha256": identity["sha256"],
                                                      "archive_receipt_sha256": sha256(d / "archive.json"),
                                                      "record_sha256": sha256(d / "record.json"),
                                                      "verdict_sha256": sha256(d / "verdict.json")})
            write_new(d / "prune.json", prune(ckpt, out_dir, identity))
        return {"label": label, "public_20m": ev["val_bpb"], "steps": rec["steps"],
                "verdict": verdict.get("verdict", "control"), "drift_alarms": verdict.get("drift_alarms"),
                "anomalies": anomalies, "checkpoint_uri": uri}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-config", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    repo, job = Path(args.repo).resolve(), None
    try:
        cfg = read_json(Path(args.host_config).resolve())
        job_path = Path(args.job).resolve()
        job = read_json(job_path)
        ensure_label(job.get("label"))
        result = run(cfg, job, repo, job_path)
        print(json.dumps(result, indent=2))
        return 0
    except JobError as exc:
        label = job.get("label") if isinstance(job, dict) else "unknown"
        try:
            if isinstance(label, str) and LABEL_RE.fullmatch(label):
                d = repo / "research/v2/experiments" / label
                if d.is_dir() and not d.is_symlink():
                    write_new(d / f"failure-{exc.stage}-{time.time_ns()}-{os.getpid()}.json",
                              {"stage": exc.stage, "code": exc.code, "error": str(exc), "utc": now()})
        except (OSError, JobError):
            pass
        print(json.dumps({"label": label, "failed_stage": exc.stage,
                          "exit_code": exc.code, "error": str(exc)}, indent=2))
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
