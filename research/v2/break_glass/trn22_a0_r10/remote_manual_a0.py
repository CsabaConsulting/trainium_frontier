#!/usr/bin/env python3
"""One-use, host-side break-glass A0 control. No SSH, AWS, or cleanup actions.

The Operator copies this file and the exact train.py into a new private package
directory by scp. This program runs only after a distinct reviewed Supervisor
assignment is copied there. It keeps the old host-wide lock through training,
fresh-process public-20M evaluation, and postflight, preserving all artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time


LABEL = "fs-20260925-v2-c5c-a0-control-trn22-r10"
ROOT = Path("/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2")
PACKAGE = ROOT / "out" / (LABEL + "-manual-source-r1")
TRAIN_WORK = ROOT / "out" / (LABEL + "-manual-work") / "train"
EVAL_WORK = ROOT / "out" / (LABEL + "-manual-work") / "eval"
REHEARSAL = ROOT / "research/scratch/submission_rehearsal.py"
LOCK = Path("/home/ubuntu/.cache/trainium_freshstart/device0.lock")
PYTHON = "/home/ubuntu/workspace/native_venv/bin/python"
HOSTNAME = "ip-172-31-32-253"
HOST_COMMIT = "e65a3097a5b00f6178188a935bb2acb9a5042c6c"
TRAIN_SHA = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREPARE_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
REHEARSAL_SHA = "7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b"
G2_SHA = "38dd39a02d86d639f6333228048f8bdb9dcc77735f03c2d45963e1057bf945b9"
FLOOR = 13 * 1024**3 // 2
ARGV = [
    "--n-embd", "1024", "--no-ngram-ve", "--no-qk-shift",
    "--compile-sdpa-direct", "--nki-local-conv", "--bf16-norm-output",
    "--pack-factor", "32", "--clip-after-reduce", "--seq-len", "1024",
    "--no-eval-public", "--matrix-lr", "0.015",
]
LAUNCH = (
    "NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py "
    + " ".join(ARGV)
)
CACHE_ROOT = Path(
    "/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1"
    "/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train"
)
CACHE_NAMES = {
    "TORCH_NEURONX_NEFF_CACHE_DIR": "neff",
    "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": "local",
    "TORCH_NEURONX_HLO_CACHE_DIR": "hlo",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "NEURON_COMPILE_CACHE_DIR": "compiler",
    "TORCH_NEURONX_DEBUG_DIR": "debug",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path: Path, value: dict) -> str:
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def plain_owned_dir(path: Path) -> None:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise RuntimeError("expected owned real directory: " + str(path))
    if metadata.st_mode & 0o022:
        raise RuntimeError("directory is group/world writable: " + str(path))


def fuser_clear() -> bool:
    devices = sorted(Path("/dev").glob("neuron[0-9]*"))
    if not devices:
        raise RuntimeError("Neuron device node is absent")
    for device in devices:
        result = subprocess.run(
            ["/usr/bin/fuser", "-s", str(device)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
        if result.returncode == 0:
            return False
        if result.returncode != 1:
            raise RuntimeError("Neuron device-holder check failed")
    return True


def job_processes() -> list[int]:
    found = []
    names = {
        "torchrun", "train.py", "prepare.py", "run_experiment.py",
        "submission_rehearsal.py", "fresh_start_guard.py", "phase_timer.py",
    }
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            words = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if any(Path(w.decode(errors="replace")).name in names for w in words if w):
            found.append(int(entry.name))
    return found


def require_idle() -> None:
    if job_processes() or not fuser_clear():
        raise RuntimeError("another training/evaluation process or Neuron holder is present")


def require_disk() -> int:
    free = shutil.disk_usage(ROOT).free
    if free < FLOOR:
        raise RuntimeError(f"root free bytes {free} below {FLOOR}")
    return free


def environment() -> dict[str, str]:
    env = {
        "HOME": "/home/ubuntu",
        "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
        "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv",
        "PYTHONUNBUFFERED": "1",
        "NEURON_LOGICAL_NC_CONFIG": "2",
        "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
        "NEURON_COMPETITION_R1_SEED": "42",
        "NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS": "1800",
    }
    for key, name in CACHE_NAMES.items():
        path = CACHE_ROOT / name
        plain_owned_dir(path)
        env[key] = str(path)
    return env


def preflight() -> dict:
    if sys.flags.optimize or "PYTHONOPTIMIZE" in os.environ:
        raise RuntimeError("optimized Python is forbidden")
    if os.getuid() != 1000 or socket.gethostname() != HOSTNAME:
        raise RuntimeError("wrong host or user")
    plain_owned_dir(ROOT)
    plain_owned_dir(PACKAGE)
    if Path(__file__).resolve() != PACKAGE / "remote_manual_a0.py":
        raise RuntimeError("manual runner is not in its unique package directory")
    commit = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
    ).strip()
    if commit != HOST_COMMIT:
        raise RuntimeError("host source checkout commit changed")
    dirty = subprocess.check_output(
        ["/usr/bin/git", "status", "--porcelain"], cwd=ROOT, text=True, timeout=10
    )
    if dirty:
        raise RuntimeError("host source checkout is dirty")
    expected = (
        (PACKAGE / "train.py", TRAIN_SHA),
        (ROOT / "prepare.py", PREPARE_SHA),
        (REHEARSAL, REHEARSAL_SHA),
    )
    for path, digest in expected:
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise RuntimeError("source SHA-256 mismatch: " + str(path))
    g2_path = PACKAGE / "G2_PASS.json"
    if not g2_path.is_file() or g2_path.is_symlink() or sha(g2_path) != G2_SHA:
        raise RuntimeError("exact G2 PASS receipt missing or mismatched")
    g2 = json.loads(g2_path.read_text())
    if (g2.get("verdict") != "PASS" or g2.get("torchrun_rc") != 0
            or g2.get("nproc") != 4 or g2.get("argv") != ARGV
            or g2.get("train_py_sha256") != TRAIN_SHA
            or g2.get("prepare", {}).get("sha256") != PREPARE_SHA):
        raise RuntimeError("G2 PASS receipt does not bind exact bytes and argv")
    provenance = json.loads((PACKAGE / "provenance.json").read_text())
    if provenance != {
        "schema": "trainium-break-glass-a0-trn22-r10-provenance-v1",
        "label": LABEL, "host": "trn22",
        "source_commit": "22bfbab5a77d6feb670d2d737504f8b70b08ddc1",
        "train_sha256": TRAIN_SHA, "prepare_sha256": PREPARE_SHA,
        "launch_command": LAUNCH,
    }:
        raise RuntimeError("manual provenance mismatch")
    if (PACKAGE / "launch-command.txt").read_text().strip() != LAUNCH:
        raise RuntimeError("manual launch command mismatch")
    assignment = json.loads((PACKAGE / "SUPERVISOR_ASSIGNMENT.json").read_text())
    if (
        assignment.get("schema") != "trainium-break-glass-assignment-v1"
        or assignment.get("scope") != "TRN22_A0_MANUAL_CONTROL_ONLY"
        or assignment.get("label") != LABEL
        or assignment.get("host") != "trn22"
        or assignment.get("train_sha256") != TRAIN_SHA
        or assignment.get("g2_receipt_sha256") != G2_SHA
        or assignment.get("launch_authorized") is not True
        or assignment.get("submission_authorized") is not False
    ):
        raise RuntimeError("missing or mismatched Supervisor assignment")
    expiry = datetime.fromisoformat(assignment["expires_at_utc"].replace("Z", "+00:00"))
    if expiry.tzinfo is None or datetime.now(timezone.utc) >= expiry:
        raise RuntimeError("Supervisor assignment expired")
    if (TRAIN_WORK.exists() or EVAL_WORK.exists()
            or (PACKAGE / "attempt.json").exists()
            or (PACKAGE / "run-record.json").exists()):
        raise RuntimeError("manual label or work path was already attempted")
    env = environment()
    free = require_disk()
    require_idle()
    return {
        "schema": "trainium-break-glass-a0-preflight-v1",
        "label": LABEL, "host": "trn22", "checked_utc": utc(),
        "host_commit": commit, "host_git_dirty": False,
        "train_py_sha256": TRAIN_SHA, "prepare_py_sha256": PREPARE_SHA,
        "rehearsal_sha256": REHEARSAL_SHA, "g2_receipt_sha256": G2_SHA,
        "assignment_sha256": sha(PACKAGE / "SUPERVISOR_ASSIGNMENT.json"),
        "root_free_bytes": free, "floor_bytes": FLOOR,
        "no_other_job": True, "neuron_device_clear": True,
        "cache_paths": {k: env[k] for k in CACHE_NAMES},
        "effective_environment": env,
        "argv": ARGV,
    }


def terminate_owned(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    # start_new_session made this child's group unique. It cannot be reused
    # while the direct child remains unreaped here.
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def phase(name: str, command: list[str], env: dict[str, str], limit: int) -> dict:
    log = PACKAGE / (name + ".log")
    started = time.monotonic()
    with log.open("xb") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(json.dumps({"event": name + "_DRIVER_STARTED", "pid": process.pid,
                          "label": LABEL, "utc": utc()}), flush=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started > limit:
                    terminate_owned(process)
                    raise RuntimeError(name + " exceeded outer wall limit")
                if shutil.disk_usage(ROOT).free < FLOOR:
                    terminate_owned(process)
                    raise RuntimeError(name + " stopped at 6.5-GiB disk floor")
                time.sleep(1)
        except BaseException:
            terminate_owned(process)
            raise
        stream.flush()
        os.fsync(stream.fileno())
    if process.returncode != 0:
        raise RuntimeError(f"{name} returned {process.returncode}; full error in {log}")
    for _ in range(30):
        if not job_processes() and fuser_clear():
            break
        time.sleep(1)
    else:
        raise RuntimeError(name + " left a process or Neuron holder")
    return {"returncode": process.returncode,
            "wall_seconds": time.monotonic() - started, "log": str(log),
            "log_sha256": sha(log), "finished_utc": utc()}


def eval_metrics(log: Path) -> dict:
    text = log.read_text(errors="replace")
    decoder = json.JSONDecoder()
    found = []
    for offset, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "val_bpb" in value:
            found.append(value)
    if len(found) != 1:
        raise RuntimeError("public-20M metric JSON missing or duplicated")
    value = found[0]
    if (value.get("num_tokens") != 20971520
            or value.get("num_bytes") != 83462186
            or value.get("eval_timeout") != 1200
            or value.get("eval_over_timeout") is not False):
        raise RuntimeError("public-20M metric schema/count/timeout mismatch")
    for key in ("val_bpb", "total_nats", "eval_seconds"):
        if not isinstance(value.get(key), (int, float)) or not math.isfinite(value[key]):
            raise RuntimeError("invalid public metric: " + key)
    check = value["total_nats"] / (value["num_bytes"] * math.log(2))
    if not math.isclose(value["val_bpb"], check, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError("public BPB arithmetic mismatch")
    if value["eval_seconds"] > 1200:
        raise RuntimeError("public evaluator exceeded timeout")
    return value


def main() -> int:
    terminal = {"schema": "trainium-break-glass-a0-terminal-v1",
                "label": LABEL, "host": "trn22", "started_utc": utc(),
                "status": "FAILED", "error": None}
    record = None
    lock_fd = None
    try:
        if Path(__file__).parent != PACKAGE:
            raise RuntimeError("wrong manual package location")
        lock_fd = os.open(LOCK, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = preflight()
        before_sha = write_new(PACKAGE / "preflight.json", before)
        write_new(PACKAGE / "attempt.json", {"label": LABEL, "host": "trn22",
                                             "started_utc": utc(), "one_attempt": True})
        env = environment()
        train_command = [PYTHON, str(REHEARSAL), "--package", str(PACKAGE),
                         "--sandbox", str(TRAIN_WORK), "--mode", "train"]
        training = phase("train", train_command, env, 4200)
        receipt = json.loads((TRAIN_WORK / "rehearsal.json").read_text())
        checkpoint = TRAIN_WORK / "out/final.pt"
        if (receipt.get("returncode") != 0
                or receipt.get("step_boundaries", 0) < 5
                or receipt.get("charged_through_process_exit_within_budget") is not True
                or not checkpoint.is_file() or checkpoint.is_symlink()):
            raise RuntimeError("training boundaries, budget, or checkpoint failed")
        checkpoint_sha = sha(checkpoint)
        if checkpoint_sha != receipt.get("checkpoint_sha256"):
            raise RuntimeError("checkpoint SHA mismatch after training")
        require_disk()
        evaluation = phase("eval", [PYTHON, str(REHEARSAL), "--package", str(PACKAGE),
                                    "--sandbox", str(EVAL_WORK), "--mode", "eval",
                                    "--checkpoint", str(checkpoint), "--eval-tokens", "20971520"],
                           env, 1800)
        metrics = eval_metrics(PACKAGE / "eval.log")
        if sha(checkpoint) != checkpoint_sha:
            raise RuntimeError("checkpoint changed during evaluation")
        quiescence = {"label": LABEL, "host": "trn22", "checked_utc": utc(),
                      "neuron_device_clear": fuser_clear(),
                      "no_job_process": not job_processes(),
                      "root_free_bytes": require_disk()}
        if not quiescence["neuron_device_clear"] or not quiescence["no_job_process"]:
            raise RuntimeError("post-eval host not quiescent")
        quiescence_sha = write_new(PACKAGE / "quiescence.json", quiescence)
        record = {
            "schema": "trn-run-record-v2", "label": LABEL, "arm_id": "A0",
            "kind": "control", "host": "trn22", "timestamp_utc": utc(),
            "base_sha256": TRAIN_SHA, "train_py_sha256": TRAIN_SHA,
            "git_commit": HOST_COMMIT, "git_dirty": False, "argv": ARGV,
            "env": {**env, "cache_state": "warm", "manual_break_glass": True},
            "g2_receipt_sha256": G2_SHA,
            "preflight_receipt_sha256": before_sha,
            "quiescence_receipt_sha256": quiescence_sha,
            "status": "evaluated", "steps": receipt["step_boundaries"],
            "training_seconds": receipt["charged_through_process_exit"],
            "startup_s": receipt.get("startup_allowance_observed"),
            "over_budget": False, "epoch_boundary_step": None,
            "median_step_s": None, "causality_passed": None,
            "eval_tokens": metrics["num_tokens"], "eval_seq_len": 1024,
            "public_20m": metrics["val_bpb"],
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_uri": str(checkpoint),
            "notes": "Exact c5c/argv break-glass manual A0, warm cache; G2 on the same bytes/argv."
        }
        write_new(PACKAGE / "run-record.json", record)
        terminal.update(status="EVALUATED", error=None, training=training,
                        evaluation=evaluation, checkpoint_sha256=checkpoint_sha,
                        steps=record["steps"], public_20m=record["public_20m"])
        print(json.dumps({"event": "A0_EVALUATED", "label": LABEL,
                          "steps": record["steps"], "public_20m": record["public_20m"]}),
              flush=True)
        return 0
    except BaseException as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps({"event": "A0_FAILED", "label": LABEL,
                          "error": terminal["error"]}), flush=True)
        return 1
    finally:
        terminal["finished_utc"] = utc()
        try:
            terminal["postflight_root_free_bytes"] = shutil.disk_usage(ROOT).free
            terminal["postflight_neuron_clear"] = fuser_clear()
            terminal["postflight_job_processes"] = job_processes()
        except BaseException as error:
            terminal["postflight_error"] = f"{type(error).__name__}: {error}"
        try:
            write_new(PACKAGE / "terminal.json", terminal)
        except BaseException as error:
            print("TERMINAL_RECEIPT_WRITE_FAILED: " + repr(error), flush=True)
        if lock_fd is not None:
            os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
