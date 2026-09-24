#!/usr/bin/env python3
"""One-label, host-local disk monitor for the retained-checkpoint G1 calibration.

The already authenticated fresh_start_launch/guard runs this as its *child*.
This wrapper starts only the exact public evaluator recorded in the approved
job manifest, monitors root free space while it runs, and stops only its own
authenticated child process group. It does not alter model or evaluator code.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time


LABEL = "fs-20260925-g1-public20m-calibration-trn21-r2"
ROOT = Path("/home/ubuntu/trainium_freshstart_20260925_g1_public20m_calibration_trn21_r2")
RUN_LOGS = ROOT / "research/run-logs"
JOB = RUN_LOGS / (LABEL + ".job.json")
EVENTS = RUN_LOGS / (LABEL + ".disk-monitor.jsonl")
HOSTNAME = "ip-172-31-34-85"
TRAIN_REL = "submissions/dense6x1024_direct_review_matrixlr015/train.py"
PREPARE_SHA256 = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
TRAIN_SHA256 = "116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d"
CHECKPOINT = Path("/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/train/out/final.pt")
CHECKPOINT_BYTES = 469831009
CHECKPOINT_SHA256 = "5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9"
FLOOR_BYTES = 6_979_321_856  # immutable 6.5 GiB hard floor
RESERVED_BYTES = 2_147_483_648  # stop before the 2 GiB reserve is consumed
STOP_BELOW_BYTES = FLOOR_BYTES + RESERVED_BYTES
POLL_SECONDS = 1.0
MARKER = "TRAINIUM_FRESH_START_RUN_ID"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular_owned(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError("required file is not an owned regular file: " + str(path))


def append_event(stream, event: str, **fields: object) -> None:
    row = {"event": event, "label": LABEL, "monotonic_seconds": time.monotonic(), **fields}
    stream.write(json.dumps(row, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def process_identity(pid: int) -> tuple[int, int, int] | None:
    proc = Path("/proc") / str(pid)
    try:
        uid = proc.stat().st_uid
        fields = (proc / "stat").read_text().rpartition(")")[2].split()
        # Linux proc fields after the closing parenthesis start at field 3.
        return uid, int(fields[19]), int(fields[2])  # uid, start ticks, process group
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None


def stop_owned_child(child: subprocess.Popen, start_ticks: int, event_stream, reason: str) -> bool:
    current = process_identity(child.pid)
    if child.poll() is not None:
        append_event(event_stream, "stop_not_needed_child_exited", reason=reason)
        return True
    if current != (os.getuid(), start_ticks, child.pid):
        append_event(event_stream, "stop_refused_identity_mismatch", reason=reason)
        return False
    try:
        env_items = (Path("/proc") / str(child.pid) / "environ").read_bytes().split(b"\0")
        argv = (Path("/proc") / str(child.pid) / "cmdline").read_bytes().split(b"\0")
        if (MARKER + "=" + LABEL).encode() not in env_items or not argv or argv[0].decode() != str(EVAL_ARGV[0]):
            append_event(event_stream, "stop_refused_marker_or_argv_mismatch", reason=reason)
            return False
        os.killpg(child.pid, signal.SIGTERM)
        append_event(event_stream, "stop_term_sent", reason=reason, pid=child.pid, start_ticks=start_ticks)
        try:
            child.wait(timeout=5)
            return True
        except subprocess.TimeoutExpired:
            if process_identity(child.pid) == (os.getuid(), start_ticks, child.pid):
                os.killpg(child.pid, signal.SIGKILL)
                append_event(event_stream, "stop_kill_sent", reason=reason, pid=child.pid, start_ticks=start_ticks)
                child.wait(timeout=5)
                return True
            append_event(event_stream, "stop_kill_refused_identity_changed", reason=reason)
            return False
    except (FileNotFoundError, ProcessLookupError, PermissionError, subprocess.TimeoutExpired) as error:
        append_event(event_stream, "stop_error", reason=reason, error_type=type(error).__name__)
        return False


def make_child_parent_death_signal(parent_pid: int):
    def set_parent_death_signal() -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            os._exit(127)
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)
    return set_parent_death_signal


EVAL_ARGV = [
    "/home/ubuntu/workspace/native_venv/bin/python",
    str(ROOT / "prepare.py"),
    "eval-public",
    "--train-py", str(ROOT / TRAIN_REL),
    "--checkpoint", str(CHECKPOINT),
    "--device", "neuron",
    "--batch-size", "1",
    "--seq-len", "1024",
    "--eval-tokens", "20971520",
    "--timeout-seconds", "1200",
]


def authenticate(job: dict) -> None:
    if os.getuid() != 1000 or Path.home() != Path("/home/ubuntu") or socket.gethostname() != HOSTNAME:
        raise RuntimeError("wrong host, effective user, or HOME")
    if job.get("launch_authorized") is not True:
        raise RuntimeError("one-label Supervisor authorization is absent")
    auth = job.get("authorization")
    if not isinstance(auth, dict) or auth.get("decision") != "AUTHORIZE_ONE_LABEL_G1_EVALUATOR":
        raise RuntimeError("one-label Supervisor authorization is invalid")
    if auth.get("stage_review_sha256") != "bf3c78a7db65ea7829386129969e7bb06cc700f11c2d4b2f251c448f04878a21":
        raise RuntimeError("stage review binding mismatch")
    if job.get("label") != LABEL or job.get("host") != "trn21" or job.get("expected_hostname") != HOSTNAME:
        raise RuntimeError("wrong label or host binding")
    if job.get("remote") != str(ROOT) or job.get("expected_commit") != "e7f11c38dc3b5e68b373322e10cfef580f1e8d36":
        raise RuntimeError("wrong staged source binding")
    if job.get("calibration_eval_command") != EVAL_ARGV:
        raise RuntimeError("evaluator argv differs from the pinned public-20M command")
    if job.get("monitor", {}).get("script_path") != str(Path(__file__).resolve()):
        raise RuntimeError("monitor script path mismatch")
    if job["monitor"].get("script_sha256") != digest(Path(__file__)):
        raise RuntimeError("monitor script hash mismatch")
    if job["monitor"].get("floor_bytes") != FLOOR_BYTES or job["monitor"].get("stop_below_bytes") != STOP_BELOW_BYTES:
        raise RuntimeError("disk threshold mismatch")
    if job["monitor"].get("poll_seconds") != POLL_SECONDS:
        raise RuntimeError("monitor cadence mismatch")
    if job.get("expected_sha256", {}).get(TRAIN_REL) != TRAIN_SHA256 or job.get("expected_sha256", {}).get("prepare.py") != PREPARE_SHA256:
        raise RuntimeError("source manifest hash mismatch")
    if job.get("checkpoint", {}).get("path") != str(CHECKPOINT) or job["checkpoint"].get("expected_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint manifest mismatch")
    if job["checkpoint"].get("expected_bytes") != CHECKPOINT_BYTES:
        raise RuntimeError("checkpoint size manifest mismatch")
    # The generic launcher authenticates the runtime closure and holds the
    # host lock. This wrapper independently rechecks its science inputs.
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if head != job["expected_commit"]:
        raise RuntimeError("staged HEAD changed")
    if subprocess.check_output(["git", "-C", str(ROOT), "diff", "--name-only", "HEAD", "--", "prepare.py", TRAIN_REL]).strip():
        raise RuntimeError("required source path became dirty")
    for path, expected in ((ROOT / "prepare.py", PREPARE_SHA256), (ROOT / TRAIN_REL, TRAIN_SHA256), (CHECKPOINT, CHECKPOINT_SHA256)):
        regular_owned(path)
        if digest(path) != expected:
            raise RuntimeError("source or checkpoint hash changed: " + str(path))
    if CHECKPOINT.stat().st_size != CHECKPOINT_BYTES:
        raise RuntimeError("checkpoint size changed")
    if not Path(EVAL_ARGV[0]).is_file() or not os.access(EVAL_ARGV[0], os.X_OK):
        raise RuntimeError("native Python missing")
    expected_env = job.get("environment", {})
    for key, value in expected_env.items():
        if os.environ.get(key) != value:
            raise RuntimeError("effective environment differs: " + key)
    forbidden_exact = set(job.get("environment_must_be_unset", []))
    forbidden_prefixes = tuple(job.get("environment_must_be_unset_prefixes", []))
    inherited = sorted(key for key in os.environ if key in forbidden_exact or key.startswith(forbidden_prefixes))
    if inherited:
        raise RuntimeError("forbidden environment names present: " + ",".join(inherited))
    for cache_key in ("TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "TORCH_NEURONX_HLO_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "NEURON_COMPILE_CACHE_DIR", "TORCH_NEURONX_DEBUG_DIR"):
        cache = Path(expected_env[cache_key])
        if not cache.is_dir() or cache.is_symlink():
            raise RuntimeError("historical warm cache is absent or symlinked: " + cache_key)
    if shutil.disk_usage(ROOT).free < STOP_BELOW_BYTES:
        raise RuntimeError("pre-evaluator disk below floor plus reserved headroom")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    if args.job != JOB:
        raise RuntimeError("unexpected job manifest path")
    regular_owned(JOB)
    job = json.loads(JOB.read_text())
    authenticate(job)

    stopped_by_signal: int | None = None
    child: subprocess.Popen | None = None
    child_start_ticks: int | None = None
    def request_stop(signum, _frame) -> None:
        nonlocal stopped_by_signal
        stopped_by_signal = signum
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    with EVENTS.open("x", encoding="utf-8") as events:
        append_event(events, "monitor_ready", free_bytes=shutil.disk_usage(ROOT).free,
                     floor_bytes=FLOOR_BYTES, stop_below_bytes=STOP_BELOW_BYTES,
                     poll_seconds=POLL_SECONDS, job_sha256=digest(JOB))
        child = subprocess.Popen(EVAL_ARGV, cwd=ROOT, env=os.environ.copy(),
                                 start_new_session=True,
                                 preexec_fn=make_child_parent_death_signal(os.getpid()))
        identity = process_identity(child.pid)
        if identity is None or identity[0] != os.getuid() or identity[2] != child.pid:
            raise RuntimeError("evaluator child identity could not be authenticated")
        child_start_ticks = identity[1]
        append_event(events, "evaluator_started", pid=child.pid, start_ticks=child_start_ticks)
        samples = 0
        min_free = shutil.disk_usage(ROOT).free
        reason = None
        while child.poll() is None:
            try:
                free = shutil.disk_usage(ROOT).free
            except OSError:
                free = None
                reason = "disk_sample_failed"
            samples += 1
            if free is not None:
                min_free = min(min_free, free)
                if samples == 1 or samples % 10 == 0:
                    append_event(events, "disk_sample", sample=samples, free_bytes=free)
                if free < STOP_BELOW_BYTES:
                    reason = "reserved_headroom_crossed"
            if stopped_by_signal is not None:
                reason = "wrapper_signal_" + str(stopped_by_signal)
            if reason is not None:
                stop_owned_child(child, child_start_ticks, events, reason)
                break
            time.sleep(POLL_SECONDS)
        rc = child.wait(timeout=10) if child.poll() is None else child.returncode
        append_event(events, "monitor_finished", evaluator_returncode=rc,
                     sample_count=samples, minimum_free_bytes=min_free,
                     stopped_reason=reason, final_free_bytes=shutil.disk_usage(ROOT).free)
        return rc if rc != 0 else (2 if reason else 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("G1_MONITOR_FAILED " + type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(3)
