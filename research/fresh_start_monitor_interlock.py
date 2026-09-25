#!/usr/bin/env python3
"""Fail-closed launch interlock for one reviewed paired root-space monitor.

Called by fresh_start_launch immediately before it creates a job/spawns a guard.
It never launches, signals, or changes a process. The operator must separately
authorize the exact one-label request and start both external monitors first.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import time


PAIR_SCHEMA = "paired-root-free-guard-monitor-v1"
AUTH_SCHEMA = "one-shot-root-free-monitor-authorization-v1"
MAX_HEARTBEAT_AGE = 1.0
MAX_FILE_BYTES = 1024 * 1024


class InterlockError(RuntimeError):
    """No guard may be spawned when this error is raised."""


@dataclass(frozen=True)
class Process:
    pid: int
    uid: int
    birth_ticks: int
    argv: tuple[str, ...]
    exe: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def secure_bytes(path: Path, uid: int, *, limit: int = MAX_FILE_BYTES) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != uid
                or info.st_mode & 0o077 or info.st_size > limit):
            raise InterlockError("external evidence owner, mode, type, or size mismatch")
        data = os.read(fd, limit + 1)
        if len(data) > limit:
            raise InterlockError("external evidence exceeds size limit")
        return data
    finally:
        os.close(fd)


def secure_json(path: Path, uid: int) -> dict:
    value = json.loads(secure_bytes(path, uid))
    if not isinstance(value, dict):
        raise InterlockError("expected external JSON object")
    return value


def sha256_secure(path: Path, uid: int) -> str:
    return hashlib.sha256(secure_bytes(path, uid)).hexdigest()


def read_proc(pid: int) -> Process | None:
    root = Path("/proc") / str(pid)
    try:
        statline = (root / "stat").read_text()
        tail = statline[statline.rfind(")") + 2:].split()
        uidline = next(line for line in (root / "status").read_text().splitlines()
                       if line.startswith("Uid:"))
        argv = tuple(item.decode("utf-8", "surrogateescape")
                     for item in (root / "cmdline").read_bytes().split(b"\0") if item)
        return Process(pid, int(uidline.split()[1]), int(tail[19]), argv,
                       os.path.realpath(os.readlink(root / "exe")))
    except (FileNotFoundError, ProcessLookupError):
        return None


def recent_events(path: Path, uid: int) -> list[dict]:
    """Return complete fsynced-style JSONL records from the last 64 KiB."""
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != uid or info.st_mode & 0o077:
            raise InterlockError("journal owner, mode, or type mismatch")
        start = max(0, info.st_size - 65536)
        os.lseek(fd, start, os.SEEK_SET)
        raw = os.read(fd, 65536)
    finally:
        os.close(fd)
    lines = raw.split(b"\n")
    if not raw.endswith(b"\n"):
        lines.pop()
    if start:
        if not lines:
            raise InterlockError("journal has no complete events")
        lines.pop(0)
    try:
        return [json.loads(line) for line in lines if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InterlockError("malformed journal event") from error


def root_free_bytes() -> int:
    info = os.statvfs("/")
    return info.f_bavail * info.f_frsize


def _exact_peer_argv(pair: dict, role: str, pair_path: Path) -> tuple[str, ...]:
    return (
        pair["expected_python"], pair["wrapper_path"],
        "--binding", pair["base_binding_path"],
        "--authorization", pair["authorization_path"],
        "--pair-binding", str(pair_path), "--role", role,
    )


def require_live_pair(request: dict) -> dict:
    """Return a receipt-ready snapshot or raise before any guard may spawn."""
    if request.get("dispatch_authorized") is not True:
        raise InterlockError("one-label dispatch is not authorized")
    config = request.get("monitor_interlock")
    if not isinstance(config, dict):
        raise InterlockError("paired monitor interlock is missing")
    pair_path = Path(config["pair_binding_path"])
    uid = os.getuid()
    if not pair_path.is_absolute():
        raise InterlockError("pair binding path is not absolute")
    pair_sha = sha256_secure(pair_path, uid)
    pair = secure_json(pair_path, uid)
    if (pair.get("schema") != PAIR_SCHEMA or pair.get("package_state") != "STATIC_ONLY"
            or pair.get("label") != request.get("label")
            or pair.get("remote") != request.get("remote")
            or pair.get("expected_hostname") != socket.gethostname()
            or pair.get("expected_uid") != uid
            or pair.get("no_spawn_path") != config.get("no_spawn_path")
            or pair.get("interlock_path") != config.get("interlock_path")
            or set(pair.get("roles", {})) != {"primary", "peer"}):
        raise InterlockError("pair binding disagrees with this one-label host")
    base_path = Path(pair["base_binding_path"])
    base_sha = sha256_secure(base_path, uid)
    binding = secure_json(base_path, uid)
    request_path = Path(binding["request_path"])
    request_sha = sha256_secure(request_path, uid)
    original = secure_json(request_path, uid)
    if (base_sha != pair["base_binding_sha256"]
            or request_sha != binding["request_sha256"]
            or request_sha != pair["request_sha256"]
            or any(request.get(key) != value for key, value in original.items())
            or binding.get("label") != request.get("label")
            or binding.get("remote") != request.get("remote")
            or binding.get("expected_commit") != request.get("expected_commit")
            or binding.get("expected_sha256") != request.get("expected_sha256")
            or binding.get("floor_free_bytes") != request.get("min_free_bytes")
            or binding.get("prelaunch_free_bytes") != request.get("pre_launch_required_free_bytes")
            or pair.get("stop_free_bytes") != binding.get("stop_free_bytes")):
        raise InterlockError("request or base binding bytes changed")
    auth_path = Path(pair["authorization_path"])
    auth = secure_json(auth_path, uid)
    if (auth.get("schema") != AUTH_SCHEMA or auth.get("authorized") is not True
            or auth.get("paired_authorized") is not True
            or auth.get("dispatch_authorized") is not True
            or auth.get("one_attempt_only") is not True
            or auth.get("label") != request["label"]
            or auth.get("binding_sha256") != base_sha
            or auth.get("pair_binding_sha256") != pair_sha
            or auth.get("request_sha256") != request_sha):
        raise InterlockError("one-label pair authorization is absent or changed")
    if (sha256_secure(Path(pair["wrapper_path"]), uid) != pair["wrapper_script_sha256"]
            or sha256_secure(Path(pair["base_monitor_path"]), uid)
            != binding["monitor_script_sha256"]):
        raise InterlockError("monitor source bytes changed")
    if request["pre_launch_required_free_bytes"] < 10737418240:
        raise InterlockError("10 GiB launch gate was lowered")
    if root_free_bytes() < request["pre_launch_required_free_bytes"]:
        raise InterlockError("root free below 10 GiB launch gate")

    now = time.monotonic()
    ready: dict[str, dict] = {}
    process: dict[str, Process] = {}
    for role in ("primary", "peer"):
        role_config = pair["roles"][role]
        ready_path = Path(role_config["ready_path"])
        info = secure_json(ready_path, uid)
        if (info.get("schema") != PAIR_SCHEMA
                or info.get("state") != "READY_BEFORE_LAUNCH"
                or info.get("label") != request["label"]
                or info.get("role") != role
                or info.get("pair_binding_sha256") != pair_sha
                or info.get("base_binding_sha256") != base_sha
                or info.get("request_sha256") != request_sha
                or info.get("wrapper_script_sha256") != pair["wrapper_script_sha256"]
                or not isinstance(info.get("pid"), int)
                or not isinstance(info.get("birth_ticks"), int)
                or info.get("uid") != uid):
            raise InterlockError(f"{role} ready identity mismatch")
        current = read_proc(info["pid"])
        if (current is None or current.uid != uid
                or current.birth_ticks != info["birth_ticks"]
                or current.exe != os.path.realpath(pair["expected_python"])
                or current.argv != _exact_peer_argv(pair, role, pair_path)):
            raise InterlockError(f"{role} process identity changed")
        ready[role], process[role] = info, current

    samples: dict[str, dict] = {}
    for role in ("primary", "peer"):
        events = recent_events(Path(pair["roles"][role]["events_path"]), uid)
        other = "peer" if role == "primary" else "primary"
        bound = [e for e in events if e.get("event") == "PEER_BOUND"
                 and e.get("peer_role") == other
                 and e.get("peer_pid") == ready[other]["pid"]
                 and e.get("peer_birth_ticks") == ready[other]["birth_ticks"]]
        samples_here = [e for e in events if e.get("event") == "ROOT_SAMPLE"]
        if not bound or not samples_here:
            raise InterlockError(f"{role} has no mutual heartbeat proof")
        if any(e.get("event") in {"MONITOR_FAILURE", "SIGNAL_REFUSED", "TERM_INTENT",
                                      "MONITOR_FINISHED"} for e in events):
            raise InterlockError(f"{role} recorded a monitor failure or stop")
        sample = samples_here[-1]
        when = sample.get("sample_monotonic")
        if (not isinstance(when, (int, float)) or not math.isfinite(when)
                or when > now + 0.1 or now - when > MAX_HEARTBEAT_AGE
                or sample.get("state") != "WAITING_FOR_GUARD"
                or sample.get("guard_pid") is not None
                or sample.get("free_bytes", 0) < request["pre_launch_required_free_bytes"]):
            raise InterlockError(f"{role} sample stale or unsafe")
        current = read_proc(ready[role]["pid"])
        if current != process[role]:
            raise InterlockError(f"{role} PID changed during interlock")
        samples[role] = sample
    final_now = time.monotonic()
    if any(final_now - samples[role]["sample_monotonic"] > MAX_HEARTBEAT_AGE
           for role in ("primary", "peer")):
        raise InterlockError("monitor heartbeat aged during interlock")
    final_free = root_free_bytes()
    if final_free < request["pre_launch_required_free_bytes"]:
        raise InterlockError("root free fell below launch gate during interlock")
    return {
        "label": request["label"], "pair_binding_sha256": pair_sha,
        "base_binding_sha256": base_sha, "request_sha256": request_sha,
        "monotonic": final_now, "root_free_bytes": final_free,
        "roles": {role: {"pid": ready[role]["pid"],
                         "birth_ticks": ready[role]["birth_ticks"],
                         "sample_monotonic": samples[role]["sample_monotonic"]}
                  for role in ("primary", "peer")},
    }


def _exclusive_fsynced_json(path: Path, payload: dict) -> None:
    if not path.is_absolute():
        raise InterlockError("receipt path is not absolute")
    parent = path.parent.stat(follow_symlinks=False)
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid()
            or parent.st_mode & 0o022):
        raise InterlockError("receipt directory is not privately owned")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        while data:
            data = data[os.write(fd, data):]
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_interlock_pass(request: dict, snapshot: dict) -> None:
    path = Path(request["monitor_interlock"]["interlock_path"])
    _exclusive_fsynced_json(path, {
        "schema": "paired-monitor-pre-spawn-interlock-v1",
        "status": "CHECK_PASSED_NOT_SPAWN_PROOF",
        "label": request["label"], "snapshot": snapshot,
    })


def write_no_spawn(request: dict, stage: str, error: BaseException) -> None:
    """Durably prove no guard spawn only when called before Popen succeeds."""
    path = Path(request["monitor_interlock"]["no_spawn_path"])
    _exclusive_fsynced_json(path, {
        "schema": "paired-monitor-no-spawn-v1", "status": "NO_GUARD_SPAWNED",
        "label": request["label"], "stage": stage,
        "expected_commit": request.get("expected_commit"),
        "pair_binding_path": request["monitor_interlock"]["pair_binding_path"],
        "error_type": type(error).__name__, "error": str(error)[:240],
    })
