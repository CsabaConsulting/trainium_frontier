#!/usr/bin/env python3
"""Host-neutral, fail-closed root-space monitor for one pinned fresh-start guard.

This is not a launcher or an authorization mechanism. A separately reviewed,
one-attempt authorization is required before this program may be armed. It
signals only the pidfd of the exact guard that wrote the expected process
receipt; it never signals a process group or an unverified PID.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time


SCHEMA = "root-free-guard-monitor-v1"
AUTH_SCHEMA = "one-shot-root-free-monitor-authorization-v1"
TWO_GIB = 2 * 1024**3


class MonitorError(RuntimeError):
    """A mandatory monitor condition failed."""


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    uid: int
    birth_ticks: int
    ppid: int
    pgrp: int
    session: int
    argv: tuple[str, ...]
    exe: str


class RealSystem:
    @staticmethod
    def free_bytes() -> int:
        vfs = os.statvfs("/")
        return vfs.f_bavail * vfs.f_frsize

    @staticmethod
    def monotonic() -> float:
        return time.monotonic()

    @staticmethod
    def sleep(seconds: float) -> None:
        time.sleep(seconds)

    @staticmethod
    def exists(path: Path) -> bool:
        return path.exists()

    @staticmethod
    def read_json(path: Path) -> dict:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise MonitorError("expected a JSON object")
        return value

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def git_head(remote: Path) -> str:
        return subprocess.check_output(
            ["git", "-C", str(remote), "rev-parse", "HEAD"],
            text=True,
            timeout=5,
        ).strip()

    @staticmethod
    def proc(pid: int) -> ProcInfo | None:
        procdir = Path("/proc") / str(pid)
        try:
            raw = (procdir / "stat").read_text()
            # The comm field is parenthesized and may contain spaces or ')'.
            end_comm = raw.rfind(")")
            if end_comm < 2:
                raise MonitorError("malformed /proc stat")
            after_comm = raw[end_comm + 2 :].split()
            if len(after_comm) < 20:
                raise MonitorError("short /proc stat")
            uid_line = next(
                line for line in (procdir / "status").read_text().splitlines()
                if line.startswith("Uid:")
            )
            uid = int(uid_line.split()[1])
            argv = tuple(
                item.decode("utf-8", "surrogateescape")
                for item in (procdir / "cmdline").read_bytes().split(b"\0")
                if item
            )
            exe = os.path.realpath(os.readlink(procdir / "exe"))
            return ProcInfo(
                pid=pid,
                uid=uid,
                birth_ticks=int(after_comm[19]),
                ppid=int(after_comm[1]),
                pgrp=int(after_comm[2]),
                session=int(after_comm[3]),
                argv=argv,
                exe=exe,
            )
        except FileNotFoundError:
            return None
        except ProcessLookupError:
            return None

    @staticmethod
    def open_pidfd(pid: int) -> int:
        # No os.kill fallback: a numeric PID can be reused between check and signal.
        return os.pidfd_open(pid, 0)

    @staticmethod
    def term_pidfd(pidfd: int) -> None:
        signal.pidfd_send_signal(pidfd, signal.SIGTERM, None, 0)

    @staticmethod
    def close_pidfd(pidfd: int) -> None:
        os.close(pidfd)


class Journal:
    """Append-only, fsynced evidence. A previous attempt is never overwritten."""

    def __init__(self, events_path: Path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        self.fd = os.open(events_path, flags, 0o600)
        parent_fd = os.open(events_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def emit(self, event: str, **fields: object) -> None:
        item = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        data = (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode()
        offset = 0
        while offset < len(data):
            offset += os.write(self.fd, data[offset:])
        os.fsync(self.fd)

    def close(self) -> None:
        os.close(self.fd)


def write_ready(path: Path, payload: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        data = (json.dumps(payload, sort_keys=True) + "\n").encode()
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


class GuardMonitor:
    def __init__(self, binding: dict, request: dict, system: RealSystem, journal: Journal):
        self.binding = binding
        self.request = request
        self.system = system
        self.journal = journal
        self.label = binding["label"]
        self.remote = Path(binding["remote"])
        logdir = self.remote / "research/run-logs"
        self.job_path = logdir / f"{self.label}.job.json"
        self.process_path = logdir / f"{self.label}.process.json"
        self.postflight_path = logdir / f"{self.label}.postflight.json"
        self.launch_path = logdir / f"{self.label}.launch.json"
        self.guard_path = self.remote / "research/fresh_start_guard.py"
        self.guard: ProcInfo | None = None
        self.child: ProcInfo | None = None
        self.guard_pidfd: int | None = None
        self.job_hash: str | None = None
        self.process_hash: str | None = None
        self.started_at = 0.0
        self.last_tick_at = 0.0
        self.job_first_seen_at: float | None = None
        self.stop_at: float | None = None
        self.stop_reason: str | None = None
        self.state = "NEW"
        self.result: int | None = None

    def _source_is_exact(self) -> None:
        b, r, s = self.binding, self.request, self.system
        if s.sha256(Path(b["request_path"])) != b["request_sha256"]:
            raise MonitorError("request bytes changed")
        if s.sha256(Path(__file__)) != b["monitor_script_sha256"]:
            raise MonitorError("monitor script bytes changed")
        if s.git_head(self.remote) != b["expected_commit"]:
            raise MonitorError("source commit changed")
        if r.get("expected_sha256") != b["expected_sha256"]:
            raise MonitorError("source hash contract changed")
        for relpath, expected in b["expected_sha256"].items():
            if s.sha256(self.remote / relpath) != expected:
                raise MonitorError(f"source bytes changed: {relpath}")

    def arm(self) -> int:
        b, r, s = self.binding, self.request, self.system
        if b.get("schema") != SCHEMA or b.get("package_state") != "STATIC_ONLY":
            raise MonitorError("invalid monitor binding")
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise MonitorError("pidfd support is required before readiness")
        if not 0 < float(b["poll_seconds"]) <= 1:
            raise MonitorError("poll interval exceeds one second")
        if not 0 < int(b["bind_timeout_seconds"]) <= 180:
            raise MonitorError("invalid bind timeout")
        if b["stop_free_bytes"] < b["floor_free_bytes"] + TWO_GIB:
            raise MonitorError("stop threshold lacks two-GiB reserve")
        if b["prelaunch_free_bytes"] <= b["stop_free_bytes"]:
            raise MonitorError("prelaunch space is not above stop threshold")
        if r.get("label") != self.label or r.get("remote") != str(self.remote):
            raise MonitorError("request label or remote changed")
        if r.get("expected_commit") != b["expected_commit"]:
            raise MonitorError("request commit changed")
        if r.get("expected_hostname") != b["expected_hostname"]:
            raise MonitorError("request hostname changed")
        if r.get("min_free_bytes") != b["floor_free_bytes"]:
            raise MonitorError("operational floor changed")
        if r.get("pre_launch_required_free_bytes") != b["prelaunch_free_bytes"]:
            raise MonitorError("prelaunch floor changed")
        if socket.gethostname() != b["expected_hostname"]:
            raise MonitorError("wrong host")
        if os.getuid() != b["expected_uid"]:
            raise MonitorError("wrong user")
        self._source_is_exact()
        if any(s.exists(p) for p in (self.job_path, self.process_path,
                                      self.postflight_path, self.launch_path)):
            raise MonitorError("science label already has launch artifacts")
        free = s.free_bytes()
        if free < b["prelaunch_free_bytes"]:
            raise MonitorError("prelaunch root free below reviewed gate")
        self.started_at = s.monotonic()
        self.last_tick_at = self.started_at
        self.state = "WAITING_FOR_GUARD"
        self.journal.emit("MONITOR_ARMED", label=self.label, free_bytes=free,
                          stop_free_bytes=b["stop_free_bytes"], poll_seconds=b["poll_seconds"])
        return free

    def _auth_guard(self, proc: ProcInfo, *, first: bool) -> None:
        b = self.binding
        if proc.uid != b["expected_uid"] or proc.pgrp != proc.pid or proc.session != proc.pid:
            raise MonitorError("guard UID/session/process-group mismatch")
        if proc.exe != os.path.realpath(b["expected_guard_python"]):
            raise MonitorError("guard executable mismatch")
        if len(proc.argv) != 5 or proc.argv[1:4] != (
            str(self.guard_path), str(self.job_path), "--lock-fd"
        ) or not proc.argv[4].isdigit():
            raise MonitorError("guard argv mismatch")
        if os.path.realpath(proc.argv[0]) != proc.exe:
            raise MonitorError("guard argv executable mismatch")
        if not first and (self.guard is None or proc.pid != self.guard.pid
                          or proc.birth_ticks != self.guard.birth_ticks):
            raise MonitorError("guard PID reused or birth ticks changed")

    def _auth_child(self, proc: ProcInfo, guard_pid: int, *, first: bool) -> None:
        if (proc.uid != self.binding["expected_uid"] or proc.ppid != guard_pid
                or proc.pgrp != proc.pid or proc.session != proc.pid):
            raise MonitorError("owned child UID/parent/process-group mismatch")
        if proc.argv != tuple(self.request["command"]):
            raise MonitorError("owned child command mismatch")
        if not first and (self.child is None or proc.pid != self.child.pid
                          or proc.birth_ticks != self.child.birth_ticks):
            raise MonitorError("owned child PID reused or birth ticks changed")

    def _manifests_are_exact(self) -> tuple[dict, dict]:
        s = self.system
        if (self.job_path.is_symlink() or self.process_path.is_symlink()):
            raise MonitorError("job or process receipt is a symlink")
        job = s.read_json(self.job_path)
        process = s.read_json(self.process_path)
        for key, value in self.request.items():
            if job.get(key) != value:
                raise MonitorError(f"job request field changed: {key}")
        if job.get("source_sha256", {}).get("train.py") != self.binding["expected_sha256"]["train.py"]:
            raise MonitorError("launch train source hash changed")
        if process.get("label") != self.label or process.get("command") != self.request["command"]:
            raise MonitorError("process receipt label/command mismatch")
        if process.get("child_session") != process.get("child_pid"):
            raise MonitorError("process receipt child session mismatch")
        if self.job_hash is not None and s.sha256(self.job_path) != self.job_hash:
            raise MonitorError("job receipt changed after binding")
        if self.process_hash is not None and s.sha256(self.process_path) != self.process_hash:
            raise MonitorError("process receipt changed after binding")
        return job, process

    def _bind(self) -> None:
        self._source_is_exact()
        _, process = self._manifests_are_exact()
        guard_pid, child_pid = process.get("guard_pid"), process.get("child_pid")
        if not isinstance(guard_pid, int) or not isinstance(child_pid, int):
            raise MonitorError("process receipt lacks PIDs")
        guard = self.system.proc(guard_pid)
        child = self.system.proc(child_pid)
        if guard is None or child is None:
            raise MonitorError("process receipt has no live guard/child pair")
        self._auth_guard(guard, first=True)
        self._auth_child(child, guard_pid, first=True)
        pidfd = self.system.open_pidfd(guard_pid)
        try:
            again = self.system.proc(guard_pid)
            if again is None or again.birth_ticks != guard.birth_ticks:
                raise MonitorError("guard changed while opening pidfd")
            self.guard = guard
            self.child = child
            self.guard_pidfd = pidfd
            self.job_hash = self.system.sha256(self.job_path)
            self.process_hash = self.system.sha256(self.process_path)
            self.state = "BOUND"
            self.journal.emit("GUARD_BOUND", guard_pid=guard.pid,
                              guard_uid=guard.uid, guard_birth_ticks=guard.birth_ticks,
                              child_pid=child.pid, child_birth_ticks=child.birth_ticks,
                              job_sha256=self.job_hash, process_sha256=self.process_hash)
        except BaseException:
            if self.guard_pidfd is None:
                self.system.close_pidfd(pidfd)
            raise

    def _stop_owned_guard(self, reason: str, *, best_effort_journal: bool = False) -> None:
        if self.state not in ("BOUND", "STOPPING") or self.guard is None or self.guard_pidfd is None:
            raise MonitorError("no authenticated owned guard to stop")
        self._source_is_exact()
        _, process = self._manifests_are_exact()
        if process["guard_pid"] != self.guard.pid or process["child_pid"] != self.child.pid:
            raise MonitorError("process receipt PID changed")
        guard = self.system.proc(self.guard.pid)
        child = self.system.proc(self.child.pid)
        if guard is None or child is None:
            raise MonitorError("guard or owned child exited before authenticated stop")
        self._auth_guard(guard, first=False)
        self._auth_child(child, guard.pid, first=False)
        try:
            self.journal.emit("TERM_INTENT", reason=reason, guard_pid=guard.pid,
                              guard_birth_ticks=guard.birth_ticks, child_pid=child.pid)
        except OSError:
            if not best_effort_journal:
                raise
            print("disk monitor: journal failed before emergency TERM", file=sys.stderr, flush=True)
        self.system.term_pidfd(self.guard_pidfd)
        self.stop_reason = reason
        self.stop_at = self.system.monotonic()
        self.state = "STOPPING"
        try:
            self.journal.emit("TERM_SENT", reason=reason, guard_pid=guard.pid)
        except OSError:
            if not best_effort_journal:
                raise
            print("disk monitor: journal failed after emergency TERM", file=sys.stderr, flush=True)

    def tick(self) -> int | None:
        s, b = self.system, self.binding
        now = s.monotonic()
        if now - self.last_tick_at > 1.0:
            raise MonitorError("root-free poll gap exceeded one second")
        self.last_tick_at = now
        free = s.free_bytes()
        if self.state == "WAITING_FOR_GUARD":
            if s.exists(self.job_path) and self.job_first_seen_at is None:
                self.job_first_seen_at = now
            if s.exists(self.process_path):
                if not s.exists(self.job_path):
                    raise MonitorError("process receipt without job")
                self._bind()
            elif self.job_first_seen_at is not None and now - self.job_first_seen_at > 10:
                raise MonitorError("guard process receipt missing after job creation")
            elif now - self.started_at > b["bind_timeout_seconds"]:
                raise MonitorError("no guard appeared before bind timeout")
            if free <= b["stop_free_bytes"]:
                if self.state != "BOUND":
                    raise MonitorError("root free crossed stop threshold before guard authentication")
                self._stop_owned_guard("LOW_ROOT_FREE")
        if self.state in ("BOUND", "STOPPING"):
            guard = s.proc(self.guard.pid)
            if guard is None:
                if not s.exists(self.postflight_path):
                    raise MonitorError("guard exited without postflight receipt")
                postflight = s.read_json(self.postflight_path)
                if postflight.get("label") != self.label or postflight.get("host_clean") is not True:
                    raise MonitorError("postflight identity or host-clean gate failed")
                self.state = "FINISHED"
                self.result = 0 if self.stop_reason is None and postflight.get("command_returncode") == 0 else 2
                self.journal.emit("MONITOR_FINISHED", free_bytes=free,
                                  stopped_by_monitor=self.stop_reason is not None,
                                  postflight_host_clean=True, result=self.result)
                return self.result
            self._auth_guard(guard, first=False)
            if self.state == "BOUND" and free <= b["stop_free_bytes"]:
                self._stop_owned_guard("LOW_ROOT_FREE")
            if (self.state == "STOPPING" and self.stop_at is not None
                    and now - self.stop_at > b["post_stop_timeout_seconds"]):
                raise MonitorError("guard did not finish after TERM")
        self.journal.emit("ROOT_SAMPLE", free_bytes=free, sample_monotonic=now,
                          state=self.state,
                          guard_pid=self.guard.pid if self.guard else None)
        return None

    def fail_closed(self, error: BaseException) -> int:
        kind = type(error).__name__
        try:
            self.journal.emit("MONITOR_FAILURE", error_type=kind, reason=str(error)[:240],
                              state=self.state)
        except OSError:
            print(f"disk monitor: journal failure ({kind})", file=sys.stderr, flush=True)
        if (self.state == "WAITING_FOR_GUARD"
                and self.system.exists(self.job_path)
                and self.system.exists(self.process_path)):
            try:
                self._bind()
            except BaseException as bind_error:
                try:
                    self.journal.emit("LATE_BIND_REFUSED", error_type=type(bind_error).__name__,
                                      reason=str(bind_error)[:240])
                except OSError:
                    pass
        if self.state == "BOUND":
            try:
                self._stop_owned_guard("MONITOR_FAILURE", best_effort_journal=True)
            except BaseException as stop_error:
                try:
                    self.journal.emit("SIGNAL_REFUSED", error_type=type(stop_error).__name__,
                                      reason=str(stop_error)[:240])
                except OSError:
                    print("disk monitor: authenticated stop refused", file=sys.stderr, flush=True)
        return 3

    def run(self) -> int:
        poll = float(self.binding["poll_seconds"])
        try:
            while True:
                began = self.system.monotonic()
                result = self.tick()
                if result is not None:
                    return result
                self.system.sleep(max(0, poll - (self.system.monotonic() - began)))
        except BaseException as error:
            return self.fail_closed(error)
        finally:
            if self.guard_pidfd is not None:
                self.system.close_pidfd(self.guard_pidfd)


def _load_authorized_binding(binding_path: Path, auth_path: Path) -> tuple[dict, dict, str]:
    system = RealSystem()
    binding_sha = system.sha256(binding_path)
    binding = system.read_json(binding_path)
    auth = system.read_json(auth_path)
    if auth.get("schema") != AUTH_SCHEMA or auth.get("authorized") is not True:
        raise MonitorError("no one-shot monitor authorization")
    if (auth.get("binding_sha256") != binding_sha
            or auth.get("label") != binding.get("label")
            or auth.get("request_sha256") != binding.get("request_sha256")
            or auth.get("one_attempt_only") is not True):
        raise MonitorError("authorization does not bind exact one-label package")
    request_path = Path(binding["request_path"])
    if system.sha256(request_path) != binding["request_sha256"]:
        raise MonitorError("reviewed request bytes unavailable")
    request = system.read_json(request_path)
    return binding, request, binding_sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    binding, request, binding_sha = _load_authorized_binding(args.binding, args.authorization)
    events = Path(binding["events_path"])
    ready = Path(binding["ready_path"])
    if not events.is_absolute() or not ready.is_absolute() or events.parent != ready.parent:
        raise MonitorError("evidence paths must share one absolute external directory")
    parent = events.parent
    parent_stat = parent.stat(follow_symlinks=False)
    if (not stat.S_ISDIR(parent_stat.st_mode) or parent_stat.st_uid != os.getuid()
            or parent_stat.st_mode & 0o022):
        raise MonitorError("evidence directory must be owned and not group/world writable")
    journal = Journal(events)
    monitor = GuardMonitor(binding, request, RealSystem(), journal)
    try:
        try:
            def interrupted(_signal_number: int, _frame: object) -> None:
                raise MonitorError("monitor received a termination signal")

            signal.signal(signal.SIGTERM, interrupted)
            signal.signal(signal.SIGINT, interrupted)
            free = monitor.arm()
            self_proc = RealSystem.proc(os.getpid())
            if self_proc is None:
                raise MonitorError("monitor process identity unavailable")
            write_ready(ready, {
                "schema": SCHEMA, "label": binding["label"],
                "binding_sha256": binding_sha, "request_sha256": binding["request_sha256"],
                "monitor_script_sha256": binding["monitor_script_sha256"],
                "pid": os.getpid(), "uid": os.getuid(),
                "birth_ticks": self_proc.birth_ticks, "initial_root_free_bytes": free,
                "poll_seconds": binding["poll_seconds"], "state": "READY_BEFORE_LAUNCH",
            })
            return monitor.run()
        except BaseException as error:
            return monitor.fail_closed(error)
    finally:
        journal.close()


if __name__ == "__main__":
    sys.exit(main())
