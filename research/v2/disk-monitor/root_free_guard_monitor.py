#!/usr/bin/env python3
"""Host-neutral, fail-closed root-space monitor for one pinned fresh-start guard.

This is not a launcher or an authorization mechanism. A separately reviewed,
one-attempt authorization is required before this program may be armed. It
first signals the pidfd of the exact guard that wrote the expected process
receipt. If TERM does not quiesce the job, it escalates through authenticated
pidfds for the bound child and descendants proven in its live kernel session,
including workers that have stripped the run marker. It never signals a
process group or an unverified numeric PID.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import socket
import stat
import subprocess
import sys
import time


SCHEMA = "root-free-guard-monitor-v1"
AUTH_SCHEMA = "one-shot-root-free-monitor-authorization-v1"
TWO_GIB = 2 * 1024**3
OWNERSHIP_MARKER = "TRAINIUM_FRESH_START_RUN_ID"
DEFAULT_KILL_SETTLE_SECONDS = 30


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
    def kill_pidfd(pidfd: int) -> None:
        signal.pidfd_send_signal(pidfd, signal.SIGKILL, None, 0)

    @staticmethod
    def pidfd_alive(pidfd: int) -> bool:
        poller = select.poll()
        poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        events = poller.poll(0)
        if not events:
            return True
        if any(flags & (select.POLLIN | select.POLLHUP) for _, flags in events):
            return False
        raise MonitorError("pidfd liveness could not be determined")

    @staticmethod
    def has_marker(pid: int, label: str) -> bool:
        needle = (OWNERSHIP_MARKER + "=" + label).encode()
        try:
            return needle in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            return False

    @staticmethod
    def owned_processes(label: str) -> list[ProcInfo]:
        """Find marker candidates; a public marker never proves ownership."""
        result = []
        for root in Path("/proc").iterdir():
            if not root.name.isdigit():
                continue
            pid = int(root.name)
            if pid == os.getpid():
                continue
            try:
                if root.stat().st_uid != os.getuid() or not RealSystem.has_marker(pid, label):
                    continue
                proc = RealSystem.proc(pid)
                if proc is not None and proc.uid == os.getuid():
                    result.append(proc)
            except (FileNotFoundError, ProcessLookupError):
                continue
        return result

    @staticmethod
    def session_processes(session: int) -> list[ProcInfo]:
        """Inventory same-UID session members, even without the run marker."""
        result = []
        for root in Path("/proc").iterdir():
            if not root.name.isdigit():
                continue
            pid = int(root.name)
            if pid == os.getpid():
                continue
            try:
                if root.stat().st_uid != os.getuid():
                    continue
                raw = (root / "stat").read_text()
                end_comm = raw.rfind(")")
                if end_comm < 2:
                    raise MonitorError("malformed /proc stat during session inventory")
                fields = raw[end_comm + 2 :].split()
                if len(fields) < 4:
                    raise MonitorError("short /proc stat during session inventory")
                if fields[0] == "Z" or int(fields[3]) != session:
                    continue
                proc = RealSystem.proc(pid)
                if proc is not None and proc.uid == os.getuid():
                    result.append(proc)
            except (FileNotFoundError, ProcessLookupError):
                continue
        return result

    @staticmethod
    def recent_processes(minimum_birth_ticks: int) -> list[ProcInfo]:
        """Find same-UID post-child processes as HOLD hints, not kill authority.

        A descendant may leave the child's session and exec with a sanitized
        environment. Such a process loses both strong session provenance and
        the public marker. Birth-at/after the bound child is deliberately
        overinclusive: unrelated new processes may cause HOLD, never a signal.
        """
        if type(minimum_birth_ticks) is not int or minimum_birth_ticks <= 0:
            raise MonitorError("invalid child birth threshold for process inventory")
        result = []
        for root in Path("/proc").iterdir():
            if not root.name.isdigit():
                continue
            pid = int(root.name)
            if pid == os.getpid():
                continue
            try:
                if root.stat().st_uid != os.getuid():
                    continue
                raw = (root / "stat").read_text()
                end_comm = raw.rfind(")")
                if end_comm < 2:
                    raise MonitorError("malformed /proc stat during birth inventory")
                fields = raw[end_comm + 2 :].split()
                if len(fields) < 20:
                    raise MonitorError("short /proc stat during birth inventory")
                if fields[0] == "Z" or int(fields[19]) < minimum_birth_ticks:
                    continue
                proc = RealSystem.proc(pid)
                if proc is not None and proc.uid == os.getuid():
                    result.append(proc)
            except (FileNotFoundError, ProcessLookupError):
                continue
        return result

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
        self.child_pidfd: int | None = None
        self.job_hash: str | None = None
        self.process_hash: str | None = None
        self.started_at = 0.0
        self.last_tick_at = 0.0
        self.job_first_seen_at: float | None = None
        self.stop_at: float | None = None
        self.stop_reason: str | None = None
        self.kill_at: float | None = None
        self.kill_sent: set[tuple[int, int]] = set()
        self.hold_reason: str | None = None
        self.term_failed = False
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
        if not 0 < int(b["post_stop_timeout_seconds"]) <= 120:
            raise MonitorError("invalid TERM grace timeout")
        if not 0 < int(b.get("kill_settle_timeout_seconds", DEFAULT_KILL_SETTLE_SECONDS)) <= 60:
            raise MonitorError("invalid KILL settle timeout")
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

    def _auth_child(self, proc: ProcInfo, guard_pid: int, *, first: bool,
                    allow_reparented: bool = False) -> None:
        if (proc.uid != self.binding["expected_uid"]
                or (not allow_reparented and proc.ppid != guard_pid)
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
        guard_pidfd = self.system.open_pidfd(guard_pid)
        child_pidfd = None
        try:
            again = self.system.proc(guard_pid)
            if again is None or again.birth_ticks != guard.birth_ticks:
                raise MonitorError("guard changed while opening pidfd")
            child_pidfd = self.system.open_pidfd(child_pid)
            child_again = self.system.proc(child_pid)
            if (child_again is None or child_again.birth_ticks != child.birth_ticks
                    or not self.system.pidfd_alive(guard_pidfd)
                    or not self.system.pidfd_alive(child_pidfd)):
                raise MonitorError("guard or child changed while opening pidfd")
            self._auth_guard(again, first=True)
            self._auth_child(child_again, guard_pid, first=True)
            self.guard = guard
            self.child = child
            self.guard_pidfd = guard_pidfd
            self.child_pidfd = child_pidfd
            self.state = "BOUND"
            self.job_hash = self.system.sha256(self.job_path)
            self.process_hash = self.system.sha256(self.process_path)
            self.journal.emit("GUARD_BOUND", guard_pid=guard.pid,
                              guard_uid=guard.uid, guard_birth_ticks=guard.birth_ticks,
                              child_pid=child.pid, child_birth_ticks=child.birth_ticks,
                              child_session=child.session,
                              job_sha256=self.job_hash, process_sha256=self.process_hash)
        except BaseException:
            if self.guard_pidfd is None:
                self.system.close_pidfd(guard_pidfd)
            if self.child_pidfd is None and child_pidfd is not None:
                self.system.close_pidfd(child_pidfd)
            raise

    def _emit_safety(self, event: str, **fields: object) -> None:
        # A full or unreadable root filesystem must not veto a pidfd signal.
        try:
            self.journal.emit(event, **fields)
        except OSError:
            print(f"disk monitor: could not persist {event}", file=sys.stderr, flush=True)

    def _bound_guard(self) -> ProcInfo | None:
        if self.guard is None or self.guard_pidfd is None:
            raise MonitorError("guard pidfd was never authenticated")
        if not self.system.pidfd_alive(self.guard_pidfd):
            return None
        current = self.system.proc(self.guard.pid)
        if current is None:
            raise MonitorError("live guard pidfd has no readable process identity")
        self._auth_guard(current, first=False)
        return current

    def _bound_child(self) -> ProcInfo | None:
        if self.child is None or self.child_pidfd is None:
            raise MonitorError("child pidfd was never authenticated")
        if not self.system.pidfd_alive(self.child_pidfd):
            return None
        current = self.system.proc(self.child.pid)
        if current is None:
            raise MonitorError("live child pidfd has no readable process identity")
        # The direct child may be reparented after the guard exits. Its pinned
        # pidfd, birth, UID, session, executable and argv still identify it.
        self._auth_child(current, self.guard.pid, first=False, allow_reparented=True)
        if current.exe != self.child.exe:
            raise MonitorError("owned child executable changed")
        return current

    def _owned_descendants(self) -> tuple[list[ProcInfo], list[ProcInfo]]:
        if self.guard is None or self.child is None:
            raise MonitorError("owned process roots are not bound")
        # A public environment marker is a discovery hint, not proof of
        # descent. Only a live, reauthenticated child can anchor its unique
        # kernel session. Include markerless session members and every
        # same-UID post-child process; anything not provably anchored remains
        # HOLD-only. This may HOLD on unrelated same-UID work, but can never
        # authorize a broad signal.
        child_anchor = self._bound_child()
        proven, unproven = [], []
        candidates = {}
        for proc in self.system.owned_processes(self.label):
            candidates[(proc.pid, proc.birth_ticks)] = proc
        for proc in self.system.session_processes(self.child.session):
            candidates[(proc.pid, proc.birth_ticks)] = proc
        for proc in self.system.recent_processes(self.child.birth_ticks):
            candidates[(proc.pid, proc.birth_ticks)] = proc
        for proc in candidates.values():
            if proc.pid == self.guard.pid:
                if proc.birth_ticks != self.guard.birth_ticks:
                    unproven.append(proc)
                continue
            if proc.pid == self.child.pid:
                if proc.birth_ticks != self.child.birth_ticks:
                    unproven.append(proc)
                continue
            if (child_anchor is None or proc.uid != self.binding["expected_uid"]
                    or proc.birth_ticks < self.child.birth_ticks
                    or proc.session != self.child.session):
                unproven.append(proc)
            else:
                proven.append(proc)
        return proven, unproven

    def _quiescent(self) -> bool:
        if self._bound_guard() is not None or self._bound_child() is not None:
            return False
        proven, unproven = self._owned_descendants()
        if unproven:
            self.hold_reason = "process remains without proven live child-session ownership"
            self._emit_safety("UNPROVEN_PROCESS_HOLD",
                              pids=[proc.pid for proc in unproven])
        return not proven and not unproven

    def _stop_owned_guard(self, reason: str) -> None:
        if self.state in ("STOPPING", "KILLING"):
            return
        if self.state != "BOUND":
            raise MonitorError("no authenticated owned guard to stop")
        guard = self._bound_guard()
        self.stop_reason = reason
        self.stop_at = self.system.monotonic()
        self.state = "STOPPING"
        if guard is None:
            self._emit_safety("TERM_GUARD_ALREADY_EXITED", reason=reason,
                              guard_pid=self.guard.pid)
            self._escalate_kill("GUARD_EXITED_BEFORE_TERM")
            return
        self._emit_safety("TERM_INTENT", reason=reason, guard_pid=guard.pid,
                          guard_birth_ticks=guard.birth_ticks, child_pid=self.child.pid)
        try:
            self.system.term_pidfd(self.guard_pidfd)
        except ProcessLookupError:
            self._emit_safety("TERM_GUARD_EXIT_RACE", guard_pid=guard.pid)
            self._escalate_kill("GUARD_EXITED_DURING_TERM")
            return
        except OSError as error:
            self.term_failed = True
            self.hold_reason = f"guard TERM failed: {error}"
            self._emit_safety("TERM_FAILED", error_type=type(error).__name__,
                              reason=str(error)[:240])
            self._escalate_kill("TERM_FAILED")
            return
        self._emit_safety("TERM_SENT", reason=reason, guard_pid=guard.pid)

    def _kill_live_targets(self) -> None:
        """KILL only bound pidfds and live-session-authenticated pidfds."""
        errors = []
        try:
            descendants, unproven = self._owned_descendants()
            if unproven:
                errors.append("process lacks proven live child-session ownership: "
                              + ",".join(str(proc.pid) for proc in unproven))
        except Exception as error:
            descendants = []
            errors.append(f"descendant inventory: {error}")
        for seen in descendants:
            key = (seen.pid, seen.birth_ticks)
            if key in self.kill_sent:
                continue
            pidfd = None
            try:
                pidfd = self.system.open_pidfd(seen.pid)
                current = self.system.proc(seen.pid)
                if current is None or not self.system.pidfd_alive(pidfd):
                    continue
                if (current.birth_ticks != seen.birth_ticks
                        or current.uid != self.binding["expected_uid"]
                        or current.argv != seen.argv or current.exe != seen.exe
                        or current.birth_ticks < self.child.birth_ticks
                        or current.session != self.child.session
                        or self._bound_child() is None):
                    raise MonitorError("descendant changed before pidfd KILL")
                self.system.kill_pidfd(pidfd)
                self.kill_sent.add(key)
                self._emit_safety("KILL_SENT", target="child_session_descendant",
                                  pid=seen.pid, birth_ticks=seen.birth_ticks)
            except ProcessLookupError:
                pass
            except Exception as error:
                errors.append(f"descendant {seen.pid}: {error}")
            finally:
                if pidfd is not None:
                    self.system.close_pidfd(pidfd)
        for target, bound, pidfd in (("child", self._bound_child, self.child_pidfd),
                                     ("guard", self._bound_guard, self.guard_pidfd)):
            try:
                current = bound()
                if current is None:
                    continue
                key = (current.pid, current.birth_ticks)
                if key in self.kill_sent:
                    continue
                self.system.kill_pidfd(pidfd)
                self.kill_sent.add(key)
                self._emit_safety("KILL_SENT", target=target, pid=current.pid,
                                  birth_ticks=current.birth_ticks)
            except ProcessLookupError:
                pass
            except Exception as error:
                errors.append(f"{target}: {error}")
        if errors:
            self.hold_reason = "; ".join(errors)[:240]
            self._emit_safety("KILL_REFUSED", reason=self.hold_reason)

    def _escalate_kill(self, reason: str) -> None:
        if self.state == "KILLING":
            return
        if self.state not in ("BOUND", "STOPPING"):
            raise MonitorError("KILL escalation lacks a bound guard")
        if self.stop_reason is None:
            self.stop_reason = reason
        self.kill_at = self.system.monotonic()
        self.state = "KILLING"
        self._emit_safety("KILL_ESCALATION", reason=reason,
                          guard_pid=self.guard.pid, child_pid=self.child.pid)
        self._kill_live_targets()

    def _hold(self, reason: str, *, quiescent: bool) -> int:
        self.state = "HOLD"
        self.result = 3
        self._emit_safety("QUIESCENT_HOLD" if quiescent else "UNRESOLVED_HOLD",
                          reason=reason[:240], guard_pid=self.guard.pid if self.guard else None,
                          child_pid=self.child.pid if self.child else None)
        print(f"disk monitor: HOLD: {reason}", file=sys.stderr, flush=True)
        return 3

    def _finish_quiescent(self, free: int | None, *, force_hold: bool = False) -> int:
        if force_hold or self.hold_reason is not None or self.kill_at is not None:
            return self._hold(self.hold_reason or "emergency stop reached quiescence after safety failure",
                              quiescent=True)
        try:
            self._source_is_exact()
            if not self.system.exists(self.postflight_path):
                raise MonitorError("guard exited without postflight receipt")
            postflight = self.system.read_json(self.postflight_path)
            if postflight.get("label") != self.label or postflight.get("host_clean") is not True:
                raise MonitorError("postflight identity or host-clean gate failed")
        except Exception as error:
            return self._hold(str(error), quiescent=True)
        self.state = "FINISHED"
        self.result = 0 if self.stop_reason is None and postflight.get("command_returncode") == 0 else 2
        self._emit_safety("MONITOR_FINISHED", free_bytes=free,
                          stopped_by_monitor=self.stop_reason is not None,
                          postflight_host_clean=True, result=self.result)
        return self.result

    def _containment_step(self, free: int | None, *, force_hold: bool = False) -> int | None:
        now = self.system.monotonic()
        try:
            if self._quiescent():
                return self._finish_quiescent(free, force_hold=force_hold)
        except Exception as error:
            self.hold_reason = f"quiescence identity uncertain: {error}"
            self._emit_safety("QUIESCENCE_UNCERTAIN", reason=str(error)[:240])
            if self.state == "STOPPING":
                self._escalate_kill("QUIESCENCE_UNCERTAIN")
        if self.state == "STOPPING":
            try:
                guard_gone = not self.system.pidfd_alive(self.guard_pidfd)
            except Exception as error:
                guard_gone = True
                self.hold_reason = f"guard pidfd liveness unreadable: {error}"
            if (self.term_failed or guard_gone or self.stop_at is None
                    or now - self.stop_at >= self.binding["post_stop_timeout_seconds"]):
                self._escalate_kill("TERM_GRACE_EXPIRED" if not guard_gone else "GUARD_EXITED_WITH_OWNED_PROCESSES")
        if self.state == "KILLING":
            self._kill_live_targets()  # Discover children created during escalation.
            try:
                if self._quiescent():
                    return self._finish_quiescent(free, force_hold=force_hold)
            except Exception as error:
                self.hold_reason = f"quiescence identity uncertain: {error}"
            if now - self.kill_at >= self.binding.get("kill_settle_timeout_seconds", DEFAULT_KILL_SETTLE_SECONDS):
                return self._hold(self.hold_reason or "owned guard or descendants remain after authenticated KILL",
                                  quiescent=False)
        return None

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
        if self.state == "BOUND":
            guard = self._bound_guard()
            if guard is None:
                if self._quiescent():
                    return self._finish_quiescent(free)
                self._escalate_kill("GUARD_EXITED_WITH_OWNED_PROCESSES")
            elif free <= b["stop_free_bytes"]:
                self._stop_owned_guard("LOW_ROOT_FREE")
            else:
                # After binding, drift is a reason to stop the owned job. It is
                # never an authorization prerequisite for that emergency stop.
                self._source_is_exact()
                self._manifests_are_exact()
        if self.state in ("STOPPING", "KILLING"):
            result = self._containment_step(free)
            if result is not None:
                return result
        self.journal.emit("ROOT_SAMPLE", free_bytes=free, sample_monotonic=now,
                          state=self.state,
                          guard_pid=self.guard.pid if self.guard else None)
        return None

    def fail_closed(self, error: BaseException) -> int:
        kind = type(error).__name__
        self.hold_reason = f"monitor failure: {kind}: {error}"[:240]
        self._emit_safety("MONITOR_FAILURE", error_type=kind, reason=str(error)[:240],
                          state=self.state)
        try:
            late_receipts = (self.state == "WAITING_FOR_GUARD"
                             and self.system.exists(self.job_path)
                             and self.system.exists(self.process_path))
        except Exception as receipt_error:
            late_receipts = False
            self._emit_safety("LATE_RECEIPT_CHECK_FAILED",
                              reason=str(receipt_error)[:240])
        if late_receipts:
            try:
                self._bind()
            except BaseException as bind_error:
                self._emit_safety("LATE_BIND_REFUSED", error_type=type(bind_error).__name__,
                                  reason=str(bind_error)[:240])
        if self.state == "BOUND":
            try:
                self._stop_owned_guard("MONITOR_FAILURE")
            except BaseException as stop_error:
                self._emit_safety("SIGNAL_REFUSED", error_type=type(stop_error).__name__,
                                  reason=str(stop_error)[:240])
                try:
                    self._escalate_kill("TERM_AUTHENTICATION_FAILED")
                except BaseException as escalation_error:
                    return self._hold(f"authenticated KILL escalation failed: {escalation_error}",
                                      quiescent=False)
        if self.state not in ("STOPPING", "KILLING"):
            return self._hold(self.hold_reason, quiescent=False)
        poll = float(self.binding["poll_seconds"])
        while True:
            try:
                try:
                    free = self.system.free_bytes()
                except OSError:
                    free = None
                result = self._containment_step(free, force_hold=True)
                if result is not None:
                    return 3
                self._emit_safety("ROOT_SAMPLE", free_bytes=free,
                                  sample_monotonic=self.system.monotonic(),
                                  state=self.state, guard_pid=self.guard.pid)
                self.system.sleep(poll)
            except BaseException as containment_error:
                return self._hold(f"containment monitoring failed: {containment_error}",
                                  quiescent=False)

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
            if self.child_pidfd is not None:
                self.system.close_pidfd(self.child_pidfd)


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
