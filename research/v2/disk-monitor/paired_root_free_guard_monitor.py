#!/usr/bin/env python3
"""Two independent, mutually watched root-space monitors for one pinned guard.

This wrapper does not launch training. Both roles must become ready and exchange
fsynced samples before an operator may launch the exact reviewed request. Each
role can authenticate and stop the same guard and owned child through pidfds
if its peer dies or stops sampling. Simultaneous process/host/kernel failure
remains a boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys

import root_free_guard_monitor as base


PAIR_SCHEMA = "paired-root-free-guard-monitor-v1"
LIVE_STATES = frozenset(("WAITING_FOR_GUARD", "BOUND", "STOPPING", "KILLING"))
MAX_SAMPLE_AGE_SECONDS = 1.0


def last_root_sample(path: Path, expected_uid: int) -> tuple[dict | None, tuple[int, int]]:
    """Read only completed, newline-terminated events; reject swapped journals."""
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != expected_uid
                or info.st_mode & 0o077):
            raise base.MonitorError("peer journal owner, type, or mode mismatch")
        start = max(0, info.st_size - 65536)
        os.lseek(fd, start, os.SEEK_SET)
        tail = os.read(fd, 65536)
    finally:
        os.close(fd)
    lines = tail.split(b"\n")
    if not tail.endswith(b"\n"):
        lines.pop()  # A concurrent append is not yet a completed fsynced event.
    if start:
        lines.pop(0)  # The first line may start in the middle of an event.
    for raw in reversed(lines):
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise base.MonitorError("malformed peer journal") from error
        if event.get("event") == "ROOT_SAMPLE":
            return event, (info.st_dev, info.st_ino)
    return None, (info.st_dev, info.st_ino)


def exact_peer_argv(pair: dict, role: str, binding: Path, auth: Path,
                    pair_path: Path) -> tuple[str, ...]:
    return (
        pair["expected_python"], pair["wrapper_path"],
        "--binding", str(binding), "--authorization", str(auth),
        "--pair-binding", str(pair_path), "--role", role,
    )


class PairedGuardMonitor(base.GuardMonitor):
    def __init__(self, binding: dict, request: dict, system: base.RealSystem,
                 journal: base.Journal, *, pair: dict, pair_path: Path,
                 binding_path: Path, auth_path: Path, auth: dict, role: str):
        super().__init__(binding, request, system, journal)
        self.pair = pair
        self.pair_path = pair_path
        self.binding_path = binding_path
        self.auth_path = auth_path
        self.auth = auth
        self.role = role
        self.peer_role = "peer" if role == "primary" else "primary"
        self.peer_identity: tuple[int, int] | None = None
        self.peer_journal_inode: tuple[int, int] | None = None
        self.peer_bound_logged = False

    def _pair_is_exact(self) -> None:
        s, p = self.system, self.pair
        if s.sha256(self.pair_path) != self.auth["pair_binding_sha256"]:
            raise base.MonitorError("pair binding bytes changed")
        if s.sha256(self.binding_path) != p["base_binding_sha256"]:
            raise base.MonitorError("base binding bytes changed")
        if s.sha256(Path(__file__)) != p["wrapper_script_sha256"]:
            raise base.MonitorError("paired wrapper bytes changed")
        if s.sha256(Path(base.__file__)) != self.binding["monitor_script_sha256"]:
            raise base.MonitorError("base monitor bytes changed")

    def arm(self) -> int:
        p, b = self.pair, self.binding
        if (p.get("schema") != PAIR_SCHEMA or p.get("package_state") != "STATIC_ONLY"
                or p.get("label") != b["label"] or p.get("remote") != b["remote"]
                or p.get("request_sha256") != b["request_sha256"]
                or p.get("expected_hostname") != b["expected_hostname"]
                or p.get("expected_uid") != b["expected_uid"]
                or p.get("base_binding_sha256") != self.auth["binding_sha256"]
                or set(p.get("roles", {})) != {"primary", "peer"}
                or p.get("expected_python") != b["expected_guard_python"]):
            raise base.MonitorError("paired binding disagrees with reviewed request")
        if self.role not in p["roles"]:
            raise base.MonitorError("unrecognized monitor role")
        self._pair_is_exact()
        free = super().arm()
        if self.role == "primary":
            # The second monitor cannot publish readiness before the first is
            # alive and has already written a fresh root-space sample.
            self._check_peer(require_present=True)
        return free

    def _check_peer(self, *, require_present: bool = False) -> bool:
        now = self.system.monotonic()
        role_info = self.pair["roles"][self.peer_role]
        ready_path = Path(role_info["ready_path"])
        if not self.system.exists(ready_path):
            if (require_present or self.peer_identity is not None
                    or now - self.started_at > self.pair["peer_start_timeout_seconds"]):
                raise base.MonitorError("peer ready missing or peer monitor died")
            return False
        if ready_path.is_symlink():
            raise base.MonitorError("peer ready file is a symlink")
        ready = self.system.read_json(ready_path)
        expected = {
            "schema": PAIR_SCHEMA, "state": "READY_BEFORE_LAUNCH",
            "label": self.label, "role": self.peer_role,
            "pair_binding_sha256": self.auth["pair_binding_sha256"],
            "base_binding_sha256": self.pair["base_binding_sha256"],
            "request_sha256": self.binding["request_sha256"],
            "wrapper_script_sha256": self.pair["wrapper_script_sha256"],
        }
        if any(ready.get(key) != value for key, value in expected.items()):
            raise base.MonitorError("peer ready identity changed")
        pid, birth = ready.get("pid"), ready.get("birth_ticks")
        if (not isinstance(pid, int) or pid <= 1 or not isinstance(birth, int)
                or ready.get("uid") != self.binding["expected_uid"]):
            raise base.MonitorError("peer PID/UID/birth receipt invalid")
        if self.peer_identity is not None and (pid, birth) != self.peer_identity:
            raise base.MonitorError("peer PID reused or identity changed")
        proc = self.system.proc(pid)
        if (proc is None or proc.uid != self.binding["expected_uid"]
                or proc.birth_ticks != birth
                or proc.exe != os.path.realpath(self.pair["expected_python"])
                or proc.argv != exact_peer_argv(self.pair, self.peer_role,
                                                self.binding_path, self.auth_path,
                                                self.pair_path)):
            raise base.MonitorError("peer process/argv identity mismatch")
        events_path = Path(role_info["events_path"])
        if not self.system.exists(events_path):
            raise base.MonitorError("peer journal missing")
        sample, inode = last_root_sample(events_path, self.binding["expected_uid"])
        if self.peer_journal_inode is not None and inode != self.peer_journal_inode:
            raise base.MonitorError("peer journal inode changed")
        if sample is None:
            ready_at = ready.get("ready_monotonic")
            if (require_present or not isinstance(ready_at, (int, float))
                    or now - ready_at > MAX_SAMPLE_AGE_SECONDS):
                raise base.MonitorError("peer has not emitted a root-space sample")
            return False
        when = sample.get("sample_monotonic")
        if (not isinstance(when, (int, float)) or not math.isfinite(when)
                or when > now + 0.1 or now - when > MAX_SAMPLE_AGE_SECONDS
                or sample.get("state") not in LIVE_STATES):
            raise base.MonitorError("peer monitor heartbeat stale or invalid")
        if (self.guard is not None and sample.get("state") == "BOUND"
                and sample.get("guard_pid") != self.guard.pid):
            raise base.MonitorError("peer bound to a different guard")
        again = self.system.proc(pid)
        if again is None or again.birth_ticks != birth or again.uid != proc.uid:
            raise base.MonitorError("peer changed during liveness check")
        self.peer_identity = (pid, birth)
        self.peer_journal_inode = inode
        if not self.peer_bound_logged:
            self.journal.emit("PEER_BOUND", peer_role=self.peer_role, peer_pid=pid,
                              peer_birth_ticks=birth, peer_events_path=str(events_path))
            self.peer_bound_logged = True
        return True

    def tick(self) -> int | None:
        self._pair_is_exact()
        result = super().tick()
        if result is None:
            self._check_peer()
        return result

    def _no_spawn_receipt(self) -> bool:
        path = Path(self.pair["no_spawn_path"])
        if not self.system.exists(path):
            return False
        if path.is_symlink():
            raise base.MonitorError("no-spawn receipt is a symlink")
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode)
                    or info.st_uid != self.binding["expected_uid"]
                    or info.st_mode & 0o077 or info.st_size > 16384):
                raise base.MonitorError("no-spawn receipt owner/type/mode invalid")
            raw = os.read(fd, 16385)
        finally:
            os.close(fd)
        if len(raw) != info.st_size or not raw.endswith(b"\n"):
            return False  # Writer has not finished its exclusive fsynced receipt.
        try:
            receipt = json.loads(raw)
        except json.JSONDecodeError:
            return False  # A concurrent write is not accepted as an abort proof.
        if (receipt.get("schema") != "paired-monitor-no-spawn-v1"
                or receipt.get("status") != "NO_GUARD_SPAWNED"
                or receipt.get("label") != self.label
                or receipt.get("expected_commit") != self.binding["expected_commit"]
                or receipt.get("pair_binding_path") != str(self.pair_path)):
            raise base.MonitorError("no-spawn receipt identity mismatch")
        return True

    def _rescue_pre_guard(self, error: BaseException) -> int:
        """Stay resident after peer failure until exact guard or no-spawn proof.

        The launcher independently refuses a dead/stale pair before guard spawn.
        If it had passed that final check just before peer death, this survivor
        authenticates the imminent process receipt and stops only that guard.
        """
        try:
            self.journal.emit("MONITOR_FAILURE_RESCUE", error_type=type(error).__name__,
                              reason=str(error)[:240], state=self.state)
        except OSError:
            print("paired monitor: rescue journal unavailable", file=sys.stderr, flush=True)
        poll = float(self.binding["poll_seconds"])
        rescue_started = self.system.monotonic()
        hold_announced = False
        while True:
            try:
                now = self.system.monotonic()
                if (not hold_announced
                        and now - rescue_started >= self.binding["bind_timeout_seconds"]):
                    # A failed no-spawn write is not proof that no guard can
                    # appear. Publish a bounded HOLD verdict but stay resident
                    # to catch an authentic late receipt; never exit clean.
                    hold_announced = True
                    self.state = "RESCUE_UNRESOLVED_HOLD"
                    self.hold_reason = "no authenticated guard or no-spawn proof within rescue deadline"
                    self._emit_safety("RESCUE_UNRESOLVED_HOLD",
                                      reason=self.hold_reason, elapsed_seconds=now - rescue_started)
                    print("paired monitor: unresolved HOLD; continuing to watch for exact guard or no-spawn proof",
                          file=sys.stderr, flush=True)
                try:
                    free = self.system.free_bytes()
                except OSError as sample_error:
                    free = None
                    self._emit_safety("RESCUE_ROOT_SAMPLE_UNAVAILABLE",
                                      reason=str(sample_error)[:240])
                if self.system.exists(self.process_path):
                    self._bind()  # Verifies request/source, both PIDs, UID, birth, argv.
                    self._stop_owned_guard("PEER_MONITOR_FAILURE")
                    while True:
                        result = base.GuardMonitor.tick(self)
                        if result is not None:
                            return 3
                        self.system.sleep(poll)
                if self._no_spawn_receipt():
                    self.journal.emit("INTERLOCK_NO_SPAWN_CONFIRMED", label=self.label)
                    return 3
                self._emit_safety("ROOT_SAMPLE", free_bytes=free,
                                  sample_monotonic=now,
                                  state=("RESCUE_UNRESOLVED_HOLD" if hold_announced
                                         else "RESCUE_WAIT_FOR_GUARD"),
                                  guard_pid=None)
                self.system.sleep(poll)
            except (KeyboardInterrupt, SystemExit, base.MonitorError, OSError,
                    ValueError, KeyError, json.JSONDecodeError) as rescue_error:
                # Do not guess at a PID or silently call a failed rescue safe.
                try:
                    self.journal.emit("RESCUE_REFUSED", error_type=type(rescue_error).__name__,
                                      reason=str(rescue_error)[:240])
                except OSError:
                    pass
                if self.state in ("BOUND", "STOPPING", "KILLING"):
                    return base.GuardMonitor.fail_closed(self, rescue_error)
                return 3

    def fail_closed(self, error: BaseException) -> int:
        if self.state == "WAITING_FOR_GUARD" and self.peer_bound_logged:
            return self._rescue_pre_guard(error)
        return super().fail_closed(error)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--pair-binding", type=Path, required=True)
    parser.add_argument("--role", choices=("primary", "peer"), required=True)
    args = parser.parse_args()
    for path in (args.binding, args.authorization, args.pair_binding):
        if not path.is_absolute():
            raise base.MonitorError("all pair package paths must be absolute")
    binding, request, binding_sha = base._load_authorized_binding(
        args.binding, args.authorization)
    system = base.RealSystem()
    auth = system.read_json(args.authorization)
    pair_sha = system.sha256(args.pair_binding)
    pair = system.read_json(args.pair_binding)
    if (auth.get("paired_authorized") is not True
            or auth.get("dispatch_authorized") is not True
            or auth.get("pair_binding_sha256") != pair_sha
            or pair.get("base_binding_path") != str(args.binding)
            or pair.get("authorization_path") != str(args.authorization)
            or pair.get("base_binding_sha256") != binding_sha
            or pair.get("wrapper_path") != str(Path(__file__))):
        raise base.MonitorError("paired one-shot authorization is not exact")
    role_info = pair["roles"][args.role]
    events, ready = Path(role_info["events_path"]), Path(role_info["ready_path"])
    if (not events.is_absolute() or not ready.is_absolute()
            or events.parent != ready.parent or events == ready
            or events.parent != args.pair_binding.parent):
        raise base.MonitorError("paired evidence paths must share one external directory")
    parent = events.parent.stat(follow_symlinks=False)
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid()
            or parent.st_mode & 0o022):
        raise base.MonitorError("paired evidence directory is not privately owned")
    journal = base.Journal(events)
    monitor = PairedGuardMonitor(
        binding, request, system, journal, pair=pair, pair_path=args.pair_binding,
        binding_path=args.binding, auth_path=args.authorization, auth=auth,
        role=args.role)
    try:
        def interrupted(_number: int, _frame: object) -> None:
            raise base.MonitorError("monitor received a termination signal")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            free = monitor.arm()
            self_proc = system.proc(os.getpid())
            if self_proc is None:
                raise base.MonitorError("monitor process identity unavailable")
            base.write_ready(ready, {
                "schema": PAIR_SCHEMA, "state": "READY_BEFORE_LAUNCH",
                "label": binding["label"], "role": args.role,
                "pair_binding_sha256": pair_sha,
                "base_binding_sha256": binding_sha,
                "request_sha256": binding["request_sha256"],
                "wrapper_script_sha256": pair["wrapper_script_sha256"],
                "pid": os.getpid(), "uid": os.getuid(),
                "birth_ticks": self_proc.birth_ticks,
                "ready_monotonic": system.monotonic(),
                "initial_root_free_bytes": free,
                "poll_seconds": binding["poll_seconds"],
            })
            return monitor.run()
        except BaseException as error:
            return monitor.fail_closed(error)
    finally:
        journal.close()


if __name__ == "__main__":
    sys.exit(main())
