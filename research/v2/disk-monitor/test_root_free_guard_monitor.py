#!/usr/bin/env python3
"""Synthetic tests only: no host connection, launcher, or signal is used."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("root_free_guard_monitor.py")
SPEC = importlib.util.spec_from_file_location("root_free_guard_monitor", MODULE_PATH)
assert SPEC and SPEC.loader
monitor_module = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = monitor_module
SPEC.loader.exec_module(monitor_module)


class MemoryJournal:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


class FakeSystem:
    def __init__(self, binding, request):
        self.binding = binding
        self.request = request
        self.time = 0.0
        self.free = binding["prelaunch_free_bytes"] + 1024**3
        self.files = {}
        self.hashes = {
            Path(binding["request_path"]): binding["request_sha256"],
            MODULE_PATH: binding["monitor_script_sha256"],
        }
        for relpath, sha in binding["expected_sha256"].items():
            self.hashes[Path(binding["remote"]) / relpath] = sha
        self.procs = {}
        self.signals = []
        self.pidfds = []
        self.pidfd_identities = {}
        self.owned_pids = set()
        self.closed = []
        self.fail_free = False

    def free_bytes(self):
        if self.fail_free:
            raise OSError("synthetic statvfs failure")
        return self.free

    def monotonic(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds

    def exists(self, path):
        return path in self.files

    def read_json(self, path):
        return self.files[path]

    def sha256(self, path):
        return self.hashes[path]

    def git_head(self, remote):
        return self.binding["expected_commit"]

    def proc(self, pid):
        return self.procs.get(pid)

    def open_pidfd(self, pid):
        self.pidfds.append(pid)
        pidfd = pid + 100000
        self.pidfd_identities[pidfd] = (pid, self.procs[pid].birth_ticks)
        return pidfd

    def pidfd_alive(self, pidfd):
        pid, birth = self.pidfd_identities[pidfd]
        proc = self.procs.get(pid)
        return proc is not None and proc.birth_ticks == birth

    def term_pidfd(self, pidfd):
        self.signals.append(pidfd)

    def kill_pidfd(self, pidfd):
        self.signals.append(("KILL", pidfd))

    def owned_processes(self, _label):
        return [self.procs[pid] for pid in self.owned_pids
                if self.procs.get(pid) is not None]

    def session_processes(self, session):
        return [proc for proc in self.procs.values()
                if proc is not None and proc.session == session]

    def recent_processes(self, minimum_birth_ticks):
        return [proc for proc in self.procs.values()
                if proc is not None and proc.birth_ticks >= minimum_birth_ticks]

    def has_marker(self, pid, _label):
        return pid in self.owned_pids

    def close_pidfd(self, pidfd):
        self.closed.append(pidfd)


class GuardMonitorTest(unittest.TestCase):
    def setUp(self):
        remote = "/home/ubuntu/trainium_freshstart_test"
        label = "fs-synthetic-a0-r2"
        floor = 6979321856
        self.binding = {
            "schema": monitor_module.SCHEMA,
            "package_state": "STATIC_ONLY",
            "label": label,
            "remote": remote,
            "request_path": "/home/ubuntu/external-monitor/request.json",
            "request_sha256": "request-sha",
            "monitor_script_sha256": "script-sha",
            "expected_commit": "e" * 40,
            "expected_sha256": {
                "train.py": "train-sha",
                "research/fresh_start_guard.py": "guard-sha",
            },
            "expected_hostname": socket.gethostname(),
            "expected_uid": os.getuid(),
            "expected_guard_python": "/usr/bin/python3",
            "floor_free_bytes": floor,
            "stop_free_bytes": floor + monitor_module.TWO_GIB,
            "prelaunch_free_bytes": 10737418240,
            "poll_seconds": 0.25,
            "bind_timeout_seconds": 120,
            "post_stop_timeout_seconds": 120,
        }
        self.request = {
            "label": label,
            "remote": remote,
            "expected_commit": "e" * 40,
            "expected_hostname": socket.gethostname(),
            "expected_sha256": self.binding["expected_sha256"],
            "min_free_bytes": floor,
            "pre_launch_required_free_bytes": 10737418240,
            "command": ["timeout", "--foreground", "3600s", "env", "-i", "torchrun"],
        }
        self.system = FakeSystem(self.binding, self.request)
        self.journal = MemoryJournal()
        self.monitor = monitor_module.GuardMonitor(
            self.binding, self.request, self.system, self.journal
        )
        self.monitor.arm()
        self.guard_pid = 12345
        self.child_pid = 12346
        guard_exe = os.path.realpath("/usr/bin/python3")
        guard_argv = (
            "/usr/bin/python3",
            f"{remote}/research/fresh_start_guard.py",
            f"{remote}/research/run-logs/{label}.job.json",
            "--lock-fd",
            "4",
        )
        self.guard = monitor_module.ProcInfo(
            self.guard_pid, os.getuid(), 88001, 1, self.guard_pid, self.guard_pid,
            guard_argv, guard_exe,
        )
        self.child = monitor_module.ProcInfo(
            self.child_pid, os.getuid(), 88002, self.guard_pid, self.child_pid,
            self.child_pid, tuple(self.request["command"]), "/usr/bin/timeout",
        )
        self.job = dict(self.request, source_sha256=self.binding["expected_sha256"])
        self.process = {
            "label": label,
            "guard_pid": self.guard_pid,
            "child_pid": self.child_pid,
            "child_session": self.child_pid,
            "command": self.request["command"],
        }

    def stage_live_guard(self):
        s, m = self.system, self.monitor
        s.files[m.job_path] = self.job
        s.files[m.process_path] = self.process
        s.hashes[m.job_path] = "job-sha"
        s.hashes[m.process_path] = "process-sha"
        s.procs[self.guard_pid] = self.guard
        s.procs[self.child_pid] = self.child
        s.owned_pids.update((self.guard_pid, self.child_pid))
        s.time += 0.25
        self.assertIsNone(m.tick())
        self.assertEqual(m.state, "BOUND")

    def test_preemptive_threshold_signals_exact_pidfd_once(self):
        self.stage_live_guard()
        self.system.free = self.binding["stop_free_bytes"]
        self.system.time += 0.25
        self.assertIsNone(self.monitor.tick())
        self.assertEqual(self.system.signals, [self.guard_pid + 100000])
        self.assertEqual(self.monitor.state, "STOPPING")
        self.assertIn("TERM_INTENT", [event for event, _ in self.journal.events])
        self.assertIn("TERM_SENT", [event for event, _ in self.journal.events])

    def test_high_free_and_clean_completion_never_signals(self):
        self.stage_live_guard()
        self.system.time += 0.25
        self.assertIsNone(self.monitor.tick())
        self.system.procs[self.guard_pid] = None
        self.system.procs[self.child_pid] = None
        self.system.files[self.monitor.postflight_path] = {
            "label": self.binding["label"], "host_clean": True, "command_returncode": 0,
        }
        self.system.time += 0.25
        self.assertEqual(self.monitor.tick(), 0)
        self.assertEqual(self.system.signals, [])

    def test_reused_guard_pid_never_signals_replacement(self):
        self.stage_live_guard()
        self.system.procs[self.guard_pid] = monitor_module.ProcInfo(
            self.guard.pid, self.guard.uid, self.guard.birth_ticks + 1,
            self.guard.ppid, self.guard.pgrp, self.guard.session,
            self.guard.argv, self.guard.exe,
        )
        self.system.free = self.binding["stop_free_bytes"]
        self.system.time += 0.25
        self.assertIsNone(self.monitor.tick())
        self.assertNotIn(self.guard_pid + 100000, self.system.signals)
        self.assertNotIn(("KILL", self.guard_pid + 100000), self.system.signals)
        self.assertIn(("KILL", self.child_pid + 100000), self.system.signals)
        self.assertIn("KILL_REFUSED", [event for event, _ in self.journal.events])

    def test_foreign_guard_uid_refuses_binding(self):
        self.system.files[self.monitor.job_path] = self.job
        self.system.files[self.monitor.process_path] = self.process
        self.system.procs[self.guard_pid] = monitor_module.ProcInfo(
            self.guard.pid, self.guard.uid + 1, self.guard.birth_ticks,
            self.guard.ppid, self.guard.pgrp, self.guard.session,
            self.guard.argv, self.guard.exe,
        )
        self.system.procs[self.child_pid] = self.child
        self.system.time += 0.25
        with self.assertRaisesRegex(monitor_module.MonitorError, "UID") as raised:
            self.monitor.tick()
        self.assertEqual(self.monitor.fail_closed(raised.exception), 3)
        self.assertEqual(self.system.signals, [])

    def test_monitor_sample_failure_stops_only_authenticated_guard(self):
        self.stage_live_guard()
        self.system.fail_free = True
        self.system.time += 0.25
        with self.assertRaises(OSError) as raised:
            self.monitor.tick()
        self.assertEqual(self.monitor.fail_closed(raised.exception), 3)
        self.assertEqual(self.system.signals[0], self.guard_pid + 100000)
        self.assertIn(("KILL", self.guard_pid + 100000), self.system.signals)
        self.assertIn(("KILL", self.child_pid + 100000), self.system.signals)
        self.assertEqual(self.monitor.stop_reason, "MONITOR_FAILURE")

    def test_missing_process_receipt_fails_without_signal(self):
        self.system.files[self.monitor.job_path] = self.job
        for _ in range(42):
            self.system.time += 0.25
            try:
                self.monitor.tick()
            except monitor_module.MonitorError as error:
                raised = error
                break
        else:
            self.fail("missing process receipt did not fail")
        self.assertIn("process receipt missing", str(raised))
        self.assertEqual(self.monitor.fail_closed(raised), 3)
        self.assertEqual(self.system.signals, [])

    def test_low_space_before_guard_binding_refuses_foreign_signal(self):
        self.system.free = self.binding["stop_free_bytes"]
        self.system.time += 0.25
        with self.assertRaisesRegex(monitor_module.MonitorError, "before guard authentication") as raised:
            self.monitor.tick()
        self.assertEqual(self.monitor.fail_closed(raised.exception), 3)
        self.assertEqual(self.system.signals, [])

    def test_reused_child_pid_is_not_signaled_but_owned_guard_is_stopped(self):
        self.stage_live_guard()
        self.system.procs[self.child_pid] = monitor_module.ProcInfo(
            self.child.pid, self.child.uid, self.child.birth_ticks + 1,
            self.child.ppid, self.child.pgrp, self.child.session,
            self.child.argv, self.child.exe,
        )
        self.system.free = self.binding["stop_free_bytes"]
        self.system.time += 0.25
        self.assertIsNone(self.monitor.tick())
        self.assertEqual(self.system.signals, [self.guard_pid + 100000])
        self.assertNotIn(("KILL", self.child_pid + 100000), self.system.signals)

    def test_poll_lag_stops_authenticated_guard(self):
        self.stage_live_guard()
        self.system.time += 1.1
        with self.assertRaisesRegex(monitor_module.MonitorError, "poll gap") as raised:
            self.monitor.tick()
        self.assertEqual(self.monitor.fail_closed(raised.exception), 3)
        self.assertEqual(self.system.signals[0], self.guard_pid + 100000)
        self.assertIn(("KILL", self.guard_pid + 100000), self.system.signals)

    def test_replaced_job_manifest_triggers_owned_stop(self):
        self.stage_live_guard()
        self.system.hashes[self.monitor.job_path] = "changed-sha"
        self.system.time += 0.25
        with self.assertRaisesRegex(monitor_module.MonitorError, "job receipt changed") as raised:
            self.monitor.tick()
        self.assertEqual(self.monitor.fail_closed(raised.exception), 3)
        self.assertEqual(self.system.signals[0], self.guard_pid + 100000)
        self.assertIn(("KILL", self.guard_pid + 100000), self.system.signals)

    def test_real_proc_reader_observes_own_uid_and_birth_ticks(self):
        proc = monitor_module.RealSystem.proc(os.getpid())
        self.assertIsNotNone(proc)
        self.assertEqual(proc.uid, os.getuid())
        self.assertGreater(proc.birth_ticks, 0)
        self.assertEqual(proc.pid, os.getpid())

    def test_static_package_requires_distinct_true_exact_hash_authorization(self):
        with tempfile.TemporaryDirectory(prefix="trainium-monitor-unit-") as directory:
            root = Path(directory)
            request_path = root / "request.json"
            binding_path = root / "binding.json"
            auth_path = root / "auth.json"
            request_path.write_text(json.dumps(self.request))
            binding = dict(self.binding, request_path=str(request_path))
            binding["request_sha256"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
            binding_path.write_text(json.dumps(binding))
            binding_sha = hashlib.sha256(binding_path.read_bytes()).hexdigest()
            auth = {
                "schema": monitor_module.AUTH_SCHEMA,
                "authorized": False,
                "binding_sha256": binding_sha,
                "label": binding["label"],
                "request_sha256": binding["request_sha256"],
                "one_attempt_only": True,
            }
            auth_path.write_text(json.dumps(auth))
            with self.assertRaisesRegex(monitor_module.MonitorError, "no one-shot"):
                monitor_module._load_authorized_binding(binding_path, auth_path)
            auth["authorized"] = True
            auth["binding_sha256"] = "wrong"
            auth_path.write_text(json.dumps(auth))
            with self.assertRaisesRegex(monitor_module.MonitorError, "exact one-label"):
                monitor_module._load_authorized_binding(binding_path, auth_path)

    def test_journal_is_append_only_and_refuses_existing_evidence_path(self):
        with tempfile.TemporaryDirectory(prefix="trainium-monitor-journal-") as directory:
            path = Path(directory) / "events.jsonl"
            journal = monitor_module.Journal(path)
            journal.emit("ROOT_SAMPLE", free_bytes=123)
            journal.close()
            self.assertEqual(json.loads(path.read_text())["free_bytes"], 123)
            with self.assertRaises(FileExistsError):
                monitor_module.Journal(path)


if __name__ == "__main__":
    unittest.main()
