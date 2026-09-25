#!/usr/bin/env python3
"""Synthetic peer-death/late-bind tests; no host access or real signals."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import root_free_guard_monitor as base
import paired_root_free_guard_monitor as pairmod
from test_root_free_guard_monitor import FakeSystem, MemoryJournal


class PairMonitorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        label = "fs-synthetic-c5c-r3"
        remote = "/home/ubuntu/trainium_freshstart_synthetic_r3"
        self.binding_path, self.pair_path, self.auth_path = (
            d / "base.json", d / "pair.json", d / "auth.json")
        self.binding = {
            "schema": base.SCHEMA, "package_state": "STATIC_ONLY",
            "label": label, "remote": remote,
            "request_path": str(d / "request.json"),
            "request_sha256": "request-sha",
            "monitor_script_sha256": "base-script-sha",
            "expected_commit": "e" * 40,
            "expected_sha256": {"train.py": "train-sha",
                                "research/fresh_start_guard.py": "guard-sha"},
            "expected_hostname": socket.gethostname(),
            "expected_uid": os.getuid(),
            "expected_guard_python": "/usr/bin/python3",
            "floor_free_bytes": 6979321856,
            "stop_free_bytes": 9126805504,
            "prelaunch_free_bytes": 10737418240,
            "poll_seconds": 0.25, "bind_timeout_seconds": 120,
            "post_stop_timeout_seconds": 120,
        }
        self.request = {
            "label": label, "remote": remote,
            "expected_commit": "e" * 40,
            "expected_hostname": socket.gethostname(),
            "expected_sha256": self.binding["expected_sha256"],
            "min_free_bytes": self.binding["floor_free_bytes"],
            "pre_launch_required_free_bytes": self.binding["prelaunch_free_bytes"],
            "command": ["timeout", "--foreground", "3600s", "env", "-i", "torchrun"],
        }
        self.pair = {
            "schema": pairmod.PAIR_SCHEMA, "package_state": "STATIC_ONLY",
            "label": label, "remote": remote,
            "request_sha256": "request-sha",
            "expected_hostname": socket.gethostname(),
            "expected_uid": os.getuid(),
            "expected_python": "/usr/bin/python3",
            "wrapper_path": str(Path(pairmod.__file__)),
            "wrapper_script_sha256": "pair-script-sha",
            "base_monitor_path": str(Path(base.__file__)),
            "base_binding_path": str(self.binding_path),
            "base_binding_sha256": "base-binding-sha",
            "authorization_path": str(self.auth_path),
            "peer_start_timeout_seconds": 120,
            "no_spawn_path": str(d / "no-spawn.json"),
            "roles": {
                "primary": {"ready_path": str(d / "primary.ready.json"),
                            "events_path": str(d / "primary.events.jsonl")},
                "peer": {"ready_path": str(d / "peer.ready.json"),
                         "events_path": str(d / "peer.events.jsonl")},
            },
        }
        self.auth = {
            "binding_sha256": "base-binding-sha",
            "pair_binding_sha256": "pair-binding-sha",
        }
        self.system = FakeSystem(self.binding, self.request)
        self.system.hashes.update({
            self.pair_path: "pair-binding-sha",
            self.binding_path: "base-binding-sha",
            Path(pairmod.__file__): "pair-script-sha",
            Path(base.__file__): "base-script-sha",
        })
        self.journal = MemoryJournal()
        self.monitor = pairmod.PairedGuardMonitor(
            self.binding, self.request, self.system, self.journal,
            pair=self.pair, pair_path=self.pair_path,
            binding_path=self.binding_path, auth_path=self.auth_path,
            auth=self.auth, role="peer")
        self.monitor.arm()
        self.peer_pid = 12300
        self.peer_birth = 999
        self.peer_proc = base.ProcInfo(
            self.peer_pid, os.getuid(), self.peer_birth, 1,
            self.peer_pid, self.peer_pid,
            pairmod.exact_peer_argv(self.pair, "primary", self.binding_path,
                                    self.auth_path, self.pair_path),
            os.path.realpath("/usr/bin/python3"))
        self.system.procs[self.peer_pid] = self.peer_proc
        self.peer_ready = {
            "schema": pairmod.PAIR_SCHEMA, "state": "READY_BEFORE_LAUNCH",
            "label": label, "role": "primary",
            "pair_binding_sha256": "pair-binding-sha",
            "base_binding_sha256": "base-binding-sha",
            "request_sha256": "request-sha",
            "wrapper_script_sha256": "pair-script-sha",
            "pid": self.peer_pid, "uid": os.getuid(),
            "birth_ticks": self.peer_birth, "ready_monotonic": 0.0,
        }
        self.system.files[Path(self.pair["roles"]["primary"]["ready_path"])] = self.peer_ready
        self.system.files[Path(self.pair["roles"]["primary"]["events_path"])] = {}
        self.sample = {"event": "ROOT_SAMPLE", "sample_monotonic": 0.25,
                       "state": "WAITING_FOR_GUARD", "guard_pid": None,
                       "free_bytes": self.system.free}
        self.sample_patch = mock.patch.object(pairmod, "last_root_sample",
                                              side_effect=lambda *_: (self.sample, (1, 2)))
        self.sample_patch.start()
        self.addCleanup(self.sample_patch.stop)

    def bind_peer(self):
        self.system.time = 0.25
        self.assertIsNone(self.monitor.tick())
        self.assertTrue(self.monitor.peer_bound_logged)

    def stage_guard(self):
        m, s = self.monitor, self.system
        guard_pid, child_pid = 12345, 12346
        s.files[m.job_path] = dict(self.request,
                                   source_sha256=self.binding["expected_sha256"])
        s.files[m.process_path] = {
            "label": self.binding["label"], "command": self.request["command"],
            "guard_pid": guard_pid, "child_pid": child_pid,
            "child_session": child_pid,
        }
        s.hashes[m.job_path] = "job-sha"
        s.hashes[m.process_path] = "process-sha"
        s.procs[guard_pid] = base.ProcInfo(
            guard_pid, os.getuid(), 100001, 1, guard_pid, guard_pid,
            ("/usr/bin/python3", str(m.guard_path), str(m.job_path), "--lock-fd", "4"),
            os.path.realpath("/usr/bin/python3"))
        s.procs[child_pid] = base.ProcInfo(
            child_pid, os.getuid(), 100002, guard_pid, child_pid, child_pid,
            tuple(self.request["command"]), "/usr/bin/timeout")
        s.owned_pids.update((guard_pid, child_pid))
        return guard_pid, child_pid

    def test_live_peer_does_not_signal(self):
        self.bind_peer()
        self.system.time = 0.50
        self.sample["sample_monotonic"] = 0.50
        self.assertIsNone(self.monitor.tick())
        self.assertEqual(self.system.signals, [])

    def test_peer_pid_reuse_is_refused(self):
        self.bind_peer()
        self.system.procs[self.peer_pid] = base.ProcInfo(
            self.peer_pid, os.getuid(), self.peer_birth + 1, 1,
            self.peer_pid, self.peer_pid, self.peer_proc.argv, self.peer_proc.exe)
        with self.assertRaisesRegex(base.MonitorError, "process/argv identity"):
            self.monitor._check_peer()
        self.assertEqual(self.system.signals, [])

    def test_stale_peer_sample_is_refused(self):
        self.bind_peer()
        self.system.time = 1.5
        with self.assertRaisesRegex(base.MonitorError, "heartbeat stale"):
            self.monitor._check_peer()
        self.assertEqual(self.system.signals, [])

    def test_peer_death_stops_only_authenticated_guard(self):
        self.bind_peer()
        guard_pid, _ = self.stage_guard()
        self.sample["sample_monotonic"] = 0.50
        self.sample["state"] = "BOUND"
        self.sample["guard_pid"] = guard_pid
        self.system.time = 0.50
        self.assertIsNone(self.monitor.tick())
        self.system.procs[self.peer_pid] = None
        self.system.time = 0.75
        with self.assertRaises(base.MonitorError) as raised:
            self.monitor.tick()
        self.assertEqual(self.monitor.fail_closed(raised.exception), 3)
        self.assertEqual(self.system.signals[0], guard_pid + 100000)
        self.assertIn(("KILL", guard_pid + 100000), self.system.signals)
        self.assertNotIn(self.peer_pid + 100000, self.system.signals)

    def test_pre_guard_failure_waits_for_authenticated_receipt(self):
        self.bind_peer()
        guard_pid, _ = self.stage_guard()
        def synthetic_term(pidfd):
            self.system.signals.append(pidfd)
            self.system.procs[guard_pid] = None
            self.system.procs[guard_pid + 1] = None
            self.system.files[self.monitor.postflight_path] = {
                "label": self.binding["label"], "host_clean": True,
                "command_returncode": 143,
            }
        self.system.term_pidfd = synthetic_term
        self.assertEqual(self.monitor.fail_closed(base.MonitorError("peer died")), 3)
        self.assertEqual(self.system.signals, [guard_pid + 100000])

    def test_late_process_receipt_still_stops_authenticated_guard(self):
        self.bind_peer()
        original_sleep = self.system.sleep
        injected = [False]
        def synthetic_term(pidfd):
            self.system.signals.append(pidfd)
            self.system.procs[12345] = None
            self.system.procs[12346] = None
            self.system.files[self.monitor.postflight_path] = {
                "label": self.binding["label"], "host_clean": True,
                "command_returncode": 143,
            }
        def delayed_sleep(seconds):
            original_sleep(seconds)
            if not injected[0]:
                injected[0] = True
                self.stage_guard()  # Arrives after a peer failure in preflight.
                self.system.term_pidfd = synthetic_term
        self.system.sleep = delayed_sleep
        self.assertEqual(self.monitor.fail_closed(base.MonitorError("peer died")), 3)
        self.assertTrue(injected[0])
        self.assertEqual(self.system.signals, [12345 + 100000])

    def test_no_spawn_receipt_ends_pre_guard_watch_without_signal(self):
        self.bind_peer()
        path = Path(self.pair["no_spawn_path"])
        receipt = {
            "schema": "paired-monitor-no-spawn-v1", "status": "NO_GUARD_SPAWNED",
            "label": self.binding["label"],
            "expected_commit": self.binding["expected_commit"],
            "pair_binding_path": str(self.pair_path),
        }
        path.write_text(json.dumps(receipt) + "\n")
        path.chmod(0o600)
        self.system.files[path] = receipt
        self.assertEqual(self.monitor.fail_closed(base.MonitorError("peer died")), 3)
        self.assertEqual(self.system.signals, [])

    def test_incomplete_no_spawn_receipt_is_not_abort_proof(self):
        self.bind_peer()
        path = Path(self.pair["no_spawn_path"])
        path.write_text('{"schema":"paired-monitor-no-spawn-v1"')
        path.chmod(0o600)
        self.system.files[path] = {}
        self.assertFalse(self.monitor._no_spawn_receipt())
        self.assertEqual(self.system.signals, [])

    def test_missing_process_and_no_abort_receipt_is_hold_not_signal(self):
        self.bind_peer()
        def fail_sleep(_seconds):
            raise OSError("synthetic stalled receipt")
        self.system.sleep = fail_sleep
        self.assertEqual(self.monitor.fail_closed(base.MonitorError("peer died")), 3)
        self.assertEqual(self.system.signals, [])
        self.assertIn("RESCUE_REFUSED", [name for name, _ in self.journal.events])

    def test_foreign_guard_is_never_signaled_during_rescue(self):
        self.bind_peer()
        guard_pid, _ = self.stage_guard()
        old = self.system.procs[guard_pid]
        self.system.procs[guard_pid] = base.ProcInfo(
            old.pid, old.uid + 1, old.birth_ticks, old.ppid, old.pgrp,
            old.session, old.argv, old.exe)
        self.assertEqual(self.monitor.fail_closed(base.MonitorError("peer died")), 3)
        self.assertEqual(self.system.signals, [])


if __name__ == "__main__":
    unittest.main()
