#!/usr/bin/env python3
"""Synthetic interlock tests; never connect to a host or spawn a guard."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import fresh_start_monitor_interlock as gate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    path.chmod(0o600)


class PairedLaunchInterlockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        self.label = "fs-synthetic-c5c-a0-r3"
        self.uid = os.getuid()
        self.free = 12 * 1024**3
        self.pair_path = d / "pair.json"
        self.base_path = d / "base.json"
        self.auth_path = d / "auth.json"
        self.request_path = d / "request.json"
        self.wrapper_path = d / "paired_root_free_guard_monitor.py"
        self.monitor_path = d / "root_free_guard_monitor.py"
        self.wrapper_path.write_text("# synthetic wrapper\n")
        self.monitor_path.write_text("# synthetic base\n")
        self.wrapper_path.chmod(0o600)
        self.monitor_path.chmod(0o600)
        self.request = {
            "label": self.label,
            "remote": "/home/ubuntu/trainium_freshstart_synthetic_r3",
            "expected_commit": "e" * 40,
            "expected_sha256": {"train.py": "c5c-sha"},
            "min_free_bytes": 6979321856,
            "pre_launch_required_free_bytes": 10737418240,
            "dispatch_authorized": True,
            "monitor_interlock": {
                "pair_binding_path": str(self.pair_path),
                "interlock_path": str(d / "interlock.json"),
                "no_spawn_path": str(d / "no-spawn.json"),
            },
        }
        write_json(self.request_path, self.request)
        self.binding = {
            "label": self.label, "remote": self.request["remote"],
            "request_path": str(self.request_path),
            "request_sha256": digest(self.request_path),
            "expected_commit": self.request["expected_commit"],
            "expected_sha256": self.request["expected_sha256"],
            "monitor_script_sha256": digest(self.monitor_path),
            "floor_free_bytes": self.request["min_free_bytes"],
            "stop_free_bytes": 9126805504,
            "prelaunch_free_bytes": self.request["pre_launch_required_free_bytes"],
        }
        write_json(self.base_path, self.binding)
        roles = {
            role: {"ready_path": str(d / f"{role}.ready.json"),
                   "events_path": str(d / f"{role}.events.jsonl")}
            for role in ("primary", "peer")
        }
        self.pair = {
            "schema": gate.PAIR_SCHEMA, "package_state": "STATIC_ONLY",
            "label": self.label, "remote": self.request["remote"],
            "expected_hostname": socket.gethostname(), "expected_uid": self.uid,
            "expected_python": "/usr/bin/python3",
            "wrapper_path": str(self.wrapper_path),
            "wrapper_script_sha256": digest(self.wrapper_path),
            "base_monitor_path": str(self.monitor_path),
            "base_binding_path": str(self.base_path),
            "base_binding_sha256": digest(self.base_path),
            "authorization_path": str(self.auth_path),
            "request_sha256": digest(self.request_path),
            "stop_free_bytes": self.binding["stop_free_bytes"],
            "interlock_path": self.request["monitor_interlock"]["interlock_path"],
            "no_spawn_path": self.request["monitor_interlock"]["no_spawn_path"],
            "roles": roles,
        }
        write_json(self.pair_path, self.pair)
        self.auth = {
            "schema": gate.AUTH_SCHEMA, "authorized": True,
            "paired_authorized": True, "dispatch_authorized": True,
            "one_attempt_only": True, "label": self.label,
            "binding_sha256": digest(self.base_path),
            "pair_binding_sha256": digest(self.pair_path),
            "request_sha256": digest(self.request_path),
        }
        write_json(self.auth_path, self.auth)
        self.procs = {}
        for role, pid, birth, peer_pid, peer_birth in (
            ("primary", 12100, 99101, 12200, 99201),
            ("peer", 12200, 99201, 12100, 99101),
        ):
            ready = {
                "schema": gate.PAIR_SCHEMA, "state": "READY_BEFORE_LAUNCH",
                "label": self.label, "role": role,
                "pair_binding_sha256": digest(self.pair_path),
                "base_binding_sha256": digest(self.base_path),
                "request_sha256": digest(self.request_path),
                "wrapper_script_sha256": digest(self.wrapper_path),
                "pid": pid, "uid": self.uid, "birth_ticks": birth,
            }
            write_json(Path(roles[role]["ready_path"]), ready)
            argv = (
                "/usr/bin/python3", str(self.wrapper_path),
                "--binding", str(self.base_path),
                "--authorization", str(self.auth_path),
                "--pair-binding", str(self.pair_path), "--role", role,
            )
            self.procs[pid] = gate.Process(pid, self.uid, birth, argv,
                                           os.path.realpath("/usr/bin/python3"))
            with Path(roles[role]["events_path"]).open("w") as stream:
                stream.write(json.dumps({"event": "PEER_BOUND", "peer_role":
                                         "peer" if role == "primary" else "primary",
                                         "peer_pid": peer_pid,
                                         "peer_birth_ticks": peer_birth}) + "\n")
                stream.write(json.dumps({"event": "ROOT_SAMPLE",
                                         "sample_monotonic": 100.25,
                                         "state": "WAITING_FOR_GUARD",
                                         "guard_pid": None,
                                         "free_bytes": self.free}) + "\n")
            Path(roles[role]["events_path"]).chmod(0o600)
        self.patches = [
            mock.patch.object(gate, "read_proc", side_effect=lambda pid: self.procs.get(pid)),
            mock.patch.object(gate, "root_free_bytes", side_effect=lambda: self.free),
            mock.patch.object(gate.time, "monotonic", return_value=100.5),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_both_exact_live_monitors_pass(self):
        snapshot = gate.require_live_pair(self.request)
        self.assertEqual(set(snapshot["roles"]), {"primary", "peer"})
        self.assertEqual(snapshot["root_free_bytes"], self.free)

    def test_stale_peer_sample_blocks_guard(self):
        events = Path(self.pair["roles"]["peer"]["events_path"])
        data = events.read_text().replace("100.25", "98.0")
        events.write_text(data)
        with self.assertRaisesRegex(gate.InterlockError, "stale or unsafe"):
            gate.require_live_pair(self.request)

    def test_peer_pid_reuse_blocks_guard(self):
        old = self.procs[12200]
        self.procs[12200] = gate.Process(old.pid, old.uid, old.birth_ticks + 1,
                                          old.argv, old.exe)
        with self.assertRaisesRegex(gate.InterlockError, "process identity"):
            gate.require_live_pair(self.request)

    def test_monitor_dies_after_first_check_before_spawn(self):
        gate.require_live_pair(self.request)  # preflight entry was safe
        self.procs[12200] = None  # peer dies while preflight is still running
        with self.assertRaisesRegex(gate.InterlockError, "process identity"):
            gate.require_live_pair(self.request)  # final no-spawn interlock

    def test_both_monitors_unavailable_block_guard(self):
        self.procs[12100] = None
        self.procs[12200] = None
        with self.assertRaisesRegex(gate.InterlockError, "process identity"):
            gate.require_live_pair(self.request)

    def test_foreign_peer_argv_blocks_guard(self):
        old = self.procs[12200]
        self.procs[12200] = gate.Process(old.pid, old.uid, old.birth_ticks,
                                          ("/usr/bin/python3", "foreign.py"), old.exe)
        with self.assertRaisesRegex(gate.InterlockError, "process identity"):
            gate.require_live_pair(self.request)

    def test_missing_mutual_handshake_blocks_guard(self):
        events = Path(self.pair["roles"]["primary"]["events_path"])
        events.write_text(events.read_text().splitlines(keepends=True)[-1])
        with self.assertRaisesRegex(gate.InterlockError, "mutual heartbeat"):
            gate.require_live_pair(self.request)

    def test_monitor_failure_event_blocks_guard(self):
        events = Path(self.pair["roles"]["primary"]["events_path"])
        with events.open("a") as stream:
            stream.write('{"event":"MONITOR_FAILURE"}\n')
        with self.assertRaisesRegex(gate.InterlockError, "monitor failure"):
            gate.require_live_pair(self.request)

    def test_sub_ten_gib_root_blocks_guard(self):
        self.free = self.request["pre_launch_required_free_bytes"] - 1
        with self.assertRaisesRegex(gate.InterlockError, "10 GiB"):
            gate.require_live_pair(self.request)

    def test_absent_pair_authorization_blocks_guard(self):
        self.auth["paired_authorized"] = False
        write_json(self.auth_path, self.auth)
        with self.assertRaisesRegex(gate.InterlockError, "authorization"):
            gate.require_live_pair(self.request)

    def test_mutated_monitor_source_blocks_guard(self):
        self.wrapper_path.write_text("# mutated source\n")
        with self.assertRaisesRegex(gate.InterlockError, "source bytes"):
            gate.require_live_pair(self.request)

    def test_no_spawn_receipt_is_durable_and_non_overwriting(self):
        gate.write_no_spawn(self.request, "before_guard_spawn", gate.InterlockError("stale"))
        receipt = json.loads(Path(self.request["monitor_interlock"]["no_spawn_path"]).read_text())
        self.assertEqual(receipt["status"], "NO_GUARD_SPAWNED")
        with self.assertRaises(FileExistsError):
            gate.write_no_spawn(self.request, "again", gate.InterlockError("again"))

    def test_launcher_interlock_precedes_guard_spawn(self):
        source = Path(__file__).with_name("fresh_start_launch.py").read_text()
        self.assertLess(source.index("require_live_pair(m)"),
                        source.index("subprocess.Popen([sys.executable"))
        self.assertIn("write_no_spawn(m, stage, error)", source)
        tree = ast.parse(source)
        remote = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "REMOTE_SCRIPT"
                              for target in node.targets))
        compile(remote, "synthetic-remote-launch-script", "exec")


if __name__ == "__main__":
    unittest.main()
