"""Offline adversarial tests only: no SSH, AWS, CPU G2 or filesystem cleanup."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


local = load("a2_final_local", "cpu_g2_invocation_final.py")
remote = load("a2_final_remote", "remote_cpu_g2_once.py")


def snapshot() -> dict:
    source = {"path": remote.SOURCE,
              "entry": {"type": "directory", "device": 66305,
                        "inode": 2432738, "uid": 1000, "mode": "0775"},
              "git": {"commit": remote.COMMIT, "rc": 0},
              "git_status": {"clean": True},
              "source_hashes": {
                  "train.py": {"sha256": remote.TRAIN_SHA},
                  "prepare.py": {"sha256": remote.PREPARE_SHA}}}
    mount = {"point": "/", "filesystem": "ext4", "mount_id": "37",
             "major_minor": "259:1", "source": "/dev/root"}
    return {"base": {"schema": "dual-host-readonly-v1", "host": "trn22",
                     "identity": {"hostname": remote.HOSTNAME, "uid": 1000,
                                  "boot_id": "11111111-2222-3333-4444-555555555555",
                                  "root_mount": mount, "home_mount": copy.deepcopy(mount),
                                  "root_device": {"device": 66305}},
                     "root_space": {"total_bytes": 250 * 1024 ** 3,
                                    "free_bytes": 54 * 1024 ** 3,
                                    "free_inodes": 50000},
                     "stops": {"present": {}},
                     "activity": {"processes": {"matched": [],
                                                "unreadable_cmdlines": 0},
                                  "neuron": {"rc": 0, "json_valid": True,
                                             "device_count": 1,
                                             "non_diagnostic_pids": []},
                                  "host_lock_fuser": {"path_exists": True,
                                                      "rc": 1, "pids": []},
                                  "neuron0_fuser": {"path_exists": True,
                                                    "rc": 1, "pids": []}},
                     "checkouts": {"discovery_truncated": False,
                                   "entries": {"e65_exact": source}}},
            "memory_available_bytes": 122 * 1024 ** 3,
            "git_pointer": {"sha256":
                            "dc53f6887bc5f9c2c1bbdbbd60911cdbeb076f7c57bafd796ee5b96f69ecbf1b",
                            "inode": 2432740, "mode": "0664"},
            "linked_gitdir": {"type": "directory", "inode": 2432736,
                              "mode": "0775"},
            "g2_tool": {"sha256": remote.G2_SHA},
            "contained_tool": {"sha256": remote.CONTAINED_SHA},
            "scratch": {path: {"exists": False} for path in
                        (remote.ROOT, remote.TMP, remote.PROCESS,
                         remote.STEP, remote.RECEIPT)},
            "label_matches": []}


class FinalPendingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = local.build_plan()

    def test_exact_disabled_plan_and_binding(self):
        p = self.plan
        self.assertFalse(p["dispatch_authorized"])
        self.assertEqual(len(p["request"]["science"]["train_argv"]), 27)
        self.assertEqual(p["request"]["science"]["train_argv"],
                         local._r3_plan()["train_argv"])
        self.assertEqual(json.loads(local.BINDING.read_text()), local.expected_binding(p))
        self.assertIn("--trace-rmsprop", p["commands"]["launch"])
        self.assertIn("--keep", p["commands"]["launch"])
        self.assertNotIn("--real-data", p["commands"]["launch"])
        self.assertIn("StrictHostKeyChecking=yes", p["audit_argv"])
        self.assertEqual(p["commands"]["launch"].count("--nproc"), 1)

    def test_direct_entrypoints_refuse_dispatch(self):
        for optimize in ([], ["-O"]):
            result = subprocess.run([sys.executable, *optimize, "-B",
                                     str(local.HERE / "cpu_g2_invocation_final.py")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("DISPATCH HOLD", result.stderr)
            remote_result = subprocess.run([sys.executable, *optimize, "-B",
                                            str(local.HERE / "remote_cpu_g2_once.py")],
                                           capture_output=True, text=True)
            self.assertNotEqual(remote_result.returncode, 0)
            self.assertEqual(json.loads(remote_result.stdout)["status"],
                             "HOLD_PRESERVE_PARTIAL_NO_RETRY")

    def test_missing_actual_review_blocks_before_marker_or_transport(self):
        calls = []

        def forbidden(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("transport must not be reached")

        with self.assertRaises(RuntimeError):
            local.operator_once(transport=forbidden)
        self.assertEqual(calls, [])

    def test_request_and_binding_mutations_fail_closed(self):
        request_raw = local.REQUEST.read_bytes()
        request = json.loads(request_raw)
        binding_raw = local.BINDING.read_bytes()
        original = Path.read_bytes
        for section, field, value in (
            ("capacity", "minimum_fresh_root_free_inodes", 0),
            ("capacity", "scratch_peak_projection_bytes", 0),
            ("capacity", "minimum_fresh_memory_available_bytes", 0),
            ("host", "git_pointer_sha256", "0" * 64),
            ("host", "linked_gitdir_mode", "0777"),
            ("science", "label", "reused-label"),
        ):
            forged = copy.deepcopy(request)
            forged[section][field] = value

            def changed(path, *args, **kwargs):
                if path == local.REQUEST:
                    return json.dumps(forged).encode()
                return original(path, *args, **kwargs)

            with self.subTest(section=section, field=field), patch.object(Path, "read_bytes", changed):
                with self.assertRaises(RuntimeError):
                    local.build_plan()

        forged_binding = json.loads(binding_raw)
        forged_binding["commands_sha256"] = "0" * 64

        def changed_binding(path, *args, **kwargs):
            if path == local.BINDING:
                return json.dumps(forged_binding).encode()
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_bytes", changed_binding):
            with self.assertRaises(RuntimeError):
                local.build_plan()

    def test_remote_authenticates_exact_request_and_commands_independently(self):
        payload = json.loads(self.plan["stdin_payload"])
        remote.authenticate_payload(payload)
        for name, mutate in (
            ("request", lambda p: p["request"]["capacity"].update(
                minimum_fresh_root_free_inodes=0)),
            ("command", lambda p: p["commands"]["wait"].append("--unexpected")),
            ("probe", lambda p: p.update(probe_source="print('not pinned')")),
        ):
            forged = copy.deepcopy(payload)
            mutate(forged)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                remote.authenticate_payload(forged)

    def test_fresh_preflight_thresholds_and_host_gates(self):
        item = snapshot()
        remote.validate_snapshot(item, self.plan["request"], phase="before")
        mutations = (
            ("root_floor", lambda s: s["base"]["root_space"].update(free_bytes=31 * 1024 ** 3)),
            ("inode_floor", lambda s: s["base"]["root_space"].update(free_inodes=9999)),
            ("memory_floor", lambda s: s.update(memory_available_bytes=95 * 1024 ** 3)),
            ("stop", lambda s: s["base"]["stops"].update(present={"/tmp/STOP": {"exists": True}})),
            ("process", lambda s: s["base"]["activity"]["processes"]["matched"].append({"pid": 1})),
            ("device", lambda s: s["base"]["activity"]["neuron0_fuser"].update(pids=[123])),
            ("lock", lambda s: s["base"]["activity"]["host_lock_fuser"].update(pids=[123])),
            ("git_pointer", lambda s: s["git_pointer"].update(mode="0777")),
            ("tool", lambda s: s["g2_tool"].update(sha256="0" * 64)),
            ("label", lambda s: s["label_matches"].append("/home/ubuntu/old-label")),
        )
        for name, mutate in mutations:
            forged = copy.deepcopy(item)
            mutate(forged)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                remote.validate_snapshot(forged, self.plan["request"], phase="before")

    def test_prepared_scratch_requires_exact_root_tmp_only(self):
        item = snapshot()
        for path in (remote.ROOT, remote.TMP):
            item["scratch"][path] = {"exists": True, "type": "directory",
                                      "device": 66305, "uid": 1000, "mode": "0700"}
        remote.validate_snapshot(item, self.plan["request"], phase="prepared")
        item["scratch"][remote.PROCESS] = {"exists": True}
        with self.assertRaises(RuntimeError):
            remote.validate_snapshot(item, self.plan["request"], phase="prepared")

    def test_transport_failure_still_attempts_distinct_postflight(self):
        class Journal:
            def __init__(self):
                self.events = []

            def emit(self, event, **fields):
                self.events.append(event)

        journal = Journal()
        calls = []

        def fake_transport(argv, *, input_data, timeout_seconds):
            calls.append(argv)
            return types.SimpleNamespace(returncode=1, stdout=b"", stderr=b"failure")

        result = local.run_with_postflight(self.plan, journal, fake_transport,
                                           {"review_sha256": "a", "assignment_sha256": "b"})
        self.assertEqual(result["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
        self.assertEqual(calls, [self.plan["audit_argv"], self.plan["audit_argv"]])
        self.assertIn("POSTFLIGHT_HOLD", journal.events)

    def test_final_scratch_guard_exact_tmp_and_no_fallback(self):
        code = (HERE / "scratch_guard.py").read_text()
        with tempfile.TemporaryDirectory(prefix="a2-final-guard-") as parent:
            root = str(Path(parent) / "new-root")
            tmp = root + "/tmp"
            env = dict(os.environ, TMPDIR=tmp)
            base = [sys.executable, "-B", "-c", code]
            fields = [root, tmp, "1", "1", "1", "1", "--end"]
            missing = subprocess.run(base + ["guard", *fields], env=env,
                                     capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            prepared = subprocess.run(base + ["prepare", *fields], env=env,
                                      capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(prepared.stdout)["tempfile_root"], tmp)
            repeated = subprocess.run(base + ["prepare", *fields], env=env,
                                      capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            guarded = subprocess.run(base + ["guard", *fields], env=env,
                                     capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(guarded.stdout)["tempfile_root"], tmp)


if __name__ == "__main__":
    unittest.main()
