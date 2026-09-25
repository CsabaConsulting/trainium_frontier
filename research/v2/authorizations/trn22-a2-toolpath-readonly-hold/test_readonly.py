"""Offline safety/decision tests. No SSH, host marker, training or eval."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


probe = module("a2_toolpath_probe_test", "remote_tool_paths.py")
collector = module("a2_toolpath_collector_test", "collect_readonly.py")
request = json.loads((HERE / "REQUEST.HOLD.json").read_text(encoding="utf-8"))


def clean_planning_fixture():
    science = request["source"]
    host = "ip-172-31-32-253"
    boot = "30f90d95-224c-48d9-bb11-8ec18dd7c4fe"
    namespace = "mnt:[4026531840]"
    snapshot = {
        "host": "trn22",
        "identity": {"hostname": host, "uid": 1000, "boot_id": boot,
                     "mount_namespace": namespace, "root_device": {"device": 66305}},
        "checkouts": {"discovery_truncated": False, "entries": {"e65_exact": {
            "path": science["e65_checkout"],
            "entry": {"type": "directory", "device": 66305,
                      "inode": science["expected_checkout_inode"], "uid": 1000},
            "git": {"commit": science["e65_commit"], "rc": 0},
            "git_status": {"clean": True},
            "source_hashes": {
                "train.py": {"sha256": science["expected_train_sha256"]},
                "prepare.py": {"sha256": science["expected_prepare_sha256"]},
            },
        }}},
        "stops": {"present": {}},
        "activity": {
            "processes": {"matched": [], "unreadable_cmdlines": 0},
            "neuron": {"rc": 0, "json_valid": True, "non_diagnostic_pids": []},
            "host_lock_fuser": {"rc": 1, "pids": []},
            "neuron0_fuser": {"rc": 1, "pids": []},
        },
        "root_space": {"free_bytes": 50 * 1024 ** 3, "free_inodes": 50000},
    }
    gitdir = {
        "hostname": host, "uid": 1000, "boot_id": boot,
        "mount_namespace": namespace,
        "pointer": {
            "metadata": {"type": "regular", "device": 66305,
                         "inode": science["expected_git_pointer_inode"],
                         "uid": 1000, "mode": "0664"},
            "sha256": science["expected_git_pointer_sha256"],
            "resolved_gitdir": "/home/ubuntu/worktrees/e65",
            "resolved_gitdir_metadata": {"type": "directory", "device": 66305,
                                         "inode": science["expected_linked_gitdir_inode"],
                                         "uid": 1000, "mode": "0775"},
        },
    }
    tools = {
        "host": "trn22", "hostname": host, "uid": 1000,
        "boot_id": boot, "mount_namespace": namespace,
        "root_device": 66305, "checkout_path": science["e65_checkout"],
        "checkout": {"type": "directory", "device": 66305,
                     "inode": science["expected_checkout_inode"],
                     "opened_same_inode": True},
        "memory_available_bytes": 110 * 1024 ** 3,
        "tools": {
            name: {"path": expected["path"], "exists": True,
                   "type": "regular", "uid": 1000, "device": 66305,
                   "inode": 3000000 + index, "nlink": 1,
                   "sha256": expected["expected_sha256"], "bytes": 150000,
                   "read_bytes": 150000}
            for index, (name, expected) in enumerate(request["exact_tools"].items())
        },
    }
    return snapshot, gitdir, tools


class NoFollowToolTests(unittest.TestCase):
    def test_regular_hash_and_inode(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "workflow" / "tools" / "tool.py"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"read-only test tool\n")
            with mock.patch.object(probe, "TOOLS", {"g2": str(target)}):
                observed = probe.nofollow_regular(str(target))
            self.assertEqual(observed["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
            self.assertEqual(observed["inode"], target.stat().st_ino)
            self.assertEqual(observed["uid"], target.stat().st_uid)
            self.assertEqual(observed["read_bytes"], target.stat().st_size)

    def test_missing_leaf_and_symlink_leaf_are_not_followed(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "original.py"
            source.write_bytes(b"source")
            absent = Path(root) / "absent.py"
            link = Path(root) / "link.py"
            link.symlink_to(source)
            with mock.patch.object(probe, "TOOLS", {"g2": str(absent), "contained": str(link)}):
                self.assertEqual(probe.nofollow_regular(str(absent))["reason"], "missing_leaf")
                observed = probe.nofollow_regular(str(link))
            self.assertEqual(observed["type"], "symlink")
            self.assertNotIn("sha256", observed)

    def test_symlink_ancestor_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            real = Path(root) / "real"
            real.mkdir()
            (real / "tool.py").write_bytes(b"source")
            link = Path(root) / "link"
            link.symlink_to(real, target_is_directory=True)
            target = link / "tool.py"
            with mock.patch.object(probe, "TOOLS", {"g2": str(target)}):
                observed = probe.nofollow_regular(str(target))
            self.assertIsNone(observed["exists"])
            self.assertEqual(observed["reason"], "unsafe_ancestor")

    def test_oversized_file_is_not_hashed(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "tool.py"
            target.write_bytes(b"too big")
            with mock.patch.object(probe, "TOOLS", {"g2": str(target)}), mock.patch.object(
                probe, "MAX_TOOL_BYTES", 3
            ):
                observed = probe.nofollow_regular(str(target))
            self.assertEqual(observed["hash_error"], "tool_exceeds_2m_bound")
            self.assertNotIn("sha256", observed)


class CollectorTests(unittest.TestCase):
    def test_disabled_capture_refuses_before_ssh_or_marker(self):
        with mock.patch.object(collector.subprocess, "run", side_effect=AssertionError("SSH touched")):
            with self.assertRaisesRegex(RuntimeError, "enablement/assignment absent"):
                collector.capture()

    def test_exact_planning_witness_then_missing_tool_hold(self):
        snapshot, gitdir, tools = clean_planning_fixture()
        first = collector.planning_decision(request, snapshot, gitdir, tools)
        self.assertEqual(first["status"], "READ_ONLY_FEASIBLE_AT_OBSERVATION_TIME_ONLY")
        tools["tools"]["g2"] = {"path": request["exact_tools"]["g2"]["path"],
                                "exists": False, "reason": "missing_leaf"}
        second = collector.planning_decision(request, snapshot, gitdir, tools)
        self.assertEqual(second["status"], "HOLD")
        self.assertIn("g2_exact_nofollow_tool_path_absent_or_mismatched", second["reasons"])

    def test_dirty_git_occupied_host_and_low_memory_hold(self):
        snapshot, gitdir, tools = clean_planning_fixture()
        snapshot["checkouts"]["entries"]["e65_exact"]["git_status"]["clean"] = False
        snapshot["activity"]["processes"]["matched"] = [{"pid": 42}]
        tools["memory_available_bytes"] = 1
        result = collector.planning_decision(request, snapshot, gitdir, tools)
        self.assertEqual(result["status"], "HOLD")
        self.assertIn("e65_checkout_commit_cleanliness_or_source_hash_mismatch", result["reasons"])
        self.assertIn("stop_process_neuron_or_lock_occupancy", result["reasons"])
        self.assertIn("root_disk_inode_or_memory_below_planning_floor", result["reasons"])

    def test_sensitive_raw_stream_is_suppressed(self):
        observed = collector.redacted_stream(b"AWS_SECRET_ACCESS_KEY=test-only")
        self.assertTrue(observed["suppressed"])
        self.assertIsNone(observed["base64"])


if __name__ == "__main__":
    unittest.main()
