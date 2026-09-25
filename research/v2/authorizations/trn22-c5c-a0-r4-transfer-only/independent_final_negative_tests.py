"""Independent offline regression checks for the disabled d682a90b repair.

The cases retain the two failures from the b80c69c4 review, using the new
preflight binding and write receipt API. They never contact a host or read a
private key. The remote writer runs only against temporary controller files.
"""

import copy
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_final_once as final
import stage_r4_remote as remote
import stage_r4_transport as transport
import test_stage_r4_transfer as fixtures


class IndependentFinalRepairTests(unittest.TestCase):
    def test_boot_a_preflight_boot_b_write_boot_a_postflight_is_rejected(self):
        pre = fixtures.probe()
        write = fixtures.receipt("write", pid=1001, preflight=pre)
        verify = fixtures.receipt("verify", pid=1002, utc_seconds=1)
        post = fixtures.probe(archive=write["target_stat"])
        for receipt in (write, verify):
            for side in ("identity_before", "identity_after"):
                receipt[side]["boot_id"] = "different-boot"
        # The old code accepted these four receipts. The revised composite
        # comparator must join the preflight to the writer before success.
        self.assertEqual(transport.compare_postflight(pre, post)["status"],
                         "READ_ONLY_POSTFLIGHT_PASS")
        with self.assertRaisesRegex(RuntimeError, "preflight-to-write.*boot_id"):
            transport.compare_sessions(write, verify, preflight=pre)

        binding = transport.write_binding(pre)
        changed = copy.deepcopy(pre["identity_after"])
        changed["boot_id"] = "different-boot"
        with tempfile.TemporaryDirectory() as directory:
            parent_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with (mock.patch.object(remote, "_open_parent",
                                        side_effect=lambda: (os.dup(parent_fd), changed["ancestry"])),
                      mock.patch.object(remote, "_identity", return_value=changed),
                      mock.patch.object(remote, "_process_identity",
                                        return_value={"pid": 456, "start_ticks": 789})):
                    with self.assertRaisesRegex(RuntimeError, "preflight-to-write.*identity"):
                        remote.write(io.BytesIO(b""), binding)
                self.assertFalse((Path(directory) / remote.TARGET_NAME).exists())
            finally:
                os.close(parent_fd)

    def test_host_stop_after_probe_blocks_target_before_exclusive_create(self):
        pre = fixtures.probe()
        binding = transport.write_binding(pre)
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "STOP"
            stop.write_text("STOP", encoding="ascii")
            parent_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with (mock.patch.object(remote, "STOP_PATHS", [str(stop)]),
                      mock.patch.object(remote, "_open_parent",
                                        side_effect=lambda: (os.dup(parent_fd),
                                                             pre["identity_after"]["ancestry"])),
                      mock.patch.object(remote, "_identity",
                                        return_value=pre["identity_after"]),
                      mock.patch.object(remote, "_process_identity",
                                        return_value={"pid": 456, "start_ticks": 789}),
                      mock.patch.object(remote, "_safe_path",
                                        side_effect=lambda path: {"exists": os.path.lexists(path)})):
                    with self.assertRaisesRegex(RuntimeError, "live host STOP"):
                        remote.write(io.BytesIO(b""), binding)
                self.assertFalse((Path(directory) / remote.TARGET_NAME).exists())
            finally:
                os.close(parent_fd)

    def test_busy_process_after_probe_blocks_target_before_exclusive_create(self):
        pre = fixtures.probe()
        binding = transport.write_binding(pre)
        with tempfile.TemporaryDirectory() as directory:
            parent_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with (mock.patch.object(remote, "_open_parent",
                                        side_effect=lambda: (os.dup(parent_fd),
                                                             pre["identity_after"]["ancestry"])),
                      mock.patch.object(remote, "_identity",
                                        return_value=pre["identity_after"]),
                      mock.patch.object(remote, "_process_identity",
                                        return_value={"pid": 456, "start_ticks": 789}),
                      mock.patch.object(remote, "_safe_path", return_value={"exists": False}),
                      mock.patch.object(remote, "_processes", return_value={
                          "matched": [{"pid": 777, "markers": ["train.py"]}],
                          "unreadable_cmdlines": 0})):
                    with self.assertRaisesRegex(RuntimeError, "live host process busy"):
                        remote.write(io.BytesIO(b""), binding)
                self.assertFalse((Path(directory) / remote.TARGET_NAME).exists())
            finally:
                os.close(parent_fd)

    def test_stop_appearing_during_live_checks_is_caught_at_final_gate(self):
        pre = fixtures.probe()
        binding = transport.write_binding(pre)
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "STOP"
            parent_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                def late_stop():
                    stop.write_text("STOP", encoding="ascii")
                    return {}

                def safe_path(path):
                    return {"exists": path == str(stop) and stop.exists()}

                with (mock.patch.object(remote, "STOP_PATHS", [str(stop)]),
                      mock.patch.object(remote, "_open_parent",
                                        side_effect=lambda: (os.dup(parent_fd),
                                                             pre["identity_after"]["ancestry"])),
                      mock.patch.object(remote, "_identity",
                                        return_value=pre["identity_after"]),
                      mock.patch.object(remote, "_process_identity",
                                        return_value={"pid": 456, "start_ticks": 789}),
                      mock.patch.object(remote, "_safe_path", side_effect=safe_path),
                      mock.patch.object(remote, "_live_prewrite_safety",
                                        side_effect=late_stop)):
                    with self.assertRaisesRegex(RuntimeError, "live host STOP"):
                        remote.write(io.BytesIO(b""), binding)
                self.assertTrue(stop.exists())
                self.assertFalse((Path(directory) / remote.TARGET_NAME).exists())
            finally:
                os.close(parent_fd)

    def test_tampered_binding_and_reported_live_safety_are_rejected(self):
        pre = fixtures.probe()
        binding = transport.write_binding(pre)
        wrong_target = copy.deepcopy(binding)
        wrong_target["target"] += ".other"
        with mock.patch.object(remote, "_process_identity",
                               return_value={"pid": 456, "start_ticks": 789}):
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                remote._require_preflight_binding(wrong_target,
                                                  pre["identity_after"])
        write = fixtures.receipt("write", pid=1001, preflight=pre)
        write["prewrite_evidence"]["preflight_receipt_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            transport.compare_preflight_write(pre, write)
        write = fixtures.receipt("write", pid=1001, preflight=pre)
        write["prewrite_evidence"]["last_stop_process_check"]["stops"][
            remote.STOP_PATHS[0]] = {"exists": True}
        with self.assertRaisesRegex(RuntimeError, "STOP/process"):
            transport.compare_preflight_write(pre, write)

    def test_exact_local_target_is_exclusive_and_partial_failure_is_preserved(self):
        pre = fixtures.probe()
        safety = {
            "stops": copy.deepcopy(pre["stops"]),
            "processes": copy.deepcopy(pre["processes"]),
            "checkpoint_fuser": copy.deepcopy(pre["checkpoint_fuser"]),
            "shared_lock_fuser": copy.deepcopy(pre["shared_lock_fuser"]),
            "neuron0_fuser": copy.deepcopy(pre["neuron0_fuser"]),
            "neuron_ls": copy.deepcopy(pre["neuron_ls"]),
            "science_label_scan": copy.deepcopy(pre["science_label_scan"]),
        }
        payload = b"small independent local-only fixture"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / remote.TARGET_NAME
            parent_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                digest = hashlib.sha256(payload).hexdigest()
                with (mock.patch.object(remote, "BYTES", len(payload)),
                      mock.patch.object(remote, "SHA256", digest),
                      mock.patch.object(transport, "SOURCE_SHA256", digest),
                      mock.patch.object(remote, "_open_parent",
                                        side_effect=lambda: (os.dup(parent_fd),
                                                             pre["identity_after"]["ancestry"])),
                      mock.patch.object(remote, "_identity",
                                        return_value=pre["identity_after"]),
                      mock.patch.object(remote, "_process_identity",
                                        return_value={"pid": 456, "start_ticks": 789}),
                      mock.patch.object(remote, "_safe_path", return_value={"exists": False}),
                      mock.patch.object(remote, "_processes", return_value={
                          "matched": [], "unreadable_cmdlines": 0}),
                      mock.patch.object(remote, "_live_prewrite_safety", return_value=safety)):
                    binding = transport.write_binding(pre)
                    result = remote.write(io.BytesIO(payload), binding)
                    self.assertEqual(result["readback_sha256"], digest)
                    self.assertEqual(transport.compare_preflight_write(pre, result)[
                        "status"], "PREFLIGHT_TO_WRITE_IDENTITY_AND_LIVE_SAFETY_PASS")
                    before = target.read_bytes()
                    with self.assertRaises(FileExistsError):
                        remote.write(io.BytesIO(payload), binding)
                    self.assertEqual(target.read_bytes(), before)
                    self.assertEqual(target.stat().st_mode & 0o777, 0o600)
            finally:
                os.close(parent_fd)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / remote.TARGET_NAME
            parent_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with (mock.patch.object(remote, "BYTES", len(payload) + 1),
                      mock.patch.object(remote, "SHA256", "0" * 64),
                      mock.patch.object(transport, "SOURCE_SHA256", "0" * 64),
                      mock.patch.object(remote, "_open_parent",
                                        side_effect=lambda: (os.dup(parent_fd),
                                                             pre["identity_after"]["ancestry"])),
                      mock.patch.object(remote, "_identity",
                                        return_value=pre["identity_after"]),
                      mock.patch.object(remote, "_process_identity",
                                        return_value={"pid": 456, "start_ticks": 789}),
                      mock.patch.object(remote, "_safe_path", return_value={"exists": False}),
                      mock.patch.object(remote, "_processes", return_value={
                          "matched": [], "unreadable_cmdlines": 0}),
                      mock.patch.object(remote, "_live_prewrite_safety", return_value=safety)):
                    binding = transport.write_binding(pre)
                    with self.assertRaisesRegex(RuntimeError, "partial target retained"):
                        remote.write(io.BytesIO(payload), binding)
                    self.assertEqual(target.read_bytes(), payload)
                    with self.assertRaises(FileExistsError):
                        remote.write(io.BytesIO(payload), binding)
                    self.assertEqual(target.read_bytes(), payload)
            finally:
                os.close(parent_fd)

    def test_disabled_package_and_failed_write_postflight(self):
        with (mock.patch.object(final, "_consume_fixed_marker",
                                side_effect=AssertionError("marker reached")),
              mock.patch.object(final.core, "_run_ssh",
                                side_effect=AssertionError("SSH reached"))):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                final.run_once()

        pre, post = fixtures.probe(), fixtures.probe()
        post["process_identity"] = {"pid": 999, "start_ticks": 9990}
        calls, events = [], []

        class Journal:
            def emit(self, event, **details):
                events.append(event)
            def close(self):
                events.append("CLOSED")

        def fake_ssh(operation, journal, **kwargs):
            calls.append(operation)
            if len(calls) == 1:
                return pre
            if operation == "write":
                self.assertIs(kwargs["preflight"], pre)
                raise RuntimeError("offline synthetic failed write")
            return post

        with (mock.patch.object(final, "authorize_final", return_value={"request_sha256": "a" * 64}),
              mock.patch.object(final.core.transport, "checked_local_source", return_value={}),
              mock.patch.object(final.core, "_check_controller_breakers", return_value={}),
              mock.patch.object(final, "_consume_fixed_marker", return_value="b" * 64),
              mock.patch.object(final.core, "_pinned", return_value=b"marker"),
              mock.patch.object(final.core, "Journal", Journal),
              mock.patch.object(final.core, "_run_ssh", side_effect=fake_ssh)):
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                final.run_once()
        self.assertEqual(calls, ["probe", "write", "probe"])
        self.assertIn("TRANSFER_HOLD", events)
        self.assertIn("FINAL_READ_ONLY_POSTFLIGHT", events)
        self.assertIn("CLOSED", events)


if __name__ == "__main__":
    unittest.main()
