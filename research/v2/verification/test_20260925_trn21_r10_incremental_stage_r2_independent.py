"""Independent offline trn21 r10 r2 boundary witnesses; no host contact."""

from contextlib import ExitStack
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock


PACKAGE = (Path(__file__).resolve().parents[1] / "authorizations" /
           "trn21-c5c-a0-r10-incremental-stage-r2-hold")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remote = load("independent_trn21_r2_remote", PACKAGE / "remote_incremental.py")
operator = load("independent_trn21_r2_operator", PACKAGE / "operator_once.py")


class FinalBoundary(unittest.TestCase):
    def witness(self, event, *, after_open=False):
        with tempfile.TemporaryDirectory(prefix="trn21-r10-r2-verifier-") as temp:
            root = Path(temp)
            bundle = root / "bundle"
            target = root / "checkout"
            stop = root / "STOP"
            label = root / ("late-" + remote.LABEL)
            payload = b"independent test bundle bytes"
            state = {"busy": False, "disk_low": False, "exclusive_open": False}
            real_gate = remote._final_no_write_continuity
            real_open = os.open

            def inject():
                if event == "stop":
                    stop.touch()
                elif event == "label":
                    label.touch()
                elif event == "activity":
                    state["busy"] = True
                elif event == "disk":
                    state["disk_low"] = True
                elif event == "age":
                    time.sleep(1.05)

            def gate():
                result = real_gate()
                if not after_open:
                    inject()
                return result

            def open_file(path, flags, *args, **kwargs):
                result = real_open(path, flags, *args, **kwargs)
                if path == bundle.name and flags & os.O_EXCL:
                    state["exclusive_open"] = True
                    if after_open:
                        inject()
                return result

            def activity():
                if stop.exists() or state["busy"]:
                    raise RuntimeError("independent STOP or activity")

            def scan():
                if label.exists():
                    raise RuntimeError("independent label collision")

            def space(_minimum):
                if state["disk_low"]:
                    raise RuntimeError("independent disk floor")
                return {"root_free_bytes": remote.PRE_FREE + 4096,
                        "root_free_inodes": remote.MIN_INODES + 2}

            with ExitStack() as stack:
                for key, value in (("ROOT", root), ("BUNDLE", bundle),
                                   ("TARGET", target), ("STOP_PATHS", [stop]),
                                   ("BUNDLE_BYTES", len(payload)),
                                   ("BUNDLE_SHA256", hashlib.sha256(payload).hexdigest())):
                    stack.enter_context(mock.patch.object(remote, key, value))
                stack.enter_context(mock.patch.object(remote, "_activity", side_effect=activity))
                stack.enter_context(mock.patch.object(remote, "_label_scan", side_effect=scan))
                stack.enter_context(mock.patch.object(remote, "_space", side_effect=space))
                stack.enter_context(mock.patch.object(
                    remote, "_final_no_write_continuity", side_effect=gate))
                stack.enter_context(mock.patch.object(remote, "_checked_bundle", return_value={}))
                stack.enter_context(mock.patch.object(remote.os, "open", side_effect=open_file))
                stack.enter_context(mock.patch.object(
                    remote.os, "unlink", side_effect=AssertionError("cleanup forbidden")))
                with self.assertRaises(RuntimeError) as caught:
                    remote._write_bundle(payload)
                self.assertTrue(state["exclusive_open"])
                self.assertTrue(bundle.is_file())
                self.assertEqual(bundle.read_bytes(), b"")
                self.assertFalse(target.exists())
            return str(caught.exception)

    def test_prior_hold_late_stop_label_activity_disk_and_age_now_refuse(self):
        for event in ("stop", "label", "activity", "disk", "age"):
            with self.subTest(event=event):
                self.assertTrue(self.witness(event))

    def test_post_open_stop_label_activity_disk_refuse_and_retain_empty(self):
        for event in ("stop", "label", "activity", "disk"):
            with self.subTest(event=event):
                self.assertTrue(self.witness(event, after_open=True))

    def test_interrupted_payload_retains_nonempty_partial_without_checkout(self):
        with tempfile.TemporaryDirectory(prefix="trn21-r10-r2-partial-") as temp:
            root = Path(temp)
            bundle = root / "bundle"
            target = root / "checkout"
            payload = b"independent interrupted payload"
            actual_write = os.write
            writes = 0
            now = time.monotonic()
            floor = {"root_free_bytes": remote.PRE_FREE + 4096,
                     "root_free_inodes": remote.MIN_INODES + 2}

            def write(fd, data):
                nonlocal writes
                writes += 1
                if writes == 1:
                    return actual_write(fd, data[:4])
                raise OSError("independent simulated transport interruption")

            with ExitStack() as stack:
                for key, value in (("ROOT", root), ("BUNDLE", bundle),
                                   ("TARGET", target), ("STOP_PATHS", []),
                                   ("BUNDLE_BYTES", len(payload)),
                                   ("BUNDLE_SHA256", hashlib.sha256(payload).hexdigest())):
                    stack.enter_context(mock.patch.object(remote, key, value))
                stack.enter_context(mock.patch.object(remote, "_activity"))
                stack.enter_context(mock.patch.object(remote, "_label_scan"))
                stack.enter_context(mock.patch.object(remote, "_space", return_value=floor))
                stack.enter_context(mock.patch.object(remote, "_final_no_write_continuity",
                    return_value={"activity_at": now, "label_at": now,
                                  "space_at": now, "root_space_before_open": floor,
                                  "label_scan_duration_seconds": 0.0}))
                stack.enter_context(mock.patch.object(remote.os, "write", side_effect=write))
                stack.enter_context(mock.patch.object(
                    remote.os, "unlink", side_effect=AssertionError("cleanup forbidden")))
                with self.assertRaisesRegex(OSError, "transport interruption"):
                    remote._write_bundle(payload)
                self.assertEqual(writes, 2)
                self.assertEqual(bundle.read_bytes(), b"inde")
                self.assertFalse(target.exists())

    def test_noncooperating_label_after_last_scan_is_a_bounded_residual(self):
        with tempfile.TemporaryDirectory(prefix="trn21-r10-r2-residual-") as temp:
            root = Path(temp)
            bundle = root / "bundle"
            target = root / "checkout"
            label = root / ("late-" + remote.LABEL)
            payload = b"independent race witness"
            activity_calls = 0
            floor = {"root_free_bytes": remote.PRE_FREE + 4096,
                     "root_free_inodes": remote.MIN_INODES + 2}

            def activity():
                nonlocal activity_calls
                activity_calls += 1
                if activity_calls == 3:
                    label.touch()  # after third/last broad scan, before payload

            def scan():
                if label.exists():
                    raise RuntimeError("label collision")

            with ExitStack() as stack:
                for key, value in (("ROOT", root), ("BUNDLE", bundle),
                                   ("TARGET", target), ("STOP_PATHS", []),
                                   ("BUNDLE_BYTES", len(payload)),
                                   ("BUNDLE_SHA256", hashlib.sha256(payload).hexdigest())):
                    stack.enter_context(mock.patch.object(remote, key, value))
                stack.enter_context(mock.patch.object(remote, "_activity", side_effect=activity))
                stack.enter_context(mock.patch.object(remote, "_label_scan", side_effect=scan))
                stack.enter_context(mock.patch.object(remote, "_space", return_value=floor))
                stack.enter_context(mock.patch.object(remote, "_checked_bundle", return_value={}))
                receipt = remote._write_bundle(payload)
                self.assertEqual(activity_calls, 3)
                self.assertTrue(receipt["post_open_label_and_activity_check"])
                self.assertTrue(label.exists())
                self.assertEqual(bundle.read_bytes(), payload)
                self.assertTrue(target.is_dir())

    def test_local_stop_blocks_postflight_contact_not_truthful_completion(self):
        with mock.patch.object(operator, "_local_stop_and_breakers",
                               side_effect=RuntimeError("local STOP")), \
             mock.patch.object(operator, "_ssh_identity") as identity, \
             mock.patch.object(operator.subprocess, "run") as contact:
            with self.assertRaisesRegex(RuntimeError, "local STOP"):
                operator._ssh_phase("postflight", None, b"", b"remote")
            identity.assert_not_called()
            contact.assert_not_called()


if __name__ == "__main__":
    unittest.main()
