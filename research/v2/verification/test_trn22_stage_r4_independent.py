"""Independent offline r4 source-stage boundary witnesses; no host contact."""

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
           "trn22-c5c-a0-r9-incremental-stage-r4-hold")
SPEC = importlib.util.spec_from_file_location(
    "independent_trn22_r4", PACKAGE / "remote_incremental.py")
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)


class NoCleanupBoundary(unittest.TestCase):
    def exercise(self, failure):
        with tempfile.TemporaryDirectory(prefix="trn22-r4-independent-") as dirname:
            root = Path(dirname)
            bundle = root / "bundle"
            target = root / "checkout"
            payload = b"independent fake incremental bundle"
            original_write = os.write
            original_fsync = os.fsync
            calls = {"write": 0, "fsync": 0}
            floor = {"root_free_bytes": remote.PRE_FREE + 1024,
                     "root_free_inodes": remote.MIN_INODES + 1}

            def samples():
                now = time.monotonic()
                return {"activity_at": now, "label_at": now,
                        "space_at": now, "root_space_before_open": floor,
                        "label_scan_duration_seconds": 0.0}

            def label_scan():
                if failure == "post_open_label":
                    raise RuntimeError("independent late label")

            def write(fd, data):
                calls["write"] += 1
                if failure == "partial_write":
                    if calls["write"] == 1:
                        return original_write(fd, data[:5])
                    raise OSError("independent write interruption")
                return original_write(fd, data)

            def fsync(fd):
                calls["fsync"] += 1
                if failure == "file_fsync" and calls["fsync"] == 1:
                    raise OSError("independent durability failure")
                return original_fsync(fd)

            with ExitStack() as stack:
                for name, value in (("ROOT", root), ("BUNDLE", bundle),
                                    ("TARGET", target), ("STOP_PATHS", []),
                                    ("BUNDLE_BYTES", len(payload)),
                                    ("BUNDLE_SHA256", hashlib.sha256(payload).hexdigest())):
                    stack.enter_context(mock.patch.object(remote, name, value))
                stack.enter_context(mock.patch.object(
                    remote, "_final_no_write_continuity", side_effect=samples))
                stack.enter_context(mock.patch.object(
                    remote, "_label_scan", side_effect=label_scan))
                stack.enter_context(mock.patch.object(remote, "_activity"))
                stack.enter_context(mock.patch.object(
                    remote, "_space", return_value=floor))
                stack.enter_context(mock.patch.object(
                    remote.os, "unlink", side_effect=AssertionError("forbidden cleanup")))
                stack.enter_context(mock.patch.object(remote.os, "write", side_effect=write))
                stack.enter_context(mock.patch.object(remote.os, "fsync", side_effect=fsync))
                with self.assertRaises((RuntimeError, OSError)):
                    remote._write_bundle(payload)
                self.assertTrue(bundle.is_file())
                self.assertFalse(target.exists())
                content = bundle.read_bytes()
            return content, calls

    def test_post_open_label_retains_empty_bundle(self):
        content, calls = self.exercise("post_open_label")
        self.assertEqual(content, b"")
        self.assertEqual(calls["write"], 0)
        self.assertEqual(calls["fsync"], 2)

    def test_interrupted_payload_retains_nonempty_partial(self):
        content, calls = self.exercise("partial_write")
        self.assertEqual(content, b"indep")
        self.assertEqual(calls["write"], 2)
        self.assertEqual(calls["fsync"], 2)

    def test_failed_file_fsync_retains_payload_and_no_checkout(self):
        content, calls = self.exercise("file_fsync")
        self.assertEqual(content, b"independent fake incremental bundle")
        self.assertEqual(calls["write"], 1)
        self.assertEqual(calls["fsync"], 2)


if __name__ == "__main__":
    unittest.main()
