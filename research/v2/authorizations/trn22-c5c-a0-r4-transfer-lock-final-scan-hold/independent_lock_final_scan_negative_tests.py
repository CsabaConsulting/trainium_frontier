"""Verifier-only offline negatives for the disabled trn22 lock repair."""

import contextlib
import fcntl
import io
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_r4_remote_lock_hold as remote
import stage_r4_transport_lock_hold as transport


class IndependentLockFinalScanNegatives(unittest.TestCase):
    def test_observed_0775_lock_directory_is_rejected_by_remote(self):
        """Reproduce the 05:11 UTC host mode without touching the host."""
        with tempfile.TemporaryDirectory(prefix="trn22-verifier-lock-") as root:
            base = Path(root)
            cache = base / ".cache"
            lock_dir = cache / "trainium_freshstart"
            cache.mkdir()
            lock_dir.mkdir()
            lock = lock_dir / "device0.lock"
            lock.write_bytes(b"")
            cache.chmod(0o700)
            lock_dir.chmod(0o775)
            lock.chmod(0o664)
            parent_fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with mock.patch.object(remote, "UID", os.geteuid()):
                    with self.assertRaisesRegex(RuntimeError, "ancestry is unsafe"):
                        remote._open_existing_lock(parent_fd)
                self.assertFalse((base / "new-archive.tar.gz").exists())
                # The OS lock is usable; the candidate's own ancestry policy
                # rejects it before its probe can return a receipt.
                lock_fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(lock_fd)
            finally:
                os.close(parent_fd)

    def test_observed_0775_lock_directory_is_rejected_by_transport(self):
        device = 66305
        identity = {"ancestry": {"ubuntu": {"device": device}}}
        record = {
            "path": remote.HOST_LOCK,
            "cache_directory": {"type": "directory", "device": device,
                                "inode": 295736, "uid": 1000, "gid": 1000,
                                "mode": "0o700"},
            "lock_directory": {"type": "directory", "device": device,
                               "inode": 832588, "uid": 1000, "gid": 1000,
                               "mode": "0o775"},
            "file": {"exists": True, "type": "regular", "device": device,
                     "inode": 832589, "uid": 1000, "gid": 1000,
                     "mode": "0o664", "nlink": 1, "bytes": 0},
        }
        with self.assertRaisesRegex(RuntimeError, "writable cooperative-lock ancestor"):
            transport._validate_lock_identity(record, identity)

    def test_late_label_after_full_checks_now_blocks_creation(self):
        """Inject the old r3 timing counterexample after the held full check."""
        with tempfile.TemporaryDirectory(prefix="trn22-verifier-label-") as root:
            base = Path(root)
            cache = base / ".cache"
            lock_dir = cache / "trainium_freshstart"
            cache.mkdir(mode=0o700)
            lock_dir.mkdir(mode=0o700)
            (lock_dir / "device0.lock").write_bytes(b"")
            parent_fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
            identity = {"root_free_bytes": remote.PRE_FREE + 1,
                        "root_free_inodes": remote.MIN_INODES + 1}
            state = {"full_checks": 0, "late_label": False}

            def full_checks(*, owned_lock=None):
                state["full_checks"] += 1
                if owned_lock is not None:
                    state["late_label"] = True
                return {}

            def last_label_scan():
                if state["late_label"]:
                    raise RuntimeError("science label appeared after held full checks")
                return {"completed_monotonic_ns": time.monotonic_ns()}

            with contextlib.ExitStack() as stack:
                for patch in (
                    mock.patch.object(remote, "UID", os.geteuid()),
                    mock.patch.object(remote, "TARGET_NAME", "new-archive.tar.gz"),
                    mock.patch.object(remote, "_open_parent", return_value=(parent_fd, {})),
                    mock.patch.object(remote, "_identity", return_value=identity),
                    mock.patch.object(remote, "_stable"),
                    mock.patch.object(remote, "_require_preflight_binding", return_value="bound"),
                    mock.patch.object(remote, "_safe_path", return_value={"exists": False}),
                    mock.patch.object(remote, "_live_prewrite_safety", side_effect=full_checks),
                    mock.patch.object(remote, "_require_no_stops_or_processes", return_value={}),
                    mock.patch.object(remote, "_final_label_observation", side_effect=last_label_scan),
                ):
                    stack.enter_context(patch)
                with self.assertRaisesRegex(RuntimeError, "science label appeared"):
                    remote.write(io.BytesIO(b"never streamed"), {})
            self.assertEqual(state["full_checks"], 2)
            self.assertFalse((base / "new-archive.tar.gz").exists())


if __name__ == "__main__":
    unittest.main()
