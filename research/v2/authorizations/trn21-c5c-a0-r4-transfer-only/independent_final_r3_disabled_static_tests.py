"""Independent local checks of the disabled final-r3 transfer package.

All filesystem fixtures are disposable and local. No SSH or host operation is
performed, and this module does not create a review or assignment file.
"""

import copy
import fcntl
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_final_r3_once as operator
import stage_r4_remote_final_r3 as remote
import stage_r4_transport_final_r3 as transport
import test_operator_transfer_final_r3_once as fixtures


def _local_lock(root):
    cache = root / ".cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    host = cache / "trainium_freshstart"
    host.mkdir(mode=0o700)
    host.chmod(0o700)
    lock = host / "device0.lock"
    lock.write_bytes(b"unchanged-local-lock")
    lock.chmod(0o600)
    return lock


class IndependentFinalR3DisabledStaticTests(unittest.TestCase):
    def test_disabled_request_stops_before_authority_marker_or_ssh(self):
        request, binding = operator._read_frozen_request_and_code()
        self.assertIs(request["dispatch_authorized"], False)
        self.assertIs(request["source_transfer_authorized"], False)
        self.assertEqual(binding["request_sha256"], operator.REQUEST_SHA256)
        with (mock.patch.object(operator, "_read_required",
                                side_effect=AssertionError("authority read reached")),
              mock.patch.object(operator, "_consume_fixed_marker",
                                side_effect=AssertionError("marker reached")),
              mock.patch.object(operator, "_run_ssh_final",
                                side_effect=AssertionError("SSH reached")),
              mock.patch.object(operator.subprocess, "Popen",
                                side_effect=AssertionError("process reached"))):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()

    def test_existing_lock_open_is_read_only_and_compatible_with_launcher_flock(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            lock = _local_lock(root)
            before = (lock.read_bytes(), lock.stat().st_size,
                      lock.stat().st_mtime_ns, stat.S_IMODE(lock.stat().st_mode))
            parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with mock.patch.object(remote, "LOCK_PATH", str(lock)):
                    fd, identity = remote._open_existing_host_lock(parent_fd)
                    try:
                        self.assertEqual(identity["file"]["inode"], lock.stat().st_ino)
                        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                        self.assertEqual(flags & os.O_ACCMODE, os.O_RDONLY)
                        self.assertTrue(flags & os.O_NONBLOCK)
                        self.assertTrue(fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC)
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        launcher_fd = os.open(lock, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
                        try:
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(launcher_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        finally:
                            os.close(launcher_fd)
                    finally:
                        os.close(fd)
            finally:
                os.close(parent_fd)
            after = (lock.read_bytes(), lock.stat().st_size,
                     lock.stat().st_mtime_ns, stat.S_IMODE(lock.stat().st_mode))
            self.assertEqual(before, after)

    def test_untrusted_or_replaced_lock_holds_without_modification(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            lock = _local_lock(root)
            parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with mock.patch.object(remote, "LOCK_PATH", str(lock)):
                    lock.parent.chmod(0o775)
                    with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                        remote._open_existing_host_lock(parent_fd)
                    lock.parent.chmod(0o700)
                    content = lock.read_bytes()
                    lock.unlink()
                    lock.symlink_to(root / "alternate")
                    with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                        remote._open_existing_host_lock(parent_fd)
                    self.assertFalse((root / "alternate").exists())
                    lock.unlink()
                    lock.write_bytes(content)
                    lock.chmod(0o600)
                    self.assertEqual(lock.read_bytes(), content)
            finally:
                os.close(parent_fd)

    def test_receipt_rejects_independently_chosen_label_and_lock_drift(self):
        pre, write, verify, post = fixtures.clean_receipts()
        self.assertEqual(transport.compare_preflight_to_write(pre, write)["status"],
                         "PREFLIGHT_TO_FINAL_LOCKED_LABEL_GATE_PASS")
        persistence = transport.compare_sessions(pre, write, verify)
        self.assertEqual(transport.compare_postflight(pre, post,
                         persistence=persistence)["status"], "READ_ONLY_POSTFLIGHT_PASS")
        changed = copy.deepcopy(write)
        changed["final_creation_gate"]["host_lock"]["access"] = "O_RDWR"
        with self.assertRaises(RuntimeError):
            transport.compare_preflight_to_write(pre, changed)
        changed = copy.deepcopy(verify)
        changed["host_lock_fuser"]["pids"] = [12345]
        with self.assertRaises(RuntimeError):
            transport.compare_sessions(pre, write, changed)
        changed = copy.deepcopy(post)
        changed["host_lock_identity"]["host_directory"]["mode"] = "0o775"
        with self.assertRaises(RuntimeError):
            transport.compare_postflight(pre, changed, persistence=persistence)

    def test_file_backed_git_is_classified_without_following(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            checkout = root / "old-checkout"
            checkout.mkdir()
            git_file = checkout / ".git"
            git_file.write_text("gitdir: elsewhere\n", encoding="utf-8")
            self.assertEqual(remote._safe_path(str(git_file))["type"], "regular")
            with self.assertRaisesRegex(RuntimeError, "non-directory ancestor"):
                remote._safe_path(str(git_file / "HEAD"))

    def test_simulated_failed_write_still_gets_distinct_full_postflight(self):
        pre = fixtures.decorated_probe(pid=101, offset=-5)
        post = fixtures.decorated_probe(pid=301, offset=1)
        calls = []
        events = []

        class LocalJournal:
            def emit(self, event, **_details):
                events.append(event)

            def close(self):
                events.append("JOURNAL_CLOSED")

        def local_phase(operation, _journal, *, preflight=None):
            calls.append(operation)
            if len(calls) == 1:
                return pre
            if len(calls) == 2:
                self.assertIs(preflight, pre)
                raise RuntimeError("synthetic local write failure")
            return post

        with (mock.patch.object(operator, "authorize_final", return_value={"local": True}),
              mock.patch.object(operator.final_transport, "checked_local_source",
                                return_value={}),
              mock.patch.object(operator.final_transport, "checked_ssh_identity",
                                return_value={}),
              mock.patch.object(operator.core, "_check_controller_breakers",
                                return_value={}),
              mock.patch.object(operator, "_consume_fixed_marker",
                                return_value="0" * 64),
              mock.patch.object(operator, "_pinned_marker"),
              mock.patch.object(operator.core, "Journal", return_value=LocalJournal()),
              mock.patch.object(operator, "_run_ssh_final", side_effect=local_phase),
              mock.patch.object(operator.subprocess, "Popen",
                                side_effect=AssertionError("process reached"))):
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                operator.run_once()
        self.assertEqual(calls, ["probe", "write", "probe"])
        self.assertIn("FINAL_READ_ONLY_POSTFLIGHT", events)
        self.assertNotIn("TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW", events)


if __name__ == "__main__":
    unittest.main()
