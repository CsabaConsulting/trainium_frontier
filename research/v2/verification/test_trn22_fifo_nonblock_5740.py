"""Independent local negatives for the disabled trn22 FIFO-lock successor.

No test contacts a host. FIFO opens run in a child with a hard timeout.
"""

import contextlib
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PACKAGE = (Path(__file__).resolve().parents[1] / "authorizations" /
           "trn22-c5c-a0-r4-transfer-lock-nonblock-hold")
sys.path.insert(0, str(PACKAGE))
import stage_r4_remote_lock_hold as remote
import stage_r4_transport_lock_hold as transport
import test_lock_final_scan_hold as supplied


class IndependentFifoLockReview(unittest.TestCase):
    def test_fifo_and_actual_stat_to_open_swaps_promptly_hold(self):
        child = r'''
import os
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
import stage_r4_remote_lock_hold as remote

root = Path(sys.argv[2])
case = sys.argv[3]
cache = root / ".cache"
inner = cache / "trainium_freshstart"
cache.mkdir(mode=0o700)
inner.mkdir(mode=0o775)
inner.chmod(0o775)
lock = inner / "device0.lock"
if case == "already_fifo":
    os.mkfifo(lock, 0o664)
else:
    lock.write_bytes(b"")
    lock.chmod(0o664)
    regular = os.stat(lock, follow_symlinks=False)
    real_stat = os.stat
    first = True
    def replace_after_stat(path, *args, **kwargs):
        global first
        if path == remote.LOCK_NAME and first:
            first = False
            lock.unlink()
            if case == "swap_fifo":
                os.mkfifo(lock, 0o664)
            else:
                lock.symlink_to(root / "outside")
            return regular
        return real_stat(path, *args, **kwargs)
    os.stat = replace_after_stat

parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
try:
    try:
        parts, _ = remote._open_existing_lock(parent_fd)
    except (RuntimeError, OSError):
        print("PROMPT_HOLD")
    else:
        remote._close_lock(parts)
        raise AssertionError("unsafe lock accepted")
finally:
    os.close(parent_fd)
'''
        for case in ("already_fifo", "swap_fifo", "swap_symlink"):
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                    prefix="trn22-fifo-independent-") as root:
                result = subprocess.run(
                    [sys.executable, "-B", "-c", child, str(PACKAGE), root, case],
                    capture_output=True, text=True, timeout=1, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "PROMPT_HOLD")

    def test_readonly_nonblocking_descriptor_and_cross_process_flock(self):
        with supplied.LockFixture() as fixture:
            parent_fd = os.open(fixture.root, os.O_RDONLY | os.O_DIRECTORY)
            with contextlib.ExitStack() as stack:
                stack.callback(os.close, parent_fd)
                stack.enter_context(mock.patch.object(remote, "UID", os.geteuid()))
                parts, record = remote._open_existing_lock(parent_fd)
                stack.callback(remote._close_lock, parts)
                flags = fcntl.fcntl(parts[2], fcntl.F_GETFL)
                self.assertEqual(flags & os.O_ACCMODE, os.O_RDONLY)
                self.assertTrue(flags & os.O_NONBLOCK)
                self.assertEqual(record["file"]["type"], "regular")
                fcntl.flock(parts[2], fcntl.LOCK_EX | fcntl.LOCK_NB)
                child = (
                    "import fcntl,os,sys\n"
                    "fd=os.open(sys.argv[1],os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)\n"
                    "try:\n"
                    "  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
                    "except BlockingIOError: print('HELD')\n"
                    "else: raise AssertionError('second process acquired lock')\n"
                    "finally: os.close(fd)\n"
                )
                result = subprocess.run(
                    [sys.executable, "-B", "-c", child, str(fixture.lock)],
                    capture_output=True, text=True, timeout=1, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "HELD")

    def test_receipt_binding_rejects_lock_and_late_safety_drift(self):
        preflight, write, verify = supplied.compared_receipts()
        self.assertEqual(
            transport.compare_sessions(write, verify, preflight=preflight)["status"],
            "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        variants = []
        changed = copy.deepcopy(write)
        changed["prewrite_evidence"]["cooperative_lock_after_write"]["file"]["inode"] += 1
        variants.append(changed)
        changed = copy.deepcopy(write)
        changed["prewrite_evidence"]["target_open_completed_monotonic_ns"] += 1_000_000_001
        variants.append(changed)
        changed = copy.deepcopy(write)
        changed["prewrite_evidence"]["last_stop_process_check"]["processes"]["matched"] = [42]
        variants.append(changed)
        changed = copy.deepcopy(write)
        changed["prewrite_evidence"]["last_live_precreate_safety"]["neuron_ls"]["non_diagnostic_pids"] = [42]
        variants.append(changed)
        changed = copy.deepcopy(preflight)
        changed["host_lock_identity"]["cache_directory"]["acl"]["access"] = "present"
        with self.assertRaises(RuntimeError):
            transport.compare_sessions(write, verify, preflight=changed)
        for changed in variants:
            with self.subTest(variant=variants.index(changed)), self.assertRaises(RuntimeError):
                transport.compare_sessions(changed, verify, preflight=preflight)

    def test_optimized_python_keeps_both_entrypoints_disabled(self):
        remote_call = subprocess.run(
            [sys.executable, "-O", "-B", str(PACKAGE / "stage_r4_remote_lock_hold.py"),
             "write"], capture_output=True, text=True, timeout=2, check=False)
        self.assertNotEqual(remote_call.returncode, 0)
        self.assertIn("DISPATCH HOLD", remote_call.stderr)
        transport_call = subprocess.run(
            [sys.executable, "-O", "-B", "-c",
             "import sys; sys.path.insert(0,sys.argv[1]); "
             "import stage_r4_transport_lock_hold as t; t.build_ssh_argv('write')",
             str(PACKAGE)], capture_output=True, text=True, timeout=2, check=False)
        self.assertNotEqual(transport_call.returncode, 0)
        self.assertIn("DISPATCH HOLD", transport_call.stderr)


if __name__ == "__main__":
    unittest.main()
