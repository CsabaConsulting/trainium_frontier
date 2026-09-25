"""Verifier-owned offline negatives for the exact disabled trn22 successor.

No test opens SSH or addresses a host. The FIFO child has a hard local timeout.
"""

import contextlib
import copy
import errno
import fcntl
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import stage_r4_remote_lock_hold as remote
import stage_r4_transport_lock_hold as transport
import test_lock_final_scan_hold as supplied


class LocalLock:
    def __enter__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trn22-verifier-lock-")
        self.root = Path(self.temporary.name)
        self.cache = self.root / ".cache"
        self.inner = self.cache / "trainium_freshstart"
        self.cache.mkdir(mode=0o700)
        self.inner.mkdir(mode=0o775)
        self.inner.chmod(0o775)
        self.lock = self.inner / "device0.lock"
        self.lock.touch(mode=0o664)
        self.lock.chmod(0o664)
        self.parent_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        return self

    def __exit__(self, *_):
        os.close(self.parent_fd)
        self.temporary.cleanup()


class IndependentDisabledStaticReview(unittest.TestCase):
    def test_exact_archive_and_frozen_cooperative_launcher_bytes(self):
        source = transport.checked_local_source()
        self.assertEqual(source["bytes"], 104154875)
        self.assertEqual(source["sha256"], remote.SHA256)
        expected = {
            "research/fresh_start_launch.py":
                "9488efa918119439a33cbe68cb5638bd1af3b15f1f8f61ec790a216e2bc1e3f4",
            "research/fresh_start_host.py":
                "b34c368c12175615514e63daebc296ec09332ef801e176e0d19cc3edc045fed3",
            "research/fresh_start_guard.py":
                "df902becb2faf009d04d7ecfaee3791df67e822e90ec6112adbc39be81599ecc",
        }
        for path, digest in expected.items():
            with self.subTest(path=path):
                raw = subprocess.check_output(
                    ["git", "show", remote.SOURCE_COMMIT + ":" + path], cwd=ROOT)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)

    def test_observed_private_modes_accept_and_acl_uncertainty_holds(self):
        with LocalLock() as fixture, mock.patch.object(remote, "UID", os.geteuid()), \
                mock.patch.object(remote, "HOST_LOCK", str(fixture.lock)):
            parts, record = remote._open_existing_lock(fixture.parent_fd)
            try:
                self.assertEqual([record[key]["mode"] for key in
                                  ("cache_directory", "lock_directory", "file")],
                                 ["0o700", "0o775", "0o664"])
            finally:
                remote._close_lock(parts)
            with mock.patch.object(remote.os, "getxattr",
                                   side_effect=OSError(errno.EACCES, "uncertain")):
                with self.assertRaisesRegex(RuntimeError, "cannot authenticate"):
                    remote._open_existing_lock(fixture.parent_fd)
            fixture.cache.chmod(0o750)
            with self.assertRaisesRegex(RuntimeError, "ancestry"):
                remote._open_existing_lock(fixture.parent_fd)

    def test_named_lock_inode_replacement_is_detected(self):
        with LocalLock() as fixture, mock.patch.object(remote, "UID", os.geteuid()), \
                mock.patch.object(remote, "HOST_LOCK", str(fixture.lock)):
            parts, _ = remote._open_existing_lock(fixture.parent_fd)
            try:
                fixture.lock.rename(fixture.inner / "device0.lock.old")
                fixture.lock.touch(mode=0o664)
                fixture.lock.chmod(0o664)
                with self.assertRaisesRegex(RuntimeError, "replaced or unsafe"):
                    remote._lock_identity(fixture.parent_fd, parts)
            finally:
                remote._close_lock(parts)

    def test_preexisting_fifo_lock_has_bounded_blocking_witness(self):
        # A pre-existing nonregular lock must be classifiable without a hang.
        # The current candidate blocks in open(2) before its type check.
        with LocalLock() as fixture:
            fixture.lock.unlink()
            os.mkfifo(fixture.lock, 0o664)
            child = (
                "import os,sys\n"
                "sys.path.insert(0,sys.argv[1])\n"
                "import stage_r4_remote_lock_hold as remote\n"
                "fd=os.open(sys.argv[2],os.O_RDONLY|os.O_DIRECTORY)\n"
                "try: remote._open_existing_lock(fd)\n"
                "finally: os.close(fd)\n"
            )
            with self.assertRaises(subprocess.TimeoutExpired):
                subprocess.run([sys.executable, "-B", "-c", child,
                                str(HERE), str(fixture.root)], capture_output=True,
                               check=False, timeout=0.5)

    def test_preexisting_fifo_checkpoint_is_classified_before_open(self):
        with tempfile.TemporaryDirectory(prefix="trn22-verifier-checkpoint-") as root:
            fifo = Path(root) / "checkpoint.fifo"
            os.mkfifo(fifo, 0o600)
            with mock.patch.object(remote, "CHECKPOINT", str(fifo)), \
                    mock.patch.object(remote, "_safe_open_regular",
                                      side_effect=AssertionError("must not open FIFO")):
                with self.assertRaisesRegex(RuntimeError, "checkpoint absent"):
                    remote._protected()

    def test_receipts_reject_lock_label_stop_process_and_neuron_mutations(self):
        pre, write, verify = supplied.compared_receipts()
        self.assertEqual(transport.compare_sessions(write, verify, preflight=pre)["status"],
                         "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        mutations = {
            "lock inode": lambda e: e["cooperative_lock_after_write"]["file"].update(
                inode=e["cooperative_lock_after_write"]["file"]["inode"] + 1),
            "label after open": lambda e: e.update(
                target_open_attempt_monotonic_ns=1_000_000_005),
            "late label": lambda e: e["final_label_observation"]["scan"].update(
                match_count=1),
            "STOP": lambda e: next(iter(e["last_stop_process_check"]["stops"].values())).update(
                exists=True),
            "process": lambda e: e["last_stop_process_check"]["processes"].update(
                matched=[{"pid": 42}]),
            "Neuron": lambda e: e["last_live_precreate_safety"]["neuron_ls"].update(
                non_diagnostic_pids=[42]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                altered = copy.deepcopy(write)
                mutate(altered["prewrite_evidence"])
                with self.assertRaises(RuntimeError):
                    transport.compare_sessions(altered, verify, preflight=pre)

    def test_postflight_rejects_boot_checkpoint_cache_disk_and_target_drift(self):
        pre, post = supplied.fresh_probe(), supplied.fresh_probe()
        self.assertEqual(transport.compare_postflight(pre, post)["status"],
                         "READ_ONLY_POSTFLIGHT_PASS")
        changes = {
            "boot": lambda p: p["identity_before"].update(boot_id="other"),
            "checkpoint": lambda p: p["protected"]["checkpoint"].update(
                sha256="0" * 64),
            "cache": lambda p: p["protected"]["six_caches"]["neff"].update(
                inode=123456),
            "disk": lambda p: p["identity_before"].update(root_free_bytes=1),
            "target": lambda p: p["targets"][remote.TARGET].update(
                exists=True, type="symlink"),
        }
        for name, mutate in changes.items():
            with self.subTest(name=name):
                altered = copy.deepcopy(post)
                mutate(altered)
                with self.assertRaises(RuntimeError):
                    transport.compare_postflight(pre, altered)

    def test_exclusive_target_collision_keeps_existing_bytes(self):
        with supplied.LockFixture() as fixture:
            target = fixture.root / "new-archive.tar.gz"

            def collision():
                target.write_bytes(b"existing")
                return {"completed_monotonic_ns": time.monotonic_ns()}

            patches = supplied.LockFinalScanTests()._writer_fixture(
                fixture, final_label=collision)
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with self.assertRaises(FileExistsError):
                    remote.write(io.BytesIO(b"payload"), {})
            self.assertEqual(target.read_bytes(), b"existing")

    def test_disabled_entrypoints_reject_before_any_source_or_key_check(self):
        with mock.patch.object(transport, "checked_ssh_identity",
                               side_effect=AssertionError("key metadata touched")), \
                mock.patch.object(transport, "checked_local_source",
                                  side_effect=AssertionError("source touched")):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                transport.build_ssh_argv("write", preflight={})
        result = subprocess.run([sys.executable, "-B", str(HERE /
                                 "stage_r4_remote_lock_hold.py"), "write"],
                                capture_output=True, text=True, timeout=5, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DISPATCH HOLD", result.stderr)


if __name__ == "__main__":
    unittest.main()
