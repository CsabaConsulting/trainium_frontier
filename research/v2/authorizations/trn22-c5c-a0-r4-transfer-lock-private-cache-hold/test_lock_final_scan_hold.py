"""Offline only: mocked remote writer and synthetic cooperative-lock fixtures."""

import copy
import contextlib
import datetime
import errno
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "trn22-c5c-a0-r4-transfer-only" / "test_stage_r4_transfer.py"
sys.path.insert(0, str(HERE))
import stage_r4_remote_lock_hold as remote
import stage_r4_transport_lock_hold as transport

spec = importlib.util.spec_from_file_location("historical_trn22_transfer_fixtures", OLD)
historical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(historical)


def lock_fixture():
    return {
        "path": remote.HOST_LOCK,
        "cache_directory": {"type": "directory", "device": 66305,
                            "inode": 800, "uid": 1000, "gid": 1000,
                            "mode": "0o700",
                            "acl": {"access": "ENODATA", "default": "ENODATA"}},
        "lock_directory": {"type": "directory", "device": 66305,
                           "inode": 801, "uid": 1000, "gid": 1000,
                           "mode": "0o775",
                           "acl": {"access": "ENODATA", "default": "ENODATA"}},
        "file": {"exists": True, "type": "regular", "device": 66305,
                 "inode": 832589, "uid": 1000, "gid": 1000,
                 "mode": "0o664", "nlink": 1, "bytes": 0,
                 "acl": {"access": "ENODATA"}},
    }


def fresh_probe(*, archive=None):
    probe = historical.probe(archive=archive)
    probe["schema"] = remote.SCHEMA
    probe["host_lock_identity"] = lock_fixture()
    return probe


def compared_receipts():
    old_pre = historical.probe()
    write = historical.receipt("write", pid=124, utc_seconds=3, preflight=old_pre)
    verify = historical.receipt("verify", pid=125, utc_seconds=4)
    preflight = old_pre
    preflight["schema"] = remote.SCHEMA
    preflight["host_lock_identity"] = lock_fixture()
    write["schema"] = remote.SCHEMA
    verify["schema"] = remote.SCHEMA
    binding = transport.write_binding(preflight)
    evidence = write["prewrite_evidence"]
    evidence["preflight_binding_sha256"] = transport.sha256(
        transport._canonical_bytes(binding))
    evidence["preflight_receipt_sha256"] = binding["preflight_receipt_sha256"]
    evidence["preflight_lock_identity"] = lock_fixture()
    evidence["cooperative_lock_after_create"] = lock_fixture()
    evidence["cooperative_lock_after_write"] = lock_fixture()
    evidence["last_live_precreate_safety"]["shared_lock_fuser"] = None
    evidence["last_live_precreate_safety"]["cooperative_lock"] = lock_fixture()
    evidence["cooperative_lock_acquired_monotonic_ns"] = 1000
    started = datetime.datetime.fromisoformat(preflight["utc"])
    evidence["final_label_observation"] = {
        "started_utc": (started + datetime.timedelta(seconds=1)).isoformat(),
        "completed_utc": (started + datetime.timedelta(seconds=2)).isoformat(),
        "started_monotonic_ns": 2000,
        "completed_monotonic_ns": 3000,
        "scan": {"rc": 0, "matches": [], "match_count": 0,
                 "truncated": False, "stderr_empty": True,
                 "stdout_sha256": hashlib.sha256(b"").hexdigest()},
    }
    evidence["target_open_attempt_monotonic_ns"] = 4000
    evidence["target_open_completed_monotonic_ns"] = 5000
    return preflight, write, verify


class LockFixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trn22-lock-repair-offline-")
        self.root = Path(self.temporary.name)
        self.cache = self.root / ".cache"
        self.lock_dir = self.cache / "trainium_freshstart"
        self.cache.mkdir(mode=0o700)
        self.lock_dir.mkdir(mode=0o775)
        self.lock_dir.chmod(0o775)
        self.lock = self.lock_dir / "device0.lock"
        self.lock.write_bytes(b"")
        self.lock.chmod(0o664)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.temporary.cleanup()


class LockFinalScanTests(unittest.TestCase):
    def test_contract_is_disabled_and_binds_exact_local_bytes(self):
        contract = json.loads((HERE / "GATE_CONTRACT.HOLD.json").read_text())
        self.assertIs(contract["dispatch_authorized"], False)
        self.assertIs(contract["transfer_authorized"], False)
        self.assertIs(contract["checkout_authorized"], False)
        for key, digest in contract["package_file_sha256"].items():
            self.assertEqual(hashlib.sha256((HERE / key).read_bytes()).hexdigest(), digest)
        runbook = (HERE / "RUNBOOK.md").read_text()
        for key in ("stage_r4_remote_lock_hold.py", "stage_r4_transport_lock_hold.py",
                    "test_lock_final_scan_hold.py"):
            digest = contract["package_file_sha256"][key]
            self.assertIn(f"| `{key}` | `{digest}` |", runbook)
        self.assertIsNone(contract["missing_authority"]["independent_static_review_sha256"])
        self.assertIsNone(contract["missing_authority"]["enabled_request_sha256"])
        self.assertIsNone(contract["missing_authority"]["supervisor_assignment_sha256"])

    def test_frozen_446b_launcher_uses_same_flock_and_passes_fd(self):
        root = HERE.parents[3]
        launch = subprocess.check_output([
            "git", "show", remote.SOURCE_COMMIT + ":research/fresh_start_launch.py"
        ], cwd=root, text=True)
        guard = subprocess.check_output([
            "git", "show", remote.SOURCE_COMMIT + ":research/fresh_start_guard.py"
        ], cwd=root, text=True)
        host = subprocess.check_output([
            "git", "show", remote.SOURCE_COMMIT + ":research/fresh_start_host.py"
        ], cwd=root, text=True)
        self.assertIn("HOST_LOCK = HOST_DIRECTORY / 'device0.lock'", host)
        self.assertIn("fcntl.flock(host_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)", launch)
        self.assertIn("pass_fds=(host_lock.fileno(),)", launch)
        self.assertIn("pass_fds=(args.lock_fd,)", guard)

    def test_existing_readonly_lock_is_exclusive_and_never_created(self):
        with LockFixture() as fixture, \
                mock.patch.object(remote, "UID", os.geteuid()), \
                mock.patch.object(remote, "HOST_LOCK", str(fixture.lock)):
            parent_fd = os.open(fixture.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                parts, record = remote._open_existing_lock(parent_fd)
                try:
                    self.assertEqual(record["file"]["bytes"], 0)
                    self.assertEqual(record["cache_directory"]["mode"], "0o700")
                    self.assertEqual(record["lock_directory"]["mode"], "0o775")
                    self.assertEqual(record["file"]["mode"], "0o664")
                    self.assertEqual(record["cache_directory"]["acl"],
                                     {"access": "ENODATA", "default": "ENODATA"})
                    fcntl.flock(parts[2], fcntl.LOCK_EX | fcntl.LOCK_NB)
                    rival_parts, _ = remote._open_existing_lock(parent_fd)
                    try:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(rival_parts[2], fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        remote._close_lock(rival_parts)
                    self.assertEqual(remote._lock_identity(parent_fd, parts), record)
                finally:
                    remote._close_lock(parts)
                fixture.lock.unlink()
                with self.assertRaises(FileNotFoundError):
                    remote._open_existing_lock(parent_fd)
                self.assertFalse(fixture.lock.exists())
            finally:
                os.close(parent_fd)

    def test_symlink_and_writable_lock_ancestor_hold(self):
        with LockFixture() as fixture, mock.patch.object(remote, "UID", os.geteuid()):
            parent_fd = os.open(fixture.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                fixture.cache.chmod(0o777)
                with self.assertRaisesRegex(RuntimeError, "ancestry"):
                    remote._open_existing_lock(parent_fd)
                fixture.cache.chmod(0o700)
                fixture.lock.unlink()
                fixture.lock.symlink_to(fixture.root / "outside")
                with self.assertRaises(OSError):
                    remote._open_existing_lock(parent_fd)
            finally:
                os.close(parent_fd)

    def test_private_parent_exception_is_exact_and_fails_closed(self):
        with LockFixture() as fixture, mock.patch.object(remote, "UID", os.geteuid()):
            parent_fd = os.open(fixture.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                for path, bad_mode in ((fixture.cache, 0o710),
                                       (fixture.cache, 0o750),
                                       (fixture.lock_dir, 0o777),
                                       (fixture.lock_dir, 0o770),
                                       (fixture.lock, 0o666),
                                       (fixture.lock, 0o660)):
                    original_mode = path.stat().st_mode & 0o777
                    with self.subTest(path=path.name, mode=oct(bad_mode)):
                        path.chmod(bad_mode)
                        with self.assertRaises(RuntimeError):
                            remote._open_existing_lock(parent_fd)
                    path.chmod(original_mode)
                parts, record = remote._open_existing_lock(parent_fd)
                try:
                    self.assertEqual(record["lock_directory"]["mode"], "0o775")
                finally:
                    remote._close_lock(parts)
            finally:
                os.close(parent_fd)

    def test_unknown_or_present_acl_blocks_private_parent_exception(self):
        with LockFixture() as fixture, mock.patch.object(remote, "UID", os.geteuid()):
            parent_fd = os.open(fixture.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with mock.patch.object(remote.os, "getxattr", return_value=b"named-entry"):
                    with self.assertRaisesRegex(RuntimeError, "ACL is not absent"):
                        remote._open_existing_lock(parent_fd)
                with mock.patch.object(remote.os, "getxattr",
                                       side_effect=OSError(errno.EACCES, "denied")):
                    with self.assertRaisesRegex(RuntimeError, "cannot authenticate"):
                        remote._open_existing_lock(parent_fd)
            finally:
                os.close(parent_fd)

    def test_transport_rejects_private_cache_or_acl_spoofing(self):
        probe = fresh_probe()
        identity = probe["identity_before"]
        self.assertEqual(transport._validate_lock_identity(lock_fixture(), identity),
                         lock_fixture())
        variants = []
        for section, key, value in (("cache_directory", "mode", "0o750"),
                                    ("cache_directory", "acl", {"access": "ENODATA"}),
                                    ("lock_directory", "mode", "0o777"),
                                    ("lock_directory", "mode", "0o1775"),
                                    ("lock_directory", "acl", {"access": "present",
                                                                  "default": "ENODATA"}),
                                    ("file", "mode", "0o666"),
                                    ("file", "mode", "0o4664"),
                                    ("file", "acl", {"access": "present"})):
            altered = copy.deepcopy(lock_fixture())
            altered[section][key] = value
            variants.append(altered)
        for altered in variants:
            with self.subTest(altered=altered), self.assertRaises(RuntimeError):
                transport._validate_lock_identity(altered, identity)

    def _writer_fixture(self, fixture, *, final_label=None, byte_count=8):
        parent_fd = os.open(fixture.root, os.O_RDONLY | os.O_DIRECTORY)
        state = {"hostname": remote.HOST, "uid": os.geteuid(),
                 "root_free_bytes": 30_000_000_000, "root_free_inodes": 100_000,
                 "boot_id": "synthetic", "mount_namespace": "mnt:[synthetic]",
                 "mount": {}, "block_aliases": {}, "ancestry": {}}
        fake_label = final_label or (lambda: {
            "completed_monotonic_ns": time.monotonic_ns()})
        patches = [
            mock.patch.object(remote, "UID", os.geteuid()),
            mock.patch.object(remote, "TARGET_NAME", "new-archive.tar.gz"),
            mock.patch.object(remote, "BYTES", byte_count),
            mock.patch.object(remote, "_open_parent", return_value=(parent_fd, {})),
            mock.patch.object(remote, "_identity", return_value=state),
            mock.patch.object(remote, "_stable"),
            mock.patch.object(remote, "_require_preflight_binding", return_value="bound"),
            mock.patch.object(remote, "_safe_path", return_value={"exists": False}),
            mock.patch.object(remote, "_live_prewrite_safety", return_value={}),
            mock.patch.object(remote, "_require_no_stops_or_processes", return_value={}),
            mock.patch.object(remote, "_final_label_observation", side_effect=fake_label),
        ]
        return patches

    def test_last_label_scan_and_expired_observation_block_before_create(self):
        with LockFixture() as fixture:
            for final in (lambda: (_ for _ in ()).throw(RuntimeError("science label appeared")),
                          lambda: {"completed_monotonic_ns": time.monotonic_ns() - 3_000_000_000}):
                patches = self._writer_fixture(fixture, final_label=final)
                with contextlib.ExitStack() as stack:
                    for patch in patches:
                        stack.enter_context(patch)
                    with self.assertRaises(RuntimeError):
                        remote.write(io.BytesIO(b"payload"), {})
                self.assertFalse((fixture.root / "new-archive.tar.gz").exists())

    def test_label_after_slow_held_checks_is_seen_under_lock_before_o_excl(self):
        with LockFixture() as fixture:
            late = {"appeared": False, "lock_held_at_final_scan": False}

            def slow_live_safety(*, owned_lock=None):
                if owned_lock is not None:
                    late["appeared"] = True
                return {}

            def final_scan():
                rival_fd = os.open(fixture.lock, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(rival_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    late["lock_held_at_final_scan"] = True
                finally:
                    os.close(rival_fd)
                if late["appeared"]:
                    raise RuntimeError("science label appeared after slow live checks")
                raise AssertionError("final scan ran before held live checks")

            patches = self._writer_fixture(fixture, final_label=final_scan)
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                stack.enter_context(mock.patch.object(remote, "_live_prewrite_safety",
                                                      side_effect=slow_live_safety))
                with self.assertRaisesRegex(RuntimeError, "science label appeared"):
                    remote.write(io.BytesIO(b"payload"), {})
            self.assertTrue(late["lock_held_at_final_scan"])
            self.assertFalse((fixture.root / "new-archive.tar.gz").exists())

    def test_held_lock_blocks_writer_before_create(self):
        with LockFixture() as fixture:
            parent_fd = os.open(fixture.root, os.O_RDONLY | os.O_DIRECTORY)
            parts, _ = remote._open_existing_lock(parent_fd)
            try:
                fcntl.flock(parts[2], fcntl.LOCK_EX | fcntl.LOCK_NB)
                patches = self._writer_fixture(fixture)
                with contextlib.ExitStack() as stack:
                    for patch in patches:
                        stack.enter_context(patch)
                    with self.assertRaisesRegex(RuntimeError, "already held"):
                        remote.write(io.BytesIO(b"payload"), {})
                self.assertFalse((fixture.root / "new-archive.tar.gz").exists())
            finally:
                remote._close_lock(parts)
                os.close(parent_fd)

    def test_partial_target_retained_and_readonly_postflight_can_report_it(self):
        with LockFixture() as fixture:
            patches = self._writer_fixture(fixture, byte_count=8)
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                    remote.write(io.BytesIO(b"payload"), {})
            target = fixture.root / "new-archive.tar.gz"
            self.assertEqual(target.read_bytes(), b"payload")
            self.assertEqual(target.stat().st_nlink, 1)
        preflight = fresh_probe()
        partial = historical.meta("regular", inode=700777, mode="0o600", size=7)
        postflight = fresh_probe(archive=partial)
        self.assertEqual(transport.compare_postflight(preflight, postflight)["status"],
                         "READ_ONLY_POSTFLIGHT_PASS")

    def test_missing_or_malformed_final_evidence_fails_separate_session(self):
        preflight, write, verify = compared_receipts()
        self.assertEqual(transport.compare_sessions(write, verify, preflight=preflight)["status"],
                         "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        variants = []
        absent = copy.deepcopy(write)
        del absent["prewrite_evidence"]["final_label_observation"]
        variants.append(absent)
        late = copy.deepcopy(write)
        late["prewrite_evidence"]["final_label_observation"]["scan"]["match_count"] = 1
        variants.append(late)
        stale = copy.deepcopy(write)
        stale["prewrite_evidence"]["target_open_attempt_monotonic_ns"] = 2_000_000_000
        variants.append(stale)
        unbound = copy.deepcopy(write)
        unbound["prewrite_evidence"]["cooperative_lock_after_create"]["file"]["inode"] += 1
        variants.append(unbound)
        for item in variants:
            with self.subTest(item=variants.index(item)), self.assertRaises(RuntimeError):
                transport.compare_sessions(item, verify, preflight=preflight)

    def test_package_entrypoints_are_dispatch_hold(self):
        with mock.patch.object(transport, "checked_ssh_identity",
                               side_effect=AssertionError("key metadata touched")):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                transport.build_ssh_argv("write", preflight=fresh_probe())
        result = subprocess.run(["python3", "-B", str(HERE / "stage_r4_remote_lock_hold.py"),
                                 "write"], capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DISPATCH HOLD", result.stderr)


if __name__ == "__main__":
    unittest.main()
