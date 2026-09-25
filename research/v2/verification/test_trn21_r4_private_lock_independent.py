"""Independent offline adversarial checks for the disabled trn21 r4 transfer."""

import copy
import errno
import fcntl
import hashlib
import io
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


PACKAGE = Path(__file__).resolve().parents[1] / "authorizations" / "trn21-c5c-a0-r4-transfer-only"
sys.path.insert(0, str(PACKAGE))
import operator_transfer_final_r4_once as operator
import stage_r4_remote_final_r4 as remote
import stage_r4_transport_final_r4 as transport
import test_operator_transfer_final_r3_once as r3_fixtures
import test_operator_transfer_final_r4_once as r4_fixtures


def _node(kind, mode, inode, *, device=66305, uid=1000, gid=1000, links=1, size=0):
    return SimpleNamespace(st_mode=kind | mode, st_ino=inode, st_dev=device,
                           st_uid=uid, st_gid=gid, st_nlink=links, st_size=size)


def _local_lock(root):
    cache = root / ".cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    directory = cache / "trainium_freshstart"
    directory.mkdir(mode=0o775)
    directory.chmod(0o775)
    lock = directory / "device0.lock"
    lock.write_bytes(b"")
    lock.chmod(0o664)
    return lock


def _mount_proof():
    return {"mount_namespace": "mnt:[synthetic]", "mountinfo_sha256": "a" * 64,
            "entry_count": 1, "root_mount_id": "27", "root_mount_point": "/",
            "root_mount_root": "/", "root_major_minor": "259:1",
            "root_filesystem": "ext4", "root_source": "/dev/root", "unsafe_aliases": []}


class IndependentPrivateLockTests(unittest.TestCase):
    def test_disabled_gate_cannot_reach_authority_marker_or_ssh(self):
        request, binding = operator._read_frozen_request_and_code()
        self.assertEqual(binding["request_sha256"], operator.REQUEST_SHA256)
        self.assertIs(request["dispatch_authorized"], False)
        self.assertIs(request["source_transfer_authorized"], False)
        with (mock.patch.object(operator, "_read_required", side_effect=AssertionError("authority reached")),
              mock.patch.object(operator, "_consume_fixed_marker", side_effect=AssertionError("marker reached")),
              mock.patch.object(operator, "_run_ssh_final", side_effect=AssertionError("SSH reached")),
              mock.patch.object(operator.subprocess, "Popen", side_effect=AssertionError("spawn reached"))):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()

    def test_exact_private_exception_and_every_critical_metadata_pin(self):
        parent = _node(stat.S_IFDIR, 0o750, 295711, links=167)
        cache = _node(stat.S_IFDIR, 0o700, 295736, links=6)
        directory = _node(stat.S_IFDIR, 0o775, 596822, links=3)
        lock = _node(stat.S_IFREG, 0o664, 596823)
        self.assertEqual(remote._lock_mode_policy(parent, cache, directory, lock),
                         "private_cache_exact_observed_modes_no_acl_no_same_ns_alias")
        mutations = (
            (1, "st_mode", stat.S_IFDIR | 0o710),
            (2, "st_mode", stat.S_IFDIR | 0o777),
            (3, "st_mode", stat.S_IFREG | 0o666),
            (1, "st_uid", 999), (2, "st_gid", 999),
            (0, "st_dev", 999), (1, "st_dev", 999),
            (1, "st_ino", 295737), (2, "st_ino", 596823),
            (3, "st_ino", 596824), (1, "st_nlink", 7),
            (2, "st_nlink", 4), (3, "st_nlink", 2),
            (3, "st_size", 1), (3, "st_mode", stat.S_IFIFO | 0o664),
        )
        for index, field, value in mutations:
            with self.subTest(index=index, field=field, value=value):
                nodes = copy.deepcopy([parent, cache, directory, lock])
                setattr(nodes[index], field, value)
                with self.assertRaises(RuntimeError):
                    remote._lock_mode_policy(*nodes)
        strict_directory = _node(stat.S_IFDIR, 0o755, 111111)
        strict_lock = _node(stat.S_IFREG, 0o644, 222222)
        self.assertEqual(remote._lock_mode_policy(parent, cache, strict_directory, strict_lock),
                         "strict_r3_non_group_writable")

    def test_opened_inode_flock_and_raced_fifo_are_bounded(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            lock = _local_lock(root)
            observed = {"cache": lock.parent.parent.stat().st_ino,
                        "host_directory": lock.parent.stat().st_ino,
                        "file": lock.stat().st_ino}
            links = {"cache": lock.parent.parent.stat().st_nlink,
                     "host_directory": lock.parent.stat().st_nlink}
            initial = (lock.read_bytes(), lock.stat().st_mtime_ns, stat.S_IMODE(lock.stat().st_mode))
            parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with (mock.patch.object(remote, "LOCK_PATH", str(lock)),
                      mock.patch.object(remote, "LOCK_OBSERVED_INODES", observed),
                      mock.patch.object(remote, "LOCK_OBSERVED_NLINKS", links),
                      mock.patch.object(remote, "LOCK_OBSERVED_DEVICE", root.stat().st_dev),
                      mock.patch.object(remote, "_lock_mount_proof", return_value=_mount_proof())):
                    fd, identity = remote._open_existing_host_lock(parent_fd)
                    try:
                        self.assertEqual(identity["file"]["inode"], observed["file"])
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        competing = os.open(lock, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
                        try:
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        finally:
                            os.close(competing)
                    finally:
                        os.close(fd)
                    self.assertEqual((lock.read_bytes(), lock.stat().st_mtime_ns,
                                      stat.S_IMODE(lock.stat().st_mode)), initial)
                    real_open = os.open

                    def substitute_fifo(path, flags, *args, **kwargs):
                        if path == "device0.lock":
                            self.assertTrue(flags & os.O_NONBLOCK)
                            self.assertTrue(flags & os.O_NOFOLLOW)
                            lock.unlink()
                            os.mkfifo(lock)
                        return real_open(path, flags, *args, **kwargs)

                    with mock.patch.object(remote.os, "open", side_effect=substitute_fifo):
                        with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                            remote._open_existing_host_lock(parent_fd)
                    self.assertTrue(stat.S_ISFIFO(lock.lstat().st_mode))
            finally:
                os.close(parent_fd)

    def test_mount_alias_and_unknown_acl_fail_closed(self):
        root = b"27 0 259:1 / / rw,relatime - ext4 /dev/root rw\n"
        self.assertEqual(remote._lock_mount_proof(root)["unsafe_aliases"], [])
        unsafe = (
            root + b"40 27 259:1 /home/ubuntu/.cache/trainium_freshstart /tmp/alias rw - ext4 /dev/root rw\n",
            root + b"40 27 259:1 /home/ubuntu/.cache/trainium_freshstart/device0.lock /tmp/alias rw - ext4 /dev/root rw\n",
            root + b"40 27 259:1 /src /home/ubuntu/.cache/child rw - ext4 /dev/root rw\n",
            b"27 0 259:1 / / rw - overlay overlay rw\n",
            b"malformed\n",
        )
        for mountinfo in unsafe:
            with self.subTest(mountinfo=mountinfo[:75]), self.assertRaises(RuntimeError):
                remote._lock_mount_proof(mountinfo)
        with mock.patch.object(remote.os, "getxattr", return_value=b"ACL"):
            with self.assertRaisesRegex(RuntimeError, "ACL is present"):
                remote._acl_absence(0, directory=True)
        with mock.patch.object(remote.os, "getxattr",
                               side_effect=OSError(errno.EOPNOTSUPP, "unknown")):
            with self.assertRaisesRegex(RuntimeError, "could not be proved absent"):
                remote._acl_absence(0, directory=True)

    def test_receipt_mismatches_and_late_label_hold(self):
        pre, write, verify, post = r4_fixtures.clean_receipts()
        self.assertEqual(transport.compare_sessions(pre, write, verify)["status"],
                         "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        failures = []
        bad = copy.deepcopy(pre)
        bad["host_lock_identity"]["trust"]["acl"]["cache"]["system.posix_acl_access"] = "present"
        failures.append(lambda receipt=bad: transport.validate_probe(receipt, before_write=True))
        bad_disk = copy.deepcopy(pre)
        bad_disk["identity_before"]["root_free_bytes"] = transport.PRE_FREE - 1
        failures.append(lambda receipt=bad_disk: transport.validate_probe(receipt, before_write=True))
        bad_old_asset = copy.deepcopy(pre)
        bad_old_asset["protected"]["e65_source"]["train.py"]["sha256"] = "0" * 64
        failures.append(lambda receipt=bad_old_asset: transport.validate_probe(receipt, before_write=True))
        bad_target = copy.deepcopy(pre)
        bad_target["targets"][transport.TARGET] = {"exists": True}
        failures.append(lambda receipt=bad_target: transport.validate_probe(receipt, before_write=True))
        bad = copy.deepcopy(write)
        bad["prewrite_probe"]["host_lock_identity"]["file"]["inode"] += 1
        failures.append(lambda receipt=bad: transport.compare_preflight_to_write(pre, receipt))
        bad = copy.deepcopy(verify)
        bad["host_lock_identity"]["trust"]["mounts"]["mount_namespace"] = "mnt:[drift]"
        failures.append(lambda receipt=bad: transport.compare_sessions(pre, write, receipt))
        bad = copy.deepcopy(post)
        bad["science_label_scan"]["match_count"] = 1
        failures.append(lambda receipt=bad: transport.compare_postflight(pre, receipt))
        bad_post_disk = copy.deepcopy(post)
        bad_post_disk["identity_after"]["root_free_bytes"] = transport.POST_FREE - 1
        failures.append(lambda receipt=bad_post_disk: transport.validate_probe(receipt, before_write=False))
        for index, action in enumerate(failures):
            with self.subTest(mutation=index), self.assertRaises(RuntimeError):
                action()
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            with (mock.patch.object(r3_fixtures, "remote", remote),
                  mock.patch.object(remote, "_lock_mount_proof", return_value=_mount_proof())):
                with r3_fixtures.simulated_writer(root) as (pin, lock_pin, target, _, _):
                    original = remote._host_lock_identity
                    late = root / ("late-" + remote.LABEL)

                    def insert_label_after_lock_recheck(parent_fd):
                        identity = original(parent_fd)
                        late.write_text("late", encoding="utf-8")
                        return identity

                    with mock.patch.object(remote, "_host_lock_identity",
                                           side_effect=insert_label_after_lock_recheck):
                        with self.assertRaisesRegex(RuntimeError, "science label"):
                            remote.write(io.BytesIO(b""), pin, lock_pin)
                    self.assertTrue(late.exists())
                    self.assertFalse(target.exists())

    def test_last_activity_gate_rejects_new_stop_process_neuron_or_target(self):
        for kind in ("STOP", "process", "Neuron", "target"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
                root = Path(temporary)
                activity = {"active": False, "kind": kind}
                with (mock.patch.object(r3_fixtures, "remote", remote),
                      mock.patch.object(remote, "_lock_mount_proof", return_value=_mount_proof())):
                    with r3_fixtures.simulated_writer(root, activity) as (pin, lock_pin, target, _, _):
                        slow_check = remote._neuron_ls
                        calls = 0

                        def introduce_during_last_slow_check():
                            nonlocal calls
                            calls += 1
                            if calls == 3 and kind == "Neuron":
                                activity["active"] = True
                            result = slow_check()
                            if calls == 3:
                                if kind == "target":
                                    target.write_bytes(b"raced target")
                                else:
                                    activity["active"] = True
                            return result

                        with mock.patch.object(remote, "_neuron_ls",
                                               side_effect=introduce_during_last_slow_check):
                            with self.assertRaises(RuntimeError):
                                remote.write(io.BytesIO(b""), pin, lock_pin)
                        self.assertEqual(calls, 3)
                        if kind == "target":
                            self.assertEqual(target.read_bytes(), b"raced target")
                        else:
                            self.assertFalse(target.exists())

    def test_inherited_end_of_probe_inode_gap_falsely_passes_full_gates(self):
        """A terminal low inode count passes the supposedly full probe validators."""
        pre, write, verify, post = r4_fixtures.clean_receipts()
        pre["identity_after"]["root_free_inodes"] = transport.MIN_INODES - 1
        post["identity_after"]["root_free_inodes"] = transport.MIN_INODES - 1
        self.assertIs(transport.validate_probe(pre, before_write=True), pre)
        self.assertIs(transport.validate_probe(post, before_write=False), post)
        persistence = transport.compare_sessions(pre, write, verify)
        self.assertEqual(persistence["status"], "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        self.assertEqual(transport.compare_postflight(pre, post,
                         persistence=persistence)["status"], "READ_ONLY_POSTFLIGHT_PASS")

    def test_short_stream_retains_partial_target_and_prevents_same_path_retry(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            with (mock.patch.object(r3_fixtures, "remote", remote),
                  mock.patch.object(remote, "_lock_mount_proof", return_value=_mount_proof())):
                with r3_fixtures.simulated_writer(root) as (pin, lock_pin, target, _, _):
                    expected = b"longer synthetic capsule"
                    partial = b"short"
                    with (mock.patch.object(remote, "BYTES", len(expected)),
                          mock.patch.object(remote, "SHA256", hashlib.sha256(expected).hexdigest())):
                        with self.assertRaisesRegex(RuntimeError, "partial target retained"):
                            remote.write(io.BytesIO(partial), pin, lock_pin)
                        self.assertEqual(target.read_bytes(), partial)
                        with self.assertRaises(RuntimeError):
                            remote.write(io.BytesIO(expected), pin, lock_pin)
                        self.assertEqual(target.read_bytes(), partial)


if __name__ == "__main__":
    unittest.main()
