"""Offline negatives for the disabled trn21 final-r4 private-cache lock policy."""

import contextlib
import copy
import errno
import fcntl
import hashlib
import io
import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_final_r4_once as operator
import stage_r4_remote_final_r4 as remote
import stage_r4_transport_final_r4 as transport
import test_operator_transfer_final_r3_once as r3_fixtures


EMPTY = hashlib.sha256(b"").hexdigest()
ACL_ABSENT = {"system.posix_acl_access": "absent_ENODATA",
              "system.posix_acl_default": "absent_ENODATA"}


def mount_proof():
    return {"mount_namespace": "mnt:[synthetic]",
            "mountinfo_sha256": "a" * 64, "entry_count": 30,
            "root_mount_id": "27", "root_mount_point": "/",
            "root_mount_root": "/", "root_major_minor": "259:1",
            "root_filesystem": "ext4", "root_source": "/dev/root",
            "unsafe_aliases": []}


def lock_identity():
    item = copy.deepcopy(r3_fixtures.lock_identity_fixture())
    for name in ("cache", "host_directory", "file"):
        item[name]["inode"] = remote.LOCK_OBSERVED_INODES[name]
        item[name]["gid"] = remote.LOCK_OBSERVED_GID
        item[name]["mode"] = oct(remote.LOCK_OBSERVED_MODES[name])
    item["trust"] = {
        "policy": "private_cache_exact_observed_modes_no_acl_no_same_ns_alias",
        "acl": {"cache": copy.deepcopy(ACL_ABSENT),
                "host_directory": copy.deepcopy(ACL_ABSENT),
                "file": {"system.posix_acl_access": "absent_ENODATA"}},
        "mounts": mount_proof(),
        "cache_nlink": remote.LOCK_OBSERVED_NLINKS["cache"],
        "host_directory_nlink": remote.LOCK_OBSERVED_NLINKS["host_directory"],
    }
    return item


def clean_receipts():
    pre, write, verify, post = r3_fixtures.clean_receipts()
    identity = lock_identity()
    pre["host_lock_identity"] = copy.deepcopy(identity)
    write["prewrite_probe"]["host_lock_identity"] = copy.deepcopy(identity)
    write["final_creation_gate"]["host_lock"]["identity"] = copy.deepcopy(identity)
    write["host_lock_identity"] = copy.deepcopy(identity)
    write["preflight_lock_sha256"] = remote._lock_identity_digest(identity)
    verify["host_lock_identity"] = copy.deepcopy(identity)
    post["host_lock_identity"] = copy.deepcopy(identity)
    return pre, write, verify, post


def local_lock(root):
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


@contextlib.contextmanager
def local_policy(root):
    lock = local_lock(root)
    cache, directory = lock.parent.parent, lock.parent
    inodes = {"cache": cache.stat().st_ino,
              "host_directory": directory.stat().st_ino,
              "file": lock.stat().st_ino}
    links = {"cache": cache.stat().st_nlink,
             "host_directory": directory.stat().st_nlink}
    with (mock.patch.object(remote, "LOCK_PATH", str(lock)),
          mock.patch.object(remote, "LOCK_OBSERVED_INODES", inodes),
          mock.patch.object(remote, "LOCK_OBSERVED_NLINKS", links),
          mock.patch.object(remote, "LOCK_OBSERVED_DEVICE", root.stat().st_dev),
          mock.patch.object(remote, "_lock_mount_proof", return_value=mount_proof())):
        parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            yield lock, parent_fd
        finally:
            os.close(parent_fd)


class FinalR4DisabledTests(unittest.TestCase):
    def test_disabled_request_stops_before_marker_or_ssh(self):
        request, binding = operator._read_frozen_request_and_code()
        self.assertFalse(request["dispatch_authorized"])
        self.assertFalse(request["source_transfer_authorized"])
        self.assertEqual(binding["request_sha256"], operator.REQUEST_SHA256)
        self.assertEqual(request["prior_final_r3_disabled_static_review_sha256"],
                         operator.PRIOR_R3_STATIC_REVIEW_SHA256)
        self.assertEqual(request["trn21_lock_acl_readonly_raw_sha256"],
                         operator.LOCK_ACL_RAW_SHA256)
        with (mock.patch.object(operator, "_consume_fixed_marker",
                                side_effect=AssertionError("marker reached")),
              mock.patch.object(operator, "_run_ssh_final",
                                side_effect=AssertionError("SSH reached")),
              mock.patch.object(operator.subprocess, "Popen",
                                side_effect=AssertionError("process spawned"))):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()

    def test_exact_private_case_is_read_only_and_flock_compatible(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as name:
            with local_policy(Path(name)) as (lock, parent_fd):
                before = (lock.read_bytes(), lock.stat().st_mtime_ns,
                          stat.S_IMODE(lock.stat().st_mode))
                fd, identity = remote._open_existing_host_lock(parent_fd)
                try:
                    self.assertEqual(identity["trust"]["policy"],
                                     "private_cache_exact_observed_modes_no_acl_no_same_ns_alias")
                    self.assertEqual(identity["trust"]["acl"]["cache"], ACL_ABSENT)
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    launcher = os.open(lock, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
                    try:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(launcher, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(launcher)
                finally:
                    os.close(fd)
                self.assertEqual((lock.read_bytes(), lock.stat().st_mtime_ns,
                                  stat.S_IMODE(lock.stat().st_mode)), before)

    def test_exact_exception_rejects_mode_inode_gid_link_and_symlink_drift(self):
        mutations = ("cache_traverse", "directory_world_write", "file_world_write",
                     "file_other_mode", "wrong_gid", "wrong_device", "wrong_inode", "hardlink",
                     "symlink", "cache_symlink", "directory_symlink", "fifo")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                    dir="/dev/shm") as name:
                root = Path(name)
                with local_policy(root) as (lock, parent_fd):
                    if mutation == "cache_traverse":
                        lock.parent.parent.chmod(0o710)
                    elif mutation == "directory_world_write":
                        lock.parent.chmod(0o777)
                    elif mutation == "file_world_write":
                        lock.chmod(0o666)
                    elif mutation == "file_other_mode":
                        lock.chmod(0o660)
                    elif mutation == "wrong_gid":
                        with mock.patch.object(remote, "LOCK_OBSERVED_GID", 987):
                            with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                                remote._open_existing_host_lock(parent_fd)
                        continue
                    elif mutation == "wrong_device":
                        with mock.patch.object(remote, "LOCK_OBSERVED_DEVICE", 987):
                            with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                                remote._open_existing_host_lock(parent_fd)
                        continue
                    elif mutation == "wrong_inode":
                        pins = dict(remote.LOCK_OBSERVED_INODES)
                        pins["file"] += 1
                        with mock.patch.object(remote, "LOCK_OBSERVED_INODES", pins):
                            with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                                remote._open_existing_host_lock(parent_fd)
                        continue
                    elif mutation == "hardlink":
                        os.link(lock, root / "outside-lock-alias")
                    elif mutation == "symlink":
                        lock.unlink()
                        lock.symlink_to(root / "outside-lock-alias")
                    elif mutation == "cache_symlink":
                        cache = lock.parent.parent
                        cache.rename(root / "hidden-cache")
                        cache.symlink_to(root / "hidden-cache", target_is_directory=True)
                    elif mutation == "directory_symlink":
                        directory = lock.parent
                        directory.rename(directory.parent / "hidden-host")
                        directory.symlink_to(directory.parent / "hidden-host",
                                             target_is_directory=True)
                    else:
                        lock.unlink()
                        os.mkfifo(lock)
                    with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                        remote._open_existing_host_lock(parent_fd)

    def test_raced_fifo_open_is_nofollow_nonblocking_and_holds(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as name:
            with local_policy(Path(name)) as (lock, parent_fd):
                real_open = os.open

                def swap_on_lock_open(path, flags, *args, **kwargs):
                    if path == remote.LOCK_NAME:
                        self.assertTrue(flags & os.O_NOFOLLOW)
                        self.assertTrue(flags & os.O_NONBLOCK)
                        lock.unlink()
                        os.mkfifo(lock)
                    return real_open(path, flags, *args, **kwargs)

                with mock.patch.object(remote.os, "open", side_effect=swap_on_lock_open):
                    with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                        remote._open_existing_host_lock(parent_fd)
                self.assertTrue(stat.S_ISFIFO(lock.lstat().st_mode))

    def test_acl_present_or_unknown_is_hold(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as name:
            with local_policy(Path(name)) as (_, parent_fd):
                for position in range(1, 6):
                    with self.subTest(xattr_position=position):
                        calls = 0

                        def one_acl(fd, key):
                            nonlocal calls
                            calls += 1
                            if calls == position:
                                return b"ACL"
                            raise OSError(errno.ENODATA, "no ACL")

                        with mock.patch.object(remote.os, "getxattr",
                                               side_effect=one_acl):
                            with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                                remote._open_existing_host_lock(parent_fd)
                for code in (errno.EACCES, errno.EOPNOTSUPP):
                    with self.subTest(errno=code), mock.patch.object(
                            remote.os, "getxattr",
                            side_effect=OSError(code, "ACL unknown")):
                        with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                            remote._open_existing_host_lock(parent_fd)

    def test_same_namespace_mount_alias_or_malformed_mountinfo_is_hold(self):
        root = b"27 0 259:1 / / rw,relatime - ext4 /dev/root rw\n"
        proof = remote._lock_mount_proof(root)
        self.assertEqual(proof["unsafe_aliases"], [])
        self.assertEqual(proof["root_mount_id"], "27")
        bad = (
            root + b"40 27 259:1 /home/ubuntu/.cache/trainium_freshstart /tmp/alias rw - ext4 /dev/root rw\n",
            root + b"40 27 259:1 /src /home/ubuntu/.cache/alias rw - ext4 /dev/root rw\n",
            b"27 0 259:1 / / rw - overlay overlay rw\n",
            b"malformed\n",
        )
        for raw in bad:
            with self.subTest(raw=raw[:55]):
                with self.assertRaises(RuntimeError):
                    remote._lock_mount_proof(raw)

    def test_literal_r4_write_argv_binds_fresh_host_and_lock_proof(self):
        pre, _, _, _ = clean_receipts()
        with mock.patch.object(transport, "checked_ssh_identity", return_value={}):
            argv = transport.build_ssh_argv("write", preflight=pre)
        self.assertEqual(shlex.split(argv[-1]),
                         ["exec", "python3", "-c", transport.reviewed_remote_source(),
                          "write", remote._identity_digest(pre["identity_after"]),
                          remote._lock_identity_digest(pre["host_lock_identity"])])
        self.assertEqual(argv[-2], transport.ENDPOINT)

    def test_receipts_bind_exact_private_parent_and_acl_proof(self):
        pre, write, verify, post = clean_receipts()
        self.assertIs(transport.validate_probe(pre, before_write=True), pre)
        continuity = transport.compare_preflight_to_write(pre, write)
        self.assertEqual(continuity["status"],
                         "PREFLIGHT_TO_FINAL_LOCKED_LABEL_GATE_PASS")
        persistence = transport.compare_sessions(pre, write, verify)
        self.assertEqual(persistence["status"], "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        self.assertEqual(transport.compare_postflight(pre, post,
                         persistence=persistence)["status"],
                         "READ_ONLY_POSTFLIGHT_PASS")
        for mutation in ("cache_mode", "file_inode", "acl_unknown",
                         "alias", "namespace", "prewrite_drift"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(pre)
                identity = bad["host_lock_identity"]
                if mutation == "cache_mode":
                    identity["cache"]["mode"] = "0o710"
                elif mutation == "file_inode":
                    identity["file"]["inode"] += 1
                elif mutation == "acl_unknown":
                    identity["trust"]["acl"]["cache"]["system.posix_acl_access"] = "unknown"
                elif mutation == "alias":
                    identity["trust"]["mounts"]["unsafe_aliases"] = ["/tmp/alias"]
                elif mutation == "namespace":
                    identity["trust"]["mounts"]["mount_namespace"] = "mnt:[other]"
                else:
                    bad_write = copy.deepcopy(write)
                    bad_write["prewrite_probe"]["host_lock_identity"]["file"]["inode"] += 1
                    with self.assertRaises(RuntimeError):
                        transport.compare_preflight_to_write(pre, bad_write)
                    continue
                with self.assertRaises(RuntimeError):
                    transport.validate_probe(bad, before_write=True)

    def test_late_science_label_still_blocks_target_creation(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as name:
            root = Path(name)
            with (mock.patch.object(r3_fixtures, "remote", remote),
                  mock.patch.object(remote, "_lock_mount_proof",
                                    return_value=mount_proof())):
                with r3_fixtures.simulated_writer(root) as (pin, lock_pin,
                                                             target, _, _):
                    original_gate = remote._activity_gate_snapshot
                    label = root / ("late-" + remote.LABEL)

                    def insert_after_slow_gate(held_lock_pid=None):
                        result = original_gate(held_lock_pid)
                        if held_lock_pid is not None:
                            label.write_text("late", encoding="utf-8")
                        return result

                    with mock.patch.object(remote, "_activity_gate_snapshot",
                                           side_effect=insert_after_slow_gate):
                        with self.assertRaisesRegex(RuntimeError, "science label"):
                            remote.write(io.BytesIO(b""), pin, lock_pin)
                    self.assertTrue(label.exists())
                    self.assertFalse(target.exists())

    def test_lock_remains_held_through_write_receipt_then_releases(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as name:
            root = Path(name)
            with (mock.patch.object(r3_fixtures, "remote", remote),
                  mock.patch.object(remote, "_lock_mount_proof",
                                    return_value=mount_proof())):
                with r3_fixtures.simulated_writer(root) as (pin, lock_pin,
                                                             target, lock, _):
                    payload = b"synthetic-local-capsule"
                    original_receipt = remote._transfer_receipt
                    saw_held = []

                    def receipt_under_lock(*args, **kwargs):
                        other = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
                        try:
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            saw_held.append(True)
                        finally:
                            os.close(other)
                        return original_receipt(*args, **kwargs)

                    with (mock.patch.object(remote, "BYTES", len(payload)),
                          mock.patch.object(remote, "SHA256",
                                            hashlib.sha256(payload).hexdigest()),
                          mock.patch.object(remote, "_transfer_receipt",
                                            side_effect=receipt_under_lock)):
                        receipt = remote.write(io.BytesIO(payload), pin, lock_pin)
                    self.assertEqual(saw_held, [True])
                    self.assertEqual(target.read_bytes(), payload)
                    self.assertEqual(receipt["final_creation_gate"]["host_lock"]["mode"],
                                     "LOCK_EX|LOCK_NB")
                    after = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
                    try:
                        fcntl.flock(after, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(after)


if __name__ == "__main__":
    unittest.main()
