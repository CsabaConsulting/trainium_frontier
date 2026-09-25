"""Offline adversarial tests for the disabled trn21 final-r3 transfer."""

import contextlib
import copy
import datetime
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
import operator_transfer_final_r3_once as operator
import stage_r4_remote_final_r3 as remote
import stage_r4_transport_final_r3 as transport
import test_stage_r4_transfer as fixtures


EMPTY = hashlib.sha256(b"").hexdigest()


def lock_identity_fixture():
    return {
        "path": remote.LOCK_PATH,
        "ubuntu": copy.deepcopy(fixtures.identity()["ancestry"]["ubuntu"]),
        "cache": {"type": "directory", "device": 66305, "inode": 800001,
                  "uid": 1000, "gid": 1000, "mode": "0o755"},
        "host_directory": {"type": "directory", "device": 66305, "inode": 800002,
                           "uid": 1000, "gid": 1000, "mode": "0o755"},
        "file": {"exists": True, "type": "regular", "device": 66305,
                 "inode": 800003, "uid": 1000, "gid": 1000,
                 "mode": "0o644", "nlink": 1, "bytes": 0},
    }


def idle_fuser():
    return {"path_exists": True, "rc": 1, "pids": [],
            "stderr_empty": True, "stdout_sha256": EMPTY,
            "stderr_sha256": EMPTY}


def self_fuser(pid):
    return {"path_exists": True, "rc": 0, "pids": [pid],
            "stderr_empty": False, "stdout_sha256": "a" * 64,
            "stderr_sha256": remote._held_lock_fuser_stderr_sha256()}


def label_scan(start, end):
    return {"rc": 0, "matches": [], "match_count": 0,
            "truncated": False, "stderr_empty": True,
            "stdout_sha256": EMPTY,
            "started_utc": fixtures.utc(start),
            "completed_utc": fixtures.utc(end)}


def decorated_probe(*, pid=101, offset=-5, archive=None):
    result = fixtures.probe(pid=pid, offset=offset, archive=archive)
    result["host_lock_identity"] = lock_identity_fixture()
    result["shared_lock_fuser"] = idle_fuser()
    result["science_label_scan"] = label_scan(offset - .2, offset - .1)
    return result


def clean_receipts():
    pre = decorated_probe()
    archive, write, verify = fixtures.transfer_receipts(pre)
    prewrite = decorated_probe(pid=201, offset=-2)
    write["prewrite_probe"] = prewrite
    first = {key: copy.deepcopy(prewrite[key]) for key in (
        "targets", "stops", "processes", "checkpoint_fusers",
        "shared_lock_fuser", "neuron0_fuser", "neuron_ls")}
    first["utc"] = fixtures.utc(-1.9)
    pre_label = copy.deepcopy(first)
    pre_label["utc"] = fixtures.utc(-1.8)
    final = copy.deepcopy(first)
    final["utc"] = fixtures.utc(-1.6)
    final["shared_lock_fuser"] = self_fuser(write["process_identity"]["pid"])
    final["pre_label_gate"] = pre_label
    final["science_label_scan"] = label_scan(-1.5, -1.4)
    final["host_lock"] = {
        "identity": lock_identity_fixture(), "acquired_utc": fixtures.utc(-1.7),
        "mode": "LOCK_EX|LOCK_NB",
        "access": "O_RDONLY|O_NONBLOCK|O_NOFOLLOW|O_CLOEXEC",
        "writer_process_identity": write["process_identity"],
        "fuser": final["shared_lock_fuser"],
        "identity_rechecked_before_final_label_scan": True,
    }
    write["immediate_gate"] = first
    write["final_creation_gate"] = final
    write["preflight_lock_sha256"] = remote._lock_identity_digest(lock_identity_fixture())
    write["host_lock_identity"] = lock_identity_fixture()
    write["host_lock_fuser"] = final["shared_lock_fuser"]
    write["science_label_scan"] = final["science_label_scan"]
    verify["host_lock_identity"] = lock_identity_fixture()
    verify["host_lock_fuser"] = idle_fuser()
    verify["science_label_scan"] = label_scan(-.4, -.3)
    post = decorated_probe(pid=301, offset=1, archive=archive)
    return pre, write, verify, post


def local_lock(root):
    cache_dir = root / ".cache"
    cache_dir.mkdir(mode=0o700)
    cache_dir.chmod(0o700)
    host_dir = cache_dir / "trainium_freshstart"
    host_dir.mkdir(mode=0o700)
    host_dir.chmod(0o700)
    lock = host_dir / "device0.lock"
    lock.write_bytes(b"existing-lock-sentinel")
    lock.chmod(0o600)
    return lock


@contextlib.contextmanager
def simulated_writer(root, activity=None):
    activity = activity if activity is not None else {}
    lock = local_lock(root)
    current = fixtures.identity()
    with mock.patch.object(remote, "LOCK_PATH", str(lock)):
        parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            lock_id = remote._host_lock_identity(parent_fd)
        finally:
            os.close(parent_fd)
        prewrite = fixtures.probe(pid=202, offset=-2)
        prewrite["host_lock_identity"] = lock_id
        lock_calls = {"count": 0}
        target = root / remote.TARGET_NAME
        original_scan = remote._label_scan

        def parent():
            return os.open(root, os.O_RDONLY | os.O_DIRECTORY), current["ancestry"]

        def path_meta(path):
            if path == remote.TARGET and target.exists():
                return {"exists": True}
            if activity.get("active") and activity.get("kind") == "STOP" and (
                    path == remote.STOP_PATHS[0]):
                return {"exists": True}
            return {"exists": False}

        def processes():
            return {"matched": ([{"pid": 987}] if activity.get("active") and
                                activity.get("kind") == "process" else []),
                    "unreadable_cmdlines": 0}

        def fuser(path):
            if path == str(lock):
                lock_calls["count"] += 1
                if lock_calls["count"] % 3 == 0:
                    return self_fuser(os.getpid())
                return idle_fuser()
            return {"path_exists": True, "rc": 1, "pids": [],
                    "stderr_empty": True}

        def neuron():
            return {"rc": 0, "json_valid": True, "schema_valid": True,
                    "device_count": 1,
                    "non_diagnostic_pids":
                    [987] if activity.get("active") and activity.get("kind") == "Neuron" else [],
                    "stderr_empty": True}

        def scan_local():
            with mock.patch.object(remote, "ROOT", str(root)):
                return original_scan()

        with (mock.patch.object(remote, "probe", return_value=prewrite),
              mock.patch.object(remote, "_open_parent", side_effect=parent),
              mock.patch.object(remote, "_identity", return_value=current),
              mock.patch.object(remote, "_safe_path", side_effect=path_meta),
              mock.patch.object(remote, "_processes", side_effect=processes),
              mock.patch.object(remote, "_fuser", side_effect=fuser),
              mock.patch.object(remote, "_neuron_ls", side_effect=neuron),
              mock.patch.object(remote, "_label_scan", side_effect=scan_local)):
            yield (remote._identity_digest(current),
                   remote._lock_identity_digest(lock_id), target, lock,
                   lock_calls)


class FinalR3DisabledTests(unittest.TestCase):
    def test_exact_disabled_request_stops_before_marker_or_ssh(self):
        request, binding = operator._read_frozen_request_and_code()
        self.assertFalse(request["dispatch_authorized"])
        self.assertFalse(request["source_transfer_authorized"])
        self.assertEqual(binding["request_sha256"], operator.REQUEST_SHA256)
        self.assertEqual(request["existing_host_wide_lock_path"], remote.LOCK_PATH)
        self.assertEqual(request["final_remote_sha256"], operator.FINAL_REMOTE_SHA256)
        self.assertEqual(request["supervisor_final_r2_hold_sha256"],
                         operator.SUPERVISOR_R2_HOLD_SHA256)
        self.assertFalse(operator.MARKER.exists())
        with (mock.patch.object(operator, "_consume_fixed_marker",
                                side_effect=AssertionError("marker reached")),
              mock.patch.object(operator, "_run_ssh_final",
                                side_effect=AssertionError("SSH reached")),
              mock.patch.object(operator.subprocess, "Popen",
                                side_effect=AssertionError("process spawned"))):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()
        self.assertFalse(operator.MARKER.exists())

    def test_existing_lock_is_read_only_nofollow_and_nonblocking(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            lock = local_lock(root)
            before = (lock.read_bytes(), lock.stat().st_mtime_ns,
                      stat.S_IMODE(lock.stat().st_mode))
            parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with mock.patch.object(remote, "LOCK_PATH", str(lock)):
                    fd, identity = remote._open_existing_host_lock(parent_fd)
                    try:
                        self.assertEqual(identity["file"]["inode"], lock.stat().st_ino)
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        observed = remote._fuser(str(lock))
                        self.assertEqual(observed["pids"], [os.getpid()])
                        self.assertFalse(observed["stderr_empty"])
                        self.assertEqual(observed["stderr_sha256"],
                                         remote._held_lock_fuser_stderr_sha256())
                        # The frozen launcher opens the same existing inode
                        # for append before taking its exclusive flock.
                        other = os.open(lock, os.O_WRONLY | os.O_APPEND |
                                        os.O_NOFOLLOW)
                        try:
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        finally:
                            os.close(other)
                    finally:
                        os.close(fd)
                    again = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
                    try:
                        fcntl.flock(again, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(again)
            finally:
                os.close(parent_fd)
            self.assertEqual((lock.read_bytes(), lock.stat().st_mtime_ns,
                              stat.S_IMODE(lock.stat().st_mode)), before)

    def test_missing_untrusted_or_contended_lock_holds_without_create(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            lock = local_lock(root)
            parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                lock.unlink()
                with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                    remote._open_existing_host_lock(parent_fd)
                self.assertFalse(lock.exists(), "lock open must never create the file")
                lock.write_bytes(b"existing-lock-sentinel")
                lock.chmod(0o660)
                with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                    remote._open_existing_host_lock(parent_fd)
                lock.chmod(0o600)
                host_dir = lock.parent
                host_dir.chmod(0o775)
                with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                    remote._open_existing_host_lock(parent_fd)
                host_dir.chmod(0o700)
                lock.unlink()
                lock.symlink_to(root / "other")
                with self.assertRaisesRegex(RuntimeError, "absent or unsafe"):
                    remote._open_existing_host_lock(parent_fd)
                lock.unlink()
                lock.write_bytes(b"existing-lock-sentinel")
                lock.chmod(0o600)
                with mock.patch.object(remote, "_immediate_activity_gate",
                                       return_value={"utc": fixtures.utc(-2)}):
                    held_fd = os.open(lock, os.O_WRONLY | os.O_APPEND |
                                      os.O_NOFOLLOW)
                    try:
                        fcntl.flock(held_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        pin = remote._lock_identity_digest(
                            remote._host_lock_identity(parent_fd))
                        with self.assertRaisesRegex(RuntimeError, "held or uncertain"):
                            remote._final_creation_gate(parent_fd, pin)
                    finally:
                        os.close(held_fd)
                self.assertEqual(lock.read_bytes(), b"existing-lock-sentinel")
            finally:
                os.close(parent_fd)

    def test_final_label_catches_file_created_after_last_slow_gate(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            with simulated_writer(root) as (pin, lock_pin, target, _, _):
                original_gate = remote._activity_gate_snapshot
                label = root / ("late-" + remote.LABEL)

                def late_label(held_lock_pid=None):
                    result = original_gate(held_lock_pid)
                    if held_lock_pid is not None:
                        label.write_text("after all slow checks", encoding="utf-8")
                    return result

                with mock.patch.object(remote, "_activity_gate_snapshot",
                                       side_effect=late_label):
                    with self.assertRaisesRegex(RuntimeError, "science label"):
                        remote.write(io.BytesIO(b""), pin, lock_pin)
                self.assertTrue(label.exists())
                self.assertFalse(target.exists())

    def test_stop_process_or_neuron_change_in_last_activity_gate_blocks(self):
        for kind in ("STOP", "process", "Neuron"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(
                    dir="/dev/shm") as directory:
                root = Path(directory)
                activity = {"kind": kind, "active": False}
                with simulated_writer(root, activity) as (pin, lock_pin, target, _, _):
                    original_gate = remote._activity_gate_snapshot

                    def activate(held_lock_pid=None):
                        if held_lock_pid is not None:
                            activity["active"] = True
                        return original_gate(held_lock_pid)

                    with mock.patch.object(remote, "_activity_gate_snapshot",
                                           side_effect=activate):
                        with self.assertRaisesRegex(RuntimeError, kind):
                            remote.write(io.BytesIO(b""), pin, lock_pin)
                    self.assertTrue(activity["active"])
                    self.assertFalse(target.exists())

    def test_partial_target_retained_and_same_path_not_retried(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            with simulated_writer(root) as (pin, lock_pin, target, lock, _):
                payload = b"short-offline-capsule"
                with mock.patch.object(remote, "BYTES", len(payload) + 1):
                    with self.assertRaisesRegex(RuntimeError, "partial target retained"):
                        remote.write(io.BytesIO(payload), pin, lock_pin)
                    before = target.read_bytes()
                    self.assertEqual(before, payload)
                    with self.assertRaisesRegex(RuntimeError, "target"):
                        remote.write(io.BytesIO(payload), pin, lock_pin)
                    self.assertEqual(target.read_bytes(), before)
                    self.assertEqual(lock.read_bytes(), b"existing-lock-sentinel")

    def test_receipt_binds_final_label_lock_readback_and_postflight(self):
        pre, write, verify, post = clean_receipts()
        self.assertEqual(transport.validate_probe(pre, before_write=True), pre)
        self.assertEqual(transport.compare_preflight_to_write(pre, write)["status"],
                         "PREFLIGHT_TO_FINAL_LOCKED_LABEL_GATE_PASS")
        persistence = transport.compare_sessions(pre, write, verify)
        self.assertEqual(persistence["status"], "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        self.assertEqual(transport.compare_postflight(pre, post,
                         persistence=persistence)["status"], "READ_ONLY_POSTFLIGHT_PASS")
        for mutation in ("label_missing", "label_before_activity", "lock_missing",
                         "lock_not_self", "lock_stderr_drift", "prewrite_lock_changed"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(write)
                if mutation == "label_missing":
                    bad["final_creation_gate"].pop("science_label_scan")
                elif mutation == "label_before_activity":
                    activity_at = datetime.datetime.fromisoformat(
                        bad["final_creation_gate"]["utc"])
                    early = (activity_at - datetime.timedelta(seconds=1)).isoformat()
                    bad["final_creation_gate"]["science_label_scan"]["started_utc"] = early
                    bad["science_label_scan"]["started_utc"] = early
                elif mutation == "lock_missing":
                    bad["final_creation_gate"].pop("host_lock")
                elif mutation == "lock_not_self":
                    bad["final_creation_gate"]["shared_lock_fuser"]["pids"] = [999]
                    bad["final_creation_gate"]["host_lock"]["fuser"]["pids"] = [999]
                    bad["host_lock_fuser"]["pids"] = [999]
                elif mutation == "lock_stderr_drift":
                    bad["final_creation_gate"]["shared_lock_fuser"]["stderr_sha256"] = EMPTY
                    bad["final_creation_gate"]["host_lock"]["fuser"]["stderr_sha256"] = EMPTY
                    bad["host_lock_fuser"]["stderr_sha256"] = EMPTY
                else:
                    bad["prewrite_probe"]["host_lock_identity"]["file"]["inode"] += 1
                with self.assertRaises(RuntimeError):
                    transport.compare_preflight_to_write(pre, bad)
        bad_verify = copy.deepcopy(verify)
        bad_verify.pop("science_label_scan")
        with self.assertRaisesRegex(RuntimeError, "science-label"):
            transport.compare_sessions(pre, write, bad_verify)
        bad_post = copy.deepcopy(post)
        bad_post["host_lock_identity"]["file"]["inode"] += 1
        with self.assertRaisesRegex(RuntimeError, "lock"):
            transport.compare_postflight(pre, bad_post, persistence=persistence)
        bad_post = copy.deepcopy(post)
        bad_post["science_label_scan"]["matches"] = [remote.ROOT + "/late-" + remote.LABEL]
        with self.assertRaisesRegex(RuntimeError, "label"):
            transport.compare_postflight(pre, bad_post, persistence=persistence)

    def test_literal_r3_argv_adds_lock_identity_pin_without_ssh(self):
        pre = decorated_probe()
        with mock.patch.object(transport, "checked_ssh_identity", return_value={}):
            argv = transport.build_ssh_argv("write", preflight=pre)
        words = ["exec", "python3", "-c", transport.reviewed_remote_source(),
                 "write", remote._identity_digest(pre["identity_after"]),
                 remote._lock_identity_digest(pre["host_lock_identity"])]
        self.assertEqual(shlex.split(argv[-1]), words)
        self.assertEqual(argv[-2], transport.ENDPOINT)
        with self.assertRaisesRegex(RuntimeError, "validated fresh preflight"):
            transport.build_ssh_argv("write")


if __name__ == "__main__":
    unittest.main()
