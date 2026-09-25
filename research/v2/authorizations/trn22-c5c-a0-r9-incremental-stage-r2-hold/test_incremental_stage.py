"""Offline-only trn22 r2 linked-source and one-shot tests; no host use."""

from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remote = load("trn22_r9_incremental_r2_remote_test", HERE / "remote_incremental.py")
operator = load("trn22_r9_incremental_r2_operator_test", HERE / "operator_once.py")


def git(repo, *args):
    env = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_ALLOW_PROTOCOL": "file"}
    result = subprocess.run(["/usr/bin/git", "-C", str(repo), *args],
                            env=env, capture_output=True, text=True, check=False)
    if result.returncode:
        raise AssertionError("test fixture Git failed: " + " ".join(args) +
                             " " + result.stderr)
    return result.stdout.strip()


class LinkedFixture:
    def __init__(self, root):
        self.root = root
        self.upstream = root / "upstream"
        self.source = root / "accepted-e65-linked"
        self.target = root / "new-r9"
        self.bundle = root / "delta.bundle"
        self.upstream.mkdir()
        git(self.upstream, "init", "-q", "--initial-branch=main")
        git(self.upstream, "config", "user.name", "Synthetic")
        git(self.upstream, "config", "user.email", "synthetic@example.invalid")
        (self.upstream / "train.py").write_text("base model\n", encoding="utf-8")
        (self.upstream / "prepare.py").write_text("fixed evaluator\n", encoding="utf-8")
        git(self.upstream, "add", "train.py", "prepare.py")
        git(self.upstream, "commit", "-qm", "base")
        self.base = git(self.upstream, "rev-parse", "HEAD")
        git(self.upstream, "worktree", "add", "-q", "--detach", str(self.source), self.base)
        os.chmod(self.source, 0o775)
        os.chmod(self.source / ".git", 0o664)
        self.gitdir = Path(git(self.source, "rev-parse", "--absolute-git-dir"))
        os.chmod(self.gitdir, 0o775)
        (self.upstream / "research.txt").write_text("safe child\n", encoding="utf-8")
        git(self.upstream, "add", "research.txt")
        git(self.upstream, "commit", "-qm", "child")
        self.head = git(self.upstream, "rev-parse", "HEAD")
        git(self.upstream, "bundle", "create", str(self.bundle), "HEAD", "^" + self.base)
        self.base_files = {name: hashlib.sha256((self.source / name).read_bytes()).hexdigest()
                           for name in ("train.py", "prepare.py")}
        self.target_files = {**self.base_files,
                             "research.txt": hashlib.sha256(b"safe child\n").hexdigest()}

    def patches(self):
        marker = self.source / ".git"
        return {
            "SOURCE": self.source,
            "TARGET": self.target,
            "BUNDLE": self.bundle,
            "BASE_COMMIT": self.base,
            "TARGET_COMMIT": self.head,
            "DELTA_COUNT": 1,
            "BASE_FILES": self.base_files,
            "TARGET_FILES": self.target_files,
            "DEV": os.lstat(self.source).st_dev,
            "UID": os.getuid(),
            "GID": os.getgid(),
            "SOURCE_INO": os.lstat(self.source).st_ino,
            "SOURCE_MODE": 0o775,
            "SOURCE_NLINK": os.lstat(self.source).st_nlink,
            "GIT_MARKER_INO": os.lstat(marker).st_ino,
            "GIT_MARKER_MODE": 0o664,
            "GIT_MARKER_BYTES": marker.stat().st_size,
            "GIT_MARKER_SHA256": hashlib.sha256(marker.read_bytes()).hexdigest(),
            "GIT_DIR": self.gitdir,
            "GIT_DIR_INO": os.lstat(self.gitdir).st_ino,
            "GIT_DIR_MODE": 0o775,
            "BUNDLE_BYTES": self.bundle.stat().st_size,
            "BUNDLE_SHA256": hashlib.sha256(self.bundle.read_bytes()).hexdigest(),
        }


class IncrementalStageTests(unittest.TestCase):
    def test_checked_in_real_bundle_and_science_identity(self):
        data = operator.BUNDLE.read_bytes()
        self.assertEqual(len(data), remote.BUNDLE_BYTES)
        self.assertEqual(hashlib.sha256(data).hexdigest(), remote.BUNDLE_SHA256)
        self.assertEqual(git(HERE.parents[3], "bundle", "list-heads",
                             str(operator.BUNDLE)),
                         remote.TARGET_COMMIT + " HEAD")
        self.assertEqual(git(HERE.parents[3], "merge-base", remote.BASE_COMMIT,
                             remote.TARGET_COMMIT), remote.BASE_COMMIT)
        self.assertEqual(git(HERE.parents[3], "rev-list", "--count",
                             remote.BASE_COMMIT + ".." + remote.TARGET_COMMIT), "14")
        for name, digest in remote.BASE_FILES.items():
            self.assertEqual(hashlib.sha256((HERE.parents[3] / name).read_bytes()).hexdigest(),
                             digest)
        for name, digest in remote.TARGET_FILES.items():
            self.assertEqual(hashlib.sha256((HERE.parents[3] / name).read_bytes()).hexdigest(),
                             digest)

    def test_synthetic_group_writable_linked_source_and_new_checkout(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r9-linked-") as temp:
            fixture = LinkedFixture(Path(temp))
            with ExitStack() as stack:
                for key, value in fixture.patches().items():
                    stack.enter_context(mock.patch.object(remote, key, value))
                stack.enter_context(mock.patch.object(remote, "_safe_chain",
                                                     side_effect=lambda path: {str(path): "checked"}))
                stack.enter_context(mock.patch.object(remote, "_acl_absent"))
                stack.enter_context(mock.patch.object(remote, "_group_isolation",
                                                     return_value={"bounded": True}))
                before = git(fixture.source, "status", "--porcelain=v1")
                self.assertEqual(remote._source()["commit"], fixture.base)
                fixture.target.mkdir(mode=0o700)
                target = remote._reconstruct()
                self.assertEqual(target["commit"], fixture.head)
                self.assertTrue(target["clean"])
                self.assertEqual(git(fixture.source, "rev-parse", "HEAD"), fixture.base)
                self.assertEqual(git(fixture.source, "status", "--porcelain=v1"), before)
                self.assertFalse((fixture.target / ".git/objects/info/alternates").exists())

    def test_linked_pointer_mode_hash_or_gitdir_drift_holds(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r9-pointer-") as temp:
            fixture = LinkedFixture(Path(temp))
            with ExitStack() as stack:
                for key, value in fixture.patches().items():
                    stack.enter_context(mock.patch.object(remote, key, value))
                stack.enter_context(mock.patch.object(remote, "_safe_chain",
                                                     side_effect=lambda path: {str(path): "checked"}))
                stack.enter_context(mock.patch.object(remote, "_acl_absent"))
                stack.enter_context(mock.patch.object(remote, "_group_isolation",
                                                     return_value={"bounded": True}))
                self.assertEqual(remote._source()["commit"], fixture.base)
                with mock.patch.object(remote, "GIT_MARKER_SHA256", "0" * 64):
                    with self.assertRaises(RuntimeError):
                        remote._source()
                with mock.patch.object(remote, "GIT_DIR_INO", 999999):
                    with self.assertRaises(RuntimeError):
                        remote._source()
                os.chmod(fixture.source / ".git", 0o666)
                with self.assertRaises(RuntimeError):
                    remote._source()

    def test_group_and_acl_unknown_or_extra_member_holds(self):
        account = types.SimpleNamespace(pw_uid=remote.UID, pw_gid=remote.GID)
        group = types.SimpleNamespace(gr_mem=["other"])
        with mock.patch.object(remote.pwd, "getpwnam", return_value=account), \
             mock.patch.object(remote.pwd, "getpwall", return_value=[
                 types.SimpleNamespace(pw_name="ubuntu", pw_uid=remote.UID,
                                       pw_gid=remote.GID)]), \
             mock.patch.object(remote.grp, "getgrgid", return_value=group):
            with self.assertRaises(RuntimeError):
                remote._group_isolation()
        with tempfile.TemporaryDirectory(prefix="trn22-r9-acl-") as temp:
            path = Path(temp) / "file"
            path.write_bytes(b"x")
            with mock.patch.object(remote.os, "getxattr", return_value=b"grant"):
                with self.assertRaises(RuntimeError):
                    remote._acl_absent(path, directory=False)
            with mock.patch.object(remote.os, "getxattr", side_effect=OSError(5, "unknown")):
                with self.assertRaises(RuntimeError):
                    remote._acl_absent(path, directory=False)

    def test_fifo_lock_refuses_without_blocking(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r9-fifo-") as temp:
            cache = Path(temp) / ".cache"
            directory = cache / "trainium_freshstart"
            directory.mkdir(parents=True)
            fifo = directory / "device0.lock"
            os.mkfifo(fifo)
            with mock.patch.object(remote, "LOCK_CACHE", cache), \
                 mock.patch.object(remote, "LOCK_DIR", directory), \
                 mock.patch.object(remote, "LOCK", fifo):
                with self.assertRaises(RuntimeError):
                    with remote._host_lock():
                        self.fail("FIFO lock was acquired")

    def test_existing_stage_target_refuses_before_write(self):
        data = operator.BUNDLE.read_bytes()
        with tempfile.TemporaryDirectory(prefix="trn22-r9-exists-") as temp:
            root = Path(temp)
            bundle = root / "bundle"
            target = root / "target"
            target.mkdir()
            with mock.patch.object(remote, "ROOT", root), \
                 mock.patch.object(remote, "BUNDLE", bundle), \
                 mock.patch.object(remote, "TARGET", target):
                with self.assertRaises(RuntimeError):
                    remote._write_bundle(data)
            self.assertFalse(bundle.exists())
            self.assertTrue(target.is_dir())

    def test_both_space_endpoints_enforce_inode_floor(self):
        good = types.SimpleNamespace(f_bavail=20_000_000, f_frsize=1024,
                                     f_favail=remote.MIN_INODES)
        low_inode = types.SimpleNamespace(f_bavail=20_000_000, f_frsize=1024,
                                          f_favail=9_999)
        with mock.patch.object(remote.os, "statvfs", side_effect=[good, low_inode]):
            self.assertEqual(remote._space(remote.PRE_FREE)["root_free_inodes"],
                             remote.MIN_INODES)
            with self.assertRaises(RuntimeError):
                remote._space(remote.PRE_FREE)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(remote.socket, "gethostname",
                                                 return_value=remote.HOST))
            stack.enter_context(mock.patch.object(remote.os, "geteuid",
                                                 return_value=remote.UID))
            for name in ("_source", "_mount_identity", "_protected", "_label_scan"):
                stack.enter_context(mock.patch.object(remote, name, return_value={}))
            stack.enter_context(mock.patch.object(remote, "_activity", return_value={}))
            space = stack.enter_context(mock.patch.object(remote, "_space",
                side_effect=[{"root_free_inodes": remote.MIN_INODES},
                             RuntimeError("synthetic after-scan 9999 inodes")]))
            with self.assertRaisesRegex(RuntimeError, "9999 inodes"):
                remote._probe({})
            self.assertEqual(space.call_count, 2)

    def test_stop_appearing_during_label_scan_holds(self):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(remote.socket, "gethostname",
                                                 return_value=remote.HOST))
            stack.enter_context(mock.patch.object(remote.os, "geteuid",
                                                 return_value=remote.UID))
            for name in ("_space", "_source", "_mount_identity", "_protected",
                         "_label_scan"):
                stack.enter_context(mock.patch.object(remote, name, return_value={}))
            activity = stack.enter_context(mock.patch.object(remote, "_activity",
                side_effect=[{}, RuntimeError("synthetic STOP during scan")]))
            with self.assertRaisesRegex(RuntimeError, "STOP during scan"):
                remote._probe({})
            self.assertEqual(activity.call_count, 2)

    def test_disabled_entry_and_missing_authority_precede_contact(self):
        hold = json.loads((HERE / "REQUEST.HOLD.json").read_text(encoding="utf-8"))
        self.assertIs(hold["dispatch_authorized"], False)
        self.assertIs(hold["source_stage_authorized"], False)
        self.assertIs(hold["native_training_authorized"], False)
        self.assertEqual(hold["bundle_sha256"], remote.BUNDLE_SHA256)
        for name, field in (("remote_incremental.py", "remote_source_sha256"),
                            ("operator_once.py", "operator_source_sha256")):
            self.assertEqual(hold[field], hashlib.sha256((HERE / name).read_bytes()).hexdigest())
        for path in (operator.ENABLED_REQUEST, operator.FINAL_REVIEW, operator.ASSIGNMENT):
            self.assertFalse(path.exists())
        with tempfile.TemporaryDirectory(prefix="trn22-r9-authority-") as temp:
            absent = Path(temp) / "enabled.json"
            marker = Path(temp) / "marker"
            fake = mock.Mock(side_effect=AssertionError("contacted"))
            with mock.patch.object(operator, "ENABLED_REQUEST", absent), \
                 mock.patch.object(operator, "MARKER", marker):
                with self.assertRaises((FileNotFoundError, RuntimeError)):
                    operator.stage_once(remote_call=fake)
            self.assertFalse(marker.exists())
            fake.assert_not_called()
        result = subprocess.run(["/usr/bin/python3", "-B", str(HERE / "operator_once.py")],
                                capture_output=True, text=True, timeout=10, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HOLD", result.stderr)

    def test_absolute_timeout_gate_blocks_fake_contact(self):
        fake = mock.Mock()
        with mock.patch.object(operator, "_local_stop_and_breakers"), \
             mock.patch.object(operator, "_ssh_identity"), \
             mock.patch.object(operator.subprocess, "run", return_value=types.SimpleNamespace(
                 returncode=0, stderr=b"", stdout=b"fake timeout\n")) as run:
            with self.assertRaises(RuntimeError):
                operator._ssh_phase("preflight", None, b"", b"# empty\n")
            self.assertEqual(run.call_args.args[0][0], "/usr/bin/timeout")
        fake.assert_not_called()

    def test_forged_enabled_request_holds_before_marker_or_contact(self):
        contact = mock.Mock(side_effect=AssertionError("remote contact"))
        with mock.patch.object(operator, "_json", side_effect=[
                ({"dispatch_authorized": True}, "forged-request"),
                ({"verdict": "PASS_SOURCE_STAGE_ONLY"}, "forged-review"),
                ({"one_attempt": True}, "forged-assignment")]), \
             mock.patch.object(operator, "_source_bytes", return_value=(b"bundle", b"remote")), \
             mock.patch.object(operator, "_regular", return_value=b"operator"), \
             mock.patch.object(operator, "_marker", side_effect=AssertionError("marker")):
            with self.assertRaises(RuntimeError):
                operator.stage_once(remote_call=contact)
        contact.assert_not_called()

    def test_prior_r1_marker_blocks_same_label_retry(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r9-r2-prior-") as temp:
            root = Path(temp)
            prior, current = root / "r1.one-attempt", root / "r2.one-attempt"
            prior.write_text("consumed\n", encoding="ascii")
            with mock.patch.object(operator, "PRIOR_MARKER", prior), \
                 mock.patch.object(operator, "MARKER", current):
                with self.assertRaises(RuntimeError):
                    operator._marker("request", "review", "assignment")
            self.assertFalse(current.exists())

    def test_write_failure_still_attempts_postflight_and_keeps_marker(self):
        if sys_flags_optimize():
            with self.assertRaises(RuntimeError):
                operator.stage_once(remote_call=mock.Mock())
            return
        with tempfile.TemporaryDirectory(prefix="trn22-r9-once-") as temp:
            root = Path(temp)
            marker = root / "marker"
            calls = []
            source = {"identity": "same"}
            protected = {"checkpoint": "same"}
            def fake_remote(phase, prior, data, remote_source):
                calls.append(phase)
                if phase == "write":
                    raise RuntimeError("synthetic partial write")
                return {"phase": phase, "snapshot": {
                    "bundle_exists": False, "target_exists": False,
                    "source": source, "protected": protected}}
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(operator, "ENABLED_REQUEST", root / "enabled"))
                stack.enter_context(mock.patch.object(operator, "MARKER", marker))
                stack.enter_context(mock.patch.object(operator, "_json", side_effect=[
                    ({"dispatch_authorized": True}, "requesthash"), ({}, "reviewhash"),
                    ({}, "assignmenthash")]))
                stack.enter_context(mock.patch.object(operator, "_source_bytes",
                                                     return_value=(b"bundle", b"remote")))
                stack.enter_context(mock.patch.object(operator, "_regular", return_value=b"operator"))
                stack.enter_context(mock.patch.object(operator, "_authority"))
                stack.enter_context(mock.patch.object(operator, "_local_stop_and_breakers"))
                stack.enter_context(mock.patch.object(operator, "_ssh_identity"))
                with self.assertRaises(RuntimeError):
                    operator.stage_once(remote_call=fake_remote)
            self.assertEqual(calls, ["preflight", "write", "postflight"])
            self.assertTrue(marker.exists())
            self.assertIn(b"UNCONDITIONAL_FULL_POSTFLIGHT", marker.read_bytes())


def sys_flags_optimize():
    import sys
    return bool(sys.flags.optimize)


if __name__ == "__main__":
    unittest.main()
