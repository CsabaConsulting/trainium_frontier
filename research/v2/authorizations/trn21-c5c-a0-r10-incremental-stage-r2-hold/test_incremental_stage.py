"""Offline-only private-shallow-source and one-shot tests; no SSH or host use."""

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


remote = load("trn21_r10_incremental_remote_test", HERE / "remote_incremental.py")
operator = load("trn21_r10_incremental_operator_test", HERE / "operator_once.py")


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


class PrivateShallowFixture:
    def __init__(self, root):
        self.root = root
        self.upstream = root / "upstream"
        self.source = root / "accepted-e65-private"
        self.target = root / "new-r10"
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
        git(root, "clone", "-q", "--depth=1", "--no-local", "--branch=main",
            "file://" + str(self.upstream), str(self.source))
        git(self.source, "remote", "remove", "origin")
        git(self.source, "checkout", "-q", "--detach", self.base)
        os.chmod(self.source, 0o700)
        self.gitdir = Path(git(self.source, "rev-parse", "--absolute-git-dir"))
        os.chmod(self.gitdir, 0o700)
        for name in ("HEAD", "config", "shallow"):
            os.chmod(self.gitdir / name, 0o600)
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
        git_controls = {}
        for name in ("HEAD", "config", "shallow"):
            path = self.gitdir / name
            git_controls[name] = (path.stat().st_ino, path.stat().st_size,
                                  hashlib.sha256(path.read_bytes()).hexdigest())
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
            "SOURCE_MODE": 0o700,
            "SOURCE_NLINK": os.lstat(self.source).st_nlink,
            "GIT_DIR": self.gitdir,
            "GIT_DIR_INO": os.lstat(self.gitdir).st_ino,
            "GIT_DIR_MODE": 0o700,
            "GIT_DIR_NLINK": os.lstat(self.gitdir).st_nlink,
            "GIT_HEAD_INO": git_controls["HEAD"][0],
            "GIT_HEAD_BYTES": git_controls["HEAD"][1],
            "GIT_HEAD_SHA256": git_controls["HEAD"][2],
            "GIT_CONFIG_INO": git_controls["config"][0],
            "GIT_CONFIG_BYTES": git_controls["config"][1],
            "GIT_CONFIG_SHA256": git_controls["config"][2],
            "GIT_SHALLOW_INO": git_controls["shallow"][0],
            "GIT_SHALLOW_BYTES": git_controls["shallow"][1],
            "GIT_SHALLOW_SHA256": git_controls["shallow"][2],
            "BUNDLE_BYTES": self.bundle.stat().st_size,
            "BUNDLE_SHA256": hashlib.sha256(self.bundle.read_bytes()).hexdigest(),
        }


class IncrementalStageTests(unittest.TestCase):
    def test_checked_in_real_bundle_and_science_identity(self):
        data = (HERE / "e65-to-r10.bundle").read_bytes()
        self.assertEqual(len(data), remote.BUNDLE_BYTES)
        self.assertEqual(hashlib.sha256(data).hexdigest(), remote.BUNDLE_SHA256)
        self.assertEqual(git(HERE.parents[3], "bundle", "list-heads",
                             str(HERE / "e65-to-r10.bundle")),
                         remote.TARGET_COMMIT + " HEAD")
        self.assertEqual(git(HERE.parents[3], "merge-base", remote.BASE_COMMIT,
                             remote.TARGET_COMMIT), remote.BASE_COMMIT)
        self.assertEqual(git(HERE.parents[3], "rev-list", "--count",
                             remote.BASE_COMMIT + ".." + remote.TARGET_COMMIT), "15")
        for name, digest in remote.BASE_FILES.items():
            self.assertEqual(hashlib.sha256((HERE.parents[3] / name).read_bytes()).hexdigest(),
                             digest)
        for name, digest in remote.TARGET_FILES.items():
            self.assertEqual(hashlib.sha256((HERE.parents[3] / name).read_bytes()).hexdigest(),
                             digest)

    def test_synthetic_private_shallow_source_and_new_checkout(self):
        with tempfile.TemporaryDirectory(prefix="trn21-r10-private-") as temp:
            fixture = PrivateShallowFixture(Path(temp))
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

    def test_private_source_mode_hash_or_gitdir_drift_holds(self):
        with tempfile.TemporaryDirectory(prefix="trn21-r10-source-") as temp:
            fixture = PrivateShallowFixture(Path(temp))
            with ExitStack() as stack:
                for key, value in fixture.patches().items():
                    stack.enter_context(mock.patch.object(remote, key, value))
                stack.enter_context(mock.patch.object(remote, "_safe_chain",
                                                     side_effect=lambda path: {str(path): "checked"}))
                stack.enter_context(mock.patch.object(remote, "_acl_absent"))
                stack.enter_context(mock.patch.object(remote, "_group_isolation",
                                                     return_value={"bounded": True}))
                self.assertEqual(remote._source()["commit"], fixture.base)
                with mock.patch.object(remote, "GIT_HEAD_SHA256", "0" * 64):
                    with self.assertRaises(RuntimeError):
                        remote._source()
                with mock.patch.object(remote, "GIT_DIR_INO", 999999):
                    with self.assertRaises(RuntimeError):
                        remote._source()
                os.chmod(fixture.source / ".git", 0o775)
                with self.assertRaises(RuntimeError):
                    remote._source()
                os.chmod(fixture.source / ".git", 0o700)
                os.chmod(fixture.source, 0o775)
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
        with tempfile.TemporaryDirectory(prefix="trn21-r10-acl-") as temp:
            path = Path(temp) / "file"
            path.write_bytes(b"x")
            with mock.patch.object(remote.os, "getxattr", return_value=b"grant"):
                with self.assertRaises(RuntimeError):
                    remote._acl_absent(path, directory=False)
            with mock.patch.object(remote.os, "getxattr", side_effect=OSError(5, "unknown")):
                with self.assertRaises(RuntimeError):
                    remote._acl_absent(path, directory=False)

    def test_protected_checkpoint_stage_file_and_cache_identity_drift_hold(self):
        with tempfile.TemporaryDirectory(prefix="trn21-r10-protected-") as temp:
            root = Path(temp)
            checkpoint = root / "retained/final.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"retained checkpoint")
            preserved = root / "e65-stage.tar.gz"
            preserved.write_bytes(b"preserved stage file")
            cache = root / "caches/train"
            (cache / "compiler").mkdir(parents=True)
            pinned_checkpoint = (checkpoint.stat().st_ino, checkpoint.stat().st_size,
                                 hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            pinned_preserved = (preserved.stat().st_ino, preserved.stat().st_size,
                                hashlib.sha256(preserved.read_bytes()).hexdigest())
            with mock.patch.object(remote, "DEV", root.stat().st_dev), \
                 mock.patch.object(remote, "UID", os.getuid()), \
                 mock.patch.object(remote, "GID", os.getgid()), \
                 mock.patch.object(remote, "CHECKPOINTS", {checkpoint: pinned_checkpoint}), \
                 mock.patch.object(remote, "PRESERVED_STAGE_FILES", {preserved: pinned_preserved}), \
                 mock.patch.object(remote, "CACHE_ROOT", cache), \
                 mock.patch.object(remote, "CACHE_ROOT_INO", cache.stat().st_ino), \
                 mock.patch.object(remote, "CACHE_INODES", {
                     "compiler": (cache / "compiler").stat().st_ino}), \
                 mock.patch.object(remote, "_safe_chain", return_value={"safe": True}), \
                 mock.patch.object(remote, "_acl_absent"):
                self.assertEqual(remote._protected()["checkpoints"][str(checkpoint)]["sha256"],
                                 pinned_checkpoint[2])
                checkpoint.write_bytes(b"changed checkpoint")
                with self.assertRaisesRegex(RuntimeError, "checkpoint length changed"):
                    remote._protected()
                checkpoint.write_bytes(b"retained checkpoint")
                preserved.write_bytes(b"tampered stage file!")
                with self.assertRaisesRegex(RuntimeError, "file bytes/hash changed"):
                    remote._protected()
                preserved.write_bytes(b"preserved stage file")
                with mock.patch.object(remote, "CACHE_INODES", {"compiler": 0}):
                    with self.assertRaisesRegex(RuntimeError, "drift"):
                        remote._protected()

    def test_fifo_lock_refuses_without_blocking(self):
        with tempfile.TemporaryDirectory(prefix="trn21-r10-fifo-") as temp:
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
        data = (HERE / "e65-to-r10.bundle").read_bytes()
        with tempfile.TemporaryDirectory(prefix="trn21-r10-exists-") as temp:
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

    def test_stop_inserted_during_final_slow_activity_prevents_first_write(self):
        data = (HERE / "e65-to-r10.bundle").read_bytes()
        with tempfile.TemporaryDirectory(prefix="trn21-r10-stop-race-") as temp:
            root = Path(temp)
            stop, bundle, target = root / "STOP", root / "bundle", root / "target"
            def late_stop():
                stop.write_text("hold", encoding="ascii")
                return {}
            with mock.patch.object(remote, "ROOT", root), \
                 mock.patch.object(remote, "STOP_PATHS", [stop]), \
                 mock.patch.object(remote, "BUNDLE", bundle), \
                 mock.patch.object(remote, "TARGET", target), \
                 mock.patch.object(remote, "_activity", side_effect=late_stop), \
                 mock.patch.object(remote, "_label_scan", return_value={}), \
                 mock.patch.object(remote, "_space", return_value={
                     "root_free_bytes": remote.PRE_FREE + 1024,
                     "root_free_inodes": remote.MIN_INODES + 1}):
                with self.assertRaisesRegex(RuntimeError, "STOP"):
                    remote._write_bundle(data)
            self.assertFalse(bundle.exists())
            self.assertFalse(target.exists())

    def test_label_inserted_during_final_slow_activity_prevents_first_write(self):
        data = (HERE / "e65-to-r10.bundle").read_bytes()
        with tempfile.TemporaryDirectory(prefix="trn21-r10-label-race-") as temp:
            root = Path(temp)
            bundle, target = root / "bundle", root / "target"
            label = "synthetic-trn21-stage-label"
            def late_label():
                (root / ("late-" + label)).write_text("x", encoding="ascii")
                return {}
            with mock.patch.object(remote, "ROOT", root), \
                 mock.patch.object(remote, "STOP_PATHS", []), \
                 mock.patch.object(remote, "BUNDLE", bundle), \
                 mock.patch.object(remote, "TARGET", target), \
                 mock.patch.object(remote, "LABEL", label), \
                 mock.patch.object(remote, "_activity", side_effect=late_label):
                with self.assertRaisesRegex(RuntimeError, "label already appears"):
                    remote._write_bundle(data)
            self.assertFalse(bundle.exists())
            self.assertFalse(target.exists())

    def test_label_arriving_during_preflight_scan_hits_final_scan_before_open(self):
        data = (HERE / "e65-to-r10.bundle").read_bytes()
        with tempfile.TemporaryDirectory(prefix="trn21-r10-preflight-label-") as temp:
            root = Path(temp)
            bundle, target = root / "bundle", root / "target"
            label = "synthetic-trn21-preflight-late"
            nested = root / "other/deep"
            nested.mkdir(parents=True)
            duplicate = nested / ("late-" + label)
            calls = []

            def scan_with_late_duplicate():
                calls.append(1)
                if len(calls) == 1:
                    duplicate.write_bytes(b"persistent duplicate")
                    return {"label": label, "matches": []}
                if duplicate.exists():
                    raise RuntimeError("synthetic stage label already appears")
                return {"label": label, "matches": []}

            with mock.patch.object(remote, "ROOT", root), \
                 mock.patch.object(remote, "BUNDLE", bundle), \
                 mock.patch.object(remote, "TARGET", target), \
                 mock.patch.object(remote, "LABEL", label), \
                 mock.patch.object(remote, "STOP_PATHS", []), \
                 mock.patch.object(remote.socket, "gethostname", return_value=remote.HOST), \
                 mock.patch.object(remote.os, "geteuid", return_value=remote.UID), \
                 mock.patch.object(remote, "_source", return_value={}), \
                 mock.patch.object(remote, "_mount_identity", return_value={}), \
                 mock.patch.object(remote, "_protected", return_value={}), \
                 mock.patch.object(remote, "_activity", return_value={}), \
                 mock.patch.object(remote, "_label_scan", side_effect=scan_with_late_duplicate), \
                 mock.patch.object(remote, "_space", return_value={
                     "root_free_bytes": remote.PRE_FREE + 1024,
                     "root_free_inodes": remote.MIN_INODES + 1}):
                remote._probe({})
                with self.assertRaisesRegex(RuntimeError, "label already appears"):
                    remote._write_bundle(data)
            self.assertEqual(len(calls), 2)
            self.assertFalse(bundle.exists())
            self.assertFalse(target.exists())

    def test_label_inserted_behind_final_scan_cursor_holds_before_payload(self):
        data = (HERE / "e65-to-r10.bundle").read_bytes()
        with tempfile.TemporaryDirectory(prefix="trn21-r10-double-scan-") as temp:
            root = Path(temp)
            bundle, target = root / "bundle", root / "target"
            label = "synthetic-trn21-final-scan"
            original_scan = remote._label_scan
            calls = []
            def scan_then_insert():
                calls.append(1)
                if len(calls) == 1:
                    (root / ("late-" + label)).write_text("x", encoding="ascii")
                    return {"label": label, "matches": []}
                return original_scan()
            with mock.patch.object(remote, "ROOT", root), \
                 mock.patch.object(remote, "STOP_PATHS", []), \
                 mock.patch.object(remote, "BUNDLE", bundle), \
                 mock.patch.object(remote, "TARGET", target), \
                 mock.patch.object(remote, "LABEL", label), \
                 mock.patch.object(remote, "_activity", return_value={}), \
                 mock.patch.object(remote, "_space", return_value={
                     "root_free_bytes": remote.PRE_FREE + 1024,
                     "root_free_inodes": remote.MIN_INODES + 1}), \
                 mock.patch.object(remote, "_label_scan", side_effect=scan_then_insert):
                with self.assertRaisesRegex(RuntimeError, "label already appears"):
                    remote._write_bundle(data)
            self.assertEqual(len(calls), 2)
            # One final pre-open scan is not equivalent to the old double
            # scan: a late duplicate may create an empty provisional inode.
            # The post-open full scan must block payload and checkout, and
            # the provisional inode is retained without cleanup/retry.
            self.assertTrue(bundle.exists())
            self.assertEqual(bundle.read_bytes(), b"")
            self.assertFalse(target.exists())

    def test_stale_final_proof_prevents_first_write(self):
        data = (HERE / "e65-to-r10.bundle").read_bytes()
        with tempfile.TemporaryDirectory(prefix="trn21-r10-stale-proof-") as temp:
            root = Path(temp)
            bundle, target = root / "bundle", root / "target"
            with mock.patch.object(remote, "ROOT", root), \
                 mock.patch.object(remote, "STOP_PATHS", []), \
                 mock.patch.object(remote, "BUNDLE", bundle), \
                 mock.patch.object(remote, "TARGET", target), \
                 mock.patch.object(remote, "_activity", return_value={}), \
                 mock.patch.object(remote, "_label_scan", return_value={}), \
                 mock.patch.object(remote, "_space", return_value={
                     "root_free_bytes": remote.PRE_FREE + 1024,
                     "root_free_inodes": remote.MIN_INODES + 1}), \
                 mock.patch.object(remote.time, "monotonic",
                                   side_effect=[0.0, 0.0, 0.0, 0.0, 1.1]):
                with self.assertRaisesRegex(RuntimeError, "too old"):
                    remote._write_bundle(data)
            self.assertFalse(bundle.exists())
            self.assertFalse(target.exists())

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
        with tempfile.TemporaryDirectory(prefix="trn21-r10-authority-") as temp:
            absent = Path(temp) / "enabled.json"
            marker = Path(temp) / "marker"
            fake = mock.Mock(side_effect=AssertionError("contacted"))
            with mock.patch.object(operator, "ENABLED_REQUEST", absent), \
                 mock.patch.object(operator, "MARKER", marker), \
                 mock.patch.object(operator, "_ssh_identity") as identity, \
                 mock.patch.object(operator, "_remote_timeout_identity") as probe:
                with self.assertRaises((FileNotFoundError, RuntimeError)):
                    operator.stage_once(remote_call=fake)
            self.assertFalse(marker.exists())
            fake.assert_not_called()
            identity.assert_not_called()
            probe.assert_not_called()

    def test_missing_final_review_or_assignment_holds_before_probe(self):
        if sys_flags_optimize():
            with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                operator.stage_once(remote_call=mock.Mock())
            return
        for missing in ("review", "assignment"):
            with self.subTest(missing=missing):
                available = [({"dispatch_authorized": True}, "request-hash")]
                if missing == "assignment":
                    available.append(({}, "review-hash"))
                available.append(FileNotFoundError("synthetic missing " + missing))
                contact = mock.Mock(side_effect=AssertionError("remote phase contacted"))
                with mock.patch.object(operator, "_json", side_effect=available), \
                     mock.patch.object(operator, "_source_bytes",
                                       return_value=(b"bundle", b"remote")), \
                     mock.patch.object(operator, "_regular", return_value=b"operator"), \
                     mock.patch.object(operator, "_ssh_identity") as identity, \
                     mock.patch.object(operator, "_remote_timeout_identity") as probe, \
                     mock.patch.object(operator, "_marker") as marker:
                    with self.assertRaisesRegex(FileNotFoundError, "synthetic missing"):
                        operator.stage_once(remote_call=contact)
                identity.assert_not_called()
                probe.assert_not_called()
                marker.assert_not_called()
                contact.assert_not_called()

    def test_pinned_remote_timeout_probe_ignores_local_uutils(self):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            # The local timeout is uutils; invoking it would fail this test.
            self.assertEqual(argv[0], "/usr/bin/ssh")
            self.assertEqual(argv[-2], operator.ENDPOINT)
            self.assertIn("HostKeyAlgorithms=ssh-ed25519", argv)
            if len(calls) == 1:
                self.assertEqual(argv[-1], "/usr/bin/timeout --version")
                self.assertEqual(kwargs["timeout"], 20)
                return types.SimpleNamespace(returncode=0, stderr=b"",
                                             stdout=b"timeout (GNU coreutils) 9.1\n")
            self.assertTrue(argv[-1].startswith("/usr/bin/timeout --foreground "))
            return types.SimpleNamespace(returncode=0, stderr=b"",
                                         stdout=b'{"phase":"preflight"}')

        with mock.patch.object(operator, "_local_stop_and_breakers") as stop, \
             mock.patch.object(operator, "_ssh_identity"), \
             mock.patch.object(operator.subprocess, "run", side_effect=fake_run):
            self.assertEqual(operator._ssh_phase("preflight", None, b"", b"# empty\n"),
                             {"phase": "preflight"})
        self.assertEqual(len(calls), 2)
        self.assertEqual(stop.call_count, 3)

    def test_remote_timeout_mismatch_blocks_phase_before_contact(self):
        with mock.patch.object(operator, "_local_stop_and_breakers"), \
             mock.patch.object(operator, "_ssh_identity"), \
             mock.patch.object(operator.subprocess, "run", return_value=types.SimpleNamespace(
                 returncode=0, stderr=b"", stdout=b"timeout (uutils coreutils) 0.8.0\n")) as run:
            with self.assertRaisesRegex(RuntimeError, "remote absolute GNU timeout"):
                operator._ssh_phase("preflight", None, b"", b"# empty\n")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][-1], "/usr/bin/timeout --version")

    def test_stop_during_remote_timeout_probe_blocks_phase(self):
        with mock.patch.object(operator, "_local_stop_and_breakers", side_effect=[
                None, None, RuntimeError("late local STOP")]) as stop, \
             mock.patch.object(operator, "_ssh_identity"), \
             mock.patch.object(operator.subprocess, "run", return_value=types.SimpleNamespace(
                 returncode=0, stderr=b"", stdout=b"timeout (GNU coreutils) 9.1\n")) as run:
            with self.assertRaisesRegex(RuntimeError, "late local STOP"):
                operator._ssh_phase("preflight", None, b"", b"# empty\n")
        self.assertEqual(stop.call_count, 3)
        self.assertEqual(run.call_count, 1)

    def test_stop_during_pre_marker_probe_preserves_unconsumed_marker(self):
        if sys_flags_optimize():
            with mock.patch.object(operator, "_remote_timeout_identity") as probe:
                with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                    operator.stage_once(remote_call=mock.Mock())
            probe.assert_not_called()
            return
        with tempfile.TemporaryDirectory(prefix="trn21-r10-probe-stop-") as temp:
            marker = Path(temp) / "marker"
            contact = mock.Mock(side_effect=AssertionError("staging phase contacted"))
            with mock.patch.object(operator, "ENABLED_REQUEST", Path(temp) / "enabled"), \
                 mock.patch.object(operator, "MARKER", marker), \
                 mock.patch.object(operator, "_json", side_effect=[
                     ({"dispatch_authorized": True}, "request"), ({}, "review"),
                     ({}, "assignment")]), \
                 mock.patch.object(operator, "_source_bytes", return_value=(b"bundle", b"remote")), \
                 mock.patch.object(operator, "_regular", return_value=b"operator"), \
                 mock.patch.object(operator, "_authority"), \
                 mock.patch.object(operator, "_local_stop_and_breakers", side_effect=[
                     None, None, RuntimeError("late local STOP")]) as stop, \
                 mock.patch.object(operator, "_ssh_identity"), \
                 mock.patch.object(operator, "_remote_timeout_identity") as probe:
                with self.assertRaisesRegex(RuntimeError, "late local STOP"):
                    operator.stage_once(remote_call=contact)
            self.assertEqual(stop.call_count, 3)
            probe.assert_called_once_with()
            contact.assert_not_called()
            self.assertFalse(marker.exists())

    def test_stale_documents_state_cannot_override_tripped_desktop_before_ssh(self):
        self.assertEqual(operator.CONTROL_ROOT,
                         Path("/home/poojith-devan/Desktop/Trainium"))
        with tempfile.TemporaryDirectory(prefix="trn21-r10-active-controller-") as temp:
            root = Path(temp)
            desktop = root / "Desktop/Trainium"
            documents = root / "Documents/ChatGPT/Trainium"
            package = root / "stage-package"
            for checkout in (desktop, documents, package):
                (checkout / "research/v2").mkdir(parents=True)
            clear = {"launch_authorized": False,
                     "breakers": {"global_correctness": {"tripped": False},
                                  "trn21": {"tripped": False}},
                     "hosts": {"trn21": {"launch_allowed": False}}}
            tripped = json.loads(json.dumps(clear))
            tripped["breakers"]["trn21"]["tripped"] = True
            (documents / "research/v2/state.json").write_text(json.dumps(clear))
            (desktop / "research/v2/state.json").write_text(json.dumps(tripped))
            with mock.patch.object(operator, "CONTROL_ROOT", desktop), \
                 mock.patch.object(operator, "LEGACY_CONTROL_ROOT", documents), \
                 mock.patch.object(operator, "PACKAGE_ROOT", package), \
                 mock.patch.object(operator, "_ssh_identity") as identity, \
                 mock.patch.object(operator.subprocess, "run") as process:
                with self.assertRaisesRegex(RuntimeError, "controller STOP/breaker"):
                    operator._ssh_phase("preflight", None, b"", b"remote source")
            identity.assert_not_called()
            process.assert_not_called()

    def test_remote_absolute_timeout_rejects_fake_version(self):
        with mock.patch.object(remote.subprocess, "run", return_value=types.SimpleNamespace(
                returncode=0, stderr=b"", stdout=b"fake timeout\n")) as run:
            with self.assertRaisesRegex(RuntimeError, "GNU timeout"):
                remote._timeout_identity()
            self.assertEqual(run.call_args.args[0], ["/usr/bin/timeout", "--version"])

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

    def test_write_failure_still_attempts_postflight_and_keeps_marker(self):
        if sys_flags_optimize():
            with self.assertRaises(RuntimeError):
                operator.stage_once(remote_call=mock.Mock())
            return
        with tempfile.TemporaryDirectory(prefix="trn21-r10-once-") as temp:
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
                stack.enter_context(mock.patch.object(operator, "_remote_timeout_identity"))
                with self.assertRaises(RuntimeError):
                    operator.stage_once(remote_call=fake_remote)
            self.assertEqual(calls, ["preflight", "write", "postflight"])
            self.assertTrue(marker.exists())
            self.assertIn(b"FULL_POSTFLIGHT_COMPLETED", marker.read_bytes())


def sys_flags_optimize():
    import sys
    return bool(sys.flags.optimize)


if __name__ == "__main__":
    unittest.main()
