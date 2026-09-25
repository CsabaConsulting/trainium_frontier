"""Offline-only trn22 e65->r10 linked-source and one-shot tests; no host use."""

from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
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


remote = load("trn22_r10_incremental_r1_remote_test", HERE / "remote_incremental.py")
operator = load("trn22_r10_incremental_r1_operator_test", HERE / "operator_once.py")


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


def git_blob(repo, commit, path):
    result = subprocess.run(["/usr/bin/git", "-C", str(repo), "show",
                             commit + ":" + path],
                            env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent",
                                 "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
                                 "GIT_CONFIG_GLOBAL": "/dev/null"},
                            capture_output=True, check=False)
    if result.returncode:
        raise AssertionError("test fixture Git show failed: " + path)
    return result.stdout


class LinkedFixture:
    def __init__(self, root):
        self.root = root
        self.upstream = root / "upstream"
        self.source = root / "accepted-e65-linked"
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
                             remote.BASE_COMMIT + ".." + remote.TARGET_COMMIT), "15")
        for name, digest in remote.BASE_FILES.items():
            self.assertEqual(hashlib.sha256(git_blob(HERE.parents[3], remote.BASE_COMMIT,
                                                     name)).hexdigest(),
                             digest)
        for name, digest in remote.TARGET_FILES.items():
            self.assertEqual(hashlib.sha256(git_blob(HERE.parents[3], remote.TARGET_COMMIT,
                                                     name)).hexdigest(),
                             digest)

    def test_synthetic_group_writable_linked_source_and_new_checkout(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r10-linked-") as temp:
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
        with tempfile.TemporaryDirectory(prefix="trn22-r10-pointer-") as temp:
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
        with tempfile.TemporaryDirectory(prefix="trn22-r10-acl-") as temp:
            path = Path(temp) / "file"
            path.write_bytes(b"x")
            with mock.patch.object(remote.os, "getxattr", return_value=b"grant"):
                with self.assertRaises(RuntimeError):
                    remote._acl_absent(path, directory=False)
            with mock.patch.object(remote.os, "getxattr", side_effect=OSError(5, "unknown")):
                with self.assertRaises(RuntimeError):
                    remote._acl_absent(path, directory=False)

    def test_fifo_lock_refuses_without_blocking(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r10-fifo-") as temp:
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
        with tempfile.TemporaryDirectory(prefix="trn22-r10-exists-") as temp:
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

    def test_disabled_r2_request_and_missing_authority_precede_contact(self):
        hold = json.loads((HERE / "REQUEST.HOLD.json").read_text(encoding="utf-8"))
        self.assertIs(hold["dispatch_authorized"], False)
        self.assertIs(hold["source_stage_authorized"], False)
        self.assertIs(hold["native_training_authorized"], False)
        self.assertIs(hold["evaluation_authorized"], False)
        self.assertEqual(hold["source_commit"], remote.BASE_COMMIT)
        self.assertEqual(hold["target_commit"], remote.TARGET_COMMIT)
        self.assertEqual(hold["source_path"], str(remote.SOURCE))
        self.assertEqual(hold["new_target_path"], str(remote.TARGET))
        self.assertEqual(hold["remote_bundle"], str(remote.BUNDLE))
        self.assertEqual(hold["bundle_source"], operator.BUNDLE.name)
        self.assertEqual(hold["attempt_label"], remote.LABEL)
        self.assertEqual(hold["one_attempt_marker"], str(operator.MARKER))
        self.assertEqual(hold["consumed_r1_marker"],
                         str(operator.CONSUMED_R1_MARKER))
        self.assertEqual(hold["consumed_r1_marker_sha256"],
                         operator.CONSUMED_R1_MARKER_SHA256)
        self.assertEqual(hold["r5_result_hold_sha256"],
                         operator.R5_RESULT_HOLD_SHA256)
        self.assertEqual(hold["controller_state_root"], str(operator.CONTROL_ROOT))
        self.assertEqual(hold["former_controller_stop_root"],
                         str(operator.FORMER_CONTROL_ROOT))
        self.assertEqual(hold["prior_one_attempt_markers"],
                         [str(path) for path in operator.PRIOR_MARKERS])
        self.assertEqual(hold["max_preopen_attempts"], remote.MAX_PREOPEN_ATTEMPTS)
        self.assertEqual(hold["inherited_max_final_check_age_seconds"],
                         remote.MAX_FINAL_CHECK_AGE_SECONDS)
        self.assertIn("PRESERVE_EVEN_EMPTY", hold["post_open_failure_policy"])
        self.assertEqual(hold["bundle_sha256"], remote.BUNDLE_SHA256)
        self.assertEqual(hold["remote_source_sha256"], hashlib.sha256(
            (HERE / "remote_incremental.py").read_bytes()).hexdigest())
        self.assertEqual(hold["operator_source_sha256"], hashlib.sha256(
            (HERE / "operator_once.py").read_bytes()).hexdigest())
        self.assertNotEqual(str(operator.MARKER), str(operator.CONSUMED_R1_MARKER))
        self.assertNotEqual(str(remote.BUNDLE), str(remote.PRIOR_R1_BUNDLE))
        self.assertNotEqual(str(remote.TARGET), str(remote.PRIOR_R1_TARGET))
        self.assertFalse(operator.ENABLED_REQUEST.exists())
        self.assertFalse(operator.FINAL_REVIEW.exists())
        self.assertFalse(operator.ASSIGNMENT.exists())
        contact = mock.Mock(side_effect=AssertionError("contacted"))
        with mock.patch.object(operator, "_marker") as marker, \
             mock.patch.object(operator, "_ssh_identity") as ssh:
            if sys_flags_optimize():
                with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                    operator.stage_once(remote_call=contact)
            else:
                with self.assertRaises(FileNotFoundError):
                    operator.stage_once(remote_call=contact)
            marker.assert_not_called()
            ssh.assert_not_called()
            contact.assert_not_called()
        with tempfile.TemporaryDirectory(prefix="trn22-r10-r2-entry-isolated-") as temp:
            copy = Path(temp)
            shutil.copyfile(HERE / "operator_once.py", copy / "operator_once.py")
            result = subprocess.run(["/usr/bin/python3", "-B", str(copy / "operator_once.py")],
                                    capture_output=True, text=True, timeout=10, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("HOLD", result.stderr)

    def test_stale_documents_state_cannot_hide_tripped_desktop_breaker(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r10-controller-root-") as temp:
            root = Path(temp)
            desktop = root / "Desktop" / "Trainium"
            documents = root / "Documents" / "ChatGPT" / "Trainium"
            fake_here = root / "isolated" / "research" / "v2" / "authorizations" / "pkg"
            for control in (desktop, documents):
                (control / "research" / "v2").mkdir(parents=True)
            safe = {
                "breakers": {"global_correctness": {"tripped": False},
                             "trn22": {"tripped": False}},
                "launch_authorized": False,
                "hosts": {"trn22": {"launch_allowed": False}},
            }
            tripped = json.loads(json.dumps(safe))
            tripped["breakers"]["trn22"]["tripped"] = True
            (documents / "research/v2/state.json").write_text(
                json.dumps(safe), encoding="utf-8")
            (desktop / "research/v2/state.json").write_text(
                json.dumps(tripped), encoding="utf-8")
            remote_call = mock.Mock(side_effect=AssertionError("SSH attempted"))
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(operator, "CONTROL_ROOT", desktop))
                stack.enter_context(mock.patch.object(operator, "FORMER_CONTROL_ROOT", documents))
                stack.enter_context(mock.patch.object(operator, "HERE", fake_here))
                stack.enter_context(mock.patch.object(operator, "_json", side_effect=[
                    ({"dispatch_authorized": True}, "requesthash"),
                    ({}, "reviewhash"), ({}, "assignmenthash")]))
                stack.enter_context(mock.patch.object(operator, "_source_bytes",
                                                     return_value=(b"bundle", b"remote")))
                stack.enter_context(mock.patch.object(operator, "_authority"))
                ssh_identity = stack.enter_context(mock.patch.object(
                    operator, "_ssh_identity", side_effect=AssertionError("SSH identity attempted")))
                marker = stack.enter_context(mock.patch.object(
                    operator, "_marker", side_effect=AssertionError("marker attempted")))
                with self.assertRaisesRegex(RuntimeError, "controller STOP/breaker/launch state"):
                    operator._local_stop_and_breakers()
                expected = ("optimized Python" if sys_flags_optimize() else
                            "controller STOP/breaker/launch state")
                with self.assertRaisesRegex(RuntimeError, expected):
                    operator.stage_once(remote_call=remote_call)
            ssh_identity.assert_not_called()
            marker.assert_not_called()
            remote_call.assert_not_called()

    def test_remote_timeout_gate_blocks_fake_contact(self):
        fake = mock.Mock()
        with mock.patch.object(operator, "_local_stop_and_breakers"), \
             mock.patch.object(operator, "_ssh_identity"), \
             mock.patch.object(operator.subprocess, "run", return_value=types.SimpleNamespace(
                 returncode=0, stderr=b"", stdout=b"fake timeout\n")) as run:
            with self.assertRaisesRegex(RuntimeError, "remote absolute GNU timeout identity"):
                operator._ssh_phase("preflight", None, b"", b"# empty\n")
            self.assertEqual(run.call_args.args[0][0], "/usr/bin/ssh")
            self.assertEqual(run.call_args.args[0][-1],
                             "/usr/bin/env -i LC_ALL=C /usr/bin/timeout --version")
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

    def test_prior_r9_markers_block_competing_stage(self):
        self.assertEqual(len(operator.PRIOR_MARKERS), 4)
        self.assertTrue(all(str(marker).endswith(
            "trn22-c5c-a0-r9-incremental-stage-%s.one-attempt.jsonl" % revision)
            for marker, revision in zip(operator.PRIOR_MARKERS, ("r1", "r2", "r3", "r4"))))
        with tempfile.TemporaryDirectory(prefix="trn22-r10-r9-prior-") as temp:
            root = Path(temp)
            prior = tuple(root / (revision + ".one-attempt")
                          for revision in ("r1", "r2", "r3", "r4"))
            current = root / "r10.one-attempt"
            for consumed in prior:
                consumed.write_text("consumed\n", encoding="ascii")
                with mock.patch.object(operator, "PRIOR_MARKERS", prior), \
                     mock.patch.object(operator, "MARKER", current):
                    with self.assertRaises(RuntimeError):
                        operator._marker("request", "review", "assignment")
                self.assertFalse(current.exists())
                consumed.unlink()

    def test_local_boundary_receipt_rejects_nonfinite_or_below_floor_values(self):
        for value in (True, -0.1, 1.01, float("nan"), float("inf"), "0"):
            self.assertFalse(operator._bounded_age(value))
        self.assertTrue(operator._bounded_age(1.0))
        self.assertFalse(operator._root_floor({"root_free_bytes": operator.PRE_FREE,
                                               "root_free_inodes": 9_999}))
        self.assertFalse(operator._root_floor({"root_free_bytes": True,
                                               "root_free_inodes": 20_000}))
        self.assertTrue(operator._root_floor({"root_free_bytes": operator.PRE_FREE,
                                              "root_free_inodes": operator.MIN_INODES}))

    def test_write_failure_still_attempts_postflight_and_keeps_marker(self):
        if sys_flags_optimize():
            with self.assertRaises(RuntimeError):
                operator.stage_once(remote_call=mock.Mock())
            return
        with tempfile.TemporaryDirectory(prefix="trn22-r10-once-") as temp:
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
                stack.enter_context(mock.patch.object(operator, "PRIOR_MARKERS", ()))
                stack.enter_context(mock.patch.object(operator, "_json", side_effect=[
                    ({"dispatch_authorized": True}, "requesthash"), ({}, "reviewhash"),
                    ({}, "assignmenthash")]))
                stack.enter_context(mock.patch.object(operator, "_source_bytes",
                                                     return_value=(b"bundle", b"remote")))
                stack.enter_context(mock.patch.object(operator, "_regular", return_value=b"operator"))
                stack.enter_context(mock.patch.object(operator, "_consumed_r1_marker_evidence"))
                stack.enter_context(mock.patch.object(operator, "_authority"))
                stack.enter_context(mock.patch.object(operator, "_local_stop_and_breakers"))
                stack.enter_context(mock.patch.object(operator, "_ssh_identity"))
                stack.enter_context(mock.patch.object(operator, "_remote_timeout_identity"))
                with self.assertRaises(RuntimeError):
                    operator.stage_once(remote_call=fake_remote)
            self.assertEqual(calls, ["preflight", "write", "postflight"])
            self.assertTrue(marker.exists())
            self.assertIn(b"UNCONDITIONAL_FULL_POSTFLIGHT", marker.read_bytes())

    def test_missing_post_open_proof_blocks_readback_and_keeps_postflight(self):
        if sys_flags_optimize():
            with self.assertRaises(RuntimeError):
                operator.stage_once(remote_call=mock.Mock())
            return
        with tempfile.TemporaryDirectory(prefix="trn22-r10-r4-proof-") as temp:
            root = Path(temp)
            marker = root / "marker"
            calls = []
            snapshot = {"bundle_exists": False, "target_exists": False,
                        "source": {"identity": "same"},
                        "protected": {"checkpoint": "same"}}

            def fake_remote(phase, prior, data, remote_source):
                calls.append(phase)
                if phase == "write":
                    return {"phase": phase, "snapshot": snapshot,
                            "final_no_write_continuity": {"double_label_scan": True}}
                return {"phase": phase, "snapshot": snapshot}

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(operator, "ENABLED_REQUEST", root / "enabled"))
                stack.enter_context(mock.patch.object(operator, "MARKER", marker))
                stack.enter_context(mock.patch.object(operator, "PRIOR_MARKERS", ()))
                stack.enter_context(mock.patch.object(operator, "_json", side_effect=[
                    ({"dispatch_authorized": True}, "requesthash"), ({}, "reviewhash"),
                    ({}, "assignmenthash")]))
                stack.enter_context(mock.patch.object(operator, "_source_bytes",
                                                     return_value=(b"bundle", b"remote")))
                stack.enter_context(mock.patch.object(operator, "_regular", return_value=b"operator"))
                stack.enter_context(mock.patch.object(operator, "_consumed_r1_marker_evidence"))
                stack.enter_context(mock.patch.object(operator, "_authority"))
                stack.enter_context(mock.patch.object(operator, "_local_stop_and_breakers"))
                stack.enter_context(mock.patch.object(operator, "_ssh_identity"))
                stack.enter_context(mock.patch.object(operator, "_remote_timeout_identity"))
                with self.assertRaisesRegex(RuntimeError, "final STOP/label/holder/disk/open proof"):
                    operator.stage_once(remote_call=fake_remote)
            self.assertEqual(calls, ["preflight", "write", "postflight"])
            self.assertTrue(marker.exists())
            self.assertIn(b"POSTFLIGHT", marker.read_bytes())


def sys_flags_optimize():
    import sys
    return bool(sys.flags.optimize)


if __name__ == "__main__":
    unittest.main()
