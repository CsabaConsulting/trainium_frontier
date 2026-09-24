#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("cleanup_contract", HERE / "cleanup_contract.py")
C = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(C)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request_for_current_constants(parent: Path, head: str = "a" * 40) -> dict:
    return {
        "schema": C.SCHEMA_REQUEST,
        "state": "finalized",
        "cleanup": True,
        "host": C.HOST,
        "expected_hostname": C.HOSTNAME,
        "parent": str(parent),
        "origin": C.ORIGIN,
        "minimum_post_cleanup_free_bytes": C.MIN_POST_CLEANUP_FREE_BYTES,
        "targets": list(C.TARGETS),
        "preserve": {
            "r2_checkout": C.R2_ROOT,
            "r2_runtime_sha256": C.R2_RUNTIME_HASHES,
            "g0_checkpoint": C.G0_CHECKPOINT,
            "all_logs_results_evidence": True,
            "all_caches": True,
            "unknown_paths": True,
            "all_other_worktrees": True,
        },
        "authorities": C.FORBIDDEN_AUTHORITIES,
        "evidence_commits": {
            "r1": "e1550203a61c53cd7c73b0c543ced8a157312612",
            "r2": "5b392db4bc8c01a47da2f5969cb33df0c50ec52b",
            "r3": "5e45a564bba0d04b052345a7c9c1812ddb71ce51",
            "analysis": "83e533546fb8b53f5c8268d84f0ccd2b5c90e1a1",
        },
        "package_head": head,
        "inventory_output": str(parent / "cleanup.inventory.json"),
        "journal_output": str(parent / "cleanup.journal.jsonl"),
        "receipt_output": str(parent / "cleanup.receipt.json"),
    }


class StaticContractTests(unittest.TestCase):
    def test_archive_backed_exact_bundle_pins(self):
        bundles = {x["role"]: x for x in C.TARGETS if x["kind"] == "bundle"}
        self.assertEqual(bundles["consumed-r1-bundle"]["bytes"], 46_549_993)
        self.assertEqual(bundles["consumed-r1-bundle"]["sha256"],
                         "942b85ec982b1834ab44d41219523c0ffbf640464d8a847417b529436c03116e")
        self.assertEqual(bundles["consumed-r2-bundle-only"]["bytes"], 46_585_255)
        self.assertEqual(bundles["consumed-r2-bundle-only"]["sha256"],
                         "bf33a3fe75e67fd23018151d8ccd1e219b96e91b7dfc52e1bb9fe34a48137bc8")
        self.assertEqual(bundles["consumed-r3-bundle"]["sha256"],
                         "e319b4d1455a31dfa84594ff3fab6a6d4a3bf8ad8179f7a39bc2fe7d2504cd4c")

    def test_exact_target_set_excludes_r2_checkout_and_all_caches(self):
        paths = {x["path"] for x in C.TARGETS}
        self.assertEqual(len(paths), 5)
        self.assertNotIn(C.R2_ROOT, paths)
        self.assertFalse(any("cache" in path for path in paths))
        self.assertFalse(any("*" in path or ".." in Path(path).parts for path in paths))

    def test_pending_request_and_authorization_are_non_executable(self):
        request = json.loads((HERE / "cleanup-request.pending.json").read_text())
        auth = json.loads((HERE / "cleanup-authorization.pending.json").read_text())
        self.assertFalse(request["cleanup"])
        self.assertFalse(auth["cleanup"])
        self.assertFalse(auth["executable"])
        with self.assertRaises(C.ContractError):
            C.validate_request(request)

    def test_no_ssh_training_or_broad_delete_capability(self):
        source = (HERE / "cleanup_contract.py").read_text()
        self.assertNotIn("import paramiko", source)
        self.assertNotIn("shutil.rmtree", source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("train.py\"]", source)
        self.assertNotIn("os.system", source)

    def test_request_mutations_fail(self):
        request = request_for_current_constants(Path(C.PARENT))
        C.validate_request(request)
        mutations = []
        for key, value in (
            ("state", "pending"), ("cleanup", False), ("host", "trn22"),
            ("expected_hostname", "wrong"), ("parent", "/tmp"),
            ("origin", "https://evil.invalid/repo"),
            ("minimum_post_cleanup_free_bytes", C.MIN_POST_CLEANUP_FREE_BYTES - 1),
            ("package_head", "PENDING"),
        ):
            item = copy.deepcopy(request)
            item[key] = value
            mutations.append(item)
        item = copy.deepcopy(request); item["targets"] = item["targets"][:-1]; mutations.append(item)
        item = copy.deepcopy(request); item["targets"].reverse(); mutations.append(item)
        item = copy.deepcopy(request); item["targets"][0]["path"] += "-other"; mutations.append(item)
        item = copy.deepcopy(request); item["preserve"]["all_caches"] = False; mutations.append(item)
        item = copy.deepcopy(request); item["authorities"]["training"] = True; mutations.append(item)
        item = copy.deepcopy(request); item["extra"] = True; mutations.append(item)
        for mutated in mutations:
            with self.subTest(mutated=mutated):
                with self.assertRaises(C.ContractError):
                    C.validate_request(mutated)

    def test_authorization_requires_external_exact_identities(self):
        request = request_for_current_constants(Path(C.PARENT))
        inventory = {"request_sha256": "b" * 64}
        auth = {
            "schema": C.SCHEMA_AUTH, "state": "finalized", "executable": True, "cleanup": True,
            "authorized_head": request["package_head"], "request_sha256": "b" * 64,
            "inventory_sha256": "c" * 64, "host": C.HOST,
            "expected_hostname": C.HOSTNAME, "targets": list(C.TARGETS),
            "authorities": C.FORBIDDEN_AUTHORITIES,
            "minimum_post_cleanup_free_bytes": C.MIN_POST_CLEANUP_FREE_BYTES,
            "independent_verifier_commit": "d" * 40, "human_authorization": True,
        }
        C.validate_authorization(auth, request, "b" * 64, inventory, "c" * 64)
        for key, value in (
            ("executable", False), ("cleanup", False), ("authorized_head", "e" * 40),
            ("request_sha256", "e" * 64), ("inventory_sha256", "e" * 64),
            ("host", "trn22"), ("human_authorization", False),
        ):
            bad = copy.deepcopy(auth); bad[key] = value
            with self.subTest(key=key), self.assertRaises(C.ContractError):
                C.validate_authorization(bad, request, "b" * 64, inventory, "c" * 64)


class SealedIOTests(unittest.TestCase):
    def test_anchored_json_read_rejects_symlink_hardlink_directory_and_wrong_mode(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(C, "PARENT", temp):
            root = Path(temp)
            regular = root / "regular"; regular.write_text('{"value":1}\n'); regular.chmod(0o600)
            value, data = C.read_json_immediate_parent(str(regular), require_mode=0o600)
            self.assertEqual(value, {"value": 1})
            self.assertEqual(data, b'{"value":1}\n')
            symlink = root / "symlink"; symlink.symlink_to(regular)
            with self.assertRaises(OSError): C.read_json_immediate_parent(str(symlink))
            hard = root / "hard"; os.link(regular, hard)
            with self.assertRaises(C.ContractError): C.read_json_immediate_parent(str(regular))
            directory = root / "directory"; directory.mkdir()
            with self.assertRaises(C.ContractError): C.read_json_immediate_parent(str(directory))
            hard.unlink(); regular.chmod(0o644)
            with self.assertRaises(C.ContractError):
                C.read_json_immediate_parent(str(regular), require_mode=0o600)

    def test_safe_dir_rejects_symlink_ancestor_and_noncanonical_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = root / "real"; real.mkdir()
            link = root / "link"; link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(OSError):
                C.open_safe_dir(str(link))
            with self.assertRaises(C.ContractError):
                C.open_safe_dir(str(root / "real" / ".." / "real"))

    def test_exclusive_durable_output_rejects_existing(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(C, "PARENT", temp):
            path = str(Path(temp) / "receipt.json")
            expected = C.sha_bytes(C.canonical({"status": "PASS"}))
            self.assertEqual(C._write_exclusive(path, {"status": "PASS"}), expected)
            self.assertEqual(stat.S_IMODE(os.lstat(path).st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                C._write_exclusive(path, {"status": "PASS"})

    def test_short_write_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(C, "PARENT", temp):
            path = str(Path(temp) / "receipt.json")
            with mock.patch.object(C.os, "write", return_value=0):
                with self.assertRaises(C.ContractError):
                    C._write_exclusive(path, {"status": "PASS"})

    def test_full_write_retries_eintr(self):
        with tempfile.TemporaryFile() as stream:
            original = C.os.write
            calls = []
            def interrupted_once(fd, data):
                calls.append(True)
                if len(calls) == 1:
                    raise InterruptedError
                return original(fd, data)
            with mock.patch.object(C.os, "write", side_effect=interrupted_once):
                C._full_write(stream.fileno(), b"durable")
            stream.seek(0)
            self.assertEqual(stream.read(), b"durable")

    def test_output_rejects_replaced_parent_identity(self):
        with tempfile.TemporaryDirectory() as outer:
            parent = Path(outer) / "parent"; parent.mkdir()
            fd = C.open_safe_dir(str(parent))
            try:
                identity = C.parent_identity(fd)
            finally:
                os.close(fd)
            moved = Path(outer) / "moved"; parent.rename(moved); parent.mkdir()
            with mock.patch.object(C, "PARENT", str(parent)):
                with self.assertRaisesRegex(C.ContractError, "parent identity"):
                    C._write_exclusive(str(parent / "receipt.json"), {"status": "PASS"}, identity)
            self.assertFalse((parent / "receipt.json").exists())

    def test_output_readback_rejects_same_size_mutation_and_inode_replacement(self):
        for attack in ("mutate", "replace"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temp, \
                    mock.patch.object(C, "PARENT", temp):
                path = Path(temp) / "receipt.json"
                def race(event, context):
                    if event != "after_output_write_before_readback":
                        return
                    if attack == "mutate":
                        data = path.read_bytes()
                        path.write_bytes(b"X" * len(data))
                    else:
                        path.unlink()
                        path.write_text('{}\n')
                        path.chmod(0o600)
                with mock.patch.object(C, "_BOUNDARY_HOOK", race):
                    with self.assertRaises(C.ContractError):
                        C._write_exclusive(str(path), {"status": "PASS"})


class FilesystemContractTests(unittest.TestCase):
    def _git(self, cwd: Path, *argv: str) -> str:
        return subprocess.check_output([C.GIT, "-C", str(cwd), *argv], text=True).strip()

    def _make_checkout(self, path: Path, origin: str) -> str:
        path.mkdir()
        subprocess.check_call([C.GIT, "-C", str(path), "init", "-q"])
        subprocess.check_call([C.GIT, "-C", str(path), "config", "user.name", "Test"])
        subprocess.check_call([C.GIT, "-C", str(path), "config", "user.email", "test@example.invalid"])
        (path / "tracked.txt").write_text("tracked\n")
        subprocess.check_call([C.GIT, "-C", str(path), "add", "tracked.txt"])
        subprocess.check_call([C.GIT, "-C", str(path), "commit", "-qm", "fixture"])
        subprocess.check_call([C.GIT, "-C", str(path), "remote", "add", "origin", origin])
        return self._git(path, "rev-parse", "HEAD")

    def test_bundle_exact_hash_size_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp); data = b"bundle-exact"
            path = parent / "one.bundle"; path.write_bytes(data)
            target = {"path": str(path), "kind": "bundle", "bytes": len(data), "sha256": digest(data)}
            fd = C.open_safe_dir(str(parent))
            try:
                record = C.verify_bundle(fd, target)
                self.assertEqual(record["sha256"], digest(data))
                path.write_bytes(b"different")
                with self.assertRaises(C.ContractError): C.verify_bundle(fd, target)
                path.unlink(); path.symlink_to(parent / "missing")
                with self.assertRaises(OSError): C.verify_bundle(fd, target)
            finally:
                os.close(fd)

    def test_bundle_mount_substitution_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp); data = b"bundle"
            path = parent / "one.bundle"; path.write_bytes(data)
            target = {"path": str(path), "kind": "bundle", "bytes": len(data), "sha256": digest(data)}
            fd = C.open_safe_dir(str(parent))
            try:
                with mock.patch.object(C, "mount_id", side_effect=[222, 111]):
                    with self.assertRaisesRegex(C.ContractError, "mount substitution"):
                        C.verify_bundle(fd, target)
            finally:
                os.close(fd)

    def test_checkout_exact_head_origin_clean_and_no_links(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp); checkout = parent / "checkout"
            with mock.patch.object(C, "ORIGIN", "fixture-origin"):
                head = self._make_checkout(checkout, C.ORIGIN)
                target = {"path": str(checkout), "kind": "checkout", "head": head}
                fd = C.open_safe_dir(str(parent))
                try:
                    self.assertEqual(C.verify_checkout(fd, target)["head"], head)
                    (checkout / "untracked").write_text("x")
                    with self.assertRaises(C.ContractError): C.verify_checkout(fd, target)
                    (checkout / "untracked").unlink()
                    (checkout / ".gitignore").write_text("ignored\n")
                    subprocess.check_call([C.GIT, "-C", str(checkout), "add", ".gitignore"])
                    subprocess.check_call([C.GIT, "-C", str(checkout), "commit", "-qm", "ignore rule"])
                    target["head"] = self._git(checkout, "rev-parse", "HEAD")
                    (checkout / "ignored").write_text("must-not-delete")
                    with self.assertRaises(C.ContractError): C.verify_checkout(fd, target)
                    (checkout / "ignored").unlink()
                    (checkout / "bad-link").symlink_to("tracked.txt")
                    with self.assertRaises(C.ContractError): C.verify_checkout(fd, target)
                finally:
                    os.close(fd)

    def test_wrong_host_fails_before_lock_or_device_probe(self):
        with mock.patch.object(C.socket, "gethostname", return_value="not-trn21"), \
                mock.patch.object(C.os, "open") as opened:
            with self.assertRaises(C.ContractError):
                with C.host_quiescence():
                    pass
            opened.assert_not_called()

    def test_missing_target_stops_before_any_delete(self):
        parent_fd = mock.Mock()
        targets = (
            {"path": "/fixed/one", "kind": "bundle"},
            {"path": "/fixed/two", "kind": "bundle"},
        )
        with mock.patch.object(C, "TARGETS", targets), \
                mock.patch.object(C, "verify_bundle", side_effect=[{"metadata": {"dev": 1}}, C.ContractError("missing")]), \
                mock.patch.object(C, "open_safe_dir", return_value=10), \
                mock.patch.object(C.os, "fstat", return_value=mock.Mock(st_dev=1)), \
                mock.patch.object(C.os, "close"), \
                mock.patch.object(C, "free_record_fd", return_value={"free_bytes": 9_000_000_000}), \
                mock.patch.object(C, "protected_records", return_value={}), \
                mock.patch.object(C, "_assert_request_head"), \
                mock.patch.object(C, "validate_request"):
            with self.assertRaises(C.ContractError):
                C._collect_inventory_locked({}, "a" * 64, {})

    def test_insufficient_conservative_projection_fails_before_delete(self):
        targets = ({"path": "/fixed/one", "kind": "bundle"},)
        record = {"kind": "bundle", "metadata": {"dev": 1, "blocks": 1}}
        with mock.patch.object(C, "TARGETS", targets), \
                mock.patch.object(C, "verify_bundle", return_value=record), \
                mock.patch.object(C, "open_safe_dir", return_value=10), \
                mock.patch.object(C.os, "fstat", return_value=mock.Mock(st_dev=1)), \
                mock.patch.object(C.os, "close"), \
                mock.patch.object(C, "free_record_fd", return_value={"free_bytes": 1}), \
                mock.patch.object(C, "protected_records", return_value={}), \
                mock.patch.object(C, "_assert_request_head"), \
                mock.patch.object(C, "validate_request"):
            with self.assertRaisesRegex(C.ContractError, "projection"):
                C._collect_inventory_locked({}, "a" * 64, {})


class EndToEndTempCleanupTests(unittest.TestCase):
    def _git_fixture(self, path: Path, origin: str) -> str:
        path.mkdir()
        subprocess.check_call([C.GIT, "-C", str(path), "init", "-q"])
        subprocess.check_call([C.GIT, "-C", str(path), "config", "user.name", "Test"])
        subprocess.check_call([C.GIT, "-C", str(path), "config", "user.email", "test@example.invalid"])
        (path / "tracked").write_text("ok\n")
        subprocess.check_call([C.GIT, "-C", str(path), "add", "tracked"])
        subprocess.check_call([C.GIT, "-C", str(path), "commit", "-qm", "fixture"])
        subprocess.check_call([C.GIT, "-C", str(path), "remote", "add", "origin", origin])
        return subprocess.check_output([C.GIT, "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

    @contextlib.contextmanager
    def _health(self):
        yield {"hostname": C.HOSTNAME, "workflow_lock_free_and_held_by_cleanup": True,
               "relevant_processes": [], "device_holders": [], "neuron_process_count": 0}

    def test_exact_temp_targets_deleted_and_protected_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp); origin = "fixture-origin"
            c1 = parent / "consumed-r1"; h1 = self._git_fixture(c1, origin)
            c3 = parent / "consumed-r3"; h3 = self._git_fixture(c3, origin)
            bundle_data = [b"r1-bundle", b"r3-bundle", b"r2-bundle"]
            bundle_names = ["r1.bundle", "r3.bundle", "r2.bundle"]
            for name, data in zip(bundle_names, bundle_data, strict=True):
                (parent / name).write_bytes(data)
            targets = (
                {"kind": "checkout", "role": "consumed-r1-checkout", "path": str(c1),
                 "head": h1, "archive_evidence_commit": "e1550203a61c53cd7c73b0c543ced8a157312612"},
                {"kind": "bundle", "role": "consumed-r1-bundle", "path": str(parent / bundle_names[0]),
                 "bytes": len(bundle_data[0]), "sha256": digest(bundle_data[0]),
                 "archive_evidence_commit": "e1550203a61c53cd7c73b0c543ced8a157312612"},
                {"kind": "checkout", "role": "consumed-r3-checkout", "path": str(c3),
                 "head": h3, "archive_evidence_commit": "5e45a564bba0d04b052345a7c9c1812ddb71ce51"},
                {"kind": "bundle", "role": "consumed-r3-bundle", "path": str(parent / bundle_names[1]),
                 "bytes": len(bundle_data[1]), "sha256": digest(bundle_data[1]),
                 "archive_evidence_commit": "5e45a564bba0d04b052345a7c9c1812ddb71ce51"},
                {"kind": "bundle", "role": "consumed-r2-bundle-only", "path": str(parent / bundle_names[2]),
                 "bytes": len(bundle_data[2]), "sha256": digest(bundle_data[2]),
                 "archive_evidence_commit": "5b392db4bc8c01a47da2f5969cb33df0c50ec52b"},
            )
            r2 = parent / "preserved-r2"; r2.mkdir()
            runtime_hashes = {}
            for index in range(8):
                relative = f"runtime/file-{index}.json"
                path = r2 / relative; path.parent.mkdir(parents=True, exist_ok=True)
                data = f"runtime-{index}".encode(); path.write_bytes(data); runtime_hashes[relative] = digest(data)
            checkpoint = parent / "g0-final.pt"; checkpoint.write_bytes(b"checkpoint")
            checkpoint_spec = {"path": str(checkpoint), "bytes": len(b"checkpoint"),
                               "sha256": digest(b"checkpoint")}
            patches = (
                mock.patch.object(C, "PARENT", str(parent)),
                mock.patch.object(C, "ORIGIN", origin),
                mock.patch.object(C, "TARGETS", targets),
                mock.patch.object(C, "R2_ROOT", str(r2)),
                mock.patch.object(C, "R2_RUNTIME_HASHES", runtime_hashes),
                mock.patch.object(C, "G0_CHECKPOINT", checkpoint_spec),
                mock.patch.object(C, "G0_ROOT", str(parent)),
                mock.patch.object(C, "MIN_POST_CLEANUP_FREE_BYTES", 1),
                mock.patch.object(C, "_assert_request_head"),
                mock.patch.object(C, "host_quiescence", self._health),
                mock.patch.object(C, "free_record_fd", return_value={"free_bytes": 10_000_000_000,
                                                                  "free_inodes": 1_000_000,
                                                                  "fragment_size": 4096}),
            )
            with contextlib.ExitStack() as stack:
                for patch in patches: stack.enter_context(patch)
                request = request_for_current_constants(parent)
                request_sha = digest(C.canonical(request))
                inventory = C.collect_inventory(request, request_sha)
                inventory_sha = digest(C.canonical(inventory))
                auth = {
                    "schema": C.SCHEMA_AUTH, "state": "finalized", "executable": True,
                    "cleanup": True, "authorized_head": request["package_head"],
                    "request_sha256": request_sha, "inventory_sha256": inventory_sha,
                    "host": C.HOST, "expected_hostname": C.HOSTNAME, "targets": list(C.TARGETS),
                    "authorities": C.FORBIDDEN_AUTHORITIES,
                    "minimum_post_cleanup_free_bytes": C.MIN_POST_CLEANUP_FREE_BYTES,
                    "independent_verifier_commit": "b" * 40, "human_authorization": True,
                }
                result = C.execute(request, request_sha, inventory, inventory_sha, auth,
                                   digest(C.canonical(auth)))
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["deleted_exact_paths"], [x["path"] for x in targets])
            self.assertTrue(all(not Path(x["path"]).exists() for x in targets))
            self.assertTrue(r2.is_dir())
            self.assertEqual(len(list(r2.glob("runtime/*"))), 8)
            self.assertEqual(checkpoint.read_bytes(), b"checkpoint")
            self.assertEqual(result["free_before"], result["free_after"])


class RaceRegressionTests(unittest.TestCase):
    def _git_fixture(self, path: Path, origin: str) -> str:
        path.mkdir()
        subprocess.check_call([C.GIT, "-C", str(path), "init", "-q"])
        subprocess.check_call([C.GIT, "-C", str(path), "config", "user.name", "Test"])
        subprocess.check_call([C.GIT, "-C", str(path), "config", "user.email", "test@example.invalid"])
        (path / "tracked").write_text("authorized\n")
        subprocess.check_call([C.GIT, "-C", str(path), "add", "tracked"])
        subprocess.check_call([C.GIT, "-C", str(path), "commit", "-qm", "fixture"])
        subprocess.check_call([C.GIT, "-C", str(path), "remote", "add", "origin", origin])
        return subprocess.check_output([C.GIT, "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

    @contextlib.contextmanager
    def _health(self):
        yield {"hostname": C.HOSTNAME, "workflow_lock_free_and_held_by_cleanup": True,
               "relevant_processes": [], "device_holders": [], "neuron_process_count": 0}

    @contextlib.contextmanager
    def fixture(self, kinds=("checkout",)):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            origin = "fixture-origin"
            targets = []
            for index, kind in enumerate(kinds):
                if kind == "checkout":
                    path = parent / f"checkout-{index}"
                    head = self._git_fixture(path, origin)
                    targets.append({"kind": "checkout", "role": f"checkout-{index}",
                                    "path": str(path), "head": head,
                                    "archive_evidence_commit": "e" * 40})
                else:
                    path = parent / f"bundle-{index}"
                    data = f"authorized-bundle-{index}".encode()
                    path.write_bytes(data)
                    targets.append({"kind": "bundle", "role": f"bundle-{index}",
                                    "path": str(path), "bytes": len(data), "sha256": digest(data),
                                    "archive_evidence_commit": "e" * 40})
            r2 = parent / "protected-r2"; r2.mkdir()
            runtime = r2 / "runtime.json"; runtime.write_bytes(b"runtime")
            runtime_hashes = {"runtime.json": digest(b"runtime")}
            checkpoint = parent / "g0.pt"; checkpoint.write_bytes(b"checkpoint")
            checkpoint_spec = {"path": str(checkpoint), "bytes": len(b"checkpoint"),
                               "sha256": digest(b"checkpoint")}
            patches = (
                mock.patch.object(C, "PARENT", str(parent)),
                mock.patch.object(C, "ORIGIN", origin),
                mock.patch.object(C, "TARGETS", tuple(targets)),
                mock.patch.object(C, "R2_ROOT", str(r2)),
                mock.patch.object(C, "R2_RUNTIME_HASHES", runtime_hashes),
                mock.patch.object(C, "G0_CHECKPOINT", checkpoint_spec),
                mock.patch.object(C, "G0_ROOT", str(parent)),
                mock.patch.object(C, "MIN_POST_CLEANUP_FREE_BYTES", 1),
                mock.patch.object(C, "_assert_request_head"),
                mock.patch.object(C, "host_quiescence", self._health),
                mock.patch.object(C, "free_record_fd", return_value={"free_bytes": 10_000_000_000,
                                                                  "free_inodes": 1_000_000,
                                                                  "fragment_size": 4096}),
            )
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                request = request_for_current_constants(parent)
                request_sha = digest(C.canonical(request))
                inventory = C.collect_inventory(request, request_sha)
                inventory_sha = digest(C.canonical(inventory))
                auth = {
                    "schema": C.SCHEMA_AUTH, "state": "finalized", "executable": True,
                    "cleanup": True, "authorized_head": request["package_head"],
                    "request_sha256": request_sha, "inventory_sha256": inventory_sha,
                    "host": C.HOST, "expected_hostname": C.HOSTNAME, "targets": list(C.TARGETS),
                    "authorities": C.FORBIDDEN_AUTHORITIES,
                    "minimum_post_cleanup_free_bytes": C.MIN_POST_CLEANUP_FREE_BYTES,
                    "independent_verifier_commit": "b" * 40, "human_authorization": True,
                }
                yield {"parent": parent, "targets": targets, "request": request,
                       "request_sha": request_sha, "inventory": inventory,
                       "inventory_sha": inventory_sha, "auth": auth,
                       "auth_sha": digest(C.canonical(auth)), "r2": r2,
                       "checkpoint": checkpoint}

    def _execute(self, fixture):
        return C.execute(fixture["request"], fixture["request_sha"], fixture["inventory"],
                         fixture["inventory_sha"], fixture["auth"], fixture["auth_sha"])

    def test_checkout_swap_after_open_never_deletes_replacement(self):
        with self.fixture(("checkout",)) as item:
            target = Path(item["targets"][0]["path"])
            moved = item["parent"] / "authorized-moved"
            def attack(event, context):
                if event == "after_target_open_before_quarantine":
                    target.rename(moved)
                    target.mkdir()
                    (target / "unknown").write_text("replacement-must-survive")
            with mock.patch.object(C, "_BOUNDARY_HOOK", attack):
                result = self._execute(item)
            self.assertEqual(result["status"], "PARTIAL_FAILURE")
            self.assertEqual((moved / "tracked").read_text(), "authorized\n")
            self.assertEqual((target / "unknown").read_text(), "replacement-must-survive")
            self.assertTrue(Path(item["request"]["journal_output"]).is_file())

    def test_same_inode_bundle_mutation_after_quarantine_is_not_unlinked(self):
        with self.fixture(("bundle",)) as item:
            target = Path(item["targets"][0]["path"])
            mutated = b"X" * item["targets"][0]["bytes"]
            def attack(event, context):
                if event == "before_bundle_final_hash":
                    quarantine = item["parent"] / context["quarantine"]
                    with quarantine.open("r+b") as stream:
                        stream.write(mutated)
                        stream.flush()
                        os.fsync(stream.fileno())
            with mock.patch.object(C, "_BOUNDARY_HOOK", attack):
                result = self._execute(item)
            self.assertEqual(result["status"], "PARTIAL_FAILURE")
            self.assertEqual(target.read_bytes(), mutated)
            self.assertFalse(result["cleanup_performed"])

    def test_post_inventory_checkout_insertion_is_preserved_and_rejected(self):
        with self.fixture(("checkout",)) as item:
            target = Path(item["targets"][0]["path"])
            def attack(event, context):
                if event == "after_checkout_quarantine_validation":
                    (item["parent"] / context["quarantine"] / "late-unknown").write_text("keep")
            with mock.patch.object(C, "_BOUNDARY_HOOK", attack):
                result = self._execute(item)
            self.assertEqual(result["status"], "PARTIAL_FAILURE")
            self.assertEqual((target / "late-unknown").read_text(), "keep")
            self.assertFalse(result["cleanup_performed"])

    def test_nested_directory_swap_before_quarantine_preserves_replacement(self):
        with self.fixture(("checkout",)) as item:
            target = Path(item["targets"][0]["path"])
            fired = []
            def attack(event, context):
                if event == "before_entry_quarantine" and context["relative"] == ".git" and not fired:
                    fired.append(True)
                    os.rename(".git", ".git-authorized-moved",
                              src_dir_fd=context["dir_fd"], dst_dir_fd=context["dir_fd"])
                    os.mkdir(".git", dir_fd=context["dir_fd"])
                    replacement = os.open(".git", os.O_RDONLY | os.O_DIRECTORY,
                                          dir_fd=context["dir_fd"])
                    try:
                        fd = os.open("unknown", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                                     dir_fd=replacement)
                        os.write(fd, b"preserve"); os.close(fd)
                    finally:
                        os.close(replacement)
            with mock.patch.object(C, "_BOUNDARY_HOOK", attack):
                result = self._execute(item)
            self.assertTrue(fired)
            self.assertEqual(result["status"], "PARTIAL_FAILURE")
            self.assertTrue((target / ".git-authorized-moved").is_dir())
            self.assertEqual((target / ".git" / "unknown").read_bytes(), b"preserve")

    def test_nested_hardlink_fifo_and_symlink_are_rejected_by_inventory(self):
        for attack in ("hardlink", "fifo", "symlink"):
            with self.subTest(attack=attack), self.fixture(("checkout",)) as item:
                root = Path(item["targets"][0]["path"])
                if attack == "hardlink":
                    os.link(root / "tracked", root / "extra")
                elif attack == "fifo":
                    os.mkfifo(root / "extra")
                else:
                    (root / "extra").symlink_to("tracked")
                with self.assertRaises(C.ContractError):
                    C.collect_inventory(item["request"], item["request_sha"])

    def test_interruption_after_each_destructive_boundary_is_never_pass(self):
        boundaries = ("after_target_quarantine", "after_entry_quarantine",
                      "after_entry_delete", "after_target_delete")
        for boundary in boundaries:
            with self.subTest(boundary=boundary), self.fixture(("checkout",)) as item:
                fired = []
                def interrupt(event, _context):
                    if event == boundary and not fired:
                        fired.append(True)
                        raise RuntimeError(f"interrupt at {boundary}")
                with mock.patch.object(C, "_BOUNDARY_HOOK", interrupt):
                    result = self._execute(item)
                self.assertTrue(fired)
                self.assertEqual(result["status"], "PARTIAL_FAILURE")
                self.assertTrue(Path(item["request"]["journal_output"]).is_file())
                events = [json.loads(line)["event"] for line in
                          Path(item["request"]["journal_output"]).read_text().splitlines()]
                self.assertEqual(events[0], "INTENT")
                self.assertEqual(events[-1], "PARTIAL_FAILURE")

    def test_main_writes_durable_partial_receipt_after_first_target_deleted(self):
        with self.fixture(("bundle", "bundle")) as item:
            second = item["targets"][1]
            def attack(event, context):
                if event == "before_bundle_final_hash" and context["target"] == second["path"]:
                    quarantine = item["parent"] / context["quarantine"]
                    quarantine.write_bytes(b"Z" * second["bytes"])
            request_path = item["parent"] / "request.json"
            inventory_path = Path(item["request"]["inventory_output"])
            auth_path = item["parent"] / "authorization.json"
            for path, value in ((request_path, item["request"]),
                                (inventory_path, item["inventory"]),
                                (auth_path, item["auth"])):
                path.write_bytes(C.canonical(value)); path.chmod(0o600)
            with mock.patch.object(C, "_BOUNDARY_HOOK", attack):
                rc = C.main(["execute", "--request", str(request_path),
                             "--inventory", str(inventory_path), "--authorization", str(auth_path)])
            self.assertEqual(rc, 2)
            receipt = json.loads(Path(item["request"]["receipt_output"]).read_text())
            self.assertEqual(receipt["status"], "PARTIAL_FAILURE")
            self.assertTrue(receipt["cleanup_performed"])
            self.assertFalse(Path(item["targets"][0]["path"]).exists())
            self.assertTrue(Path(item["targets"][1]["path"]).exists())
            self.assertEqual(json.loads(Path(item["request"]["journal_output"]).read_text().splitlines()[-1])
                             ["event"], "PARTIAL_FAILURE")

    def test_stale_v1_and_mutated_journal_paths_are_rejected(self):
        with self.fixture(("bundle",)) as item:
            stale = copy.deepcopy(item["request"]); stale["schema"] = "a1-native-g1-consumed-staging-cleanup-request-v1"
            with self.assertRaises(C.ContractError): C.validate_request(stale)
            missing = copy.deepcopy(item["request"]); missing.pop("journal_output")
            with self.assertRaises(C.ContractError): C.validate_request(missing)
            alias = copy.deepcopy(item["request"]); alias["journal_output"] = alias["receipt_output"]
            with self.assertRaises(C.ContractError): C.validate_request(alias)
            stale_auth = copy.deepcopy(item["auth"]); stale_auth["schema"] = "a1-native-g1-consumed-staging-cleanup-authorization-v1"
            with self.assertRaises(C.ContractError):
                C.validate_authorization(stale_auth, item["request"], item["request_sha"],
                                         item["inventory"], item["inventory_sha"])


if __name__ == "__main__":
    unittest.main()
