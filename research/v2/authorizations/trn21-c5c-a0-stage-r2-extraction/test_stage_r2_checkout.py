"""Local-only synthetic and pinned-capsule checks; never SSH or train."""

import ast
import copy
import datetime
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = HERE / "stage_r2_checkout.py"
SPEC = importlib.util.spec_from_file_location("stage_r2_checkout_test", SOURCE)
STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE)


def fixture_archive(path, entries, *, fmt=tarfile.USTAR_FORMAT):
    """Entries are (name, type, payload, mode, optional pax headers)."""
    with tarfile.open(path, "w:gz", format=fmt) as archive:
        for entry in entries:
            name, kind, payload, mode, *extra = entry
            info = tarfile.TarInfo(name)
            info.mode = mode
            info.uid = os.geteuid()
            info.gid = os.getegid()
            if extra:
                info.pax_headers = extra[0]
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            elif kind == "file":
                info.type = tarfile.REGTYPE
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            else:
                info.type = kind
                info.linkname = "./elsewhere" if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ""
                archive.addfile(info)


def safe_entries():
    return [
        (".", "dir", b"", 0o700),
        ("./dir", "dir", b"", 0o775),
        ("./dir/hello.txt", "file", b"safe synthetic text\n", 0o664),
        ("./tool.sh", "file", b"#!/bin/sh\nexit 0\n", 0o775),
    ]


def fake_bundle():
    start = datetime.datetime(2026, 9, 25, 1, 0, tzinfo=datetime.timezone.utc)
    identity = {
        "hostname": STAGE.HOST, "uid": STAGE.UID, "gid": STAGE.UID,
        "boot_id": "synthetic-boot", "mount_namespace": "mnt:[synthetic]",
        "mount": {"mount_id": "1", "major_minor": "259:1", "mount_point": "/",
                  "filesystem": "ext4", "source": "/dev/synthetic"},
        "parent_device": 99, "parent_inode": 123, "parent_uid": STAGE.UID,
        "parent_gid": STAGE.UID, "parent_mode": "0o755",
        "root_free_bytes": 50 * 1024**3,
    }
    bundle = {}
    for number, (artifact, (path, size, digest)) in enumerate(STAGE.TRANSFER_ARTIFACTS.items()):
        metadata = {"device": 99, "inode": 200 + number, "bytes": size,
                    "uid": STAGE.UID, "gid": STAGE.UID, "mode": "0o600", "nlink": 1}
        group = {}
        for position, key in enumerate(("write", "verify", "final_verify")):
            # Every final readback occurs after every first readback.
            seconds = (number if position == 0 else 10 + number if position == 1
                       else 100 + number)
            group[key] = {
                "schema": STAGE.TRANSFER_SCHEMA,
                "status": "HASH_VERIFIED_IN_THIS_SESSION_ONLY",
                "operation": "write" if position == 0 else "verify",
                "artifact": artifact, "path": path,
                "expected_bytes": size, "expected_sha256": digest,
                "streamed_sha256": digest if position == 0 else None,
                "readback_sha256": digest,
                "target_stat": copy.deepcopy(metadata),
                "identity_before": copy.deepcopy(identity),
                "identity_after": copy.deepcopy(identity),
                "process_id": 1000 + number * 3 + position,
                "process_start_ticks": 2000 + number * 3 + position,
                "utc": (start + datetime.timedelta(seconds=seconds)).isoformat(),
            }
        bundle[artifact] = group
    return bundle


class StageR2CheckoutTests(unittest.TestCase):
    def test_hold_contract_pins_implementation_and_missing_authority(self):
        contract = json.loads((HERE / "GATE_CONTRACT.json").read_text())
        self.assertIn("HOLD", contract["status"])
        for key in ("extraction_authorized", "cpu_g2_authorized",
                    "native_training_authorized", "evaluation_authorized",
                    "submission_authorized"):
            self.assertIs(contract[key], False)
        for key in ("complete_operator_transfer_receipt", "independent_transfer_result_review",
                    "fresh_full_host_preflight", "future_exact_execution_request_and_argv_sha256",
                    "independent_implementation_and_request_review",
                    "supervisor_one_shot_checkout_assignment", "operator_checkout_receipt",
                    "independent_checkout_result_review"):
            self.assertIsNone(contract["missing_gates"][key])
        implementation = contract["implementation"]
        for path_key, sha_key in (("repo_path", "sha256"),
                                  ("test_path", "test_sha256"),
                                  ("runbook_path", "runbook_sha256")):
            path = REPO / implementation[path_key]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             implementation[sha_key])
        self.assertIsNone(implementation["ssh_or_host_command_in_this_package"])
        self.assertIsNone(implementation["one_shot_source_checkout_assignment"])

    def test_frozen_sources_and_real_6338_member_scan(self):
        old = REPO / "research/v2/authorizations/trn21-c5c-a0-stage-r2"
        assignment = REPO / ("research/v2/authorizations/"
                             "fs-20260925-v2-c5c-a0-control-trn21-r1."
                             "source-stage-r2-transfer-assignment.json")
        self.assertEqual(hashlib.sha256((old / "GATE_CONTRACT.json").read_bytes()).hexdigest(),
                         STAGE.SOURCE_CONTRACT_SHA256)
        self.assertEqual(hashlib.sha256(assignment.read_bytes()).hexdigest(),
                         STAGE.TRANSFER_ASSIGNMENT_SHA256)
        capsule = Path("/dev/shm/trainium-c5c-e65-one-commit-20260925.tar.gz")
        named = capsule.lstat()
        self.assertTrue(stat.S_ISREG(named.st_mode))
        self.assertEqual(stat.S_IMODE(named.st_mode), 0o600)
        self.assertEqual(named.st_size, STAGE.CAPSULE_BYTES)
        with capsule.open("rb") as stream:
            summary, entries = STAGE.scan_archive(stream)
        self.assertEqual(summary["members"], 6338)
        self.assertEqual(summary["files"], 6283)
        self.assertEqual(summary["directories"], 55)
        self.assertEqual(summary["gnu_long_headers"], 4)
        self.assertEqual(summary["unpacked_bytes"], STAGE.EXPECTED_UNPACKED_BYTES)
        self.assertEqual(summary["manifest_sha256"],
                         "a9eff72e760f314717e712a5e3d23d300232db4feeb361896c57b16f7bdd1bf4")
        self.assertEqual(len(entries), STAGE.EXPECTED_MEMBERS)
        with capsule.open("rb") as stream:
            self.assertEqual(hashlib.sha256(stream.read()).hexdigest(), STAGE.CAPSULE_SHA256)

    def test_synthetic_scanner_rejects_unsafe_paths_types_and_pax(self):
        unsafe = {
            "absolute": safe_entries()[:2] + [("/escape", "file", b"x", 0o600)],
            "traversal": safe_entries()[:2] + [("./dir/../escape", "file", b"x", 0o600)],
            "duplicate": safe_entries() + [("./dir/hello.txt", "file", b"x", 0o600)],
            "missing_parent": safe_entries()[:1] + [("./missing/file", "file", b"x", 0o600)],
            "credential": safe_entries()[:1] + [("./id_rsa", "file", b"x", 0o600)],
            "checkpoint": safe_entries()[:1] + [("./out/final.pt", "file", b"x", 0o600)],
            "symlink": safe_entries()[:1] + [("./link", tarfile.SYMTYPE, b"", 0o777)],
            "hardlink": safe_entries()[:1] + [("./link", tarfile.LNKTYPE, b"", 0o777)],
            "fifo": safe_entries()[:1] + [("./pipe", tarfile.FIFOTYPE, b"", 0o600)],
            "setuid": safe_entries()[:1] + [("./privileged", "file", b"x", 0o4755)],
        }
        with tempfile.TemporaryDirectory(prefix="stage-r2-scan-synthetic-") as temporary:
            for label, entries in unsafe.items():
                with self.subTest(label=label):
                    path = Path(temporary, label + ".tar.gz")
                    fixture_archive(path, entries)
                    with path.open("rb") as stream, self.assertRaises(RuntimeError):
                        STAGE.scan_archive(stream, pinned=False)
            pax = Path(temporary, "pax.tar.gz")
            fixture_archive(pax, safe_entries()[:1] + [
                ("./safe", "file", b"x", 0o600, {"path": "./safe"}),
            ], fmt=tarfile.PAX_FORMAT)
            with pax.open("rb") as stream, self.assertRaises(RuntimeError):
                STAGE.scan_archive(stream, pinned=False)
            remote = Path(temporary, "remote.tar.gz")
            fixture_archive(remote, safe_entries()[:1] + [
                ("./.git", "dir", b"", 0o700),
                ("./.git/config", "file", b"[remote \"evil\"]\nurl = x\n", 0o600),
            ])
            with remote.open("rb") as stream, self.assertRaises(RuntimeError):
                STAGE.scan_archive(stream, pinned=False)

    def test_synthetic_extraction_private_no_overwrite_and_partial_preserved(self):
        with tempfile.TemporaryDirectory(prefix="stage-r2-extract-synthetic-") as temporary:
            archive_path = Path(temporary, "synthetic.tar.gz")
            fixture_archive(archive_path, safe_entries())
            with archive_path.open("rb") as stream:
                summary, manifest = STAGE.scan_archive(stream, pinned=False)
            root = Path(temporary, "checkout")
            root.mkdir(mode=0o700)
            fd = os.open(root, STAGE.OPEN_DIR)
            try:
                with archive_path.open("rb") as stream:
                    result = STAGE._extract_archive(stream, fd, manifest,
                                                    owner_uid=os.geteuid())
            finally:
                os.close(fd)
            self.assertEqual(result["members_extracted"], summary["members"])
            self.assertEqual(Path(root, "dir/hello.txt").read_bytes(), b"safe synthetic text\n")
            self.assertEqual(stat.S_IMODE(Path(root, "dir").stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(Path(root, "dir/hello.txt").stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(Path(root, "tool.sh").stat().st_mode), 0o700)

            partial = Path(temporary, "partial")
            partial.mkdir(mode=0o700)
            Path(partial, "tool.sh").write_bytes(b"existing")
            fd = os.open(partial, STAGE.OPEN_DIR)
            try:
                with archive_path.open("rb") as stream, self.assertRaises(FileExistsError):
                    STAGE._extract_archive(stream, fd, manifest, owner_uid=os.geteuid())
            finally:
                os.close(fd)
            self.assertEqual(Path(partial, "tool.sh").read_bytes(), b"existing")
            self.assertEqual(Path(partial, "dir/hello.txt").read_bytes(),
                             b"safe synthetic text\n")

    def test_symlink_collision_cannot_escape_private_checkout(self):
        with tempfile.TemporaryDirectory(prefix="stage-r2-symlink-synthetic-") as temporary:
            archive_path = Path(temporary, "synthetic.tar.gz")
            fixture_archive(archive_path, safe_entries())
            with archive_path.open("rb") as stream:
                _summary, manifest = STAGE.scan_archive(stream, pinned=False)
            outside = Path(temporary, "outside")
            outside.mkdir()
            root = Path(temporary, "checkout")
            root.mkdir(mode=0o700)
            Path(root, "dir").symlink_to(outside, target_is_directory=True)
            fd = os.open(root, STAGE.OPEN_DIR)
            try:
                with archive_path.open("rb") as stream, self.assertRaises(FileExistsError):
                    STAGE._extract_archive(stream, fd, manifest, owner_uid=os.geteuid())
            finally:
                os.close(fd)
            self.assertEqual(list(outside.iterdir()), [])
            self.assertTrue(Path(root, "dir").is_symlink())

    def test_persistence_bundle_needs_three_later_distinct_sessions(self):
        bundle = fake_bundle()
        passed = STAGE.validate_persistence_bundle(bundle)
        self.assertEqual(passed["capsule_target_stat"]["bytes"], STAGE.CAPSULE_BYTES)
        self.assertEqual(set(passed["target_stats"]), {"capsule", "preflight", "contained"})
        mutations = []
        missing = copy.deepcopy(bundle)
        del missing["preflight"]["final_verify"]
        mutations.append(missing)
        replaced = copy.deepcopy(bundle)
        replaced["capsule"]["final_verify"]["target_stat"]["inode"] += 1
        mutations.append(replaced)
        same_process = copy.deepcopy(bundle)
        same_process["contained"]["verify"]["process_id"] = bundle["capsule"]["write"]["process_id"]
        same_process["contained"]["verify"]["process_start_ticks"] = bundle["capsule"]["write"]["process_start_ticks"]
        mutations.append(same_process)
        changed_boot = copy.deepcopy(bundle)
        changed_boot["preflight"]["final_verify"]["identity_after"]["boot_id"] = "other"
        mutations.append(changed_boot)
        early_final = copy.deepcopy(bundle)
        early_final["capsule"]["final_verify"]["utc"] = (
            datetime.datetime(2026, 9, 25, 1, 0, 12, tzinfo=datetime.timezone.utc).isoformat())
        mutations.append(early_final)
        missing_mount = copy.deepcopy(bundle)
        for operation in ("write", "verify", "final_verify"):
            del missing_mount["capsule"][operation]["identity_before"]["mount_namespace"]
            del missing_mount["capsule"][operation]["identity_after"]["mount_namespace"]
        mutations.append(missing_mount)
        for candidate in mutations:
            with self.subTest(index=mutations.index(candidate)), self.assertRaises(RuntimeError):
                STAGE.validate_persistence_bundle(candidate)

    def test_transferred_cpu_tool_is_rehashed_against_final_inode(self):
        with tempfile.TemporaryDirectory(prefix="stage-r2-tool-synthetic-") as temporary:
            tool = Path(temporary, "preflight.py")
            original = b"synthetic tool"
            tool.write_bytes(original)
            tool.chmod(0o600)
            info = tool.stat()
            expected_stat = {"device": info.st_dev, "inode": info.st_ino,
                             "bytes": info.st_size, "uid": info.st_uid,
                             "gid": info.st_gid, "mode": "0o600", "nlink": 1}
            artifacts = {"preflight": (str(tool), len(original),
                                       hashlib.sha256(original).hexdigest())}
            fd = os.open(temporary, STAGE.OPEN_DIR)
            try:
                with patch.object(STAGE, "TRANSFER_ARTIFACTS", artifacts), \
                        patch.object(STAGE, "UID", os.geteuid()):
                    checked = STAGE._verify_transferred_tool(fd, "preflight", expected_stat)
                    self.assertEqual(checked["target_stat"], expected_stat)
                    tool.write_bytes(b"other tool----")
                    with self.assertRaises(RuntimeError):
                        STAGE._verify_transferred_tool(fd, "preflight", expected_stat)
            finally:
                os.close(fd)

    def test_protected_snapshot_detects_drift_without_writing_assets(self):
        with tempfile.TemporaryDirectory(prefix="stage-r2-assets-synthetic-") as temporary:
            root = Path(temporary)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"fixed synthetic checkpoint")
            cache = root / "cache"
            cache.mkdir()
            children = {"": cache.stat().st_ino}
            for name in ("neff", "local", "hlo", "inductor", "compiler", "debug"):
                child = cache / name
                child.mkdir()
                children[name] = child.stat().st_ino
            info = checkpoint.stat()
            spec = ((str(checkpoint), info.st_size,
                     hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     info.st_dev, info.st_ino),)
            with patch.object(STAGE, "PROTECTED_CHECKPOINTS", spec), \
                    patch.object(STAGE, "CACHE_ROOT", str(cache)), \
                    patch.object(STAGE, "PROTECTED_CACHE_INODES", children), \
                    patch.object(STAGE, "PROTECTED_CACHE_DEVICE", cache.stat().st_dev), \
                    patch.object(STAGE, "UID", os.geteuid()):
                before = STAGE.snapshot_protected_assets()
                self.assertEqual(len(before["warm_cache_directories"]), 7)
                checkpoint.write_bytes(b"changed synthetic checkpoint")
                with self.assertRaises(RuntimeError):
                    STAGE.snapshot_protected_assets()

    def test_no_dispatch_entry_point_or_cleanup_calls(self):
        tree = ast.parse(SOURCE.read_text())
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertTrue({"unlink", "remove", "rename", "replace", "rmdir", "rmtree",
                         "system", "popen"}.isdisjoint(called))
        with patch.object(STAGE.os, "open", side_effect=AssertionError("filesystem touched")):
            with self.assertRaises(RuntimeError):
                STAGE.stage_fixed_checkout({})
        result = subprocess.run(["python3", "-B", str(SOURCE)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DISPATCH HOLD", result.stderr)


if __name__ == "__main__":
    unittest.main()
