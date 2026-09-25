"""Local-only synthetic checks for the trn22 r4 source-capsule builder."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("trn22_r4_capsule_builder",
                                              HERE / "build_source_capsule.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
EXPECTED_MANIFEST_SHA256 = "7ec8798c3e6d0284a4514a15a0caeb17bc5eb2cb5df935fb7dd6e8f3aabee42a"


class BuilderTests(unittest.TestCase):
    def test_frozen_manifest_is_complete_and_safe(self):
        path = HERE / "MEMBERS.jsonl"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         EXPECTED_MANIFEST_SHA256)
        records = [json.loads(line) for line in path.read_bytes().splitlines()]
        self.assertEqual(len(records), 6368)
        seen = set()
        directories = set()
        files = 0
        total = 0
        for row in records:
            self.assertEqual(set(row), {"path", "type", "mode", "bytes", "sha256"})
            self.assertNotIn(row["path"], seen)
            BUILDER._safe_path(row["path"], row["type"] == "directory")
            parent = row["path"].rsplit("/", 1)[0] if "/" in row["path"] else "."
            if row["path"] != ".":
                self.assertIn(parent, directories)
            seen.add(row["path"])
            if row["type"] == "directory":
                self.assertEqual(row["mode"], "0o700")
                self.assertEqual((row["bytes"], row["sha256"]), (0, None))
                directories.add(row["path"])
            else:
                self.assertIn(row["mode"], ("0o600", "0o700"))
                self.assertEqual(len(row["sha256"]), 64)
                files += 1
                total += row["bytes"]
        self.assertEqual((files, len(directories), total), (6305, 63, 188917148))
        by_path = {row["path"]: row for row in records}
        for relative, digest in BUILDER.EXPECTED_SHA256.items():
            self.assertEqual(by_path[relative]["sha256"], digest)

    def test_path_gate_rejects_credentials_data_traversal_and_active_hooks(self):
        unsafe = ("../escape", "a/../escape", "/abs", "a//b", "a/./b",
                  "data/secret.txt", "out/final.pt", ".aws/credentials",
                  "id_ed25519", ".env", ".git/hooks/pre-commit",
                  ".git/objects/info/alternates", ".git/config.worktree")
        for name in unsafe:
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                BUILDER._safe_path(name, False)

    def test_synthetic_marker_aborts_without_echoing_value(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r4-marker-test-") as temporary:
            path = Path(temporary, "fixture.txt")
            path.write_bytes(b"synthetic-only-marker")
            with patch.object(BUILDER, "SECRET_MARKERS",
                              {"synthetic": re.compile(rb"synthetic-only-marker")}):
                with self.assertRaisesRegex(RuntimeError, "high-confidence secret marker"):
                    BUILDER._digest_file(path, "fixture.txt")

    def test_local_capsule_is_exclusive_and_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r4-exclusive-test-") as temporary:
            capsule = Path(temporary, "capsule.tar.gz")
            manifest = Path(temporary, "MEMBERS.jsonl")
            capsule.write_bytes(b"existing synthetic artifact")
            with patch.object(BUILDER, "LOCAL_CAPSULE", capsule), \
                    patch.object(BUILDER, "MEMBER_MANIFEST", manifest):
                with self.assertRaises(FileExistsError):
                    BUILDER._make_capsule([(".", "directory", Path(temporary), 0o700)], {})
            self.assertEqual(capsule.read_bytes(), b"existing synthetic artifact")

    def test_raw_tar_scan_rejects_links_devices_and_pax(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r4-unsafe-tar-") as temporary:
            for label, kind, fmt in (
                ("symlink", tarfile.SYMTYPE, tarfile.GNU_FORMAT),
                ("hardlink", tarfile.LNKTYPE, tarfile.GNU_FORMAT),
                ("character", tarfile.CHRTYPE, tarfile.GNU_FORMAT),
                ("block", tarfile.BLKTYPE, tarfile.GNU_FORMAT),
                ("pax", tarfile.REGTYPE, tarfile.PAX_FORMAT),
            ):
                with self.subTest(label=label):
                    archive = Path(temporary, label + ".tar.gz")
                    with tarfile.open(archive, "w:gz", format=fmt) as tar:
                        root = tarfile.TarInfo(".")
                        root.type = tarfile.DIRTYPE
                        root.mode = 0o700
                        tar.addfile(root)
                        member = tarfile.TarInfo("./unsafe")
                        member.type = kind
                        member.mode = 0o600
                        if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                            member.linkname = "./elsewhere"
                        if kind == tarfile.REGTYPE:
                            member.size = 1
                            member.pax_headers = {"path": "./unsafe"}
                            tar.addfile(member, io.BytesIO(b"x"))
                        else:
                            tar.addfile(member)
                    with patch.object(BUILDER, "LOCAL_CAPSULE", archive):
                        with self.assertRaises(RuntimeError):
                            BUILDER._raw_scan()

    def test_package_contains_no_host_dispatch_call(self):
        source = (HERE / "build_source_capsule.py").read_text()
        for forbidden in ("subprocess.Popen(", "os.system(", "paramiko.",
                          "boto3.", "ssh -", "scp ", "rsync "):
            self.assertNotIn(forbidden, source)
        self.assertEqual(BUILDER.SOURCE_COMMIT,
                         "446b657695f086f9b484fa601c9a3b61c2495912")

    def test_complete_artifact_pins_fail_closed(self):
        BUILDER._require_frozen_artifacts(BUILDER.EXPECTED_CAPSULE,
                                          BUILDER.EXPECTED_MANIFEST)
        with self.assertRaises(RuntimeError):
            BUILDER._require_frozen_artifacts((0, BUILDER.EXPECTED_CAPSULE[1]),
                                              BUILDER.EXPECTED_MANIFEST)
        with self.assertRaises(RuntimeError):
            BUILDER._require_frozen_artifacts(BUILDER.EXPECTED_CAPSULE,
                                              (BUILDER.EXPECTED_MANIFEST[0], "0" * 64))

    def test_hold_contract_binds_package_files_and_never_dispatches(self):
        contract = json.loads((HERE / "GATE_CONTRACT.json").read_text())
        for field in ("source_transfer_authorized", "source_checkout_authorized",
                      "cpu_g2_authorized", "native_training_authorized",
                      "fresh_eval_authorized", "submission_authorized"):
            self.assertIs(contract[field], False)
        artifact = contract["local_exact_source_artifact"]
        self.assertEqual((artifact["bytes"], artifact["sha256"]),
                         BUILDER.EXPECTED_CAPSULE)
        self.assertEqual(contract["frozen_source_commit"], BUILDER.SOURCE_COMMIT)
        self.assertEqual(contract["frozen_source_sha256"], BUILDER.EXPECTED_SHA256)
        package = contract["local_package_files"]
        self.assertEqual((package["manifest_bytes"], package["manifest_sha256"]),
                         BUILDER.EXPECTED_MANIFEST)
        for stem in ("manifest", "builder", "tests", "runbook"):
            path = HERE / Path(package[stem + "_path"]).name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             package[stem + "_sha256"])
        targets = contract["proposed_trn22_targets_not_created"]
        self.assertTrue(targets["transfer_only_archive"].startswith("/home/ubuntu/"))
        self.assertTrue(targets["host_dev_shm_target_forbidden"])


if __name__ == "__main__":
    unittest.main()
