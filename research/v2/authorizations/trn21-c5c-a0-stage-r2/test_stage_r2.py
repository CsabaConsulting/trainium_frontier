"""Local-only synthetic checks. No SSH, host stage, CPU G2 or training."""

import ast
import copy
import datetime
import hashlib
import importlib.util
import io
import json
import os
import shlex
import stat
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WRITER = load("trn21_stage_r2_remote_writer_test", HERE / "stage_r2_remote_writer.py")
TRANSPORT = load("trn21_stage_r2_transport_test", HERE / "stage_r2_transport.py")


class StageR2LocalTests(unittest.TestCase):
    def test_local_capsule_member_scan_is_pinned_and_safe(self):
        capsule = TRANSPORT.ARTIFACTS["capsule"]["source"]
        seen = set()
        members = 0
        git_config_seen = False
        with tarfile.open(capsule, mode="r:gz") as archive:
            for member in archive:
                members += 1
                self.assertTrue(member.isfile() or member.isdir())
                self.assertFalse(member.issym() or member.islnk())
                self.assertFalse({"path", "linkpath"}.intersection(member.pax_headers))
                self.assertEqual(member.mode & 0o6000, 0)
                raw = member.name
                self.assertFalse(raw.startswith("/"))
                components = PurePosixPath(raw).parts
                self.assertNotIn("..", components)
                normalized = str(PurePosixPath(raw))
                self.assertNotIn(normalized, seen)
                seen.add(normalized)
                if not components:
                    self.assertEqual(normalized, ".")
                    self.assertTrue(member.isdir())
                    continue
                lowered = [part.lower() for part in components]
                basename = lowered[-1]
                self.assertNotIn(".aws", lowered)
                self.assertNotIn(basename, {"id_rsa", "id_ed25519", "credentials", "config.json"})
                self.assertFalse(basename.endswith((".pem", ".key", ".p12", ".pfx")))
                self.assertFalse(basename.startswith(".env") and basename != ".env.example")
                if normalized.startswith(".git/hooks/"):
                    self.assertTrue(basename.endswith(".sample"))
                if normalized == ".git/config":
                    git_config_seen = True
                    config = archive.extractfile(member).read().lower()
                    for forbidden in (b"[remote ", b"[filter ", b"[include]", b"[includeif ",
                                      b"credential", b"hookspath"):
                        self.assertNotIn(forbidden, config)
        self.assertEqual(members, 6338)
        self.assertTrue(git_config_seen)

    def test_exact_capsule_tools_and_e65_source_binding(self):
        contract = json.loads((HERE / "GATE_CONTRACT.json").read_text())
        for artifact in ("capsule", "preflight", "contained"):
            checked = TRANSPORT.checked_local_source(artifact)
            self.assertEqual(checked["target"], contract["persistent_artifacts"][artifact]["host_path"])
            self.assertEqual(checked["bytes"], contract["persistent_artifacts"][artifact]["bytes"])
            self.assertEqual(checked["sha256"], contract["persistent_artifacts"][artifact]["sha256"])
            self.assertEqual(checked["target"], WRITER.ROOT + "/" + WRITER.ARTIFACTS[artifact]["name"])
        self.assertEqual(hashlib.sha256((HERE / "stage_r2_remote_writer.py").read_bytes()).hexdigest(),
                         TRANSPORT.WRITER_SHA256)
        self.assertEqual(contract["writer"]["sha256"], TRANSPORT.WRITER_SHA256)
        self.assertEqual(hashlib.sha256((HERE / "stage_r2_transport.py").read_bytes()).hexdigest(),
                         contract["writer"]["command_builder_sha256"])
        for name, digest in (("operator_hold", "operator_hold_sha256"),
                             ("read_only_addendum", "read_only_addendum_sha256")):
            referenced = REPO / contract["prior_stage_failure"][name + "_path"]
            self.assertEqual(hashlib.sha256(referenced.read_bytes()).hexdigest(),
                             contract["prior_stage_failure"][digest])
        for source_path, expected in (("train.py", contract["immutable_science"]["train_sha256"]),
                                      ("prepare.py", contract["immutable_science"]["prepare_sha256"])):
            raw = subprocess.check_output(["git", "show", contract["immutable_science"]["source_commit"]
                                           + ":" + source_path], cwd=REPO)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)

    def test_exact_ssh_argv_is_quoted_static_source_and_never_executes(self):
        source = TRANSPORT.reviewed_writer_source()
        for operation in ("write", "verify"):
            for artifact in ("capsule", "preflight", "contained"):
                argv = TRANSPORT.build_ssh_argv(operation, artifact)
                self.assertEqual(argv[0], "ssh")
                self.assertEqual(argv[-2], TRANSPORT.ENDPOINT)
                self.assertEqual(shlex.split(argv[-1]),
                                 ["exec", "python3", "-c", source, operation, artifact])
                self.assertEqual(len(TRANSPORT.command_sha256(argv)), 64)
                self.assertIn("StrictHostKeyChecking=yes", argv)
                self.assertIn("BatchMode=yes", argv)
                self.assertIn("IdentitiesOnly=yes", argv)
        with self.assertRaises(ValueError):
            TRANSPORT.build_ssh_argv("write; touch /tmp/bad", "capsule")
        with self.assertRaises(ValueError):
            TRANSPORT.build_ssh_argv("write", "../capsule")

    def test_writer_has_no_delete_or_shell_process_calls(self):
        tree = ast.parse((HERE / "stage_r2_remote_writer.py").read_text())
        banned = {"unlink", "remove", "rename", "replace", "rmdir", "system", "popen", "subprocess", "chdir"}
        called = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called.append(node.func.attr)
        self.assertTrue(banned.isdisjoint(called))
        source = (HERE / "stage_r2_remote_writer.py").read_text()
        self.assertIn("os.O_EXCL", source)
        self.assertIn("os.O_NOFOLLOW", source)
        self.assertIn("os.fsync(fd)", source)
        self.assertIn("os.fsync(parent_fd)", source)

    def test_synthetic_exclusive_write_readback_persistence_and_no_cleanup(self):
        body = b"stage-r2-synthetic\x00body"
        digest = hashlib.sha256(body).hexdigest()
        with tempfile.TemporaryDirectory(prefix="trn21-stage-r2-local-") as root:
            fake_spec = {"name": "synthetic.bin", "bytes": len(body), "sha256": digest}

            def local_identity(parent_fd):
                parent = os.fstat(parent_fd)
                return {"hostname": WRITER.HOST, "uid": WRITER.UID, "gid": 1000,
                        "boot_id": "synthetic-boot", "mount_namespace": "mnt:[synthetic]",
                        "mount": {"filesystem": "ext4", "mount_id": "synthetic"},
                        "parent_device": parent.st_dev, "parent_inode": parent.st_ino,
                        "parent_uid": parent.st_uid, "parent_gid": parent.st_gid,
                        "parent_mode": oct(stat.S_IMODE(parent.st_mode)),
                        "root_free_bytes": 60000000000}

            real_open = os.open
            target_flags = []

            def recording_open(path, flags, *args, **kwargs):
                if path == "synthetic.bin":
                    target_flags.append(flags)
                return real_open(path, flags, *args, **kwargs)

            with patch.object(WRITER, "ROOT", root), patch.object(WRITER, "ARTIFACTS", {"tiny": fake_spec}), \
                    patch.object(WRITER, "_identity", side_effect=local_identity), \
                    patch.object(WRITER.os, "open", side_effect=recording_open):
                written = WRITER.write_artifact("tiny", io.BytesIO(body))
                self.assertEqual(written["readback_sha256"], digest)
                self.assertEqual(written["streamed_sha256"], digest)
                self.assertEqual(written["target_stat"]["mode"], "0o600")
                self.assertEqual(Path(root, "synthetic.bin").read_bytes(), body)
                self.assertTrue(target_flags[0] & os.O_EXCL)
                self.assertTrue(target_flags[0] & os.O_NOFOLLOW)
                with self.assertRaises(FileExistsError):
                    WRITER.write_artifact("tiny", io.BytesIO(body))
                checked = WRITER.verify_artifact("tiny")
                self.assertEqual(checked["target_stat"], written["target_stat"])
                self.assertEqual(checked["readback_sha256"], digest)
                self.assertEqual(Path(root, "synthetic.bin").read_bytes(), body)
                with self.assertRaises(ValueError):
                    TRANSPORT.compare_cross_session(written, checked)  # same test process is not a new SSH session
                separate = copy.deepcopy(checked)
                separate["process_start_ticks"] += 1
                separate["utc"] = (datetime.datetime.fromisoformat(written["utc"])
                                   + datetime.timedelta(seconds=1)).isoformat()
                result = TRANSPORT.compare_cross_session(written, separate)
                self.assertEqual(result["status"], "CROSS_SESSION_PERSISTENCE_HASH_PASS")
                separate["identity_before"]["boot_id"] = "other-boot"
                with self.assertRaises(ValueError):
                    TRANSPORT.compare_cross_session(written, separate)

        with tempfile.TemporaryDirectory(prefix="trn21-stage-r2-fail-local-") as root:
            fake_spec = {"name": "partial.bin", "bytes": len(body), "sha256": digest}
            with patch.object(WRITER, "ROOT", root), patch.object(WRITER, "ARTIFACTS", {"tiny": fake_spec}), \
                    patch.object(WRITER, "_identity", side_effect=local_identity):
                with self.assertRaises(RuntimeError):
                    WRITER.write_artifact("tiny", io.BytesIO(b"incorrect"))
                self.assertEqual(Path(root, "partial.bin").read_bytes(), b"incorrect")

    def test_static_hold_and_no_collision_outside_new_package(self):
        contract = json.loads((HERE / "GATE_CONTRACT.json").read_text())
        self.assertIn("HOLD", contract["status"])
        for key in ("source_stage_authorized", "cpu_g2_authorized", "native_training_authorized",
                    "fresh_eval_authorized", "submission_authorized"):
            self.assertIs(contract[key], False)
        for artifact, spec in contract["persistent_artifacts"].items():
            self.assertTrue(spec["host_path"].startswith("/home/ubuntu/"))
            self.assertNotEqual(spec["host_path"],
                                contract["prior_stage_failure"]["consumed_host_archive_target_do_not_retry"])
            search = subprocess.run(["git", "grep", "--cached", "-l", "-F", spec["host_path"],
                                     "--", ":!research/v2/authorizations/trn21-c5c-a0-stage-r2/"],
                                    cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(search.returncode, 1, (artifact, search.stdout, search.stderr))
        targets = [spec["host_path"] for spec in contract["persistent_artifacts"].values()]
        targets.extend((contract["source_scan_and_checkout_after_verified_persistence_only"]["new_checkout_must_be_absent"],
                        contract["cpu_g2"]["planned_new_root_must_be_absent"]))
        self.assertEqual(len(targets), len(set(targets)))
        self.assertIsNone(contract["stage_result_receipt"])
        self.assertIsNone(contract["independent_stage_review"])
        self.assertIsNone(contract["cpu_g2"]["independent_g2_receipt"])


if __name__ == "__main__":
    unittest.main()
