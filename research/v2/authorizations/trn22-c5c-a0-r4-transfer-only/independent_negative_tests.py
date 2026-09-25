"""Independent offline probes of the frozen fcaf3780 trn22 transfer draft.

No test opens SSH, reads a private key, or contacts/changes the host.
"""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_once as operator
import stage_r4_remote as remote
import stage_r4_transport as transport


def digest(data):
    return hashlib.sha256(data).hexdigest()


class IndependentNegativeTests(unittest.TestCase):
    def test_frozen_request_is_not_dispatchable(self):
        with mock.patch.object(operator.subprocess, "Popen", side_effect=AssertionError("SSH opened")):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()

    def test_capsule_is_exact_but_package_head_is_not_source_head(self):
        artifact = transport.checked_local_source()
        self.assertEqual(artifact["bytes"], 104154875)
        self.assertEqual(artifact["sha256"], "767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a")
        with tarfile.open(transport.SOURCE, mode="r:gz") as archive:
            for member in ("./.git/HEAD", "./.git/shallow"):
                self.assertEqual(archive.extractfile(member).read().strip(),
                                 b"446b657695f086f9b484fa601c9a3b61c2495912")
            self.assertEqual(digest(archive.extractfile("./train.py").read()),
                             "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586")
            self.assertEqual(digest(archive.extractfile("./prepare.py").read()),
                             "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb")
        self.assertEqual(remote.HOST, "ip-172-31-32-253")
        self.assertEqual(remote.TARGET, transport.TARGET)

    def test_fixed_journal_consumes_one_attempt_before_any_ssh(self):
        with tempfile.TemporaryDirectory() as directory:
            journal_path = Path(directory) / "one-attempt.jsonl"
            with mock.patch.object(operator, "JOURNAL", journal_path):
                journal = operator.Journal()
                journal.emit("ONE_ATTEMPT_CONSUMED")
                journal.close()
                self.assertEqual(stat.S_IMODE(journal_path.stat().st_mode), 0o600)
                with self.assertRaises(FileExistsError):
                    operator.Journal()

    def test_missing_explicit_host_key_and_config_pin_reproduced(self):
        argv = transport.build_ssh_argv("probe")
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertNotIn("-F", argv)
        self.assertFalse(any("UserKnownHostsFile=" in part for part in argv))
        self.assertFalse(any("GlobalKnownHostsFile=" in part for part in argv))
        self.assertFalse(any("HostKeyAlgorithms=" in part for part in argv))
        self.assertFalse(any("Fingerprint" in part for part in argv))

    def test_local_stop_is_ignored_by_breaker_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.json"
            state.write_text(json.dumps({"breakers": {
                "global_correctness": {"tripped": False},
                "trn22": {"tripped": False}}}), encoding="utf-8")
            (root / "STOP").write_text("operator stop", encoding="utf-8")
            with mock.patch.object(operator, "CONTROL_STATE", state):
                self.assertEqual(operator._check_controller_breakers()["trn22_tripped"], False)

    def test_self_declared_pass_and_assignment_without_request_hash_are_accepted(self):
        frozen = json.loads(operator.REQUEST.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path, review_path, assignment_path = (
                root / "request.json", root / "review.md", root / "assignment.json")
            review = b"PASS: this text is not an independent review.\n"
            review_path.write_bytes(review)
            assignment = {
                "dispatch_authorized": True,
                "scope": frozen["assignment_scope"],
                "host": "trn22",
                "source_commit": frozen["source_commit"],
                "capsule_sha256": frozen["capsule_sha256"],
                "target": frozen["target"],
                "one_attempt": True,
            }
            assignment_bytes = json.dumps(assignment).encode("utf-8")
            assignment_path.write_bytes(assignment_bytes)
            forged = copy.deepcopy(frozen)
            forged.update({
                "dispatch_authorized": True,
                "independent_review_status": "PASS",
                "independent_review_sha256": digest(review),
                "assignment_sha256": digest(assignment_bytes),
            })
            request_path.write_text(json.dumps(forged), encoding="utf-8")
            with (mock.patch.object(operator, "REQUEST", request_path),
                  mock.patch.object(operator, "INDEPENDENT_REVIEW", review_path),
                  mock.patch.object(operator, "ASSIGNMENT", assignment_path)):
                accepted_request, accepted_assignment = operator.authorize()
            self.assertTrue(accepted_request["dispatch_authorized"])
            self.assertNotIn("request_sha256", accepted_assignment)
            self.assertNotIn("independent_review_sha256", accepted_assignment)

    def test_writer_is_exclusive_private_fsynced_and_read_back_on_local_fixture(self):
        payload = b"local-only capsule fixture"
        with tempfile.TemporaryDirectory() as directory:
            name = "new-capsule.bin"
            target = Path(directory) / name
            parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            identity = {"hostname": remote.HOST, "uid": remote.UID,
                        "boot_id": "fixture", "mount_namespace": "mnt:[fixture]",
                        "mount": {"source": "/dev/root"}, "ancestry": {"ubuntu": {"inode": 1}},
                        "root_free_bytes": 30_000_000_000,
                        "root_free_inodes": 100_000}
            fsynced = []
            original_fsync = os.fsync

            def recorded_fsync(fd):
                fsynced.append(stat.S_ISDIR(os.fstat(fd).st_mode))
                return original_fsync(fd)

            with (mock.patch.object(remote, "TARGET", str(target)),
                  mock.patch.object(remote, "TARGET_NAME", name),
                  mock.patch.object(remote, "BYTES", len(payload)),
                  mock.patch.object(remote, "SHA256", digest(payload)),
                  mock.patch.object(remote, "PRE_FREE", 0),
                  mock.patch.object(remote, "POST_FREE", 0),
                  mock.patch.object(remote, "_open_parent", side_effect=lambda: (os.dup(parent), {})),
                  mock.patch.object(remote, "_identity", return_value=identity),
                  mock.patch.object(remote, "_safe_path", side_effect=lambda path: {"exists": os.path.lexists(path)}),
                  mock.patch.object(remote.os, "fsync", side_effect=recorded_fsync)):
                result = remote.write(io.BytesIO(payload))
                self.assertEqual(result["readback_sha256"], digest(payload))
                self.assertEqual(result["target_stat"]["mode"], "0o600")
                self.assertIn(False, fsynced)  # file fd
                self.assertIn(True, fsynced)   # parent dir fd
                with self.assertRaises(FileExistsError):
                    remote.write(io.BytesIO(payload))
            os.close(parent)
            self.assertEqual(target.read_bytes(), payload)

    def test_partial_write_is_preserved_and_not_overwritten(self):
        payload = b"too short"
        with tempfile.TemporaryDirectory() as directory:
            name = "new-capsule.bin"
            target = Path(directory) / name
            parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            identity = {"hostname": remote.HOST, "uid": remote.UID,
                        "boot_id": "fixture", "mount_namespace": "mnt:[fixture]",
                        "mount": {"source": "/dev/root"}, "ancestry": {"ubuntu": {"inode": 1}},
                        "root_free_bytes": 30_000_000_000,
                        "root_free_inodes": 100_000}
            with (mock.patch.object(remote, "TARGET", str(target)),
                  mock.patch.object(remote, "TARGET_NAME", name),
                  mock.patch.object(remote, "BYTES", len(payload) + 1),
                  mock.patch.object(remote, "SHA256", "0" * 64),
                  mock.patch.object(remote, "PRE_FREE", 0),
                  mock.patch.object(remote, "POST_FREE", 0),
                  mock.patch.object(remote, "_open_parent", side_effect=lambda: (os.dup(parent), {})),
                  mock.patch.object(remote, "_identity", return_value=identity),
                  mock.patch.object(remote, "_safe_path", side_effect=lambda path: {"exists": os.path.lexists(path)})):
                with self.assertRaisesRegex(RuntimeError, "partial target retained"):
                    remote.write(io.BytesIO(payload))
                self.assertEqual(target.read_bytes(), payload)
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
                with self.assertRaises(FileExistsError):
                    remote.write(io.BytesIO(payload))
            os.close(parent)

    def test_preflight_exception_still_attempts_separate_postflight(self):
        calls, events = [], []

        class Journal:
            def emit(self, event, **details):
                events.append(event)

            def close(self):
                events.append("CLOSED")

        def ssh(operation, journal):
            calls.append(operation)
            raise RuntimeError("synthetic transport failure")

        with (mock.patch.object(operator, "authorize", return_value=({"assignment_sha256": "0" * 64}, {})),
              mock.patch.object(operator.transport, "checked_local_source", return_value={}),
              mock.patch.object(operator, "_check_controller_breakers", return_value={}),
              mock.patch.object(operator, "Journal", Journal),
              mock.patch.object(operator, "_run_ssh", side_effect=ssh)):
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                operator.run_once()
        self.assertEqual(calls, ["probe", "probe"])
        self.assertIn("POSTFLIGHT_HOLD", events)


if __name__ == "__main__":
    unittest.main()
