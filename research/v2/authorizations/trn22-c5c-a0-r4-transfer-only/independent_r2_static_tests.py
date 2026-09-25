"""Independent, local-only adversarial tests for repaired r4 transfer draft.

No SSH connection, host mutation, private-key read, transfer, or checkout.
All writer fixtures are created under TemporaryDirectory on the controller.
"""

import base64
import copy
import hashlib
import io
import json
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
import operator_transfer_once as operator
import stage_r4_remote as remote
import stage_r4_transport as transport


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


class IndependentRepairedDraftTests(unittest.TestCase):
    def test_disabled_request_and_flag_flip_stop_before_ssh(self):
        with mock.patch.object(operator.subprocess, "Popen", side_effect=AssertionError("SSH spawned")):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()
        forged = json.loads(operator.REQUEST.read_text())
        forged["dispatch_authorized"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "enabled.json"
            path.write_bytes(encoded(forged))
            with mock.patch.object(operator, "REQUEST", path):
                with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                    operator.authorize()

    def test_literal_ssh_configuration_and_host_key_are_pinned(self):
        identity = transport.checked_ssh_identity()
        self.assertEqual(identity["known_hosts_sha256"], transport.KNOWN_HOSTS_SHA256)
        self.assertEqual(identity["host_fingerprint"], transport.HOST_FINGERPRINT)
        argv = transport.build_ssh_argv("probe")
        self.assertEqual(argv[:4], ["ssh", "-F", "/dev/null", "-T"])
        self.assertEqual(argv[-2], transport.ENDPOINT)
        for option in ("UserKnownHostsFile=" + str(transport.KNOWN_HOSTS),
                       "GlobalKnownHostsFile=/dev/null", "HostKeyAlgorithms=ssh-ed25519",
                       "UpdateHostKeys=no", "VerifyHostKeyDNS=no", "BatchMode=yes",
                       "IdentitiesOnly=yes", "StrictHostKeyChecking=yes"):
            self.assertIn(option, argv)
        self.assertEqual(shlex.split(argv[-1]),
                         ["exec", "python3", "-c", transport.reviewed_remote_source(), "probe"])
        with mock.patch.object(transport, "KNOWN_HOSTS_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "known-hosts bytes changed"):
                transport.build_ssh_argv("probe")
        with mock.patch.object(transport, "HOST_FINGERPRINT", "SHA256:" + "0" * 43):
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                transport.build_ssh_argv("probe")

    def test_every_controller_and_worktree_stop_blocks_clear_breakers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "controller/research/v2/state.json"
            state.parent.mkdir(parents=True)
            state.write_bytes(encoded({"breakers": {"global_correctness": {"tripped": False},
                                                    "trn22": {"tripped": False}}}))
            repo = root / "isolated"
            (repo / "research/agentic").mkdir(parents=True)
            (repo / "research/v2").mkdir(parents=True)
            stops = (state.parent / "STOP", state.parent.parent / "agentic/STOP",
                     repo / "research/agentic/STOP", repo / "research/v2/STOP")
            with (mock.patch.object(operator, "CONTROL_STATE", state),
                  mock.patch.object(operator, "REPO", repo)):
                self.assertTrue(operator._check_controller_breakers()["local_stops_clear"])
                for path in stops:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("STOP")
                    with self.assertRaisesRegex(RuntimeError, "local STOP"):
                        operator._check_controller_breakers()
                    path.unlink()
                state.write_bytes(encoded({"breakers": {"global_correctness": {"tripped": True},
                                                    "trn22": {"tripped": False}}}))
                with self.assertRaisesRegex(RuntimeError, "breaker"):
                    operator._check_controller_breakers()

    def test_real_review_and_assignment_bytes_form_one_way_hash_chain(self):
        request = json.loads(operator.REQUEST.read_text())
        request["dispatch_authorized"] = True
        request_bytes = encoded(request)
        operator_sha = digest(Path(operator.__file__).read_bytes())
        transport_sha = digest(Path(transport.__file__).read_bytes())
        review = {"schema": "trn22-r4-independent-transfer-enable-review-v1",
                  "verdict": "PASS", "reviewer_role": "independent_verifier",
                  "request_sha256": digest(request_bytes), "operator_sha256": operator_sha,
                  "transport_sha256": transport_sha, "remote_sha256": transport.REMOTE_SHA256,
                  "capsule_sha256": transport.SOURCE_SHA256,
                  "source_commit": request["source_commit"], "host": "trn22",
                  "scope": request["assignment_scope"], "no_host_action_by_reviewer": True}
        review_bytes = encoded(review)
        assignment = {"schema": "trn22-r4-supervisor-transfer-only-one-attempt-v1",
                      "issuer_role": "supervisor", "dispatch_authorized": True,
                      "scope": request["assignment_scope"], "host": "trn22",
                      "request_sha256": digest(request_bytes),
                      "independent_review_sha256": digest(review_bytes),
                      "operator_sha256": operator_sha, "transport_sha256": transport_sha,
                      "remote_sha256": transport.REMOTE_SHA256,
                      "source_commit": request["source_commit"],
                      "capsule_sha256": transport.SOURCE_SHA256,
                      "target": transport.TARGET, "journal": str(operator.JOURNAL),
                      "one_attempt": True}
        self.assertEqual(operator._validate_authority_bundle(
            request_bytes, review_bytes, encoded(assignment))[1], assignment)
        changed = copy.deepcopy(review)
        changed["request_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "structured PASS"):
            operator._validate_authority_bundle(request_bytes, encoded(changed), encoded(assignment))
        changed = copy.deepcopy(assignment)
        changed["independent_review_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "assignment mismatch"):
            operator._validate_authority_bundle(request_bytes, review_bytes, encoded(changed))
        changed = copy.deepcopy(request)
        changed["independent_review_sha256"] = digest(review_bytes)
        with self.assertRaisesRegex(RuntimeError, "self-attest"):
            operator._validate_authority_bundle(encoded(changed), review_bytes, encoded(assignment))

    def test_remote_alias_requires_matching_kernel_rdev_and_sysfs(self):
        class Block:
            st_mode = stat.S_IFBLK | 0o600
            st_rdev = os.makedev(259, 1)
            st_dev = 5
            st_ino = 99
        def named(path):
            if path == "/dev/root":
                raise FileNotFoundError(path)
            return Block()
        with (mock.patch.object(remote.os, "lstat", side_effect=named),
              mock.patch.object(remote.os, "stat", return_value=Block()),
              mock.patch.object(remote.os, "readlink", return_value="../../devices/nvme0n1p1"),
              mock.patch.object(remote.os.path, "realpath",
                                return_value="/sys/devices/pci/nvme0n1p1")):
            aliases = remote._block_aliases({"major_minor": "259:1"})
            self.assertEqual(aliases["historical_names"]["/dev/nvme0n1p1"]
                             ["resolved_rdev_major_minor"], "259:1")
        class Other(Block):
            st_rdev = os.makedev(259, 2)
        with (mock.patch.object(remote.os, "lstat", side_effect=named),
              mock.patch.object(remote.os, "stat", return_value=Other())):
            with self.assertRaisesRegex(RuntimeError, "different block device"):
                remote._block_aliases({"major_minor": "259:1"})

    def test_exclusive_private_journal_and_partial_writer_preserve_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one-attempt.jsonl"
            with mock.patch.object(operator, "JOURNAL", path):
                journal = operator.Journal()
                journal.emit("ONE_ATTEMPT_CONSUMED")
                journal.close()
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                with self.assertRaises(FileExistsError):
                    operator.Journal()
        payload = b"short local-only fixture"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new-capsule.bin"
            parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            identity = {"hostname": remote.HOST, "uid": remote.UID,
                        "boot_id": "fixture", "mount_namespace": "mnt:[fixture]",
                        "mount": {"source": "/dev/root"}, "block_aliases": {},
                        "ancestry": {"ubuntu": {"inode": 1}},
                        "root_free_bytes": 30_000_000_000, "root_free_inodes": 100_000}
            try:
                with (mock.patch.object(remote, "TARGET", str(target)),
                      mock.patch.object(remote, "TARGET_NAME", target.name),
                      mock.patch.object(remote, "BYTES", len(payload) + 1),
                      mock.patch.object(remote, "SHA256", "0" * 64),
                      mock.patch.object(remote, "PRE_FREE", 0),
                      mock.patch.object(remote, "POST_FREE", 0),
                      mock.patch.object(remote, "_open_parent", side_effect=lambda: (os.dup(parent), {})),
                      mock.patch.object(remote, "_identity", return_value=identity),
                      mock.patch.object(remote, "_safe_path",
                                        side_effect=lambda path: {"exists": os.path.lexists(path)})):
                    with self.assertRaisesRegex(RuntimeError, "partial target retained"):
                        remote.write(io.BytesIO(payload))
                    self.assertEqual(target.read_bytes(), payload)
                    with self.assertRaises(FileExistsError):
                        remote.write(io.BytesIO(payload))
            finally:
                os.close(parent)

    def test_local_writer_success_fsyncs_file_and_parent_then_reads_back(self):
        payload = b"complete local-only capsule fixture"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new-capsule.bin"
            parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            identity = {"hostname": remote.HOST, "uid": remote.UID,
                        "boot_id": "fixture", "mount_namespace": "mnt:[fixture]",
                        "mount": {"source": "/dev/root"}, "block_aliases": {},
                        "ancestry": {"ubuntu": {"inode": 1}},
                        "root_free_bytes": 30_000_000_000, "root_free_inodes": 100_000}
            fsynced = []
            original_fsync = os.fsync
            def recorded_fsync(fd):
                fsynced.append(stat.S_ISDIR(os.fstat(fd).st_mode))
                return original_fsync(fd)
            try:
                with (mock.patch.object(remote, "TARGET", str(target)),
                      mock.patch.object(remote, "TARGET_NAME", target.name),
                      mock.patch.object(remote, "BYTES", len(payload)),
                      mock.patch.object(remote, "SHA256", digest(payload)),
                      mock.patch.object(remote, "PRE_FREE", 0),
                      mock.patch.object(remote, "POST_FREE", 0),
                      mock.patch.object(remote, "_open_parent", side_effect=lambda: (os.dup(parent), {})),
                      mock.patch.object(remote, "_identity", return_value=identity),
                      mock.patch.object(remote, "_safe_path",
                                        side_effect=lambda path: {"exists": os.path.lexists(path)}),
                      mock.patch.object(remote.os, "fsync", side_effect=recorded_fsync)):
                    receipt = remote.write(io.BytesIO(payload))
                    self.assertEqual(receipt["readback_sha256"], digest(payload))
                    self.assertEqual(receipt["target_stat"]["mode"], "0o600")
                    self.assertIn(False, fsynced)  # target FD
                    self.assertIn(True, fsynced)   # parent directory FD
            finally:
                os.close(parent)
            self.assertEqual(target.read_bytes(), payload)

    def test_preflight_failure_still_attempts_read_only_postflight(self):
        calls, events = [], []
        class Journal:
            def emit(self, event, **details):
                events.append(event)
            def close(self):
                events.append("CLOSED")
        def ssh(operation, journal):
            calls.append(operation)
            raise RuntimeError("synthetic transport failure")
        with (mock.patch.object(operator, "authorize", return_value=({}, {"request_sha256": "a" * 64})),
              mock.patch.object(operator.transport, "checked_local_source", return_value={}),
              mock.patch.object(operator, "_check_controller_breakers", return_value={}),
              mock.patch.object(operator, "_read_evidence", return_value=b"assignment"),
              mock.patch.object(operator, "Journal", Journal),
              mock.patch.object(operator, "_run_ssh", side_effect=ssh)):
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                operator.run_once()
        self.assertEqual(calls, ["probe", "probe"])
        self.assertIn("POSTFLIGHT_HOLD", events)

    def test_raw_ssh_transport_is_archived_before_receipt_parse(self):
        events = []
        stdout = encoded({"schema": "fixture", "operation": "probe"})
        stderr = b"local-only synthetic diagnostic"
        class Journal:
            def emit(self, event, **details):
                events.append((event, details))
        class Process:
            pid = 4242
            returncode = 0
            def communicate(self, timeout):
                return stdout, stderr
        with (mock.patch.object(operator.transport, "build_ssh_argv",
                                return_value=["ssh", "fixed-local-fixture"]),
              mock.patch.object(operator.subprocess, "Popen", return_value=Process())):
            self.assertEqual(operator._run_ssh("probe", Journal()), json.loads(stdout))
        self.assertEqual([item[0] for item in events], ["RAW_SSH_TRANSPORT", "REMOTE_RECEIPT"])
        raw = events[0][1]
        self.assertEqual(raw["argv"], ["ssh", "fixed-local-fixture"])
        self.assertEqual(raw["argv_sha256"], transport.argv_sha256(raw["argv"]))
        self.assertEqual(raw["local_ssh_pid"], 4242)
        self.assertEqual(raw["exit_code"], 0)
        self.assertEqual(raw["stdout_bytes"], len(stdout))
        self.assertEqual(raw["stdout_sha256"], digest(stdout))
        self.assertEqual(base64.b64decode(raw["stdout_b64"]), stdout)
        self.assertEqual(base64.b64decode(raw["stderr_b64"]), stderr)


if __name__ == "__main__":
    unittest.main()
