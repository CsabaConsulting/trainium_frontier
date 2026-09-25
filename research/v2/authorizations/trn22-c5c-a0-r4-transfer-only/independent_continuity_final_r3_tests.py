"""Independent final-r3 transfer-only tests. All writes are temporary/local.

No SSH, host, AWS or private-key-content access is permitted by these tests.
"""

import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_continuity_final_r3 as final
import stage_r4_remote as remote
import stage_r4_transport as transport
import test_stage_r4_transfer as fixtures


class IndependentContinuityFinalR3Tests(unittest.TestCase):
    def test_actual_review_assignment_chain_blocks_fake_or_changed_files(self):
        _request, binding = final._read_frozen_request_and_code()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, assignment = root / "review.json", root / "assignment.json"
            with (mock.patch.object(final, "FINAL_REVIEW", review),
                  mock.patch.object(final, "ASSIGNMENT", assignment),
                  mock.patch.object(final.core.transport, "checked_ssh_identity",
                                    return_value={"synthetic": True}),
                  mock.patch.object(final.marker_support, "_consume_fixed_marker",
                                    side_effect=AssertionError("marker reached")),
                  mock.patch.object(final.core, "Journal",
                                    side_effect=AssertionError("journal reached")),
                  mock.patch.object(final.core, "_run_ssh",
                                    side_effect=AssertionError("SSH reached"))):
                with self.assertRaisesRegex(RuntimeError, "FINAL R3 REVIEW HOLD"):
                    final.run_once()
                role_claim = {"verdict": "PASS", "reviewer_role": "independent_verifier"}
                review.write_bytes(final.core._json_bytes(role_claim))
                with self.assertRaisesRegex(RuntimeError, "SUPERVISOR R3 ASSIGNMENT HOLD"):
                    final.run_once()
                assignment.write_bytes(final.core._json_bytes(
                    final._expected_assignment(binding,
                                               final.core._sha(review.read_bytes()))))
                with self.assertRaisesRegex(RuntimeError, "not exact PASS"):
                    final.run_once()
                exact_review = final.core._json_bytes(final._expected_final_review(binding))
                review.write_bytes(exact_review)
                with self.assertRaisesRegex(RuntimeError, "assignment binding mismatch"):
                    final.run_once()
                exact_assignment = final.core._json_bytes(
                    final._expected_assignment(binding, final.core._sha(exact_review)))
                assignment.write_bytes(exact_assignment)
                self.assertEqual(final.authorize_final()["final_review_sha256"],
                                 final.core._sha(exact_review))
                review.write_bytes(exact_review + b"\n")
                with self.assertRaisesRegex(RuntimeError, "assignment binding mismatch"):
                    final.run_once()

    def test_local_one_attempt_four_phase_success_and_replay_refusal(self):
        pre = fixtures.probe()
        written = fixtures.receipt("write", pid=1001, preflight=pre)
        verified = fixtures.receipt("verify", pid=1002, utc_seconds=1)
        post = fixtures.probe(archive=written["target_stat"])
        post["utc"] = fixtures.utc(2)
        receipts = (pre, written, verified, post)
        operations = []

        def fake_ssh(operation, _journal, **kwargs):
            operations.append(operation)
            if operation == "write":
                self.assertIs(kwargs["preflight"], pre)
            return receipts[len(operations) - 1]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker, journal_path = root / "marker", root / "journal.jsonl"
            real_fsync = os.fsync
            with (mock.patch.object(final, "authorize_final",
                                    return_value={"synthetic_authority": True}),
                  mock.patch.object(final, "MARKER", marker),
                  mock.patch.object(final.marker_support, "MARKER", marker),
                  mock.patch.object(final.marker_support, "_open_marker_parent",
                                    side_effect=lambda: os.open(root, os.O_RDONLY |
                                                               os.O_DIRECTORY |
                                                               os.O_NOFOLLOW)),
                  mock.patch.object(final.core, "JOURNAL", journal_path),
                  mock.patch.object(final.core.transport, "checked_local_source",
                                    return_value={"sha256": remote.SHA256}),
                  mock.patch.object(final.core, "_check_controller_breakers",
                                    return_value={"local_stops_clear": True}),
                  mock.patch.object(final.core, "_run_ssh", side_effect=fake_ssh),
                  mock.patch.object(os, "fsync", wraps=real_fsync) as fsync):
                result = final.run_once()
                self.assertEqual(result["status"], "CROSS_SESSION_PERSISTENCE_HASH_PASS")
                self.assertEqual(operations, ["probe", "write", "verify", "probe"])
                self.assertGreaterEqual(fsync.call_count, 10)
                with self.assertRaises(FileExistsError):
                    final.run_once()
                self.assertEqual(operations, ["probe", "write", "verify", "probe"])
            self.assertEqual(marker.stat().st_mode & 0o777, 0o600)
            self.assertEqual(journal_path.stat().st_mode & 0o777, 0o600)
            events = [json.loads(line)["event"] for line in journal_path.read_bytes().splitlines()]
            self.assertEqual(events, [
                "FINAL_R3_ONE_ATTEMPT_CONSUMED", "FRESH_PREFLIGHT_PASS",
                "PREFLIGHT_TO_WRITE_CONTINUITY_PASS", "CROSS_SESSION_PERSISTENCE_PASS",
                "FINAL_READ_ONLY_POSTFLIGHT",
                "TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW"])

    def test_failed_write_retains_attempt_and_runs_full_postflight(self):
        pre, post = fixtures.probe(), fixtures.probe()
        operations = []

        def fake_ssh(operation, _journal, **kwargs):
            operations.append(operation)
            if operation == "write":
                self.assertIs(kwargs["preflight"], pre)
                raise RuntimeError("synthetic partial remote write")
            return pre if len(operations) == 1 else post

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker, journal_path = root / "marker", root / "journal.jsonl"
            with (mock.patch.object(final, "authorize_final",
                                    return_value={"synthetic_authority": True}),
                  mock.patch.object(final, "MARKER", marker),
                  mock.patch.object(final.marker_support, "MARKER", marker),
                  mock.patch.object(final.marker_support, "_open_marker_parent",
                                    side_effect=lambda: os.open(root, os.O_RDONLY |
                                                               os.O_DIRECTORY |
                                                               os.O_NOFOLLOW)),
                  mock.patch.object(final.core, "JOURNAL", journal_path),
                  mock.patch.object(final.core.transport, "checked_local_source",
                                    return_value={"sha256": remote.SHA256}),
                  mock.patch.object(final.core, "_check_controller_breakers",
                                    return_value={"local_stops_clear": True}),
                  mock.patch.object(final.core, "_run_ssh", side_effect=fake_ssh)):
                with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                    final.run_once()
                with self.assertRaises(FileExistsError):
                    final.run_once()
            self.assertEqual(operations, ["probe", "write", "probe"])
            self.assertTrue(marker.exists())
            events = [json.loads(line)["event"] for line in journal_path.read_bytes().splitlines()]
            self.assertIn("TRANSFER_HOLD", events)
            self.assertIn("FINAL_READ_ONLY_POSTFLIGHT", events)
            self.assertNotIn("TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW", events)

    def test_boot_mount_and_parent_races_fail_before_target_creation(self):
        pre = fixtures.probe()
        binding = transport.write_binding(pre)
        identity = pre["identity_after"]
        for name, mutate in (
                ("boot", lambda item: item.update({"boot_id": "new-boot"})),
                ("mount", lambda item: item["mount"].update({"mount_id": "new-mount"})),
                ("parent", lambda item: item["ancestry"]["ubuntu"].update(
                    {"inode": 999999}))):
            with self.subTest(race=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / remote.TARGET_NAME
                changed = copy.deepcopy(identity)
                mutate(changed)
                calls = {"identity": 0}

                def identity_at_boundary(_fd, _ancestry):
                    calls["identity"] += 1
                    return identity if calls["identity"] == 1 else changed

                parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    with (mock.patch.object(remote, "_open_parent",
                                            side_effect=lambda: (os.dup(parent_fd),
                                                                 identity["ancestry"])),
                          mock.patch.object(remote, "_identity",
                                            side_effect=identity_at_boundary),
                          mock.patch.object(remote, "_process_identity",
                                            return_value={"pid": 456, "start_ticks": 789}),
                          mock.patch.object(remote, "_safe_path",
                                            return_value={"exists": False}),
                          mock.patch.object(remote, "_live_prewrite_safety",
                                            return_value={"synthetic": "clean"})):
                        with self.assertRaises(RuntimeError):
                            remote.write(io.BytesIO(b"never-written"), binding)
                    self.assertEqual(calls["identity"], 2)
                    self.assertFalse(target.exists())
                finally:
                    os.close(parent_fd)

    def test_late_stop_label_and_lock_prevent_o_excl(self):
        pre = fixtures.probe()
        binding = transport.write_binding(pre)
        identity = pre["identity_after"]
        for signal in ("STOP", "label", "lock"):
            with self.subTest(signal=signal), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / remote.TARGET_NAME
                state = {"busy": False, "identity_calls": 0}

                def identity_at_boundary(_fd, _ancestry):
                    state["identity_calls"] += 1
                    if state["identity_calls"] == 2:
                        state["busy"] = True
                    return identity

                def path_state(path):
                    return {"exists": target.exists() if path == remote.TARGET else
                            bool(state["busy"] and signal == "STOP" and
                                 path == remote.STOP_PATHS[0])}

                def fuser(path):
                    key = ("checkpoint_fuser" if path == remote.CHECKPOINT else
                           "neuron0_fuser" if path == "/dev/neuron0" else
                           "shared_lock_fuser")
                    result = copy.deepcopy(pre[key])
                    if state["busy"] and signal == "lock" and key == "shared_lock_fuser":
                        result.update({"rc": 0, "pids": [777]})
                    return result

                def label_scan():
                    result = copy.deepcopy(pre["science_label_scan"])
                    if state["busy"] and signal == "label":
                        result.update({"matches": ["synthetic-r4-label"], "match_count": 1})
                    return result

                parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    with (mock.patch.object(remote, "_open_parent",
                                            side_effect=lambda: (os.dup(parent_fd),
                                                                 identity["ancestry"])),
                          mock.patch.object(remote, "_identity",
                                            side_effect=identity_at_boundary),
                          mock.patch.object(remote, "_process_identity",
                                            return_value={"pid": 456, "start_ticks": 789}),
                          mock.patch.object(remote, "_safe_path", side_effect=path_state),
                          mock.patch.object(remote, "_processes",
                                            return_value={"matched": [],
                                                          "unreadable_cmdlines": 0}),
                          mock.patch.object(remote, "_fuser", side_effect=fuser),
                          mock.patch.object(remote, "_neuron_ls",
                                            return_value=copy.deepcopy(pre["neuron_ls"])),
                          mock.patch.object(remote, "_label_scan", side_effect=label_scan)):
                        with self.assertRaises(RuntimeError):
                            remote.write(io.BytesIO(b"never-written"), binding)
                    self.assertTrue(state["busy"])
                    self.assertFalse(target.exists())
                finally:
                    os.close(parent_fd)

    def test_partial_target_is_preserved_and_same_path_retry_is_rejected(self):
        pre = fixtures.probe()
        binding = transport.write_binding(pre)
        identity = pre["identity_after"]
        payload = b"synthetic-partial"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / remote.TARGET_NAME
            parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with (mock.patch.object(remote, "BYTES", len(payload) + 1),
                      mock.patch.object(remote, "_open_parent",
                                        side_effect=lambda: (os.dup(parent_fd),
                                                             identity["ancestry"])),
                      mock.patch.object(remote, "_identity", return_value=identity),
                      mock.patch.object(remote, "_process_identity",
                                        return_value={"pid": 456, "start_ticks": 789}),
                      mock.patch.object(remote, "_safe_path",
                                        side_effect=lambda path: {"exists": target.exists()
                                                                  if path == remote.TARGET
                                                                  else False}),
                      mock.patch.object(remote, "_live_prewrite_safety",
                                        return_value={"synthetic": "clean"}),
                      mock.patch.object(remote, "_require_no_stops_or_processes",
                                        return_value={"stops": {}, "processes": {}})):
                    with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                        remote.write(io.BytesIO(payload), binding)
                    self.assertEqual(target.read_bytes(), payload)
                    with self.assertRaises(FileExistsError):
                        remote.write(io.BytesIO(payload), binding)
                    self.assertEqual(target.read_bytes(), payload)
            finally:
                os.close(parent_fd)

    def test_each_ssh_process_raw_transport_is_journaled_before_receipt(self):
        class Journal:
            def __init__(self):
                self.events = []

            def emit(self, event, **details):
                self.events.append((event, details))

        class Process:
            pid = 10101
            returncode = 0

            def communicate(self, timeout):
                self.timeout = timeout
                return b'{"synthetic":true}', b""

        journal = Journal()
        with (mock.patch.object(final.core.transport, "build_ssh_argv",
                                return_value=["ssh", "synthetic-local-fixture"]),
              mock.patch.object(final.core.subprocess, "Popen",
                                return_value=Process()) as spawn):
            self.assertEqual(final.core._run_ssh("probe", journal),
                             {"synthetic": True})
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual([event for event, _ in journal.events],
                         ["RAW_SSH_TRANSPORT", "REMOTE_RECEIPT"])
        raw = journal.events[0][1]
        self.assertEqual(raw["exit_code"], 0)
        self.assertEqual(raw["stdout_bytes"], len(b'{"synthetic":true}'))
        self.assertEqual(raw["stdout_sha256"], final.core._sha(b'{"synthetic":true}'))
        self.assertEqual(raw["stderr_bytes"], 0)

        failed = Process()
        failed.returncode = 255
        failed.communicate = lambda timeout: (b"", b"synthetic SSH failure")
        journal = Journal()
        with (mock.patch.object(final.core.transport, "build_ssh_argv",
                                return_value=["ssh", "synthetic-local-fixture"]),
              mock.patch.object(final.core.subprocess, "Popen", return_value=failed)):
            with self.assertRaisesRegex(RuntimeError, "raw transport archived"):
                final.core._run_ssh("probe", journal)
        self.assertEqual([event for event, _ in journal.events],
                         ["RAW_SSH_TRANSPORT"])
        self.assertEqual(journal.events[0][1]["exit_code"], 255)
        self.assertEqual(journal.events[0][1]["stderr_bytes"],
                         len(b"synthetic SSH failure"))


if __name__ == "__main__":
    unittest.main()
