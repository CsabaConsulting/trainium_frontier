"""Offline final-r3 request tests; every SSH/marker path is mocked."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_continuity_final_r3 as final
import stage_r4_remote as remote
import test_stage_r4_transfer as fixtures


class ContinuityFinalR3Tests(unittest.TestCase):
    def test_exact_static_pass_and_repaired_bytes_are_pinned(self):
        request, binding = final._read_frozen_request_and_code()
        self.assertEqual(binding["request_sha256"], final.REQUEST_SHA256)
        self.assertEqual(request["independent_r3_static_review_sha256"],
                         final.STATIC_REVIEW_SHA256)
        self.assertEqual(request["reviewed_r3_implementer_commit"],
                         final.R3_IMPLEMENTER_COMMIT)
        self.assertEqual(request["remote_sha256"], final.REMOTE_SHA256)
        self.assertEqual(request["transport_sha256"], final.TRANSPORT_SHA256)
        self.assertTrue(request["dispatch_authorized"])  # Intent, not authority.
        self.assertFalse(json.loads(final.DISABLED_R3_REQUEST.read_text())[
            "dispatch_authorized"])

    def test_missing_final_pass_blocks_before_marker_journal_or_ssh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (mock.patch.object(final, "FINAL_REVIEW", root / "absent-review.json"),
                  mock.patch.object(final, "ASSIGNMENT", root / "absent-assignment.json"),
                  mock.patch.object(final.marker_support, "_consume_fixed_marker",
                                    side_effect=AssertionError("marker reached")),
                  mock.patch.object(final.core, "Journal",
                                    side_effect=AssertionError("journal reached")),
                  mock.patch.object(final.core, "_run_ssh",
                                    side_effect=AssertionError("SSH reached"))):
                with self.assertRaisesRegex(RuntimeError, "FINAL R3 REVIEW HOLD"):
                    final.run_once()

    def test_review_alone_still_blocks_before_ssh(self):
        _request, binding = final._read_frozen_request_and_code()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = root / "review.json"
            review.write_bytes(final.core._json_bytes(
                final._expected_final_review(binding)))
            with (mock.patch.object(final, "FINAL_REVIEW", review),
                  mock.patch.object(final, "ASSIGNMENT", root / "absent-assignment.json"),
                  mock.patch.object(final.marker_support, "_consume_fixed_marker",
                                    side_effect=AssertionError("marker reached")),
                  mock.patch.object(final.core, "_run_ssh",
                                    side_effect=AssertionError("SSH reached"))):
                with self.assertRaisesRegex(RuntimeError, "SUPERVISOR R3 ASSIGNMENT HOLD"):
                    final.run_once()

    def test_request_code_or_static_review_drift_blocks_before_authority(self):
        with mock.patch.object(final, "STATIC_REVIEW_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                final._read_frozen_request_and_code()
        with mock.patch.object(final, "REMOTE_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                final._read_frozen_request_and_code()
        forged = json.loads(final.REQUEST.read_text(encoding="utf-8"))
        forged["source_checkout_authorized"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged.json"
            path.write_text(json.dumps(forged), encoding="utf-8")
            with mock.patch.object(final, "REQUEST", path):
                with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                    final._read_frozen_request_and_code()

    def test_actual_final_review_bytes_bind_exact_supervisor_assignment(self):
        _request, binding = final._read_frozen_request_and_code()
        review = final._expected_final_review(binding)
        review_bytes = final.core._json_bytes(review)
        assignment = final._expected_assignment(binding, final.core._sha(review_bytes))
        assignment_bytes = final.core._json_bytes(assignment)
        authority = final._validate_authority_files(binding, review_bytes,
                                                   assignment_bytes)
        self.assertEqual(authority["final_review_sha256"],
                         final.core._sha(review_bytes))
        changed_review = copy.deepcopy(review)
        changed_review["final_wrapper_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "not exact PASS"):
            final._validate_authority_files(binding,
                                            final.core._json_bytes(changed_review),
                                            assignment_bytes)
        changed_assignment = copy.deepcopy(assignment)
        changed_assignment["final_independent_review_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "assignment binding mismatch"):
            final._validate_authority_files(binding, review_bytes,
                                            final.core._json_bytes(changed_assignment))
        changed_assignment = copy.deepcopy(assignment)
        changed_assignment["native_run_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "assignment binding mismatch"):
            final._validate_authority_files(binding, review_bytes,
                                            final.core._json_bytes(changed_assignment))
        with self.assertRaisesRegex(RuntimeError, "assignment binding mismatch"):
            final._validate_authority_files(binding, json.dumps(review, indent=2).encode(),
                                            assignment_bytes)

    def test_reading_actual_files_is_still_local_until_run_once(self):
        _request, binding = final._read_frozen_request_and_code()
        review_bytes = final.core._json_bytes(final._expected_final_review(binding))
        assignment_bytes = final.core._json_bytes(final._expected_assignment(
            binding, final.core._sha(review_bytes)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, assignment = root / "review.json", root / "assignment.json"
            review.write_bytes(review_bytes)
            assignment.write_bytes(assignment_bytes)
            with (mock.patch.object(final, "FINAL_REVIEW", review),
                  mock.patch.object(final, "ASSIGNMENT", assignment),
                  mock.patch.object(final.core.transport, "checked_ssh_identity",
                                    return_value={"synthetic": True}),
                  mock.patch.object(final.marker_support, "_consume_fixed_marker",
                                    side_effect=AssertionError("marker reached")),
                  mock.patch.object(final.core, "_run_ssh",
                                    side_effect=AssertionError("SSH reached"))):
                authority = final.authorize_final()
                self.assertEqual(authority["supervisor_assignment_sha256"],
                                 final.core._sha(assignment_bytes))
                assignment.write_bytes(assignment_bytes.replace(
                    b'"native_run_authorized":false',
                    b'"native_run_authorized":true'))
                with self.assertRaisesRegex(RuntimeError, "assignment binding mismatch"):
                    final.authorize_final()

    def test_authority_revocation_before_marker_does_not_consume_attempt(self):
        with (mock.patch.object(final, "authorize_final",
                                side_effect=[{"authority": "first"},
                                             {"authority": "changed"}]),
              mock.patch.object(final.core.transport, "checked_local_source",
                                return_value={"sha256": remote.SHA256}),
              mock.patch.object(final.core, "_check_controller_breakers",
                                return_value={}),
              mock.patch.object(final.marker_support, "_consume_fixed_marker",
                                side_effect=AssertionError("marker reached")),
              mock.patch.object(final.core, "Journal",
                                side_effect=AssertionError("journal reached")),
              mock.patch.object(final.core, "_run_ssh",
                                side_effect=AssertionError("SSH reached"))):
            with self.assertRaisesRegex(RuntimeError, "drifted before marker"):
                final.run_once()

    def test_late_controller_stop_prevents_write_and_still_postflights(self):
        pre, post = fixtures.probe(), fixtures.probe()
        calls, events, checks = [], [], {"count": 0}

        class Journal:
            def emit(self, event, **_details):
                events.append(event)
            def close(self):
                events.append("CLOSED")

        def breakers():
            checks["count"] += 1
            if checks["count"] == 2:
                raise RuntimeError("local STOP appeared")
            return {}

        def fake_ssh(operation, _journal, **_kwargs):
            calls.append(operation)
            return pre if len(calls) == 1 else post

        with (mock.patch.object(final, "authorize_final", return_value={}),
              mock.patch.object(final.core.transport, "checked_local_source",
                                return_value={"sha256": remote.SHA256}),
              mock.patch.object(final.core, "_check_controller_breakers",
                                side_effect=breakers),
              mock.patch.object(final.marker_support, "_consume_fixed_marker",
                                return_value="a" * 64),
              mock.patch.object(final.core, "_pinned", return_value=b"marker"),
              mock.patch.object(final.core, "Journal", Journal),
              mock.patch.object(final.core, "_run_ssh", side_effect=fake_ssh)):
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                final.run_once()
        self.assertEqual(calls, ["probe", "probe"])
        self.assertIn("FINAL_READ_ONLY_POSTFLIGHT", events)
        self.assertIn("TRANSFER_HOLD", events)

    def test_success_path_has_four_distinct_ordered_remote_receipts(self):
        pre = fixtures.probe()
        written = fixtures.receipt("write", pid=1001, preflight=pre)
        verified = fixtures.receipt("verify", pid=1002, utc_seconds=1)
        post = fixtures.probe(archive=written["target_stat"])
        calls, events = [], []

        class Journal:
            def emit(self, event, **_details):
                events.append(event)
            def close(self):
                events.append("CLOSED")

        def fake_ssh(operation, _journal, **kwargs):
            calls.append(operation)
            if operation == "write":
                self.assertIs(kwargs["preflight"], pre)
            return (pre, written, verified, post)[len(calls) - 1]

        with (mock.patch.object(final, "authorize_final", return_value={}),
              mock.patch.object(final.core.transport, "checked_local_source",
                                return_value={"sha256": remote.SHA256}),
              mock.patch.object(final.core, "_check_controller_breakers",
                                return_value={}),
              mock.patch.object(final.marker_support, "_consume_fixed_marker",
                                return_value="a" * 64),
              mock.patch.object(final.core, "_pinned", return_value=b"marker"),
              mock.patch.object(final.core, "Journal", Journal),
              mock.patch.object(final.core, "_run_ssh", side_effect=fake_ssh)):
            result = final.run_once()
        self.assertEqual(calls, ["probe", "write", "verify", "probe"])
        self.assertEqual(result["status"], "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        self.assertIn("FINAL_READ_ONLY_POSTFLIGHT", events)
        self.assertIn("TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW", events)

    def test_failed_remote_write_still_attempts_distinct_read_only_postflight(self):
        pre, post = fixtures.probe(), fixtures.probe()
        calls, events = [], []

        class Journal:
            def emit(self, event, **_details):
                events.append(event)
            def close(self):
                events.append("CLOSED")

        def fake_ssh(operation, _journal, **kwargs):
            calls.append(operation)
            if len(calls) == 1:
                return pre
            if operation == "write":
                self.assertIs(kwargs["preflight"], pre)
                raise RuntimeError("synthetic remote late STOP before O_EXCL")
            return post

        with (mock.patch.object(final, "authorize_final", return_value={}),
              mock.patch.object(final.core.transport, "checked_local_source",
                                return_value={"sha256": remote.SHA256}),
              mock.patch.object(final.core, "_check_controller_breakers",
                                return_value={}),
              mock.patch.object(final.marker_support, "_consume_fixed_marker",
                                return_value="a" * 64),
              mock.patch.object(final.core, "_pinned", return_value=b"marker"),
              mock.patch.object(final.core, "Journal", Journal),
              mock.patch.object(final.core, "_run_ssh", side_effect=fake_ssh)):
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                final.run_once()
        self.assertEqual(calls, ["probe", "write", "probe"])
        self.assertIn("FINAL_READ_ONLY_POSTFLIGHT", events)
        self.assertIn("TRANSFER_HOLD", events)
        self.assertIn("CLOSED", events)


if __name__ == "__main__":
    unittest.main()
