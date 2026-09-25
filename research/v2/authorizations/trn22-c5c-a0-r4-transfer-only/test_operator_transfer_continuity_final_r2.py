"""Local synthetic request tests only; never SSH or read private-key bytes."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))
import operator_transfer_continuity_final_r2 as final
import stage_r4_remote as remote
import test_stage_r4_transfer as fixtures


class ContinuityFinalR2Tests(unittest.TestCase):
    def test_exact_request_pins_actual_static_pass_and_unchanged_hold_archive(self):
        request, binding = final._read_frozen_request_and_code()
        self.assertEqual(binding["request_sha256"], final.REQUEST_SHA256)
        self.assertEqual(request["source_commit"], final.SOURCE_COMMIT)
        self.assertEqual(request["capsule_sha256"], final.core.transport.SOURCE_SHA256)
        self.assertEqual(request["independent_continuity_static_review_sha256"],
                         final.STATIC_REVIEW_SHA256)
        self.assertEqual(request["reviewed_disabled_final_wrapper_sha256"],
                         final.DISABLED_FINAL_WRAPPER_SHA256)
        self.assertEqual(final.MARKER, final.marker_support.MARKER)
        self.assertEqual(final.JOURNAL, final.core.JOURNAL)
        self.assertFalse(json.loads(final.DISABLED_FINAL_REQUEST.read_text())[
            "dispatch_authorized"])
        self.assertFalse(json.loads(final.DISABLED_CORE_REQUEST.read_text())[
            "dispatch_authorized"])

    def test_missing_final_pass_holds_before_marker_journal_or_ssh(self):
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
                with self.assertRaisesRegex(RuntimeError, "FINAL REVIEW HOLD"):
                    final.run_once()

    def test_review_file_without_supervisor_assignment_cannot_reach_ssh(self):
        _request, binding = final._read_frozen_request_and_code()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = root / "review.json"
            review.write_text(json.dumps(final._expected_final_review(binding)),
                              encoding="utf-8")
            with (mock.patch.object(final, "FINAL_REVIEW", review),
                  mock.patch.object(final, "ASSIGNMENT", root / "absent-assignment.json"),
                  mock.patch.object(final.marker_support, "_consume_fixed_marker",
                                    side_effect=AssertionError("marker reached")),
                  mock.patch.object(final.core, "_run_ssh",
                                    side_effect=AssertionError("SSH reached"))):
                with self.assertRaisesRegex(RuntimeError, "SUPERVISOR ASSIGNMENT HOLD"):
                    final.run_once()

    def test_changed_request_code_or_static_review_fails_before_authority(self):
        with mock.patch.object(final, "STATIC_REVIEW_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                final._read_frozen_request_and_code()
        with mock.patch.object(final, "TRANSPORT_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                final._read_frozen_request_and_code()
        forged = json.loads(final.REQUEST.read_text(encoding="utf-8"))
        forged["target"] += ".other"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged-request.json"
            path.write_text(json.dumps(forged), encoding="utf-8")
            with mock.patch.object(final, "REQUEST", path):
                with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                    final._read_frozen_request_and_code()

    def test_independent_review_and_supervisor_bind_actual_bytes(self):
        _request, binding = final._read_frozen_request_and_code()
        review = final._expected_final_review(binding)
        review_bytes = json.dumps(review, sort_keys=True).encode("utf-8")
        assignment = final._expected_assignment(binding, final.core._sha(review_bytes))
        assignment_bytes = json.dumps(assignment, sort_keys=True).encode("utf-8")
        authority = final._validate_authority_files(binding, review_bytes,
                                                    assignment_bytes)
        self.assertEqual(authority["final_review_sha256"],
                         final.core._sha(review_bytes))
        self.assertEqual(authority["supervisor_assignment_sha256"],
                         final.core._sha(assignment_bytes))
        wrong_review = copy.deepcopy(review)
        wrong_review["final_wrapper_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "actual file is not exact PASS"):
            final._validate_authority_files(binding,
                                            json.dumps(wrong_review).encode("utf-8"),
                                            assignment_bytes)
        wrong_assignment = copy.deepcopy(assignment)
        wrong_assignment["final_independent_review_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            final._validate_authority_files(binding, review_bytes,
                                            json.dumps(wrong_assignment).encode("utf-8"))
        wrong_assignment = copy.deepcopy(assignment)
        wrong_assignment["native_run_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            final._validate_authority_files(binding, review_bytes,
                                            json.dumps(wrong_assignment).encode("utf-8"))
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            final._validate_authority_files(binding,
                                            json.dumps(review, indent=2).encode("utf-8"),
                                            assignment_bytes)

    def test_authority_reads_both_actual_fixed_path_files_without_dispatch(self):
        _request, binding = final._read_frozen_request_and_code()
        review_bytes = json.dumps(final._expected_final_review(binding),
                                  sort_keys=True).encode("utf-8")
        assignment_bytes = json.dumps(final._expected_assignment(
            binding, final.core._sha(review_bytes)), sort_keys=True).encode("utf-8")
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
                self.assertEqual(authority["final_review_sha256"],
                                 final.core._sha(review_bytes))
                changed = json.loads(assignment.read_text(encoding="utf-8"))
                changed["final_independent_review_sha256"] = "0" * 64
                assignment.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
                    final.authorize_final()

    def test_late_local_stop_prevents_write_and_still_postflights(self):
        pre, post = fixtures.probe(), fixtures.probe()
        calls, events, checks = [], [], 0

        class Journal:
            def emit(self, event, **details):
                events.append(event)
            def close(self):
                events.append("CLOSED")

        def breakers():
            nonlocal checks
            checks += 1
            if checks == 2:
                raise RuntimeError("local STOP appeared")
            return {}

        def fake_ssh(operation, journal, **kwargs):
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

    def test_failed_remote_write_receives_exact_preflight_and_postflights(self):
        pre, post = fixtures.probe(), fixtures.probe()
        calls, events = [], []

        class Journal:
            def emit(self, event, **details):
                events.append(event)
            def close(self):
                events.append("CLOSED")

        def fake_ssh(operation, journal, **kwargs):
            calls.append(operation)
            if len(calls) == 1:
                return pre
            if operation == "write":
                self.assertIs(kwargs["preflight"], pre)
                raise RuntimeError("synthetic remote STOP before create")
            return post

        with (mock.patch.object(final, "authorize_final", return_value={}),
              mock.patch.object(final.core.transport, "checked_local_source",
                                return_value={"sha256": remote.SHA256}),
              mock.patch.object(final.core, "_check_controller_breakers", return_value={}),
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

    def test_offline_success_path_requires_four_ordered_receipts(self):
        pre = fixtures.probe()
        written = fixtures.receipt("write", pid=1001, preflight=pre)
        verified = fixtures.receipt("verify", pid=1002, utc_seconds=1)
        post = fixtures.probe(archive=written["target_stat"])
        calls, events = [], []

        class Journal:
            def emit(self, event, **details):
                events.append(event)
            def close(self):
                events.append("CLOSED")

        def fake_ssh(operation, journal, **kwargs):
            calls.append(operation)
            if len(calls) == 2:
                self.assertIs(kwargs["preflight"], pre)
            return (pre, written, verified, post)[len(calls) - 1]

        with (mock.patch.object(final, "authorize_final", return_value={}),
              mock.patch.object(final.core.transport, "checked_local_source",
                                return_value={"sha256": remote.SHA256}),
              mock.patch.object(final.core, "_check_controller_breakers", return_value={}),
              mock.patch.object(final.marker_support, "_consume_fixed_marker",
                                return_value="a" * 64),
              mock.patch.object(final.core, "_pinned", return_value=b"marker"),
              mock.patch.object(final.core, "Journal", Journal),
              mock.patch.object(final.core, "_run_ssh", side_effect=fake_ssh)):
            result = final.run_once()
        self.assertEqual(calls, ["probe", "write", "verify", "probe"])
        self.assertEqual(result["status"], "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        self.assertEqual(result["preflight_continuity"]["status"],
                         "PREFLIGHT_TO_WRITE_IDENTITY_AND_LIVE_SAFETY_PASS")
        self.assertIn("FINAL_READ_ONLY_POSTFLIGHT", events)
        self.assertIn("TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW", events)


if __name__ == "__main__":
    unittest.main()
