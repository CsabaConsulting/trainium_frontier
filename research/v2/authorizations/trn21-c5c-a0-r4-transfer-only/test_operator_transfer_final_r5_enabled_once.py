"""Offline authority and inheritance checks for trn21 r5 transfer candidate.

No test writes an authority file, marker, journal, or contacts trn21.
"""

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_final_r5_enabled_once as candidate
import test_operator_transfer_final_r4_once as r4_fixtures


class FinalR5EnabledCandidateTests(unittest.TestCase):
    def _guarded_run(self, expected_error, *, read_required=None):
        patches = [
            mock.patch.object(candidate, "_consume_fixed_marker",
                              side_effect=AssertionError("marker reached")),
            mock.patch.object(candidate.core, "Journal",
                              side_effect=AssertionError("journal reached")),
            mock.patch.object(candidate, "_run_ssh_final",
                              side_effect=AssertionError("SSH reached")),
            mock.patch.object(candidate.subprocess, "Popen",
                              side_effect=AssertionError("process spawned")),
        ]
        if read_required is not None:
            patches.append(mock.patch.object(candidate, "_read_required",
                                             side_effect=read_required))
        with patches[0], patches[1], patches[2], patches[3]:
            if len(patches) == 5:
                with patches[4], self.assertRaisesRegex(RuntimeError, expected_error):
                    candidate.run_once()
            else:
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    candidate.run_once()

    @staticmethod
    def _review_bytes(binding):
        return candidate.core._json_bytes(candidate._expected_final_review(binding))

    def test_exact_request_binds_disabled_r5_and_separate_static_pass(self):
        request, binding = candidate._read_frozen_request_and_code()
        self.assertEqual(binding["request_sha256"], candidate.REQUEST_SHA256)
        self.assertEqual(binding["candidate_tests_sha256"],
                         candidate.core._sha(candidate.core._read_evidence(
                             candidate.ENABLED_TESTS)))
        self.assertEqual(binding["candidate_runbook_sha256"],
                         candidate.core._sha(candidate.core._read_evidence(
                             candidate.ENABLED_RUNBOOK)))
        self.assertEqual(request["schema"],
                         "trn21-c5c-a0-r4-transfer-final-request-v5-enabled-candidate-r1")
        self.assertIs(request["dispatch_authorized"], True)
        self.assertIs(request["source_transfer_authorized"], True)
        self.assertEqual(request["dispatch_meaning"],
                         "TRANSFER_INTENT_ONLY_REQUIRES_EXACT_FINAL_PASS_AND_SUPERVISOR_ONE_SHOT_ASSIGNMENT")
        self.assertEqual(request["prior_final_r5_request_sha256"],
                         candidate.PRIOR_R5_REQUEST_SHA256)
        self.assertEqual(request["prior_final_r5_wrapper_sha256"],
                         candidate.PRIOR_R5_WRAPPER_SHA256)
        self.assertEqual(request["prior_final_r5_independent_pass_sha256"],
                         candidate.PRIOR_R5_INDEPENDENT_PASS_SHA256)
        self.assertEqual(request["prior_final_r5_independent_tests_sha256"],
                         candidate.PRIOR_R5_INDEPENDENT_TESTS_SHA256)
        self.assertEqual(request["transfer_attempt_label"],
                         candidate.TRANSFER_ATTEMPT_LABEL)
        self.assertNotEqual(candidate.MARKER, candidate.prior_r5.MARKER)
        self.assertFalse(candidate.FINAL_REVIEW.exists())
        self.assertFalse(candidate.ASSIGNMENT.exists())
        for key in ("source_checkout_authorized", "cpu_g2_authorized",
                    "native_run_authorized", "eval_authorized",
                    "monitor_operation_authorized", "cleanup_authorized",
                    "submission_authorized"):
            self.assertIs(request[key], False)
        self.assertIs(request["frozen_446b_native_launcher_separate_hold_acknowledged"],
                      True)

    def test_missing_final_review_stops_before_marker_journal_or_ssh(self):
        self._guarded_run("FINAL REVIEW HOLD: exact fixed-path file is absent")

    def test_review_alone_cannot_bypass_absent_supervisor_assignment(self):
        _, binding = candidate._read_frozen_request_and_code()
        review_bytes = self._review_bytes(binding)

        def only_review(path, purpose):
            if path == candidate.FINAL_REVIEW and purpose == "FINAL REVIEW":
                return review_bytes
            raise RuntimeError("SUPERVISOR ASSIGNMENT HOLD: exact fixed-path file is absent")

        self._guarded_run("SUPERVISOR ASSIGNMENT HOLD", read_required=only_review)

    def test_mismatched_final_review_stops_before_marker_journal_or_ssh(self):
        _, binding = candidate._read_frozen_request_and_code()
        review = candidate._expected_final_review(binding)
        review["request_sha256"] = "0" * 64

        def changed_review(path, _purpose):
            if path == candidate.FINAL_REVIEW:
                return candidate.core._json_bytes(review)
            return b"{}"

        self._guarded_run("FINAL REVIEW HOLD", read_required=changed_review)

    def test_mismatched_assignment_stops_before_marker_journal_or_ssh(self):
        _, binding = candidate._read_frozen_request_and_code()
        review_bytes = self._review_bytes(binding)
        assignment = candidate._expected_assignment(binding,
                                                    candidate.core._sha(review_bytes))
        assignment["request_sha256"] = "0" * 64

        def changed_assignment(path, _purpose):
            if path == candidate.FINAL_REVIEW:
                return review_bytes
            return candidate.core._json_bytes(assignment)

        self._guarded_run("SUPERVISOR ASSIGNMENT HOLD",
                          read_required=changed_assignment)

    def test_only_exact_in_memory_chain_is_accepted(self):
        _, binding = candidate._read_frozen_request_and_code()
        review_bytes = self._review_bytes(binding)
        assignment = candidate._expected_assignment(binding,
                                                    candidate.core._sha(review_bytes))
        assignment_bytes = candidate.core._json_bytes(assignment)
        authority = candidate._validate_authority_files(binding, review_bytes,
                                                        assignment_bytes)
        self.assertEqual(authority["final_review_sha256"],
                         candidate.core._sha(review_bytes))
        self.assertEqual(authority["supervisor_assignment_sha256"],
                         candidate.core._sha(assignment_bytes))
        for mutation in ({"native_run_authorized": True},
                         {"source_transfer_authorized": False},
                         {"transfer_attempt_label": "reused"}):
            changed = copy.deepcopy(assignment)
            changed.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    RuntimeError, "SUPERVISOR ASSIGNMENT HOLD"):
                candidate._validate_authority_files(binding, review_bytes,
                                                    candidate.core._json_bytes(changed))
        changed_review = json.loads(review_bytes)
        changed_review["native_launch_remains_hold"] = False
        with self.assertRaisesRegex(RuntimeError, "FINAL REVIEW HOLD"):
            candidate._validate_authority_files(binding,
                                                candidate.core._json_bytes(changed_review),
                                                assignment_bytes)
        duplicate_review = review_bytes[:-1] + b',"verdict":"PASS"}'
        with self.assertRaisesRegex(RuntimeError, "FINAL AUTHORITY HOLD"):
            candidate._validate_authority_files(binding, duplicate_review,
                                                assignment_bytes)

    def test_prior_independent_pass_drift_stops_before_authority_or_ssh(self):
        original_pinned = candidate.core._pinned

        def changed_pass(path, digest):
            if path == candidate.PRIOR_R5_INDEPENDENT_PASS:
                raise RuntimeError("reviewed fixed-path evidence drift")
            return original_pinned(path, digest)

        with (mock.patch.object(candidate.core, "_pinned", side_effect=changed_pass),
              mock.patch.object(candidate, "_read_required",
                                side_effect=AssertionError("authority reached"))):
            self._guarded_run("reviewed fixed-path evidence drift")

    def test_reused_r5_transport_preserves_full_probe_and_lock_chain(self):
        self.assertIs(candidate.final_transport, candidate.prior_r5.final_transport)
        pre, write, verify, post = r4_fixtures.clean_receipts()
        transport = candidate.final_transport
        self.assertIs(transport.validate_probe(pre, before_write=True), pre)
        self.assertIs(transport.validate_probe(post, before_write=False), post)
        self.assertEqual(transport.compare_preflight_to_write(pre, write)["status"],
                         "PREFLIGHT_TO_FINAL_LOCKED_LABEL_GATE_PASS")
        persistence = transport.compare_sessions(pre, write, verify)
        self.assertEqual(transport.compare_postflight(pre, post,
                                                      persistence=persistence)["status"],
                         "READ_ONLY_POSTFLIGHT_PASS")
        unsafe = copy.deepcopy(pre)
        unsafe["identity_after"]["root_free_inodes"] = 9999
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.validate_probe(unsafe, before_write=True)
        unsafe = copy.deepcopy(post)
        unsafe["host_lock_identity"]["trust"]["acl"]["cache"][
            "system.posix_acl_access"] = "present"
        with self.assertRaises(RuntimeError):
            transport.validate_probe(unsafe, before_write=False)


if __name__ == "__main__":
    unittest.main()
