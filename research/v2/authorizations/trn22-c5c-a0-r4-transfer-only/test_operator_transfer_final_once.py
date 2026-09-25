"""Offline/synthetic final-request tests; never contact trn22 or read PEM bytes."""

import copy
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))
import operator_transfer_final_once as final


class FinalTransferRequestTests(unittest.TestCase):
    def _binding(self):
        return final._read_frozen_request_and_code()

    def test_exact_frozen_request_and_old_independent_review_are_pinned(self):
        request, binding = self._binding()
        self.assertEqual(binding["request_sha256"], final.REQUEST_SHA256)
        self.assertEqual(request["source_commit"], final.SOURCE_COMMIT)
        self.assertEqual(request["capsule_sha256"], final.core.transport.SOURCE_SHA256)
        self.assertEqual(request["historical_repaired_static_review_sha256_not_authority"],
                         final.REPAIRED_STATIC_REVIEW_SHA256)
        self.assertFalse(request["dispatch_authorized"])
        self.assertTrue(request["preflight_to_write_identity_binding_required"])
        self.assertTrue(request["live_remote_stop_process_safety_before_o_excl_required"])
        self.assertFalse(request["source_checkout_authorized"])
        self.assertFalse(request["native_run_authorized"])
        self.assertFalse(request["eval_authorized"])

    def test_final_review_absence_holds_before_marker_or_any_ssh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (mock.patch.object(final, "FINAL_REVIEW", root / "absent-review.json"),
                  mock.patch.object(final, "ASSIGNMENT", root / "absent-assignment.json"),
                  mock.patch.object(final, "_consume_fixed_marker",
                                    side_effect=AssertionError("marker reached")),
                  mock.patch.object(final.core, "_run_ssh",
                                    side_effect=AssertionError("SSH reached"))):
                with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                    final.run_once()

    def test_old_or_forged_review_files_cannot_enable_disabled_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = root / "forged-review.json"
            assignment = root / "forged-assignment.json"
            review.write_text('{"verdict":"PASS"}', encoding="utf-8")
            assignment.write_text('{"dispatch_authorized":true}', encoding="utf-8")
            with (mock.patch.object(final, "FINAL_REVIEW", review),
                  mock.patch.object(final, "ASSIGNMENT", assignment),
                  mock.patch.object(final, "_consume_fixed_marker",
                                    side_effect=AssertionError("marker reached")),
                  mock.patch.object(final.core, "_run_ssh",
                                    side_effect=AssertionError("SSH reached"))):
                with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                    final.run_once()

    def test_flag_or_code_binding_drift_cannot_inherit_static_pass(self):
        request = json.loads(final.REQUEST.read_text(encoding="utf-8"))
        request["source_checkout_authorized"] = True
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "altered.json"
            altered.write_text(json.dumps(request), encoding="utf-8")
            with mock.patch.object(final, "REQUEST", altered):
                with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                    final._read_frozen_request_and_code()
        with mock.patch.object(final, "CORE_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                final._read_frozen_request_and_code()

    def test_structured_review_and_supervisor_bind_actual_file_bytes(self):
        request, binding = self._binding()
        review = final._expected_final_review(request, binding)
        review_bytes = json.dumps(review, sort_keys=True).encode("utf-8")
        assignment = final._expected_assignment(request, binding,
                                                final.core._sha(review_bytes))
        assignment_bytes = json.dumps(assignment, sort_keys=True).encode("utf-8")
        authority = final._validate_authority_files(request, binding,
                                                    review_bytes, assignment_bytes)
        self.assertEqual(authority["final_review_sha256"], final.core._sha(review_bytes))
        self.assertEqual(authority["supervisor_assignment_sha256"],
                         final.core._sha(assignment_bytes))
        with self.assertRaisesRegex(RuntimeError, "malformed review"):
            final._validate_authority_files(request, binding, b"PASS", assignment_bytes)
        stale = copy.deepcopy(review)
        stale["final_wrapper_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "not exact PASS"):
            final._validate_authority_files(request, binding,
                                            json.dumps(stale).encode("utf-8"),
                                            assignment_bytes)
        unbound = copy.deepcopy(assignment)
        unbound["final_independent_review_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            final._validate_authority_files(request, binding, review_bytes,
                                            json.dumps(unbound).encode("utf-8"))
        expanded = copy.deepcopy(assignment)
        expanded["native_run_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            final._validate_authority_files(request, binding, review_bytes,
                                            json.dumps(expanded).encode("utf-8"))
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            final._validate_authority_files(request, binding,
                                            json.dumps(review, indent=2).encode("utf-8"),
                                            assignment_bytes)

    def test_fixed_marker_is_private_exclusive_and_never_retried(self):
        authority = {"request_sha256": final.REQUEST_SHA256,
                     "final_review_sha256": "a" * 64,
                     "supervisor_assignment_sha256": "b" * 64}
        with tempfile.TemporaryDirectory(dir=final.MARKER.parent) as directory:
            marker = Path(directory) / "one-attempt.marker"
            with mock.patch.object(final, "MARKER", marker):
                digest = final._consume_fixed_marker(authority)
                before = marker.read_bytes()
                self.assertEqual(final.core._sha(before), digest)
                self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
                with self.assertRaises(FileExistsError):
                    final._consume_fixed_marker(authority)
                self.assertEqual(marker.read_bytes(), before)

    def test_prewrite_local_stop_attempts_postflight_but_no_write(self):
        calls = []
        checks = 0
        authority = {"request_sha256": final.REQUEST_SHA256,
                     "final_review_sha256": "a" * 64,
                     "supervisor_assignment_sha256": "b" * 64}

        class FakeJournal:
            def emit(self, event, **details):
                pass
            def close(self):
                pass

        def breaker_check():
            nonlocal checks
            checks += 1
            if checks == 2:
                raise RuntimeError("local STOP appeared")
            return {}

        def ssh(operation, journal):
            calls.append(operation)
            return {"old_inventory": {}}

        with (mock.patch.object(final, "authorize_final", return_value=authority),
              mock.patch.object(final.core.transport, "checked_local_source",
                                return_value={"sha256": final.core.transport.SOURCE_SHA256}),
              mock.patch.object(final.core, "_check_controller_breakers",
                                side_effect=breaker_check),
              mock.patch.object(final, "_consume_fixed_marker", return_value="c" * 64),
              mock.patch.object(final.core, "Journal", FakeJournal),
              mock.patch.object(final.core, "_pinned", return_value=b"marker"),
              mock.patch.object(final.core, "_run_ssh", side_effect=ssh),
              mock.patch.object(final.core.transport, "validate_probe"),
              mock.patch.object(final.core.transport, "compare_postflight")):
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                final.run_once()
        self.assertEqual(calls, ["probe", "probe"])


if __name__ == "__main__":
    unittest.main()
