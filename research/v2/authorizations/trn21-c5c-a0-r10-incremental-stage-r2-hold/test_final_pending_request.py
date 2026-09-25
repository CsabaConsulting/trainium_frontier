"""Offline checks for the exact final-review-pending trn21 stage request.

No test creates a final review or assignment, invokes SSH, or uses the live
one-attempt marker. A later review-file integration must not invalidate these
synthetic missing-authority checks.
"""

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("trn21_r10_r4_final_pending", HERE / "operator_once.py")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


class FinalPendingRequestTests(unittest.TestCase):
    def test_exact_request_is_canonical_and_stage_only(self):
        request, request_hash = operator._json(operator.ENABLED_REQUEST)
        bundle, remote_source = operator._source_bytes()
        operator_hash = operator._sha(operator._regular(HERE / "operator_once.py"))
        self.assertEqual(request_hash,
                         "cc2bddbdda631c79f6fc8b6a7555504a4d994c3c3a5665d203c74e1fdfd72a76")
        self.assertEqual(request["remote_source_sha256"], operator._sha(remote_source))
        self.assertEqual(request["operator_source_sha256"], operator_hash)
        self.assertEqual(request["bundle_sha256"], operator._sha(bundle))
        self.assertEqual(request["bundle_bytes"], len(bundle))
        self.assertEqual(request["source_commit"], operator.BASE_COMMIT)
        self.assertEqual(request["target_commit"], operator.TARGET_COMMIT)
        self.assertEqual(request["attempt_label"], operator.LABEL)
        self.assertEqual(request["marker"], str(operator.MARKER))
        self.assertIs(request["source_stage_authorized"], True)
        self.assertIs(request["native_training_authorized"], False)
        self.assertIs(request["evaluation_authorized"], False)
        self.assertEqual(hashlib.sha256((operator.PACKAGE_ROOT / "train.py").read_bytes()).hexdigest(),
                         "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586")
        self.assertEqual(hashlib.sha256((operator.PACKAGE_ROOT / "prepare.py").read_bytes()).hexdigest(),
                         "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb")

        # The frozen operator accepts this exact request shape, but cannot
        # interpret a missing independent final review as authority.
        with self.assertRaisesRegex(RuntimeError, "independent final source-stage PASS"):
            operator._authority(request, request_hash, {}, "no-review", {},
                                operator._sha(remote_source), operator_hash)
        forged = dict(request)
        forged["native_training_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "no exact enabled"):
            operator._authority(forged, request_hash, {}, "no-review", {},
                                operator._sha(remote_source), operator_hash)

    def test_pre_review_invocation_refuses_before_marker_or_ssh(self):
        with tempfile.TemporaryDirectory(prefix="trn21-r4-final-pending-") as temp:
            root = Path(temp)
            marker = root / "one-attempt.jsonl"
            contact = mock.Mock(side_effect=AssertionError("remote phase contacted"))
            with mock.patch.object(operator, "FINAL_REVIEW", root / "missing-review.json"), \
                 mock.patch.object(operator, "ASSIGNMENT", root / "missing-assignment.json"), \
                 mock.patch.object(operator, "MARKER", marker), \
                 mock.patch.object(operator, "_local_stop_and_breakers") as controller, \
                 mock.patch.object(operator, "_ssh_identity") as identity, \
                 mock.patch.object(operator, "_remote_timeout_identity") as probe:
                if sys.flags.optimize:
                    with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                        operator.stage_once(remote_call=contact)
                else:
                    with self.assertRaises(FileNotFoundError):
                        operator.stage_once(remote_call=contact)
            self.assertFalse(marker.exists())
            controller.assert_not_called()
            identity.assert_not_called()
            probe.assert_not_called()
            contact.assert_not_called()


if __name__ == "__main__":
    unittest.main()
