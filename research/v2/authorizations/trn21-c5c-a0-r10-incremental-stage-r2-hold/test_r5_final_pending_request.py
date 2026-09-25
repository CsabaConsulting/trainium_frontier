"""Offline authority checks for the r5 final-review-pending stage request.

Missing review/assignment checks are synthetic so later legitimate file
integration does not make these tests contact a host or fail by lifecycle.
"""

import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("trn21_r10_r5_final_pending", HERE / "operator_once.py")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


class R5FinalPendingRequestTests(unittest.TestCase):
    def test_exact_canonical_source_stage_request(self):
        request, request_hash = operator._json(operator.ENABLED_REQUEST)
        bundle, remote_source = operator._source_bytes()
        operator_hash = operator._sha(operator._regular(HERE / "operator_once.py"))
        self.assertEqual(request_hash,
                         "ad7a4dd6d4b93b705e936d781c0817bfdec7cb12560567aa3128e3f2b879c36a")
        self.assertEqual(request["operator_source_sha256"], operator_hash)
        self.assertEqual(request["remote_source_sha256"], operator._sha(remote_source))
        self.assertEqual(request["bundle_sha256"], operator._sha(bundle))
        self.assertEqual(request["bundle_bytes"], len(bundle))
        self.assertEqual(request["source_commit"], operator.BASE_COMMIT)
        self.assertEqual(request["target_commit"], operator.TARGET_COMMIT)
        self.assertEqual(request["source"], operator.SOURCE)
        self.assertEqual(request["target"], operator.TARGET)
        self.assertEqual(request["attempt_label"], operator.LABEL)
        self.assertEqual(request["marker"], str(operator.MARKER))
        self.assertIs(request["source_stage_authorized"], True)
        self.assertIs(request["native_training_authorized"], False)
        self.assertIs(request["evaluation_authorized"], False)
        self.assertEqual(hashlib.sha256((operator.PACKAGE_ROOT / "train.py").read_bytes()).hexdigest(),
                         "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586")
        self.assertEqual(hashlib.sha256((operator.PACKAGE_ROOT / "prepare.py").read_bytes()).hexdigest(),
                         "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb")

        # This is the exact frozen v2 request shape; an absent independent
        # final review is still HOLD, and native authority cannot be added.
        with self.assertRaisesRegex(RuntimeError, "independent final source-stage PASS"):
            operator._authority(request, request_hash, {}, "no-review", {},
                                operator._sha(remote_source), operator_hash)
        forged = dict(request)
        forged["native_training_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "no exact enabled"):
            operator._authority(forged, request_hash, {}, "no-review", {},
                                operator._sha(remote_source), operator_hash)

    def test_missing_final_review_or_assignment_refuses_before_ssh(self):
        request, request_hash = operator._json(operator.ENABLED_REQUEST)
        for missing in ("review", "assignment"):
            with self.subTest(missing=missing):
                replies = [(request, request_hash)]
                if missing == "assignment":
                    replies.append(({}, "synthetic-review-hash"))
                replies.append(FileNotFoundError("synthetic missing " + missing))
                phase = mock.Mock(side_effect=AssertionError("phase SSH contacted"))
                with mock.patch.object(operator, "_json", side_effect=replies), \
                     mock.patch.object(operator, "_local_stop_and_breakers") as controller, \
                     mock.patch.object(operator, "_ssh_identity") as identity, \
                     mock.patch.object(operator, "_remote_timeout_identity") as probe, \
                     mock.patch.object(operator, "_marker") as marker:
                    if sys.flags.optimize:
                        with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                            operator.stage_once(remote_call=phase)
                    else:
                        with self.assertRaisesRegex(FileNotFoundError, "synthetic missing"):
                            operator.stage_once(remote_call=phase)
                controller.assert_not_called()
                identity.assert_not_called()
                probe.assert_not_called()
                marker.assert_not_called()
                phase.assert_not_called()


if __name__ == "__main__":
    unittest.main()
