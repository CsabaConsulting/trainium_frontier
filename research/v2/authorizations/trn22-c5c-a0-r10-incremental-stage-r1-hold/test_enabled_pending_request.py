"""Offline tests: an enabled request alone is never stage authority."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REQUEST = HERE / "REQUEST.ENABLED_PENDING_FINAL_REVIEW.json"
EXPECTED_REQUEST_SHA256 = "b33a72fc3b5588f537c23ff4bb565344a92eabef58a33970db2bf2160a60ee91"

spec = importlib.util.spec_from_file_location(
    "trn22_r10_stage_operator_under_test", HERE / "operator_once.py"
)
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class EnabledPendingRequestTests(unittest.TestCase):
    def test_request_is_canonical_and_matches_source_authority_schema(self):
        raw = REQUEST.read_bytes()
        request = json.loads(raw)
        self.assertEqual(raw.strip(), json.dumps(request, sort_keys=True,
                                                 separators=(",", ":")).encode())
        self.assertEqual(sha256(raw), EXPECTED_REQUEST_SHA256)
        self.assertTrue(request["dispatch_authorized"])
        self.assertTrue(request["source_stage_authorized"])
        self.assertFalse(request["native_training_authorized"])
        self.assertFalse(request["evaluation_authorized"])
        # The source itself is unchanged; the request contains every exact
        # field the local authority gate compares before any marker/SSH.
        expected = {
            "schema": "trn22-r10-incremental-stage-r1-enabled-request-v1",
            "dispatch_authorized": True,
            "source_stage_authorized": True,
            "scope": operator.SCOPE,
            "host": "trn22",
            "hostname": "ip-172-31-32-253",
            "source_commit": operator.BASE_COMMIT,
            "target_commit": operator.TARGET_COMMIT,
            "source": operator.SOURCE,
            "target": operator.TARGET,
            "remote_bundle": operator.REMOTE_BUNDLE,
            "bundle_sha256": operator.BUNDLE_SHA256,
            "bundle_bytes": operator.BUNDLE_BYTES,
            "remote_source_sha256": request["remote_source_sha256"],
            "operator_source_sha256": request["operator_source_sha256"],
            "attempt_label": operator.LABEL,
            "marker": str(operator.MARKER),
            "one_attempt": True,
            "prior_r3_hold_commit": operator.PRIOR_R3_HOLD_COMMIT,
            "prior_r3_hold_sha256": operator.PRIOR_R3_HOLD_SHA256,
            "r4_static_pass_commit": operator.R4_STATIC_PASS_COMMIT,
            "r4_static_pass_sha256": operator.R4_STATIC_PASS_SHA256,
            "r10_source_pass_commit": operator.R10_SOURCE_PASS_COMMIT,
            "r10_source_pass_sha256": operator.R10_SOURCE_PASS_SHA256,
            "provisional_failure_policy": operator.PROVISIONAL_FAILURE_POLICY,
            "native_training_authorized": False,
            "evaluation_authorized": False,
        }
        self.assertEqual(request, expected)

    def test_exact_unchanged_source_bundle_and_science_root(self):
        request = json.loads(REQUEST.read_text())
        self.assertEqual(
            sha256((HERE / "remote_incremental.py").read_bytes()),
            request["remote_source_sha256"],
        )
        self.assertEqual(
            sha256((HERE / "operator_once.py").read_bytes()),
            request["operator_source_sha256"],
        )
        bundle = (HERE / "e65-to-r10.bundle").read_bytes()
        self.assertEqual((len(bundle), sha256(bundle)),
                         (request["bundle_bytes"], request["bundle_sha256"]))
        self.assertEqual(
            sha256((ROOT / "train.py").read_bytes()),
            "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586",
        )
        self.assertEqual(
            sha256((ROOT / "prepare.py").read_bytes()),
            "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb",
        )
        disabled = json.loads((HERE / "REQUEST.HOLD.json").read_text())
        self.assertFalse(disabled["dispatch_authorized"])
        self.assertFalse(disabled["source_stage_authorized"])
        self.assertEqual(
            (request["source_commit"], request["target_commit"], request["source"],
             request["target"], request["remote_bundle"], request["attempt_label"],
             request["marker"]),
            (disabled["source_commit"], disabled["target_commit"], disabled["source_path"],
             disabled["new_target_path"], disabled["remote_bundle"],
             disabled["attempt_label"], disabled["one_attempt_marker"]),
        )

    def test_no_final_review_or_assignment_files_exist(self):
        self.assertFalse(operator.FINAL_REVIEW.exists())
        self.assertFalse(operator.ASSIGNMENT.exists())
        self.assertEqual(operator.ENABLED_REQUEST, REQUEST)

    def test_enabled_request_alone_refuses_before_ssh_or_marker(self):
        remote_call = mock.Mock(side_effect=AssertionError("remote called"))
        with (
            mock.patch.object(operator, "_local_stop_and_breakers") as stop,
            mock.patch.object(operator, "_ssh_identity") as ssh,
            mock.patch.object(operator, "_marker") as marker,
        ):
            if sys.flags.optimize:
                with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                    operator.stage_once(remote_call=remote_call)
            else:
                with self.assertRaises(FileNotFoundError):
                    operator.stage_once(remote_call=remote_call)
            stop.assert_not_called()
            ssh.assert_not_called()
            marker.assert_not_called()
            remote_call.assert_not_called()

    def test_even_an_in_memory_review_without_assignment_refuses(self):
        if sys.flags.optimize:
            self.skipTest("optimized Python refuses before any authority read")
        original_json = operator._json

        def without_assignment(path):
            if path == operator.FINAL_REVIEW:
                return {"schema": "in-memory-only"}, "0" * 64
            return original_json(path)

        remote_call = mock.Mock(side_effect=AssertionError("remote called"))
        with (
            mock.patch.object(operator, "_json", side_effect=without_assignment),
            mock.patch.object(operator, "_local_stop_and_breakers") as stop,
            mock.patch.object(operator, "_ssh_identity") as ssh,
            mock.patch.object(operator, "_marker") as marker,
        ):
            with self.assertRaises(FileNotFoundError):
                operator.stage_once(remote_call=remote_call)
            stop.assert_not_called()
            ssh.assert_not_called()
            marker.assert_not_called()
            remote_call.assert_not_called()

    def test_direct_entrypoint_holds_in_normal_and_optimized_python(self):
        for flags in (["-B"], ["-O", "-B"]):
            with self.subTest(flags=flags):
                result = subprocess.run(
                    [sys.executable, *flags, str(HERE / "operator_once.py")],
                    cwd=ROOT, stdin=subprocess.DEVNULL,
                    text=True, capture_output=True, timeout=15, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("HOLD:", result.stderr)


if __name__ == "__main__":
    unittest.main()
