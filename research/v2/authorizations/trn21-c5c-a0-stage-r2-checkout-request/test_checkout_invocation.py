"""Synthetic, local-only checks for the disabled one-attempt checkout draft."""

import copy
import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("checkout_invocation", HERE / "checkout_invocation.py")
invocation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(invocation)


def audit_snapshot(plan, *, checkout_exists=False):
    snapshot = copy.deepcopy(plan["terminal"]["final_postflight_snapshot"])
    snapshot["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if checkout_exists:
        snapshot["targets"][plan["request"]["checkout_path"]] = {
            "exists": True, "type": "directory", "mode": "0o700",
            "uid": 1000, "device": 66305, "inode": 3999999,
        }
    return snapshot


def result(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class CheckoutInvocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = invocation.build_plan()

    def run_synthetic(self, replies):
        calls = []

        def transport(argv, *, input_text, timeout_seconds):
            calls.append((argv, input_text, timeout_seconds))
            reply = replies[len(calls) - 1]
            if isinstance(reply, BaseException):
                raise reply
            return reply

        with tempfile.TemporaryDirectory() as directory:
            journal_path = Path(directory) / "one-attempt.jsonl"
            journal = invocation.Journal(journal_path)
            try:
                outcome = invocation._run_with_unconditional_postflight(
                    self.plan, journal, transport)
            finally:
                journal.close()
            lines = [json.loads(line) for line in journal_path.read_text().splitlines()]
        return outcome, calls, lines

    def enabled_local_fixture(self, directory, *, write_review=True):
        plan = copy.deepcopy(self.plan)
        plan["request"]["checkout_authorized"] = True
        plan["binding"]["dispatch_authorized"] = True
        marker = Path(directory) / "assignment-bound-attempt-once.jsonl"
        plan["request"]["attempt_marker_path"] = str(marker)
        plan["binding"]["attempt_marker_path"] = str(marker)
        relative = plan["request"]["independent_exact_request_review_path"]
        review = {
            "schema": "trn21-c5c-a0-checkout-independent-final-request-review-v1",
            "verdict": "PASS_EXACT_CHECKOUT_REQUEST_ONLY",
            "prepared_by_role": "verifier",
            "host": plan["request"]["host"],
            "checkout_path": plan["request"]["checkout_path"],
            "source_commit": plan["request"]["source_commit"],
            "reviewed_request_sha256": plan["request_sha256"],
            "reviewed_binding_sha256": plan["binding_sha256"],
            "reviewed_operator_wrapper_sha256": plan["request"]["operator_wrapper_sha256"],
            "reviewed_remote_adapter_sha256": plan["request"]["remote_adapter_sha256"],
            "reviewed_audit_argv_sha256": plan["binding"]["audit_argv_sha256"],
            "reviewed_stage_argv_sha256": plan["binding"]["stage_argv_sha256"],
            "reviewed_stage_stdin_sha256": plan["binding"]["stage_stdin_sha256"],
            "attempt_marker_path": str(marker),
            "checkout_only": True,
            "cpu_g2_authorized": False,
            "native_training_authorized": False,
            "evaluation_authorized": False,
        }
        review_bytes = invocation.canonical_json(review) + b"\n"
        review_file = Path(directory) / relative
        if write_review:
            review_file.parent.mkdir(parents=True, exist_ok=True)
            review_file.write_bytes(review_bytes)
            review_file.chmod(0o600)
        assignment = {
            "schema": "trn21-c5c-a0-checkout-only-assignment-v1",
            "checkout_only_authorized": True, "one_attempt_only": True,
            "host": plan["request"]["host"],
            "source_commit": plan["request"]["source_commit"],
            "request_sha256": plan["request_sha256"],
            "binding_sha256": plan["binding_sha256"],
            "stage_argv_sha256": plan["binding"]["stage_argv_sha256"],
            "audit_argv_sha256": plan["binding"]["audit_argv_sha256"],
            "attempt_marker_path": str(marker),
            "independent_request_review_path": relative,
            "independent_request_review_sha256": invocation.sha256(review_bytes),
            "fresh_local_breaker_and_stop_gate": True,
        }
        return plan, assignment, marker, review_file

    def test_exact_evidence_request_and_argv_binding(self):
        p = self.plan
        self.assertEqual(p["request_sha256"], p["binding"]["request_sha256"])
        self.assertEqual(p["request"]["persistence_bundle_sha256"],
                         invocation.PERSISTENCE_BUNDLE_SHA256)
        self.assertEqual(invocation.argv_sha256(p["audit_argv"]),
                         p["binding"]["audit_argv_sha256"])
        self.assertEqual(invocation.argv_sha256(p["stage_argv"]),
                         p["binding"]["stage_argv_sha256"])
        self.assertEqual(invocation.sha256(p["stage_stdin"].encode()),
                         p["binding"]["stage_stdin_sha256"])
        self.assertEqual(len(p["stage_stdin"].encode()),
                         p["binding"]["stage_stdin_bytes"])
        self.assertEqual(p["audit_argv"][:-1], p["stage_argv"][:-1])
        self.assertFalse(p["request"]["checkout_authorized"])
        self.assertFalse(p["binding"]["dispatch_authorized"])

    def test_operator_entry_and_remote_adapter_refuse_disabled_request(self):
        calls = []
        with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
            invocation.operator_once({}, "/nonexistent/no-journal", lambda *a, **k: calls.append(a))
        self.assertEqual(calls, [])
        p = self.plan
        run = subprocess.run(
            [sys.executable, str(invocation.REMOTE_ADAPTER), p["request_sha256"],
             invocation.EXTRACTION_CODE_SHA256, invocation.PERSISTENCE_BUNDLE_SHA256],
            input=p["stage_stdin"], text=True, capture_output=True, timeout=10)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(json.loads(run.stdout)["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
        self.assertIn("disabled", json.loads(run.stdout)["error"])

    def test_stage_exception_always_gets_separate_full_postflight(self):
        pre = result(stdout=json.dumps(audit_snapshot(self.plan)))
        post = result(stdout=json.dumps(audit_snapshot(self.plan, checkout_exists=True)))
        outcome, calls, lines = self.run_synthetic(
            [pre, RuntimeError("synthetic extraction exception"), post])
        self.assertEqual(outcome["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
        self.assertTrue(outcome["stage_attempted"])
        self.assertEqual([c[0] for c in calls], [self.plan["audit_argv"],
                                                  self.plan["stage_argv"],
                                                  self.plan["audit_argv"]])
        self.assertEqual(sum(c[0] == self.plan["stage_argv"] for c in calls), 1)
        events = [entry["event"] for entry in lines]
        self.assertIn("CHECKOUT_ATTEMPT_HOLD", events)
        self.assertIn("FULL_READ_ONLY_POSTFLIGHT_RECORDED", events)
        self.assertEqual(events[-1], "TERMINAL_HOLD")
        self.assertIn("synthetic extraction exception", json.dumps(lines))
        self.assertFalse(lines[-1]["retry_authorized"])

    def test_preflight_failure_still_gets_postflight_without_stage(self):
        bad_pre = result(returncode=2, stderr="synthetic preflight failure")
        post = result(stdout=json.dumps(audit_snapshot(self.plan)))
        outcome, calls, lines = self.run_synthetic([bad_pre, post])
        self.assertEqual(outcome["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
        self.assertFalse(outcome["stage_attempted"])
        self.assertEqual([c[0] for c in calls], [self.plan["audit_argv"]] * 2)
        self.assertIn("FULL_READ_ONLY_POSTFLIGHT_RECORDED",
                      [entry["event"] for entry in lines])

    def test_nonzero_stage_and_postflight_failure_both_archive_hold(self):
        pre = result(stdout=json.dumps(audit_snapshot(self.plan)))
        stage = result(returncode=2, stdout='{"status":"HOLD"}')
        bad_post = RuntimeError("synthetic postflight transport exception")
        outcome, calls, lines = self.run_synthetic([pre, stage, bad_post])
        self.assertEqual(outcome["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
        self.assertEqual(len(calls), 3)
        events = [entry["event"] for entry in lines]
        self.assertIn("CHECKOUT_ATTEMPT_HOLD", events)
        self.assertIn("FULL_POSTFLIGHT_HOLD", events)
        self.assertEqual(events[-1], "TERMINAL_HOLD")

    def test_review_only_success_never_authorizes_g2_native_or_eval(self):
        pre = result(stdout=json.dumps(audit_snapshot(self.plan)))
        stage = result(stdout=json.dumps({
            "status": "SOURCE_CHECKOUT_VERIFIED_FOR_INDEPENDENT_REVIEW_ONLY",
            "cpu_g2_authorized": False, "native_training_authorized": False,
            "evaluation_authorized": False,
        }))
        post = result(stdout=json.dumps(audit_snapshot(self.plan, checkout_exists=True)))
        outcome, calls, lines = self.run_synthetic([pre, stage, post])
        self.assertEqual(len(calls), 3)
        self.assertEqual(outcome["status"], "STAGE_EVIDENCE_READY_FOR_INDEPENDENT_REVIEW_ONLY")
        self.assertFalse(lines[-1]["cpu_g2_authorized"])
        self.assertFalse(lines[-1]["native_training_authorized"])
        self.assertFalse(lines[-1]["evaluation_authorized"])

    def test_snapshot_tamper_rejected(self):
        snapshot = audit_snapshot(self.plan)
        snapshot["targets"][self.plan["request"]["source_capsule_path"]]["inode"] += 1
        with self.assertRaisesRegex(RuntimeError, "persistent artifact changed"):
            invocation.validate_full_snapshot(snapshot, self.plan["request"], before_stage=True)

    def test_same_assignment_cannot_replay_with_a_different_journal_path(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, assignment, marker, _ = self.enabled_local_fixture(directory)
            calls = []

            def transport(argv, *, input_text, timeout_seconds):
                self.assertTrue(marker.exists(), "exclusive marker must precede host contact")
                calls.append(argv)
                if argv == plan["stage_argv"]:
                    return result(returncode=2, stdout='{"status":"HOLD"}')
                return result(stdout=json.dumps(audit_snapshot(plan)))

            with patch.object(invocation, "build_plan", return_value=plan), \
                    patch.object(invocation, "REPO", Path(directory)):
                first = invocation.operator_once(assignment, marker, transport)
                self.assertEqual(first["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
                alternate = Path(directory) / "second-journal.jsonl"
                with self.assertRaisesRegex(RuntimeError, "caller path differs"):
                    invocation.operator_once(assignment, alternate, transport)
                with self.assertRaises(FileExistsError):
                    invocation.operator_once(assignment, marker, transport)
            self.assertEqual(calls, [plan["audit_argv"], plan["stage_argv"],
                                     plan["audit_argv"]])
            self.assertFalse(alternate.exists())
            self.assertTrue(marker.exists())
            records = [json.loads(line) for line in marker.read_text().splitlines()]
            self.assertEqual(records[0]["event"], "EXACT_CHECKOUT_ASSIGNMENT_BOUND")
            self.assertTrue(records[0]["marker_consumed_before_host_contact"])
            self.assertFalse(records[-1]["retry_authorized"])

    def test_preflight_failure_consumes_marker_and_still_postflights(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, assignment, marker, _ = self.enabled_local_fixture(directory)
            calls = []

            def transport(argv, *, input_text, timeout_seconds):
                self.assertTrue(marker.exists())
                calls.append(argv)
                if len(calls) == 1:
                    return result(returncode=2, stderr="synthetic preflight failure")
                return result(stdout=json.dumps(audit_snapshot(plan)))

            with patch.object(invocation, "build_plan", return_value=plan), \
                    patch.object(invocation, "REPO", Path(directory)):
                first = invocation.operator_once(assignment, marker, transport)
                self.assertEqual(first["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
                with self.assertRaises(FileExistsError):
                    invocation.operator_once(assignment, marker, transport)
            self.assertEqual(calls, [plan["audit_argv"], plan["audit_argv"]])

    def test_nonhex_nonexistent_or_wrong_review_never_reaches_marker_or_host(self):
        for case in ("nonhex", "nonexistent", "wrong_hash", "hold_verdict"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                plan, assignment, marker, review_file = self.enabled_local_fixture(directory)
                if case == "nonhex":
                    assignment["independent_request_review_sha256"] = "z" * 64
                elif case == "nonexistent":
                    review_file.unlink()
                elif case == "wrong_hash":
                    assignment["independent_request_review_sha256"] = "0" * 64
                else:
                    content = json.loads(review_file.read_text())
                    content["verdict"] = "HOLD"
                    data = invocation.canonical_json(content) + b"\n"
                    review_file.write_bytes(data)
                    assignment["independent_request_review_sha256"] = invocation.sha256(data)
                calls = []
                with patch.object(invocation, "build_plan", return_value=plan), \
                        patch.object(invocation, "REPO", Path(directory)):
                    with self.assertRaises(RuntimeError):
                        invocation.operator_once(assignment, marker,
                                                 lambda *a, **k: calls.append(a))
                self.assertEqual(calls, [])
                self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
