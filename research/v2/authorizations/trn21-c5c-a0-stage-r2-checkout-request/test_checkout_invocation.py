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


if __name__ == "__main__":
    unittest.main()
