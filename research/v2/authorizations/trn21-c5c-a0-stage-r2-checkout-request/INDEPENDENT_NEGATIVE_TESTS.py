"""Verifier-only local synthetic checks. This file never opens an SSH connection.

The current request remains disabled. These tests inject an in-memory enabled
plan solely to exercise the future Operator wrapper's authority checks.
"""

import copy
import datetime
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("checkout_invocation", HERE / "checkout_invocation.py")
invocation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(invocation)


def synthetic_snapshot(plan):
    snapshot = copy.deepcopy(plan["terminal"]["final_postflight_snapshot"])
    snapshot["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return snapshot


def synthetic_assignment(plan):
    return {
        "schema": "trn21-c5c-a0-checkout-only-assignment-v1",
        "checkout_only_authorized": True,
        "one_attempt_only": True,
        "host": plan["request"]["host"],
        "source_commit": plan["request"]["source_commit"],
        "request_sha256": plan["request_sha256"],
        "binding_sha256": plan["binding_sha256"],
        "stage_argv_sha256": plan["binding"]["stage_argv_sha256"],
        "audit_argv_sha256": plan["binding"]["audit_argv_sha256"],
        "independent_request_review_sha256": "z" * 64,
        "fresh_local_breaker_and_stop_gate": True,
    }


class IndependentNegativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = invocation.build_plan()

    def enabled_in_memory(self):
        plan = copy.deepcopy(self.plan)
        plan["request"]["checkout_authorized"] = True
        plan["binding"]["dispatch_authorized"] = True
        return plan

    def test_same_assignment_can_issue_two_stage_ssh_attempts(self):
        plan = self.enabled_in_memory()
        assignment = synthetic_assignment(plan)
        calls = []

        def transport(argv, *, input_text, timeout_seconds):
            calls.append(argv)
            if argv == plan["stage_argv"]:
                return SimpleNamespace(returncode=2, stdout='{"status":"HOLD"}', stderr="")
            return SimpleNamespace(returncode=0, stdout=json.dumps(synthetic_snapshot(plan)), stderr="")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(invocation, "build_plan", return_value=plan):
                first = invocation.operator_once(assignment, Path(directory) / "attempt-one.jsonl", transport)
                second = invocation.operator_once(assignment, Path(directory) / "attempt-two.jsonl", transport)
            self.assertEqual(first["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
            self.assertEqual(second["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
            self.assertEqual(len(list(Path(directory).glob("*.jsonl"))), 2)
        self.assertEqual(calls.count(plan["stage_argv"]), 2)
        self.assertEqual(calls, [plan["audit_argv"], plan["stage_argv"], plan["audit_argv"]] * 2)

    def test_non_hex_nonexistent_review_digest_is_accepted(self):
        plan = self.enabled_in_memory()
        assignment = synthetic_assignment(plan)
        calls = []

        def transport(argv, *, input_text, timeout_seconds):
            calls.append(argv)
            if argv == plan["stage_argv"]:
                return SimpleNamespace(returncode=2, stdout='{"status":"HOLD"}', stderr="")
            return SimpleNamespace(returncode=0, stdout=json.dumps(synthetic_snapshot(plan)), stderr="")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(invocation, "build_plan", return_value=plan):
                outcome = invocation.operator_once(assignment, Path(directory) / "attempt.jsonl", transport)
        self.assertEqual(outcome["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
        self.assertEqual(calls.count(plan["stage_argv"]), 1)


if __name__ == "__main__":
    unittest.main()
