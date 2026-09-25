"""Independent, local-only adversarial checks of enabled trn21 checkout bytes.

All Operator transports in this module are synthetic. Never use a real marker
path or a real SSH transport here.
"""

import copy
import datetime
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request"
spec = importlib.util.spec_from_file_location("independent_final_checkout", PACKAGE / "checkout_invocation_final.py")
inv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inv)


def answer(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def fresh_snapshot(plan, checkout=False):
    item = copy.deepcopy(plan["terminal"]["final_postflight_snapshot"])
    item["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if checkout:
        item["targets"][plan["request"]["checkout_path"]] = {
            "exists": True, "type": "directory", "device": 66305,
            "inode": 3999999, "uid": 1000, "mode": "0o700",
        }
    return item


def synthetic_authority(original, temp):
    plan = copy.deepcopy(original)
    marker = Path(temp) / "fixed-marker.jsonl"
    plan["request"]["attempt_marker_path"] = str(marker)
    plan["binding"]["attempt_marker_path"] = str(marker)
    request = plan["request"]
    binding = plan["binding"]
    review = {
        "schema": "trn21-c5c-a0-checkout-independent-final-request-review-v1",
        "verdict": "PASS_EXACT_CHECKOUT_REQUEST_ONLY",
        "prepared_by_role": "verifier",
        "host": request["host"],
        "checkout_path": request["checkout_path"],
        "source_commit": request["source_commit"],
        "source_capsule_sha256": request["source_capsule_sha256"],
        "train_sha256": request["train_sha256"],
        "prepare_sha256": request["prepare_sha256"],
        "reviewed_request_sha256": plan["request_sha256"],
        "reviewed_binding_sha256": plan["binding_sha256"],
        "reviewed_operator_wrapper_sha256": request["operator_wrapper_sha256"],
        "reviewed_remote_adapter_sha256": request["remote_adapter_sha256"],
        "reviewed_extraction_code_sha256": request["extraction_code_sha256"],
        "reviewed_persistence_bundle_sha256": request["persistence_bundle_sha256"],
        "reviewed_audit_argv_sha256": binding["audit_argv_sha256"],
        "reviewed_stage_argv_sha256": binding["stage_argv_sha256"],
        "reviewed_stage_stdin_sha256": binding["stage_stdin_sha256"],
        "attempt_marker_path": str(marker),
        "checkout_only": True,
        "cpu_g2_authorized": False,
        "native_training_authorized": False,
        "evaluation_authorized": False,
    }
    relative = request["independent_exact_request_review_path"]
    review_file = Path(temp) / relative
    review_file.parent.mkdir(parents=True)
    review_bytes = inv.canonical_json(review) + b"\n"
    review_file.write_bytes(review_bytes)
    review_file.chmod(0o600)
    assignment = {
        "schema": "trn21-c5c-a0-checkout-only-assignment-v1",
        "checkout_only_authorized": True, "one_attempt_only": True,
        "host": request["host"],
        "checkout_path": request["checkout_path"],
        "source_commit": request["source_commit"],
        "source_capsule_sha256": request["source_capsule_sha256"],
        "train_sha256": request["train_sha256"],
        "prepare_sha256": request["prepare_sha256"],
        "request_sha256": plan["request_sha256"],
        "binding_sha256": plan["binding_sha256"],
        "operator_wrapper_sha256": request["operator_wrapper_sha256"],
        "remote_adapter_sha256": request["remote_adapter_sha256"],
        "extraction_code_sha256": request["extraction_code_sha256"],
        "audit_argv_sha256": binding["audit_argv_sha256"],
        "stage_argv_sha256": binding["stage_argv_sha256"],
        "stage_stdin_sha256": binding["stage_stdin_sha256"],
        "attempt_marker_path": str(marker),
        "independent_request_review_path": relative,
        "independent_request_review_sha256": inv.sha256(review_bytes),
        "fresh_local_breaker_and_stop_gate": True,
        "cpu_g2_authorized": False,
        "native_training_authorized": False,
        "evaluation_authorized": False,
    }
    return plan, assignment, marker, review_file


class FinalCheckoutIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = inv.build_plan()

    def test_exact_frozen_source_target_and_argv_pins(self):
        contract = json.loads((PACKAGE / "GATE_CONTRACT.FINAL_PENDING_REVIEW.json").read_text())
        pins = contract["exact_enabled_files_pending_final_review"]
        for key, name in (
            ("enabled_request_sha256", "CHECKOUT_REQUEST.FINAL_PENDING_REVIEW.json"),
            ("enabled_invocation_binding_sha256", "INVOCATION_BINDING.FINAL_PENDING_REVIEW.json"),
            ("offline_operator_wrapper_sha256", "checkout_invocation_final.py"),
            ("remote_one_attempt_adapter_sha256", "remote_checkout_once.py"),
            ("synthetic_tests_sha256", "test_checkout_invocation_final.py"),
        ):
            self.assertEqual(inv.sha256((PACKAGE / name).read_bytes()), pins[key])
        exact = contract["exact_invocation"]
        self.assertEqual(inv.argv_sha256(self.plan["audit_argv"]), exact["fresh_full_read_only_preflight_argv_sha256"])
        self.assertEqual(inv.argv_sha256(self.plan["stage_argv"]), exact["one_checkout_attempt_argv_sha256"])
        self.assertEqual(inv.sha256(self.plan["stage_stdin"].encode()), exact["one_checkout_attempt_stdin_sha256"])
        self.assertEqual(len(self.plan["stage_stdin"].encode()), exact["one_checkout_attempt_stdin_bytes"])
        self.assertEqual(self.plan["request"]["source_commit"], contract["source_commit"])
        self.assertEqual(self.plan["request"]["checkout_path"], contract["absent_checkout_one_attempt_path"])
        self.assertEqual(self.plan["request"]["source_capsule_sha256"], contract["source_capsule_sha256"])
        self.assertTrue(self.plan["request"]["checkout_authorized"])
        self.assertTrue(self.plan["binding"]["dispatch_authorized"])
        self.assertFalse(contract["checkout_dispatch_authorized"])
        self.assertFalse(contract["cpu_g2_authorized"])
        self.assertFalse(contract["native_training_authorized"])
        self.assertFalse(contract["evaluation_authorized"])
        self.assertIsNone(self.plan["request"]["independent_exact_request_review_sha256"])

    def test_actual_entry_refuses_absent_assignment_without_transport(self):
        calls = []
        with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
            inv.operator_once({}, "/nonexistent/test-marker", lambda *a, **k: calls.append(a))
        self.assertEqual(calls, [])

    def test_actual_final_review_file_authenticates_without_host_contact(self):
        relative = self.plan["request"]["independent_exact_request_review_path"]
        data = (REPO / relative).read_bytes()
        digest = inv.sha256(data)
        matched = inv.verify_independent_request_review(self.plan, {
            "independent_request_review_path": relative,
            "independent_request_review_sha256": digest,
        })
        self.assertEqual(matched, {"path": relative, "sha256": digest})

    def test_forged_review_and_assignment_fail_before_marker_or_transport(self):
        cases = ("absent", "nonhex", "wrong_path", "wrong_sha", "hold", "wrong_binding",
                 "wrong_source", "wrong_target", "symlink", "world_writable", "assignment_argv",
                 "assignment_g2")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                plan, assignment, marker, review_file = synthetic_authority(self.plan, temp)
                if case == "absent":
                    review_file.unlink()
                elif case == "nonhex":
                    assignment["independent_request_review_sha256"] = "z" * 64
                elif case == "wrong_path":
                    assignment["independent_request_review_path"] = "other.json"
                elif case == "wrong_sha":
                    assignment["independent_request_review_sha256"] = "0" * 64
                elif case in ("hold", "wrong_binding", "wrong_source", "wrong_target"):
                    review = json.loads(review_file.read_text())
                    field = {"hold": "verdict", "wrong_binding": "reviewed_binding_sha256",
                             "wrong_source": "source_capsule_sha256", "wrong_target": "checkout_path"}[case]
                    review[field] = "HOLD" if case == "hold" else "wrong"
                    data = inv.canonical_json(review) + b"\n"
                    review_file.write_bytes(data)
                    assignment["independent_request_review_sha256"] = inv.sha256(data)
                elif case == "symlink":
                    source = Path(temp) / "alternate-review.json"
                    review_file.rename(source)
                    review_file.symlink_to(source)
                elif case == "world_writable":
                    review_file.chmod(0o666)
                elif case == "assignment_argv":
                    assignment["stage_argv_sha256"] = "0" * 64
                else:
                    assignment["cpu_g2_authorized"] = True
                calls = []
                with patch.object(inv, "build_plan", return_value=plan), patch.object(inv, "REPO", Path(temp)):
                    with self.assertRaises((RuntimeError, OSError)):
                        inv.operator_once(assignment, marker, lambda *a, **k: calls.append(a))
                self.assertEqual(calls, [])
                self.assertFalse(marker.exists())

    def test_precreation_stage_failure_consumes_one_marker_and_postflights(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, assignment, marker, _ = synthetic_authority(self.plan, temp)
            calls = []

            def transport(argv, *, input_text, timeout_seconds):
                self.assertTrue(marker.exists())
                calls.append(argv)
                if argv == plan["stage_argv"]:
                    raise RuntimeError("synthetic precreation extraction failure")
                return answer(stdout=json.dumps(fresh_snapshot(plan)))

            with patch.object(inv, "build_plan", return_value=plan), patch.object(inv, "REPO", Path(temp)):
                result = inv.operator_once(assignment, marker, transport)
                self.assertEqual(result["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
                with self.assertRaises(RuntimeError):
                    inv.operator_once(assignment, Path(temp) / "other-marker.jsonl", transport)
                with self.assertRaises(FileExistsError):
                    inv.operator_once(assignment, marker, transport)
            self.assertEqual(calls, [plan["audit_argv"], plan["stage_argv"], plan["audit_argv"]])
            self.assertFalse((Path(temp) / "other-marker.jsonl").exists())
            records = [json.loads(line) for line in marker.read_text().splitlines()]
            self.assertEqual(records[0]["event"], "EXACT_CHECKOUT_ASSIGNMENT_BOUND")
            self.assertEqual(records[-1]["event"], "TERMINAL_HOLD")
            self.assertFalse(records[-1]["retry_authorized"])
            self.assertTrue(records[-1]["preserve_partial"])

    def test_preflight_failure_never_stages_but_postflights(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, assignment, marker, _ = synthetic_authority(self.plan, temp)
            calls = []

            def transport(argv, *, input_text, timeout_seconds):
                calls.append(argv)
                if len(calls) == 1:
                    return answer(returncode=2, stderr="synthetic preflight error")
                return answer(stdout=json.dumps(fresh_snapshot(plan)))

            with patch.object(inv, "build_plan", return_value=plan), patch.object(inv, "REPO", Path(temp)):
                result = inv.operator_once(assignment, marker, transport)
            self.assertEqual(result["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
            self.assertEqual(calls, [plan["audit_argv"], plan["audit_argv"]])
            records = [json.loads(line) for line in marker.read_text().splitlines()]
            self.assertEqual(records[-1]["event"], "TERMINAL_HOLD")
            self.assertFalse(records[-1]["stage_attempted"])

    def test_protected_assets_host_and_target_drift_fail_closed(self):
        request = self.plan["request"]
        for case in ("hostname", "root_free", "capsule_inode", "checkpoint_sha",
                     "cache_inode", "stop", "checkout_exists"):
            with self.subTest(case=case):
                snap = fresh_snapshot(self.plan)
                if case == "hostname":
                    snap["hostname"] = "other-host"
                elif case == "root_free":
                    snap["root_free_bytes"] = 1
                elif case == "capsule_inode":
                    snap["targets"][request["source_capsule_path"]]["inode"] += 1
                elif case == "checkpoint_sha":
                    snap["checkpoints"][0]["sha256"] = "0" * 64
                elif case == "cache_inode":
                    snap["caches"]["neff"]["inode"] += 1
                elif case == "stop":
                    path = request["required_stop_paths"][0]
                    snap["stops"][path]["exists"] = True
                else:
                    snap["targets"][request["checkout_path"]] = {
                        "exists": True, "type": "directory", "mode": "0o700",
                        "uid": 1000, "device": 66305, "inode": 3999999,
                    }
                with self.assertRaises(RuntimeError):
                    inv.validate_full_snapshot(snap, request, before_stage=True)


if __name__ == "__main__":
    unittest.main()
