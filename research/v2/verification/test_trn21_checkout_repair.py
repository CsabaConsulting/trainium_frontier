"""Independent offline adversarial tests for the repaired checkout request.

All transports are synthetic; this module must never contact a host.
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
spec = importlib.util.spec_from_file_location("independent_checkout_invocation", PACKAGE / "checkout_invocation.py")
inv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inv)


def response(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def snapshot(plan, *, checkout=False):
    value = copy.deepcopy(plan["terminal"]["final_postflight_snapshot"])
    value["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if checkout:
        value["targets"][plan["request"]["checkout_path"]] = {
            "exists": True, "type": "directory", "mode": "0o700",
            "uid": 1000, "device": 66305, "inode": 3999999,
        }
    return value


def fixture(plan, temp):
    plan = copy.deepcopy(plan)
    plan["request"]["checkout_authorized"] = True
    plan["binding"]["dispatch_authorized"] = True
    marker = Path(temp) / "one-fixed-marker.jsonl"
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
    review_file = Path(temp) / relative
    review_file.parent.mkdir(parents=True)
    review_bytes = inv.canonical_json(review) + b"\n"
    review_file.write_bytes(review_bytes)
    review_file.chmod(0o600)
    assignment = {
        "schema": "trn21-c5c-a0-checkout-only-assignment-v1",
        "checkout_only_authorized": True,
        "one_attempt_only": True,
        "host": plan["request"]["host"],
        "source_commit": plan["request"]["source_commit"],
        "request_sha256": plan["request_sha256"],
        "binding_sha256": plan["binding_sha256"],
        "stage_argv_sha256": plan["binding"]["stage_argv_sha256"],
        "audit_argv_sha256": plan["binding"]["audit_argv_sha256"],
        "attempt_marker_path": str(marker),
        "independent_request_review_path": relative,
        "independent_request_review_sha256": inv.sha256(review_bytes),
        "fresh_local_breaker_and_stop_gate": True,
    }
    return plan, assignment, marker, review_file


class IndependentCheckoutRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = inv.build_plan()

    def test_frozen_hashes_argv_payload_and_disabled_status(self):
        contract = json.loads((PACKAGE / "GATE_CONTRACT.json").read_text())
        pins = contract["draft_files"]
        for key, name in (
            ("disabled_request_sha256", "CHECKOUT_REQUEST.HOLD.json"),
            ("disabled_invocation_binding_sha256", "INVOCATION_BINDING.HOLD.json"),
            ("offline_operator_wrapper_sha256", "checkout_invocation.py"),
            ("remote_one_attempt_adapter_sha256", "remote_checkout_once.py"),
            ("synthetic_tests_sha256", "test_checkout_invocation.py"),
        ):
            self.assertEqual(inv.sha256((PACKAGE / name).read_bytes()), pins[key])
        exact = contract["exact_invocation"]
        self.assertEqual(inv.argv_sha256(self.plan["audit_argv"]), exact["fresh_full_read_only_preflight_argv_sha256"])
        self.assertEqual(inv.argv_sha256(self.plan["stage_argv"]), exact["one_checkout_attempt_argv_sha256"])
        self.assertEqual(inv.sha256(self.plan["stage_stdin"].encode()), exact["one_checkout_attempt_stdin_sha256"])
        self.assertEqual(len(self.plan["stage_stdin"].encode()), exact["one_checkout_attempt_stdin_bytes"])
        self.assertEqual(self.plan["request"]["source_commit"], contract["source_commit"])
        self.assertEqual(self.plan["request"]["source_capsule_sha256"], contract["source_capsule_sha256"])
        self.assertEqual(self.plan["request"]["checkout_path"], contract["absent_checkout_one_attempt_path"])
        self.assertFalse(self.plan["request"]["checkout_authorized"])
        self.assertFalse(self.plan["binding"]["dispatch_authorized"])
        self.assertIsNone(self.plan["request"]["independent_exact_request_review_sha256"])

    def test_same_assignment_precreation_failure_single_stage_even_alternate_path(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, assignment, marker, _ = fixture(self.plan, temp)
            calls = []

            def transport(argv, *, input_text, timeout_seconds):
                calls.append(argv)
                if argv == plan["stage_argv"]:
                    raise RuntimeError("synthetic precreation stage failure")
                return response(stdout=json.dumps(snapshot(plan)))

            with patch.object(inv, "build_plan", return_value=plan), patch.object(inv, "REPO", Path(temp)):
                first = inv.operator_once(assignment, marker, transport)
                self.assertEqual(first["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
                self.assertEqual(calls, [plan["audit_argv"], plan["stage_argv"], plan["audit_argv"]])
                with self.assertRaises(RuntimeError):
                    inv.operator_once(assignment, Path(temp) / "alternate.jsonl", transport)
                with self.assertRaises(FileExistsError):
                    inv.operator_once(assignment, marker, transport)
            self.assertEqual(len(calls), 3)
            self.assertFalse((Path(temp) / "alternate.jsonl").exists())
            self.assertTrue(marker.exists())

    def test_marker_file_and_parent_fsync_before_first_transport(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, assignment, marker, _ = fixture(self.plan, temp)
            real_fsync = os.fsync
            fsynced = set()
            calls = []

            def checked_fsync(fd):
                info = os.fstat(fd)
                if stat.S_ISREG(info.st_mode):
                    fsynced.add("marker")
                elif stat.S_ISDIR(info.st_mode):
                    fsynced.add("parent")
                return real_fsync(fd)

            def transport(argv, *, input_text, timeout_seconds):
                self.assertTrue(marker.exists())
                self.assertEqual(fsynced, {"marker", "parent"})
                self.assertEqual(json.loads(marker.read_text().splitlines()[0])["event"],
                                 "EXACT_CHECKOUT_ASSIGNMENT_BOUND")
                calls.append(argv)
                return response(stdout=json.dumps(snapshot(plan)))

            with patch.object(inv, "build_plan", return_value=plan), patch.object(inv, "REPO", Path(temp)), patch.object(inv.os, "fsync", side_effect=checked_fsync):
                outcome = inv.operator_once(assignment, marker, transport)
            self.assertEqual(outcome["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
            self.assertEqual(calls, [plan["audit_argv"], plan["stage_argv"], plan["audit_argv"]])

    def test_nonprivate_or_symlink_marker_parent_fails_before_host(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, assignment, marker, _ = fixture(self.plan, temp)
            calls = []
            Path(temp).chmod(0o750)
            try:
                with patch.object(inv, "build_plan", return_value=plan), patch.object(inv, "REPO", Path(temp)):
                    with self.assertRaises(RuntimeError):
                        inv.operator_once(assignment, marker, lambda *a, **k: calls.append(a))
                self.assertFalse(marker.exists())
            finally:
                Path(temp).chmod(0o700)
            self.assertEqual(calls, [])
        with tempfile.TemporaryDirectory() as temp:
            real = Path(temp) / "real"
            real.mkdir(mode=0o700)
            link = Path(temp) / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                inv.Journal(link / "marker.jsonl")
            self.assertFalse((real / "marker.jsonl").exists())

    def test_fsync_failure_consumes_marker_without_host_contact(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, assignment, marker, _ = fixture(self.plan, temp)
            calls = []
            with patch.object(inv, "build_plan", return_value=plan), patch.object(inv, "REPO", Path(temp)), patch.object(inv.os, "fsync", side_effect=OSError("synthetic fsync failure")):
                with self.assertRaises(OSError):
                    inv.operator_once(assignment, marker, lambda *a, **k: calls.append(a))
            self.assertEqual(calls, [])
            self.assertTrue(marker.exists(), "failed marker creation must not permit replay")

    def test_build_plan_rejects_changed_request_target_or_argv_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            request = json.loads((PACKAGE / "CHECKOUT_REQUEST.HOLD.json").read_text())
            request["checkout_path"] = "/home/ubuntu/other-target"
            altered_request = Path(temp) / "changed-request.json"
            altered_request.write_text(json.dumps(request))
            with patch.object(inv, "REQUEST", altered_request):
                with self.assertRaises(RuntimeError):
                    inv.build_plan()
        with tempfile.TemporaryDirectory() as temp:
            binding = json.loads((PACKAGE / "INVOCATION_BINDING.HOLD.json").read_text())
            binding["stage_argv_sha256"] = "0" * 64
            altered_binding = Path(temp) / "changed-binding.json"
            altered_binding.write_text(json.dumps(binding))
            with patch.object(inv, "BINDING", altered_binding):
                with self.assertRaises(RuntimeError):
                    inv.build_plan()

    def test_review_absent_nonhex_wrong_path_wrong_sha_wrong_binding_and_hold(self):
        for case in ("absent", "nonhex", "wrong_path", "wrong_sha", "wrong_binding", "hold"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                plan, assignment, marker, review_file = fixture(self.plan, temp)
                if case == "absent":
                    review_file.unlink()
                elif case == "nonhex":
                    assignment["independent_request_review_sha256"] = "z" * 64
                elif case == "wrong_path":
                    assignment["independent_request_review_path"] = "elsewhere.json"
                elif case == "wrong_sha":
                    assignment["independent_request_review_sha256"] = "0" * 64
                else:
                    review = json.loads(review_file.read_text())
                    if case == "wrong_binding":
                        review["reviewed_binding_sha256"] = "0" * 64
                    else:
                        review["verdict"] = "HOLD"
                    data = inv.canonical_json(review) + b"\n"
                    review_file.write_bytes(data)
                    assignment["independent_request_review_sha256"] = inv.sha256(data)
                calls = []
                with patch.object(inv, "build_plan", return_value=plan), patch.object(inv, "REPO", Path(temp)):
                    with self.assertRaises(RuntimeError):
                        inv.operator_once(assignment, marker, lambda *a, **k: calls.append(a))
                self.assertEqual(calls, [])
                self.assertFalse(marker.exists())

    def test_assignment_source_and_argv_drift_refused_before_marker(self):
        for field in ("source_commit", "request_sha256", "binding_sha256", "stage_argv_sha256", "audit_argv_sha256", "attempt_marker_path"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                plan, assignment, marker, _ = fixture(self.plan, temp)
                assignment[field] = "0" * 64
                calls = []
                with patch.object(inv, "build_plan", return_value=plan), patch.object(inv, "REPO", Path(temp)):
                    with self.assertRaises(RuntimeError):
                        inv.operator_once(assignment, marker, lambda *a, **k: calls.append(a))
                self.assertEqual(calls, [])
                self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
