"""Offline adversarial tests for the disabled post-checkout CPU-G2 request."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("trn21_cpu_g2_request", HERE / "build_cpu_g2_request.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class ExactCPURequestHoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = json.loads(BUILDER.REQUEST.read_text())
        cls.binding = json.loads(BUILDER.BINDING.read_text())
        root = BUILDER.REPO
        result = cls.request["accepted_checkout_result"]
        cls.review = json.loads((root / result["independent_result_review_path"]).read_text())
        cls.stage = json.loads((root / result["raw_checkout_stage_path"]).read_text())
        cls.post = json.loads((root / result["raw_unconditional_postflight_path"]).read_text())

    def _reject(self, **overrides):
        with self.assertRaises((RuntimeError, ValueError, TypeError, KeyError)):
            BUILDER.build_plan(**overrides)

    def test_default_plan_binds_reviewed_checkout_but_is_not_authorized(self):
        plan = BUILDER.build_plan()
        self.assertEqual(len(plan["launch_argv"]), 64)
        self.assertFalse(plan["request"]["cpu_g2_dispatch_authorized"])
        self.assertFalse(plan["binding"]["dispatch_authorized"])
        self.assertEqual(plan["request"]["accepted_checkout_result"]["independent_result_review_sha256"],
                         "b3c82bd594ae312142925cc3888f1de532473854f246a17b4ac4251baf37b89c")
        hard_module, _ = BUILDER._hardened_plan()
        digest = hard_module._frozen_builder().argv_digest
        for name, argv in (("exact_launch_argv_sha256", "launch_argv"),
                           ("exact_wait_argv_sha256", "wait_argv"),
                           ("exact_stop_on_failure_argv_sha256", "stop_on_failure_argv"),
                           ("exact_quiesce_argv_sha256", "quiesce_argv")):
            self.assertEqual(plan["request"][name], digest(plan[argv]))

    def test_result_or_hardened_review_substitution_rejected(self):
        for section, field in (("accepted_checkout_result", "operator_terminal_sha256"),
                               ("accepted_checkout_result", "independent_result_review_sha256"),
                               ("accepted_checkout_result", "raw_checkout_stage_sha256"),
                               ("hardened_offline_g2", "independent_static_review_sha256")):
            with self.subTest(section=section, field=field):
                request = copy.deepcopy(self.request)
                request[section][field] = "0" * 64
                self._reject(request=request)

    def test_accepted_review_and_raw_stage_must_match_source(self):
        review = copy.deepcopy(self.review)
        review["verdict"] = "HOLD"
        self._reject(review=review)
        review = copy.deepcopy(self.review)
        review["evidence_sha256"]["operator_terminal_summary"] = "0" * 64
        self._reject(review=review)
        stage = copy.deepcopy(self.stage)
        stage["checkout"]["git_checks"]["head"]["stdout"] = "0" * 40 + "\n"
        self._reject(stage=stage)
        stage = copy.deepcopy(self.stage)
        stage["checkout"]["git_checks"]["status"]["stdout"] = "?? unknown"
        self._reject(stage=stage)

    def test_checkout_and_cpu_tool_identity_must_match_postflight(self):
        post = copy.deepcopy(self.post)
        post["targets"][self.request["source_checkout"]]["inode"] = 0
        self._reject(postflight=post)
        tool = self.request["persistent_cpu_tools"]["preflight_path"]
        post = copy.deepcopy(self.post)
        post["targets"][tool]["mode"] = "0o777"
        self._reject(postflight=post)
        post = copy.deepcopy(self.post)
        post["processes"]["hits"] = [{"pid": 1}]
        self._reject(postflight=post)

    def test_exact_future_cpu_root_requires_fresh_absence_gate(self):
        root = self.request["cpu_g2_isolation"]["root_must_be_absent_at_fresh_preflight"]
        self.assertNotIn(root, self.post["targets"])
        self.assertTrue(self.request["fresh_preflight_required"]["exact_new_g2_root_process_dir_step_log_and_cpu_label_absent"])
        post = copy.deepcopy(self.post)
        post["targets"][root] = {"exists": True}
        self._reject(postflight=post)

    def test_request_rejects_safety_and_authority_mutations(self):
        cases = (
            (None, "cpu_g2_dispatch_authorized", True),
            (None, "native_training_authorized", True),
            ("cpu_g2_isolation", "one_attempt_per_label", False),
            ("cpu_g2_isolation", "root_must_be_absent_at_fresh_preflight", "/dev/shm/fallback"),
            ("cpu_g2_isolation", "root_parent_filesystem", "tmpfs"),
            ("persistent_cpu_tools", "required_mode", "0777"),
            ("persistent_cpu_tools", "preflight_inode", 0),
            ("persistent_cpu_tools", "contained_inode", True),
            ("fresh_preflight_required", "exact_new_g2_root_process_dir_step_log_and_cpu_label_absent", False),
        )
        for section, field, value in cases:
            with self.subTest(section=section, field=field):
                request = copy.deepcopy(self.request)
                target = request if section is None else request[section]
                target[field] = value
                self._reject(request=request)
        request = copy.deepcopy(self.request)
        del request["fresh_preflight_required"]["exact_new_g2_root_process_dir_step_log_and_cpu_label_absent"]
        self._reject(request=request)

    def test_invocation_binding_cannot_enable_or_relabel_request(self):
        for field, value in (("dispatch_authorized", True),
                             ("one_attempt_only", False),
                             ("request_sha256", "0" * 64),
                             ("offline_builder_sha256", "0" * 64),
                             ("independent_static_review_sha256", "0" * 64),
                             ("exact_launch_argv_sha256", "0" * 64)):
            with self.subTest(field=field):
                binding = copy.deepcopy(self.binding)
                binding[field] = value
                self._reject(binding=binding)

    def test_direct_execution_is_dispatch_hold(self):
        result = subprocess.run([sys.executable, str(HERE / "build_cpu_g2_request.py")],
                                capture_output=True, text=True, check=False, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DISPATCH HOLD", result.stderr)


if __name__ == "__main__":
    unittest.main()
