"""Adversarial offline tests only; no CPU G2, host contact or training."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("stage_r2_g2_final_pending", HERE / "build_g2_argv.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class StageR2G2FinalPendingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(BUILDER.CONTRACT.read_text())
        cls.old = json.loads((BUILDER.FROZEN / "GATE_CONTRACT.json").read_text())
        cls.transfer = json.loads((BUILDER.REPO / cls.contract["transfer_lineage"]["operator_terminal_receipt_path"]).read_text())
        cls.checkout = json.loads((BUILDER.REPO / cls.contract["checkout_lineage"]["final_request_path"]).read_text())
        cls.review = json.loads((BUILDER.REPO / cls.contract["checkout_lineage"]["independent_request_review_path"]).read_text())
        cls.assignment = json.loads((BUILDER.REPO / cls.contract["checkout_lineage"]["supervisor_checkout_only_assignment_path"]).read_text())

    def _reject(self, **overrides):
        with self.assertRaises((RuntimeError, KeyError, TypeError, ValueError)):
            BUILDER.build_plan(**overrides)

    def test_baseline_is_exact_and_non_dispatchable(self):
        plan = BUILDER.build_plan()
        c = plan["contract"]
        self.assertFalse(c["cpu_g2_dispatch_authorized"])
        self.assertFalse(c["checkout_lineage"]["source_checkout_result_independently_accepted"])
        self.assertEqual(len(plan["launch_argv"]), 64)
        for digest_field, argv_field in (
            ("exact_launch_argv_sha256", "launch_argv"),
            ("exact_wait_argv_sha256", "wait_argv"),
            ("exact_stop_on_failure_argv_sha256", "stop_on_failure_argv"),
            ("exact_quiesce_argv_sha256", "quiesce_argv"),
        ):
            self.assertEqual(c["frozen_draft"][digest_field],
                             BUILDER._frozen_builder().argv_digest(plan[argv_field]))

    def test_new_contract_rejects_all_three_reported_mutations(self):
        cases = (
            ("one_attempt", "one_attempt_per_label", False),
            ("tool_mode", "persistent_cpu_tool_required_mode", "0777"),
            ("inode_zero", "preflight_inode", 0),
            ("inode_bool", "preflight_inode", True),
            ("inode_wrong", "preflight_inode", 1),
            ("contained_inode_zero", "contained_inode", 0),
        )
        for name, field, value in cases:
            with self.subTest(name=name):
                c = copy.deepcopy(self.contract)
                c["safety_pins"][field] = value
                self._reject(contract=c)

    def test_frozen_draft_rejects_all_three_reported_mutations(self):
        mutations = (
            ("one_attempt", "cpu_g2_isolation", "one_attempt_per_label", False),
            ("tool_mode", "persistent_cpu_tools", "required_mode", "0777"),
            ("inode_zero", "persistent_cpu_tools", "preflight_inode", 0),
            ("inode_wrong", "persistent_cpu_tools", "preflight_inode", 1),
        )
        for name, section, field, value in mutations:
            with self.subTest(name=name):
                old = copy.deepcopy(self.old)
                old[section][field] = value
                self._reject(hold_contract=old)

    def test_inode_and_mode_must_match_independently_retained_transfer_result(self):
        path = self.old["persistent_cpu_tools"]["preflight_path"]
        for name, location, field, value in (
            ("cross_session_inode", ("artifacts", "preflight", "operator_comparator_result"), "inode", 0),
            ("final_inode", ("final_postflight_snapshot", "targets", path), "inode", 0),
            ("final_mode", ("final_postflight_snapshot", "targets", path), "mode", "0o777"),
            ("final_symlink", ("final_postflight_snapshot", "targets", path), "type", "other"),
            ("final_multilink", ("final_postflight_snapshot", "targets", path), "nlink", 2),
        ):
            with self.subTest(name=name):
                transfer = copy.deepcopy(self.transfer)
                target = transfer
                for key in location:
                    target = target[key]
                target[field] = value
                self._reject(transfer_receipt=transfer)

    def test_inode_and_mode_must_match_reviewed_checkout_request(self):
        path = self.old["persistent_cpu_tools"]["preflight_path"]
        for field, value in (("inode", 0), ("mode", "0o777"), ("nlink", 2)):
            with self.subTest(field=field):
                checkout = copy.deepcopy(self.checkout)
                checkout["persistent_targets"][path][field] = value
                self._reject(checkout_request=checkout)

    def test_source_review_assignment_must_match_frozen_checkout(self):
        review = copy.deepcopy(self.review)
        review["verdict"] = "HOLD"
        self._reject(checkout_review=review)
        assignment = copy.deepcopy(self.assignment)
        assignment["cpu_g2_authorized"] = True
        self._reject(checkout_assignment=assignment)
        checkout = copy.deepcopy(self.checkout)
        checkout["source_commit"] = "0" * 40
        self._reject(checkout_request=checkout)

    def test_no_spoofed_checkout_result_or_dispatch(self):
        for section, field, value in (
            (None, "cpu_g2_dispatch_authorized", True),
            ("checkout_lineage", "source_checkout_result_independently_accepted", True),
            ("checkout_lineage", "independent_checkout_result_review_sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                c = copy.deepcopy(self.contract)
                target = c if section is None else c[section]
                target[field] = value
                self._reject(contract=c)

    def test_review_file_digests_cannot_be_relabeled(self):
        for section, field in (
            ("frozen_draft", "contract_sha256"),
            ("transfer_lineage", "operator_terminal_receipt_sha256"),
            ("checkout_lineage", "final_request_sha256"),
            ("checkout_lineage", "independent_request_review_sha256"),
            ("checkout_lineage", "supervisor_checkout_only_assignment_sha256"),
        ):
            with self.subTest(field=field):
                c = copy.deepcopy(self.contract)
                c[section][field] = "0" * 64
                self._reject(contract=c)

    def test_direct_execution_refuses_dispatch(self):
        result = subprocess.run([sys.executable, str(HERE / "build_g2_argv.py")],
                                capture_output=True, text=True, check=False, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DISPATCH HOLD", result.stderr)


if __name__ == "__main__":
    unittest.main()
