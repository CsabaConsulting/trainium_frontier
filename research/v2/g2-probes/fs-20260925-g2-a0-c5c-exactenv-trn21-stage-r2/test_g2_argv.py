"""Offline static/synthetic checks only: never execute CPU G2 or contact a host."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("stage_r2_g2_builder", HERE / "build_g2_argv.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class StageR2G2HoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((HERE / "GATE_CONTRACT.json").read_text())
        cls.plan = BUILDER.build_plan()

    def test_pinned_local_evidence_tools_and_exact_e65_source(self):
        BUILDER.check_local_evidence()
        for filename, expected in (("train.py", self.contract["source_train_sha256"]),
                                   ("prepare.py", self.contract["source_prepare_sha256"])):
            data = subprocess.check_output(
                ["git", "show", self.contract["source_commit"] + ":" + filename],
                cwd=BUILDER.REPO)
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
        science_contract = json.loads((BUILDER.REPO / "research/submission-validation/20260925-v2-c5c-a0-control-trn21-r1/GATE_CONTRACT.json").read_text())
        ratification = json.loads((BUILDER.REPO / "research/v2/authorizations/20260925-trn21-c5c-a0-r1-proposal-ratification.json").read_text())
        self.assertEqual(self.contract["selected_science"]["train_argv"],
                         science_contract["immutable_science"]["train_argv"])
        self.assertEqual(self.contract["selected_science"]["train_argv"],
                         ratification["train_argv"])

    def test_exact_four_rank_command_bindings(self):
        c, p = self.contract, self.plan
        self.assertEqual(BUILDER.argv_digest(p["launch_argv"]), c["exact_launch_argv_sha256"])
        self.assertEqual(BUILDER.argv_digest(p["wait_argv"]), c["exact_wait_argv_sha256"])
        self.assertEqual(BUILDER.argv_digest(p["stop_on_failure_argv"]),
                         c["exact_stop_on_failure_argv_sha256"])
        self.assertEqual(BUILDER.argv_digest(p["quiesce_argv"]), c["exact_quiesce_argv_sha256"])
        self.assertEqual(len(p["launch_argv"]), 64)
        self.assertEqual(p["launch_argv"][0:3],
                         [BUILDER.PYTHON, c["persistent_cpu_tools"]["contained_path"], "launch"])
        self.assertEqual(p["launch_argv"].count("--"), 2)
        self.assertEqual(p["launch_argv"][-len(BUILDER.TRAIN_ARGV):], BUILDER.TRAIN_ARGV)
        self.assertEqual(p["launch_argv"].count("--nproc"), 1)
        self.assertEqual(p["launch_argv"][p["launch_argv"].index("--nproc") + 1], "4")
        self.assertNotIn("--real-data", p["launch_argv"])
        self.assertNotIn("--stub-prepare", p["launch_argv"])

    def test_new_ext4_isolation_and_unconsumed_science(self):
        c, argv = self.contract, self.plan["launch_argv"]
        g2 = c["cpu_g2_isolation"]
        self.assertTrue(g2["root_must_be_absent"].startswith("/home/ubuntu/"))
        self.assertNotIn("/dev/shm/", " ".join(argv))
        self.assertIn("--cwd", argv)
        self.assertEqual(argv[argv.index("--cwd") + 1], c["source_checkout"])
        self.assertIn("--train-py", argv)
        self.assertEqual(argv[argv.index("--train-py") + 1],
                         c["source_checkout"] + "/train.py")
        self.assertIn("TMPDIR=" + g2["root_must_be_absent"], argv)
        self.assertIn("NEURON_COMPETITION_R1_STEP_LOG=" +
                      g2["step_log_must_be_absent_before_launch"], argv)
        self.assertIn("NEURON_COMPETITION_R1_OUT_DIR=" + c["source_checkout"] +
                      "/out/" + BUILDER.SCIENCE_LABEL, argv)
        self.assertNotIn("/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r1/train.py", argv)
        self.assertNotEqual(g2["new_cpu_label"], c["selected_science"]["science_run_label_unconsumed"])

    def test_all_authority_and_checkout_result_gates_stay_hold(self):
        c = self.contract
        for key in ("cpu_g2_dispatch_authorized", "native_training_authorized",
                    "evaluation_authorized", "submission_authorized",
                    "source_checkout_clean_shallow_one_commit_and_independently_reviewed"):
            self.assertIs(c[key], False)
        self.assertTrue(all(value is None for value in c["missing_gates"].values()))
        self.assertFalse(c["source_stage_lineage"]["disabled_checkout_request_draft_accepted_for_dispatch"])
        self.assertIsNone(c["source_stage_lineage"]["independent_checkout_result_review_sha256"])

    def test_mutated_science_source_tool_isolation_or_authority_rejected(self):
        mutations = [
            lambda c: c.__setitem__("cpu_g2_dispatch_authorized", True),
            lambda c: c.__setitem__("source_checkout", "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r1"),
            lambda c: c["selected_science"]["train_argv"].append("--real-data"),
            lambda c: c["selected_science"]["four_effective_environment_values"].__setitem__("NEURON_COMPETITION_R1_SEED", "43"),
            lambda c: c["persistent_cpu_tools"].__setitem__("preflight_sha256", "0" * 64),
            lambda c: c["cpu_g2_isolation"].__setitem__("root_must_be_absent", "/dev/shm/stale"),
            lambda c: c["cpu_g2_isolation"].__setitem__("new_cpu_label", "fs-20260925-g2-a0-c5c-exactenv-trn21-r1"),
            lambda c: c["missing_gates"].__setitem__("independent_checkout_result_review", "fake-pass"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(self.contract)
                mutate(changed)
                with self.assertRaises(RuntimeError):
                    BUILDER.build_plan(changed)

    def test_direct_execution_refuses_dispatch(self):
        result = subprocess.run([sys.executable, str(HERE / "build_g2_argv.py")],
                                capture_output=True, text=True, check=False, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DISPATCH HOLD", result.stderr)


if __name__ == "__main__":
    unittest.main()
