"""Local-only adversarial checks for the disabled A2 CPU G2 plan."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


MODULE = Path(__file__).with_name("plan_g2.py")
SPEC = importlib.util.spec_from_file_location("a2_disabled_plan_g2", MODULE)
plan_g2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan_g2)


def receipt() -> dict:
    appended = ["--device-type", "cpu", "--no-compile", "--num-steps", "2", "--out-dir",
                plan_g2.SCRATCH + "/tmp/g2-unit/out"]
    checks = [
        {"step_t": step, "numel": 67108864, "rows_touched": 7,
         "max_abs_err_over_lr": 0.01, "untouched_rows_unchanged": True,
         "grad_abs_median_touched": 0.0001}
        for step in (1, 2) for _ in range(4)
    ]
    ranks = []
    for number in range(4):
        ranks.append({
            "rank": number, "status": "PASS", "appended_argv": appended,
            "plan_sha256": "a" * 64, "plan_summary": {"elements_by_owner": [1, 2, 3, 4]},
            "losses": [9.0, 8.9], "tables": list(plan_g2.KEYS),
            "unchanged_tables": [], "cross_rank_mismatches": [], "n_changed": 70,
            "rmsprop_trace": {"calls": 2, "optimizer_steps": 2, "checks": copy.deepcopy(checks)},
        })
    ranks[0]["checkpoint_check"] = {
        "ok": True, "path_exists": True, "deterministic": True,
        "logits_shape": [1, 256, 8192],
    }
    ranks[0]["final_digests"] = {key: "a1b2c3d4e5f60000" for key in plan_g2.KEYS}
    return {
        "schema": "g2-preflight-v1", "verdict": "PASS",
        "train_py": plan_g2.CHECKOUT + "/train.py", "train_py_sha256": plan_g2.TRAIN_SHA,
        "argv": plan_g2.build_plan()["train_argv"],
        "prepare": {"dir": plan_g2.CHECKOUT, "sha256": plan_g2.PREPARE_SHA, "stub": False},
        "steps": 2, "nproc": 4, "micro_batch": [2, 256],
        "host": "ip-172-31-32-253", "torchrun_rc": 0, "timed_out": False,
        "appended_argv": appended, "plan_sha256": "a" * 64, "ranks": ranks,
    }


class DisabledPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = plan_g2.build_plan()

    def test_plan_is_disabled_and_exact(self):
        self.assertFalse(self.plan["dispatch_authorized"])
        self.assertEqual(self.plan["label"], plan_g2.LABEL)
        self.assertEqual(self.plan["train_argv"][0:4], ["--n-embd", "1024", "--ngram-ve", "--ngram-ve-layers"])
        launch = self.plan["commands"]["launch"]
        self.assertIn("--cgroup", launch)
        self.assertEqual(launch.count("--trace-rmsprop"), 1)
        self.assertEqual(launch.count("--keep"), 1)
        self.assertIn("TRN_EXP_LABEL=" + plan_g2.LABEL, launch)
        self.assertEqual(launch[launch.index("/usr/bin/env") + 1], "-i")
        self.assertNotIn("--real-data", launch)
        self.assertNotIn("--stub-prepare", launch)

    def test_exact_numeric_receipt_shape_accepts(self):
        plan_g2.validate_g2_receipt(receipt(), plan=self.plan)
        plan_g2.validate_step_log(b"0\n1\n")

    def test_m8_or_reordered_flags_rejected(self):
        candidate = receipt()
        candidate["argv"][candidate["argv"].index("64")] = "8"
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)

    def test_stub_prepare_rejected(self):
        candidate = receipt()
        candidate["prepare"]["stub"] = True
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)

    def test_missing_rank_and_trace_rejected(self):
        candidate = receipt()
        candidate["ranks"].pop()
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)
        candidate = receipt()
        candidate["ranks"][2]["rmsprop_trace"]["checks"].pop()
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)

    def test_changed_or_nonfinite_table_update_rejected(self):
        candidate = receipt()
        candidate["ranks"][1]["unchanged_tables"] = [plan_g2.KEYS[0]]
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)
        candidate = receipt()
        candidate["ranks"][3]["rmsprop_trace"]["checks"][0]["max_abs_err_over_lr"] = float("nan")
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)

    def test_checkpoint_and_boundary_failure_rejected(self):
        candidate = receipt()
        candidate["ranks"][0]["checkpoint_check"]["ok"] = False
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)
        for raw in (b"0\n", b"1\n0\n", b"0\n1\n2\n"):
            with self.assertRaises(ValueError):
                plan_g2.validate_step_log(raw)

    def test_forged_cpu_or_host_identity_rejected(self):
        candidate = receipt()
        candidate["host"] = "other-host"
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)
        candidate = receipt()
        candidate["appended_argv"][1] = "neuron"
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)


if __name__ == "__main__":
    unittest.main()
