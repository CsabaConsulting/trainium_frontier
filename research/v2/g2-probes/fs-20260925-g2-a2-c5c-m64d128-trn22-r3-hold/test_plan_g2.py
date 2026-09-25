"""Local-only adversarial checks for the disabled A2 CPU G2 plan."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE = Path(__file__).with_name("plan_g2.py")
SPEC = importlib.util.spec_from_file_location("a2_disabled_plan_g2", MODULE)
plan_g2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan_g2)
GUARD_SPEC = importlib.util.spec_from_file_location(
    "a2_disabled_scratch_guard", Path(__file__).with_name("scratch_guard.py"))
scratch_guard = importlib.util.module_from_spec(GUARD_SPEC)
GUARD_SPEC.loader.exec_module(scratch_guard)


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
            "plan_sha256": "a" * 64, "plan_summary": {
                "elements_by_owner": [128, 256, 384, 512],
                "estimated_ms_by_owner": [1.0, 2.0, 3.0, 4.0],
                "shard_elements": 512, "total_padded_bytes": 8192},
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
        self.assertEqual(self.plan["scratch_projection_binding"], "UNBOUND_PROJECTION")
        self.assertIn("prepare_scratch_after_marker", self.plan["commands"])
        self.assertIn("scratch_guard", " ".join(launch))

    def test_exact_receipt_is_structural_only_not_numeric_pass(self):
        verdict = plan_g2.validate_g2_receipt(receipt(), plan=self.plan)
        self.assertEqual(verdict["structural_checks"], "PASS")
        self.assertEqual(verdict["numeric_gate"], "HOLD")
        self.assertEqual(verdict["per_table_attribution"],
                         "HOLD_INDEPENDENT_RAW_PROOF_REQUIRED")
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

    def test_incoherent_summary_and_impossible_rows_rejected(self):
        candidate = receipt()
        candidate["ranks"][3]["plan_summary"] = {"inconsistent_with_common_digest": True}
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)
        candidate = receipt()
        candidate["ranks"][2]["rmsprop_trace"]["checks"][0]["rows_touched"] = 524289
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)

    def test_unauthenticated_table_name_cannot_claim_per_table_proof(self):
        candidate = receipt()
        for rank in candidate["ranks"]:
            for check in rank["rmsprop_trace"]["checks"]:
                check["table_name"] = plan_g2.KEYS[0]
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)

    def test_request_pin_mutations_are_fail_closed(self):
        original_request = json.loads(plan_g2.REQUEST.read_text())
        original_read = Path.read_text
        for section, key, bad in (
            ("capacity_and_preflight", "minimum_fresh_root_free_inodes", 0),
            ("host", "git_pointer_sha256", "0" * 64),
            ("host", "git_pointer_mode", "0777"),
            ("host", "linked_gitdir_mode", "0777"),
            ("scratch_contract", "guard_sha256", "0" * 64),
        ):
            forged = copy.deepcopy(original_request)
            forged[section][key] = bad

            def changed(path, *args, **kwargs):
                if path == plan_g2.REQUEST:
                    return json.dumps(forged)
                return original_read(path, *args, **kwargs)

            with self.subTest(section=section, key=key), patch.object(Path, "read_text", changed):
                with self.assertRaises(ValueError):
                    plan_g2.build_plan()

    def test_exact_scratch_creation_and_tempfile_root(self):
        code = Path(__file__).with_name("scratch_guard.py").read_text()
        with tempfile.TemporaryDirectory(prefix="a2-r3-guard-") as parent:
            root, tmp = str(Path(parent) / "new-run"), str(Path(parent) / "new-run" / "tmp")
            env = dict(os.environ, TMPDIR=tmp)
            base = [sys.executable, "-B", "-c", code]
            args = [root, tmp, "1", "1", "1", "1", "--end"]
            prepared = subprocess.run(base + ["prepare", *args], env=env,
                                      capture_output=True, text=True, check=True)
            info = json.loads(prepared.stdout)
            self.assertEqual(info["tempfile_root"], tmp)
            self.assertEqual(info["mode"], "0700")
            guarded = subprocess.run(base + ["guard", *args], env=env,
                                     capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(guarded.stdout)["tempfile_root"], tmp)
            repeated = subprocess.run(base + ["prepare", *args], env=env,
                                      capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)

    def test_missing_or_symlinked_tmp_never_falls_back(self):
        code = Path(__file__).with_name("scratch_guard.py").read_text()
        with tempfile.TemporaryDirectory(prefix="a2-r3-missing-") as parent:
            root, tmp = str(Path(parent) / "new-run"), str(Path(parent) / "new-run" / "tmp")
            Path(root).mkdir(mode=0o700)
            env = dict(os.environ, TMPDIR=tmp)
            args = [sys.executable, "-B", "-c", code, "guard", root, tmp,
                    "1", "1", "1", "1", "--end"]
            missing = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("SCRATCH_HOLD", missing.stderr)
            Path(tmp).symlink_to(parent)
            symlink = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertNotEqual(symlink.returncode, 0)
            self.assertIn("SCRATCH_HOLD", symlink.stderr)

    def test_contained_process_directory_is_expected_only_at_launch(self):
        with tempfile.TemporaryDirectory(prefix="a2-r3-process-") as parent:
            root, tmp = str(Path(parent) / "new-run"), str(Path(parent) / "new-run" / "tmp")
            Path(root).mkdir(mode=0o700)
            Path(tmp).mkdir(mode=0o700)
            (Path(root) / "process").mkdir(mode=0o755)
            previous = tempfile.tempdir
            try:
                with patch.dict(os.environ, {"TMPDIR": tmp}):
                    result = scratch_guard._inspect(
                        root, tmp, create=False, allow_process=True,
                        min_free=1, min_inodes=1, projection=1, floor=1)
                    self.assertEqual(result["tempfile_root"], tmp)
                    with self.assertRaises(ValueError):
                        scratch_guard._inspect(
                            root, tmp, create=False, allow_process=False,
                            min_free=1, min_inodes=1, projection=1, floor=1)
            finally:
                tempfile.tempdir = previous

    def test_unbound_projection_draft_cannot_create_scratch(self):
        code = Path(__file__).with_name("scratch_guard.py").read_text()
        with tempfile.TemporaryDirectory(prefix="a2-r3-unbound-") as parent:
            root, tmp = str(Path(parent) / "new-run"), str(Path(parent) / "new-run" / "tmp")
            env = dict(os.environ, TMPDIR=tmp)
            done = subprocess.run([sys.executable, "-B", "-c", code,
                                   "prepare", root, tmp, "1", "1",
                                   "UNBOUND_PROJECTION", "1", "--end"],
                                  env=env, capture_output=True, text=True)
            self.assertNotEqual(done.returncode, 0)
            self.assertFalse(Path(root).exists())
        candidate = receipt()
        candidate["appended_argv"][1] = "neuron"
        with self.assertRaises(ValueError):
            plan_g2.validate_g2_receipt(candidate, plan=self.plan)


if __name__ == "__main__":
    unittest.main()
