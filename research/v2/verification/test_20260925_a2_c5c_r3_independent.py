"""Offline adversarial review of the disabled A2 CPU-G2 r3 repair.

These tests import only local source and use temporary scratch. They neither
contact trn22 nor execute model training, evaluation, or the G2 child.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PACKAGE = (
    Path(__file__).resolve().parents[1]
    / "g2-probes/fs-20260925-g2-a2-c5c-m64d128-trn22-r3-hold"
)


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = load("independent_a2_r3_plan", "plan_g2.py")
fixture = load("independent_a2_r3_fixture", "test_plan_g2.py")
guard = load("independent_a2_r3_guard", "scratch_guard.py")


class IndependentR3Tests(unittest.TestCase):
    def test_disabled_plan_has_exact_unbound_scratch_commands(self):
        built = plan.build_plan()
        self.assertFalse(built["dispatch_authorized"])
        self.assertEqual(len(built["train_argv"]), 27)
        self.assertEqual(built["scratch_projection_binding"], "UNBOUND_PROJECTION")
        for name in ("prepare_scratch_after_marker", "launch"):
            self.assertIn("UNBOUND_PROJECTION", built["commands"][name])
        self.assertEqual(built["scratch_guard_sha256"], plan.SCRATCH_GUARD_SHA)

    def test_prior_request_pin_bypasses_are_closed(self):
        original = json.loads(plan.REQUEST.read_text())
        reader = Path.read_text
        for section, key, value in (
            ("capacity_and_preflight", "minimum_fresh_root_free_inodes", 0),
            ("host", "git_pointer_sha256", "0" * 64),
            ("host", "git_pointer_mode", "0777"),
            ("host", "linked_gitdir_mode", "0777"),
            ("scratch_contract", "guard_sha256", "0" * 64),
        ):
            forged = copy.deepcopy(original)
            forged[section][key] = value

            def fake_read(path, *args, **kwargs):
                if path == plan.REQUEST:
                    return json.dumps(forged)
                return reader(path, *args, **kwargs)

            with self.subTest(section=section, key=key), patch.object(Path, "read_text", fake_read):
                with self.assertRaises(ValueError):
                    plan.build_plan()

    def test_prior_receipt_bypasses_are_closed_without_numeric_promotion(self):
        baseline = fixture.receipt()
        structural = plan.validate_g2_receipt(baseline, plan=plan.build_plan())
        self.assertEqual(structural["numeric_gate"], "HOLD")
        self.assertEqual(structural["per_table_attribution"],
                         "HOLD_INDEPENDENT_RAW_PROOF_REQUIRED")
        for mutator in (
            lambda r: r["ranks"][3].update(plan_summary={"forged": True}),
            lambda r: r["ranks"][2]["rmsprop_trace"]["checks"][0].update(rows_touched=524289),
            lambda r: r["ranks"][2]["rmsprop_trace"]["checks"][0].update(rows_touched=0),
            lambda r: [check.update(table_name=plan.KEYS[0])
                       for rank in r["ranks"] for check in rank["rmsprop_trace"]["checks"]],
        ):
            candidate = copy.deepcopy(baseline)
            mutator(candidate)
            with self.assertRaises(ValueError):
                plan.validate_g2_receipt(candidate, plan=plan.build_plan())

    def test_scratch_refuses_missing_tmp_and_symlink_before_fallback(self):
        with tempfile.TemporaryDirectory(prefix="a2-r3-review-", dir="/dev/shm") as parent:
            root = Path(parent) / "new"
            tmp = root / "tmp"
            root.mkdir(mode=0o700)
            env = dict(os.environ, TMPDIR=str(tmp))
            command = [sys.executable, "-B", "-c", (PACKAGE / "scratch_guard.py").read_text(),
                       "guard", str(root), str(tmp), "1", "1", "1", "1", "--end"]
            missing = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("SCRATCH_HOLD", missing.stderr)
            tmp.symlink_to(parent)
            linked = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(linked.returncode, 2)
            self.assertIn("SCRATCH_HOLD", linked.stderr)

    def test_scratch_prepare_is_exclusive_and_unbound_projection_no_write(self):
        with tempfile.TemporaryDirectory(prefix="a2-r3-review-", dir="/dev/shm") as parent:
            root = Path(parent) / "new"
            tmp = root / "tmp"
            env = dict(os.environ, TMPDIR=str(tmp))
            base = [sys.executable, "-B", "-c", (PACKAGE / "scratch_guard.py").read_text(),
                    "prepare", str(root), str(tmp)]
            unbound = subprocess.run(base + ["1", "1", "UNBOUND_PROJECTION", "1", "--end"],
                                     env=env, capture_output=True, text=True)
            self.assertEqual(unbound.returncode, 2)
            self.assertFalse(root.exists())
            prepared = subprocess.run(base + ["1", "1", "1", "1", "--end"],
                                      env=env, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(prepared.stdout)["tempfile_root"], str(tmp))
            repeated = subprocess.run(base + ["1", "1", "1", "1", "--end"],
                                      env=env, capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 2)


if __name__ == "__main__":
    unittest.main()
