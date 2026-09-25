"""Offline reproductions of gaps in the frozen A2 CPU-G2 disabled draft.

These tests intentionally PASS when the reviewed 701ecb65 candidate accepts an
invalid request/receipt. They never contact a host or run training/evaluation.
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
    / "g2-probes/fs-20260925-g2-a2-c5c-m64d128-trn22-r2-hold"
)


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan_g2 = load("reviewed_a2_plan_g2", "plan_g2.py")
fixture = load("reviewed_a2_test_fixture", "test_plan_g2.py")


class IndependentNegatives(unittest.TestCase):
    def test_weakened_inode_and_git_identity_are_accepted(self):
        request = json.loads(plan_g2.REQUEST.read_text())
        weakened = copy.deepcopy(request)
        weakened["capacity_and_preflight"]["minimum_fresh_root_free_inodes"] = 0
        weakened["host"]["git_pointer_sha256"] = "0" * 64
        weakened["host"]["linked_gitdir_mode"] = "0777"
        original = Path.read_text

        def changed(path, *args, **kwargs):
            if path == plan_g2.REQUEST:
                return json.dumps(weakened)
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", changed):
            result = plan_g2.build_plan()  # Unexpectedly accepts all three drifts.
        self.assertFalse(result["dispatch_authorized"])

    def test_incoherent_rank_plan_summary_is_accepted(self):
        result = fixture.receipt()
        result["ranks"][3]["plan_summary"] = {"inconsistent_with_common_digest": True}
        plan_g2.validate_g2_receipt(result, plan=plan_g2.build_plan())

    def test_impossible_rows_touched_is_accepted(self):
        result = fixture.receipt()
        result["ranks"][2]["rmsprop_trace"]["checks"][0]["rows_touched"] = 524289
        plan_g2.validate_g2_receipt(result, plan=plan_g2.build_plan())

    def test_per_table_trace_identity_is_not_verified(self):
        result = fixture.receipt()
        for rank in result["ranks"]:
            for check in rank["rmsprop_trace"]["checks"]:
                check["table_name"] = plan_g2.KEYS[0]
        plan_g2.validate_g2_receipt(result, plan=plan_g2.build_plan())

    def test_missing_tmpdir_silently_falls_back(self):
        with tempfile.TemporaryDirectory(prefix="a2-g2-review-") as root:
            required = Path(root) / "absent-tmp"
            env = dict(os.environ, TMPDIR=str(required))
            done = subprocess.run(
                [sys.executable, "-B", "-c", "import tempfile; print(tempfile.gettempdir())"],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            self.assertFalse(required.exists())
            self.assertNotEqual(done.stdout.strip(), str(required))


if __name__ == "__main__":
    unittest.main()
