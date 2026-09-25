"""Independent offline A2 r4 CPU-G2 feasibility/binding witnesses.

No SSH, AWS, Trainium, training, evaluator, or authority file is used here.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "research/v2/authorizations/trn22-c5c-a2-m64d128-cpu-g2-final-pending"
PRIOR_LAUNCH = (REPO / "research/v2/g2-probes/"
                "fs-20260925-g2-a0-c5c-exactenv-trn22-r3/remote-run/"
                "process-fs-20260925-g2-a0-c5c-exactenv-trn22-r3/launch.json")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


local = load("independent_a2_r4_local", PACKAGE / "cpu_g2_invocation_final.py")
remote = load("independent_a2_r4_remote", PACKAGE / "remote_cpu_g2_once.py")
supplied = load("a2_r4_supplied_fixture", PACKAGE / "test_cpu_g2_final.py")


class IndependentA2R4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = local.build_plan()

    def test_pinned_e65_tree_has_science_but_not_planned_cpu_tools(self):
        commit = self.plan["request"]["host"]["source_commit"]
        for name, expected in (("train.py", remote.TRAIN_SHA),
                               ("prepare.py", remote.PREPARE_SHA)):
            result = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=REPO,
                                    capture_output=True, check=True)
            self.assertEqual(hashlib.sha256(result.stdout).hexdigest(), expected)
        result = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit,
                                 "workflow/tools"], cwd=REPO, capture_output=True,
                                text=True, check=True)
        self.assertEqual(result.stdout, "", "pinned checkout commit unexpectedly has tools")
        for name in ("preflight_dryrun.py", "contained.py"):
            self.assertIn("/workflow/tools/" + name,
                          json.dumps(self.plan["commands"]))

    def test_prior_host_cpu_g2_used_a_separate_tool_path(self):
        prior = json.loads(PRIOR_LAUNCH.read_text())
        command = json.dumps(prior["argv"])
        self.assertIn("/dev/shm/trainium-v2-g2-20260925-trn22-r1/"
                      "workflow/tools/preflight_dryrun.py", command)
        self.assertNotIn(local.SOURCE + "/workflow/tools/preflight_dryrun.py", command)

    def test_missing_e65_tools_fail_readonly_audit(self):
        for key in ("g2_tool", "contained_tool"):
            item = supplied.snapshot()
            item[key] = {"exists": False}
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "source.*CPU tool"):
                remote.validate_snapshot(item, self.plan["request"], phase="before")

    def test_current_package_has_no_actual_authority(self):
        request = self.plan["request"]
        self.assertFalse(request["authority"]["cpu_g2_dispatch_authorized"])
        self.assertFalse(request["authority"]["host_contact_authorized"])
        self.assertIsNone(request["capacity"]["fresh_host_values"])
        self.assertFalse(self.plan["dispatch_authorized"])
        self.assertFalse(local.FINAL_REVIEW.exists())
        self.assertFalse(local.ASSIGNMENT.exists())

    def test_transport_exception_still_causes_distinct_postflight_attempt(self):
        events = []
        calls = []

        class Journal:
            def emit(self, event, **fields):
                events.append(event)

        def transport(argv, *, input_data, timeout_seconds):
            calls.append(argv)
            if len(calls) == 1:
                raise TimeoutError("offline injected first-SSH timeout")
            return types.SimpleNamespace(returncode=1, stdout=b"", stderr=b"offline")

        result = local.run_with_postflight(self.plan, Journal(), transport,
                                           {"review_sha256": "x", "assignment_sha256": "y"})
        self.assertEqual(result["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
        self.assertEqual(calls, [self.plan["audit_argv"], self.plan["audit_argv"]])
        self.assertIn("POSTFLIGHT_HOLD", events)

    def test_symlinked_tmp_is_rejected_before_g2_tool(self):
        code = (PACKAGE / "scratch_guard.py").read_text()
        with tempfile.TemporaryDirectory(prefix="a2-r4-independent-") as parent:
            root = Path(parent) / "scratch"
            root.mkdir(mode=0o700)
            target = Path(parent) / "real-tmp"
            target.mkdir(mode=0o700)
            (root / "tmp").symlink_to(target)
            command = [sys.executable, "-B", "-c", code, "guard", str(root),
                       str(root / "tmp"), "1", "1", "1", "1", "--end"]
            result = subprocess.run(command, env={**os.environ,
                                                   "TMPDIR": str(root / "tmp")},
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SCRATCH_HOLD", result.stderr)


if __name__ == "__main__":
    unittest.main()
