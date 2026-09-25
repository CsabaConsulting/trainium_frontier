"""Independent, offline characterization of the disabled trn22 A0 r5 package.

The optimized-launcher test intentionally passes when the frozen defect is
reproduced. It never invokes SSH or trains a model.
"""

import argparse
import ast
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "research/v2/authorizations/trn22-c5c-a0-r5-native-hold"
SOURCE = "446b657695f086f9b484fa601c9a3b61c2495912"


def frozen_source(path):
    return subprocess.check_output(
        ["git", "show", f"{SOURCE}:{path}"], cwd=ROOT, text=True
    )


def frozen_function(path, name):
    tree = ast.parse(frozen_source(path))
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class IndependentR5NegativeTests(unittest.TestCase):
    def test_optimized_launcher_reaches_fake_host_contact(self):
        request_path = PACKAGE / "NATIVE_REQUEST.DISABLED.json"
        request = json.loads(request_path.read_text())
        self.assertIs(request["dispatch_authorized"], False)
        main = frozen_function("research/fresh_start_launch.py", "main")
        fake_host = types.ModuleType("fresh_start_host")
        fake_host.MIN_OPERATIONAL_FREE_BYTES = 6979321856

        with tempfile.TemporaryDirectory(prefix="trn22-r5-launcher-negative-") as tmp:
            receipt_dir = Path(tmp) / "research/run-logs"
            receipt_dir.mkdir(parents=True)
            attempted = []

            def block_contact(*args, **kwargs):
                attempted.append(args)
                raise RuntimeError("FAKE_HOST_CONTACT_REACHED")

            common = dict(
                argparse=argparse, Path=Path, json=json, re=re, shlex=shlex,
                subprocess=types.SimpleNamespace(run=block_contact),
                ROOT=Path(tmp), HOSTS={"trn22": "dummy"}, SSH=["ssh"],
                REMOTE_SCRIPT="pass", datetime=datetime, timezone=timezone,
            )
            module = ast.Module(body=[main], type_ignores=[])
            argv = ["fresh_start_launch.py", str(request_path)]
            with patch.dict(sys.modules, {"fresh_start_host": fake_host}), patch.object(
                sys, "argv", argv
            ):
                normal = dict(common)
                exec(compile(module, "frozen-launcher-main", "exec", optimize=0), normal)
                with self.assertRaisesRegex(AssertionError, "one-label dispatch"):
                    normal["main"]()
                self.assertEqual(attempted, [])
                self.assertEqual(list(receipt_dir.iterdir()), [])

                optimized = dict(common)
                exec(compile(module, "frozen-launcher-main", "exec", optimize=1), optimized)
                with self.assertRaisesRegex(RuntimeError, "FAKE_HOST_CONTACT_REACHED"):
                    optimized["main"]()

            self.assertEqual(len(attempted), 1)
            receipt = receipt_dir / (request["label"] + ".dispatch.json")
            self.assertTrue(receipt.is_file())

    def test_step_log_parent_is_in_capsule_but_missing_parent_fails(self):
        manifest = ROOT / (
            "research/v2/authorizations/trn22-c5c-a0-r4-source-stage/MEMBERS.jsonl"
        )
        entries = [json.loads(line) for line in manifest.read_text().splitlines()]
        parent = next(item for item in entries if item["path"] == "research/run-logs")
        self.assertEqual((parent["type"], parent["mode"]), ("directory", "0o700"))
        children = sum(
            item["type"] == "file" and item["path"].startswith("research/run-logs/")
            for item in entries
        )
        self.assertEqual(children, 5560)
        tree = subprocess.check_output(
            ["git", "ls-tree", "-d", SOURCE, "research/run-logs"],
            cwd=ROOT, text=True,
        )
        self.assertTrue(tree.startswith("040000 tree "))

        step = frozen_function("prepare.py", "record_step_boundary")
        with tempfile.TemporaryDirectory(prefix="trn22-r5-step-parent-negative-") as tmp:
            destination = Path(tmp) / "missing-parent/r5.steps.jsonl"
            namespace = {"os": os, "STEP_LOG_PATH": str(destination)}
            exec(compile(ast.Module(body=[step], type_ignores=[]),
                         "frozen-prepare-step-boundary", "exec"), namespace)
            with patch.dict(os.environ, {"RANK": "0"}):
                with self.assertRaises(FileNotFoundError):
                    namespace["record_step_boundary"](0)
                destination.parent.mkdir()
                namespace["record_step_boundary"](0)
            self.assertEqual(destination.read_text(), "0\n")

    def test_supplied_suite_rejects_seven_tampered_variants(self):
        spec = importlib.util.spec_from_file_location(
            "r5_package_tests", PACKAGE / "test_disabled_package.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite = module.DisabledPackageTests
        suite.setUpClass()
        cases = [
            ("request", "test_no_host_or_dispatch_or_evaluation_authority",
             lambda value: value.__setitem__("dispatch_authorized", True)),
            ("base", "test_draft_hash_chain_and_runbook_pins",
             lambda value: value.__setitem__("request_sha256", "0" * 64)),
            ("pair", "test_draft_hash_chain_and_runbook_pins",
             lambda value: value.__setitem__("base_binding_sha256", "0" * 64)),
            ("eval", "test_no_host_or_dispatch_or_evaluation_authority",
             lambda value: value.__setitem__("evaluation_authorized", True)),
            ("contract", "test_source_bytes_are_exact_frozen_blobs",
             lambda value: value["source_sha256"].__setitem__("train.py", "0" * 64)),
            ("request", "test_r5_uses_r4_checkout_with_new_science_and_monitor_paths",
             lambda value: value.__setitem__("label", "fs-wrong-label")),
            ("eval", "test_postrun_public20m_plan_matches_calibrated_g1",
             lambda value: value.__setitem__("eval_tokens", 2097152)),
        ]
        for field, method, mutate in cases:
            with self.subTest(field=field, method=method):
                altered = copy.deepcopy(getattr(suite, field))
                mutate(altered)
                with patch.object(suite, field, altered):
                    with self.assertRaises(AssertionError):
                        getattr(suite(method), method)()


if __name__ == "__main__":
    unittest.main()
