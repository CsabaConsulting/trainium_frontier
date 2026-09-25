"""Offline Implementer checks for the disabled trn22 A0 r6 HOLD packet.

No test calls a real host. The frozen-launcher reproduction substitutes a
fake SSH seam and writes only to a disposable local temporary directory.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
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


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parent
SOURCE = "446b657695f086f9b484fa601c9a3b61c2495912"
R5_COMMIT = "9f5459e670f3feb8422222e49d0462ea81c16974"
R5_REVIEW_COMMIT = "919b6f99688d5bd8a23ec35e8cd133faa717b0b6"
LABEL = "fs-20260925-v2-c5c-a0-control-trn22-r6"
R5_LABEL = "fs-20260925-v2-c5c-a0-control-trn22-r5"
CHECKOUT = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r4"
PACKAGE_DIR = "research/v2/authorizations/trn22-c5c-a0-r6-optimization-hold/"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def read_json(path):
    return json.loads(path.read_text())


class DisabledR6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = read_json(PACKAGE / "GATE_CONTRACT.json")
        cls.request = read_json(PACKAGE / "NATIVE_REQUEST.DISABLED.json")
        cls.base = read_json(PACKAGE / "BASE_BINDING.STATIC_ONLY.json")
        cls.pair = read_json(PACKAGE / "PAIR_BINDING.STATIC_ONLY.json")
        cls.eval = read_json(PACKAGE / "PUBLIC20M_EVAL.DISABLED.json")

    def test_exact_byte_chain_and_source_pins(self):
        c, r, b, p = self.contract, self.request, self.base, self.pair
        for key in ("request", "base_binding", "pair_binding",
                    "public20m_eval_plan", "diagnostic_no_dispatch_gate"):
            with self.subTest(artifact=key):
                artifact = c[key]
                self.assertTrue(artifact["path"].startswith(PACKAGE_DIR))
                self.assertEqual(sha256((ROOT / artifact["path"]).read_bytes()),
                                 artifact["sha256"])
                self.assertIn(artifact["sha256"], (PACKAGE / "RUNBOOK.md").read_text())
        self.assertEqual(b["request_sha256"], c["request"]["sha256"])
        self.assertEqual(p["request_sha256"], c["request"]["sha256"])
        self.assertEqual(p["base_binding_sha256"], c["base_binding"]["sha256"])
        self.assertEqual(c["source_sha256"], r["expected_sha256"])
        self.assertEqual(c["source_sha256"], b["expected_sha256"])
        self.assertEqual(set(r["fingerprint_paths"]), set(c["source_sha256"]))
        self.assertEqual(c["frozen_source_commit"], SOURCE)
        self.assertEqual(r["expected_commit"], SOURCE)
        self.assertEqual(c["launcher_safety"]["legacy_frozen_launcher_sha256"],
                         c["source_sha256"]["research/fresh_start_launch.py"])
        for path, digest in c["source_sha256"].items():
            with self.subTest(source=path):
                self.assertEqual(sha256(git_blob(SOURCE, path)), digest)

    def test_disabled_authority_and_required_source_successor(self):
        c, r, b, p, e = self.contract, self.request, self.base, self.pair, self.eval
        self.assertIn("HOLD", c["status"])
        self.assertFalse(c["host_action_authorized"])
        self.assertFalse(c["native_dispatch_authorized"])
        self.assertFalse(c["fresh_public20m_eval_authorized"])
        self.assertIs(r["dispatch_authorized"], False)
        self.assertIs(e["evaluation_authorized"], False)
        self.assertIsNone(e["checkpoint_sha256"])
        self.assertEqual((b["package_state"], p["package_state"]),
                         ("STATIC_ONLY", "STATIC_ONLY"))
        self.assertTrue(all(value is False for value in c["required_gates"].values()))
        self.assertFalse(c["launcher_safety"]["legacy_frozen_launcher_dispatch_eligible"])
        self.assertIsNone(c["launcher_safety"]["successor_launcher_source_commit"])
        self.assertIsNone(c["launcher_safety"]["successor_launcher_source_review_sha256"])
        self.assertTrue(c["launcher_safety"]["final_request_must_rebind_source_and_gate"])
        self.assertFalse(c["required_gates"][
            "r6_nonassert_local_and_remote_launcher_source_implementation_and_independent_review"])
        for key in ("source_transfer_receipt_sha256", "source_checkout_result_review_sha256",
                    "r6_proposal_ratification_sha256", "r6_final_request_review_sha256",
                    "r6_one_shot_native_assignment_sha256", "r6_operator_terminal_receipt_sha256",
                    "r6_public20m_eval_receipt_sha256"):
            self.assertIsNone(c[key])
        self.assertEqual(r["launcher_safety_state"],
                         "HOLD_FROZEN_446B_ASSERT_GATES_BYPASSABLE_UNDER_OPTIMIZATION")
        self.assertFalse(list(PACKAGE.glob("*authorization*.json")))

    def test_unique_label_paths_and_unchanged_control_science(self):
        c, r, b, p, e = self.contract, self.request, self.base, self.pair, self.eval
        self.assertEqual(c["science_label"], LABEL)
        self.assertEqual((r["label"], b["label"], p["label"], e["training_label"]),
                         (LABEL,) * 4)
        self.assertEqual((r["remote"], b["remote"], p["remote"], e["source_checkout"]),
                         (CHECKOUT,) * 4)
        self.assertEqual(c["legacy_r4_checkout_for_static_binding_only"], CHECKOUT)
        self.assertNotIn(R5_LABEL, json.dumps((r, b, p, e)))
        self.assertEqual(c["new_external_monitor_target"],
                         "/home/ubuntu/trainium_a0_disk_monitor_20260925_r6")
        old = json.loads(git_blob(
            R5_COMMIT,
            "research/v2/authorizations/trn22-c5c-a0-r5-native-hold/NATIVE_REQUEST.DISABLED.json"
        ))
        self.assertEqual(r["command"],
                         [arg.replace(R5_LABEL, LABEL) for arg in old["command"]])
        self.assertEqual(r["environment"], old["environment"])
        self.assertEqual(r["warm_cache_policy"], old["warm_cache_policy"])
        self.assertEqual(r["external_boundary_observer"]["command"],
                         [arg.replace(R5_LABEL, LABEL)
                          for arg in old["external_boundary_observer"]["command"]])
        self.assertNotIn("PYTHONOPTIMIZE", r["environment"])
        self.assertFalse(any(arg.startswith("PYTHONOPTIMIZE=") for arg in r["command"]))
        self.assertEqual(r["command"][:7],
                         ["timeout", "--foreground", "--signal=TERM",
                          "--kill-after=60s", "3600s", "env", "-i"])
        self.assertEqual(r["command"][-2:], ["--matrix-lr", "0.015"])
        self.assertEqual((r["environment"]["NEURON_LOGICAL_NC_CONFIG"],
                          r["environment"]["NEURON_COMPETITION_R1_SEED"],
                          r["environment"]["NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS"]),
                         ("2", "42", "1800"))
        self.assertIn("--optlevel=1", r["environment"]["NEURON_CC_FLAGS"])
        self.assertEqual(r["command"][r["command"].index(
            "/home/ubuntu/workspace/native_venv/bin/torchrun"):
            r["command"].index("/home/ubuntu/workspace/native_venv/bin/torchrun") + 3],
            ["/home/ubuntu/workspace/native_venv/bin/torchrun", "--standalone",
             "--nproc_per_node=4"])
        for path in (b["request_path"], b["events_path"], b["ready_path"],
                     p["authorization_path"], p["base_monitor_path"], p["wrapper_path"],
                     p["interlock_path"], p["no_spawn_path"]):
            self.assertTrue(path.startswith(c["new_external_monitor_target"] + "/"), path)

    def test_disk_floor_and_fresh_public20m_plan(self):
        c, r, b, p, e = self.contract, self.request, self.base, self.pair, self.eval
        t = c["safety_thresholds_bytes"]
        self.assertEqual((r["min_free_bytes"], b["floor_free_bytes"],
                          t["immutable_operational_floor"]), (6979321856,) * 3)
        self.assertEqual((r["pre_launch_required_free_bytes"], b["prelaunch_free_bytes"],
                          t["minimum_prelaunch_root_free"]), (10737418240,) * 3)
        self.assertEqual((b["stop_free_bytes"], p["stop_free_bytes"],
                          t["paired_monitor_stop"]), (9126805504,) * 3)
        self.assertLessEqual(b["poll_seconds"], t["monitor_poll_max_seconds"])
        self.assertTrue(r["external_boundary_observer"]["must_start_before_guard"])
        self.assertEqual((e["fresh_process"], e["context_tokens"], e["eval_tokens"],
                          e["expected_num_bytes"]), (True, 1024, 20971520, 83462186))
        self.assertEqual(e["command"][:5],
                         ["timeout", "--foreground", "--signal=TERM",
                          "--kill-after=60s", "1800s"])
        self.assertEqual(e["expected_source_commit"], SOURCE)
        self.assertEqual(e["train_sha256"], c["source_sha256"]["train.py"])
        self.assertEqual(e["prepare_sha256"], c["source_sha256"]["prepare.py"])

    def test_r5_independent_hold_is_pinned_not_authority(self):
        ev = self.contract["upstream_evidence_not_dispatch_authority"]
        self.assertEqual(ev["r5_independent_hold_commit"], R5_REVIEW_COMMIT)
        self.assertEqual(sha256(git_blob(R5_REVIEW_COMMIT,
                                        ev["r5_independent_hold_path"])),
                         ev["r5_independent_hold_sha256"])
        self.assertFalse(ev["r2_ratification_covers_r6_label"])
        self.assertEqual(ev["r4_capsule_sha256"],
                         "767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a")

    def test_no_dispatch_gate_refuses_normal_optimized_and_environment_modes(self):
        gate = PACKAGE / "no_dispatch_gate.py"
        request = PACKAGE / "NATIVE_REQUEST.DISABLED.json"
        clean = dict(os.environ)
        clean.pop("PYTHONOPTIMIZE", None)
        clean["PYTHONDONTWRITEBYTECODE"] = "1"
        cases = (
            ("normal", [], None, "R6_FROZEN_446B_LAUNCHER_UNSAFE_HOLD"),
            ("dash_O", ["-O"], None, "R6_OPTIMIZED_PYTHON_REFUSED"),
            ("dash_OO", ["-OO"], None, "R6_OPTIMIZED_PYTHON_REFUSED"),
            ("env_1", [], "1", "R6_OPTIMIZED_PYTHON_REFUSED"),
            ("env_0", [], "0", "R6_PYTHONOPTIMIZE_ENV_REFUSED"),
            ("env_empty", [], "", "R6_PYTHONOPTIMIZE_ENV_REFUSED"),
        )
        for stage in ("local", "remote"):
            for name, flags, setting, marker in cases:
                with self.subTest(stage=stage, mode=name), tempfile.TemporaryDirectory(
                    prefix="trn22-r6-gate-offline-"
                ) as tmp:
                    env = dict(clean)
                    if setting is not None:
                        env["PYTHONOPTIMIZE"] = setting
                    args = [sys.executable, *flags, str(gate), stage]
                    if stage == "local":
                        args.append(str(request))
                    completed = subprocess.run(args, cwd=tmp, env=env, capture_output=True,
                                               text=True, timeout=10, check=False)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(marker, completed.stderr)
                    self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_gate_has_no_dispatch_import_or_call_path(self):
        tree = ast.parse((PACKAGE / "no_dispatch_gate.py").read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add((node.module or "").split(".")[0])
        self.assertTrue({"subprocess", "socket", "paramiko"}.isdisjoint(imports))
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
        source = (PACKAGE / "no_dispatch_gate.py").read_text()
        self.assertIn('if sys.flags.optimize != 0:', source)
        self.assertIn('if "PYTHONOPTIMIZE" in os.environ:', source)
        self.assertIn('raise DispatchHold("R6_FROZEN_446B_LAUNCHER_UNSAFE_HOLD")', source)

    def test_frozen_launcher_optimized_bypass_is_known_hold_not_a_pass(self):
        source = git_blob(SOURCE, "research/fresh_start_launch.py").decode()
        tree = ast.parse(source)
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        remote = next(node.value.value for node in tree.body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "REMOTE_SCRIPT"
                              for target in node.targets))
        self.assertIn("assert request.get('dispatch_authorized') is True", source)
        self.assertIn("assert sufficient_free_space(free)", remote)
        self.assertIn("assert commit == m['expected_commit']", remote)
        self.assertNotIn("PYTHONOPTIMIZE", remote)
        self.assertNotIn("m.get('dispatch_authorized')", remote)

        fake_host = types.ModuleType("fresh_start_host")
        fake_host.MIN_OPERATIONAL_FREE_BYTES = 6979321856
        with tempfile.TemporaryDirectory(prefix="trn22-r6-known-defect-offline-") as tmp:
            receipts = Path(tmp) / "research/run-logs"
            receipts.mkdir(parents=True)
            attempted = []

            def fake_contact(*args, **kwargs):
                attempted.append((args, kwargs))
                raise RuntimeError("FAKE_HOST_CONTACT_REACHED")

            namespace = dict(
                argparse=argparse, Path=Path, json=json, re=re, shlex=shlex,
                subprocess=types.SimpleNamespace(run=fake_contact),
                ROOT=Path(tmp), HOSTS={"trn22": "fake-host"}, SSH=["fake-ssh"],
                REMOTE_SCRIPT="pass", datetime=datetime, timezone=timezone,
            )
            module = ast.Module(body=[main], type_ignores=[])
            argv = ["frozen-launcher.py", str(PACKAGE / "NATIVE_REQUEST.DISABLED.json")]
            with patch.dict(sys.modules, {"fresh_start_host": fake_host}), patch.object(
                sys, "argv", argv
            ):
                normal = dict(namespace)
                exec(compile(module, "frozen-launcher-main", "exec", optimize=0), normal)
                with self.assertRaisesRegex(AssertionError, "one-label dispatch"):
                    normal["main"]()
                self.assertEqual(attempted, [])
                self.assertEqual(list(receipts.iterdir()), [])

                optimized = dict(namespace)
                exec(compile(module, "frozen-launcher-main", "exec", optimize=1), optimized)
                with self.assertRaisesRegex(RuntimeError, "FAKE_HOST_CONTACT_REACHED"):
                    optimized["main"]()
            self.assertEqual(len(attempted), 1)
            self.assertTrue((receipts / (LABEL + ".dispatch.json")).is_file())


if __name__ == "__main__":
    unittest.main()
