"""Independent offline checks for the frozen, disabled trn22 r10 A0 package.

The launcher contact seam is fake and the remote script runs only in a local
Python child. This file never connects to a training host or starts a job.
"""

import argparse
import ast
import copy
from datetime import datetime, timezone
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
PACKAGE = ROOT / "research/v2/authorizations/trn22-c5c-a0-r10-native-disabled"
SOURCE = "22bfbab5a77d6feb670d2d737504f8b70b08ddc1"


def frozen_launcher():
    return subprocess.check_output(
        ["git", "show", f"{SOURCE}:research/fresh_start_launch.py"],
        cwd=ROOT, text=True,
    )


def function_from(source, name):
    node = next(
        item for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<frozen>", "exec"), namespace)
    return namespace[name]


def remote_script(source):
    return next(
        node.value.value for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REMOTE_SCRIPT"
                for target in node.targets)
    )


class IndependentDisabledR10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request_path = PACKAGE / "NATIVE_REQUEST.DISABLED.json"
        cls.request = json.loads(cls.request_path.read_text())
        cls.source = frozen_launcher()

    def test_exact_disabled_local_main_has_no_contact_or_receipt(self):
        """Execute frozen main with its real gate and a fake contact seam."""
        names = {"require_unoptimized_python", "validate_request",
                 "require_gnu_timeout", "bounded_timeout_command", "main"}
        body = [node for node in ast.parse(self.source).body
                if isinstance(node, ast.FunctionDef) and node.name in names]
        calls = []

        def deny_contact(*args, **kwargs):
            calls.append((args, kwargs))
            raise RuntimeError("FAKE_CONTACT_REACHED")

        with tempfile.TemporaryDirectory(prefix="r10-disabled-local-") as temp:
            logs = Path(temp) / "research/run-logs"
            logs.mkdir(parents=True)
            namespace = {
                "argparse": argparse, "Path": Path, "json": json,
                "os": os, "re": re, "shlex": shlex, "sys": sys,
                "subprocess": types.SimpleNamespace(run=deny_contact),
                "ROOT": Path(temp), "HOSTS": {"trn22": "fake-host"},
                "SSH": ["fake-ssh"], "REMOTE_SCRIPT": remote_script(self.source),
                "datetime": datetime, "timezone": timezone,
            }
            with patch.object(sys, "argv", ["fresh_start_launch.py", str(self.request_path)]):
                for compiled_optimize in (0, 1):
                    with self.subTest(compiled_optimize=compiled_optimize):
                        isolated = dict(namespace)
                        exec(compile(ast.Module(body=body, type_ignores=[]),
                                     "<frozen-launcher>", "exec",
                                     optimize=compiled_optimize), isolated)
                        reason = ("optimized Python" if sys.flags.optimize or
                                  "PYTHONOPTIMIZE" in os.environ else "not authorized")
                        with self.assertRaisesRegex(RuntimeError, reason):
                            isolated["main"]()
                        self.assertEqual(calls, [])
                        self.assertEqual(list(logs.iterdir()), [])

    def test_exact_disabled_remote_script_refuses_in_real_python_modes(self):
        """Run embedded remote code locally; disabled and optimized modes exit early."""
        script = remote_script(self.source)
        clean = dict(os.environ)
        clean.pop("PYTHONOPTIMIZE", None)
        clean["PYTHONDONTWRITEBYTECODE"] = "1"
        cases = (("normal", [], None, "not authorized"),
                 ("dash_o", ["-O"], None, "optimized Python"),
                 ("env_optimize", [], "1", "optimized Python"))
        for name, flags, pythonoptimize, expected in cases:
            with self.subTest(mode=name), tempfile.TemporaryDirectory(
                    prefix="r10-disabled-remote-") as temp:
                env = dict(clean)
                if pythonoptimize is not None:
                    env["PYTHONOPTIMIZE"] = pythonoptimize
                result = subprocess.run(
                    [sys.executable, *flags, "-c", script],
                    input=json.dumps(self.request), text=True,
                    capture_output=True, cwd=temp, env=env, timeout=10,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                self.assertEqual(list(Path(temp).iterdir()), [])

    def test_local_and_embedded_profile_reject_drift(self):
        """The exact child profile cannot acquire an extra flag or environment."""
        local = function_from(self.source, "bounded_timeout_command")
        remote = function_from(remote_script(self.source), "bounded_timeout_command")
        mutations = (
            lambda r: r["command"].__setitem__(0, "timeout"),
            lambda r: r["command"].__setitem__(5, "env"),
            lambda r: r["command"].append("--extra"),
            lambda r: r["command"].__setitem__(4, "0s"),
            lambda r: r["command"].__setitem__(
                r["command"].index("--nproc_per_node=4"), "--nproc_per_node=8"),
            lambda r: r["environment"].update(NEURON_COMPETITION_R1_SEED="43"),
            lambda r: r["environment"].update(NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS="1801"),
            lambda r: r["environment"].update(NEURON_LOGICAL_NC_CONFIG="1"),
            lambda r: r["environment"].update(NEURON_CC_FLAGS="--optlevel=2"),
            lambda r: r["environment"].update(NEURON_COMPILE_CACHE_DIR="/tmp/fake"),
            lambda r: r["environment"].update(PYTHONOPTIMIZE="1"),
            lambda r: r["environment"].update(PATH="/tmp/fake:/usr/bin"),
        )
        for validator in (local, remote):
            self.assertTrue(validator(
                self.request["command"], self.request["host"],
                self.request["label"], self.request["remote"],
                self.request["environment"],
            ))
            for index, mutation in enumerate(mutations):
                with self.subTest(validator=validator, mutation=index):
                    altered = copy.deepcopy(self.request)
                    mutation(altered)
                    self.assertFalse(validator(
                        altered["command"], altered["host"],
                        altered["label"], altered["remote"],
                        altered["environment"],
                    ))


if __name__ == "__main__":
    unittest.main()
