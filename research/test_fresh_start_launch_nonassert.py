#!/usr/bin/env python3
"""Offline launcher safety tests. All SSH and guard calls are fakes."""

import argparse
import ast
import copy
from datetime import datetime, timezone
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SOURCE_PATH = Path(__file__).with_name("fresh_start_launch.py")
SOURCE = SOURCE_PATH.read_text()
TREE = ast.parse(SOURCE)
REMOTE_SCRIPT = next(
    node.value.value for node in TREE.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "REMOTE_SCRIPT"
            for target in node.targets)
)
LOCAL_FUNCTIONS = [
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {"require_unoptimized_python", "validate_request", "main"}
]
FLOOR = 13 * 1024**3 // 2
PRELAUNCH = 10 * 1024**3
COMMIT = "a" * 40
LABEL = "fs-synthetic-nonassert-r7"
SOURCE_SHA256 = "ece127ad02f72871341369736732f9b324bad9ffa865bc503def7732bb8ad3ab"


class FlagsProxy:
    """Change only optimize; retain every other interpreter flag."""

    def __init__(self, original, optimize):
        self.original = original
        self.optimize = optimize

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, index):
        return self.original[index]


class SourceBindingTests(unittest.TestCase):
    def test_exact_launcher_and_protected_training_bytes(self):
        self.assertEqual(hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(), SOURCE_SHA256)
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(TREE)))
        self.assertFalse(any(isinstance(node, ast.Assert)
                             for node in ast.walk(ast.parse(REMOTE_SCRIPT))))
        root = SOURCE_PATH.parent.parent
        for path, expected in (
            ("train.py", "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"),
            ("prepare.py", "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"),
        ):
            with self.subTest(path=path):
                blob = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=root)
                self.assertEqual(hashlib.sha256(blob).hexdigest(), expected)


def request_template(remote="/home/ubuntu/trainium_freshstart_synthetic_r7",
                     monitor="/home/ubuntu/monitor-synthetic-r7"):
    return {
        "dispatch_authorized": True,
        "host": "trn22",
        "expected_hostname": socket.gethostname(),
        "label": LABEL,
        "expected_commit": COMMIT,
        "remote": str(remote),
        "command": ["timeout", "1s", "true"],
        "environment": {"NEURON_LOGICAL_NC_CONFIG": "2",
                        "NEURON_CC_FLAGS": "--optlevel=1"},
        "min_free_bytes": FLOOR,
        "pre_launch_required_free_bytes": PRELAUNCH,
        "runner_owns_manifest": False,
        "monitor_interlock": {
            "pair_binding_path": str(Path(monitor) / "pair.json"),
            "interlock_path": str(Path(monitor) / "interlock.json"),
            "no_spawn_path": str(Path(monitor) / "no-spawn.json"),
        },
    }


def local_namespace(root, contacts, optimize=0):
    def fake_ssh(*args, **kwargs):
        contacts.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, stdout="FAKE_ONLY\n", stderr="")

    values = dict(
        argparse=argparse, datetime=datetime, timezone=timezone, json=json,
        os=os, Path=Path, re=re, shlex=shlex,
        subprocess=types.SimpleNamespace(run=fake_ssh), sys=sys,
        ROOT=root, HOSTS={"trn22": "fake-host"}, SSH=["fake-ssh"],
        REMOTE_SCRIPT=REMOTE_SCRIPT,
    )
    code = compile(ast.Module(body=LOCAL_FUNCTIONS, type_ignores=[]),
                   "launcher-local-functions", "exec", optimize=optimize)
    exec(code, values)
    return values


def probe_real_local_interpreter():
    """Child-process probe with a fake SSH seam and the real sys.flags."""
    with tempfile.TemporaryDirectory(prefix="launcher-r7-real-local-probe-") as tmp:
        root = Path(tmp)
        receipts = root / "research/run-logs"
        receipts.mkdir(parents=True)
        request_path = root / "request.json"
        request_path.write_text(json.dumps(request_template()))
        contacts = []
        local = local_namespace(root, contacts, sys.flags.optimize)
        fake_host = types.ModuleType("fresh_start_host")
        fake_host.MIN_OPERATIONAL_FREE_BYTES = FLOOR
        with (patch.dict(sys.modules, {"fresh_start_host": fake_host}),
              patch.object(sys, "argv", ["fresh_start_launch.py", str(request_path)])):
            try:
                local["main"]()
            except RuntimeError as error:
                if "optimized Python" not in str(error):
                    raise
            else:
                raise RuntimeError("optimized local probe was not refused")
        if contacts or list(receipts.iterdir()):
            raise RuntimeError("optimized local probe reached fake contact or receipt")
    print("LOCAL_REAL_MODE_REFUSED")


def probe_real_remote_interpreter():
    """Child-process probe; fake Popen and temporary paths only."""
    with tempfile.TemporaryDirectory(prefix="launcher-r7-real-remote-probe-") as tmp:
        root = Path(tmp)
        monitor = root / "monitor"
        monitor.mkdir(mode=0o700)
        request = request_template(root / "missing-checkout", monitor)
        with (patch.object(sys, "stdin", io.StringIO(json.dumps(request))),
              patch.object(subprocess, "Popen",
                           side_effect=RuntimeError("FAKE_GUARD_REACHED")),
              patch.object(subprocess, "check_output",
                           side_effect=RuntimeError("FAKE_GIT_REACHED"))):
            try:
                exec(compile(REMOTE_SCRIPT, "real-mode-fake-remote", "exec",
                             optimize=sys.flags.optimize), {"__name__": "__main__"})
            except RuntimeError as error:
                if "optimized Python" not in str(error):
                    raise
            else:
                raise RuntimeError("optimized remote probe was not refused")
        if list(monitor.iterdir()) or (root / "host-lock-dir").exists():
            raise RuntimeError("optimized remote probe wrote a target")
    print("REMOTE_REAL_MODE_REFUSED")


class ActualInterpreterModeTests(unittest.TestCase):
    def test_real_dash_o_and_pythonoptimize_refuse_without_contact(self):
        code = (
            "import importlib.util,sys;"
            "spec=importlib.util.spec_from_file_location('launcher_probe',sys.argv[1]);"
            "module=importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(module);"
            "getattr(module,sys.argv[2])()"
        )
        clean = dict(os.environ)
        clean.pop("PYTHONOPTIMIZE", None)
        clean["PYTHONDONTWRITEBYTECODE"] = "1"
        cases = (("dash_o", ["-O"], None), ("dash_oo", ["-OO"], None),
                 ("env_1", [], "1"), ("env_0", [], "0"),
                 ("env_empty", [], ""))
        for name, flags, setting in cases:
            for probe, marker in (("probe_real_local_interpreter", "LOCAL_REAL_MODE_REFUSED"),
                                  ("probe_real_remote_interpreter", "REMOTE_REAL_MODE_REFUSED")):
                with self.subTest(mode=name, path=probe):
                    env = dict(clean)
                    if setting is not None:
                        env["PYTHONOPTIMIZE"] = setting
                    result = subprocess.run(
                        [sys.executable, *flags, "-c", code, str(Path(__file__).resolve()), probe],
                        cwd=SOURCE_PATH.parent, env=env, capture_output=True, text=True,
                        timeout=10, check=False)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(marker, result.stdout)


class LocalNonAssertTests(unittest.TestCase):
    def run_local(self, request, *, compiled_optimize=0, runtime_optimize=0,
                  pythonoptimize=None):
        with tempfile.TemporaryDirectory(prefix="launcher-r7-local-offline-") as tmp:
            root = Path(tmp)
            receipt_dir = root / "research/run-logs"
            receipt_dir.mkdir(parents=True)
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request))
            contacts = []
            local = local_namespace(root, contacts, compiled_optimize)
            fake_host = types.ModuleType("fresh_start_host")
            fake_host.MIN_OPERATIONAL_FREE_BYTES = FLOOR
            env = dict(os.environ)
            env.pop("PYTHONOPTIMIZE", None)
            if pythonoptimize is not None:
                env["PYTHONOPTIMIZE"] = pythonoptimize
            with (patch.dict(sys.modules, {"fresh_start_host": fake_host}),
                  patch.dict(os.environ, env, clear=True),
                  patch.object(sys, "flags", FlagsProxy(sys.flags, runtime_optimize)),
                  patch.object(sys, "argv", ["fresh_start_launch.py", str(request_path)]),
                  redirect_stdout(io.StringIO())):
                try:
                    local["main"]()
                    error = None
                except BaseException as caught:
                    error = caught
            receipts = list(receipt_dir.iterdir())
            return error, contacts, receipts

    def test_disabled_and_hostile_requests_never_reach_receipt_or_ssh(self):
        changes = {
            "disabled": lambda r: r.update(dispatch_authorized=False),
            "missing_interlock": lambda r: r.pop("monitor_interlock"),
            "relative_interlock": lambda r: r["monitor_interlock"].update(no_spawn_path="relative"),
            "split_interlock": lambda r: r["monitor_interlock"].update(no_spawn_path="/tmp/other"),
            "duplicate_interlock": lambda r: r["monitor_interlock"].update(
                no_spawn_path=r["monitor_interlock"]["interlock_path"]),
            "bad_host": lambda r: r.update(host=["trn22"]),
            "missing_hostname": lambda r: r.pop("expected_hostname"),
            "bad_label": lambda r: r.update(label="fs-../other"),
            "bad_commit": lambda r: r.update(expected_commit="not-a-commit"),
            "bad_checkout": lambda r: r.update(remote="/tmp/hostile-checkout"),
            "unbounded_command": lambda r: r.update(command=["torchrun", "train.py"]),
            "optimized_child_command": lambda r: r["command"].append("PYTHONOPTIMIZE=1"),
            "bad_environment": lambda r: r["environment"].update(NEURON_LOGICAL_NC_CONFIG="0"),
            "optimized_child_environment": lambda r: r["environment"].update(PYTHONOPTIMIZE="1"),
            "low_floor": lambda r: r.update(min_free_bytes=FLOOR - 1),
            "low_prelaunch": lambda r: r.update(pre_launch_required_free_bytes=PRELAUNCH - 1),
            "manifest_policy": lambda r: r.update(runner_owns_manifest="false"),
            "hostile_fingerprint": lambda r: r.update(
                fingerprint_paths=["research/agentic/.env"]),
        }
        for optimize in (0, 1):
            for name, change in changes.items():
                with self.subTest(compiled_optimize=optimize, mutation=name):
                    request = copy.deepcopy(request_template())
                    change(request)
                    error, contacts, receipts = self.run_local(
                        request, compiled_optimize=optimize)
                    self.assertIsInstance(error, RuntimeError)
                    self.assertEqual(contacts, [])
                    self.assertEqual(receipts, [])

    def test_runtime_optimization_and_environment_refuse_before_receipt(self):
        for compiled in (0, 1):
            for runtime, setting in ((1, None), (2, None), (0, "1"), (0, "0"), (0, "")):
                with self.subTest(compiled=compiled, runtime=runtime, setting=setting):
                    error, contacts, receipts = self.run_local(
                        request_template(), compiled_optimize=compiled,
                        runtime_optimize=runtime, pythonoptimize=setting)
                    self.assertIsInstance(error, RuntimeError)
                    self.assertIn("optimized Python", str(error))
                    self.assertEqual(contacts, [])
                    self.assertEqual(receipts, [])

    def test_valid_unoptimized_request_reaches_only_fake_ssh(self):
        for compiled in (0, 1):
            with self.subTest(compiled=compiled):
                error, contacts, receipts = self.run_local(
                    request_template(), compiled_optimize=compiled)
                self.assertIsNone(error)
                self.assertEqual(len(contacts), 1)
                self.assertEqual(contacts[0][0][0][:2], ["fake-ssh", "fake-host"])
                self.assertEqual(len(receipts), 1)
                self.assertEqual(receipts[0].name, LABEL + ".dispatch.json")


class RemoteNonAssertTests(unittest.TestCase):
    def run_remote(self, *, compiled_optimize=0, runtime_optimize=0,
                   pythonoptimize=None, disk_free=11 * 1024**3,
                   dirty=b"", untracked=b"", commit=COMMIT,
                   interlock_fails=False, mutation=None, allow_fake_guard=False):
        with tempfile.TemporaryDirectory(prefix="launcher-r7-remote-offline-") as tmp:
            root = Path(tmp)
            checkout = root / "checkout"
            runlogs = checkout / "research/run-logs"
            runlogs.mkdir(parents=True)
            for name in ("train.py", "prepare.py", "research/run_experiment.py",
                         "research/fresh_start_host.py", "research/fresh_start_guard.py",
                         "research/fresh_start_backup.py",
                         "research/fresh_start_monitor_interlock.py"):
                target = checkout / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("synthetic source\n")
            monitor = root / "monitor"
            monitor.mkdir(mode=0o700)
            monitor.chmod(0o700)
            request = request_template(checkout, monitor)
            if mutation is not None:
                mutation(request, root, checkout, monitor)

            calls = {"git": [], "guard": [], "interlock": [],
                     "preflight": [], "no_spawn": []}

            def fake_git(argv, **kwargs):
                calls["git"].append(argv)
                if argv[:2] == ["git", "diff"]:
                    return dirty
                if argv[:2] == ["git", "ls-files"]:
                    return untracked
                if argv[:3] == ["git", "rev-parse", "HEAD"]:
                    return commit + "\n"
                raise AssertionError("UNEXPECTED_FAKE_GIT_COMMAND")

            def fake_guard(*args, **kwargs):
                calls["guard"].append((args, kwargs))
                if not allow_fake_guard:
                    raise AssertionError("FAKE_GUARD_SPAWN_REACHED")
                return types.SimpleNamespace(pid=12345)

            def live_pair(_request):
                calls["interlock"].append("require_live_pair")
                if interlock_fails:
                    raise RuntimeError("FAKE_INTERLOCK_REFUSAL")
                return {"synthetic": "live pair"}

            def fake_preflight(_checkout, _label):
                calls["preflight"].append((_checkout, _label))
                return {"passed": True}

            fake_host = types.ModuleType("fresh_start_host")
            fake_host.HOST_DIRECTORY = root / "host-lock-dir"
            fake_host.HOST_LOCK = fake_host.HOST_DIRECTORY / "device0.lock"
            fake_host.MIN_OPERATIONAL_FREE_BYTES = FLOOR
            fake_host.sufficient_free_space = lambda free: type(free) is int and free >= FLOOR
            fake_host.preflight = fake_preflight
            fake_interlock = types.ModuleType("fresh_start_monitor_interlock")
            fake_interlock.require_live_pair = live_pair
            fake_interlock.write_interlock_pass = lambda *_args: calls["interlock"].append("pass")
            fake_interlock.write_no_spawn = lambda *_args: calls["no_spawn"].append(_args)

            env = dict(os.environ)
            env.pop("PYTHONOPTIMIZE", None)
            if pythonoptimize is not None:
                env["PYTHONOPTIMIZE"] = pythonoptimize
            output = io.StringIO()
            errors = io.StringIO()
            remote_namespace = {"__name__": "__main__"}
            with (patch.dict(sys.modules, {"fresh_start_host": fake_host,
                                           "fresh_start_monitor_interlock": fake_interlock}),
                  patch.object(sys, "stdin", io.StringIO(json.dumps(request))),
                  patch.object(sys, "path", list(sys.path)),
                  patch.object(sys, "flags", FlagsProxy(sys.flags, runtime_optimize)),
                  patch.dict(os.environ, env, clear=True),
                  patch.object(subprocess, "check_output", side_effect=fake_git),
                  patch.object(subprocess, "Popen", side_effect=fake_guard),
                  patch.object(shutil, "disk_usage",
                               return_value=types.SimpleNamespace(free=disk_free)),
                  redirect_stdout(output), patch.object(sys, "stderr", errors)):
                try:
                    exec(compile(REMOTE_SCRIPT, "fake-remote-launcher", "exec",
                                 optimize=compiled_optimize), remote_namespace)
                    error = None
                except BaseException as caught:
                    error = caught
                finally:
                    lock = remote_namespace.get("host_lock")
                    if lock is not None:
                        lock.close()
            label = request.get("label", LABEL)
            receipt_path = Path(request.get("monitor_interlock", {}).get(
                "no_spawn_path", monitor / "no-spawn.json"))
            receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
            return error, calls, {
                "host_dir": fake_host.HOST_DIRECTORY.exists(),
                "manifest": (runlogs / (label + ".launch.json")).exists(),
                "job": (runlogs / (label + ".job.json")).exists(),
                "log": (runlogs / (label + ".log")).exists(),
                "no_spawn": receipt,
                "stdout": output.getvalue(),
                "stderr": errors.getvalue(),
            }

    def test_remote_optimization_refuses_before_checkout_lock_or_receipt(self):
        cases = ((0, 1, None), (1, 1, None), (1, 0, "1"),
                 (0, 0, "0"), (0, 0, ""))
        for compiled, runtime, setting in cases:
            with self.subTest(compiled=compiled, runtime=runtime, setting=setting):
                error, calls, state = self.run_remote(
                    compiled_optimize=compiled, runtime_optimize=runtime,
                    pythonoptimize=setting)
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("optimized Python", str(error))
                self.assertFalse(state["host_dir"])
                self.assertIsNone(state["no_spawn"])
                self.assertFalse(state["manifest"] or state["job"] or state["log"])
                self.assertEqual(calls["git"], [])
                self.assertEqual(calls["guard"], [])

    def test_remote_disabled_and_malformed_interlock_refuse_before_writes(self):
        changes = {
            "disabled": lambda r, *_: r.update(dispatch_authorized=False),
            "missing_interlock": lambda r, *_: r.pop("monitor_interlock"),
            "relative_interlock": lambda r, *_: r["monitor_interlock"].update(
                no_spawn_path="relative"),
            "split_interlock": lambda r, *_: r["monitor_interlock"].update(
                no_spawn_path="/tmp/other"),
            "wrong_hostname": lambda r, *_: r.update(expected_hostname="wrong"),
            "bad_commit_field": lambda r, *_: r.update(expected_commit="wrong"),
            "optimized_child": lambda r, *_: r["environment"].update(PYTHONOPTIMIZE="1"),
            "low_floor_field": lambda r, *_: r.update(min_free_bytes=FLOOR - 1),
        }
        for optimize in (0, 1):
            for name, mutate in changes.items():
                with self.subTest(compiled_optimize=optimize, mutation=name):
                    error, calls, state = self.run_remote(
                        compiled_optimize=optimize, mutation=mutate)
                    self.assertIsInstance(error, RuntimeError)
                    self.assertFalse(state["host_dir"])
                    self.assertIsNone(state["no_spawn"])
                    self.assertFalse(state["manifest"] or state["job"] or state["log"])
                    self.assertEqual(calls["git"], [])
                    self.assertEqual(calls["guard"], [])

    def test_disk_source_commit_and_label_gates_survive_optimized_compile(self):
        cases = (
            ("disk", {"disk_free": FLOOR - 1}, "6.5 GiB", "early_disk"),
            ("dirty", {"dirty": b"train.py\n"}, "not committed", "source_identity"),
            ("untracked", {"untracked": b"research/fresh_start_launch.py\n"},
             "not committed", "source_identity"),
            ("fingerprint", {"mutation": lambda r, *_: r.update(
                fingerprint_paths=["research/agentic/.env"])},
             "invalid source fingerprint paths", "source_identity"),
            ("commit", {"commit": "b" * 40}, "unexpected source commit", "source_identity"),
            ("label", {"mutation": lambda r, _root, checkout, _monitor: (
                checkout / "research/run-logs" / (r["label"] + ".launch.json")
            ).write_text("occupied")}, "label already exists", "label_identity"),
            ("dangling_label", {"mutation": lambda r, _root, checkout, _monitor: (
                checkout / "research/run-logs" / (r["label"] + ".launch.json")
            ).symlink_to("absent-target")}, "label already exists", "label_identity"),
        )
        for optimize in (0, 1):
            for name, kwargs, message, stage in cases:
                with self.subTest(compiled_optimize=optimize, failure=name):
                    error, calls, state = self.run_remote(
                        compiled_optimize=optimize, **kwargs)
                    self.assertIsInstance(error, RuntimeError)
                    self.assertIn(message, str(error))
                    self.assertEqual(calls["guard"], [])
                    self.assertFalse(state["job"] or state["log"])
                    self.assertIsNotNone(state["no_spawn"])
                    self.assertEqual(state["no_spawn"]["stage"], stage)
                    self.assertEqual(state["no_spawn"]["status"], "NO_GUARD_SPAWNED")

    def test_interlock_refusal_prevents_manifest_job_and_guard(self):
        for optimize in (0, 1):
            with self.subTest(compiled_optimize=optimize):
                error, calls, state = self.run_remote(
                    compiled_optimize=optimize, interlock_fails=True)
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("FAKE_INTERLOCK_REFUSAL", str(error))
                self.assertEqual(calls["guard"], [])
                self.assertFalse(state["manifest"] or state["job"] or state["log"])
                self.assertEqual(len(calls["no_spawn"]), 1)

    def test_valid_unoptimized_remote_reaches_only_fake_guard(self):
        for optimize in (0, 1):
            with self.subTest(compiled_optimize=optimize):
                error, calls, state = self.run_remote(
                    compiled_optimize=optimize, allow_fake_guard=True)
                self.assertIsNone(error)
                self.assertEqual(len(calls["guard"]), 1)
                self.assertEqual(len(calls["preflight"]), 1)
                self.assertEqual(calls["interlock"].count("require_live_pair"), 4)
                self.assertTrue(state["manifest"] and state["job"] and state["log"])
                self.assertIsNone(state["no_spawn"])
                self.assertIn('"pid": 12345', state["stdout"])


if __name__ == "__main__":
    unittest.main()
