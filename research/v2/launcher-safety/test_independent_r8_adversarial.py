#!/usr/bin/env python3
"""Independent, local-only r8 launcher source witnesses.

All launcher SSH, Git, preflight, monitor and guard contacts use fakes. The
two child-process probes execute only a temporary shell marker or Python's
optimization flag; they never run training or contact a host.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RESEARCH = Path(__file__).resolve().parents[2]
REPO = RESEARCH.parent
sys.path.insert(0, str(RESEARCH))

import fresh_start_launch as launcher
import test_fresh_start_launch_nonassert as fixture

SUBJECT = "d3f39dd8cd8bfa16e25306d4d06293e188fd2cff"
LAUNCHER_SHA256 = "dcf9fdc0e1d8ecc459e6687b87511f00a5747d3fff19dfdfeae0a6b57d818c07"
REQUEST_PATH = "research/v2/disk-monitor/fs-20260925-v2-c5c-a0-control-trn22-r3.request.DISABLED.json"
EXPECTED_PATHS = (
    "train.py", "prepare.py", "research/run_experiment.py",
    "research/fresh_start_launch.py", "research/fresh_start_sync.py",
    "research/fresh_start_guard.py", "research/fresh_start_host.py",
    "research/fresh_start_backup.py", "research/fresh_start_monitor_interlock.py",
    "research/v2/disk-monitor/root_free_guard_monitor.py",
    "research/v2/disk-monitor/paired_root_free_guard_monitor.py",
)


def frozen_request():
    raw = subprocess.check_output(["git", "show", f"{SUBJECT}:{REQUEST_PATH}"], cwd=REPO)
    return json.loads(raw)


class IndependentR8Tests(unittest.TestCase):
    def setUp(self):
        self.local = fixture.LocalNonAssertTests(
            "test_valid_unoptimized_request_reaches_only_fake_ssh")
        self.remote = fixture.RemoteNonAssertTests(
            "test_valid_unoptimized_remote_reaches_only_fake_guard")

    def request(self):
        request = fixture.request_template()
        request["fingerprint_paths"] = list(EXPECTED_PATHS)
        return request

    def assert_early_refusal(self, mutate):
        for optimized_compile in (0, 1):
            with self.subTest(optimized_compile=optimized_compile):
                request = self.request()
                mutate(request)
                error, contacts, receipts = self.local.run_local(
                    request, compiled_optimize=optimized_compile)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(contacts, [])
                self.assertEqual(receipts, [])
                error, calls, state = self.remote.run_remote(
                    compiled_optimize=optimized_compile,
                    mutation=lambda remote, *_: mutate(remote), allow_fake_guard=True)
                self.assertIsInstance(error, RuntimeError)
                self.assertFalse(state["host_dir"])
                self.assertIsNone(state["no_spawn"])
                self.assertFalse(state["manifest"] or state["job"] or state["log"])
                self.assertEqual(calls["git"], [])
                self.assertEqual(calls["preflight"], [])
                self.assertEqual(calls["interlock"], [])
                self.assertEqual(calls["guard"], [])

    def test_exact_subject_and_disabled_request_binding(self):
        self.assertEqual(hashlib.sha256((RESEARCH / "fresh_start_launch.py").read_bytes()).hexdigest(),
                         LAUNCHER_SHA256)
        self.assertEqual(launcher.SOURCE_FINGERPRINT_PATHS, EXPECTED_PATHS)
        self.assertEqual(frozen_request()["fingerprint_paths"], list(EXPECTED_PATHS))
        self.assertIs(frozen_request()["dispatch_authorized"], False)
        self.assertNotEqual(frozen_request()["expected_commit"], SUBJECT)
        self.assert_early_refusal(lambda request: request.update(dispatch_authorized=False))

    def test_r7_zero_timeout_and_unbounded_forms_refuse_before_contact(self):
        bad = (
            ["timeout", "0", "true"],
            ["timeout", "--kill-after=60s", "0s", "true"],
            ["timeout", "--kill-after=0s", "1s", "true"],
            ["timeout", "1s", "true"],
            ["timeout", "--kill-after=60s", "-1s", "true"],
            ["timeout", "--kill-after=60s", "99999999d", "true"],
        )
        for command in bad:
            with self.subTest(command=command):
                self.assert_early_refusal(lambda request: request.update(command=command))

    def test_r7_nonsource_and_incomplete_fingerprints_refuse_before_contact(self):
        expected = list(EXPECTED_PATHS)
        bad = (
            ["research/agentic/private.json"],
            expected[:-1],
            expected[:-1] + ["research/scratch"],
            expected[:-1] + [expected[0]],
            expected + ["research/agentic/private.json"],
            expected[:-1] + ["research/../prepare.py"],
            expected[:-1] + ["/tmp/private.json"],
        )
        for paths in bad:
            with self.subTest(paths=paths):
                self.assert_early_refusal(
                    lambda request: request.update(fingerprint_paths=paths))

    def test_launcher_optimization_modes_refuse_before_contact(self):
        for kwargs in ({"runtime_optimize": 1}, {"runtime_optimize": 2},
                       {"pythonoptimize": "1"}, {"pythonoptimize": "0"},
                       {"pythonoptimize": ""}):
            with self.subTest(kwargs=kwargs):
                error, contacts, receipts = self.local.run_local(self.request(), **kwargs)
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("optimized Python", str(error))
                self.assertEqual(contacts, [])
                self.assertEqual(receipts, [])
                error, calls, state = self.remote.run_remote(**kwargs)
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("optimized Python", str(error))
                self.assertFalse(state["host_dir"])
                self.assertEqual(calls["guard"], [])

    def test_valid_frozen_a0_shape_reaches_only_fakes(self):
        request = copy.deepcopy(frozen_request())
        self.assertTrue(launcher.bounded_timeout_command(request["command"]))
        request["dispatch_authorized"] = True  # in-memory synthetic probe only
        for optimized_compile in (0, 1):
            error, contacts, receipts = self.local.run_local(
                request, compiled_optimize=optimized_compile)
            self.assertIsNone(error)
            self.assertEqual(len(contacts), 1)
            self.assertEqual(len(receipts), 1)
            error, calls, state = self.remote.run_remote(
                compiled_optimize=optimized_compile,
                mutation=lambda remote, *_: remote.update(command=list(request["command"])),
                allow_fake_guard=True)
            self.assertIsNone(error)
            self.assertEqual(len(calls["guard"]), 1)
            self.assertTrue(state["host_dir"] and state["manifest"] and state["job"])

    def test_child_python_optimization_can_hide_inside_shell_argument(self):
        command = ["timeout", "--kill-after=2s", "2s", "sh", "-c",
                   "export PYTHONOPTIMIZE=1; python3 -c 'import sys; print(sys.flags.optimize)'"]
        self.assertTrue(launcher.bounded_timeout_command(command))
        request = self.request()
        request["command"] = command
        error, contacts, receipts = self.local.run_local(request)
        self.assertIsNone(error)
        self.assertEqual(len(contacts), 1)
        self.assertEqual(len(receipts), 1)
        error, calls, _state = self.remote.run_remote(
            mutation=lambda remote, *_: remote.update(command=command), allow_fake_guard=True)
        self.assertIsNone(error)
        self.assertEqual(len(calls["guard"]), 1)
        clean_env = dict(os.environ)
        clean_env.pop("PYTHONOPTIMIZE", None)
        completed = subprocess.run(command, env=clean_env, capture_output=True,
                                   text=True, timeout=5, check=True)
        self.assertEqual(completed.stdout.strip(), "1")

    def test_request_path_can_select_a_non_gnu_timeout(self):
        command = ["timeout", "--kill-after=1s", "1s", "true"]
        with tempfile.TemporaryDirectory(prefix="r8-independent-fake-timeout-") as tmp:
            fake_timeout = Path(tmp) / "timeout"
            fake_timeout.write_text("#!/bin/sh\nprintf 'FAKE_TIMEOUT_NO_DEADLINE\\n'\n")
            fake_timeout.chmod(0o700)
            path = f"{tmp}:/usr/bin:/bin"
            request = self.request()
            request["command"] = command
            request["environment"]["PATH"] = path
            error, contacts, receipts = self.local.run_local(request)
            self.assertIsNone(error)
            self.assertEqual(len(contacts), 1)
            self.assertEqual(len(receipts), 1)
            error, calls, _state = self.remote.run_remote(
                mutation=lambda remote, *_: (
                    remote.update(command=command), remote["environment"].update(PATH=path)),
                allow_fake_guard=True)
            self.assertIsNone(error)
            self.assertEqual(len(calls["guard"]), 1)
            env = dict(os.environ)
            env["PATH"] = path
            completed = subprocess.run(command, env=env, capture_output=True,
                                       text=True, timeout=5, check=True)
            self.assertEqual(completed.stdout.strip(), "FAKE_TIMEOUT_NO_DEADLINE")


if __name__ == "__main__":
    unittest.main()
