#!/usr/bin/env python3
"""Independent, workstation-only r9 launcher adversarial checks.

All would-be SSH, timeout-version, Git, preflight, and guard calls are fakes.
The only files created are disposable local request and receipt fixtures.
"""

import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


RESEARCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESEARCH))
import fresh_start_launch as launcher
import test_fresh_start_launch_nonassert as fixture


TORCHRUN = "/home/ubuntu/workspace/native_venv/bin/torchrun"


def replace_child(request, child):
    argv = request["command"]
    argv[argv.index(TORCHRUN):] = child


def replace_assignment(request, key, value):
    argv = request["command"]
    index = next(i for i, arg in enumerate(argv) if arg.startswith(key + "="))
    argv[index] = key + "=" + value


class IndependentR9AdversarialTests(unittest.TestCase):
    def setUp(self):
        self.local = fixture.LocalNonAssertTests(
            "test_valid_unoptimized_request_reaches_only_fake_ssh")
        self.remote = fixture.RemoteNonAssertTests(
            "test_valid_unoptimized_remote_reaches_only_fake_guard")

    def assert_request_refused_before_contacts(self, mutate):
        for compile_mode in (0, 1):
            with self.subTest(compile_mode=compile_mode):
                request = fixture.request_template()
                mutate(request)
                error, contacts, receipts = self.local.run_local(
                    copy.deepcopy(request), compiled_optimize=compile_mode)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(contacts, [])
                self.assertEqual(receipts, [])

                error, calls, state = self.remote.run_remote(
                    compiled_optimize=compile_mode,
                    mutation=lambda remote, *_: mutate(remote),
                    allow_fake_guard=True)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(calls["timeout_version"], [])
                self.assertEqual(calls["git"], [])
                self.assertEqual(calls["preflight"], [])
                self.assertEqual(calls["interlock"], [])
                self.assertEqual(calls["guard"], [])
                self.assertFalse(state["host_dir"])
                self.assertIsNone(state["no_spawn"])
                self.assertFalse(state["manifest"] or state["job"] or state["log"])

    def test_exact_profile_rejects_command_and_environment_perturbations(self):
        cases = {
            "alternate_absolute_timeout": lambda r: r["command"].__setitem__(0, "/usr/local/bin/timeout"),
            "relative_timeout": lambda r: r["command"].__setitem__(0, "timeout"),
            "changed_kill_after": lambda r: r["command"].__setitem__(3, "--kill-after=120s"),
            "changed_deadline": lambda r: r["command"].__setitem__(4, "3601s"),
            "zero_deadline": lambda r: r["command"].__setitem__(4, "0s"),
            "alternate_absolute_env": lambda r: r["command"].__setitem__(5, "/bin/env"),
            "relative_env": lambda r: r["command"].__setitem__(5, "env"),
            "alternate_env_clear": lambda r: r["command"].__setitem__(6, "--ignore-environment"),
            "fake_path_in_argv": lambda r: replace_assignment(r, "PATH", "/tmp/fake:/usr/bin:/bin"),
            "extra_env_assignment": lambda r: r["command"].insert(r["command"].index(TORCHRUN), "PYTHONOPTIMIZE=1"),
            "second_env_wrapper": lambda r: replace_child(r, ["/usr/bin/env", "PYTHONOPTIMIZE=1", TORCHRUN, "train.py"]),
            "shell_wrapper": lambda r: replace_child(r, ["/bin/sh", "-c", "PYTHONOPTIMIZE=1 python3 -O train.py"]),
            "python_wrapper": lambda r: replace_child(r, ["/usr/bin/python3", "-O", "train.py"]),
            "torchrun_world8": lambda r: r["command"].__setitem__(r["command"].index("--nproc_per_node=4"), "--nproc_per_node=8"),
            "alternate_train_path": lambda r: r["command"].__setitem__(r["command"].index("train.py"), "research/scratch/train.py"),
            "different_matrix_lr": lambda r: r["command"].__setitem__(-1, "0.025"),
            "extra_training_flag": lambda r: r["command"].append("--nki-relu2"),
            "out_dir_not_label_bound": lambda r: replace_assignment(r, "NEURON_COMPETITION_R1_OUT_DIR", "/tmp/other"),
            "step_log_not_label_bound": lambda r: replace_assignment(r, "NEURON_COMPETITION_R1_STEP_LOG", "/tmp/other"),
            "different_host": lambda r: r.update(host="trn21"),
            "different_label": lambda r: r.update(label="fs-other-a0-label"),
            "different_checkout": lambda r: r.update(remote="/home/ubuntu/trainium_freshstart_other"),
            "request_path_override": lambda r: r["environment"].update(PATH="/tmp/fake:/usr/bin"),
            "request_pythonpath_override": lambda r: r["environment"].update(PYTHONPATH="/tmp/fake"),
            "request_ld_preload_override": lambda r: r["environment"].update(LD_PRELOAD="/tmp/fake.so"),
            "request_pythonoptimize_override": lambda r: r["environment"].update(PYTHONOPTIMIZE="0"),
            "compiler_flags_drift": lambda r: r["environment"].update(NEURON_CC_FLAGS="--optlevel=1 --other"),
            "cache_path_drift": lambda r: r["environment"].update(NEURON_COMPILE_CACHE_DIR="/tmp/fake"),
            "seed_drift": lambda r: r["environment"].update(NEURON_COMPETITION_R1_SEED="43"),
            "extra_fingerprint": lambda r: r["fingerprint_paths"].append("research/agentic/private.json"),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                self.assert_request_refused_before_contacts(mutate)

    def test_absolute_gnu_version_check_fails_before_local_receipt(self):
        argv = ["/usr/bin/timeout", "--version"]
        good = "timeout (GNU coreutils) 9.1\n"
        outcomes = {
            "nonzero": subprocess.CompletedProcess(argv, 1, good, ""),
            "malformed": subprocess.CompletedProcess(argv, 0, "timeout (GNU coreutils)\n", ""),
            "stderr": subprocess.CompletedProcess(argv, 0, good, "synthetic error"),
            "hung": subprocess.TimeoutExpired(argv, 5),
            "missing": FileNotFoundError("synthetic missing GNU timeout"),
        }
        for name, outcome in outcomes.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix="r9-independent-local-") as tmp:
                root = Path(tmp)
                receipt_dir = root / "research/run-logs"
                receipt_dir.mkdir(parents=True)
                request_path = root / "request.json"
                request_path.write_text(json.dumps(fixture.request_template()))
                calls = []

                def fake_run(command, **kwargs):
                    calls.append((command, kwargs))
                    if command != argv:
                        raise AssertionError("FAKE_SSH_WAS_REACHED")
                    if isinstance(outcome, BaseException):
                        raise outcome
                    return outcome

                clean_env = dict(os.environ)
                clean_env.pop("PYTHONOPTIMIZE", None)
                with (patch.object(launcher, "ROOT", root),
                      patch.object(launcher, "HOSTS", {"trn22": "fake-host"}),
                      patch.object(launcher, "SSH", ["fake-ssh"]),
                      patch.object(launcher.subprocess, "run", side_effect=fake_run),
                      patch.object(sys, "flags", fixture.FlagsProxy(sys.flags, 0)),
                      patch.object(sys, "argv", ["fresh_start_launch.py", str(request_path)]),
                      patch.dict(os.environ, clean_env, clear=True)):
                    with self.assertRaises((RuntimeError, FileNotFoundError, subprocess.TimeoutExpired)):
                        launcher.main()
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0], argv)
                self.assertEqual(calls[0][1]["env"], {"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
                self.assertEqual(calls[0][1]["timeout"], 5)
                self.assertEqual(list(receipt_dir.iterdir()), [])

    def test_absolute_gnu_version_check_fails_before_remote_checkout(self):
        argv = ["/usr/bin/timeout", "--version"]
        outcomes = {
            "malformed": subprocess.CompletedProcess(argv, 0, "not GNU\n", ""),
            "hung": subprocess.TimeoutExpired(argv, 5),
            "missing": FileNotFoundError("synthetic missing GNU timeout"),
        }
        for name, outcome in outcomes.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix="r9-independent-remote-") as tmp:
                root = Path(tmp)
                monitor = root / "monitor"
                monitor.mkdir(mode=0o700)
                request = fixture.request_template(root / "missing-checkout", monitor)
                calls = []

                def fake_run(command, **kwargs):
                    calls.append(command)
                    if command != argv:
                        raise AssertionError("UNEXPECTED_REMOTE_PROCESS")
                    if isinstance(outcome, BaseException):
                        raise outcome
                    return outcome

                clean_env = dict(os.environ)
                clean_env.pop("PYTHONOPTIMIZE", None)
                with (patch.object(sys, "stdin", io.StringIO(json.dumps(request))),
                      patch.object(sys, "flags", fixture.FlagsProxy(sys.flags, 0)),
                      patch.dict(os.environ, clean_env, clear=True),
                      patch.object(subprocess, "run", side_effect=fake_run),
                      patch.object(subprocess, "check_output", side_effect=AssertionError("FAKE_GIT_REACHED")),
                      patch.object(subprocess, "Popen", side_effect=AssertionError("FAKE_GUARD_REACHED"))):
                    with self.assertRaises((RuntimeError, FileNotFoundError, subprocess.TimeoutExpired)):
                        exec(compile(launcher.REMOTE_SCRIPT, "independent-r9-remote", "exec", optimize=1),
                             {"__name__": "__main__"})
                self.assertEqual(calls, [argv])
                self.assertEqual(list(monitor.iterdir()), [])
                self.assertFalse((root / "missing-checkout").exists())

    def test_optimization_mode_refuses_before_version_probe(self):
        request = fixture.request_template()
        for name, optimized, setting in (("dash_o", 1, None),
                                         ("inherited_one", 0, "1"),
                                         ("inherited_zero", 0, "0"),
                                         ("inherited_empty", 0, "")):
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix="r9-independent-optimized-") as tmp:
                root = Path(tmp)
                receipt_dir = root / "research/run-logs"
                receipt_dir.mkdir(parents=True)
                request_path = root / "request.json"
                request_path.write_text(json.dumps(request))
                env = dict(os.environ)
                env.pop("PYTHONOPTIMIZE", None)
                if setting is not None:
                    env["PYTHONOPTIMIZE"] = setting
                with (patch.object(launcher, "ROOT", root),
                      patch.object(launcher.subprocess, "run", side_effect=AssertionError("PROCESS_REACHED")),
                      patch.object(sys, "flags", fixture.FlagsProxy(sys.flags, optimized)),
                      patch.object(sys, "argv", ["fresh_start_launch.py", str(request_path)]),
                      patch.dict(os.environ, env, clear=True)):
                    with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                        launcher.main()
                    with patch.object(sys, "stdin", io.StringIO(json.dumps(request))):
                        with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                            exec(compile(launcher.REMOTE_SCRIPT, "independent-r9-optimized-remote", "exec", optimize=1),
                                 {"__name__": "__main__"})
                self.assertEqual(list(receipt_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
