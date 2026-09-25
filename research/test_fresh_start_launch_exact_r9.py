#!/usr/bin/env python3
"""Offline r9 A0 argv/timeout regressions; all contacts are local fakes."""

import copy
from pathlib import Path
import tempfile
import unittest

import fresh_start_launch as launcher
import test_fresh_start_launch_nonassert as fake


TORCHRUN = "/home/ubuntu/workspace/native_venv/bin/torchrun"


def child_as(request, child):
    command = request["command"]
    command[command.index(TORCHRUN):] = child


class ExactA0R9Tests(unittest.TestCase):
    def setUp(self):
        self.local = fake.LocalNonAssertTests(
            "test_valid_unoptimized_request_reaches_only_fake_ssh")
        self.remote = fake.RemoteNonAssertTests(
            "test_valid_unoptimized_remote_reaches_only_fake_guard")

    def assert_early_refusal(self, mutate):
        for compiled in (0, 1):
            with self.subTest(compiled=compiled):
                request = fake.request_template()
                mutate(request)
                error, contacts, receipts = self.local.run_local(
                    request, compiled_optimize=compiled)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(contacts, [])
                self.assertEqual(receipts, [])
                error, calls, state = self.remote.run_remote(
                    compiled_optimize=compiled,
                    mutation=lambda remote, *_: mutate(remote),
                    allow_fake_guard=True)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(calls["timeout_version"], [])
                self.assertEqual(calls["git"], [])
                self.assertEqual(calls["preflight"], [])
                self.assertEqual(calls["guard"], [])
                self.assertFalse(state["host_dir"])
                self.assertIsNone(state["no_spawn"])
                self.assertFalse(state["manifest"] or state["job"] or state["log"])

    def test_valid_a0_argv_reaches_only_fake_contact(self):
        frozen = copy.deepcopy(fake.A0_REQUEST)
        self.assertIs(frozen["dispatch_authorized"], False)
        self.assertFalse(launcher.bounded_timeout_command(
            frozen["command"], frozen["host"], frozen["label"], frozen["remote"],
            frozen["environment"]))
        frozen["command"] = fake.a0_command(frozen["label"], frozen["remote"])
        self.assertTrue(launcher.bounded_timeout_command(
            frozen["command"], frozen["host"], frozen["label"], frozen["remote"],
            frozen["environment"]))
        frozen["dispatch_authorized"] = True  # in-memory fake-contact probe only
        error, contacts, receipts = self.local.run_local(frozen)
        self.assertIsNone(error)
        self.assertEqual(len(contacts), 1)
        self.assertEqual(len(receipts), 1)

        request = fake.request_template()
        self.assertTrue(launcher.bounded_timeout_command(
            request["command"], request["host"], request["label"], request["remote"],
            request["environment"]))
        self.assertEqual(request["command"][:7], [
            "/usr/bin/timeout", "--foreground", "--signal=TERM",
            "--kill-after=60s", "3600s", "/usr/bin/env", "-i"])
        self.assertEqual(request["command"][request["command"].index(TORCHRUN):],
                         [TORCHRUN, "--standalone", "--nproc_per_node=4",
                          "train.py", "--n-embd", "1024", "--no-ngram-ve",
                          "--no-qk-shift", "--compile-sdpa-direct",
                          "--nki-local-conv", "--bf16-norm-output",
                          "--pack-factor", "32", "--clip-after-reduce",
                          "--seq-len", "1024", "--no-eval-public",
                          "--matrix-lr", "0.015"])
        for compiled in (0, 1):
            with self.subTest(compiled=compiled):
                error, contacts, receipts = self.local.run_local(
                    copy.deepcopy(request), compiled_optimize=compiled)
                self.assertIsNone(error)
                self.assertEqual(len(contacts), 1)
                self.assertEqual(len(receipts), 1)
                error, calls, state = self.remote.run_remote(
                    compiled_optimize=compiled, allow_fake_guard=True)
                self.assertIsNone(error)
                self.assertEqual(calls["timeout_version"],
                                 [["/usr/bin/timeout", "--version"]])
                self.assertEqual(len(calls["guard"]), 1)
                self.assertTrue(state["host_dir"] and state["manifest"] and state["job"])

    def test_old_relative_timeout_and_env_names_are_not_legal(self):
        for name, index, value in (("relative_timeout", 0, "timeout"),
                                   ("relative_env", 5, "env"),
                                   ("fake_timeout", 0, "/tmp/fake-timeout")):
            with self.subTest(name=name):
                self.assert_early_refusal(
                    lambda request: request["command"].__setitem__(index, value))

    def test_request_path_cannot_select_fake_timeout_or_child_path(self):
        with tempfile.TemporaryDirectory(prefix="r9-fake-timeout-", dir="/dev/shm") as tmp:
            fake_timeout = Path(tmp) / "timeout"
            fake_timeout.write_text("#!/bin/sh\nprintf 'NO_DEADLINE\\n'\n")
            fake_timeout.chmod(0o700)
            hostile_path = tmp + ":/usr/bin:/bin"
            self.assert_early_refusal(
                lambda request: request["environment"].update(PATH=hostile_path))
            self.assert_early_refusal(
                lambda request: request["command"].__setitem__(8,
                    "PATH=" + hostile_path))

    def test_shell_python_and_extra_assignment_bypasses_are_not_legal(self):
        children = (
            ["sh", "-c", "export PYTHONOPTIMIZE=1; python3 -O train.py"],
            ["/bin/bash", "-lc", "PYTHONOPTIMIZE=1 python3 train.py"],
            ["python3", "-O", "train.py"],
            ["env", "PYTHONOPTIMIZE=1", TORCHRUN, "train.py"],
            [TORCHRUN, "--standalone", "--nproc_per_node=8", "train.py"],
        )
        for child in children:
            with self.subTest(child=child):
                self.assert_early_refusal(lambda request: child_as(request, list(child)))
        self.assert_early_refusal(
            lambda request: request["command"].insert(
                request["command"].index(TORCHRUN), "PYTHONOPTIMIZE=1"))
        self.assert_early_refusal(
            lambda request: request["environment"].update(PYTHONPATH="/tmp/fake"))
        self.assert_early_refusal(
            lambda request: request["environment"].update(PYTHONOPTIMIZE="1"))

    def test_command_is_bound_to_label_checkout_and_exact_training_profile(self):
        changes = (
            lambda request: request.update(host="trn21"),
            lambda request: request.update(label="fs-other-a0-label"),
            lambda request: request.update(
                remote="/home/ubuntu/trainium_freshstart_other_checkout"),
            lambda request: request["command"].__setitem__(-1, "0.025"),
            lambda request: request["environment"].update(
                NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS="9999"),
        )
        for change in changes:
            with self.subTest(change=change):
                self.assert_early_refusal(change)

    def test_non_gnu_absolute_timeout_refuses_before_receipt_or_remote_lock(self):
        for compiled in (0, 1):
            with self.subTest(compiled=compiled):
                error, contacts, receipts = self.local.run_local(
                    fake.request_template(), compiled_optimize=compiled,
                    timeout_banner="FAKE_TIMEOUT_NO_DEADLINE\n")
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("GNU timeout", str(error))
                self.assertEqual(contacts, [])
                self.assertEqual(receipts, [])
                error, calls, state = self.remote.run_remote(
                    compiled_optimize=compiled, allow_fake_guard=True,
                    timeout_banner="FAKE_TIMEOUT_NO_DEADLINE\n")
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("GNU timeout", str(error))
                self.assertEqual(calls["timeout_version"],
                                 [["/usr/bin/timeout", "--version"]])
                self.assertEqual(calls["git"], [])
                self.assertEqual(calls["guard"], [])
                self.assertFalse(state["host_dir"])
                self.assertIsNone(state["no_spawn"])
                self.assertFalse(state["manifest"] or state["job"] or state["log"])


if __name__ == "__main__":
    unittest.main()
