#!/usr/bin/env python3
"""Independent r7-HOLD witness repairs using only fake local/remote contacts.

No SSH, host, guard, training, evaluation, or transfer operation occurs here.
Run this file with normal Python, ``-O``, and ``PYTHONOPTIMIZE=1``.
"""

import ast
import copy
import unittest

import fresh_start_launch as launcher
import test_fresh_start_launch_nonassert as fake


# This is the complete source list from the disabled A0 request, deliberately
# independent of the launcher's own constant.
EXPECTED_FINGERPRINT_PATHS = (
    "train.py", "prepare.py", "research/run_experiment.py",
    "research/fresh_start_launch.py", "research/fresh_start_sync.py",
    "research/fresh_start_guard.py", "research/fresh_start_host.py",
    "research/fresh_start_backup.py", "research/fresh_start_monitor_interlock.py",
    "research/v2/disk-monitor/root_free_guard_monitor.py",
    "research/v2/disk-monitor/paired_root_free_guard_monitor.py",
)
VALID_COMMAND = [
    "timeout", "--foreground", "--signal=TERM", "--kill-after=60s",
    "3600s", "env", "-i", "true",
]


def remote_literal(name):
    tree = ast.parse(launcher.REMOTE_SCRIPT)
    return ast.literal_eval(next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in node.targets)
    ))


class R8BoundedWitnessTests(unittest.TestCase):
    def setUp(self):
        self.local = fake.LocalNonAssertTests(
            "test_valid_unoptimized_request_reaches_only_fake_ssh")
        self.remote = fake.RemoteNonAssertTests(
            "test_valid_unoptimized_remote_reaches_only_fake_guard")

    def request(self):
        request = fake.request_template()
        request["fingerprint_paths"] = list(EXPECTED_FINGERPRINT_PATHS)
        request["command"] = list(VALID_COMMAND)
        return request

    def local_refuses_without_contact(self, request):
        for compiled in (0, 1):
            with self.subTest(compiled=compiled):
                error, contacts, receipts = self.local.run_local(
                    copy.deepcopy(request), compiled_optimize=compiled)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(contacts, [])
                self.assertEqual(receipts, [])

    def remote_refuses_without_writes(self, change):
        for compiled in (0, 1):
            with self.subTest(compiled=compiled):
                error, calls, state = self.remote.run_remote(
                    compiled_optimize=compiled, mutation=change,
                    allow_fake_guard=True)
                self.assertIsInstance(error, RuntimeError)
                self.assertFalse(state["host_dir"])
                self.assertIsNone(state["no_spawn"])
                self.assertFalse(state["manifest"] or state["job"] or state["log"])
                self.assertEqual(calls["git"], [])
                self.assertEqual(calls["preflight"], [])
                self.assertEqual(calls["interlock"], [])
                self.assertEqual(calls["guard"], [])

    def test_allowlist_is_exactly_the_a0_source_set_locally_and_remotely(self):
        self.assertEqual(launcher.SOURCE_FINGERPRINT_PATHS, EXPECTED_FINGERPRINT_PATHS)
        self.assertEqual(remote_literal("SOURCE_FINGERPRINT_PATHS"),
                         EXPECTED_FINGERPRINT_PATHS)
        self.assertFalse(any(isinstance(node, ast.Assert)
                             for node in ast.walk(ast.parse(launcher.REMOTE_SCRIPT))))

    def test_r7_zero_timeout_witness_refuses_before_local_receipt_and_remote_lock(self):
        request = self.request()
        request["command"] = ["timeout", "0", "true"]
        self.local_refuses_without_contact(request)
        self.remote_refuses_without_writes(
            lambda remote, *_: remote.update(command=["timeout", "0", "true"]))

    def test_r7_private_fingerprint_witness_refuses_before_any_contact_or_write(self):
        value = ["research/agentic/private.json"]
        request = self.request()
        request["fingerprint_paths"] = value
        self.local_refuses_without_contact(request)
        self.remote_refuses_without_writes(
            lambda remote, *_: remote.update(fingerprint_paths=value))

    def test_nonpositive_unbounded_or_malformed_timeout_matrix(self):
        commands = {
            "zero_with_actual_flags": ["timeout", "--foreground", "--signal=TERM",
                                       "--kill-after=60s", "0s", "env", "true"],
            "zero_kill_after": ["timeout", "--kill-after=0s", "1s", "true"],
            "no_kill_after": ["timeout", "1s", "true"],
            "negative": ["timeout", "--kill-after=60s", "-1s", "true"],
            "nan": ["timeout", "--kill-after=60s", "NaN", "true"],
            "infinite": ["timeout", "--kill-after=60s", "inf", "true"],
            "huge": ["timeout", "--kill-after=60s", "99999999d", "true"],
            "exponent": ["timeout", "--kill-after=60s", "1e309s", "true"],
            "missing_duration": ["timeout", "--kill-after=60s", "true"],
            "missing_child": ["timeout", "--kill-after=60s", "1s"],
            "empty_child": ["timeout", "--kill-after=60s", "1s", ""],
            "help": ["timeout", "--help", "1s", "true"],
            "help_as_child": ["timeout", "--kill-after=60s", "1s", "--help"],
            "unknown_option": ["timeout", "--preserve-status", "--kill-after=60s",
                               "1s", "true"],
            "duplicate_option": ["timeout", "--kill-after=60s", "--kill-after=30s",
                                 "1s", "true"],
            "child_optimize": ["timeout", "--kill-after=60s", "1s", "env",
                               "PYTHONOPTIMIZE=1", "true"],
            "nul_arg": ["timeout", "--kill-after=60s", "1s", "true\x00"],
        }
        for name, command in commands.items():
            with self.subTest(command=name):
                request = self.request()
                request["command"] = command
                self.local_refuses_without_contact(request)
                self.remote_refuses_without_writes(
                    lambda remote, *_: remote.update(command=command))

    def test_missing_extra_duplicate_or_nonsource_fingerprint_matrix(self):
        expected = list(EXPECTED_FINGERPRINT_PATHS)
        variants = {
            "absent": None,
            "missing_one": expected[:-1],
            "extra_private": expected + ["research/agentic/private.json"],
            "duplicate": expected[:-1] + [expected[0]],
            "scratch_directory": expected[:-1] + ["research/scratch"],
            "traversal": expected[:-1] + ["research/../prepare.py"],
            "absolute": expected[:-1] + ["/tmp/private.json"],
            "nonstr": expected[:-1] + [42],
        }
        for name, value in variants.items():
            with self.subTest(fingerprint=name):
                request = self.request()
                request["fingerprint_paths"] = value
                self.local_refuses_without_contact(request)
                self.remote_refuses_without_writes(
                    lambda remote, *_: remote.update(fingerprint_paths=value))

    def test_positive_a0_shape_reaches_only_fakes(self):
        for compiled in (0, 1):
            with self.subTest(compiled=compiled):
                error, contacts, receipts = self.local.run_local(
                    self.request(), compiled_optimize=compiled)
                self.assertIsNone(error)
                self.assertEqual(len(contacts), 1)
                self.assertEqual(len(receipts), 1)
                error, calls, state = self.remote.run_remote(
                    compiled_optimize=compiled,
                    mutation=lambda remote, *_: remote.update(
                        command=list(VALID_COMMAND)),
                    allow_fake_guard=True)
                self.assertIsNone(error)
                self.assertEqual(len(calls["guard"]), 1)
                self.assertTrue(state["host_dir"] and state["manifest"] and state["job"])

    def test_actual_optimization_flags_and_environment_refuse_before_contact(self):
        for name, kwargs in (("dash_o", {"runtime_optimize": 1}),
                             ("dash_oo", {"runtime_optimize": 2}),
                             ("pythonoptimize", {"pythonoptimize": "1"}),
                             ("pythonoptimize_zero", {"pythonoptimize": "0"}),
                             ("pythonoptimize_empty", {"pythonoptimize": ""})):
            with self.subTest(mode=name):
                error, contacts, receipts = self.local.run_local(self.request(), **kwargs)
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("optimized Python", str(error))
                self.assertEqual(contacts, [])
                self.assertEqual(receipts, [])
                error, calls, state = self.remote.run_remote(**kwargs)
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("optimized Python", str(error))
                self.assertFalse(state["host_dir"])
                self.assertIsNone(state["no_spawn"])
                self.assertEqual(calls["git"], [])
                self.assertEqual(calls["guard"], [])


if __name__ == "__main__":
    unittest.main()
