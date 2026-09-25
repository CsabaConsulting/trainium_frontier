#!/usr/bin/env python3
"""Independent offline counterexamples for the e74f999 launcher source gate.

Every contact and process signal in this file is synthetic. The tests assert
observed behavior, including unsafe acceptance, so the HOLD is reproducible.
"""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

RESEARCH = Path(__file__).parents[2]
DISK_MONITOR = RESEARCH / "v2/disk-monitor"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(DISK_MONITOR))

import fresh_start_launch as launcher
import test_fresh_start_launch_nonassert as supplied_fixture
import test_root_free_guard_monitor as base_fixtures
import test_paired_root_free_guard_monitor as pair_fixtures

SOURCE_SHA256 = "ece127ad02f72871341369736732f9b324bad9ffa865bc503def7732bb8ad3ab"
FLOOR = 13 * 1024**3 // 2
NORMAL_MODE = sys.flags.optimize == 0 and "PYTHONOPTIMIZE" not in os.environ


def request():
    return {
        "dispatch_authorized": True,
        "host": "trn22",
        "expected_hostname": "synthetic-hostname",
        "label": "fs-independent-r7-source-gate",
        "expected_commit": "a" * 40,
        "remote": "/home/ubuntu/trainium_freshstart_independent_r7",
        "command": ["timeout", "1s", "true"],
        "environment": {"NEURON_LOGICAL_NC_CONFIG": "2",
                        "NEURON_CC_FLAGS": "--optlevel=1"},
        "min_free_bytes": FLOOR,
        "pre_launch_required_free_bytes": 10 * 1024**3,
        "monitor_interlock": {
            "pair_binding_path": "/tmp/synthetic-monitor/pair.json",
            "interlock_path": "/tmp/synthetic-monitor/interlock.json",
            "no_spawn_path": "/tmp/synthetic-monitor/no-spawn.json",
        },
    }


class FlagsProxy:
    def __init__(self, original, optimize):
        self.original = original
        self.optimize = optimize

    def __getattr__(self, name):
        return getattr(self.original, name)


class IndependentR7SourceGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        actual = hashlib.sha256((RESEARCH / "fresh_start_launch.py").read_bytes()).hexdigest()
        if actual != SOURCE_SHA256:
            raise AssertionError(f"subject launcher bytes changed: {actual}")

    def local_probe(self, change, *, optimized=False, pythonoptimize=False):
        with tempfile.TemporaryDirectory(prefix="r7-verifier-fake-contact-") as directory:
            root = Path(directory)
            receipts = root / "research/run-logs"
            receipts.mkdir(parents=True)
            payload = request()
            change(payload)
            request_path = root / "request.json"
            request_path.write_text(json.dumps(payload))
            contacts = []

            def fake_ssh(*args, **kwargs):
                contacts.append(args[0])
                return launcher.subprocess.CompletedProcess(args[0], 0, "FAKE_ONLY\n", "")

            env = {"PATH": os.environ.get("PATH", "")}
            if pythonoptimize:
                env["PYTHONOPTIMIZE"] = "1"
            with (mock.patch.object(launcher, "ROOT", root),
                  mock.patch.object(launcher, "HOSTS", {"trn22": "fake-host"}),
                  mock.patch.object(launcher, "SSH", ["fake-ssh"]),
                  mock.patch.object(launcher.subprocess, "run", side_effect=fake_ssh),
                  mock.patch.object(sys, "argv", ["fake-launch", str(request_path)]),
                  mock.patch.object(sys, "flags", FlagsProxy(sys.flags, int(optimized))),
                  mock.patch.dict(os.environ, env, clear=True),
                  contextlib.redirect_stdout(io.StringIO())):
                try:
                    launcher.main()
                except BaseException as error:
                    outcome = error
                else:
                    outcome = None
            return outcome, contacts, [p.name for p in receipts.iterdir()]

    @unittest.skipUnless(NORMAL_MODE, "normal-mode counterexample")
    def test_local_untrusted_fingerprint_reaches_fake_ssh_and_receipt(self):
        outcome, contacts, receipts = self.local_probe(
            lambda value: value.update(fingerprint_paths=["research/agentic/private.json"]))
        self.assertIsNone(outcome)
        self.assertEqual(len(contacts), 1)
        self.assertEqual(receipts, ["fs-independent-r7-source-gate.dispatch.json"])

    @unittest.skipUnless(NORMAL_MODE, "normal-mode counterexample")
    def test_local_zero_duration_reaches_fake_ssh_and_receipt(self):
        outcome, contacts, receipts = self.local_probe(
            lambda value: value.update(command=["timeout", "0", "true"]))
        self.assertIsNone(outcome)
        self.assertEqual(len(contacts), 1)
        self.assertEqual(receipts, ["fs-independent-r7-source-gate.dispatch.json"])

    def test_local_authority_and_optimization_controls_fail_before_contact(self):
        for name, change, options in (
            ("nonboolean_authority", lambda value: value.update(dispatch_authorized=1), {}),
            ("child_optimize", lambda value: value["environment"].update(PYTHONOPTIMIZE="1"), {}),
            ("runtime_optimize", lambda value: None, {"optimized": True}),
            ("inherited_optimize", lambda value: None, {"pythonoptimize": True}),
        ):
            with self.subTest(name=name):
                outcome, contacts, receipts = self.local_probe(change, **options)
                self.assertIsInstance(outcome, RuntimeError)
                self.assertEqual(contacts, [])
                self.assertEqual(receipts, [])

    @unittest.skipUnless(NORMAL_MODE, "normal-mode counterexample")
    def test_remote_zero_duration_reaches_only_fake_guard(self):
        fake = supplied_fixture.RemoteNonAssertTests(
            "test_valid_unoptimized_remote_reaches_only_fake_guard")
        error, calls, state = fake.run_remote(
            mutation=lambda value, *_: value.update(command=["timeout", "0", "true"]),
            allow_fake_guard=True)
        self.assertIsNone(error)
        self.assertEqual(len(calls["guard"]), 1)
        self.assertTrue(state["host_dir"] and state["manifest"] and state["job"])

    @unittest.skipUnless(NORMAL_MODE, "normal-mode counterexample")
    def test_remote_bad_fingerprint_writes_lock_and_no_spawn_before_refusal(self):
        fake = supplied_fixture.RemoteNonAssertTests(
            "test_valid_unoptimized_remote_reaches_only_fake_guard")
        error, calls, state = fake.run_remote(
            mutation=lambda value, *_: value.update(
                fingerprint_paths=["research/agentic/private.json"]),
            allow_fake_guard=True)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("invalid source fingerprint paths", str(error))
        self.assertEqual(calls["guard"], [])
        self.assertTrue(state["host_dir"])
        self.assertIsNotNone(state["no_spawn"])

    def test_remote_authority_and_optimization_controls_fail_before_writes(self):
        fake = supplied_fixture.RemoteNonAssertTests(
            "test_valid_unoptimized_remote_reaches_only_fake_guard")
        for name, kwargs in (
            ("nonboolean_authority", {"mutation": lambda value, *_: value.update(dispatch_authorized=1)}),
            ("runtime_optimize", {"runtime_optimize": 1}),
            ("inherited_optimize", {"pythonoptimize": "1"}),
        ):
            with self.subTest(name=name):
                error, calls, state = fake.run_remote(**kwargs)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(calls["guard"], [])
                self.assertFalse(state["host_dir"] or state["manifest"] or state["job"])
                self.assertIsNone(state["no_spawn"])

    def test_three_historical_r3_failures_have_safe_r4_outcomes(self):
        base = base_fixtures.GuardMonitorTest(
            methodName="test_preemptive_threshold_signals_exact_pidfd_once")
        base.setUp()
        base.stage_live_guard()
        base.system.hashes[Path(base.binding["remote"]) / "train.py"] = "changed-after-bind"
        base.system.free = base.binding["stop_free_bytes"]
        base.system.time += 0.25
        self.assertIsNone(base.monitor.tick())
        self.assertEqual(base.monitor.state, "STOPPING")
        self.assertEqual(base.system.signals, [base.monitor.guard_pidfd])

        pair = pair_fixtures.PairMonitorTest(
            methodName="test_peer_death_stops_only_authenticated_guard")
        pair.setUp()
        self.addCleanup(pair.doCleanups)
        pair.bind_peer()
        guard_pid, _ = pair.stage_guard()
        pair.sample.update(sample_monotonic=0.50, state="BOUND", guard_pid=guard_pid)
        pair.system.time = 0.50
        self.assertIsNone(pair.monitor.tick())
        pair.system.hashes[Path(pair.binding["remote"]) / "train.py"] = "changed-after-bind"
        pair.system.free = pair.binding["stop_free_bytes"]
        pair.system.time = 0.75
        pair.sample["sample_monotonic"] = 0.75
        self.assertIsNone(pair.monitor.tick())
        self.assertEqual(pair.monitor.state, "STOPPING")
        self.assertEqual(pair.system.signals, [pair.monitor.guard_pidfd])

        stuck = base_fixtures.GuardMonitorTest(
            methodName="test_preemptive_threshold_signals_exact_pidfd_once")
        stuck.setUp()
        stuck.stage_live_guard()
        stuck.system.free = stuck.binding["stop_free_bytes"]
        self.assertEqual(stuck.monitor.run(), 3)
        self.assertEqual(stuck.monitor.state, "HOLD")
        self.assertIn(stuck.monitor.guard_pidfd, stuck.system.signals)
        self.assertIn(("KILL", stuck.monitor.guard_pidfd), stuck.system.signals)
        self.assertIn(("KILL", stuck.monitor.child_pidfd), stuck.system.signals)


if __name__ == "__main__":
    unittest.main()
