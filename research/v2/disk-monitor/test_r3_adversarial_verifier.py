#!/usr/bin/env python3
"""Independent static/synthetic safety probes for the disabled r3 package.

These tests assert the observed unsafe outcome to keep the failure reproducible.
They do not launch a host job or send a real signal.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))

import test_root_free_guard_monitor as fixtures
import test_paired_root_free_guard_monitor as pairfixtures


class R3AdversarialVerifierTest(unittest.TestCase):
    def test_source_drift_after_pidfd_binding_refuses_disk_stop(self):
        case = fixtures.GuardMonitorTest(methodName="test_preemptive_threshold_signals_exact_pidfd_once")
        case.setUp()
        case.stage_live_guard()
        monitor, system = case.monitor, case.system
        self.assertIsNotNone(monitor.guard_pidfd)
        self.assertEqual(monitor.state, "BOUND")

        # No guard, child, job, or process identity changes. Only a source hash
        # drifts after the running process has already been authenticated.
        system.hashes[Path(case.binding["remote"]) / "train.py"] = "changed-after-bind"
        system.free = case.binding["stop_free_bytes"]
        system.time += 0.25

        with self.assertRaisesRegex(Exception, "source bytes changed") as raised:
            monitor.tick()
        self.assertEqual(type(raised.exception).__name__, "MonitorError")
        self.assertEqual(monitor.fail_closed(raised.exception), 3)
        self.assertEqual(system.signals, [])
        self.assertEqual(monitor.state, "BOUND")
        self.assertIn("SIGNAL_REFUSED", [name for name, _ in case.journal.events])

    def test_paired_wrapper_inherits_the_same_bound_stop_refusal(self):
        case = pairfixtures.PairMonitorTest(methodName="test_peer_death_stops_only_authenticated_guard")
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.bind_peer()
        guard_pid, _ = case.stage_guard()
        case.sample.update(sample_monotonic=0.50, state="BOUND", guard_pid=guard_pid)
        case.system.time = 0.50
        self.assertIsNone(case.monitor.tick())
        self.assertEqual(case.monitor.state, "BOUND")

        case.system.hashes[Path(case.binding["remote"]) / "train.py"] = "changed-after-bind"
        case.system.free = case.binding["stop_free_bytes"]
        case.system.time = 0.75
        case.sample["sample_monotonic"] = 0.75
        with self.assertRaisesRegex(Exception, "source bytes changed") as raised:
            case.monitor.tick()
        self.assertEqual(type(raised.exception).__name__, "MonitorError")
        self.assertEqual(case.monitor.fail_closed(raised.exception), 3)
        self.assertEqual(case.system.signals, [])
        self.assertIn("SIGNAL_REFUSED", [name for name, _ in case.journal.events])

    def test_unresponsive_authenticated_guard_outlives_monitor_timeout(self):
        case = fixtures.GuardMonitorTest(methodName="test_preemptive_threshold_signals_exact_pidfd_once")
        case.setUp()
        case.stage_live_guard()
        monitor, system = case.monitor, case.system
        system.free = case.binding["stop_free_bytes"]
        system.time += 0.25
        self.assertIsNone(monitor.tick())
        self.assertEqual(monitor.state, "STOPPING")
        self.assertEqual(system.signals, [case.guard_pid + 100000])

        # The authenticated guard and child remain live despite TERM. The
        # monitor does not wait indefinitely or perform another contained stop.
        for _ in range(500):
            system.time += 0.25
            try:
                monitor.tick()
            except fixtures.monitor_module.MonitorError as error:
                self.assertIn("did not finish after TERM", str(error))
                self.assertEqual(monitor.fail_closed(error), 3)
                break
        else:
            self.fail("unresponsive guard never reached post-stop timeout")
        self.assertEqual(monitor.state, "STOPPING")
        self.assertIsNotNone(system.proc(case.guard_pid))
        self.assertIsNotNone(system.proc(case.child_pid))
        self.assertEqual(system.signals, [case.guard_pid + 100000])


if __name__ == "__main__":
    unittest.main()
