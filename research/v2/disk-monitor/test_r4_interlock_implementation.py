#!/usr/bin/env python3
"""Local r4 containment tests; all process signals and clocks are synthetic."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))

import test_root_free_guard_monitor as fixtures
import test_paired_root_free_guard_monitor as pairfixtures
import root_free_guard_monitor as base


class R4InterlockImplementationTest(unittest.TestCase):
    def base_case(self):
        case = fixtures.GuardMonitorTest(
            methodName="test_preemptive_threshold_signals_exact_pidfd_once")
        case.setUp()
        case.stage_live_guard()
        return case

    def paired_case(self):
        case = pairfixtures.PairMonitorTest(
            methodName="test_peer_death_stops_only_authenticated_guard")
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.bind_peer()
        guard_pid, _ = case.stage_guard()
        case.sample.update(sample_monotonic=0.50, state="BOUND", guard_pid=guard_pid)
        case.system.time = 0.50
        self.assertIsNone(case.monitor.tick())
        self.assertEqual(case.monitor.state, "BOUND")
        return case

    def assert_read_failure_stops(self, case):
        target = Path(case.binding["remote"]) / "train.py"
        original = case.system.sha256

        def unreadable(path):
            if path == target:
                raise OSError("synthetic source read failure")
            return original(path)

        case.system.sha256 = unreadable
        case.system.time += 0.25
        if hasattr(case, "sample"):
            case.sample["sample_monotonic"] = case.system.time
        with self.assertRaisesRegex(OSError, "source read failure") as raised:
            case.monitor.tick()
        self.assertEqual(case.monitor.fail_closed(raised.exception), 3)
        self.assertEqual(case.system.signals[0], case.monitor.guard_pidfd)
        self.assertNotIn("SIGNAL_REFUSED", [event for event, _ in case.journal.events])
        self.assertIn("UNRESOLVED_HOLD", [event for event, _ in case.journal.events])

    def assert_hash_drift_stops(self, case):
        case.system.hashes[Path(case.binding["remote"]) / "train.py"] = "changed"
        case.system.time += 0.25
        if hasattr(case, "sample"):
            case.sample["sample_monotonic"] = case.system.time
        with self.assertRaisesRegex(Exception, "source bytes changed") as raised:
            case.monitor.tick()
        self.assertEqual(case.monitor.fail_closed(raised.exception), 3)
        self.assertEqual(case.system.signals[0], case.monitor.guard_pidfd)
        self.assertNotIn("SIGNAL_REFUSED", [event for event, _ in case.journal.events])

    def test_base_low_free_with_source_hash_drift_still_terms_bound_guard(self):
        case = self.base_case()
        case.system.hashes[Path(case.binding["remote"]) / "train.py"] = "changed"
        case.system.free = case.binding["stop_free_bytes"]
        case.system.time += 0.25
        self.assertIsNone(case.monitor.tick())
        self.assertEqual(case.system.signals, [case.monitor.guard_pidfd])
        self.assertEqual(case.monitor.state, "STOPPING")

    def test_base_source_read_failure_after_bind_stops_bound_guard(self):
        self.assert_read_failure_stops(self.base_case())

    def test_base_hash_drift_above_stop_floor_triggers_owned_stop(self):
        self.assert_hash_drift_stops(self.base_case())

    def test_base_low_free_source_read_failure_does_not_veto_term(self):
        case = self.base_case()
        target = Path(case.binding["remote"]) / "train.py"
        original = case.system.sha256
        def unreadable(path):
            if path == target:
                raise OSError("synthetic source read failure")
            return original(path)
        case.system.sha256 = unreadable
        case.system.free = case.binding["stop_free_bytes"]
        case.system.time += 0.25
        self.assertIsNone(case.monitor.tick())
        self.assertEqual(case.system.signals, [case.monitor.guard_pidfd])

    def test_paired_low_free_with_source_hash_drift_still_terms_bound_guard(self):
        case = self.paired_case()
        case.system.hashes[Path(case.binding["remote"]) / "train.py"] = "changed"
        case.system.free = case.binding["stop_free_bytes"]
        case.system.time += 0.25
        case.sample["sample_monotonic"] = case.system.time
        self.assertIsNone(case.monitor.tick())
        self.assertEqual(case.system.signals, [case.monitor.guard_pidfd])
        self.assertEqual(case.monitor.state, "STOPPING")

    def test_paired_source_read_failure_after_bind_stops_bound_guard(self):
        self.assert_read_failure_stops(self.paired_case())

    def test_paired_hash_drift_above_stop_floor_triggers_owned_stop(self):
        self.assert_hash_drift_stops(self.paired_case())

    def test_paired_low_free_source_read_failure_does_not_veto_term(self):
        case = self.paired_case()
        target = Path(case.binding["remote"]) / "train.py"
        original = case.system.sha256
        def unreadable(path):
            if path == target:
                raise OSError("synthetic source read failure")
            return original(path)
        case.system.sha256 = unreadable
        case.system.free = case.binding["stop_free_bytes"]
        case.system.time += 0.25
        case.sample["sample_monotonic"] = case.system.time
        self.assertIsNone(case.monitor.tick())
        self.assertEqual(case.system.signals, [case.monitor.guard_pidfd])

    def test_source_drift_before_binding_refuses_new_guard_ownership(self):
        case = fixtures.GuardMonitorTest(
            methodName="test_preemptive_threshold_signals_exact_pidfd_once")
        case.setUp()
        case.system.files[case.monitor.job_path] = case.job
        case.system.files[case.monitor.process_path] = case.process
        case.system.procs[case.guard_pid] = case.guard
        case.system.procs[case.child_pid] = case.child
        case.system.hashes[Path(case.binding["remote"]) / "train.py"] = "changed"
        case.system.time += 0.25
        with self.assertRaisesRegex(Exception, "source bytes changed") as raised:
            case.monitor.tick()
        self.assertEqual(case.monitor.fail_closed(raised.exception), 3)
        self.assertEqual(case.system.pidfds, [])
        self.assertEqual(case.system.signals, [])

    def test_reused_child_pid_is_never_killed_at_escalation(self):
        case = self.base_case()
        original = case.child
        case.system.procs[case.child_pid] = base.ProcInfo(
            original.pid, original.uid, original.birth_ticks + 1,
            1, original.pgrp, original.session, original.argv, original.exe)
        case.system.free = case.binding["stop_free_bytes"]
        case.system.time += 0.25
        self.assertIsNone(case.monitor.tick())
        for _ in range(500):
            if case.monitor.state == "KILLING":
                break
            case.system.time += 0.25
            self.assertIsNone(case.monitor.tick())
        self.assertEqual(case.monitor.state, "KILLING")
        self.assertNotIn(("KILL", case.monitor.child_pidfd), case.system.signals)
        self.assertIn(("KILL", case.monitor.guard_pidfd), case.system.signals)

    def test_late_worker_is_reenumerated_and_killed_by_authenticated_pidfd(self):
        case = self.base_case()
        monitor, system = case.monitor, case.system
        foreign_pid = 45678
        system.procs[foreign_pid] = base.ProcInfo(
            foreign_pid, os.getuid(), 88004, 1, foreign_pid, foreign_pid,
            ("/usr/bin/unrelated",), "/usr/bin/unrelated")
        system.free = case.binding["stop_free_bytes"]
        system.time += 0.25
        self.assertIsNone(monitor.tick())

        # The worker appears only after TERM. It has the exact inherited run
        # marker and is discovered during the later KILL-stage scan.
        worker_pid = 12347
        system.procs[worker_pid] = base.ProcInfo(
            worker_pid, os.getuid(), 88003, case.child_pid,
            case.child_pid, case.child_pid, ("/usr/bin/worker",), "/usr/bin/worker")
        system.owned_pids.add(worker_pid)
        for _ in range(500):
            if monitor.state != "STOPPING":
                break
            system.time += 0.25
            self.assertIsNone(monitor.tick())
        self.assertEqual(monitor.state, "KILLING")
        self.assertIn(("KILL", worker_pid + 100000), system.signals)
        self.assertIn(("KILL", case.child_pid + 100000), system.signals)
        self.assertIn(("KILL", case.guard_pid + 100000), system.signals)
        self.assertNotIn(("KILL", foreign_pid + 100000), system.signals)

        # KILL may not take effect on the same poll. The monitor remains live
        # until all three authenticated processes have actually disappeared.
        system.time += 0.25
        self.assertIsNone(monitor.tick())
        for pid in (worker_pid, case.child_pid, case.guard_pid):
            system.procs[pid] = None
        system.time += 0.25
        self.assertEqual(monitor.tick(), 3)
        self.assertEqual(monitor.state, "HOLD")
        self.assertIn("QUIESCENT_HOLD", [event for event, _ in case.journal.events])

    def test_term_and_kill_ignoring_guard_child_end_in_explicit_unresolved_hold(self):
        case = self.base_case()
        case.system.free = case.binding["stop_free_bytes"]
        result = case.monitor.run()
        self.assertEqual(result, 3)
        self.assertIn(case.monitor.guard_pidfd, case.system.signals)
        self.assertIn(("KILL", case.monitor.guard_pidfd), case.system.signals)
        self.assertIn(("KILL", case.monitor.child_pidfd), case.system.signals)
        self.assertIsNotNone(case.system.proc(case.guard_pid))
        self.assertIsNotNone(case.system.proc(case.child_pid))
        self.assertEqual(case.monitor.state, "HOLD")
        self.assertIn("UNRESOLVED_HOLD", [event for event, _ in case.journal.events])
        self.assertIn(case.monitor.guard_pidfd, case.system.closed)
        self.assertIn(case.monitor.child_pidfd, case.system.closed)

    def test_paired_term_ignoring_guard_and_child_escalates_and_holds(self):
        case = self.paired_case()
        case.system.free = case.binding["stop_free_bytes"]
        for _ in range(650):
            if case.monitor.state == "HOLD":
                break
            case.system.time += 0.25
            case.sample["sample_monotonic"] = case.system.time
            result = case.monitor.tick()
            if result is not None:
                self.assertEqual(result, 3)
                break
        self.assertIn(case.monitor.guard_pidfd, case.system.signals)
        self.assertIn(("KILL", case.monitor.guard_pidfd), case.system.signals)
        self.assertIn(("KILL", case.monitor.child_pidfd), case.system.signals)
        self.assertIn("UNRESOLVED_HOLD", [event for event, _ in case.journal.events])


if __name__ == "__main__":
    unittest.main()
