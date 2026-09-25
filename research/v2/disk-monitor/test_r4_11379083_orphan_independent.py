#!/usr/bin/env python3
"""Synthetic negative proof for a markerless worker that leaves the child session."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))

import test_root_free_guard_monitor as fixtures
import test_paired_root_free_guard_monitor as pairfixtures


class ChangedSessionOrphanIndependentTest(unittest.TestCase):
    def test_markerless_changed_session_orphan_prevents_clean_finish(self):
        case = fixtures.GuardMonitorTest(
            methodName="test_preemptive_threshold_signals_exact_pidfd_once"
        )
        case.setUp()
        case.stage_live_guard()
        monitor, system = case.monitor, case.system

        # A run descendant may start a new session and exec with a sanitized
        # environment. The old marker and child-session inventories then both
        # miss it, even though it remains alive after guard and child exit.
        orphan_pid = 12350
        system.procs[orphan_pid] = fixtures.monitor_module.ProcInfo(
            orphan_pid, os.getuid(), case.child.birth_ticks + 2,
            1, orphan_pid, orphan_pid,
            ("/usr/bin/sleep", "900"), "/usr/bin/sleep",
        )
        self.assertNotIn(orphan_pid, system.owned_pids)
        system.procs[case.guard_pid] = None
        system.procs[case.child_pid] = None
        system.files[monitor.postflight_path] = {
            "label": case.binding["label"],
            "host_clean": True,
            "command_returncode": 0,
        }
        system.free = case.binding["stop_free_bytes"] - 1
        system.time += 0.25
        self.assertNotEqual(monitor.tick(), 0,
                            "live markerless changed-session orphan was certified clean")

    def test_paired_monitor_inherits_changed_session_false_clean(self):
        case = pairfixtures.PairMonitorTest(
            methodName="test_peer_death_stops_only_authenticated_guard"
        )
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.bind_peer()
        guard_pid, child_pid = case.stage_guard()
        case.system.time = 0.50
        case.sample.update(sample_monotonic=0.50, state="BOUND", guard_pid=guard_pid)
        self.assertIsNone(case.monitor.tick())
        self.assertEqual(case.monitor.state, "BOUND")

        orphan_pid = 12351
        case.system.procs[orphan_pid] = fixtures.monitor_module.ProcInfo(
            orphan_pid, os.getuid(), 100004, 1, orphan_pid, orphan_pid,
            ("/usr/bin/sleep", "900"), "/usr/bin/sleep",
        )
        case.system.procs[guard_pid] = None
        case.system.procs[child_pid] = None
        case.system.files[case.monitor.postflight_path] = {
            "label": case.binding["label"],
            "host_clean": True,
            "command_returncode": 0,
        }
        case.system.free = case.binding["stop_free_bytes"] - 1
        case.system.time = 0.75
        case.sample["sample_monotonic"] = 0.75
        self.assertNotEqual(case.monitor.tick(), 0,
                            "paired monitor certified a live changed-session orphan clean")


if __name__ == "__main__":
    unittest.main()
