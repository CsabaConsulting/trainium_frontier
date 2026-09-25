#!/usr/bin/env python3
"""Independent synthetic proof: a run marker alone must not convey ownership.

No real process is spawned or signaled by this test.  The fake system records
would-be pidfd signals so the old r4 source can fail safely.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))

import root_free_guard_monitor as monitor_module
import test_root_free_guard_monitor as fixtures


class MarkerSpoofIndependentTest(unittest.TestCase):
    def test_unrelated_same_uid_marker_cannot_be_killed(self):
        case = fixtures.GuardMonitorTest(
            methodName="test_preemptive_threshold_signals_exact_pidfd_once"
        )
        case.setUp()
        case.stage_live_guard()
        monitor, system = case.monitor, case.system

        # The marker is a normal same-UID environment variable, not an
        # unforgeable descendant credential.  This process belongs to a
        # different session/process group and has never been a child of the
        # bound guard or run child.
        spoof_pid = 45679
        system.procs[spoof_pid] = monitor_module.ProcInfo(
            spoof_pid, os.getuid(), case.child.birth_ticks + 2,
            1, spoof_pid, spoof_pid,
            ("/usr/bin/unrelated",), "/usr/bin/unrelated",
        )
        system.owned_pids.add(spoof_pid)

        # Trigger the disk stop; the synthetic guard/child ignore TERM, so
        # the monitor must escalate.  The unrelated marker-bearer must never
        # receive an owned pidfd signal at any point.
        system.free = case.binding["stop_free_bytes"]
        system.time += 0.25
        self.assertIsNone(monitor.tick())
        for _ in range(500):
            if monitor.state != "STOPPING":
                break
            system.time += 0.25
            self.assertIsNone(monitor.tick())
        self.assertEqual(monitor.state, "KILLING")
        self.assertNotIn(
            ("KILL", spoof_pid + 100000), system.signals,
            "a same-UID marker alone cannot authenticate a run descendant",
        )


if __name__ == "__main__":
    unittest.main()
