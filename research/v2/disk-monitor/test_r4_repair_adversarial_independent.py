#!/usr/bin/env python3
"""Independent synthetic interlock challenges; never sends a real signal."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fresh_start_host as host
import root_free_guard_monitor as base
import test_root_free_guard_monitor as fixtures


class R4RepairAdversarialIndependentTest(unittest.TestCase):
    @staticmethod
    def base_case():
        case = fixtures.GuardMonitorTest(
            methodName="test_preemptive_threshold_signals_exact_pidfd_once"
        )
        case.setUp()
        case.stage_live_guard()
        return case

    def test_same_uid_marker_spoof_in_other_session_never_signaled(self):
        case = self.base_case()
        monitor, system = case.monitor, case.system
        outsider_pid = 45679
        system.procs[outsider_pid] = base.ProcInfo(
            outsider_pid, os.getuid(), case.child.birth_ticks + 2,
            1, outsider_pid, outsider_pid,
            ("/usr/bin/unrelated",), "/usr/bin/unrelated",
        )
        system.owned_pids.add(outsider_pid)
        system.free = case.binding["stop_free_bytes"]
        system.time += 0.25
        self.assertIsNone(monitor.tick())
        for _ in range(500):
            if monitor.state != "STOPPING":
                break
            system.time += 0.25
            self.assertIsNone(monitor.tick())
        self.assertEqual(monitor.state, "KILLING")
        self.assertNotIn(("KILL", outsider_pid + 100000), system.signals)
        self.assertNotIn(outsider_pid + 100000, system.signals)

    def test_live_same_session_worker_without_marker_prevents_clean_finish(self):
        case = self.base_case()
        monitor, system = case.monitor, case.system
        worker_pid = 12347
        # A descendant can exec with a sanitized environment, while retaining
        # its kernel session.  The child's session remains allocated here.
        system.procs[worker_pid] = base.ProcInfo(
            worker_pid, os.getuid(), case.child.birth_ticks + 1,
            case.child_pid, case.child_pid, case.child_pid,
            ("/usr/bin/sleep", "900"), "/usr/bin/sleep",
        )
        self.assertNotIn(worker_pid, system.owned_pids)
        system.procs[case.guard_pid] = None
        system.procs[case.child_pid] = None
        system.files[monitor.postflight_path] = {
            "label": case.binding["label"],
            "host_clean": True,
            "command_returncode": 0,
        }
        system.time += 0.25
        result = monitor.tick()
        self.assertNotEqual(
            result, 0,
            "a live same-session worker with stripped marker must not yield clean finish",
        )

    def test_guard_cleanup_rejects_unrelated_same_uid_marker(self):
        outsider_pid = 45679
        outsider = {
            "pid": outsider_pid,
            "start_ticks": 88005,
            "argv": ["/usr/bin/unrelated"],
        }
        would_signal = []

        def marked_processes(_label):
            # An unrelated same-UID process has independently copied the
            # public marker. After any attempted signal, make the synthetic
            # process disappear so cleanup remains instant and bounded.
            return [outsider] if not would_signal else []

        with patch.object(host, "owned_processes", marked_processes), \
             patch.object(host, "process_record", lambda _pid: outsider), \
             patch.object(host.os, "kill", lambda pid, sig: would_signal.append((pid, sig))):
            host.cleanup_owned("fs-synthetic-a0-r4")

        self.assertEqual(
            would_signal, [],
            "a marker without guard/child provenance must not authorize cleanup signal",
        )


if __name__ == "__main__":
    unittest.main()
