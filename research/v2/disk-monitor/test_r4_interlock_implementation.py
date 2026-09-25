#!/usr/bin/env python3
"""Local r4 containment tests; all process signals and clocks are synthetic."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))

import test_root_free_guard_monitor as fixtures
import test_paired_root_free_guard_monitor as pairfixtures
import root_free_guard_monitor as base
import fresh_start_guard as guard_host
import fresh_start_host as host


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
        # until all three authenticated processes have disappeared; the
        # unrelated same-UID post-child process is HOLD-only, never signaled.
        system.time += 0.25
        self.assertIsNone(monitor.tick())
        for pid in (worker_pid, case.child_pid, case.guard_pid):
            system.procs[pid] = None
        for _ in range(130):
            system.time += 0.25
            result = monitor.tick()
            if result is not None:
                break
        self.assertEqual(result, 3)
        self.assertEqual(monitor.state, "HOLD")
        self.assertNotIn(("KILL", foreign_pid + 100000), system.signals)
        self.assertIn("UNRESOLVED_HOLD", [event for event, _ in case.journal.events])

    def test_markerless_worker_in_live_child_session_is_reenumerated_and_killed(self):
        case = self.base_case()
        monitor, system = case.monitor, case.system
        system.free = case.binding["stop_free_bytes"]
        system.time += 0.25
        self.assertIsNone(monitor.tick())
        worker_pid = 12348
        system.procs[worker_pid] = base.ProcInfo(
            worker_pid, os.getuid(), 88006, case.child_pid,
            case.child_pid, case.child_pid, ("/usr/bin/env", "-i", "worker"),
            "/usr/bin/env")
        self.assertNotIn(worker_pid, system.owned_pids)
        for _ in range(500):
            if monitor.state == "KILLING":
                break
            system.time += 0.25
            self.assertIsNone(monitor.tick())
        self.assertEqual(monitor.state, "KILLING")
        self.assertIn(("KILL", worker_pid + 100000), system.signals)
        self.assertIn(("KILL", case.child_pid + 100000), system.signals)
        self.assertIn(("KILL", case.guard_pid + 100000), system.signals)

    def test_markerless_session_worker_surviving_guard_child_blocks_clean_finish(self):
        case = self.base_case()
        monitor, system = case.monitor, case.system
        worker_pid = 12349
        system.procs[worker_pid] = base.ProcInfo(
            worker_pid, os.getuid(), 88007, 1, worker_pid, case.child_pid,
            ("/usr/bin/sleep", "3600"), "/usr/bin/sleep")
        self.assertNotIn(worker_pid, system.owned_pids)
        system.procs[case.guard_pid] = None
        system.procs[case.child_pid] = None
        system.files[monitor.postflight_path] = {
            "label": case.binding["label"], "host_clean": True,
            "command_returncode": 0,
        }
        system.time += 0.25
        self.assertIsNone(monitor.tick())
        self.assertEqual(monitor.state, "KILLING")
        self.assertNotIn(("KILL", worker_pid + 100000), system.signals)
        for _ in range(130):
            system.time += 0.25
            result = monitor.tick()
            if result is not None:
                break
        self.assertEqual(result, 3)
        self.assertEqual(monitor.state, "HOLD")
        self.assertIn("UNPROVEN_PROCESS_HOLD", [event for event, _ in case.journal.events])
        self.assertIn("UNRESOLVED_HOLD", [event for event, _ in case.journal.events])

    def test_changed_session_orphan_with_marker_is_hold_only(self):
        case = self.base_case()
        monitor, system = case.monitor, case.system
        orphan_pid = 12350
        system.procs[orphan_pid] = base.ProcInfo(
            orphan_pid, os.getuid(), 88008, 1, orphan_pid, orphan_pid,
            ("/usr/bin/worker",), "/usr/bin/worker")
        system.owned_pids.add(orphan_pid)
        system.procs[case.guard_pid] = None
        system.procs[case.child_pid] = None
        system.files[monitor.postflight_path] = {
            "label": case.binding["label"], "host_clean": True,
            "command_returncode": 0,
        }
        system.time += 0.25
        self.assertIsNone(monitor.tick())
        self.assertEqual(monitor.state, "KILLING")
        self.assertNotIn(("KILL", orphan_pid + 100000), system.signals)
        for _ in range(130):
            system.time += 0.25
            result = monitor.tick()
            if result is not None:
                break
        self.assertEqual(result, 3)
        self.assertEqual(monitor.state, "HOLD")
        self.assertIn("UNPROVEN_PROCESS_HOLD", [event for event, _ in case.journal.events])

    def test_markerless_changed_session_orphan_at_low_free_blocks_base_clean(self):
        case = self.base_case()
        monitor, system = case.monitor, case.system
        orphan_pid = 12352
        system.procs[orphan_pid] = base.ProcInfo(
            orphan_pid, os.getuid(), case.child.birth_ticks + 2,
            1, orphan_pid, orphan_pid,
            ("/usr/bin/sleep", "900"), "/usr/bin/sleep")
        self.assertNotIn(orphan_pid, system.owned_pids)
        system.procs[case.guard_pid] = None
        system.procs[case.child_pid] = None
        system.files[monitor.postflight_path] = {
            "label": case.binding["label"], "host_clean": True,
            "command_returncode": 0,
        }
        system.free = case.binding["stop_free_bytes"] - 1
        system.time += 0.25
        self.assertIsNone(monitor.tick())
        self.assertEqual(monitor.state, "KILLING")
        self.assertNotIn(("KILL", orphan_pid + 100000), system.signals)
        for _ in range(130):
            system.time += 0.25
            result = monitor.tick()
            if result is not None:
                break
        self.assertEqual(result, 3)
        self.assertIn("UNRESOLVED_HOLD", [event for event, _ in case.journal.events])

    def test_markerless_changed_session_orphan_at_low_free_blocks_pair_clean(self):
        case = self.paired_case()
        monitor, system = case.monitor, case.system
        orphan_pid = 12353
        system.procs[orphan_pid] = base.ProcInfo(
            orphan_pid, os.getuid(), monitor.child.birth_ticks + 2,
            1, orphan_pid, orphan_pid,
            ("/usr/bin/sleep", "900"), "/usr/bin/sleep")
        system.procs[monitor.guard.pid] = None
        system.procs[monitor.child.pid] = None
        system.files[monitor.postflight_path] = {
            "label": case.binding["label"], "host_clean": True,
            "command_returncode": 0,
        }
        system.free = case.binding["stop_free_bytes"] - 1
        system.time += 0.25
        case.sample["sample_monotonic"] = system.time
        self.assertIsNone(monitor.tick())
        self.assertEqual(monitor.state, "KILLING")
        self.assertNotIn(("KILL", orphan_pid + 100000), system.signals)
        for _ in range(130):
            system.time += 0.25
            case.sample["sample_monotonic"] = system.time
            result = monitor.tick()
            if result is not None:
                break
        self.assertEqual(result, 3)
        self.assertIn("UNRESOLVED_HOLD", [event for event, _ in case.journal.events])

    def test_replaced_worker_pid_between_inventory_and_pidfd_is_not_killed(self):
        case = self.base_case()
        monitor, system = case.monitor, case.system
        worker_pid = 12351
        worker = base.ProcInfo(
            worker_pid, os.getuid(), 88009, case.child_pid,
            case.child_pid, case.child_pid,
            ("/usr/bin/worker",), "/usr/bin/worker")
        system.procs[worker_pid] = worker
        old_open_pidfd = system.open_pidfd

        def open_with_reuse(pid):
            pidfd = old_open_pidfd(pid)
            if pid == worker_pid:
                system.procs[worker_pid] = base.ProcInfo(
                    worker_pid, os.getuid(), worker.birth_ticks + 1, 1,
                    worker_pid, worker_pid,
                    ("/usr/bin/unrelated",), "/usr/bin/unrelated")
            return pidfd

        system.open_pidfd = open_with_reuse
        monitor._escalate_kill("synthetic worker replacement")
        self.assertEqual(monitor.state, "KILLING")
        self.assertNotIn(("KILL", worker_pid + 100000), system.signals)
        self.assertIn(("KILL", case.child_pid + 100000), system.signals)
        self.assertIn(("KILL", case.guard_pid + 100000), system.signals)

    def test_same_uid_marker_spoof_outside_child_session_is_never_killed(self):
        case = self.base_case()
        monitor, system = case.monitor, case.system
        spoof_pid = 45679
        system.procs[spoof_pid] = base.ProcInfo(
            spoof_pid, os.getuid(), 88005, 1, spoof_pid, spoof_pid,
            ("/usr/bin/unrelated",), "/usr/bin/unrelated")
        system.owned_pids.add(spoof_pid)  # Exact public marker, unrelated session.
        system.free = case.binding["stop_free_bytes"]
        system.time += 0.25
        self.assertIsNone(monitor.tick())
        for _ in range(500):
            if monitor.state == "KILLING":
                break
            system.time += 0.25
            self.assertIsNone(monitor.tick())
        self.assertEqual(monitor.state, "KILLING")
        self.assertNotIn(("KILL", spoof_pid + 100000), system.signals)
        self.assertIn(("KILL", case.guard_pid + 100000), system.signals)
        self.assertIn(("KILL", case.child_pid + 100000), system.signals)
        for pid in (case.guard_pid, case.child_pid):
            system.procs[pid] = None
        for _ in range(130):
            system.time += 0.25
            result = monitor.tick()
            if result is not None:
                break
        self.assertEqual(result, 3)
        self.assertEqual(monitor.state, "HOLD")
        self.assertNotIn(("KILL", spoof_pid + 100000), system.signals)
        self.assertIn("UNPROVEN_PROCESS_HOLD", [event for event, _ in case.journal.events])
        self.assertIn("UNRESOLVED_HOLD", [event for event, _ in case.journal.events])

    def test_host_cleanup_never_signals_same_uid_marker_spoof(self):
        spoof_pid = 45679
        spoof = {"pid": spoof_pid, "start_ticks": 88005,
                 "session": spoof_pid, "state": "S", "argv": ["unrelated"]}
        with (patch.object(host, "owned_processes", return_value=[spoof]),
              patch.object(host, "session_processes", return_value=[]),
              patch.object(host.os, "kill", side_effect=AssertionError("numeric signal"))):
            result = host.cleanup_owned("fs-synthetic-marker-spoof", 12346)
        self.assertEqual(result["actions"], [])
        self.assertEqual(result["remaining"], [spoof])
        self.assertEqual(result["signal_policy"], "inventory_only_unproven_ownership_hold")

    def test_guard_and_host_have_no_numeric_process_signal_calls(self):
        for module in (host, guard_host):
            tree = ast.parse(Path(module.__file__).read_text())
            numeric_calls = [node for node in ast.walk(tree)
                             if isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Attribute)
                             and isinstance(node.func.value, ast.Name)
                             and node.func.value.id == "os"
                             and node.func.attr in ("kill", "killpg")]
            self.assertEqual(numeric_calls, [], module.__name__)

    def test_missing_no_spawn_proof_announces_bounded_hold_and_keeps_rescuing(self):
        case = pairfixtures.PairMonitorTest(
            methodName="test_no_spawn_receipt_ends_pre_guard_watch_without_signal")
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.bind_peer()
        receipt_path = Path(case.pair["no_spawn_path"])
        receipt = {
            "schema": "paired-monitor-no-spawn-v1", "status": "NO_GUARD_SPAWNED",
            "label": case.binding["label"],
            "expected_commit": case.binding["expected_commit"],
            "pair_binding_path": str(case.pair_path),
        }
        original_sleep = case.system.sleep
        inserted = [False]

        def late_receipt(seconds):
            original_sleep(seconds)
            if (not inserted[0]
                    and case.system.time >= 0.25 + case.binding["bind_timeout_seconds"]):
                inserted[0] = True
                receipt_path.write_text(json.dumps(receipt) + "\n")
                receipt_path.chmod(0o600)
                case.system.files[receipt_path] = receipt

        case.system.sleep = late_receipt
        self.assertEqual(case.monitor.fail_closed(base.MonitorError("peer died")), 3)
        self.assertTrue(inserted[0])
        self.assertEqual(case.system.signals, [])
        self.assertIn("RESCUE_UNRESOLVED_HOLD",
                      [event for event, _ in case.journal.events])

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
