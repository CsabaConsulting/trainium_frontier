"""Offline boundary regressions for the disabled trn21 r10 stage successor.

Disposable local files only. Passing here is Implementer evidence, not an
independent review, host preflight, source-stage assignment or native gate.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remote = load("trn21_r10_stage_r2_boundary", "remote_incremental.py")
operator = load("trn21_r10_stage_r2_operator_boundary", "operator_once.py")


class FinalBoundaryTests(unittest.TestCase):
    def exercise(self, event=None):
        with tempfile.TemporaryDirectory(prefix="trn21-stage-r2-", dir="/dev/shm") as dirname:
            root = Path(dirname)
            bundle, target = root / "exact.bundle", root / "new-checkout"
            stop, label = root / "STOP", root / ("late-" + remote.LABEL)
            protected = root / "protected-checkpoint.pt"
            protected.write_bytes(b"untouched protected bytes")
            protected_inode = protected.stat().st_ino
            payload = b"synthetic exact incremental bundle"
            state = {"scan_count": 0, "activity_count": 0,
                     "busy": None, "disk_low": False, "open_elapsed": 0.0}

            def arrive(suffix):
                if event == "stop_" + suffix:
                    stop.touch()
                elif event == "label_" + suffix:
                    label.touch()
                elif event == "process_" + suffix:
                    state["busy"] = "process"
                elif event == "neuron_" + suffix:
                    state["busy"] = "Neuron"
                elif event == "disk_" + suffix:
                    state["disk_low"] = True

            def activity():
                state["activity_count"] += 1
                if stop.exists():
                    raise RuntimeError("synthetic STOP active")
                if state["busy"]:
                    raise RuntimeError("synthetic " + state["busy"] + " active")
                return {"stops_absent": True, "neuron_holder_absent": True}

            def label_scan():
                state["scan_count"] += 1
                if label.exists():
                    raise RuntimeError("synthetic stage label already appears")
                arrive("after_first_scan" if state["scan_count"] == 1 else
                       "after_second_scan" if state["scan_count"] == 2 else "none")
                return {"matches": []}

            def space(_minimum):
                if state["disk_low"]:
                    raise RuntimeError("synthetic root byte/inode floor failed")
                return {"root_free_bytes": remote.PRE_FREE + 1024,
                        "root_free_inodes": remote.MIN_INODES + 1}

            original_final = remote._final_no_write_continuity
            original_open = os.open

            def final_then_arrive():
                samples = original_final()
                arrive("after_helper")
                return samples

            def exclusive_open(path, flags, *args, **kwargs):
                if path == bundle.name and flags & os.O_EXCL:
                    started = time.monotonic()
                    fd = original_open(path, flags, *args, **kwargs)
                    arrive("after_open")
                    if event == "mutate_open":
                        os.write(fd, b"unexpected")
                    if event == "late_open":
                        time.sleep(1.05)
                    state["open_elapsed"] = time.monotonic() - started
                    return fd
                return original_open(path, flags, *args, **kwargs)

            # Deliberately reject deletion. A failed post-open check must keep
            # its zero-byte provisional bundle and leave the checkout absent.
            with mock.patch.object(remote, "ROOT", root), \
                 mock.patch.object(remote, "BUNDLE", bundle), \
                 mock.patch.object(remote, "TARGET", target), \
                 mock.patch.object(remote, "STOP_PATHS", [stop]), \
                 mock.patch.object(remote, "BUNDLE_BYTES", len(payload)), \
                 mock.patch.object(remote, "BUNDLE_SHA256", hashlib.sha256(payload).hexdigest()), \
                 mock.patch.object(remote, "_activity", side_effect=activity), \
                 mock.patch.object(remote, "_label_scan", side_effect=label_scan), \
                 mock.patch.object(remote, "_space", side_effect=space), \
                 mock.patch.object(remote, "_final_no_write_continuity",
                                   side_effect=final_then_arrive), \
                 mock.patch.object(remote, "_checked_bundle", return_value={"checked": True}), \
                 mock.patch.object(remote.os, "open", side_effect=exclusive_open), \
                 mock.patch.object(remote.os, "unlink",
                                   side_effect=AssertionError("stage cleanup is forbidden")):
                try:
                    proof = remote._write_bundle(payload)
                    error = None
                except RuntimeError as exc:
                    proof, error = None, str(exc)
            return {"bundle_exists": bundle.exists(),
                    "bundle_bytes": bundle.read_bytes() if bundle.exists() else None,
                    "target_exists": target.exists(), "proof": proof,
                    "error": error, "protected_bytes": protected.read_bytes(),
                    "protected_inode": protected.stat().st_ino,
                    "protected_inode_before": protected_inode, **state}

    def test_clean_fixture_has_real_open_and_post_open_proof(self):
        result = self.exercise()
        self.assertIsNone(result["error"])
        self.assertEqual(result["scan_count"], 2)
        self.assertEqual(result["activity_count"], 3)
        self.assertTrue(result["bundle_exists"])
        self.assertTrue(result["target_exists"])
        proof = result["proof"]
        self.assertIs(proof["pre_open_broad_label_scan"], True)
        self.assertIs(proof["post_label_activity_check"], True)
        self.assertIs(proof["post_open_broad_label_scan"], True)
        self.assertIs(proof["post_open_label_and_activity_check"], True)
        operator._checked_final_gate(proof)

    def test_stop_during_final_scan_refuses_before_exclusive_creation(self):
        result = self.exercise("stop_after_first_scan")
        self.assertIsNotNone(result["error"])
        self.assertFalse(result["bundle_exists"])
        self.assertFalse(result["target_exists"])

    def test_label_during_final_scan_retains_only_empty_provisional(self):
        result = self.exercise("label_after_first_scan")
        self.assertIn("stage label already appears", result["error"])
        self.assertEqual(result["scan_count"], 2)
        self.assertTrue(result["bundle_exists"])
        self.assertEqual(result["bundle_bytes"], b"")
        self.assertFalse(result["target_exists"])
        self.assertEqual(result["protected_bytes"], b"untouched protected bytes")
        self.assertEqual(result["protected_inode"], result["protected_inode_before"])

    def test_noncooperating_label_after_last_scan_is_explicit_residual(self):
        # This is an adversarial limitation witness, not a safety PASS. The
        # same race exists after r3's last post-open scan. A final reviewer
        # must accept the bounded cooperative-flock model before enablement.
        result = self.exercise("label_after_second_scan")
        self.assertIsNone(result["error"])
        self.assertTrue(result["bundle_exists"])
        self.assertTrue(result["target_exists"])
        self.assertEqual(result["bundle_bytes"], b"synthetic exact incremental bundle")

    def test_changes_after_post_open_scan_leave_empty_provisional(self):
        for event in ("process_after_second_scan", "neuron_after_second_scan",
                      "disk_after_second_scan"):
            with self.subTest(event=event):
                result = self.exercise(event)
                self.assertIsNotNone(result["error"])
                self.assertTrue(result["bundle_exists"])
                self.assertEqual(result["bundle_bytes"], b"")
                self.assertFalse(result["target_exists"])

    def test_helper_to_open_changes_leave_empty_partial_and_hold(self):
        for event in ("stop_after_helper", "label_after_helper",
                      "process_after_helper", "neuron_after_helper",
                      "disk_after_helper"):
            with self.subTest(event=event):
                result = self.exercise(event)
                self.assertIsNotNone(result["error"])
                self.assertTrue(result["bundle_exists"])
                self.assertEqual(result["bundle_bytes"], b"")
                self.assertFalse(result["target_exists"])

    def test_changes_at_actual_open_leave_empty_partial_and_hold(self):
        for event in ("stop_after_open", "label_after_open",
                      "process_after_open", "neuron_after_open",
                      "disk_after_open"):
            with self.subTest(event=event):
                result = self.exercise(event)
                self.assertIsNotNone(result["error"])
                self.assertTrue(result["bundle_exists"])
                self.assertEqual(result["bundle_bytes"], b"")
                self.assertFalse(result["target_exists"])

    def test_actual_open_over_one_second_is_not_success(self):
        result = self.exercise("late_open")
        self.assertGreaterEqual(result["open_elapsed"], 1.05)
        self.assertIn("too old", result["error"])
        self.assertTrue(result["bundle_exists"])
        self.assertEqual(result["bundle_bytes"], b"")
        self.assertFalse(result["target_exists"])

    def test_nonempty_provisional_file_is_preserved_and_holds(self):
        result = self.exercise("mutate_open")
        self.assertIn("provisional bundle changed", result["error"])
        self.assertEqual(result["bundle_bytes"], b"unexpected")
        self.assertFalse(result["target_exists"])

    def test_forged_or_nonfinite_final_evidence_is_refused(self):
        proof = self.exercise()["proof"]
        for name, value in (("open_call_elapsed_seconds", float("nan")),
                            ("post_open_space_age_seconds", float("inf")),
                            ("label_scan_duration_seconds", float("nan")),
                            ("pre_open_broad_label_scan", False),
                            ("post_open_broad_label_scan", False),
                            ("post_label_activity_check", False),
                            ("root_space_after_open", {"root_free_bytes": 1,
                                                       "root_free_inodes": 0})):
            with self.subTest(name=name):
                altered = dict(proof)
                altered[name] = value
                with self.assertRaises(RuntimeError):
                    operator._checked_final_gate(altered)

    def test_contradictory_final_chronology_is_refused(self):
        proof = self.exercise()["proof"]
        for older, newer in (("label_age_seconds", "activity_age_seconds"),
                             ("activity_age_seconds", "space_age_seconds"),
                             ("space_age_seconds", "open_call_elapsed_seconds"),
                             ("post_open_label_age_seconds", "post_open_activity_age_seconds"),
                             ("post_open_activity_age_seconds", "post_open_space_age_seconds")):
            with self.subTest(older=older, newer=newer):
                altered = dict(proof)
                altered[older] = 0.0
                altered[newer] = 0.5
                with self.assertRaisesRegex(RuntimeError, "contradictory"):
                    operator._checked_final_gate(altered)

    def test_local_stop_can_block_full_readonly_postflight_before_ssh(self):
        with mock.patch.object(operator, "_local_stop_and_breakers",
                               side_effect=RuntimeError("synthetic local STOP")), \
             mock.patch.object(operator, "_ssh_identity") as identity, \
             mock.patch.object(operator.subprocess, "run") as process:
            with self.assertRaisesRegex(RuntimeError, "synthetic local STOP"):
                operator._ssh_phase("postflight", None, b"", b"remote source")
            identity.assert_not_called()
            process.assert_not_called()

    def test_partial_write_then_local_stop_records_postflight_hold(self):
        if sys.flags.optimize:
            # The real entry point refuses optimized Python before a marker.
            with self.assertRaises(RuntimeError):
                operator.stage_once(remote_call=mock.Mock())
            return
        with tempfile.TemporaryDirectory(prefix="trn21-stage-r2-marker-",
                                         dir="/dev/shm") as dirname:
            root = Path(dirname)
            marker = root / "one-attempt.jsonl"

            def remote_call(phase, _prior, _data, _source):
                if phase == "preflight":
                    return {"phase": phase, "snapshot": {
                        "bundle_exists": False, "target_exists": False}}
                if phase == "write":
                    raise RuntimeError("synthetic partial write")
                if phase == "postflight":
                    raise RuntimeError("local STOP blocked read-only SSH")
                raise AssertionError("unexpected phase")

            with mock.patch.object(operator, "ENABLED_REQUEST", root / "enabled.json"), \
                 mock.patch.object(operator, "MARKER", marker), \
                 mock.patch.object(operator, "_json", side_effect=[
                     ({"dispatch_authorized": True}, "request-hash"),
                     ({}, "review-hash"), ({}, "assignment-hash")]), \
                 mock.patch.object(operator, "_source_bytes", return_value=(b"bundle", b"remote")), \
                 mock.patch.object(operator, "_regular", return_value=b"operator"), \
                 mock.patch.object(operator, "_authority"), \
                 mock.patch.object(operator, "_local_stop_and_breakers"), \
                 mock.patch.object(operator, "_ssh_identity"), \
                 mock.patch.object(operator, "_remote_timeout_identity"):
                with self.assertRaisesRegex(RuntimeError, "marker and all partials retained"):
                    operator.stage_once(remote_call=remote_call)
            events = [json.loads(line)["event"] for line in marker.read_text().splitlines()]
            self.assertEqual(events, ["ONE_ATTEMPT_CONSUMED", "FRESH_PREFLIGHT",
                                      "STAGE_HOLD", "POSTFLIGHT_HOLD"])


if __name__ == "__main__":
    unittest.main()
