"""Independent, offline-only adversarial checks for disabled trn22 r9 stage r2.

Only disposable local files are written. There is no SSH or host contact.
"""

from contextlib import nullcontext
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock


SOURCE = (Path(__file__).resolve().parents[1] / "authorizations" /
          "trn22-c5c-a0-r9-incremental-stage-r2-hold" / "remote_incremental.py")
SPEC = importlib.util.spec_from_file_location("trn22_stage_r2_independent", SOURCE)
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)


class LastBoundaryTests(unittest.TestCase):
    def exercise(self, arrival=None, delay=0.0, open_delay=0.0):
        with tempfile.TemporaryDirectory(prefix="trn22-r2-independent-") as dirname:
            root = Path(dirname)
            bundle = root / "exact.bundle"
            target = root / "new-checkout"
            stop = root / "STOP"
            label = root / ("late-" + remote.LABEL)
            payload = b"synthetic-bundle-content"
            state = {"finds": 0, "activities": 0, "process": False,
                     "neuron": False, "disk_low": False,
                     "open_delay_elapsed": 0.0}
            snapshot = {"bundle_exists": False, "target_exists": False,
                        "source": {"commit": "frozen"}}

            def stop_process():
                state["activities"] += 1
                if stop.exists():
                    raise RuntimeError("late STOP observed")
                if state["process"]:
                    raise RuntimeError("late training process observed")

            def command(argv, **_kwargs):
                if argv[0] == "/usr/bin/fuser":
                    if arrival == "stop_in_fuser":
                        stop.touch()
                    if arrival == "disk_during_fuser":
                        state["disk_low"] = True
                    return types.SimpleNamespace(returncode=1, stdout=b"", stderr=b"")
                if argv[0] == "neuron-ls":
                    output = (b'[{"neuron_processes":[{"command":"train.py"}]}]'
                              if state["neuron"] else b"[]")
                    return types.SimpleNamespace(returncode=0, stdout=output, stderr=b"")
                if argv[0] == "/usr/bin/find":
                    state["finds"] += 1
                    if delay and state["finds"] == 1:
                        time.sleep(delay)
                    seen = label.exists()
                    if arrival == "label_after_first_find" and state["finds"] == 1:
                        label.touch()
                    if arrival == "stop_after_first_find" and state["finds"] == 1:
                        stop.touch()
                    if arrival == "process_after_first_find" and state["finds"] == 1:
                        state["process"] = True
                    if arrival == "neuron_after_first_find" and state["finds"] == 1:
                        state["neuron"] = True
                    output = (str(label) + "\n").encode() if seen else b""
                    return types.SimpleNamespace(returncode=0, stdout=output, stderr=b"")
                raise AssertionError("unexpected subprocess command: " + repr(argv))

            real_open = os.open

            def delayed_open(path, flags, *args, **kwargs):
                if path == bundle.name and flags & os.O_CREAT and open_delay:
                    before = time.monotonic()
                    time.sleep(open_delay)
                    state["open_delay_elapsed"] = time.monotonic() - before
                return real_open(path, flags, *args, **kwargs)

            def space(_minimum):
                if state["disk_low"]:
                    raise RuntimeError("late root disk floor observed")
                return {}

            with mock.patch.object(remote, "ROOT", root), \
                 mock.patch.object(remote, "BUNDLE", bundle), \
                 mock.patch.object(remote, "TARGET", target), \
                 mock.patch.object(remote, "STOP_PATHS", [stop]), \
                 mock.patch.object(remote, "BUNDLE_BYTES", len(payload)), \
                 mock.patch.object(remote, "BUNDLE_SHA256", hashlib.sha256(payload).hexdigest()), \
                 mock.patch.object(remote, "_host_lock", return_value=nullcontext({})), \
                 mock.patch.object(remote, "_probe", return_value=snapshot), \
                 mock.patch.object(remote, "_stable"), \
                 mock.patch.object(remote, "_stop_and_process", side_effect=stop_process), \
                 mock.patch.object(remote.subprocess, "run", side_effect=command), \
                 mock.patch.object(remote, "_checked_bundle", return_value={}), \
                 mock.patch.object(remote, "_reconstruct", return_value={}), \
                 mock.patch.object(remote, "_source", return_value=snapshot["source"]), \
                 mock.patch.object(remote, "_space", side_effect=space), \
                 mock.patch.object(remote.os, "open", side_effect=delayed_open):
                try:
                    receipt = remote.perform("write", prior={"synthetic": True}, data=payload)
                    error = None
                except RuntimeError as exc:
                    receipt = None
                    error = str(exc)
            return {"bundle": bundle.exists(), "target": target.exists(),
                    "receipt": receipt, "error": error, **state}

    def test_stop_during_fuser_refuses_before_first_write(self):
        result = self.exercise("stop_in_fuser")
        self.assertIn("STOP", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_stop_after_first_label_scan_refuses_before_first_write(self):
        result = self.exercise("stop_after_first_find")
        self.assertIn("STOP", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_label_after_first_scan_refuses_before_first_write(self):
        result = self.exercise("label_after_first_find")
        self.assertIn("label", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_1_1_second_scan_refuses_before_first_write(self):
        result = self.exercise(delay=1.1)
        self.assertIn("too old", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_fast_clean_fixture_reaches_only_local_synthetic_write(self):
        result = self.exercise()
        self.assertIsNone(result["error"])
        self.assertTrue(result["bundle"])
        self.assertTrue(result["target"])
        self.assertLessEqual(result["receipt"]["final_no_write_continuity"]
                             ["activity_age_seconds"], 1.0)

    def test_new_training_process_during_final_scan_reaches_exclusive_write(self):
        result = self.exercise("process_after_first_find")
        self.assertIn("late training process", result["error"])
        self.assertTrue(result["bundle"], result)
        self.assertTrue(result["target"], result)

    def test_new_neuron_holder_during_final_scan_reaches_exclusive_write(self):
        result = self.exercise("neuron_after_first_find")
        self.assertIn("Neuron process active", result["error"])
        self.assertTrue(result["bundle"], result)
        self.assertTrue(result["target"], result)

    def test_disk_floor_loss_during_final_activity_reaches_exclusive_write(self):
        result = self.exercise("disk_during_fuser")
        self.assertIn("late root disk floor", result["error"])
        self.assertTrue(result["bundle"], result)
        self.assertTrue(result["target"], result)

    def test_scheduler_delay_after_freshness_gate_makes_open_stale(self):
        result = self.exercise(open_delay=1.1)
        self.assertIsNone(result["error"])
        self.assertGreaterEqual(result["open_delay_elapsed"], 1.1)
        self.assertTrue(result["bundle"], result)
        self.assertTrue(result["target"], result)
        self.assertLessEqual(result["receipt"]["final_no_write_continuity"]
                             ["activity_age_seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()
