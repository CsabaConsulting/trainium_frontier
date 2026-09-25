"""Offline r3 regressions adapted from independent r2 HOLD witness c5c9a8bd.

Only disposable local paths are used; there is no SSH, AWS or host contact.
The actual activity, label, exclusive open and write flow remain in scope.
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


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("trn22_stage_r3_boundary", HERE / "remote_incremental.py")
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)


class FinalBoundaryTests(unittest.TestCase):
    def exercise(self, arrival=None, *, open_delay=0.0, mutate_open=False):
        with tempfile.TemporaryDirectory(prefix="trn22-r3-boundary-") as dirname:
            root = Path(dirname)
            bundle = root / "exact.bundle"
            target = root / "new-checkout"
            stop = root / "STOP"
            label = root / ("late-" + remote.LABEL)
            payload = b"synthetic-bundle-content"
            state = {"finds": 0, "process": False, "neuron": False,
                     "disk_low": False, "open_delay_elapsed": 0.0}
            snapshot = {"bundle_exists": False, "target_exists": False,
                        "source": {"commit": "frozen"}}

            def stop_process():
                if stop.exists():
                    raise RuntimeError("late STOP observed")
                if state["process"]:
                    raise RuntimeError("late training process observed")

            def command(argv, **_kwargs):
                if argv[0] == "/usr/bin/fuser":
                    if arrival == "disk_during_fuser":
                        state["disk_low"] = True
                    return types.SimpleNamespace(returncode=1, stdout=b"", stderr=b"")
                if argv[0] == "neuron-ls":
                    output = (b'[{"neuron_processes":[{"command":"train.py"}]}]'
                              if state["neuron"] else b"[]")
                    return types.SimpleNamespace(returncode=0, stdout=output, stderr=b"")
                if argv[0] == "/usr/bin/find":
                    state["finds"] += 1
                    seen = label.exists()
                    if arrival == "process_after_first_find" and state["finds"] == 1:
                        state["process"] = True
                    if arrival == "process_after_second_find" and state["finds"] == 2:
                        state["process"] = True
                    if arrival == "neuron_after_first_find" and state["finds"] == 1:
                        state["neuron"] = True
                    if arrival == "neuron_after_second_find" and state["finds"] == 2:
                        state["neuron"] = True
                    if arrival == "disk_after_second_find" and state["finds"] == 2:
                        state["disk_low"] = True
                    output = (str(label) + "\n").encode() if seen else b""
                    return types.SimpleNamespace(returncode=0, stdout=output, stderr=b"")
                raise AssertionError("unexpected subprocess command: " + repr(argv))

            real_open = os.open

            def delayed_open(path, flags, *args, **kwargs):
                if path == bundle.name and flags & os.O_CREAT:
                    before = time.monotonic()
                    fd = real_open(path, flags, *args, **kwargs)
                    if arrival == "process_after_open":
                        state["process"] = True
                    if arrival == "neuron_after_open":
                        state["neuron"] = True
                    if arrival == "label_after_open":
                        label.touch()
                    if arrival == "stop_after_open":
                        stop.touch()
                    if arrival == "disk_after_open":
                        state["disk_low"] = True
                    if mutate_open:
                        os.write(fd, b"unexpected")
                    if open_delay:
                        time.sleep(open_delay)
                    state["open_delay_elapsed"] = time.monotonic() - before
                    return fd
                return real_open(path, flags, *args, **kwargs)

            def space(_minimum):
                if state["disk_low"]:
                    raise RuntimeError("late root disk floor observed")
                return {"root_free_bytes": remote.PRE_FREE + 4096,
                        "root_free_inodes": remote.MIN_INODES + 10}

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
                    receipt, error = None, str(exc)
                return {"bundle": bundle.exists(), "target": target.exists(),
                        "receipt": receipt, "error": error, **state}

    def test_training_process_after_first_scan_blocks_before_write(self):
        result = self.exercise("process_after_first_find")
        self.assertIn("training process", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_training_process_after_second_scan_blocks_before_write(self):
        result = self.exercise("process_after_second_find")
        self.assertIn("training process", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_neuron_holder_after_first_scan_blocks_before_write(self):
        result = self.exercise("neuron_after_first_find")
        self.assertIn("Neuron process active", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_neuron_holder_after_second_scan_blocks_before_write(self):
        result = self.exercise("neuron_after_second_find")
        self.assertIn("Neuron process active", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_disk_loss_during_final_fuser_blocks_before_write(self):
        result = self.exercise("disk_during_fuser")
        self.assertIn("root disk floor", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_disk_loss_during_last_scan_blocks_before_write(self):
        result = self.exercise("disk_after_second_find")
        self.assertIn("root disk floor", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_delayed_exclusive_open_rolls_back_empty_bundle_and_holds(self):
        result = self.exercise(open_delay=1.1)
        self.assertIn("too old", result["error"])
        self.assertGreaterEqual(result["open_delay_elapsed"], 1.1)
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_disk_floor_loss_at_exclusive_open_rolls_back_empty_bundle(self):
        result = self.exercise("disk_after_open")
        self.assertIn("root disk floor", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_process_entering_helper_to_open_gap_rolls_back_empty_bundle(self):
        result = self.exercise("process_after_open")
        self.assertIn("training process", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_neuron_entering_helper_to_open_gap_rolls_back_empty_bundle(self):
        result = self.exercise("neuron_after_open")
        self.assertIn("Neuron process active", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_label_entering_helper_to_open_gap_rolls_back_empty_bundle(self):
        result = self.exercise("label_after_open")
        self.assertIn("label already appears", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_stop_entering_helper_to_open_gap_rolls_back_empty_bundle(self):
        result = self.exercise("stop_after_open")
        self.assertIn("STOP", result["error"])
        self.assertFalse(result["bundle"])
        self.assertFalse(result["target"])

    def test_unexpected_nonempty_open_result_is_preserved_and_holds(self):
        result = self.exercise(open_delay=1.1, mutate_open=True)
        self.assertIn("provisional bundle changed", result["error"])
        self.assertTrue(result["bundle"])
        self.assertFalse(result["target"])

    def test_fast_clean_local_fixture_has_post_open_boundary_evidence(self):
        result = self.exercise()
        self.assertIsNone(result["error"])
        self.assertEqual(result["finds"], 3)
        self.assertTrue(result["bundle"])
        self.assertTrue(result["target"])
        proof = result["receipt"]["final_no_write_continuity"]
        self.assertIs(proof["post_label_activity_check"], True)
        self.assertIs(proof["post_open_label_and_activity_check"], True)
        self.assertIs(proof["double_label_scan"], True)
        for field in ("activity_age_seconds", "label_age_seconds",
                      "space_age_seconds", "label_scan_duration_seconds",
                      "open_call_elapsed_seconds", "post_open_label_age_seconds",
                      "post_open_activity_age_seconds", "post_open_space_age_seconds"):
            self.assertGreaterEqual(proof[field], 0)
            self.assertLessEqual(proof[field], 1)
        self.assertGreaterEqual(proof["root_space_before_open"]["root_free_bytes"],
                                remote.PRE_FREE)
        self.assertGreaterEqual(proof["root_space_after_open"]["root_free_inodes"],
                                remote.MIN_INODES)


if __name__ == "__main__":
    unittest.main()
