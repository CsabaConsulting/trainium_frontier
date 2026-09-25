"""Offline r2 regressions adapted from independent HOLD witness 40b49c32.

These keep real _activity, _label_scan, perform and O_EXCL write flow while
faking only host probes/subprocess output on a disposable local directory.
"""

from contextlib import nullcontext
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("trn22_r2_late_remote",
                                              HERE / "remote_incremental.py")
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)


class FinalScanRaceTests(unittest.TestCase):
    def _run(self, *, arrival=None, marker_name=None, find_delay=0.0):
        with tempfile.TemporaryDirectory(prefix="trn22-stage-r2-race-") as temp:
            root = Path(temp)
            marker = root / (marker_name or "unused")
            bundle, target = root / "delta.bundle", root / "r9"
            payload = b"synthetic exact bundle"
            snapshot = {"bundle_exists": False, "target_exists": False,
                        "source": {"commit": "same"}}
            events = {"find_calls": 0, "fuser_calls": 0}

            def arrive():
                if arrival is not None and not marker.exists():
                    marker.write_text("arrived during final checks\n", encoding="ascii")

            def fake_run(argv, **_kwargs):
                if argv[0] == "/usr/bin/fuser":
                    events["fuser_calls"] += 1
                    if arrival == "fuser":
                        arrive()
                    return types.SimpleNamespace(returncode=1, stdout=b"", stderr=b"")
                if argv[0] == "neuron-ls":
                    return types.SimpleNamespace(returncode=0, stdout=b"[]", stderr=b"")
                if argv[0] == "/usr/bin/find":
                    events["find_calls"] += 1
                    if find_delay and events["find_calls"] == 1:
                        time.sleep(find_delay)
                    found = marker.exists() and remote.LABEL in marker.name
                    output = (str(marker) + "\n").encode() if found else b""
                    if arrival == "first_find" and events["find_calls"] == 1:
                        arrive()  # Created after the first traversal result.
                    return types.SimpleNamespace(returncode=0, stdout=output, stderr=b"")
                raise AssertionError("unexpected subprocess command")

            def fake_scandir(path):
                self.assertEqual(path, "/proc")
                return iter(())

            with mock.patch.object(remote, "ROOT", root), \
                 mock.patch.object(remote, "BUNDLE", bundle), \
                 mock.patch.object(remote, "TARGET", target), \
                 mock.patch.object(remote, "BUNDLE_BYTES", len(payload)), \
                 mock.patch.object(remote, "BUNDLE_SHA256", hashlib.sha256(payload).hexdigest()), \
                 mock.patch.object(remote, "STOP_PATHS", [root / "STOP"]), \
                 mock.patch.object(remote, "_host_lock", return_value=nullcontext({})), \
                 mock.patch.object(remote, "_probe", return_value=snapshot), \
                 mock.patch.object(remote, "_stable"), \
                 mock.patch.object(remote, "_proc_ancestry", return_value=set()), \
                 mock.patch.object(remote.os, "scandir", side_effect=fake_scandir), \
                 mock.patch.object(remote.subprocess, "run", side_effect=fake_run), \
                 mock.patch.object(remote, "_checked_bundle", return_value={}), \
                 mock.patch.object(remote, "_reconstruct", return_value={}), \
                 mock.patch.object(remote, "_source", return_value=snapshot["source"]), \
                 mock.patch.object(remote, "_space", return_value={}):
                try:
                    events["result"] = remote.perform(
                        "write", prior={"synthetic": "preflight"}, data=payload)
                except RuntimeError as error:
                    events["error"] = str(error)
            events["bundle_exists"] = bundle.exists()
            events["target_exists"] = target.exists()
            return events

    def _assert_no_first_write(self, events):
        self.assertIn("error", events)
        self.assertFalse(events["bundle_exists"], events)
        self.assertFalse(events["target_exists"], events)

    def test_stop_during_fuser_blocks_first_exclusive_write(self):
        events = self._run(arrival="fuser", marker_name="STOP")
        self._assert_no_first_write(events)
        self.assertGreaterEqual(events["fuser_calls"], 1)
        self.assertIn("STOP", events["error"])

    def test_label_during_fuser_blocks_first_exclusive_write(self):
        events = self._run(arrival="fuser", marker_name=remote.LABEL)
        self._assert_no_first_write(events)
        self.assertGreaterEqual(events["find_calls"], 1)
        self.assertIn("label already appears", events["error"])

    def test_label_after_first_final_scan_is_caught_by_second(self):
        events = self._run(arrival="first_find", marker_name=remote.LABEL)
        self._assert_no_first_write(events)
        self.assertEqual(events["find_calls"], 2)
        self.assertIn("label already appears", events["error"])

    def test_stop_during_final_label_scan_is_caught_directly(self):
        events = self._run(arrival="first_find", marker_name="STOP")
        self._assert_no_first_write(events)
        self.assertEqual(events["find_calls"], 2)
        self.assertIn("STOP", events["error"])

    def test_1_1_second_final_scan_is_too_old_to_write(self):
        events = self._run(find_delay=1.1)
        self._assert_no_first_write(events)
        self.assertEqual(events["find_calls"], 2)
        self.assertIn("too old", events["error"])

    def test_clean_fast_final_sequence_reaches_only_synthetic_write(self):
        events = self._run()
        self.assertNotIn("error", events)
        self.assertTrue(events["bundle_exists"])
        self.assertTrue(events["target_exists"])
        self.assertEqual(events["find_calls"], 2)
        proof = events["result"]["final_no_write_continuity"]
        self.assertIs(proof["double_label_scan"], True)
        self.assertLessEqual(proof["activity_age_seconds"], 1.0)
        self.assertLessEqual(proof["label_age_seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()
