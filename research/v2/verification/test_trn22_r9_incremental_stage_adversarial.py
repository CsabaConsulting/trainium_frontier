"""Independent, offline-only timing-race tests for the disabled trn22 stage."""

from contextlib import nullcontext
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock


PACKAGE = (Path(__file__).resolve().parents[1] / "authorizations"
           / "trn22-c5c-a0-r9-incremental-stage-hold")
SPEC = importlib.util.spec_from_file_location("trn22_stage_adversarial_remote",
                                              PACKAGE / "remote_incremental.py")
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)


class FinalScanRaceTests(unittest.TestCase):
    def _run_with_late_marker(self, marker_name, delay):
        with tempfile.TemporaryDirectory(prefix="trn22-stage-race-") as temp:
            root = Path(temp)
            marker = root / marker_name
            bundle = root / "delta.bundle"
            target = root / "r9"
            payload = b"synthetic exact bundle"
            snapshot = {"bundle_exists": False, "target_exists": False,
                        "source": {"commit": "same"}}
            events = {}

            def fake_run(argv, **_kwargs):
                # Keep the real _activity() control flow: its STOP sweep is
                # first, and fuser can race a new STOP or attempt label.
                if argv[0] == "/usr/bin/fuser":
                    if not marker.exists():
                        time.sleep(delay)
                        marker.write_text("arrived after scan\n", encoding="ascii")
                        events["marker_time"] = time.monotonic()
                    return types.SimpleNamespace(returncode=1, stdout=b"", stderr=b"")
                if argv[0] == "neuron-ls":
                    return types.SimpleNamespace(returncode=0, stdout=b"[]", stderr=b"")
                raise AssertionError("unexpected subprocess")

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
            events["bundle_bytes"] = bundle.read_bytes() if bundle.exists() else b""
            return events, payload

    def test_stop_arriving_after_final_stop_scan_is_written_before_recheck(self):
        events, payload = self._run_with_late_marker("STOP", 0)
        # This is an adversarial observation of the current HOLD source, not
        # an expected safety property: the STOP is detected only after write.
        self.assertTrue(events["bundle_exists"])
        self.assertTrue(events["target_exists"])
        self.assertEqual(events["bundle_bytes"], payload)
        self.assertEqual(events["error"], "host/source/target STOP is present")

    def test_label_arriving_after_final_label_scan_still_returns_success(self):
        events, payload = self._run_with_late_marker(remote.LABEL, 0)
        self.assertTrue(events["bundle_exists"])
        self.assertTrue(events["target_exists"])
        self.assertEqual(events["bundle_bytes"], payload)
        self.assertEqual(events["result"]["phase"], "write")

    def test_more_than_one_second_after_scan_still_writes(self):
        events, payload = self._run_with_late_marker("STOP", 1.1)
        self.assertTrue(events["bundle_exists"])
        self.assertEqual(events["bundle_bytes"], payload)


if __name__ == "__main__":
    unittest.main()
