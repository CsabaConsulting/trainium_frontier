"""Independent offline witnesses for the disabled trn21 r10 stage boundary.

These tests use a disposable in-memory filesystem directory and never contact a
host.  They intentionally document behavior of the frozen c03e3ac3 subject;
they are not production tests or authority to dispatch the stage package.
"""

from contextlib import ExitStack
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock


STAGE = (Path(__file__).resolve().parents[1] / "authorizations" /
         "trn21-c5c-a0-r10-incremental-stage-hold")
REMOTE_SOURCE = STAGE / "remote_incremental.py"
REMOTE_SHA256 = "eb1b1d9150dc35d6c96c00ad49c321bc9944c0cbaeef0b1494272346eb78c9f0"
OPERATOR_SOURCE = STAGE / "operator_once.py"
OPERATOR_SHA256 = "8ed2266c20774c5f4f1f750bbe9f94c8d68a8e8459670308ef689c8e714901c1"

spec = importlib.util.spec_from_file_location("trn21_r10_stage_boundary_subject", REMOTE_SOURCE)
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)
operator_spec = importlib.util.spec_from_file_location("trn21_r10_stage_operator_subject",
                                                      OPERATOR_SOURCE)
operator = importlib.util.module_from_spec(operator_spec)
operator_spec.loader.exec_module(operator)


class BoundaryWitnesses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if hashlib.sha256(REMOTE_SOURCE.read_bytes()).hexdigest() != REMOTE_SHA256:
            raise AssertionError("frozen remote stage source digest changed")
        if hashlib.sha256(OPERATOR_SOURCE.read_bytes()).hexdigest() != OPERATOR_SHA256:
            raise AssertionError("frozen operator stage source digest changed")

    def _witness(self, after_final_gate, *, statvfs=None):
        data = (STAGE / "e65-to-r10.bundle").read_bytes()
        with tempfile.TemporaryDirectory(prefix="trn21-r10-review-", dir="/dev/shm") as temp:
            root = Path(temp)
            stop = root / "STOP"
            label = root / ("late-" + remote.LABEL)
            bundle = root / "delta.bundle"
            target = root / "new-r10"
            state = {"busy": False, "disk_low": False, "final_gate_at": None,
                     "first_exclusive_open_at": None, "label": label, "stop": stop}
            real_gate = remote._final_no_write_continuity
            real_open = os.open

            def activity():
                if state["busy"]:
                    raise RuntimeError("synthetic train/Neuron activity")
                return {"idle": True}

            def label_scan():
                if label.exists():
                    raise RuntimeError("synthetic label collision")
                return {"matches": []}

            def final_gate():
                result = real_gate()
                state["final_gate_at"] = time.monotonic()
                after_final_gate(state)
                return result

            def record_open(path, flags, *args, **kwargs):
                if flags & os.O_EXCL:
                    state["first_exclusive_open_at"] = time.monotonic()
                return real_open(path, flags, *args, **kwargs)

            patches = [
                mock.patch.object(remote, "ROOT", root),
                mock.patch.object(remote, "STOP_PATHS", [stop]),
                mock.patch.object(remote, "BUNDLE", bundle),
                mock.patch.object(remote, "TARGET", target),
                mock.patch.object(remote, "_activity", side_effect=activity),
                mock.patch.object(remote, "_label_scan", side_effect=label_scan),
                mock.patch.object(remote, "_final_no_write_continuity", side_effect=final_gate),
                mock.patch.object(remote, "_checked_bundle", return_value={"checked": True}),
                mock.patch.object(remote.os, "open", side_effect=record_open),
            ]
            if statvfs is not None:
                patches.append(mock.patch.object(remote.os, "statvfs",
                                                 side_effect=lambda _: statvfs(state)))
            with ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                if statvfs is not None:
                    self.assertGreaterEqual(remote._space(remote.PRE_FREE)["root_free_bytes"],
                                            remote.PRE_FREE)
                receipt = remote._write_bundle(data)
                self.assertTrue(bundle.is_file())
                self.assertTrue(target.is_dir())
                self.assertEqual(bundle.read_bytes(), data)
                self.assertTrue(receipt["double_label_scan"])
                state["reported_activity_age"] = receipt["activity_age_seconds"]
                state["reported_label_age"] = receipt["label_age_seconds"]
                self.assertIsNotNone(state["final_gate_at"])
                self.assertIsNotNone(state["first_exclusive_open_at"])
                state["stop_exists_after_write"] = stop.exists()
                state["label_exists_after_write"] = label.exists()
                return dict(state)

    def test_stop_appearing_after_final_gate_still_creates_bundle_and_target(self):
        state = self._witness(lambda s: s["stop"].write_text("STOP", encoding="ascii"))
        self.assertTrue(state["stop_exists_after_write"])

    def test_label_appearing_after_final_gate_still_creates_bundle_and_target(self):
        state = self._witness(lambda s: s["label"].write_text("collision", encoding="ascii"))
        self.assertTrue(state["label_exists_after_write"])

    def test_activity_appearing_after_final_gate_still_creates_bundle_and_target(self):
        state = self._witness(lambda s: s.__setitem__("busy", True))
        self.assertTrue(state["busy"])

    def test_disk_floor_falling_after_probe_still_creates_bundle_and_target(self):
        good = types.SimpleNamespace(f_bavail=20_000_000, f_frsize=1024,
                                     f_favail=remote.MIN_INODES)
        low = types.SimpleNamespace(f_bavail=1, f_frsize=1024,
                                    f_favail=remote.MIN_INODES - 1)
        state = self._witness(lambda s: s.__setitem__("disk_low", True),
                              statvfs=lambda s: low if s["disk_low"] else good)
        self.assertTrue(state["disk_low"])

    def test_elapsed_more_than_one_second_after_gate_still_creates_targets(self):
        state = self._witness(lambda _: time.sleep(1.10))
        self.assertGreater(state["first_exclusive_open_at"] - state["final_gate_at"], 1.0)
        self.assertLessEqual(state["reported_activity_age"], 1.0)
        self.assertLessEqual(state["reported_label_age"], 1.0)

    def test_local_stop_blocks_remote_full_postflight_call(self):
        with mock.patch.object(operator, "_local_stop_and_breakers",
                               side_effect=RuntimeError("synthetic local STOP")), \
             mock.patch.object(operator, "_ssh_identity") as identity, \
             mock.patch.object(operator.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "synthetic local STOP"):
                operator._ssh_phase("postflight", None, b"", b"remote source")
            identity.assert_not_called()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
