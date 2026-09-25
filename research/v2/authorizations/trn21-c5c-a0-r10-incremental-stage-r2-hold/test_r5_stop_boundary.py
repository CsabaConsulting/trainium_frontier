"""No-host adversarial tests for the disabled trn21 r5 local STOP seam.

Synthetic authority, controller roots and SSH identity are in-memory/local
test fixtures. Nothing here creates a real review, assignment or marker.
"""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("trn21_r10_r5_operator_stop", HERE / "operator_once.py")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


class LateLocalStopTests(unittest.TestCase):
    def test_real_optimized_entry_refuses_before_authority(self):
        if not sys.flags.optimize:
            return
        with mock.patch.object(operator, "_json") as authority, \
             mock.patch.object(operator, "_remote_timeout_identity") as probe, \
             mock.patch.object(operator, "_marker") as marker:
            with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                operator.stage_once(remote_call=mock.Mock())
        authority.assert_not_called()
        probe.assert_not_called()
        marker.assert_not_called()

    def _exercise(self, entry, late_change):
        with tempfile.TemporaryDirectory(prefix="trn21-r5-late-stop-", dir="/dev/shm") as temp:
            root = Path(temp)
            active = root / "active-Desktop-Trainium"
            legacy = root / "historical-Documents-Trainium"
            package = root / "r5-package"
            for checkout in (active, legacy, package):
                (checkout / "research/v2").mkdir(parents=True)
            state_path = active / "research/v2/state.json"
            state = {"launch_authorized": False,
                     "breakers": {"global_correctness": {"tripped": False},
                                  "trn21": {"tripped": False}},
                     "hosts": {"trn21": {"launch_allowed": False}}}
            state_path.write_text(json.dumps(state), encoding="utf-8")

            def identity_injects_late_stop():
                if late_change == "STOP":
                    (active / "research/v2/STOP").touch()
                else:
                    state["breakers"]["trn21"]["tripped"] = True
                    state_path.write_text(json.dumps(state), encoding="utf-8")

            phase = mock.Mock(side_effect=AssertionError("phase SSH contacted"))
            with mock.patch.object(operator, "CONTROL_ROOT", active), \
                 mock.patch.object(operator, "LEGACY_CONTROL_ROOT", legacy), \
                 mock.patch.object(operator, "PACKAGE_ROOT", package), \
                 mock.patch.object(operator, "_ssh_identity",
                                   side_effect=identity_injects_late_stop) as identity, \
                 mock.patch.object(operator, "_remote_timeout_identity") as probe, \
                 mock.patch.object(operator, "_marker") as marker, \
                 mock.patch.object(operator.subprocess, "run") as process:
                if entry == "phase":
                    with self.assertRaisesRegex(RuntimeError, "STOP|breaker"):
                        operator._ssh_phase("preflight", None, b"", b"remote source")
                else:
                    # Exercise the same seam under -O with only the optimized
                    # entry guard synthetically cleared; the real guard above
                    # must still refuse in an optimized process.
                    with mock.patch.object(operator.sys, "flags",
                                           types.SimpleNamespace(optimize=0)), \
                         mock.patch.dict(os.environ, {}, clear=True), \
                         mock.patch.object(operator, "_json", side_effect=[
                             ({"dispatch_authorized": True}, "synthetic-request"),
                             ({}, "synthetic-review"), ({}, "synthetic-assignment")]), \
                         mock.patch.object(operator, "_source_bytes",
                                           return_value=(b"bundle", b"remote")), \
                         mock.patch.object(operator, "_authority"):
                        with self.assertRaisesRegex(RuntimeError, "STOP|breaker"):
                            operator.stage_once(remote_call=phase)
            identity.assert_called_once_with()
            probe.assert_not_called()
            marker.assert_not_called()
            process.assert_not_called()
            phase.assert_not_called()

    def test_stop_or_breaker_during_identity_blocks_every_ssh_boundary(self):
        for entry in ("stage", "phase"):
            for late_change in ("STOP", "breaker"):
                with self.subTest(entry=entry, late_change=late_change):
                    self._exercise(entry, late_change)


if __name__ == "__main__":
    unittest.main()
