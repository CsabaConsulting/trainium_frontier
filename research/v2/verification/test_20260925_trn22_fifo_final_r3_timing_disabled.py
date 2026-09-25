"""Independent offline negative checks for the held trn22 timing collector."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PACKAGE = (Path(__file__).resolve().parents[1] / "authorizations" /
           "trn22-c5c-a0-r4-fifo-final-r3-timing-hold")
SPEC = importlib.util.spec_from_file_location(
    "independent_trn22_timing_collector", PACKAGE / "operator_timing_readonly.py")
operator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(operator)


class DisabledCollectorChecks(unittest.TestCase):
    def test_missing_enabled_request_prevents_subprocess_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            enabled = Path(directory) / "TIMING_REQUEST.ENABLED.json"
            result = Path(directory) / "OPERATOR_RAW_TIMING_RESULT.json"
            with mock.patch.object(operator, "ENABLED", enabled), mock.patch.object(
                    operator, "OUTPUT", result), mock.patch.object(
                    operator.subprocess, "run", side_effect=AssertionError("subprocess reached")):
                with self.assertRaises(FileNotFoundError) as caught:
                    operator.main()
            self.assertEqual(caught.exception.filename, str(enabled))
            self.assertFalse(result.exists())

    def test_non_readonly_operation_is_rejected_before_transport(self):
        class ForbiddenTransport:
            @staticmethod
            def checked_ssh_identity():
                raise AssertionError("transport reached")

        for operation, pin in (("write", None), ("verify", None),
                               ("timing", None), ("probe", {})):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(RuntimeError, "wrong exact read-only"):
                    operator._ssh_argv(operation, pin=pin, transport=ForbiddenTransport())

    def test_timing_ssh_failure_keeps_local_raw_receipt_and_postflight(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "OPERATOR_RAW_TIMING_RESULT.json"
            local_gates = {"synthetic": "clean"}
            authority = {"local_gates": local_gates, "transport": object()}
            calls = []
            pre = {"utc": "2026-09-25T00:00:00+00:00",
                   "process_identity": {"pid": 1, "start_ticks": 1},
                   "identity_after": {key: "synthetic" for key in operator.STABLE_KEYS},
                   "host_lock_identity": {"synthetic": "lock"},
                   "old_inventory": {}, "targets": {}, "protected": {}}

            def run_ssh(operation, **kwargs):
                calls.append(operation)
                if operation == "timing":
                    raise OSError("synthetic SSH failure")
                return {"operation": operation, "returncode": 0}

            with mock.patch.object(operator, "OUTPUT", result), mock.patch.object(
                    operator, "_local_stops_and_breakers", return_value=local_gates), mock.patch.object(
                    operator, "_run_ssh", side_effect=run_ssh), mock.patch.object(
                    operator, "_require_probe", side_effect=[pre, {"post": True}]):
                status = operator.collect_once(authority)
            self.assertEqual(status, "FAILED_CLOSED")
            self.assertEqual(calls, ["probe", "timing", "probe"])
            record = json.loads(result.read_bytes())
            self.assertEqual([phase["operation"] for phase in record["phases"]],
                             ["probe", "probe"])
            self.assertEqual(record["errors"][0]["message"], "synthetic SSH failure")


if __name__ == "__main__":
    unittest.main()
