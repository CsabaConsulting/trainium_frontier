"""Independent offline witnesses for the disabled trn22 r2 timing diagnostic.

This file never invokes SSH, a host writer, training, or evaluation.
"""

import ast
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest import mock


PACKAGE = (Path(__file__).resolve().parents[1] / "authorizations" /
           "trn22-c5c-a0-r9-incremental-final-continuity-timing-r4-hold")


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


operator = load("independent_trn22_timing_operator", "operator_timing_readonly.py")
remote = load("independent_trn22_timing_remote", "remote_timing_readonly.py")


class IndependentTimingTests(unittest.TestCase):
    def test_disabled_gate_prevents_ssh_and_result_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / operator.OUTPUT.name
            with mock.patch.object(operator, "ENABLED", Path(tmp) / "absent.json"), \
                 mock.patch.object(operator, "OUTPUT", output), \
                 mock.patch.object(operator, "_ssh_identity") as ssh_identity, \
                 mock.patch.object(operator, "_run_ssh") as ssh:
                with self.assertRaises(FileNotFoundError):
                    operator.main()
                ssh_identity.assert_not_called()
                ssh.assert_not_called()
            self.assertFalse(output.exists())

    def test_one_second_boundary_is_inclusive_and_one_nanosecond_late_misses(self):
        ticks = [0, 10, 10, 20, 20, 30, 30, 40, 1_000_000_010]
        with mock.patch.object(remote, "_activity"), \
             mock.patch.object(remote, "_label_scan"), \
             mock.patch.object(remote, "_direct_stop_target"), \
             mock.patch.object(remote.time, "monotonic_ns", side_effect=ticks):
            exact = remote._trial(1)
        self.assertEqual(exact["activity_age_ns"], 1_000_000_000)
        self.assertEqual(exact["status"], "WITHIN_WINDOW_NO_OPEN")
        ticks[-1] += 1
        with mock.patch.object(remote, "_activity"), \
             mock.patch.object(remote, "_label_scan"), \
             mock.patch.object(remote, "_direct_stop_target"), \
             mock.patch.object(remote.time, "monotonic_ns", side_effect=ticks):
            late = remote._trial(1)
        self.assertEqual(late["status"], "TIMING_MISS")

    def test_target_appearing_after_both_scans_aborts_before_open_clock(self):
        with mock.patch.object(remote, "_activity"), \
             mock.patch.object(remote, "_label_scan"), \
             mock.patch.object(remote, "_direct_stop_target",
                               side_effect=RuntimeError("target appeared")), \
             mock.patch.object(remote.time, "monotonic_ns", wraps=remote.time.monotonic_ns) as clock:
            trial = remote._trial(1)
        self.assertEqual(trial["status"], "ABORTED")
        self.assertEqual(len(trial["steps"]), 3)
        self.assertNotIn("would_open_ns", trial)
        self.assertEqual(clock.call_count, 7)  # 3 finished stages, direct-check start.

    def test_postflight_failure_retains_timing_prefix_and_fails_closed(self):
        raw_pre = {"operation": "probe", "parsed": {"records": [{"schema": "probe"}]}}
        raw_timing = {"operation": "timing", "timed_out": True,
                      "parsed": {"records": [{"kind": "start"}, {"kind": "trial"}]}}
        raw_post = {"operation": "probe", "parsed": {"records": []}}
        authority = {"local_gates": {"synthetic": True}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "raw.json"
            with mock.patch.object(operator, "OUTPUT", output), \
                 mock.patch.object(operator, "_local_gates", return_value=authority["local_gates"]), \
                 mock.patch.object(operator, "_run_ssh",
                                   side_effect=[raw_pre, raw_timing, raw_post]) as ssh, \
                 mock.patch.object(operator, "_require_probe",
                                   side_effect=[{"identity": "before"},
                                                RuntimeError("postflight failed")]):
                status = operator.collect_once(authority)
            record = json.loads(output.read_text())
        self.assertEqual(status, "FAILED_CLOSED")
        self.assertEqual(ssh.call_count, 3)
        self.assertTrue(ssh.call_args.kwargs["postflight"])
        self.assertEqual(record["phases"][1]["parsed"]["records"][1]["kind"], "trial")
        self.assertEqual(record["errors"][0]["message"], "postflight failed")

    def test_local_one_attempt_receipt_cannot_be_reused(self):
        authority = {"local_gates": {"synthetic": True}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "raw.json"
            output.write_text("existing attempt\n")
            with mock.patch.object(operator, "OUTPUT", output), \
                 mock.patch.object(operator, "_run_ssh") as ssh:
                with self.assertRaises(FileExistsError):
                    operator.collect_once(authority)
                ssh.assert_not_called()
            self.assertEqual(output.read_text(), "existing attempt\n")

    def test_remote_source_has_no_filesystem_write_primitive(self):
        tree = ast.parse((PACKAGE / "remote_timing_readonly.py").read_text())
        forbidden = {"mkdir", "makedirs", "unlink", "remove", "rename",
                     "replace", "write_text", "write_bytes", "rmtree", "kill"}
        calls = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertFalse(forbidden.intersection(calls))
        source = (PACKAGE / "remote_timing_readonly.py").read_text()
        self.assertNotIn("os.O_CREAT", source)
        self.assertNotIn("os.O_WRONLY", source)
        self.assertNotIn("os.O_RDWR", source)
        with mock.patch.object(operator, "_pinned", return_value=b"pinned-remote-script"):
            words = shlex.split(operator._ssh_argv("timing", pin={"test": True})[-1])
        self.assertEqual(words[:2], ["/usr/bin/timeout", "--signal=TERM"])
        self.assertIn("/usr/bin/python3", words)
        self.assertIn("-I", words)
        self.assertIn("-B", words)


if __name__ == "__main__":
    unittest.main()
