"""Local/static, no-host checks for the one-label G1 guard package."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


class GuardPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.staged = json.loads((HERE / "request.staged.json").read_text())
        cls.pending = json.loads((HERE / "request.guard-pending.json").read_text())
        spec = importlib.util.spec_from_file_location("guarded_eval", HERE / "guarded_eval.py")
        cls.monitor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.monitor)

    def test_pending_is_not_dispatchable(self):
        self.assertFalse(self.pending["launch_authorized"])
        self.assertIsNone(self.pending["authorization"])
        self.assertEqual(self.pending["label"], self.staged["label"])
        self.assertEqual(self.pending["host"], "trn21")

    def test_science_argv_environment_and_inputs_unchanged(self):
        self.assertEqual(self.pending["calibration_eval_command"], self.staged["command"][5:])
        self.assertEqual(self.pending["environment"], self.staged["environment"])
        self.assertEqual(self.pending["checkpoint"], {
            key: self.staged["checkpoint"][key] for key in self.pending["checkpoint"]
        })
        self.assertEqual(self.pending["expected_commit"], self.staged["expected_commit"])
        self.assertEqual(self.pending["expected_sha256"]["prepare.py"], self.staged["expected_sha256"]["prepare.py"])

    def test_monitor_script_and_threshold_bound(self):
        import hashlib
        self.assertEqual(self.pending["monitor"]["script_sha256"], hashlib.sha256((HERE / "guarded_eval.py").read_bytes()).hexdigest())
        self.assertEqual(self.pending["monitor"]["stop_below_bytes"], self.staged["disk_gate"]["pre_launch_required_free_bytes"])
        self.assertEqual(self.monitor.EVAL_ARGV, self.pending["calibration_eval_command"])

    def test_controller_adapter_rejects_pending_without_host_action(self):
        result = subprocess.run([
            sys.executable, str(HERE / "dispatch_one_label.py"),
            "--request", str(HERE / "request.guard-pending.json"),
            "--operator-checkout", str(HERE),
            "--ssh-key-path", "/dev/null",
        ], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pending request is never dispatchable", result.stderr)

    def test_pid_mismatch_never_signals(self):
        child = mock.Mock(pid=12345)
        child.poll.return_value = None
        with mock.patch.object(self.monitor, "process_identity", return_value=(1000, 99, 12345)), \
             mock.patch.object(self.monitor, "append_event"), \
             mock.patch.object(self.monitor.os, "killpg") as killpg, \
             mock.patch.object(self.monitor.os, "getuid", return_value=1000):
            self.assertFalse(self.monitor.stop_owned_child(child, 100, object(), "test"))
            killpg.assert_not_called()


if __name__ == "__main__":
    unittest.main()
