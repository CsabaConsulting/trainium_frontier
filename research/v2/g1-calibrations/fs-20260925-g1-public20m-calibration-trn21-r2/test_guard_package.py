"""Local/static, no-host checks for the one-label G1 guard package."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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
        adapter_spec = importlib.util.spec_from_file_location("dispatch_one_label", HERE / "dispatch_one_label.py")
        cls.adapter = importlib.util.module_from_spec(adapter_spec)
        adapter_spec.loader.exec_module(cls.adapter)

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

    def _authorized_copy(self):
        copy = dict(self.pending)
        copy["status"] = "AUTHORIZED_ONE_LABEL_G1_EVALUATOR"
        copy["launch_authorized"] = True
        copy["authorization"] = {
            "decision": "AUTHORIZE_ONE_LABEL_G1_EVALUATOR",
            "pending_package_sha256": hashlib.sha256((HERE / "request.guard-pending.json").read_bytes()).hexdigest(),
            "stage_review_sha256": self.pending["source_stage_binding"]["independent_stage_review_sha256"],
            "guard_package_verifier": {"path": "/no/verifier/receipt", "sha256": "0" * 64},
            "supervisor_authorized_at_utc": "2026-09-24T23:30:00Z",
        }
        return copy

    def _assert_authorization_rejected_before_transport(self, copy, expected_message):
        with tempfile.TemporaryDirectory(prefix="trainium-g1-auth-test-") as directory:
            request_path = Path(directory) / "request.authorized.json"
            request_path.write_text(json.dumps(copy))
            result = subprocess.run([
                sys.executable, str(HERE / "dispatch_one_label.py"),
                "--request", str(request_path),
                "--operator-checkout", "/no/operator/checkout",
                "--ssh-key-path", "/no/ssh/key",
            ], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected_message, result.stderr)
            self.assertNotIn("No such file or directory: '/no/ssh/key'", result.stderr)

    def test_authorized_copy_rejects_extra_and_missing_auth_keys(self):
        valid = self._authorized_copy()
        self.assertEqual(
            self.adapter.require_exact_authorization_shape(valid["authorization"]),
            valid["authorization"]["guard_package_verifier"],
        )
        extra = self._authorized_copy()
        extra["authorization"]["unexpected"] = "unreviewed"
        self._assert_authorization_rejected_before_transport(extra, "authorization keys differ")
        missing = self._authorized_copy()
        missing["authorization"].pop("pending_package_sha256")
        self._assert_authorization_rejected_before_transport(missing, "authorization keys differ")

    def test_authorized_copy_rejects_extra_and_missing_verifier_keys(self):
        extra = self._authorized_copy()
        extra["authorization"]["guard_package_verifier"]["unexpected"] = "unreviewed"
        self._assert_authorization_rejected_before_transport(extra, "guard_package_verifier keys differ")
        missing = self._authorized_copy()
        missing["authorization"]["guard_package_verifier"].pop("sha256")
        self._assert_authorization_rejected_before_transport(missing, "guard_package_verifier keys differ")

    def test_authorized_copy_rejects_malformed_or_non_utc_timestamps(self):
        for stamp in ("tomorrow", "2026-09-24T23:30:00+05:30", "2026-13-24T23:30:00Z", ""):
            with self.subTest(stamp=stamp):
                copy = self._authorized_copy()
                copy["authorization"]["supervisor_authorized_at_utc"] = stamp
                self._assert_authorization_rejected_before_transport(copy, "timestamp is not UTC ISO-8601")

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
