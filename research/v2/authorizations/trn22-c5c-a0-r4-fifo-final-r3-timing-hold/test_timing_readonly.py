"""Offline synthetic tests; these never SSH or inspect a Trn2 host."""

import importlib.util
import os
from pathlib import Path
import shlex
import tempfile
import time
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remote_timing = load("timing_readonly_under_test", "remote_timing_readonly.py")
operator = load("operator_timing_under_test", "operator_timing_readonly.py")


class FakeFrozenSource:
    def __init__(self, lock_path, *, post_error=None, prelabel_error=None):
        self.lock_path = lock_path
        self.post_error = post_error
        self.prelabel_error = prelabel_error
        self.calls = []
        self.identity = {key: value for key, value in (
            ("hostname", "ip-172-31-32-253"), ("uid", 1000),
            ("boot_id", "synthetic-boot"), ("mount_namespace", "synthetic-mount"),
            ("mount", {"filesystem": "ext4"}), ("block_aliases", {}),
            ("ancestry", {}))}
        self.lock = {"path": "/synthetic/device0.lock"}
        self.namespace = {
            "CHECKPOINT": "/synthetic/checkpoint",
            "TARGET": "/synthetic/archive-target",
            "MAX_FINAL_SCAN_TO_OPEN_NS": 1_000_000_000,
            "_open_parent": self.open_parent,
            "_identity": lambda parent, ancestry: self.identity,
            "_safe_path": lambda path: {"exists": False},
            "_open_existing_lock": self.open_lock,
            "_close_lock": self.close_lock,
            "_lock_identity": lambda parent, parts: self.lock,
            "_require_no_stops_or_processes": self.stop_process,
            "_processes": lambda: {"matched": [], "unreadable_cmdlines": 0},
            "_final_label_observation": self.final_label,
            "_fuser": lambda path: {"path_exists": True, "rc": 1, "pids": []},
            "_neuron_ls": lambda: {"rc": 0, "non_diagnostic_pids": []},
            "_require_no_host_stops": lambda: {"/synthetic/STOP": {"exists": False}},
            "_post_final_label_safety": self.post_label,
        }

    def open_parent(self):
        return os.open(self.lock_path.parent, os.O_RDONLY), {}

    def open_lock(self, parent):
        return (os.dup(parent), os.dup(parent), os.open(self.lock_path, os.O_RDONLY)), self.lock

    @staticmethod
    def close_lock(parts):
        for fd in reversed(parts):
            os.close(fd)

    def stop_process(self):
        self.calls.append("stop_process")
        if self.prelabel_error is not None:
            raise RuntimeError(self.prelabel_error)
        return {"stops": {"/synthetic/STOP": {"exists": False}},
                "processes": self.namespace["_processes"]()}

    def final_label(self):
        self.calls.append("final_label")
        observed = time.monotonic_ns()
        return {"started_monotonic_ns": observed, "completed_monotonic_ns": observed,
                "scan": {"rc": 0, "match_count": 0, "matches": [],
                         "stderr_empty": True}}

    def post_label(self, parent, parts, lock_pin, final_label):
        self.calls.append("post_label")
        ns = self.namespace
        ns["_fuser"](ns["CHECKPOINT"])
        ns["_fuser"]("/dev/neuron0")
        ns["_neuron_ls"]()
        ns["_lock_identity"](parent, parts)
        ns["_require_no_stops_or_processes"]()
        ns["_require_no_host_stops"]()
        if self.post_error is not None:
            raise RuntimeError(self.post_error)
        return {"completed_monotonic_ns": time.monotonic_ns()}

    def pin(self):
        return {"identity": self.identity, "lock": self.lock}


class TimingReadOnlySyntheticTests(unittest.TestCase):
    def test_exact_post_helper_order_and_monotonic_substages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.lock"
            path.touch()
            fake = FakeFrozenSource(path)
            original_fuser = fake.namespace["_fuser"]
            trial = remote_timing._trial(fake.namespace, fake.pin(), 1)
            self.assertEqual(trial["status"], "WITHIN_SOURCE_WINDOW_NO_OPEN")
            self.assertEqual(fake.calls, ["stop_process", "final_label", "post_label",
                                          "stop_process"])
            self.assertIs(fake.namespace["_fuser"], original_fuser)
            self.assertEqual([event["name"] for event in trial["post_label"]["events"],
                             ["checkpoint_fuser", "neuron0_fuser", "neuron_ls",
                              "lock_acl", "stop_process", "process_inventory", "final_stops"])
            self.assertTrue(all(event["duration_ns"] >= 0 and event["status"] == "OK"
                                for event in trial["post_label"]["events"]))
            self.assertEqual(trial["lock_stat_before"], trial["lock_stat_after"])

    def test_slow_exact_helper_is_recorded_as_timing_miss_without_open(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.lock"
            path.touch()
            fake = FakeFrozenSource(path, post_error=
                                    "post-label host safety expired before O_EXCL")
            trial = remote_timing._trial(fake.namespace, fake.pin(), 1)
            self.assertEqual(trial["status"], "TIMING_MISS")
            self.assertEqual(len(trial["post_label"]["events"]), 7)
            self.assertNotIn("would_open_attempt_monotonic_ns", trial)
            self.assertIn("lock_released_monotonic_ns", trial)

    def test_stop_activity_aborts_before_final_label_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.lock"
            path.touch()
            fake = FakeFrozenSource(path, prelabel_error="synthetic STOP appeared")
            trial = remote_timing._trial(fake.namespace, fake.pin(), 1)
            self.assertEqual(trial["status"], "ABORTED")
            self.assertEqual(fake.calls, ["stop_process"])
            self.assertIn("lock_released_monotonic_ns", trial)
            self.assertNotIn("final_label", fake.calls)

    def test_post_label_neuron_activity_aborts_instead_of_counting_timing_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.lock"
            path.touch()
            fake = FakeFrozenSource(path, post_error=
                                    "post-label checkpoint or Neuron busy/uncertain")
            trial = remote_timing._trial(fake.namespace, fake.pin(), 1)
            self.assertEqual(trial["status"], "ABORTED")
            self.assertIn("Neuron busy", trial["error"]["message"])
            self.assertIn("lock_released_monotonic_ns", trial)

    def test_protected_snapshot_drift_fails_before_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.lock"
            path.touch()
            fake = FakeFrozenSource(path)
            identity = {**fake.identity, "root_free_bytes": 100,
                        "root_free_inodes": 100}
            fake.namespace.update({
                "PRE_FREE": 1, "MIN_INODES": 1,
                "_identity": lambda parent, ancestry: identity,
                "_inspect_existing_lock": lambda parent: fake.lock,
                "_old_inventory": lambda parent: {"paths": {}},
                "_protected": lambda: {"checkpoint": "changed"},
            })
            pin = {"identity": identity, "lock": fake.lock,
                   "old_inventory": {"paths": {}},
                   "targets": {fake.namespace["TARGET"]: {"exists": False}},
                   "protected": {"checkpoint": "pinned"}}
            with self.assertRaisesRegex(RuntimeError, "protected checkpoint/cache drift"):
                remote_timing._check_snapshot(fake.namespace, pin)

    def test_lock_contention_aborts_before_final_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.lock"
            path.touch()
            fake = FakeFrozenSource(path)
            with mock.patch.object(remote_timing.fcntl, "flock",
                                   side_effect=BlockingIOError("synthetic busy lock")):
                trial = remote_timing._trial(fake.namespace, fake.pin(), 1)
            self.assertEqual(trial["status"], "ABORTED")
            self.assertEqual(fake.calls, [])
            self.assertIn("synthetic busy lock", trial["error"]["message"])

    def test_operator_is_disabled_before_any_ssh(self):
        with tempfile.TemporaryDirectory() as directory:
            absent = Path(directory) / "TIMING_REQUEST.ENABLED.json"
            with mock.patch.object(operator, "ENABLED", absent), mock.patch.object(
                    operator, "_run_ssh") as ssh:
                with self.assertRaises(FileNotFoundError):
                    operator.main()
                ssh.assert_not_called()

    def test_ssh_shape_has_no_transfer_stdin_or_write_operation(self):
        class Transport:
            IDENTITY_FILE = "/synthetic/private-key-path"
            KNOWN_HOSTS = Path("/synthetic/known_hosts")

            @staticmethod
            def checked_ssh_identity():
                return {"host_fingerprint": operator.HOST_FINGERPRINT}

        with mock.patch.object(operator, "_pinned", side_effect=lambda path, digest: b"synthetic"):
            argv = operator._ssh_argv("timing", pin={"synthetic": True},
                                      transport=Transport())
        words = shlex.split(argv[-1])
        self.assertEqual(words[:4], ["exec", "python3", "-B", "-c"])
        self.assertEqual(words[6], "timing")
        self.assertNotIn("write", [word for word in words if len(word) < 20])
        with mock.patch.object(operator, "_ssh_argv", return_value=argv), mock.patch.object(
                operator.subprocess, "run", return_value=mock.Mock(
                    stdout=b"{}", stderr=b"", returncode=0)) as run:
            raw = operator._run_ssh("timing", pin={"synthetic": True},
                                    transport=Transport())
        self.assertEqual(run.call_args.kwargs["stdin"], operator.subprocess.DEVNULL)
        self.assertEqual(raw["stdout_bytes"], 2)


if __name__ == "__main__":
    unittest.main()
