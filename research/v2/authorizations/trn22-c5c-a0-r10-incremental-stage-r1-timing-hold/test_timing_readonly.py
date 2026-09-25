"""Offline synthetic r10 timing tests only; no trn22 or stage-writer contact."""

import base64
from contextlib import nullcontext
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remote = load("trn22_r10_remote_timing_test", "remote_timing_readonly.py")
operator = load("trn22_r10_operator_timing_test", "operator_timing_readonly.py")


def line(value):
    return operator._canonical(value) + b"\n"


class FlushBytes(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1


class RemoteSequenceTests(unittest.TestCase):
    def test_exact_three_activity_three_scan_two_space_order_is_clock_only(self):
        calls = []

        def call(name):
            calls.append(name)

        with mock.patch.object(remote, "_activity", side_effect=lambda: call("activity")), \
             mock.patch.object(remote, "_label_scan", side_effect=lambda: call("label")), \
             mock.patch.object(remote, "_space_floor",
                               side_effect=lambda: (call("space") or {
                                   "root_free_bytes": remote.MIN_FREE + 1,
                                   "root_free_inodes": remote.MIN_INODES + 1})), \
             mock.patch.object(remote, "_direct_stop_target",
                               side_effect=lambda: call("direct")), \
             mock.patch.object(remote.os, "open",
                               side_effect=AssertionError("no remote open in trial")), \
             mock.patch.object(remote.os, "mkdir",
                               side_effect=AssertionError("no remote mkdir in trial")), \
             mock.patch.object(remote.os, "unlink",
                               side_effect=AssertionError("no remote unlink in trial")):
            trial = remote._trial(1)
        self.assertEqual(calls, ["activity", "label", "label", "activity",
                                 "space", "direct", "label", "activity",
                                 "space", "direct"])
        self.assertEqual(trial["status"], "WITHIN_MODEL_NO_OPEN")
        self.assertEqual(trial["model"], "CLOCK_ONLY_NO_OPEN")
        self.assertEqual([step["name"] for step in trial["steps"]],
                         list(operator.STEP_SEQUENCE))
        self.assertIn("would_open_ns", trial)
        self.assertIn("would_payload_ns", trial)
        self.assertNotIn("target_open", trial)
        self.assertFalse(operator._require_trial(trial, 1))

    def test_slow_double_scan_stops_before_second_activity_or_would_open(self):
        ticks = [0, 1, 2, 3, 4, 1_100_000_004]
        with mock.patch.object(remote, "_activity") as activity, \
             mock.patch.object(remote, "_label_scan") as scan, \
             mock.patch.object(remote, "_space_floor") as space, \
             mock.patch.object(remote.time, "monotonic_ns", side_effect=ticks):
            trial = remote._trial(1)
        self.assertEqual(trial["status"], "TIMING_MISS")
        self.assertEqual(trial["stopped_at"], "double_scan")
        self.assertEqual(activity.call_count, 1)
        self.assertEqual(scan.call_count, 2)
        space.assert_not_called()
        self.assertNotIn("would_open_ns", trial)
        self.assertTrue(operator._require_trial(trial, 1))

    def test_post_scan_activity_delay_misses_before_would_open(self):
        ticks = [0, 1, 2, 3, 4, 5, 6, 7,
                 1_100_000_010, 1_100_000_011,
                 1_100_000_012, 1_100_000_013, 1_100_000_014]
        with mock.patch.object(remote, "_activity"), \
             mock.patch.object(remote, "_label_scan"), \
             mock.patch.object(remote, "_space_floor",
                               return_value={"root_free_bytes": remote.MIN_FREE,
                                             "root_free_inodes": remote.MIN_INODES}), \
             mock.patch.object(remote, "_direct_stop_target"), \
             mock.patch.object(remote.time, "monotonic_ns", side_effect=ticks):
            trial = remote._trial(1)
        self.assertEqual(trial["status"], "TIMING_MISS")
        self.assertEqual(trial["stopped_at"], "pre_open")
        self.assertGreater(trial["pre_open_ages_ns"]["label_age_ns"],
                           remote.MAX_AGE_NS)
        self.assertNotIn("would_open_ns", trial)
        self.assertTrue(operator._require_trial(trial, 1))

    def test_scheduler_gap_at_clock_only_would_open_is_timing_miss(self):
        ticks = list(range(13)) + [1_100_000_013]
        with mock.patch.object(remote, "_activity"), \
             mock.patch.object(remote, "_label_scan"), \
             mock.patch.object(remote, "_space_floor",
                               return_value={"root_free_bytes": remote.MIN_FREE,
                                             "root_free_inodes": remote.MIN_INODES}), \
             mock.patch.object(remote, "_direct_stop_target"), \
             mock.patch.object(remote.time, "monotonic_ns", side_effect=ticks):
            trial = remote._trial(1)
        self.assertEqual(trial["status"], "TIMING_MISS")
        self.assertEqual(trial["stopped_at"], "would_open")
        self.assertEqual(len(trial["steps"]), 6)
        self.assertGreater(trial["would_open_ages_ns"]["label_age_ns"],
                           remote.MAX_AGE_NS)
        self.assertTrue(operator._require_trial(trial, 1))

    def test_slow_post_would_open_model_stops_before_payload(self):
        ticks = list(range(22)) + [1_100_000_022]
        with mock.patch.object(remote, "_activity"), \
             mock.patch.object(remote, "_label_scan"), \
             mock.patch.object(remote, "_space_floor",
                               return_value={"root_free_bytes": remote.MIN_FREE,
                                             "root_free_inodes": remote.MIN_INODES}), \
             mock.patch.object(remote, "_direct_stop_target"), \
             mock.patch.object(remote.time, "monotonic_ns", side_effect=ticks):
            trial = remote._trial(1)
        self.assertEqual(trial["status"], "TIMING_MISS")
        self.assertEqual(trial["stopped_at"], "would_payload")
        self.assertEqual(len(trial["steps"]), 10)
        self.assertGreater(trial["post_would_open_ages_ns"]["label_age_ns"],
                           remote.MAX_AGE_NS)
        self.assertTrue(operator._require_trial(trial, 1))

    def test_late_third_scan_or_activity_or_space_signal_aborts(self):
        for failed in ("label", "activity", "space"):
            with self.subTest(failed=failed):
                counts = {"label": 0, "activity": 0, "space": 0}

                def helper(name):
                    counts[name] += 1
                    if (name, counts[name]) == (
                            failed, 3 if failed != "space" else 2):
                        raise RuntimeError("synthetic persistent " + failed)

                with mock.patch.object(remote, "_activity",
                                       side_effect=lambda: helper("activity")), \
                     mock.patch.object(remote, "_label_scan",
                                       side_effect=lambda: helper("label")), \
                     mock.patch.object(remote, "_space_floor",
                                       side_effect=lambda: helper("space")), \
                     mock.patch.object(remote, "_direct_stop_target"):
                    trial = remote._trial(1)
                self.assertEqual(trial["status"], "ABORTED")
                self.assertIn("synthetic persistent", trial["error"]["message"])
                self.assertIn("would_open_ns", trial)
                self.assertNotIn("would_payload_ns", trial)

    def test_exact_one_second_boundary(self):
        self.assertFalse(remote._too_old({"label_age_ns": remote.MAX_AGE_NS,
                                          "activity_age_ns": 0, "space_age_ns": 0}))
        self.assertTrue(remote._too_old({"label_age_ns": remote.MAX_AGE_NS + 1,
                                         "activity_age_ns": 0, "space_age_ns": 0}))

    def test_finished_jsonl_trials_flush_before_next(self):
        stream = FlushBytes()
        identity = {"hostname": remote.HOST, "uid": remote.UID}
        pre = {"schema": remote.SCHEMA + "-probe", "utc": remote._utc(),
               "identity": identity,
               "stage_r10": operator.STAGE_R10_PIN,
               "lock": {"synthetic": True},
               "process_identity": {"pid": 123, "start_ticks": 456}}
        fake_out = types.SimpleNamespace(buffer=stream)
        completed = []

        def trial(index):
            completed.append((index, stream.flushes))
            return {"schema": remote.SCHEMA, "kind": "trial", "index": index,
                    "status": "WITHIN_MODEL_NO_OPEN", "steps": []}

        with mock.patch.object(remote, "TRIALS", 3), \
             mock.patch.object(remote, "_held_existing_lock",
                               return_value=nullcontext({"synthetic": True})), \
             mock.patch.object(remote, "probe_without_nested_lock", return_value=pre), \
             mock.patch.object(remote, "_compatible"), \
             mock.patch.object(remote, "_trial", side_effect=trial), \
             mock.patch.object(remote.sys, "stdout", fake_out):
            status = remote.timing(pre)
        self.assertEqual(status, "COMPLETE_READ_ONLY")
        self.assertEqual(completed, [(1, 1), (2, 2), (3, 3)])
        self.assertEqual(stream.flushes, 5)


class CollectorParserTests(unittest.TestCase):
    @staticmethod
    def complete_sample():
        identity = {"hostname": remote.HOST, "uid": remote.UID,
                    "root_device": remote.DEV, "boot_id": "synthetic-boot",
                    "mount_namespace": "synthetic-ns",
                    "source": {"commit": remote.BASE_COMMIT}}
        lock = {"synthetic": True}
        pre = {"schema": operator.SCHEMA + "-probe", "identity": identity,
               "stage_r10": operator.STAGE_R10_PIN, "lock": lock,
               "process_identity": {"pid": 1, "start_ticks": 100}}
        post = {**pre, "process_identity": {"pid": 3, "start_ticks": 300}}
        start = {"schema": operator.SCHEMA, "kind": "start", "trial_count": 20,
                 "preflight_sha256": operator._sha(operator._canonical(pre)),
                 "identity": identity, "stage_r10": operator.STAGE_R10_PIN,
                 "lock": lock,
                 "process_identity": {"pid": 2, "start_ticks": 200}}
        records = [start]
        for index in range(1, 21):
            base = index * 1000
            steps = []
            for offset, name in enumerate(operator.STEP_SEQUENCE):
                step = {"name": name, "started_ns": base + offset * 10,
                        "ended_ns": base + offset * 10 + 5, "duration_ns": 5}
                if name.startswith("space_"):
                    step["observation"] = {
                        "root_free_bytes": operator.PRE_FREE + 1000,
                        "root_free_inodes": operator.MIN_INODES + 1}
                steps.append(step)
            records.append({
                "schema": operator.SCHEMA, "kind": "trial", "index": index,
                "model": "CLOCK_ONLY_NO_OPEN", "status": "WITHIN_MODEL_NO_OPEN",
                "stopped_at": "none", "steps": steps,
                "max_age_ns": operator.MAX_AGE_NS, "scan_pair_duration_ns": 15,
                "pre_open_check_ns": base + 56,
                "pre_open_ages_ns": {"label_age_ns": 31, "activity_age_ns": 21,
                                     "space_age_ns": 11},
                "would_open_ns": base + 57,
                "would_open_ages_ns": {"label_age_ns": 32, "activity_age_ns": 22,
                                       "space_age_ns": 12},
                "would_payload_ns": base + 96,
                "post_would_open_ages_ns": {
                    "label_age_ns": 31, "activity_age_ns": 21,
                    "space_age_ns": 11}})
        records.append({"schema": operator.SCHEMA, "kind": "end",
                        "trial_count": 20, "completed_trials": 20,
                        "timing_misses": 0, "status": "COMPLETE_READ_ONLY"})
        raw = {"returncode": 0, "timed_out": False, "stderr_bytes": 0,
               "parsed": {"records": records, "error": None}}
        return raw, pre, post

    def test_full_exact_receipt_and_mutated_age_fail_closed(self):
        raw, pre, post = self.complete_sample()
        result = operator._require_timing(raw, pre, post)
        self.assertEqual(result["within_model_trials"], 20)
        self.assertEqual(result["max_double_scan_ns"], 15)
        raw["parsed"]["records"][1]["would_open_ages_ns"]["label_age_ns"] += 1
        with self.assertRaisesRegex(RuntimeError, "age evidence malformed"):
            operator._require_timing(raw, pre, post)

    def test_step_order_floor_and_stage_pin_mutations_fail_closed(self):
        raw, pre, post = self.complete_sample()
        bad = copy.deepcopy(raw)
        bad["parsed"]["records"][1]["steps"][6]["started_ns"] = 54
        with self.assertRaises(RuntimeError):
            operator._require_timing(bad, pre, post)
        bad = copy.deepcopy(raw)
        bad["parsed"]["records"][1]["steps"][4]["observation"]["root_free_inodes"] = 9999
        with self.assertRaisesRegex(RuntimeError, "below floor"):
            operator._require_timing(bad, pre, post)
        bad_pre = copy.deepcopy(pre)
        bad_pre["stage_r10"]["remote_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "binding malformed"):
            operator._require_timing(raw, bad_pre, post)

    def test_wrong_process_or_status_classification_fails_closed(self):
        raw, pre, post = self.complete_sample()
        raw["parsed"]["records"][0]["process_identity"] = pre["process_identity"]
        with self.assertRaisesRegex(RuntimeError, "distinct SSH process"):
            operator._require_timing(raw, pre, post)
        raw, pre, post = self.complete_sample()
        raw["parsed"]["records"][1]["status"] = "TIMING_MISS"
        with self.assertRaisesRegex(RuntimeError, "trial malformed"):
            operator._require_timing(raw, pre, post)

    def test_complete_prefix_survives_partial_jsonl(self):
        start = {"schema": operator.SCHEMA, "kind": "start", "trial_count": 20}
        trial = {"schema": operator.SCHEMA, "kind": "trial", "index": 1}
        parsed = operator._parse_jsonl(line(start) + line(trial) + b'{"index":2',
                                       "timing")
        self.assertEqual([row["kind"] for row in parsed["records"]], ["start", "trial"])
        self.assertEqual(parsed["error"], "partial trailing JSONL record")
        self.assertEqual(base64.b64decode(parsed["trailing_base64"]), b'{"index":2')

    def test_wrong_jsonl_order_and_duplicate_index_fail_closed(self):
        start = {"schema": operator.SCHEMA, "kind": "start"}
        one = {"schema": operator.SCHEMA, "kind": "trial", "index": 1}
        wrong = {"schema": operator.SCHEMA, "kind": "trial", "index": 1}
        parsed = operator._parse_jsonl(line(start) + line(one) + line(wrong),
                                       "timing")
        self.assertEqual(len(parsed["records"]), 2)
        self.assertEqual(parsed["error"], "nonconsecutive trial index")
        parsed = operator._parse_jsonl(line(one), "timing")
        self.assertEqual(parsed["error"], "out-of-order JSONL kind")

    def test_ssh_timeout_retains_flushed_prefix_without_success(self):
        start = {"schema": operator.SCHEMA, "kind": "start", "trial_count": 20}
        trial = {"schema": operator.SCHEMA, "kind": "trial", "index": 1}
        partial = line(start) + line(trial) + b'{"kind":"trial"'
        argv = ["/usr/bin/ssh", "-i", "/synthetic/key", "endpoint", "command"]
        expired = subprocess.TimeoutExpired(argv, 135, output=partial, stderr=b"")
        with mock.patch.object(operator, "_local_gates"), \
             mock.patch.object(operator, "_ssh_identity"), \
             mock.patch.object(operator, "_ssh_argv", return_value=argv), \
             mock.patch.object(operator.subprocess, "run", side_effect=expired):
            raw = operator._run_ssh("timing", pin={"synthetic": True})
        self.assertTrue(raw["timed_out"])
        self.assertEqual(len(raw["parsed"]["records"]), 2)
        self.assertEqual(base64.b64decode(raw["stdout_base64"]), partial)

    def test_partial_timeout_consumes_receipt_and_attempts_postflight(self):
        start = {"schema": operator.SCHEMA, "kind": "start", "trial_count": 20}
        trial = {"schema": operator.SCHEMA, "kind": "trial", "index": 1}
        raw_timing = {"returncode": None, "timed_out": True, "stderr_bytes": 0,
                      "parsed": operator._parse_jsonl(line(start) + line(trial),
                                                       "timing")}
        pre_raw = {"operation": "probe"}
        post_raw = {"operation": "probe"}
        authority = {"local_gates": {"synthetic": True}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "raw.json"
            with mock.patch.object(operator, "OUTPUT", output), \
                 mock.patch.object(operator, "_local_gates",
                                   return_value=authority["local_gates"]), \
                 mock.patch.object(operator, "_run_ssh",
                                   side_effect=[pre_raw, raw_timing, post_raw]) as ssh, \
                 mock.patch.object(operator, "_require_probe",
                                   side_effect=[{"pre": True}, {"post": True}]):
                status = operator.collect_once(authority)
            receipt = json.loads(output.read_text())
        self.assertEqual(status, "FAILED_CLOSED")
        self.assertEqual(ssh.call_count, 3)
        self.assertEqual(ssh.call_args.kwargs, {"postflight": True})
        self.assertEqual(receipt["phases"][1]["parsed"]["records"][1]["index"], 1)
        self.assertTrue(receipt["errors"])


class PinnedAuthorityTests(unittest.TestCase):
    def test_r10_stage_boundary_identity_matches_timing_model(self):
        stage = load("trn22_r10_stage_remote_timing_pin", operator.STAGE_R10_REMOTE)
        for name in ("SOURCE", "TARGET", "BUNDLE", "LABEL", "BASE_COMMIT",
                     "HOST", "UID", "DEV", "STOP_PATHS", "PROCESS_MARKERS",
                     "LOCK", "LOCK_INODES", "LOCK_MODES"):
            self.assertEqual(getattr(remote, name), getattr(stage, name), name)
        self.assertEqual(remote.MIN_FREE, stage.PRE_FREE)
        self.assertEqual(remote.MIN_INODES, stage.MIN_INODES)
        self.assertEqual(remote.MAX_AGE_NS,
                         int(stage.MAX_FINAL_CHECK_AGE_SECONDS * 1_000_000_000))

    def test_exact_stage_r10_and_diagnostic_hashes(self):
        for path, digest in ((operator.STAGE_R10_REMOTE,
                              operator.STAGE_R10_REMOTE_SHA256),
                             (operator.STAGE_R10_OPERATOR,
                              operator.STAGE_R10_OPERATOR_SHA256),
                             (operator.STAGE_R10_HOLD,
                              operator.STAGE_R10_HOLD_SHA256),
                             (operator.REMOTE, operator.REMOTE_SHA256)):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(operator.STAGE_R10_COMMIT,
                         "020bfa7ebd048f947d5207cbedaf028a29a4b159")

    def test_disabled_package_exits_before_ssh_or_local_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / operator.OUTPUT.name
            with mock.patch.object(operator, "ENABLED", Path(tmp) / "absent.json"), \
                 mock.patch.object(operator, "OUTPUT", output), \
                 mock.patch.object(operator, "_run_ssh") as ssh:
                with self.assertRaises(FileNotFoundError):
                    operator.main()
                ssh.assert_not_called()
            self.assertFalse(output.exists())

    def test_ssh_command_is_pinned_readonly_and_has_no_input_stream(self):
        with mock.patch.object(operator, "_pinned", return_value=b"synthetic-readonly"):
            argv = operator._ssh_argv("timing", pin={"synthetic": True})
        self.assertEqual(argv[:4], ["/usr/bin/ssh", "-T", "-F", "/dev/null"])
        self.assertEqual(argv[-2], operator.ENDPOINT)
        self.assertEqual(argv[-3], str(operator.IDENTITY_FILE))
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertIn("HostKeyAlgorithms=ssh-ed25519", argv)
        words = shlex.split(argv[-1])
        self.assertEqual(words[0], "/usr/bin/timeout")
        self.assertIn("/usr/bin/python3", words)
        self.assertEqual(words[-2], "timing")
        for operation in ("write", "verify", "stage", "cleanup"):
            with self.assertRaisesRegex(RuntimeError, "wrong read-only operation shape"):
                operator._ssh_argv(operation)


if __name__ == "__main__":
    unittest.main()
