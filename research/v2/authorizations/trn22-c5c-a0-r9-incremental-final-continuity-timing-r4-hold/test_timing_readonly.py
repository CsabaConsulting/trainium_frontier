"""Local synthetic tests only. Never connect to trn22 or run a stage writer."""

import base64
from contextlib import nullcontext
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


remote = load("trn22_r4_remote_timing_under_test", "remote_timing_readonly.py")
operator = load("trn22_r4_operator_timing_under_test", "operator_timing_readonly.py")


def line(value):
    return operator._canonical(value) + b"\n"


class FlushBytes(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1


class RemoteSequenceTests(unittest.TestCase):
    def test_exact_readonly_sequence_and_clock_only_open(self):
        calls = []

        def step(name):
            calls.append(name)

        with mock.patch.object(remote, "_activity", side_effect=lambda: step("activity")), \
             mock.patch.object(remote, "_label_scan", side_effect=lambda: step("label")), \
             mock.patch.object(remote, "_direct_stop_target", side_effect=lambda: step("direct")):
            trial = remote._trial(1)
        self.assertEqual(calls, ["activity", "label", "label", "direct"])
        self.assertEqual(trial["status"], "WITHIN_WINDOW_NO_OPEN")
        self.assertEqual([step["name"] for step in trial["steps"]],
                         ["activity", "label_scan_1", "label_scan_2", "direct_stop_target"])
        self.assertIn("would_open_ns", trial)
        self.assertNotIn("target_open", trial)

    def test_activity_age_over_one_second_is_timing_miss(self):
        ticks = [0, 10, 20, 30, 40, 1_100_000_040,
                 1_100_000_050, 1_100_000_060, 1_100_000_070]
        with mock.patch.object(remote, "_activity"), mock.patch.object(remote, "_label_scan"), \
             mock.patch.object(remote, "_direct_stop_target"), \
             mock.patch.object(remote.time, "monotonic_ns", side_effect=ticks):
            trial = remote._trial(1)
        self.assertEqual(trial["status"], "TIMING_MISS")
        self.assertGreater(trial["activity_age_ns"], remote.MAX_AGE_NS)
        self.assertGreater(trial["label_age_ns"], 0)

    def test_second_scan_stop_aborts_before_direct_check_or_open_clock(self):
        calls = []

        def label():
            calls.append("scan")
            if len(calls) == 2:
                raise RuntimeError("synthetic STOP during second scan")

        with mock.patch.object(remote, "_activity"), \
             mock.patch.object(remote, "_label_scan", side_effect=label), \
             mock.patch.object(remote, "_direct_stop_target") as direct:
            trial = remote._trial(1)
        self.assertEqual(trial["status"], "ABORTED")
        self.assertEqual(calls, ["scan", "scan"])
        direct.assert_not_called()
        self.assertNotIn("would_open_ns", trial)

    def test_every_completed_trial_is_flushed_before_next(self):
        stream = FlushBytes()
        identity = {"hostname": remote.HOST, "uid": remote.UID}
        pre = {"schema": remote.SCHEMA + "-probe", "utc": remote._utc(),
               "identity": identity,
               "process_identity": {"pid": 123, "start_ticks": 456}}
        fake_out = types.SimpleNamespace(buffer=stream)
        completed = []

        def trial(index):
            completed.append((index, stream.flushes))
            return {"schema": remote.SCHEMA, "kind": "trial", "index": index,
                    "status": "WITHIN_WINDOW_NO_OPEN", "steps": []}

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
        self.assertEqual(stream.flushes, 5)  # start, three trials, end
        self.assertEqual(len(stream.getvalue().splitlines()), 5)


class CollectorParserTests(unittest.TestCase):
    @staticmethod
    def complete_sample():
        identity = {"hostname": remote.HOST, "uid": remote.UID,
                    "root_device": remote.DEV, "boot_id": "synthetic-boot",
                    "mount_namespace": "synthetic-ns",
                    "source": {"commit": remote.BASE_COMMIT}}
        pre = {"schema": operator.SCHEMA + "-probe", "identity": identity,
               "process_identity": {"pid": 1, "start_ticks": 100},
               "lock": {"synthetic": True}}
        post = {**pre, "process_identity": {"pid": 3, "start_ticks": 300}}
        start = {"schema": operator.SCHEMA, "kind": "start", "trial_count": 20,
                 "preflight_sha256": operator._sha(operator._canonical(pre)),
                 "identity": identity,
                 "process_identity": {"pid": 2, "start_ticks": 200}}
        records = [start]
        for index in range(1, 21):
            base = index * 1000
            steps = [{"name": name, "started_ns": base + offset,
                      "ended_ns": base + offset + 10, "duration_ns": 10}
                     for name, offset in (("activity", 0), ("label_scan_1", 10),
                                          ("label_scan_2", 20),
                                          ("direct_stop_target", 30))]
            records.append({"schema": operator.SCHEMA, "kind": "trial",
                            "index": index, "status": "WITHIN_WINDOW_NO_OPEN",
                            "steps": steps, "would_open_ns": base + 41,
                            "activity_age_ns": 31, "label_age_ns": 11,
                            "max_age_ns": 1_000_000_000})
        records.append({"schema": operator.SCHEMA, "kind": "end",
                        "trial_count": 20, "completed_trials": 20,
                        "timing_misses": 0, "status": "COMPLETE_READ_ONLY"})
        raw = {"returncode": 0, "timed_out": False, "stderr_bytes": 0,
               "parsed": {"records": records, "error": None}}
        return raw, pre, post

    def test_full_exact_timing_receipt_and_mutated_age_fail_closed(self):
        raw, pre, post = self.complete_sample()
        result = operator._require_timing(raw, pre, post)
        self.assertEqual(result["status"],
                         "COMPLETE_READ_ONLY_PENDING_INDEPENDENT_REVIEW")
        raw["parsed"]["records"][1]["activity_age_ns"] = 32
        with self.assertRaisesRegex(RuntimeError, "age evidence malformed"):
            operator._require_timing(raw, pre, post)

    def test_wrong_timing_process_or_step_order_fails_closed(self):
        raw, pre, post = self.complete_sample()
        raw["parsed"]["records"][0]["process_identity"] = pre["process_identity"]
        with self.assertRaisesRegex(RuntimeError, "distinct SSH process"):
            operator._require_timing(raw, pre, post)
        raw, pre, post = self.complete_sample()
        steps = raw["parsed"]["records"][1]["steps"]
        steps[2]["started_ns"] = steps[1]["started_ns"] + 5
        steps[2]["ended_ns"] = steps[2]["started_ns"] + 10
        with self.assertRaisesRegex(RuntimeError, "sequence time order"):
            operator._require_timing(raw, pre, post)

    def test_complete_prefix_survives_trailing_partial_line(self):
        start = {"schema": operator.SCHEMA, "kind": "start", "trial_count": 20}
        trial = {"schema": operator.SCHEMA, "kind": "trial", "index": 1}
        parsed = operator._parse_jsonl(line(start) + line(trial) + b'{"index":2', "timing")
        self.assertEqual([row["kind"] for row in parsed["records"]], ["start", "trial"])
        self.assertEqual(parsed["error"], "partial trailing JSONL record")
        self.assertEqual(base64.b64decode(parsed["trailing_base64"]), b'{"index":2')

    def test_wrong_order_duplicate_index_and_bad_json_fail_closed(self):
        start = {"schema": operator.SCHEMA, "kind": "start"}
        one = {"schema": operator.SCHEMA, "kind": "trial", "index": 1}
        wrong = {"schema": operator.SCHEMA, "kind": "trial", "index": 1}
        parsed = operator._parse_jsonl(line(start) + line(one) + line(wrong), "timing")
        self.assertEqual(len(parsed["records"]), 2)
        self.assertEqual(parsed["error"], "nonconsecutive trial index")
        parsed = operator._parse_jsonl(line(one), "timing")
        self.assertEqual(parsed["error"], "out-of-order JSONL kind")
        parsed = operator._parse_jsonl(line(start) + b"not-json\n", "timing")
        self.assertIn("malformed JSONL", parsed["error"])

    def test_probe_parser_requires_single_correct_schema(self):
        good = {"schema": operator.SCHEMA + "-probe", "hostname": "synthetic"}
        parsed = operator._parse_jsonl(line(good), "probe")
        self.assertIsNone(parsed["error"])
        self.assertEqual(parsed["records"], [good])
        parsed = operator._parse_jsonl(line(good) + line(good), "probe")
        self.assertEqual(parsed["error"], "wrong or repeated probe JSONL record")

    def test_timing_ssh_timeout_retains_flushed_trial_records(self):
        start = {"schema": operator.SCHEMA, "kind": "start", "trial_count": 20}
        trial = {"schema": operator.SCHEMA, "kind": "trial", "index": 1}
        partial = line(start) + line(trial) + b'{"kind":"trial"'
        argv = ["/usr/bin/ssh", "-i", "/synthetic/key", "synthetic-endpoint", "synthetic-command"]
        expired = subprocess.TimeoutExpired(argv, 135, output=partial, stderr=b"")
        with mock.patch.object(operator, "_local_gates"), \
             mock.patch.object(operator, "_ssh_identity"), \
             mock.patch.object(operator, "_ssh_argv", return_value=argv), \
             mock.patch.object(operator.subprocess, "run", side_effect=expired):
            raw = operator._run_ssh("timing", pin={"synthetic": True})
        self.assertTrue(raw["timed_out"])
        self.assertIsNone(raw["returncode"])
        self.assertEqual(len(raw["parsed"]["records"]), 2)
        self.assertEqual(base64.b64decode(raw["stdout_base64"]), partial)
        self.assertEqual(raw["argv_redacted"][-3], "<identity-file-path-redacted>")

    def test_partial_timeout_consumes_receipt_and_attempts_postflight(self):
        start = {"schema": operator.SCHEMA, "kind": "start", "trial_count": 20}
        trial = {"schema": operator.SCHEMA, "kind": "trial", "index": 1}
        raw_timing = {"returncode": None, "timed_out": True, "stderr_bytes": 0,
                      "parsed": operator._parse_jsonl(line(start) + line(trial), "timing"),
                      "stdout_base64": base64.b64encode(line(start) + line(trial)).decode()}
        pre_raw = {"operation": "probe"}
        post_raw = {"operation": "probe"}
        authority = {"local_gates": {"synthetic": True}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "raw.json"
            with mock.patch.object(operator, "OUTPUT", output), \
                 mock.patch.object(operator, "_local_gates", return_value=authority["local_gates"]), \
                 mock.patch.object(operator, "_run_ssh",
                                   side_effect=[pre_raw, raw_timing, post_raw]) as ssh, \
                 mock.patch.object(operator, "_require_probe", side_effect=[{"pre": True},
                                                                            {"post": True}]):
                status = operator.collect_once(authority)
            receipt = json.loads(output.read_text())
        self.assertEqual(status, "FAILED_CLOSED")
        self.assertEqual(ssh.call_count, 3)
        self.assertEqual(ssh.call_args.kwargs, {"postflight": True})
        self.assertEqual(receipt["phases"][1]["parsed"]["records"][1]["index"], 1)
        self.assertTrue(receipt["errors"])


class PinnedAuthorityTests(unittest.TestCase):
    def test_exact_stage_r2_source_hash_is_local_and_pinned(self):
        actual = hashlib.sha256(operator.STAGE_R2.read_bytes()).hexdigest()
        self.assertEqual(actual, operator.STAGE_R2_SHA256)
        self.assertEqual(operator.STAGE_R2_COMMIT,
                         "a7643f05c5468da6537c2d7b403a05ac048679f0")

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
        self.assertIn("GlobalKnownHostsFile=/dev/null", argv)
        words = shlex.split(argv[-1])
        self.assertEqual(words[0], "/usr/bin/timeout")
        self.assertIn("/usr/bin/python3", words)
        self.assertEqual(words[-2], "timing")
        for operation in ("write", "verify", "stage"):
            with self.assertRaisesRegex(RuntimeError, "wrong read-only operation shape"):
                operator._ssh_argv(operation)


if __name__ == "__main__":
    unittest.main()
