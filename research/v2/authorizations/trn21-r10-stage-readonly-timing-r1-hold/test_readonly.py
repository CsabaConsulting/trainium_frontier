"""Offline-only regressions for disabled trn21 timing. No SSH or host use."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = load("trn21_r10_timing_collector_test", HERE / "collect_readonly.py")
timing = load("trn21_r10_remote_timing_test", HERE / "remote_timing.py")


def complete_records():
    utc = "2026-09-25T10:00:00+00:00"
    identity = {"hostname": timing.EXPECTED["HOST"], "uid": timing.EXPECTED["UID"],
                "source": timing.EXPECTED["SOURCE"],
                "target": timing.EXPECTED["TARGET"],
                "bundle": timing.EXPECTED["BUNDLE"],
                "label": timing.EXPECTED["LABEL"],
                "lock": timing.EXPECTED["LOCK"],
                "source_commit": timing.EXPECTED["BASE_COMMIT"],
                "target_commit": timing.EXPECTED["TARGET_COMMIT"]}
    start = {"schema": collector.SCHEMA, "kind": "start", "utc": utc,
             "identity": identity,
             "boot_id": "30f90d95-224c-48d9-bb11-8ec18dd7c4fe",
             "mount_namespace": "mnt:[4026531840]",
             "root_free_bytes": collector.PRE_FREE + 1000,
             "root_free_inodes": collector.MIN_INODES + 1,
             "full_readonly_preflight_seconds": 0.5,
             "max_trials": collector.TRIALS, "actual_open": False}
    trials = [{"schema": collector.SCHEMA, "kind": "trial", "index": index,
               "utc": utc, "preopen_seconds": 0.6,
               "status": "WITHIN_WINDOW_NO_OPEN",
               "label_scan_duration_seconds": 0.4,
               "activity_age_seconds": 0.2, "label_age_seconds": 0.3,
               "space_age_seconds": 0.1,
               "root_free_bytes": collector.PRE_FREE + 1000,
               "root_free_inodes": collector.MIN_INODES + 1}
              for index in range(1, collector.TRIALS + 1)]
    end = {"schema": collector.SCHEMA, "kind": "end", "utc": utc,
           "completed_trials": collector.TRIALS, "max_trials": collector.TRIALS,
           "status": "READONLY_TIMING_COMPLETE_NO_OPEN", "actual_open": False}
    return [start, *trials, end]


def jsonl(records):
    return b"\n".join(collector._canonical(item) for item in records) + b"\n"


def fake_stage(final=None):
    stage = dict(timing.EXPECTED)
    for key in ("SOURCE", "TARGET", "BUNDLE", "LOCK"):
        stage[key] = Path(stage[key])
    calls = []

    @contextmanager
    def lock():
        calls.append("lock")
        yield {"pin": "synthetic"}

    def probe(lock_value, *, minimum):
        calls.append("probe")
        if lock_value != {"pin": "synthetic"} or minimum != timing.EXPECTED["PRE_FREE"]:
            raise AssertionError("wrong exact-source preflight shape")
        return {"bundle_exists": False, "target_exists": False,
                "boot_id": "synthetic-boot",
                "mount": {"namespace": "mnt:[1]"},
                "identity_after": {"root_free_bytes": minimum + 1000,
                                   "root_free_inodes": 10001}}

    def default_final():
        calls.append("exact_final_no_write_continuity")
        import time
        now = time.monotonic()
        return {"activity_at": now, "label_at": now, "space_at": now,
                "label_scan_duration_seconds": 0.2,
                "root_space_before_open": {"root_free_bytes": 14_000_000_000,
                                           "root_free_inodes": 10001}}

    def freshness(samples, now):
        calls.append("exact_freshness_evidence")
        return {name + "_age_seconds": now - samples[name + "_at"]
                for name in ("activity", "label", "space")}

    stage.update({"_host_lock": lock, "_probe": probe,
                  "_final_no_write_continuity": final or default_final,
                  "_freshness_evidence": freshness,
                  "_write_bundle": mock.Mock(side_effect=AssertionError("writer called"))})
    return stage, calls


class ReadonlyTimingTests(unittest.TestCase):
    def test_disabled_package_pins_exact_r3_source_and_refuses_contact(self):
        inputs = collector._pins()
        self.assertEqual(len(inputs), 5)
        self.assertFalse(collector.ENABLED.exists())
        self.assertFalse(collector.REVIEW.exists())
        self.assertFalse(collector.ASSIGNMENT.exists())
        with mock.patch.object(collector, "_ssh_identity") as identity, \
             mock.patch.object(collector.subprocess, "run") as process:
            with self.assertRaises(FileNotFoundError):
                collector.capture()
            identity.assert_not_called()
            process.assert_not_called()

    def test_exact_helper_is_timed_three_times_without_writer_or_open(self):
        stage, calls = fake_stage()
        records = []
        with mock.patch.object(timing, "_emit", side_effect=records.append):
            if sys.flags.optimize:
                with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                    timing.run(stage)
                self.assertEqual(records, [])
                return
            result = timing.run(stage)
        self.assertEqual(result, "READONLY_TIMING_COMPLETE_NO_OPEN")
        self.assertEqual([item["kind"] for item in records],
                         ["start", "trial", "trial", "trial", "end"])
        self.assertEqual(calls, ["lock", "probe"] +
                         ["exact_final_no_write_continuity",
                          "exact_freshness_evidence"] * 3)
        self.assertTrue(all(item["actual_open"] is False
                            for item in (records[0], records[-1])))
        stage["_write_bundle"].assert_not_called()

    def test_failed_exact_helper_stops_after_one_trial(self):
        def failed_final():
            raise RuntimeError("final label scans too old for first O_EXCL")
        stage, calls = fake_stage(failed_final)
        records = []
        with mock.patch.object(timing, "_emit", side_effect=records.append):
            if sys.flags.optimize:
                with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                    timing.run(stage)
                return
            result = timing.run(stage)
        self.assertEqual(result, "READONLY_TIMING_HOLD_NO_OPEN")
        self.assertEqual(records[1]["status"], "HOLD_NO_OPEN")
        self.assertEqual(records[-1]["completed_trials"], 1)
        stage["_write_bundle"].assert_not_called()

    def test_stale_documents_clear_cannot_override_active_desktop_stop_or_breaker(self):
        self.assertEqual(collector.ACTIVE_CONTROL,
                         Path("/home/poojith-devan/Desktop/Trainium"))
        with tempfile.TemporaryDirectory(prefix="trn21-timing-local-gate-") as temp:
            root = Path(temp)
            desktop = root / "Desktop/Trainium"
            documents = root / "Documents/ChatGPT/Trainium"
            package = root / "package"
            for checkout in (desktop, documents, package):
                (checkout / "research/v2").mkdir(parents=True)
            clear = {"launch_authorized": False,
                     "breakers": {"global_correctness": {"tripped": False},
                                  "trn21": {"tripped": False}},
                     "hosts": {"trn21": {"launch_allowed": False}}}
            (documents / "research/v2/state.json").write_text(json.dumps(clear))
            (desktop / "research/v2/state.json").write_text(json.dumps(clear))
            stop = desktop / "research/v2/STOP"
            with mock.patch.object(collector, "ACTIVE_CONTROL", desktop), \
                 mock.patch.object(collector, "LEGACY_CONTROL", documents), \
                 mock.patch.object(collector, "REPO", package), \
                 mock.patch.object(collector, "_pins", return_value={}), \
                 mock.patch.object(collector, "_authority", return_value={}), \
                 mock.patch.object(collector, "_ssh_identity") as identity, \
                 mock.patch.object(collector.subprocess, "run") as process:
                stop.touch()
                with self.assertRaisesRegex(RuntimeError, "local STOP"):
                    collector.capture()
                stop.unlink()
                tripped = json.loads(json.dumps(clear))
                tripped["breakers"]["trn21"]["tripped"] = True
                (desktop / "research/v2/state.json").write_text(json.dumps(tripped))
                with self.assertRaisesRegex(RuntimeError, "active Desktop controller"):
                    collector.capture()
                identity.assert_not_called()
                process.assert_not_called()

    def test_jsonl_parser_rejects_nonconsecutive_or_open_claim(self):
        records = complete_records()
        self.assertEqual(len(collector._parse_jsonl(jsonl(records))), collector.TRIALS + 2)
        records[1]["index"] = 2
        with self.assertRaises(RuntimeError):
            collector._parse_jsonl(jsonl(records))
        records[1]["index"] = 1
        records[-1]["actual_open"] = True
        with self.assertRaises(RuntimeError):
            collector._parse_jsonl(jsonl(records))

    def test_forged_ordered_jsonl_fails_numeric_identity_and_resource_gates(self):
        for kind, field, value in (
                ("trial", "activity_age_seconds", 9999.0),
                ("trial", "label_age_seconds", float("nan")),
                ("trial", "space_age_seconds", float("inf")),
                ("trial", "label_scan_duration_seconds", 1.001),
                ("trial", "preopen_seconds", -1),
                ("trial", "root_free_bytes", 1),
                ("start", "identity", None),
                ("start", "boot_id", "not-a-boot-id"),
                ("start", "root_free_inodes", 9999)):
            with self.subTest(kind=kind, field=field):
                records = complete_records()
                records[1 if kind == "trial" else 0][field] = value
                with self.assertRaises(RuntimeError):
                    collector._parse_jsonl(jsonl(records))

    def test_forged_jsonl_archives_hold_not_positive_timing(self):
        forged = complete_records()
        for trial in forged[1:-1]:
            trial["activity_age_seconds"] = 9999.0
            trial["label_age_seconds"] = 9999.0
            trial["space_age_seconds"] = 9999.0
        with tempfile.TemporaryDirectory(
                    prefix="trn21-timing-forged-") as temp:
            archived = self._capture_mocked(jsonl(forged), b"", Path(temp))
            self.assertEqual(archived["status"], "HOLD")
            self.assertIsNone(archived["parsed"])
            self.assertIn("freshness", archived["parse_error"])

    def test_nonempty_stderr_with_valid_jsonl_archives_hold(self):
        with tempfile.TemporaryDirectory(prefix="trn21-timing-stderr-") as temp:
            archived = self._capture_mocked(jsonl(complete_records()),
                                            b"remote warning", Path(temp))
            self.assertEqual(archived["status"], "HOLD")
            self.assertIsNone(archived["parsed"])
            self.assertEqual(archived["parse_error"], "nonempty_remote_stderr")

    def test_valid_jsonl_with_empty_stderr_is_only_observation(self):
        with tempfile.TemporaryDirectory(prefix="trn21-timing-valid-") as temp:
            archived = self._capture_mocked(jsonl(complete_records()), b"", Path(temp))
            self.assertEqual(archived["status"],
                             "READONLY_TIMING_OBSERVED_NOT_STAGE_AUTHORITY")
            self.assertEqual(len(archived["parsed"]), collector.TRIALS + 2)

    def _capture_mocked(self, stdout, stderr, root):
        response = types.SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)
        with mock.patch.object(collector, "HERE", root), \
             mock.patch.object(collector, "_pins", return_value={
                 "stage": b"stage", "timing": b"timing", "hold": b"hold"}), \
             mock.patch.object(collector, "_authority", return_value={}), \
             mock.patch.object(collector, "_controller_gate",
                               return_value={"active_state_sha256": "stable"}), \
             mock.patch.object(collector, "_ssh_identity", return_value={}), \
             mock.patch.object(collector, "_ssh_argv", return_value=["ssh", "dummy"]), \
             mock.patch.object(collector.subprocess, "run", return_value=response) as process:
            receipt = collector.capture()
        process.assert_called_once()
        return json.loads(receipt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
