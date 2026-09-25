"""Offline-only regressions for disabled trn21 timing. No SSH or host use."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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
        start = {"schema": collector.SCHEMA, "kind": "start", "actual_open": False}
        trial = {"schema": collector.SCHEMA, "kind": "trial", "index": 1,
                 "status": "HOLD_NO_OPEN"}
        end = {"schema": collector.SCHEMA, "kind": "end",
               "completed_trials": 1, "actual_open": False,
               "status": "READONLY_TIMING_HOLD_NO_OPEN"}
        raw = b"\n".join(collector._canonical(item) for item in (start, trial, end)) + b"\n"
        self.assertEqual(len(collector._parse_jsonl(raw)), 3)
        trial["index"] = 2
        bad = b"\n".join(collector._canonical(item) for item in (start, trial, end)) + b"\n"
        with self.assertRaises(RuntimeError):
            collector._parse_jsonl(bad)
        trial["index"] = 1
        end["actual_open"] = True
        bad = b"\n".join(collector._canonical(item) for item in (start, trial, end)) + b"\n"
        with self.assertRaises(RuntimeError):
            collector._parse_jsonl(bad)


if __name__ == "__main__":
    unittest.main()
