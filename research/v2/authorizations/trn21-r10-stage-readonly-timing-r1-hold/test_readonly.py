"""Offline-only regressions for gated trn21 timing. No SSH or host use."""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
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
invoker = load("trn21_r10_timing_invoker_test", HERE / "invoke_timing_readonly.py")


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
               "utc": utc, "preopen_seconds": 0.8,
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
                "boot_id": "30f90d95-224c-48d9-bb11-8ec18dd7c4fe",
                "mount": {"namespace": "mnt:[1]"},
                "identity_after": {"root_free_bytes": minimum + 1000,
                                   "root_free_inodes": 10001}}

    def default_final():
        calls.append("exact_final_no_write_continuity")
        import time
        now = time.monotonic()
        return {"activity_at": now, "label_at": now, "space_at": now,
                "label_scan_duration_seconds": 0.0,
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
    def test_enabled_request_still_needs_distinct_final_review_and_assignment(self):
        inputs = collector._pins()
        self.assertEqual(len(inputs), 6)
        self.assertTrue(collector.ENABLED.exists())
        self.assertFalse(collector.REVIEW.exists())
        self.assertFalse(collector.ASSIGNMENT.exists())
        self.assertFalse(collector.OUTPUT.exists())
        enabled_raw, enabled = collector._exact_json(collector.ENABLED)
        self.assertEqual(enabled_raw, collector._canonical(enabled) + b"\n")
        self.assertEqual(enabled["collector_sha256"],
                         hashlib.sha256(Path(collector.__file__).read_bytes()).hexdigest())
        self.assertEqual(enabled["invoker_sha256"],
                         hashlib.sha256(Path(invoker.__file__).read_bytes()).hexdigest())
        self.assertEqual(enabled["timing_r3_review_sha256"],
                         collector.TIMING_REVIEW_SHA256)
        self.assertEqual(enabled["execution_root"], str(invoker.STAGE_ROOT))
        self.assertEqual(enabled["execution_head"], invoker.STAGE_HEAD)
        self.assertEqual(enabled["output_basename"], collector.OUTPUT.name)
        self.assertEqual(enabled["endpoint"], collector.ENDPOINT)
        self.assertEqual(enabled["fingerprint"], collector.FINGERPRINT)
        self.assertEqual(enabled["known_hosts_sha256"], collector.KNOWN_HOSTS_SHA256)
        self.assertTrue(enabled["host_contact_authorized"])
        for name in ("host_write_authorized", "host_process_signals_authorized",
                     "archive_target_open_authorized", "one_shot_marker_authorized",
                     "training_authorized", "evaluation_authorized",
                     "aws_action_authorized"):
            self.assertFalse(enabled[name])
        with mock.patch.object(collector, "_reject_optimized"), \
             mock.patch.object(collector, "_ssh_identity") as identity, \
             mock.patch.object(collector, "_execution_root_gate") as root_gate, \
             mock.patch.object(collector.subprocess, "run") as process:
            with self.assertRaises(FileNotFoundError):
                collector.capture()
            root_gate.assert_not_called()
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
        self.assertEqual(len(collector._parse_jsonl(jsonl(records))),
                         collector.TRIALS + 2)
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
                 mock.patch.object(collector, "_reject_optimized"), \
                 mock.patch.object(collector, "_execution_root_gate", return_value={}), \
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
                ("trial", "activity_age_seconds", float("inf")),
                ("trial", "label_age_seconds", float("nan")),
                ("trial", "label_age_seconds", -0.001),
                ("trial", "space_age_seconds", float("-inf")),
                ("trial", "space_age_seconds", 1.001),
                ("trial", "label_scan_duration_seconds", float("nan")),
                ("trial", "label_scan_duration_seconds", 1.001),
                ("trial", "preopen_seconds", -1),
                ("trial", "preopen_seconds", float("nan")),
                ("trial", "preopen_seconds", float("inf")),
                ("trial", "preopen_seconds", collector.MAX_REMOTE_SECONDS + 1),
                ("trial", "root_free_bytes", 1),
                ("start", "identity", None),
                ("start", "boot_id", "not-a-boot-id"),
                ("start", "root_free_inodes", 9999)):
            with self.subTest(kind=kind, field=field):
                records = complete_records()
                records[1 if kind == "trial" else 0][field] = value
                with self.assertRaises(RuntimeError):
                    collector._parse_jsonl(jsonl(records))

    def test_forged_ordered_jsonl_rejects_impossible_remote_chronology(self):
        for field, value in (("preopen_seconds", 0.0),
                             ("label_age_seconds", 0.19),
                             ("activity_age_seconds", 0.09)):
            with self.subTest(field=field):
                records = complete_records()
                records[1][field] = value
                with self.assertRaisesRegex(RuntimeError, "chronology"):
                    collector._parse_jsonl(jsonl(records))

        # Subtracting large monotonic float readings can lose a few low bits.
        # The allowance is one microsecond, never a material part of the window.
        near_boundary = complete_records()
        near_boundary[1]["preopen_seconds"] = (
            0.7 - collector.CHRONOLOGY_TOLERANCE_SECONDS / 2)
        self.assertEqual(len(collector._parse_jsonl(jsonl(near_boundary))),
                         collector.TRIALS + 2)
        near_boundary[1]["preopen_seconds"] = (
            0.7 - 2 * collector.CHRONOLOGY_TOLERANCE_SECONDS)
        with self.assertRaisesRegex(RuntimeError, "chronology"):
            collector._parse_jsonl(jsonl(near_boundary))

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

    def test_impossible_remote_chronology_archives_hold_not_observation(self):
        forged = complete_records()
        forged[1]["preopen_seconds"] = 0.0
        with tempfile.TemporaryDirectory(prefix="trn21-timing-chronology-") as temp:
            archived = self._capture_mocked(jsonl(forged), b"", Path(temp))
            self.assertEqual(archived["status"], "HOLD")
            self.assertIsNone(archived["parsed"])
            self.assertIn("chronology", archived["parse_error"])

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

    def test_late_local_stop_or_root_drift_archives_hold(self):
        stable_controller = {"active_state_sha256": "stable"}
        stable_root = {"head": "stable"}
        for kind, controller_states, root_states in (
                ("controller_stop",
                 [stable_controller] * 3 + [RuntimeError("local STOP appeared")],
                 None),
                ("root_drift", None,
                 [stable_root] * 3 + [{"head": "changed"}])):
            with self.subTest(kind=kind), \
                 tempfile.TemporaryDirectory(prefix="trn21-timing-late-gate-") as temp:
                archived = self._capture_mocked(
                    jsonl(complete_records()), b"", Path(temp),
                    controller_states=controller_states, root_states=root_states)
                self.assertEqual(archived["status"], "HOLD")
                field = ("active_controller_stable" if kind == "controller_stop"
                         else "execution_root_stable")
                self.assertFalse(archived[field])

    def test_fixed_local_result_consumes_mocked_one_attempt(self):
        with tempfile.TemporaryDirectory(prefix="trn21-timing-once-") as temp:
            root = Path(temp)
            self._capture_mocked(jsonl(complete_records()), b"", root)
            output = root / collector.OUTPUT.name
            with mock.patch.object(collector, "HERE", root), \
                 mock.patch.object(collector, "OUTPUT", output), \
                 mock.patch.object(collector, "_pins", return_value={}), \
                 mock.patch.object(collector, "_authority", return_value={}), \
                 mock.patch.object(collector, "_reject_optimized"), \
                 mock.patch.object(collector, "_execution_root_gate", return_value={}), \
                 mock.patch.object(collector, "_controller_gate", return_value={}), \
                 mock.patch.object(collector, "_ssh_identity", return_value={}), \
                 mock.patch.object(collector.subprocess, "run") as process:
                with self.assertRaisesRegex(RuntimeError, "prior local timing receipt"):
                    collector.capture()
                process.assert_not_called()

    def test_synthetic_final_review_and_assignment_are_both_required(self):
        inputs = collector._pins()
        enabled_raw, _ = collector._exact_json(collector.ENABLED)
        review = {
            "schema": collector.SCHEMA + "-review", "role": "independent_verifier",
            "verdict": "PASS_READONLY_TIMING_ONLY", "host": "trn21",
            "scope": collector.SCOPE, "enabled_sha256": collector._sha(enabled_raw),
            "collector_sha256": hashlib.sha256(Path(collector.__file__).read_bytes()).hexdigest(),
            "invoker_sha256": hashlib.sha256(Path(invoker.__file__).read_bytes()).hexdigest(),
            "timing_r3_review_sha256": collector.TIMING_REVIEW_SHA256,
            "timing_source_sha256": collector.TIMING_SHA256,
            "stage_remote_sha256": collector.STAGE_SHA256,
            "execution_root": str(invoker.STAGE_ROOT),
            "execution_head": invoker.STAGE_HEAD,
            "output_basename": collector.OUTPUT.name,
        }
        with tempfile.TemporaryDirectory(prefix="trn21-timing-fake-gates-") as temp:
            root = Path(temp)
            fake_review = root / "synthetic-review.json"
            fake_assignment = root / "synthetic-assignment.json"
            with mock.patch.object(collector, "REVIEW", fake_review), \
                 mock.patch.object(collector, "ASSIGNMENT", fake_assignment), \
                 mock.patch.object(collector, "_reject_optimized"), \
                 mock.patch.object(collector, "_execution_root_gate") as root_gate, \
                 mock.patch.object(collector, "_ssh_identity") as identity, \
                 mock.patch.object(collector.subprocess, "run") as process:
                with self.assertRaises(FileNotFoundError):
                    collector.capture()
                fake_review.write_bytes(collector._canonical(review) + b"\n")
                with self.assertRaises(FileNotFoundError):
                    collector.capture()
                fake_assignment.write_bytes(collector._canonical({
                    "schema": collector.SCHEMA + "-assignment",
                    "role": "supervisor", "status": "ASSIGNED_ONE_READONLY_CAPTURE",
                    "host": "trn21", "scope": collector.SCOPE,
                    "enabled_sha256": collector._sha(enabled_raw),
                    "review_sha256": "0" * 64,
                    "max_trials": collector.TRIALS,
                    "output_basename": collector.OUTPUT.name,
                    "one_attempt": True}) + b"\n")
                with self.assertRaisesRegex(RuntimeError, "assignment missing or mismatched"):
                    collector.capture()
                root_gate.assert_not_called()
                identity.assert_not_called()
                process.assert_not_called()

    def test_optimized_capture_refuses_before_pins_or_ssh(self):
        with mock.patch.dict(os.environ, {"PYTHONOPTIMIZE": "1"}), \
             mock.patch.object(collector, "_pins") as pins, \
             mock.patch.object(collector.subprocess, "run") as process:
            with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                collector.capture()
            pins.assert_not_called()
            process.assert_not_called()

    def test_direct_capture_wrong_root_refuses_before_controller_or_ssh(self):
        with tempfile.TemporaryDirectory(prefix="trn21-timing-wrong-root-") as temp:
            wrong_root = Path(temp)

            def real_wrong_root_gate():
                return invoker.check_execution_root(wrong_root, "0" * 40, {})

            with mock.patch.object(collector, "_reject_optimized"), \
                 mock.patch.object(collector, "_pins", return_value={}), \
                 mock.patch.object(collector, "_authority", return_value={}), \
                 mock.patch.object(collector, "_execution_root_gate",
                                   side_effect=real_wrong_root_gate), \
                 mock.patch.object(collector, "_controller_gate") as controller, \
                 mock.patch.object(collector, "_ssh_identity") as identity, \
                 mock.patch.object(collector.subprocess, "run") as process:
                with self.assertRaisesRegex(RuntimeError, "exact execution root"):
                    collector.capture()
                controller.assert_not_called()
                identity.assert_not_called()
                process.assert_not_called()

    def _capture_mocked(self, stdout, stderr, root, *,
                        controller_states=None, root_states=None):
        response = types.SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)
        output = root / collector.OUTPUT.name
        pending = []

        def fake_contact(*_args, **_kwargs):
            self.assertTrue(output.exists())
            pending.append(json.loads(output.read_text(encoding="utf-8"))["status"])
            return response

        with mock.patch.object(collector, "HERE", root), \
             mock.patch.object(collector, "OUTPUT", output), \
             mock.patch.object(collector, "_pins", return_value={
                 "stage": b"stage", "timing": b"timing", "hold": b"hold"}), \
             mock.patch.object(collector, "_authority", return_value={}), \
             mock.patch.object(collector, "_reject_optimized"), \
             mock.patch.object(collector, "_execution_root_gate",
                               return_value={"head": "stable"},
                               side_effect=root_states), \
             mock.patch.object(collector, "_controller_gate",
                               return_value={"active_state_sha256": "stable"},
                               side_effect=controller_states), \
             mock.patch.object(collector, "_ssh_identity", return_value={}), \
             mock.patch.object(collector, "_ssh_argv", return_value=["ssh", "dummy"]), \
             mock.patch.object(collector.subprocess, "run",
                               side_effect=fake_contact) as process:
            receipt = collector.capture()
        process.assert_called_once()
        self.assertEqual(pending, ["HOLD_IN_PROGRESS"])
        self.assertEqual(receipt, output)
        return json.loads(receipt.read_text(encoding="utf-8"))


class ExecutionRootTests(unittest.TestCase):
    def test_pinned_trn21_science_and_stage_bytes_are_not_trn22(self):
        self.assertEqual(invoker.STAGE_ROOT,
                         Path("/home/poojith-devan/Desktop/Trainium-trn21-r10-stage-r3-active-root"))
        self.assertEqual(invoker.STAGE_HEAD, collector.STAGE_COMMIT)
        self.assertEqual(collector.EXECUTION_ROOT, invoker.STAGE_ROOT)
        self.assertIn("ec2-16-26-246-98", collector.ENDPOINT)
        self.assertNotIn("trn22", str(invoker.STAGE_ROOT))
        for relative, digest in invoker.STAGE_FILES.items():
            self.assertEqual(hashlib.sha256((collector.REPO / relative).read_bytes()).hexdigest(),
                             digest, relative)

    def test_clean_synthetic_root_then_wrong_head_bytes_dirty_stop_and_alias(self):
        with tempfile.TemporaryDirectory(prefix="trn21-timing-root-") as temp:
            base = Path(temp)
            root = base / "stage"
            root.mkdir()
            fixture = root / "fixture.txt"
            fixture.write_text("synthetic source\n", encoding="ascii")
            env = {**invoker.GIT_ENV, "GIT_AUTHOR_NAME": "Synthetic",
                   "GIT_AUTHOR_EMAIL": "synthetic@example.invalid",
                   "GIT_COMMITTER_NAME": "Synthetic",
                   "GIT_COMMITTER_EMAIL": "synthetic@example.invalid"}
            subprocess.run(["/usr/bin/git", "init", "-q", str(root)],
                           env=env, capture_output=True, check=True)
            subprocess.run(["/usr/bin/git", "-C", str(root), "add", "fixture.txt"],
                           env=env, capture_output=True, check=True)
            subprocess.run(["/usr/bin/git", "-C", str(root), "commit", "-qm", "fixture"],
                           env=env, capture_output=True, check=True)
            head = invoker._git(root, "rev-parse", "HEAD").decode("ascii")
            source = {"fixture.txt": hashlib.sha256(fixture.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(RuntimeError, "exact execution root"):
                invoker.check_execution_root(root, head, source)
            former = Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(invoker.check_execution_root(root, head, source)["head"], head)
                with self.assertRaisesRegex(RuntimeError, "HEAD changed"):
                    invoker.check_execution_root(root, "0" * 40, source)
                with self.assertRaisesRegex(RuntimeError, "source bytes changed"):
                    invoker.check_execution_root(root, head, {"fixture.txt": "0" * 64})
                dirty = root / "untracked.txt"
                dirty.write_text("dirty\n", encoding="ascii")
                with self.assertRaisesRegex(RuntimeError, "dirty"):
                    invoker.check_execution_root(root, head, source)
                dirty.unlink()
                (root / "research/v2").mkdir(parents=True)
                stop = root / "research/v2/STOP"
                stop.write_text("synthetic STOP\n", encoding="ascii")
                with self.assertRaisesRegex(RuntimeError, "STOP present"):
                    invoker.check_execution_root(root, head, source)
                stop.unlink()
                alias = base / "alias"
                alias.symlink_to(root, target_is_directory=True)
                with self.assertRaisesRegex(RuntimeError, "not exact and real"):
                    invoker.check_execution_root(alias, head, source)
            finally:
                os.chdir(former)

    def test_wrapper_refuses_before_ssh_without_final_review(self):
        with mock.patch.object(invoker, "check_execution_root") as root_gate, \
             mock.patch.object(invoker.sys, "argv", [str(invoker.__file__)]), \
             mock.patch.dict(sys.modules, {"collect_readonly": collector}), \
             mock.patch.object(collector, "_reject_optimized"), \
             mock.patch.object(collector, "_ssh_identity") as identity, \
             mock.patch.object(collector.subprocess, "run") as process:
            if sys.flags.optimize:
                with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                    invoker.main()
                root_gate.assert_not_called()
            else:
                with self.assertRaises(FileNotFoundError):
                    invoker.main()
                root_gate.assert_called_once_with()
            identity.assert_not_called()
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
