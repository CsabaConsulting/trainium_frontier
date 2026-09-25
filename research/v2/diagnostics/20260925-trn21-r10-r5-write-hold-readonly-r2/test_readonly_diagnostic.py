"""Offline-only tests; no SSH, EC2, training, evaluation, or host writes."""

import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remote = _module("trn21_r10_readonly_remote_test", "remote_readonly.py")
operator = _module("trn21_r10_readonly_operator_test", "operator_readonly.py")


def _template():
    return operator._archived_snapshot(operator._exact(operator.R5_JOURNAL,
                                                       operator.R5_JOURNAL_SHA256))


def _pass_inventory(phase, snapshot):
    return {"schema": "trn21-r10-r5-write-hold-readonly-diagnostic-v1",
            "phase": phase, "status": "PASS_READONLY", "stage": "completed",
            "snapshot": snapshot, "elapsed_seconds": 0.25,
            "remote_write_attempted": False,
            "source_stage_authorized": False,
            "native_training_authorized": False}


def _pass_preopen():
    return {"schema": "trn21-r10-r5-write-hold-readonly-diagnostic-v1",
            "phase": "preopen", "status": "PASS_READONLY", "stage": "completed",
            "snapshot_sha256": "f" * 64,
            "would_open": {"activity_age_seconds": 0.3,
                           "label_age_seconds": 0.4,
                           "space_age_seconds": 0.2,
                           "pre_open_broad_label_scan": True,
                           "post_label_activity_check": True,
                           "label_scan_duration_seconds": 0.1,
                           "root_space_before_open": {
                               "root_free_bytes": operator.PRE_FREE + 1024,
                               "root_free_inodes": operator.MIN_INODES + 1}},
            "trace": [{"check": name, "elapsed_seconds": 0.01}
                      for name in operator.TRACE_CHECKS],
            "bundle_exists": False, "target_exists": False,
            "elapsed_seconds": 0.3,
            "remote_write_attempted": False,
            "source_stage_authorized": False,
            "native_training_authorized": False}


def _transport(phase, receipt, template):
    return operator.classify_transport(phase, 0, json.dumps(receipt).encode("utf-8"),
                                       b"", 0.5, template)


class ReadonlyRemoteTests(unittest.TestCase):
    def _reference(self, root, data):
        calls = []
        reference = {
            "PRE_FREE": 100,
            "BUNDLE_BYTES": len(data),
            "BUNDLE_SHA256": hashlib.sha256(data).hexdigest(),
            "ROOT": root,
            "BUNDLE": root / "future-bundle",
            "TARGET": root / "future-target",
            "STOP_PATHS": [],
            "UID": os.getuid(),
            "GID": os.getgid(),
            "_timeout_identity": lambda: calls.append("timeout"),
            "_host_lock": lambda: contextlib.nullcontext({"existing_lock": True}),
            "_probe": lambda lock, minimum: {
                "schema": "synthetic-full-readonly-v1",
                "process_identity": {"pid": 2},
                "bundle_exists": False,
                "target_exists": False,
                "lock": lock,
                "minimum": minimum,
            },
            "_stable": lambda prior, snapshot: calls.append("stable"),
            "_activity": lambda: calls.append("activity"),
            "_label_scan": lambda: calls.append("label"),
            "_space": lambda minimum: {"root_free_bytes": 1000,
                                       "root_free_inodes": 1000},
            "_freshness_evidence": lambda samples, now: {
                "activity_age_seconds": now - samples["activity_at"],
                "label_age_seconds": now - samples["label_at"],
                "space_age_seconds": now - samples["space_at"],
                "label_scan_duration_seconds": 0.001,
            },
            "_write_bundle": mock.Mock(side_effect=AssertionError("writer called")),
            "_reconstruct": mock.Mock(side_effect=AssertionError("Git fetch called")),
        }

        def final():
            reference["_activity"]()
            reference["_label_scan"]()
            reference["_activity"]()
            reference["_space"](100)
            now = time.monotonic()
            sample = {"activity_at": now, "label_at": now, "space_at": now,
                      "label_scan_duration_seconds": 0.001}
            reference["_freshness_evidence"](sample, now)
            return sample

        reference["_final_no_write_continuity"] = final
        return reference, calls

    def test_preopen_proves_no_exclusive_or_write_open(self):
        data = b"synthetic bundle bytes"
        with tempfile.TemporaryDirectory(prefix="trn21-readonly-", dir="/dev/shm") as temp:
            reference, calls = self._reference(Path(temp), data)
            original_open = os.open
            opened_flags = []

            def checked_open(path, flags, *args, **kwargs):
                opened_flags.append(flags)
                forbidden = (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_EXCL |
                             os.O_TRUNC | os.O_APPEND)
                self.assertEqual(flags & forbidden, 0)
                return original_open(path, flags, *args, **kwargs)

            with mock.patch.object(remote.os, "open", side_effect=checked_open):
                result = remote.probe(reference, "preopen", {"prior": True}, data)
        self.assertEqual(result["status"], "PASS_READONLY", result)
        self.assertEqual(result["stage"], "completed")
        self.assertEqual(result["remote_write_attempted"], False)
        self.assertEqual(len(opened_flags), 1)
        self.assertIn("stable", calls)
        self.assertEqual(calls.count("activity"), 2)
        self.assertEqual(calls.count("label"), 1)
        self.assertEqual(len(result["trace"]), 5)
        reference["_write_bundle"].assert_not_called()
        reference["_reconstruct"].assert_not_called()

    def test_identity_drift_is_typed_without_raw_error_text(self):
        data = b"synthetic bundle bytes"
        with tempfile.TemporaryDirectory(prefix="trn21-readonly-", dir="/dev/shm") as temp:
            reference, _ = self._reference(Path(temp), data)

            def drift(prior, snapshot):
                raise RuntimeError("preflight-to-stage identity drift: mount SECRET_TOKEN")

            reference["_stable"] = drift
            result = remote.probe(reference, "preopen", {"prior": True}, data)
        self.assertEqual(result["status"], "HOLD_READONLY")
        self.assertEqual(result["stage"], "distinct_preflight_binding")
        self.assertEqual(result["error_code"], "PREFLIGHT_IDENTITY_DRIFT")
        self.assertNotIn("SECRET_TOKEN", json.dumps(result))

    def test_final_scan_expiry_is_typed_and_never_opens_bundle(self):
        data = b"synthetic bundle bytes"
        with tempfile.TemporaryDirectory(prefix="trn21-readonly-", dir="/dev/shm") as temp:
            reference, _ = self._reference(Path(temp), data)
            reference["_final_no_write_continuity"] = mock.Mock(
                side_effect=RuntimeError("final label scan too old for first O_EXCL"))
            result = remote.probe(reference, "preopen", {"prior": True}, data)
        self.assertEqual(result["status"], "HOLD_READONLY")
        self.assertEqual(result["error_code"], "FINAL_LABEL_SCAN_AGE")
        self.assertFalse(result["remote_write_attempted"])
        reference["_write_bundle"].assert_not_called()

    def test_preflight_and_postflight_reject_existing_stage_target(self):
        for phase in ("preflight", "postflight"):
            with self.subTest(phase=phase):
                reference = {"perform": lambda which, prior, data: {
                    "phase": which,
                    "snapshot": {"bundle_exists": True, "target_exists": False}}}
                result = remote.probe(reference, phase)
                self.assertEqual(result["status"], "HOLD_READONLY")
                self.assertEqual(result["error_code"], "TARGET_OR_BUNDLE_PRESENT")

    def test_unknown_phase_refuses(self):
        with self.assertRaises(ValueError):
            remote.probe({}, "write", {}, b"bytes")


class DisabledLocalOperatorTests(unittest.TestCase):
    def test_frozen_r5_references_match_and_main_is_disabled(self):
        reference, remote_source, bundle, diagnostic_source, template = operator._load_reference()
        self.assertEqual(hashlib.sha256(remote_source).hexdigest(), operator.R5_REMOTE_SHA256)
        self.assertEqual(hashlib.sha256(bundle).hexdigest(), operator.R5_BUNDLE_SHA256)
        self.assertEqual(len(bundle), operator.R5_BUNDLE_BYTES)
        self.assertEqual(hashlib.sha256(diagnostic_source).hexdigest(),
                         hashlib.sha256(operator.REMOTE_DIAGNOSTIC.read_bytes()).hexdigest())
        self.assertIn("_ssh_argv", reference)
        self.assertEqual(template["source"]["commit"], operator.SOURCE_COMMIT)
        self.assertFalse(operator.ENABLED_REQUEST.exists())
        with mock.patch.object(operator, "_load_reference") as source, \
             mock.patch.object(operator, "_phase") as phase:
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                operator.run_once()
        source.assert_not_called()
        phase.assert_not_called()

    def test_optimized_interpreter_refuses_before_authority(self):
        if not operator.sys.flags.optimize:
            return
        with mock.patch.object(operator, "_json") as read_authority, \
             mock.patch.object(operator, "_phase") as phase:
            with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                operator.run_once()
        read_authority.assert_not_called()
        phase.assert_not_called()

    def test_each_phase_uses_separate_remote_gnu_probe(self):
        for phase in ("preflight", "preopen", "postflight"):
            with self.subTest(phase=phase):
                events = []
                reference = {
                    "_local_stop_and_breakers": lambda: events.append("STOP"),
                    "_ssh_identity": lambda: events.append("PINNED_IDENTITY"),
                    "_remote_timeout_identity": lambda: events.append("SEPARATE_GNU_PROBE"),
                    "_ssh_argv": lambda command: ["synthetic-ssh", command],
                }
                remote_receipt = {"schema": "trn21-r10-r5-write-hold-readonly-diagnostic-v1",
                                  "phase": phase, "status": "HOLD_READONLY",
                                  "stage": "entry", "error_code": "OTHER_CHECK_FAILED",
                                  "exception_class": "RuntimeError",
                                  "elapsed_seconds": 0.1, "trace": [],
                                  "remote_write_attempted": False,
                                  "source_stage_authorized": False,
                                  "native_training_authorized": False}

                def fake_run(argv, **kwargs):
                    events.append("PHASE_SSH")
                    self.assertEqual(argv[0], "synthetic-ssh")
                    self.assertEqual(kwargs["input"], b"synthetic bundle")
                    return types.SimpleNamespace(returncode=0,
                        stdout=json.dumps(remote_receipt).encode("utf-8"), stderr=b"")

                with mock.patch.object(operator.subprocess, "run", side_effect=fake_run):
                    result = operator._phase(reference, phase, None, b"synthetic bundle",
                                             b"remote", b"diagnostic", _template())
                self.assertEqual(result["transport"], "PASS", result)
                self.assertEqual(events, ["STOP", "PINNED_IDENTITY", "STOP",
                                          "SEPARATE_GNU_PROBE", "STOP", "PHASE_SSH"])

    def test_stop_after_identity_blocks_gnu_probe_and_phase(self):
        events = []

        def stop():
            events.append("STOP")
            if events.count("STOP") == 2:
                raise RuntimeError("late STOP")

        reference = {"_local_stop_and_breakers": stop,
                     "_ssh_identity": lambda: events.append("PINNED_IDENTITY"),
                     "_remote_timeout_identity": lambda: events.append("SEPARATE_GNU_PROBE"),
                     "_ssh_argv": lambda command: ["synthetic-ssh", command]}
        with mock.patch.object(operator.subprocess, "run") as transport:
            result = operator._phase(reference, "preflight", None, b"",
                                     b"remote", b"diagnostic", _template())
        self.assertEqual(result["transport"], "HOLD")
        self.assertEqual(result["stage"], "local_stop_before_remote_gnu_probe")
        self.assertEqual(events, ["STOP", "PINNED_IDENTITY", "STOP"])
        transport.assert_not_called()

    def test_transport_classifies_without_leaking_raw_output(self):
        secret = b"SECRET_CONTENT_MUST_NOT_PRINT"
        result = operator.classify_transport("preopen", 124, secret, secret, 3.0,
                                             _template())
        self.assertEqual(result["error_code"], "SSH_NONZERO")
        self.assertEqual(result["stdout_bytes"], len(secret))
        self.assertNotIn(secret.decode(), json.dumps(result))
        result = operator.classify_transport("preopen", 0, b"garbage", b"", 1.0,
                                             _template())
        self.assertEqual(result["error_code"], "REMOTE_JSON_INVALID")
        result = operator.classify_transport("preopen", 0,
                     b"a" * (operator.MAX_REMOTE_STDOUT + 1), b"", 1.0,
                     _template())
        self.assertEqual(result["error_code"], "SSH_OUTPUT_OVERSIZE")

    def test_no_write_phase_can_be_built(self):
        with self.assertRaises(ValueError):
            operator._remote_command("write", None, b"remote", b"diagnostic")


class PhaseReceiptAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.template = _template()

    def _valid_triplet(self):
        first = copy.deepcopy(self.template)
        last = copy.deepcopy(self.template)
        last["process_identity"]["pid"] += 1
        preflight = _transport("preflight", _pass_inventory("preflight", first),
                               self.template)
        preopen = _transport("preopen", _pass_preopen(), self.template)
        postflight = _transport("postflight", _pass_inventory("postflight", last),
                                self.template)
        return preflight, preopen, postflight

    def _offline_run(self, replies):
        reference = {"_local_stop_and_breakers": lambda: None}
        with mock.patch.object(operator, "_json", side_effect=[
                ({"dispatch_authorized": True}, "request"), ({}, "review"),
                ({}, "assignment")]), \
             mock.patch.object(operator, "_load_reference", return_value=(
                 reference, b"archived remote", b"synthetic bundle",
                 b"read-only diagnostic", self.template)), \
             mock.patch.object(operator, "_regular", return_value=b"local source"), \
             mock.patch.object(operator, "_authority"), \
             mock.patch.object(operator, "_phase", side_effect=replies):
            return operator.run_once()

    def test_pinned_full_snapshot_and_positive_offline_receipt_path(self):
        self.assertTrue(operator._snapshot_valid(self.template, self.template,
                                                 "preflight"))
        replies = self._valid_triplet()
        self.assertTrue(all(row["transport"] == "PASS" for row in replies))
        if operator.sys.flags.optimize:
            for phase, reply in zip(("preflight", "preopen", "postflight"), replies):
                _, passed, _ = operator._public_phase(phase, reply, self.template)
                self.assertTrue(passed)
            return  # The real optimized entrypoint must remain disabled.
        result = self._offline_run(replies)
        self.assertEqual(result["verdict"],
                         "READONLY_DIAGNOSTIC_PASS_NOT_STAGE_AUTHORITY")
        self.assertEqual(result["full_snapshot_identity"], "PASS")
        self.assertTrue(result["distinct_preopen_snapshot"])
        for phase in result["phases"]:
            self.assertNotIn("_snapshot", phase)
            self.assertNotIn("snapshot", phase["remote"])
        self.assertNotIn("/home/ubuntu/", json.dumps(result))

    def test_original_missing_would_open_counterexample_now_holds(self):
        preflight, preopen, postflight = self._valid_triplet()
        malformed = copy.deepcopy(preopen)
        del malformed["remote"]["would_open"]
        if operator.sys.flags.optimize:
            public, passed, _ = operator._public_phase("preopen", malformed,
                                                       self.template)
            self.assertFalse(passed)
            self.assertEqual(public["error_code"], "PHASE_RECEIPT_INVALID")
            return
        result = self._offline_run((preflight, malformed, postflight))
        self.assertEqual(result["verdict"], "HOLD_READONLY_DIAGNOSTIC")
        self.assertEqual(result["phases"][1]["error_code"],
                         "PHASE_RECEIPT_INVALID")

    def test_original_missing_postflight_snapshot_counterexample_now_holds(self):
        preflight, preopen, postflight = self._valid_triplet()
        malformed = copy.deepcopy(postflight)
        del malformed["_snapshot"]
        if operator.sys.flags.optimize:
            public, passed, _ = operator._public_phase("postflight", malformed,
                                                       self.template)
            self.assertFalse(passed)
            self.assertEqual(public["error_code"], "PHASE_RECEIPT_INVALID")
            return
        result = self._offline_run((preflight, preopen, malformed))
        self.assertEqual(result["verdict"], "HOLD_READONLY_DIAGNOSTIC")
        self.assertEqual(result["phases"][-1]["error_code"],
                         "PHASE_RECEIPT_INVALID")

    def test_minimal_preflight_pass_cannot_start_preopen(self):
        minimal = {"phase": "preflight", "transport": "PASS",
                   "remote": {"status": "PASS_READONLY"}}
        postflight = self._valid_triplet()[2]
        if operator.sys.flags.optimize:
            public, passed, _ = operator._public_phase("preflight", minimal,
                                                       self.template)
            self.assertFalse(passed)
            self.assertEqual(public["error_code"], "PHASE_RECEIPT_INVALID")
            return
        result = self._offline_run((minimal, postflight))
        self.assertEqual(result["verdict"], "HOLD_READONLY_DIAGNOSTIC")
        self.assertEqual(len(result["phases"]), 2)
        self.assertEqual(result["phases"][0]["error_code"],
                         "PHASE_RECEIPT_INVALID")

    def test_phase_specific_missing_freshness_space_trace_and_raw_fields_hold(self):
        cases = {}
        cases["no_would_open"] = lambda row: row.pop("would_open")
        cases["age_over_one"] = lambda row: row["would_open"].__setitem__(
            "label_age_seconds", 1.001)
        cases["nan_age"] = lambda row: row["would_open"].__setitem__(
            "label_age_seconds", float("nan"))
        cases["root_bytes_below_floor"] = lambda row: row["would_open"][
            "root_space_before_open"].__setitem__("root_free_bytes", operator.PRE_FREE - 1)
        cases["root_inodes_below_floor"] = lambda row: row["would_open"][
            "root_space_before_open"].__setitem__("root_free_inodes", operator.MIN_INODES - 1)
        cases["no_timed_trace"] = lambda row: row.pop("trace")
        cases["wrong_trace_order"] = lambda row: row["trace"][1].__setitem__(
            "check", "space")
        cases["new_raw_field"] = lambda row: row.__setitem__(
            "extra_raw_message", "SECRET_DO_NOT_PUBLISH")
        for name, change in cases.items():
            with self.subTest(name=name):
                row = _pass_preopen()
                change(row)
                result = _transport("preopen", row, self.template)
                self.assertEqual(result["transport"], "HOLD")
                self.assertEqual(result["error_code"],
                                 "REMOTE_JSON_INVALID" if name == "nan_age" else
                                 "REMOTE_RECEIPT_INVALID")
                self.assertNotIn("SECRET_DO_NOT_PUBLISH", json.dumps(result))

    def test_full_inventory_shape_absence_and_floor_are_required(self):
        cases = {}
        cases["empty_snapshot"] = lambda row: row.__setitem__("snapshot", {})
        cases["missing_protected"] = lambda row: row["snapshot"].pop("protected")
        cases["nested_raw_field"] = lambda row: row["snapshot"][
            "source"].__setitem__("raw_path", "SECRET_DO_NOT_PUBLISH")
        cases["bundle_present"] = lambda row: row["snapshot"].__setitem__(
            "bundle_exists", True)
        cases["root_bytes_below_floor"] = lambda row: row["snapshot"][
            "identity_after"].__setitem__("root_free_bytes", operator.PRE_FREE - 1)
        cases["root_inodes_below_floor"] = lambda row: row["snapshot"][
            "identity_before"].__setitem__("root_free_inodes", operator.MIN_INODES - 1)
        for name, change in cases.items():
            with self.subTest(name=name):
                row = _pass_inventory("preflight", copy.deepcopy(self.template))
                change(row)
                result = _transport("preflight", row, self.template)
                self.assertEqual(result["transport"], "HOLD")
                self.assertEqual(result["error_code"], "REMOTE_RECEIPT_INVALID")
                self.assertNotIn("SECRET_DO_NOT_PUBLISH", json.dumps(result))

    def test_full_preflight_postflight_identity_must_match(self):
        preflight, preopen, _ = self._valid_triplet()
        last = copy.deepcopy(self.template)
        last["process_identity"]["pid"] += 1
        last["mount"]["sha256"] = "0" * 64
        postflight = _transport("postflight", _pass_inventory("postflight", last),
                                self.template)
        self.assertEqual(postflight["transport"], "PASS")
        if operator.sys.flags.optimize:
            self.assertFalse(operator._same_full_inventory_identity(
                preflight["_snapshot"], postflight["_snapshot"]))
            return
        result = self._offline_run((preflight, preopen, postflight))
        self.assertEqual(result["verdict"], "HOLD_READONLY_DIAGNOSTIC")
        self.assertEqual(result["full_snapshot_identity"], "HOLD")

    def test_hold_and_pass_extra_raw_fields_never_escape(self):
        hold = {"schema": "trn21-r10-r5-write-hold-readonly-diagnostic-v1",
                "phase": "preopen", "status": "HOLD_READONLY", "stage": "entry",
                "error_code": "OTHER_CHECK_FAILED", "exception_class": "RuntimeError",
                "elapsed_seconds": 0.2, "trace": [],
                "remote_write_attempted": False,
                "source_stage_authorized": False,
                "native_training_authorized": False,
                "extra_raw_message": "SECRET_DO_NOT_PUBLISH"}
        for row in (hold, {**_pass_preopen(),
                           "extra_raw_message": "SECRET_DO_NOT_PUBLISH"}):
            result = _transport("preopen", row, self.template)
            self.assertEqual(result["transport"], "HOLD")
            self.assertNotIn("SECRET_DO_NOT_PUBLISH", json.dumps(result))

    def test_duplicate_json_key_is_rejected(self):
        raw = b'{"phase":"preopen","phase":"preopen"}'
        result = operator.classify_transport("preopen", 0, raw, b"", 0.1,
                                             self.template)
        self.assertEqual(result["error_code"], "REMOTE_JSON_INVALID")


if __name__ == "__main__":
    unittest.main()
