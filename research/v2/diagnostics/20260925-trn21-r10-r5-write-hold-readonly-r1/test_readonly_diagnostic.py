"""Offline-only tests; no SSH, EC2, training, evaluation, or host writes."""

import contextlib
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
        reference, remote_source, bundle, diagnostic_source = operator._load_reference()
        self.assertEqual(hashlib.sha256(remote_source).hexdigest(), operator.R5_REMOTE_SHA256)
        self.assertEqual(hashlib.sha256(bundle).hexdigest(), operator.R5_BUNDLE_SHA256)
        self.assertEqual(len(bundle), operator.R5_BUNDLE_BYTES)
        self.assertEqual(hashlib.sha256(diagnostic_source).hexdigest(),
                         hashlib.sha256(operator.REMOTE_DIAGNOSTIC.read_bytes()).hexdigest())
        self.assertIn("_ssh_argv", reference)
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
                                  "phase": phase, "status": "PASS_READONLY",
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
                                             b"remote", b"diagnostic")
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
                                     b"remote", b"diagnostic")
        self.assertEqual(result["transport"], "HOLD")
        self.assertEqual(result["stage"], "local_stop_before_remote_gnu_probe")
        self.assertEqual(events, ["STOP", "PINNED_IDENTITY", "STOP"])
        transport.assert_not_called()

    def test_transport_classifies_without_leaking_raw_output(self):
        secret = b"SECRET_CONTENT_MUST_NOT_PRINT"
        result = operator.classify_transport("preopen", 124, secret, secret, 3.0)
        self.assertEqual(result["error_code"], "SSH_NONZERO")
        self.assertEqual(result["stdout_bytes"], len(secret))
        self.assertNotIn(secret.decode(), json.dumps(result))
        result = operator.classify_transport("preopen", 0, b"garbage", b"", 1.0)
        self.assertEqual(result["error_code"], "REMOTE_JSON_INVALID")
        result = operator.classify_transport("preopen", 0,
                     b"a" * (operator.MAX_REMOTE_STDOUT + 1), b"", 1.0)
        self.assertEqual(result["error_code"], "SSH_OUTPUT_OVERSIZE")

    def test_no_write_phase_can_be_built(self):
        with self.assertRaises(ValueError):
            operator._remote_command("write", None, b"remote", b"diagnostic")


if __name__ == "__main__":
    unittest.main()
