"""Offline r6 adversarial checks; disposable local paths, no SSH or host writes."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
remote_spec = importlib.util.spec_from_file_location("trn22_r6_remote", HERE / "remote_incremental.py")
remote = importlib.util.module_from_spec(remote_spec)
remote_spec.loader.exec_module(remote)
operator_spec = importlib.util.spec_from_file_location("trn22_r6_operator", HERE / "operator_once.py")
operator = importlib.util.module_from_spec(operator_spec)
operator_spec.loader.exec_module(operator)


def samples():
    now = time.monotonic()
    return {"activity_at": now, "label_at": now, "space_at": now,
            "label_scan_duration_seconds": 0.01,
            "root_space_before_open": {"root_free_bytes": remote.PRE_FREE + 4096,
                                       "root_free_inodes": remote.MIN_INODES + 10}}


class PreopenOnlyRetryTests(unittest.TestCase):
    def exercise(self, failures, *, post_open_fail=False,
                 prior_r1_artifact=False):
        with tempfile.TemporaryDirectory(prefix="trn22-r6-preopen-") as dirname:
            root = Path(dirname)
            bundle, target = root / "fresh-r2.bundle", root / "fresh-r2-target"
            payload = b"synthetic exact bundle"
            opened = []
            real_open = os.open
            if prior_r1_artifact:
                (root / "old-r1.bundle").write_bytes(b"preserved old evidence")

            def counted_open(path, flags, *args, **kwargs):
                if path == bundle.name and flags & os.O_EXCL:
                    opened.append(path)
                return real_open(path, flags, *args, **kwargs)

            final = mock.Mock(side_effect=[
                *(remote.PreOpenFreshnessExpired("pre-open timing expired")
                  if code == "TIME" else
                  remote.StageGateError("STOP or holder appeared", code)
                  for code in failures),
                samples(),
            ])
            with mock.patch.object(remote, "ROOT", root), \
                 mock.patch.object(remote, "BUNDLE", bundle), \
                 mock.patch.object(remote, "TARGET", target), \
                 mock.patch.object(remote, "PRIOR_R1_BUNDLE", root / "old-r1.bundle"), \
                 mock.patch.object(remote, "PRIOR_R1_TARGET", root / "old-r1-target"), \
                 mock.patch.object(remote, "BUNDLE_BYTES", len(payload)), \
                 mock.patch.object(remote, "BUNDLE_SHA256",
                                   hashlib.sha256(payload).hexdigest()), \
                 mock.patch.object(remote, "_final_no_write_continuity", final), \
                 mock.patch.object(remote, "_label_scan", return_value={}), \
                 mock.patch.object(remote, "_activity", return_value={}), \
                 mock.patch.object(remote, "_space", return_value={
                     "root_free_bytes": remote.PRE_FREE + 4096,
                     "root_free_inodes": remote.MIN_INODES + 10}), \
                 mock.patch.object(remote, "STOP_PATHS", []), \
                 mock.patch.object(remote, "_checked_bundle", return_value={}), \
                 mock.patch.object(remote.os, "open", side_effect=counted_open):
                if post_open_fail:
                    actual = remote._freshness_evidence

                    def freshness(value, now, *, pre_open=False):
                        if not pre_open:
                            raise remote.StageGateError(
                                "final proof too old at O_EXCL return", "OTHER_GATE")
                        return actual(value, now, pre_open=True)

                    patch = mock.patch.object(remote, "_freshness_evidence",
                                              side_effect=freshness)
                else:
                    patch = mock.patch.object(remote, "_freshness_evidence",
                                              wraps=remote._freshness_evidence)
                with patch:
                    try:
                        proof = remote._write_bundle(payload)
                        failure = None
                    except RuntimeError as error:
                        proof, failure = None, error
            return {"proof": proof, "failure": failure, "final_calls": final.call_count,
                    "exclusive_opens": len(opened), "bundle_exists": bundle.exists(),
                    "target_exists": target.exists(),
                    "bundle_bytes": bundle.read_bytes() if bundle.exists() else None}

    def test_one_preopen_timing_expiry_rechecks_then_opens_once(self):
        result = self.exercise(["TIME"])
        self.assertIsNone(result["failure"])
        self.assertEqual((result["final_calls"], result["exclusive_opens"]), (2, 1))
        self.assertEqual(result["proof"]["preopen_attempts"], 2)
        self.assertTrue(result["target_exists"])
        self.assertEqual(result["bundle_bytes"], b"synthetic exact bundle")

    def test_three_preopen_timing_expiries_never_open(self):
        result = self.exercise(["TIME"] * remote.MAX_PREOPEN_ATTEMPTS)
        self.assertIsInstance(result["failure"], remote.PreOpenFreshnessExpired)
        self.assertEqual(result["failure"].failure_code, "PREOPEN_TIMING_EXPIRED")
        self.assertEqual((result["final_calls"], result["exclusive_opens"]), (3, 0))
        self.assertFalse(result["bundle_exists"])
        self.assertFalse(result["target_exists"])

    def test_non_timing_stop_or_holder_never_retries(self):
        for code in ("STOP_GATE", "ACTIVITY_GATE", "LABEL_GATE",
                     "SPACE_GATE", "IDENTITY_GATE", "DATA_GATE"):
            with self.subTest(code=code):
                result = self.exercise(["TIME", code])
                self.assertEqual(result["failure"].failure_code, code)
                self.assertEqual((result["final_calls"], result["exclusive_opens"]),
                                 (2, 0))
                self.assertFalse(result["bundle_exists"])

    def test_post_open_timing_refusal_keeps_empty_bundle_without_retry(self):
        result = self.exercise([], post_open_fail=True)
        self.assertIsInstance(result["failure"], remote.StageGateError)
        self.assertEqual((result["final_calls"], result["exclusive_opens"]), (1, 1))
        self.assertEqual(result["bundle_bytes"], b"")
        self.assertFalse(result["target_exists"])

    def test_old_r1_artifact_forces_hold_without_new_open(self):
        result = self.exercise([], prior_r1_artifact=True)
        self.assertIn("target already exists", str(result["failure"]))
        self.assertEqual((result["final_calls"], result["exclusive_opens"]), (0, 0))
        self.assertFalse(result["bundle_exists"])


class NonsecretDiagnosticTests(unittest.TestCase):
    def phase(self, reply):
        with mock.patch.object(operator, "_local_stop_and_breakers"), \
             mock.patch.object(operator, "_ssh_identity"), \
             mock.patch.object(operator, "_remote_timeout_identity"), \
             mock.patch.object(operator.subprocess, "run", return_value=reply):
            return operator._ssh_phase("write", {}, b"synthetic", b"# inert source\n")

    def test_structured_remote_timing_hold_without_stderr(self):
        hold = {"phase": "write", "remote_status": "HOLD",
                "failure_code": "PREOPEN_TIMING_EXPIRED"}
        reply = types.SimpleNamespace(returncode=70, stderr=b"",
                                      stdout=json.dumps(hold).encode())
        with self.assertRaises(operator.RemotePhaseHold) as raised:
            self.phase(reply)
        self.assertEqual(raised.exception.classification["category"],
                         "REMOTE_GATE_HOLD")
        self.assertEqual(raised.exception.classification["remote_code"],
                         "PREOPEN_TIMING_EXPIRED")
        self.assertNotIn("synthetic", str(raised.exception))

    def test_actual_local_remote_wrapper_suppresses_exception_text(self):
        source = (b"class Refusal(RuntimeError):\n"
                  b" failure_code='STOP_GATE'\n"
                  b"def perform(phase, prior, data):\n"
                  b" raise Refusal('synthetic-secret-must-not-escape')\n")
        real_run = subprocess.run

        def local_inert_process(argv, **kwargs):
            parts = shlex.split(argv[-1])
            program = parts[parts.index("-c") + 1]
            return real_run(["/usr/bin/python3", "-I", "-B", "-c",
                             program, "write", "{}"],
                            input=kwargs["input"], capture_output=True,
                            timeout=5, check=False)

        with mock.patch.object(operator, "_local_stop_and_breakers"), \
             mock.patch.object(operator, "_ssh_identity"), \
             mock.patch.object(operator, "_remote_timeout_identity"), \
             mock.patch.object(operator.subprocess, "run",
                               side_effect=local_inert_process):
            with self.assertRaises(operator.RemotePhaseHold) as raised:
                operator._ssh_phase("write", {}, b"", source)
        self.assertEqual(raised.exception.classification["remote_code"], "STOP_GATE")
        self.assertNotIn("synthetic-secret", str(raised.exception))
        self.assertNotIn("synthetic-secret",
                         json.dumps(raised.exception.classification))

    def test_unknown_remote_stderr_remains_secret_and_categorized(self):
        secret = b"private-synthetic-token-not-for-journal"
        reply = types.SimpleNamespace(returncode=1, stderr=secret, stdout=b"")
        with self.assertRaises(operator.RemotePhaseHold) as raised:
            self.phase(reply)
        classification = raised.exception.classification
        self.assertEqual(classification["category"], "REMOTE_NONZERO")
        self.assertEqual(classification["return_code"], 1)
        self.assertEqual(classification["stderr_bytes"], len(secret))
        self.assertNotIn(secret.decode(), str(raised.exception))
        self.assertNotIn(secret.decode(), json.dumps(classification))

    def test_ssh_transport_timeout_and_malformed_success_are_distinct(self):
        cases = ((255, b"", b"transport secret", "SSH_TRANSPORT"),
                 (124, b"", b"", "REMOTE_TIMEOUT"),
                 (0, b"not-json", b"", "STDOUT_MALFORMED"))
        for code, stdout, stderr, category in cases:
            with self.subTest(category=category):
                reply = types.SimpleNamespace(returncode=code,
                                              stderr=stderr, stdout=stdout)
                with self.assertRaises(operator.RemotePhaseHold) as raised:
                    self.phase(reply)
                self.assertEqual(raised.exception.classification["category"], category)

    def test_consumed_r1_marker_must_match_without_reuse(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r6-marker-") as dirname:
            root = Path(dirname)
            old = root / "old-r1.one-attempt.jsonl"
            new = root / "new-r2.one-attempt.jsonl"
            old.write_bytes(b"synthetic-consumed-r1")
            old.chmod(0o600)
            with mock.patch.object(operator, "CONSUMED_R1_MARKER", old), \
                 mock.patch.object(operator, "CONSUMED_R1_MARKER_BYTES", old.stat().st_size), \
                 mock.patch.object(operator, "CONSUMED_R1_MARKER_SHA256",
                                   hashlib.sha256(old.read_bytes()).hexdigest()), \
                 mock.patch.object(operator, "MARKER", new), \
                 mock.patch.object(operator, "PRIOR_MARKERS", ()):
                fd = operator._marker("request", "review", "assignment")
                os.close(fd)
                self.assertTrue(old.exists())
                self.assertTrue(new.exists())
                old.write_bytes(b"changed-old-marker")
                with self.assertRaisesRegex(RuntimeError, "consumed r1 marker bytes changed"):
                    operator._consumed_r1_marker_evidence()


if __name__ == "__main__":
    unittest.main()
