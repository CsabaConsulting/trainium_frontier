"""Offline adversarial checks; no SSH, Trainium, evaluation or host mutation."""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = _load("r10_readonly_remote_test", HERE / "remote_readonly.py")
operator = _load("r10_readonly_operator_test", HERE / "operator_readonly.py")


def _snapshot(pid):
    return {"schema": "trn22-r10-incremental-full-readonly-v1",
            "hostname": "ip-172-31-32-253", "uid": 1000, "boot_id": "fixed-boot",
            "mount": {"fixed": True}, "lock": {"fixed": True},
            "protected": {"fixed": True}, "process_identity": {"pid": pid},
            "source": {"commit": operator.BASE_COMMIT,
                       "science_sha256": {"train.py": operator.TRAIN_SHA256,
                                          "prepare.py": operator.PREPARE_SHA256}},
            "identity_after": {"root_free_bytes": operator.PRE_FREE + 1,
                               "root_free_inodes": operator.MIN_INODES + 1},
            "activity": {"neuron_holder_absent": True, "stops_absent": True},
            "label": {"matches": []}, "bundle_exists": False,
            "target_exists": False}


class FakeStage:
    def __init__(self):
        self.calls = []
        self.root = Path("/tmp")
        stat_root = os.lstat(self.root)
        self.bundle = self.root / "r10-diagnostic-never-create-bundle"
        self.target = self.root / "r10-diagnostic-never-create-target"
        self.data = b"fixed bundle bytes"
        self.values = {
            "PRE_FREE": 1,
            "BUNDLE_BYTES": len(self.data),
            "BUNDLE_SHA256": hashlib.sha256(self.data).hexdigest(),
            "ROOT": self.root,
            "BUNDLE": self.bundle,
            "TARGET": self.target,
            "DEV": stat_root.st_dev,
            "UID": stat_root.st_uid,
            "GID": stat_root.st_gid,
            "_host_lock": self.lock,
            "_probe": self.probe,
            "_stable": self.stable,
            "_final_no_write_continuity": self.final,
            "_freshness_evidence": self.fresh,
        }

    @contextmanager
    def lock(self):
        self.calls.append("lock_enter")
        try:
            yield {"same_flock": True}
        finally:
            self.calls.append("lock_exit")

    def probe(self, lock, *, minimum):
        self.calls.append("probe")
        if lock != {"same_flock": True} or minimum != 1:
            raise RuntimeError("probe arguments changed")
        return {"bundle_exists": False, "target_exists": False,
                "process_identity": {"pid": 2}}

    def stable(self, prior, current):
        self.calls.append("stable")
        if prior != {"prior": True} or current["process_identity"]["pid"] != 2:
            raise RuntimeError("preflight-to-stage identity drift")

    def final(self):
        self.calls.append("final")
        return {"activity_at": 0.0, "label_at": 0.0,
                "space_at": 0.0, "label_scan_duration_seconds": 0.2}

    def fresh(self, samples, now):
        self.calls.append("fresh")
        if not isinstance(now, float):
            raise RuntimeError("missing monotonic sample")
        return {"activity_age_seconds": 0.1, "label_age_seconds": 0.1,
                "space_age_seconds": 0.1,
                "label_scan_duration_seconds": 0.2,
                "double_label_scan": True,
                "post_label_activity_check": True}


class RemoteReadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.stage = FakeStage()

    def _run(self, data=None):
        with mock.patch.object(diag.os.path, "lexists", return_value=False):
            return diag.perform_readonly({"prior": True},
                                         self.stage.data if data is None else data,
                                         self.stage.values)

    def test_success_stops_at_would_open_under_same_lock(self):
        real_open = os.open
        opened_flags = []

        def checked_open(path, flags, *args, **kwargs):
            opened_flags.append(flags)
            self.assertFalse(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT |
                                      os.O_EXCL | os.O_TRUNC))
            return real_open(path, flags, *args, **kwargs)

        self.stage.values["_write_bundle"] = mock.Mock(side_effect=AssertionError("write"))
        self.stage.values["_reconstruct"] = mock.Mock(side_effect=AssertionError("fetch"))
        with mock.patch.object(diag.os, "open", side_effect=checked_open):
            result = self._run()
        self.assertEqual(result["status"], "PASS_NO_WRITE")
        self.assertEqual(self.stage.calls,
                         ["lock_enter", "probe", "stable", "final", "fresh", "lock_exit"])
        self.assertEqual(len(opened_flags), 1)
        self.assertFalse(self.stage.values["_write_bundle"].called)
        self.assertFalse(self.stage.values["_reconstruct"].called)
        self.assertFalse(result["source_stage_authorized"])

    def test_wrong_bundle_fails_before_parent_open(self):
        with mock.patch.object(diag.os, "open", side_effect=AssertionError("open")):
            result = self._run(b"wrong")
        self.assertEqual(result["status"], "FAIL_NO_WRITE")
        self.assertEqual(result["gate"], "BUNDLE_STDIN_IDENTITY")
        self.assertNotIn("wrong", json.dumps(result))

    def test_full_probe_and_preflight_stability_fail_closed(self):
        self.stage.values["_probe"] = mock.Mock(side_effect=RuntimeError("root byte/inode floor failed"))
        result = self._run()
        self.assertEqual((result["gate"], result["reason_class"]),
                         ("FULL_PROBE", "SPACE"))
        self.stage = FakeStage()
        self.stage.values["_stable"] = mock.Mock(side_effect=RuntimeError("preflight-to-stage identity drift"))
        result = self._run()
        self.assertEqual((result["gate"], result["reason_class"]),
                         ("PREFLIGHT_STABLE", "PREFLIGHT_STABILITY"))

    def test_late_stop_and_one_second_freshness_fail_closed(self):
        self.stage.values["_final_no_write_continuity"] = mock.Mock(
            side_effect=RuntimeError("STOP appeared during final continuity checks"))
        result = self._run()
        self.assertEqual((result["gate"], result["reason_class"]),
                         ("FINAL_NO_WRITE_CONTINUITY", "STOP"))
        self.stage = FakeStage()
        self.stage.values["_freshness_evidence"] = mock.Mock(
            side_effect=RuntimeError("final activity proof too old for first O_EXCL"))
        result = self._run()
        self.assertEqual((result["gate"], result["reason_class"]),
                         ("WOULD_OPEN_FRESHNESS", "FRESHNESS"))

    def test_target_appearance_and_readonly_open_refusal(self):
        with mock.patch.object(diag.os.path, "lexists", side_effect=lambda path: path == self.stage.target):
            result = diag.perform_readonly({"prior": True}, self.stage.data, self.stage.values)
        self.assertEqual(result["gate"], "TARGET_BUNDLE_ABSENT")
        self.stage = FakeStage()
        with mock.patch.object(diag.os, "open", side_effect=PermissionError("permission denied")):
            result = self._run()
        self.assertEqual(result["gate"], "ROOT_READONLY_FD")

    def test_error_receipt_never_copies_exception_message(self):
        self.stage.values["_probe"] = mock.Mock(
            side_effect=RuntimeError("private-token-should-not-be-in-receipt"))
        result = self._run()
        self.assertEqual(result["status"], "FAIL_NO_WRITE")
        self.assertNotIn("private-token", json.dumps(result))
        self.assertEqual(len(result["error_sha256"]), 64)


class LocalOperatorTests(unittest.TestCase):
    def test_disabled_request_is_canonical_and_bound_to_exact_readonly_sources(self):
        raw = (HERE / "REQUEST.DISABLED.json").read_bytes()
        value = json.loads(raw)
        self.assertEqual(raw.strip(), operator._canonical(value))
        self.assertIs(value["dispatch_authorized"], False)
        self.assertIs(value["source_stage_authorized"], False)
        self.assertEqual(value["diagnostic_label"], operator.LABEL)
        self.assertNotEqual(value["diagnostic_label"],
                            "fs-20260925-trn22-c5c-a0-r10-readonly-write-like-diagnostic-r1")
        self.assertNotEqual(value["diagnostic_label"],
                            "fs-20260925-trn22-c5c-a0-r10-incremental-stage-r1")
        self.assertEqual(value["operator_source_sha256"],
                         hashlib.sha256((HERE / "operator_readonly.py").read_bytes()).hexdigest())
        self.assertEqual(value["diagnostic_remote_sha256"],
                         hashlib.sha256((HERE / "remote_readonly.py").read_bytes()).hexdigest())
        self.assertEqual(value["stage_remote_sha256"], operator.STAGE_REMOTE_SHA256)
        self.assertEqual(value["bundle_sha256"], operator.BUNDLE_SHA256)
        enabled = {**value, "dispatch_authorized": True,
                   "schema": "trn22-r10-r5-readonly-write-like-diagnostic-r2-enabled-v1"}
        with mock.patch.object(operator, "_json", side_effect=[(enabled, "request-hash")]):
            # Exhausting the mock at the *review* read proves the request
            # matches the exact enabled schema; it creates no authority.
            with self.assertRaises(StopIteration):
                operator._authority(value["diagnostic_remote_sha256"],
                                    value["operator_source_sha256"])

    def test_disabled_request_refuses_before_ssh_or_marker(self):
        attempted = []

        def no_ssh(*args, **kwargs):
            attempted.append((args, kwargs))
            raise AssertionError("SSH must not be reached")

        with (mock.patch.object(operator, "_package_bytes", return_value=(b"stage", b"bundle", b"diag")),
              mock.patch.object(operator, "ENABLED_REQUEST", Path("/nonexistent-r10-diagnostic-enabled-request")),
              mock.patch.object(operator, "_ssh_identity", side_effect=no_ssh)):
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                operator.run_diagnostic(remote_call=no_ssh)
        self.assertEqual(attempted, [])
        self.assertNotIn("\nMARKER =", (HERE / "operator_readonly.py").read_text())

    def test_optimized_mode_refuses_before_any_authority_or_ssh(self):
        code = ("import importlib.util,pathlib,sys\n"
                "p=pathlib.Path(sys.argv[1]);s=importlib.util.spec_from_file_location('d',p);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
                "m._package_bytes=lambda: (_ for _ in ()).throw(AssertionError('read'))\n"
                "try:m.run_diagnostic(remote_call=lambda *a: (_ for _ in ()).throw(AssertionError('ssh')))\n"
                "except RuntimeError as e:print(type(e).__name__)\n")
        result = subprocess.run([sys.executable, "-O", "-B", "-c", code,
                                 str(HERE / "operator_readonly.py")],
                                capture_output=True, timeout=15, check=False)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), b"RuntimeError")
        self.assertEqual(result.stderr, b"")

    def test_generated_remote_program_has_only_preflight_postflight_or_diag(self):
        program = operator._remote_program(b"pass\n", b"pass\n")
        compile(program, "<readonly-remote>", "exec")
        exact_program = operator._remote_program(
            operator._regular(operator.STAGE_REMOTE),
            operator._regular(operator.DIAG_REMOTE))
        compile(exact_program, "<exact-readonly-remote>", "exec")
        self.assertLess(len(exact_program.encode()), 128 * 1024)
        self.assertIn("perform_readonly", program)
        self.assertIn("stage['perform'](phase)", program)
        self.assertNotIn("stage['perform']('write')", program)
        self.assertNotIn("_write_bundle", (HERE / "remote_readonly.py").read_text().replace(
            "``_write_bundle``", ""))

    def test_write_phase_name_refused_before_ssh(self):
        with mock.patch.object(operator.subprocess, "run", side_effect=AssertionError("SSH")):
            with self.assertRaisesRegex(RuntimeError, "unknown diagnostic phase"):
                operator._ssh_phase("write", None, b"", b"pass\n", b"pass\n")

    def test_separate_pinned_timeout_probe_precedes_phase_ssh(self):
        events = []

        def interlock():
            events.append("interlock")

        def identity():
            events.append("identity")

        def remote(argv, **kwargs):
            self.assertEqual(argv[:4], ["/usr/bin/ssh", "-T", "-F", "/dev/null"])
            if len([event for event in events if event in ("probe", "phase")]) == 0:
                events.append("probe")
                self.assertEqual(argv[-1],
                                 "/usr/bin/env -i LC_ALL=C /usr/bin/timeout --version")
                self.assertEqual(kwargs["timeout"], 25)
                self.assertEqual(kwargs["input"], b"")
                return subprocess.CompletedProcess(argv, 0,
                    b"timeout (GNU coreutils) 9.1\n", b"")
            events.append("phase")
            self.assertTrue(argv[-1].startswith("/usr/bin/timeout --foreground"))
            return subprocess.CompletedProcess(argv, 255, b"", b"")

        with (mock.patch.object(operator, "_local_interlock", side_effect=interlock),
              mock.patch.object(operator, "_ssh_identity", side_effect=identity),
              mock.patch.object(operator.subprocess, "run", side_effect=remote)):
            detail, _ = operator._ssh_phase("preflight", None, b"", b"pass\n", b"pass\n")
        self.assertEqual(events,
                         ["interlock", "identity", "interlock", "probe",
                          "interlock", "interlock", "phase"])
        self.assertEqual(detail["status"], "SSH_OR_STREAM_FAIL")

    def test_malformed_remote_timeout_identity_never_starts_phase(self):
        bad = (
            subprocess.CompletedProcess([], 0, b"timeout (uutils coreutils) 0.8.0\n", b""),
            subprocess.CompletedProcess([], 0, b"timeout (GNU coreutils) 9.1\n", b"warning"),
            subprocess.CompletedProcess([], 1, b"timeout (GNU coreutils) 9.1\n", b""),
            subprocess.CompletedProcess([], 0, b"x" * 4097, b""),
        )
        for result in bad:
            with self.subTest(returncode=result.returncode,
                              stdout_bytes=len(result.stdout),
                              stderr_bytes=len(result.stderr)):
                with (mock.patch.object(operator, "_local_interlock"),
                      mock.patch.object(operator, "_ssh_identity"),
                      mock.patch.object(operator.subprocess, "run",
                                        return_value=result) as ssh):
                    with self.assertRaisesRegex(RuntimeError, "GNU timeout identity"):
                        operator._ssh_phase("preflight", None, b"", b"pass\n", b"pass\n")
                self.assertEqual(ssh.call_count, 1)  # Probe only; no timed phase.

    def test_stop_before_or_after_probe_prevents_phase_ssh(self):
        for stop_at, expected_ssh in ((2, 0), (3, 1), (4, 1)):
            with self.subTest(stop_at=stop_at):
                checks = [0]

                def interlock():
                    checks[0] += 1
                    if checks[0] == stop_at:
                        raise RuntimeError("local STOP present")

                response = subprocess.CompletedProcess([], 0,
                                                       b"timeout (GNU coreutils) 9.1\n", b"")
                with (mock.patch.object(operator, "_local_interlock", side_effect=interlock),
                      mock.patch.object(operator, "_ssh_identity"),
                      mock.patch.object(operator.subprocess, "run",
                                        return_value=response) as ssh):
                    with self.assertRaisesRegex(RuntimeError, "local STOP"):
                        operator._ssh_phase("write_like", {}, b"bundle", b"pass\n", b"pass\n")
                self.assertEqual(ssh.call_count, expected_ssh)

    def test_timeout_probe_timeout_is_bounded_and_prevents_phase(self):
        timed_out = subprocess.TimeoutExpired("ssh", 25, output=b"partial-secret")
        with (mock.patch.object(operator, "_local_interlock"),
              mock.patch.object(operator, "_ssh_identity"),
              mock.patch.object(operator.subprocess, "run", side_effect=timed_out) as ssh):
            with self.assertRaisesRegex(RuntimeError, "probe timed out"):
                operator._ssh_phase("preflight", None, b"", b"pass\n", b"pass\n")
        self.assertEqual(ssh.call_count, 1)

    def test_ssh_failure_summary_hashes_but_never_emits_raw_stderr(self):
        response = subprocess.CompletedProcess([], 255, b"partial", b"SECRET-RAW-STDERR")
        with (mock.patch.object(operator, "_local_interlock"),
              mock.patch.object(operator, "_ssh_identity"),
              mock.patch.object(operator, "_remote_timeout_identity"),
              mock.patch.object(operator.subprocess, "run", return_value=response)):
            detail, value = operator._ssh_phase("preflight", None, b"", b"pass\n", b"pass\n")
        self.assertIsNone(value)
        self.assertEqual(detail["status"], "SSH_OR_STREAM_FAIL")
        self.assertEqual(detail["failure_modes"], ["NONZERO_RC", "NONEMPTY_STDERR"])
        self.assertEqual(detail["transport"]["stderr_bytes"], len(b"SECRET-RAW-STDERR"))
        self.assertNotIn("SECRET-RAW-STDERR", json.dumps(detail))

    def test_timeout_partial_streams_are_hashed(self):
        timeout = subprocess.TimeoutExpired("ssh", 350,
                                            output=b"partial-secret-output",
                                            stderr=b"partial-secret-error")
        with (mock.patch.object(operator, "_local_interlock"),
              mock.patch.object(operator, "_ssh_identity"),
              mock.patch.object(operator, "_remote_timeout_identity"),
              mock.patch.object(operator.subprocess, "run", side_effect=timeout)):
            detail, _ = operator._ssh_phase("write_like", {}, b"", b"pass\n", b"pass\n")
        self.assertEqual(detail["status"], "SSH_TIMEOUT")
        self.assertEqual(detail["transport"]["stdout_bytes"], 21)
        self.assertNotIn("secret", json.dumps(detail))

    def test_malformed_final_age_cannot_be_recorded_as_pass(self):
        value = {"schema": "trn22-r10-r5-write-like-readonly-diagnostic-v1",
                 "phase": "write_like", "status": "PASS_NO_WRITE",
                 "gate": operator.GATES[-1], "passed_gates": list(operator.GATES),
                 "timings": {key: 0.1 for key in operator.TIMING_FIELDS},
                 "freshness": {"activity_age_seconds": 1.01,
                               "label_age_seconds": 0.1,
                               "space_age_seconds": 0.1,
                               "label_scan_duration_seconds": 0.2,
                               "double_label_scan": True,
                               "post_label_activity_check": True},
                 "snapshot": _snapshot(2)}
        response = subprocess.CompletedProcess([], 0, operator._canonical(value) + b"\n", b"")
        with (mock.patch.object(operator, "_local_interlock"),
              mock.patch.object(operator, "_ssh_identity"),
              mock.patch.object(operator, "_remote_timeout_identity"),
              mock.patch.object(operator.subprocess, "run", return_value=response)):
            detail, parsed = operator._ssh_phase("write_like", {}, b"bundle", b"pass\n", b"pass\n")
        self.assertEqual(detail["status"], "MALFORMED_REMOTE_RECEIPT")
        self.assertIsNone(parsed)

    def test_failed_preflight_still_attempts_distinct_postflight(self):
        calls = []

        def fake_remote(phase, prior, data, stage_source, diag_source):
            calls.append(phase)
            return {"phase": phase, "status": "FAIL_NO_WRITE", "transport": {}}, None

        with (mock.patch.object(operator, "_package_bytes", return_value=(b"stage", b"bundle", b"diag")),
              mock.patch.object(operator, "_authority", return_value=("request", "review")),
              mock.patch.object(operator, "_local_interlock"),
              mock.patch.object(operator, "_ssh_identity"),
              mock.patch.object(operator, "_regular", return_value=b"operator")):
            if sys.flags.optimize:
                with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                    operator.run_diagnostic(remote_call=fake_remote)
                self.assertEqual(calls, [])
            else:
                receipt = operator.run_diagnostic(remote_call=fake_remote)
                self.assertEqual(calls, ["preflight", "postflight"])
                self.assertEqual(receipt["status"], "HOLD")
                self.assertFalse(receipt["source_stage_authorized"])

    def test_clean_three_process_diagnostic_never_authorizes_stage(self):
        calls = []

        def fake_remote(phase, prior, data, stage_source, diag_source):
            calls.append(phase)
            if phase == "preflight":
                return ({"phase": phase, "status": "PASS_NO_WRITE", "transport": {}},
                        {"result": {"snapshot": _snapshot(1)}})
            if phase == "write_like":
                self.assertEqual(prior["process_identity"]["pid"], 1)
                value = {"status": "PASS_NO_WRITE", "snapshot": _snapshot(2),
                         "freshness": {"activity_age_seconds": 0.1,
                                       "label_age_seconds": 0.1,
                                       "space_age_seconds": 0.1,
                                       "label_scan_duration_seconds": 0.2,
                                       "double_label_scan": True,
                                       "post_label_activity_check": True},
                         "source_stage_authorized": False,
                         "native_training_authorized": False,
                         "evaluation_authorized": False}
                return {"phase": phase, "status": "PASS_NO_WRITE", "transport": {}}, value
            return ({"phase": phase, "status": "PASS_NO_WRITE", "transport": {}},
                    {"result": {"snapshot": _snapshot(3)}})

        with (mock.patch.object(operator, "_package_bytes", return_value=(b"stage", b"bundle", b"diag")),
              mock.patch.object(operator, "_authority", return_value=("request", "review")),
              mock.patch.object(operator, "_local_interlock"),
              mock.patch.object(operator, "_ssh_identity"),
              mock.patch.object(operator, "_regular", return_value=b"operator")):
            if sys.flags.optimize:
                with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                    operator.run_diagnostic(remote_call=fake_remote)
                self.assertEqual(calls, [])
            else:
                result = operator.run_diagnostic(remote_call=fake_remote)
                self.assertEqual(calls, ["preflight", "write_like", "postflight"])
                self.assertEqual(result["status"], "READONLY_DIAGNOSTIC_COMPLETE")
                self.assertFalse(result["source_stage_authorized"])


if __name__ == "__main__":
    unittest.main()
