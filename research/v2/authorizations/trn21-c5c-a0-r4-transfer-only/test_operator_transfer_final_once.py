"""Offline final trn21 request tests. No SSH, host contact or PEM read."""

import copy
import io
import json
import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_final_once as final
import stage_r4_remote_final as remote
import stage_r4_transport_final as transport
import test_stage_r4_transfer as fixtures


class FinalTransferRequestTests(unittest.TestCase):
    def _binding(self):
        return final._read_frozen_request_and_code()

    def test_exact_request_pins_actual_disabled_static_pass_and_final_payload(self):
        request, binding = self._binding()
        self.assertEqual(binding["request_sha256"], final.REQUEST_SHA256)
        self.assertTrue(request["dispatch_authorized"])
        self.assertEqual(request["disabled_static_review_sha256_not_final_authority"],
                         final.STATIC_REVIEW_SHA256)
        self.assertEqual(request["disabled_request_sha256"], final.DISABLED_REQUEST_SHA256)
        self.assertEqual(request["final_transport_sha256"], final.FINAL_TRANSPORT_SHA256)
        self.assertEqual(request["final_remote_sha256"], final.FINAL_REMOTE_SHA256)
        self.assertEqual(request["source_commit"], final.SOURCE_COMMIT)
        self.assertEqual(request["target"], remote.TARGET)
        self.assertEqual(request["one_shot_marker"], str(final.MARKER))
        self.assertEqual(request["journal"], str(final.core.JOURNAL))
        self.assertFalse(request["live_path_absences_verified"])
        for key in ("source_checkout_authorized", "cpu_g2_authorized",
                    "native_run_authorized", "eval_authorized",
                    "monitor_operation_authorized", "cleanup_authorized",
                    "submission_authorized",
                    "same_path_overwrite_delete_rename_or_retry_authorized"):
            self.assertIs(request[key], False)
        self.assertFalse(final.FINAL_REVIEW.exists())
        self.assertFalse(final.ASSIGNMENT.exists())

    def test_no_actual_final_pass_or_assignment_means_no_marker_or_ssh(self):
        with (mock.patch.object(final, "_consume_fixed_marker",
                                side_effect=AssertionError("marker reached")),
              mock.patch.object(final, "_run_ssh_final",
                                side_effect=AssertionError("SSH reached")),
              mock.patch.object(final.subprocess, "Popen",
                                side_effect=AssertionError("process spawned"))):
            with self.assertRaisesRegex(RuntimeError, "FINAL REVIEW HOLD"):
                final.run_once()

    def test_request_and_code_drift_stop_before_authority_files(self):
        request = json.loads(final.REQUEST.read_text(encoding="utf-8"))
        request["target"] = "/home/ubuntu/changed.tar.gz"
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "altered-request.json"
            altered.write_text(json.dumps(request), encoding="utf-8")
            with mock.patch.object(final, "REQUEST", altered):
                with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                    final._read_frozen_request_and_code()
        with mock.patch.object(final, "CORE_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                final._read_frozen_request_and_code()
        with mock.patch.object(final, "FINAL_TRANSPORT_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                final._read_frozen_request_and_code()

    def test_actual_review_and_assignment_bytes_must_bind_exact_scope(self):
        _, binding = self._binding()
        review = final._expected_final_review(binding)
        review_bytes = json.dumps(review, sort_keys=True).encode("utf-8")
        assignment = final._expected_assignment(binding, final.core._sha(review_bytes))
        assignment_bytes = json.dumps(assignment, sort_keys=True).encode("utf-8")
        authority = final._validate_authority_files(binding, review_bytes,
                                                    assignment_bytes)
        self.assertEqual(authority["final_review_sha256"], final.core._sha(review_bytes))
        self.assertEqual(authority["supervisor_assignment_sha256"],
                         final.core._sha(assignment_bytes))
        with self.assertRaisesRegex(RuntimeError, "malformed review"):
            final._validate_authority_files(binding, b"PASS", assignment_bytes)
        stale = copy.deepcopy(review)
        stale["final_remote_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "not exact PASS"):
            final._validate_authority_files(binding, json.dumps(stale).encode(),
                                            assignment_bytes)
        unbound = copy.deepcopy(assignment)
        unbound["final_independent_review_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            final._validate_authority_files(binding, review_bytes,
                                            json.dumps(unbound).encode())
        expanded = copy.deepcopy(assignment)
        expanded["native_run_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            final._validate_authority_files(binding, review_bytes,
                                            json.dumps(expanded).encode())

    def test_fixed_marker_is_private_exclusive_and_preserved(self):
        authority = {"request_sha256": final.REQUEST_SHA256,
                     "final_review_sha256": "a" * 64,
                     "supervisor_assignment_sha256": "b" * 64}
        with tempfile.TemporaryDirectory(dir=final.MARKER.parent) as directory:
            marker = Path(directory) / "one-attempt.marker"
            with mock.patch.object(final, "MARKER", marker):
                digest = final._consume_fixed_marker(authority)
                before = marker.read_bytes()
                self.assertEqual(final.core._sha(before), digest)
                self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
                final._pinned_marker(digest)
                with self.assertRaises(FileExistsError):
                    final._consume_fixed_marker(authority)
                self.assertEqual(marker.read_bytes(), before)

    def test_final_argv_is_literal_pinned_trn21_payload(self):
        pre = fixtures.probe()
        with mock.patch.object(transport, "checked_ssh_identity", return_value={}):
            for operation in ("probe", "write", "verify"):
                argv = transport.build_ssh_argv(
                    operation, preflight=pre if operation == "write" else None)
                self.assertEqual(argv[:4], ["ssh", "-F", "/dev/null", "-T"])
                self.assertEqual(argv[-2], transport.ENDPOINT)
                self.assertIn("StrictHostKeyChecking=yes", argv)
                self.assertIn("HostKeyAlgorithms=ssh-ed25519", argv)
                words = ["exec", "python3", "-c", transport.reviewed_remote_source(),
                         operation]
                if operation == "write":
                    words.append(remote._identity_digest(pre["identity_after"]))
                self.assertEqual(shlex.split(argv[-1]), words)
        with self.assertRaisesRegex(RuntimeError, "validated fresh preflight"):
            transport.build_ssh_argv("write")

    def test_final_receipt_requires_second_clean_neuron_stop_process_gate(self):
        pre = fixtures.probe()
        archive, write, verify = fixtures.transfer_receipts(pre)
        first = copy.deepcopy(write["immediate_gate"])
        first["utc"] = fixtures.utc(-1.8)
        second = copy.deepcopy(first)
        second["utc"] = fixtures.utc(-1.5)
        write["immediate_gate"] = first
        write["final_creation_gate"] = second
        self.assertEqual(transport.compare_preflight_to_write(pre, write)["status"],
                         "PREFLIGHT_TO_FINAL_CREATION_GATE_PASS")
        self.assertEqual(transport.compare_sessions(pre, write, verify)["status"],
                         "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        missing = copy.deepcopy(write)
        missing.pop("final_creation_gate")
        with self.assertRaisesRegex(RuntimeError, "final pre-O_EXCL"):
            transport.compare_preflight_to_write(pre, missing)
        for mutation in ("neuron_ls", "stops", "processes", "shared_lock_fuser"):
            bad = copy.deepcopy(write)
            gate = bad["final_creation_gate"]
            if mutation == "neuron_ls":
                gate[mutation]["non_diagnostic_pids"] = [987]
            elif mutation == "stops":
                gate[mutation][next(iter(gate[mutation]))] = {"exists": True}
            elif mutation == "processes":
                gate[mutation]["matched"] = [{"pid": 987, "markers": ["train.py"]}]
            else:
                gate[mutation]["pids"] = [987]
            with self.assertRaisesRegex(RuntimeError, "immediate writer"):
                transport.compare_preflight_to_write(pre, bad)
        reversed_gate = copy.deepcopy(write)
        reversed_gate["final_creation_gate"]["utc"] = pre["utc"]
        with self.assertRaisesRegex(RuntimeError, "did not follow"):
            transport.compare_preflight_to_write(pre, reversed_gate)

    def test_file_backed_git_inventory_fails_closed_with_clear_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "old-checkout"
            checkout.mkdir()
            (checkout / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
            path = str(checkout / ".git/HEAD")
            with self.assertRaisesRegex(RuntimeError, "HOLD: non-directory ancestor"):
                remote._safe_path(path)
            with self.assertRaisesRegex(RuntimeError, "HOLD: non-directory ancestor"):
                remote._safe_open_regular(path)

    def test_late_activity_between_gates_cannot_create_target(self):
        pre = fixtures.probe()
        prewrite = fixtures.probe(pid=201, offset=-2)
        pin = remote._identity_digest(pre["identity_after"])
        current = fixtures.identity()
        clear_fuser = {"path_exists": True, "rc": 1,
                       "pids": [], "stderr_empty": True}
        clear_neuron = {"rc": 0, "json_valid": True, "schema_valid": True,
                        "device_count": 1, "non_diagnostic_pids": [],
                        "stderr_empty": True}
        for mutation in ("STOP", "process", "Neuron", "lock", "target"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / remote.TARGET_NAME
                activity = {"late": False}
                real_gate = remote._immediate_activity_gate

                def parent():
                    return os.open(directory, os.O_RDONLY | os.O_DIRECTORY), current["ancestry"]

                def path_meta(path):
                    if activity["late"] and mutation == "STOP" and path == remote.STOP_PATHS[0]:
                        return fixtures.meta("regular", mode="0o600", size=1)
                    if activity["late"] and mutation == "target" and path == remote.CHECKOUT:
                        return fixtures.meta(mode="0o700")
                    return {"exists": False}

                def processes():
                    return {"matched": ([{"pid": 987, "markers": ["train.py"]}]
                                        if activity["late"] and mutation == "process" else []),
                            "unreadable_cmdlines": 0}

                def fuser(path):
                    if (activity["late"] and mutation == "lock" and
                            path.endswith("device0.lock")):
                        return {**clear_fuser, "pids": [987]}
                    return clear_fuser

                def neuron():
                    return {**clear_neuron,
                            "non_diagnostic_pids": ([987] if activity["late"] and
                                                    mutation == "Neuron" else [])}

                def switch_after_first_gate():
                    result = real_gate()
                    activity["late"] = True
                    return result

                with (mock.patch.object(remote, "probe", return_value=prewrite),
                      mock.patch.object(remote, "_open_parent", side_effect=parent),
                      mock.patch.object(remote, "_identity", return_value=current),
                      mock.patch.object(remote, "_safe_path", side_effect=path_meta),
                      mock.patch.object(remote, "_processes", side_effect=processes),
                      mock.patch.object(remote, "_fuser", side_effect=fuser),
                      mock.patch.object(remote, "_neuron_ls", side_effect=neuron),
                      mock.patch.object(remote, "_immediate_activity_gate",
                                        side_effect=switch_after_first_gate)):
                    with self.assertRaises(RuntimeError):
                        remote.write(io.BytesIO(b""), pin)
                self.assertFalse(target.exists(), "late activity must precede O_EXCL")

    def test_neuron_busy_during_writer_identity_is_rechecked_before_create(self):
        pre = fixtures.probe()
        prewrite = fixtures.probe(pid=201, offset=-2)
        pin = remote._identity_digest(pre["identity_after"])
        current = fixtures.identity()
        active = {"neuron": False}
        clear_fuser = {"path_exists": True, "rc": 1,
                       "pids": [], "stderr_empty": True}

        def identity(parent_fd, ancestry):
            active["neuron"] = True
            return current

        def neuron():
            return {"rc": 0, "json_valid": True, "schema_valid": True,
                    "device_count": 1, "non_diagnostic_pids": ([987] if active["neuron"] else []),
                    "stderr_empty": True}

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / remote.TARGET_NAME

            def parent():
                return os.open(directory, os.O_RDONLY | os.O_DIRECTORY), current["ancestry"]

            with (mock.patch.object(remote, "probe", return_value=prewrite),
                  mock.patch.object(remote, "_open_parent", side_effect=parent),
                  mock.patch.object(remote, "_identity", side_effect=identity),
                  mock.patch.object(remote, "_safe_path", return_value={"exists": False}),
                  mock.patch.object(remote, "_processes",
                                    return_value={"matched": [], "unreadable_cmdlines": 0}),
                  mock.patch.object(remote, "_fuser", return_value=clear_fuser),
                  mock.patch.object(remote, "_neuron_ls", side_effect=neuron)):
                with self.assertRaisesRegex(RuntimeError, "Neuron"):
                    remote.write(io.BytesIO(b""), pin)
            self.assertFalse(target.exists(), "Neuron holder appeared before O_EXCL")

    def test_failed_write_still_attempts_distinct_full_postflight(self):
        seen, events = [], []
        pre = fixtures.probe()
        post = fixtures.probe(pid=301, offset=1)
        authority = {"request_sha256": final.REQUEST_SHA256,
                     "final_review_sha256": "a" * 64,
                     "supervisor_assignment_sha256": "b" * 64}

        class FakeJournal:
            def emit(self, event, **details):
                events.append(event)
            def close(self):
                events.append("CLOSED")

        def fake_ssh(operation, journal, *, preflight=None):
            seen.append(operation)
            if len(seen) == 1:
                return pre
            if operation == "write":
                raise RuntimeError("synthetic partial write")
            return post

        with (mock.patch.object(final, "authorize_final", return_value=authority),
              mock.patch.object(transport, "checked_local_source", return_value={}),
              mock.patch.object(transport, "checked_ssh_identity", return_value={}),
              mock.patch.object(final.core, "_check_controller_breakers", return_value={}),
              mock.patch.object(final, "_consume_fixed_marker", return_value="c" * 64),
              mock.patch.object(final, "_pinned_marker", return_value=None),
              mock.patch.object(final.core, "Journal", FakeJournal),
              mock.patch.object(final, "_run_ssh_final", side_effect=fake_ssh)):
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                final.run_once()
        self.assertEqual(seen, ["probe", "write", "probe"])
        self.assertIn("FINAL_READ_ONLY_POSTFLIGHT", events)
        self.assertIn("TRANSFER_HOLD", events)


if __name__ == "__main__":
    unittest.main()
