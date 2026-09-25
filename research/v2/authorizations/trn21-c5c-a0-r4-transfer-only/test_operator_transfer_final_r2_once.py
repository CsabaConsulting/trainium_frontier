"""Offline adversarial tests for the separate trn21 final-r2 transfer package."""

import copy
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_final_r2_once as final
import stage_r4_remote_final_r2 as remote
import stage_r4_transport_final_r2 as transport
import test_stage_r4_transfer as fixtures


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def clean_r2_receipts():
    pre = fixtures.probe()
    _, write, verify = fixtures.transfer_receipts(pre)
    first = copy.deepcopy(write["immediate_gate"])
    first["utc"] = fixtures.utc(-1.8)
    pre_label = copy.deepcopy(first)
    pre_label["utc"] = fixtures.utc(-1.6)
    final_gate = copy.deepcopy(first)
    final_gate["utc"] = fixtures.utc(-1.3)
    final_gate["pre_label_gate"] = pre_label
    final_gate["science_label_scan"] = {
        "rc": 0, "matches": [], "match_count": 0, "truncated": False,
        "stderr_empty": True, "stdout_sha256": EMPTY_SHA256,
        "started_utc": fixtures.utc(-1.5),
        "completed_utc": fixtures.utc(-1.4),
    }
    write["immediate_gate"] = first
    write["final_creation_gate"] = final_gate
    return pre, write, verify


class FinalR2TransferTests(unittest.TestCase):
    def test_exact_successor_pins_held_candidate_and_hold_evidence(self):
        request, binding = final._read_frozen_request_and_code()
        self.assertEqual(binding["request_sha256"], final.REQUEST_SHA256)
        self.assertEqual(request["final_transport_sha256"],
                         final.FINAL_TRANSPORT_SHA256)
        self.assertEqual(request["final_remote_sha256"], final.FINAL_REMOTE_SHA256)
        self.assertEqual(request["held_final_review_sha256"],
                         final.HELD_FINAL_REVIEW_SHA256)
        self.assertEqual(request["held_final_negative_tests_sha256"],
                         final.HELD_FINAL_NEGATIVE_TESTS_SHA256)
        self.assertEqual(request["source_commit"], final.SOURCE_COMMIT)
        self.assertEqual(request["target"], remote.TARGET)
        self.assertEqual(request["one_shot_marker"], str(final.MARKER))
        self.assertFalse(request["live_path_absences_verified"])
        self.assertTrue(request["writer_final_science_label_scan_and_post_scan_activity_gate_required"])
        self.assertTrue(request["writer_receipt_final_label_evidence_required"])
        for key in ("source_checkout_authorized", "cpu_g2_authorized",
                    "native_run_authorized", "eval_authorized",
                    "monitor_operation_authorized", "cleanup_authorized",
                    "submission_authorized",
                    "same_path_overwrite_delete_rename_or_retry_authorized"):
            self.assertIs(request[key], False)
        for path, digest in (
                (final.HELD_FINAL_REQUEST, final.HELD_FINAL_REQUEST_SHA256),
                (final.HELD_FINAL_WRAPPER, final.HELD_FINAL_WRAPPER_SHA256),
                (final.HELD_FINAL_TRANSPORT, final.HELD_FINAL_TRANSPORT_SHA256),
                (final.HELD_FINAL_REMOTE, final.HELD_FINAL_REMOTE_SHA256),
                (final.HELD_FINAL_REVIEW, final.HELD_FINAL_REVIEW_SHA256),
                (final.HELD_FINAL_NEGATIVE_TESTS,
                 final.HELD_FINAL_NEGATIVE_TESTS_SHA256)):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_future_authority_absent_means_no_marker_or_ssh(self):
        self.assertFalse(final.FINAL_REVIEW.exists())
        self.assertFalse(final.ASSIGNMENT.exists())
        self.assertFalse(final.MARKER.exists())
        with (mock.patch.object(final, "_consume_fixed_marker",
                                side_effect=AssertionError("marker reached")),
              mock.patch.object(final, "_run_ssh_final",
                                side_effect=AssertionError("SSH reached")),
              mock.patch.object(final.subprocess, "Popen",
                                side_effect=AssertionError("process spawned"))):
            with self.assertRaisesRegex(RuntimeError, "FINAL REVIEW HOLD"):
                final.run_once()
        self.assertFalse(final.MARKER.exists())

    def test_preserved_independent_negative_tests_pass_against_successor(self):
        """Run the immutable historical negatives with successor module aliases."""
        aliases = {
            "operator_transfer_final_once": final,
            "stage_r4_remote_final": remote,
            "stage_r4_transport_final": transport,
        }
        spec = importlib.util.spec_from_file_location(
            "independent_final_enablement_negatives_against_r2",
            final.HELD_FINAL_NEGATIVE_TESTS)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, aliases):
            spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        result = unittest.TestResult()
        suite.run(result)
        details = [str(case) + ": " + traceback for case, traceback
                   in result.failures + result.errors]
        self.assertEqual(result.testsRun, 3)
        self.assertEqual(details, [])

    def test_final_receipt_authenticates_clean_label_and_both_activity_gates(self):
        pre, write, verify = clean_r2_receipts()
        continuity = transport.compare_preflight_to_write(pre, write)
        self.assertEqual(continuity["status"],
                         "PREFLIGHT_TO_FINAL_CREATION_GATE_PASS")
        self.assertEqual(continuity["label_stdout_sha256"], EMPTY_SHA256)
        self.assertEqual(transport.compare_sessions(pre, write, verify)["status"],
                         "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        for mutation in ("missing", "match", "hash", "time", "pre_label_missing"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(write)
                gate = bad["final_creation_gate"]
                if mutation == "missing":
                    gate.pop("science_label_scan")
                elif mutation == "match":
                    gate["science_label_scan"]["matches"] = [remote.ROOT + "/late-" + remote.LABEL]
                elif mutation == "hash":
                    gate["science_label_scan"]["stdout_sha256"] = "0" * 64
                elif mutation == "time":
                    gate["science_label_scan"]["completed_utc"] = fixtures.utc(-1.9)
                else:
                    gate.pop("pre_label_gate")
                with self.assertRaisesRegex(RuntimeError, "label"):
                    transport.compare_preflight_to_write(pre, bad)
        for gate_name in ("pre_label_gate", "final_creation_gate"):
            with self.subTest(gate_name=gate_name):
                bad = copy.deepcopy(write)
                gate = bad["final_creation_gate"]
                activity = gate if gate_name == "final_creation_gate" else gate["pre_label_gate"]
                activity["neuron_ls"]["non_diagnostic_pids"] = [987]
                with self.assertRaisesRegex(RuntimeError, "immediate writer Neuron"):
                    transport.compare_preflight_to_write(pre, bad)

    def test_label_introduced_after_second_gate_blocks_before_exclusive_open(self):
        pre = fixtures.probe()
        prewrite = fixtures.probe(pid=202, offset=-2)
        pin = remote._identity_digest(pre["identity_after"])
        identity = fixtures.identity()
        payload = b"offline-final-r2-late-label"
        clear_holder = {"path_exists": True, "rc": 1, "pids": [],
                        "stderr_empty": True}
        clear_neuron = {"rc": 0, "json_valid": True, "schema_valid": True,
                        "device_count": 1, "non_diagnostic_pids": [],
                        "stderr_empty": True}
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            target = Path(directory) / remote.TARGET_NAME
            label_path = Path(directory) / ("late-" + remote.LABEL)
            real_gate = remote._immediate_activity_gate
            real_scan = remote._label_scan
            calls = []

            def parent():
                return os.open(directory, os.O_RDONLY | os.O_DIRECTORY), identity["ancestry"]

            def second_gate_then_label():
                result = real_gate()
                calls.append("gate")
                if len(calls) == 2:
                    label_path.write_text("after second activity gate", encoding="utf-8")
                return result

            def scan_local():
                with mock.patch.object(remote, "ROOT", directory):
                    return real_scan()

            with (mock.patch.object(remote, "BYTES", len(payload)),
                  mock.patch.object(remote, "SHA256", hashlib.sha256(payload).hexdigest()),
                  mock.patch.object(remote, "probe", return_value=prewrite),
                  mock.patch.object(remote, "_open_parent", side_effect=parent),
                  mock.patch.object(remote, "_identity", return_value=identity),
                  mock.patch.object(remote, "_safe_path", return_value={"exists": False}),
                  mock.patch.object(remote, "_processes",
                                    return_value={"matched": [], "unreadable_cmdlines": 0}),
                  mock.patch.object(remote, "_fuser", return_value=clear_holder),
                  mock.patch.object(remote, "_neuron_ls", return_value=clear_neuron),
                  mock.patch.object(remote, "_immediate_activity_gate",
                                    side_effect=second_gate_then_label),
                  mock.patch.object(remote, "_label_scan", side_effect=scan_local)):
                with self.assertRaisesRegex(RuntimeError, "science label"):
                    remote.write(io.BytesIO(payload), pin)
            self.assertEqual(len(calls), 2)
            self.assertTrue(label_path.exists())
            self.assertFalse(target.exists())

    def test_neuron_busy_during_label_scan_blocks_before_exclusive_open(self):
        pre = fixtures.probe()
        prewrite = fixtures.probe(pid=202, offset=-2)
        pin = remote._identity_digest(pre["identity_after"])
        identity = fixtures.identity()
        busy = {"neuron": False}
        clear_holder = {"path_exists": True, "rc": 1, "pids": [],
                        "stderr_empty": True}

        def scan_then_busy():
            busy["neuron"] = True
            return {"rc": 0, "matches": [], "match_count": 0,
                    "truncated": False, "stderr_empty": True,
                    "stdout_sha256": EMPTY_SHA256}

        def neuron():
            return {"rc": 0, "json_valid": True, "schema_valid": True,
                    "device_count": 1,
                    "non_diagnostic_pids": [987] if busy["neuron"] else [],
                    "stderr_empty": True}

        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            target = Path(directory) / remote.TARGET_NAME

            def parent():
                return os.open(directory, os.O_RDONLY | os.O_DIRECTORY), identity["ancestry"]

            with (mock.patch.object(remote, "probe", return_value=prewrite),
                  mock.patch.object(remote, "_open_parent", side_effect=parent),
                  mock.patch.object(remote, "_identity", return_value=identity),
                  mock.patch.object(remote, "_safe_path", return_value={"exists": False}),
                  mock.patch.object(remote, "_processes",
                                    return_value={"matched": [], "unreadable_cmdlines": 0}),
                  mock.patch.object(remote, "_fuser", return_value=clear_holder),
                  mock.patch.object(remote, "_neuron_ls", side_effect=neuron),
                  mock.patch.object(remote, "_label_scan", side_effect=scan_then_busy)):
                with self.assertRaisesRegex(RuntimeError, "Neuron"):
                    remote.write(io.BytesIO(b""), pin)
            self.assertTrue(busy["neuron"])
            self.assertFalse(target.exists())

    def test_every_activity_class_arising_during_label_scan_blocks(self):
        pre = fixtures.probe()
        prewrite = fixtures.probe(pid=202, offset=-2)
        pin = remote._identity_digest(pre["identity_after"])
        identity = fixtures.identity()
        clear_holder = {"path_exists": True, "rc": 1, "pids": [],
                        "stderr_empty": True}
        clear_neuron = {"rc": 0, "json_valid": True, "schema_valid": True,
                        "device_count": 1, "non_diagnostic_pids": [],
                        "stderr_empty": True}
        errors = {
            "STOP": "STOP", "process": "process", "checkpoint": "checkpoint holder",
            "lock": "shared lock", "Neuron": "Neuron", "target": "target",
        }
        for mutation, expected in errors.items():
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                    dir="/dev/shm") as directory:
                target = Path(directory) / remote.TARGET_NAME
                active = {"after_scan": False}

                def parent():
                    return os.open(directory, os.O_RDONLY | os.O_DIRECTORY), identity["ancestry"]

                def path_meta(path):
                    if active["after_scan"]:
                        if mutation == "STOP" and path == remote.STOP_PATHS[0]:
                            return {"exists": True}
                        if mutation == "target" and path == remote.CHECKOUT:
                            return {"exists": True}
                    return {"exists": False}

                def processes():
                    return {"matched": ([{"pid": 987}] if active["after_scan"] and
                                        mutation == "process" else []),
                            "unreadable_cmdlines": 0}

                def fuser(path):
                    if active["after_scan"] and (
                            (mutation == "checkpoint" and path in remote.CHECKPOINTS) or
                            (mutation == "lock" and path.endswith("device0.lock"))):
                        return {**clear_holder, "pids": [987]}
                    return clear_holder

                def neuron():
                    return {**clear_neuron,
                            "non_diagnostic_pids":
                            [987] if active["after_scan"] and mutation == "Neuron" else []}

                def scan_then_activity():
                    active["after_scan"] = True
                    return {"rc": 0, "matches": [], "match_count": 0,
                            "truncated": False, "stderr_empty": True,
                            "stdout_sha256": EMPTY_SHA256}

                with (mock.patch.object(remote, "probe", return_value=prewrite),
                      mock.patch.object(remote, "_open_parent", side_effect=parent),
                      mock.patch.object(remote, "_identity", return_value=identity),
                      mock.patch.object(remote, "_safe_path", side_effect=path_meta),
                      mock.patch.object(remote, "_processes", side_effect=processes),
                      mock.patch.object(remote, "_fuser", side_effect=fuser),
                      mock.patch.object(remote, "_neuron_ls", side_effect=neuron),
                      mock.patch.object(remote, "_label_scan",
                                        side_effect=scan_then_activity)):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        remote.write(io.BytesIO(b""), pin)
                self.assertTrue(active["after_scan"])
                self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
