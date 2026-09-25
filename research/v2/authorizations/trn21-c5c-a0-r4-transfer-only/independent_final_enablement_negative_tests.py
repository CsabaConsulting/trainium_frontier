"""Independent offline final-transfer regression tests; never contact a host.

The authority fixtures and writer target live under /dev/shm.  The late-label
tests intentionally fail against Implementer commit 2cda5476.
"""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
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


class IndependentFinalEnablementNegativeTests(unittest.TestCase):
    def test_actual_future_authority_files_bind_bytes_before_marker_or_ssh(self):
        _, binding = final._read_frozen_request_and_code()
        review = final._expected_final_review(binding)
        review_bytes = json.dumps(review, sort_keys=True, separators=(",", ":")).encode()
        assignment = final._expected_assignment(binding, final.core._sha(review_bytes))
        assignment_bytes = json.dumps(assignment, sort_keys=True,
                                      separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            review_path = Path(directory) / "final-review.json"
            assignment_path = Path(directory) / "supervisor-assignment.json"
            with (mock.patch.object(final, "FINAL_REVIEW", review_path),
                  mock.patch.object(final, "ASSIGNMENT", assignment_path),
                  mock.patch.object(final.core.transport, "checked_ssh_identity",
                                    return_value={}),
                  mock.patch.object(final, "_consume_fixed_marker",
                                    side_effect=AssertionError("marker reached")),
                  mock.patch.object(final, "_run_ssh_final",
                                    side_effect=AssertionError("SSH reached")),
                  mock.patch.object(final.subprocess, "Popen",
                                    side_effect=AssertionError("process spawned"))):
                with self.assertRaisesRegex(RuntimeError, "FINAL REVIEW HOLD"):
                    final.run_once()
                review_path.write_bytes(review_bytes)
                with self.assertRaisesRegex(RuntimeError, "SUPERVISOR ASSIGNMENT HOLD"):
                    final.run_once()
                assignment_path.write_bytes(assignment_bytes)
                authority = final.authorize_final()
                self.assertEqual(authority["final_review_sha256"],
                                 final.core._sha(review_bytes))
                self.assertEqual(authority["supervisor_assignment_sha256"],
                                 final.core._sha(assignment_bytes))

                tampered_review = dict(review, final_remote_sha256="0" * 64)
                review_path.write_text(json.dumps(tampered_review), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "FINAL REVIEW HOLD"):
                    final.run_once()
                review_path.write_bytes(review_bytes)

                tampered_assignment = dict(assignment,
                                           final_independent_review_sha256="0" * 64)
                assignment_path.write_text(json.dumps(tampered_assignment),
                                           encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "SUPERVISOR ASSIGNMENT HOLD"):
                    final.run_once()
                assignment_path.write_bytes(assignment_bytes)

                good_review = Path(directory) / "good-review.json"
                good_review.write_bytes(review_bytes)
                review_path.unlink()
                review_path.symlink_to(good_review)
                with self.assertRaisesRegex(RuntimeError, "unsafe fixed-path"):
                    final.run_once()

    def test_late_science_label_must_block_before_exclusive_target_create(self):
        pre = fixtures.probe(pid=101, offset=-3)
        prewrite = fixtures.probe(pid=202, offset=-2)
        remote._prewrite_safety(prewrite)
        identity = fixtures.identity()
        pin = remote._identity_digest(pre["identity_after"])
        payload = b"offline-late-label-regression"
        clear_holder = {"path_exists": True, "rc": 1, "pids": [],
                        "stderr_empty": True}
        clear_neuron = {"rc": 0, "json_valid": True, "schema_valid": True,
                        "device_count": 1, "non_diagnostic_pids": [],
                        "stderr_empty": True}
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            label_file = root / ("late-" + remote.LABEL)
            target_file = root / remote.TARGET_NAME
            original_gate = remote._immediate_activity_gate
            original_scan = remote._label_scan
            gate_calls = []

            def parent():
                return (os.open(directory, os.O_RDONLY | os.O_DIRECTORY),
                        identity["ancestry"])

            def gate_with_late_label():
                if not gate_calls:
                    label_file.write_text("created after full writer probe",
                                          encoding="utf-8")
                gate_calls.append("gate")
                return original_gate()

            def scan_local_label():
                with mock.patch.object(remote, "ROOT", directory):
                    return original_scan()

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
                                    side_effect=gate_with_late_label),
                  mock.patch.object(remote, "_label_scan", side_effect=scan_local_label)):
                blocked = None
                try:
                    remote.write(io.BytesIO(payload), pin)
                except RuntimeError as error:
                    blocked = error
                self.assertEqual(scan_local_label()["match_count"], 1)
            self.assertEqual(len(gate_calls), 2)
            self.assertFalse(target_file.exists(),
                             "late science label passed both gates and target was created")
            self.assertIsNotNone(blocked, "late science label did not cause HOLD")
            self.assertIn("label", str(blocked).lower())

    def test_final_receipt_must_prove_late_label_scan(self):
        pre = fixtures.probe()
        _, write, _ = fixtures.transfer_receipts(pre)
        first = copy.deepcopy(write["immediate_gate"])
        first["utc"] = fixtures.utc(-1.8)
        final_gate = copy.deepcopy(first)
        final_gate["utc"] = fixtures.utc(-1.5)
        write["immediate_gate"] = first
        write["final_creation_gate"] = final_gate
        with self.assertRaisesRegex(RuntimeError, "label"):
            transport.compare_preflight_to_write(pre, write)


if __name__ == "__main__":
    unittest.main()
