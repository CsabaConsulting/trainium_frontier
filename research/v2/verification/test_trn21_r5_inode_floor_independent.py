"""Independent, local-only checks of the disabled trn21 final-r5 inode repair."""

import copy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest import mock


PACKAGE = (Path(__file__).resolve().parents[1] / "authorizations" /
           "trn21-c5c-a0-r4-transfer-only")
sys.path.insert(0, str(PACKAGE))
import operator_transfer_final_r5_once as operator
import stage_r4_transport_final_r4 as prior_transport
import stage_r4_transport_final_r5 as transport
import test_operator_transfer_final_r4_once as fixtures


class IndependentR5InodeFloorTests(unittest.TestCase):
    def test_exact_disabled_subject_stops_before_authority_or_io(self):
        request, binding = operator._read_frozen_request_and_code()
        self.assertEqual(binding["request_sha256"], operator.REQUEST_SHA256)
        self.assertIs(request["dispatch_authorized"], False)
        self.assertIs(request["source_transfer_authorized"], False)
        self.assertEqual(request["minimum_free_inodes"], 10000)
        self.assertIs(request["full_probe_before_and_after_inode_floor_required"], True)
        with (mock.patch.object(operator, "_read_required",
                                side_effect=AssertionError("authority opened")),
              mock.patch.object(operator, "_consume_fixed_marker",
                                side_effect=AssertionError("marker created")),
              mock.patch.object(operator.core, "Journal",
                                side_effect=AssertionError("journal created")),
              mock.patch.object(operator.subprocess, "Popen",
                                side_effect=AssertionError("SSH spawned"))):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()

    def test_prior_r4_counterexample_is_rejected_in_both_full_probes(self):
        pre, write, verify, post = fixtures.clean_receipts()
        pre["identity_after"]["root_free_inodes"] = 9999
        post["identity_after"]["root_free_inodes"] = 9999
        self.assertIsNotNone(prior_transport.validate_probe(copy.deepcopy(pre),
                                                            before_write=True))
        self.assertIsNotNone(prior_transport.validate_probe(copy.deepcopy(post),
                                                            before_write=False))
        for receipt, before_write in ((pre, True), (post, False)):
            with self.subTest(before_write=before_write):
                with self.assertRaisesRegex(RuntimeError,
                                            "identity_after free-inode floor failed"):
                    transport.validate_probe(receipt, before_write=before_write)
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.compare_sessions(pre, write, verify)
        clean_pre = fixtures.clean_receipts()[0]
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.compare_postflight(clean_pre, post)

    def test_full_probe_matrix_has_exact_boundary_and_plain_integer_rule(self):
        for before_write in (True, False):
            for phase in ("identity_before", "identity_after"):
                for low in (0, 9999):
                    with self.subTest(before_write=before_write, phase=phase, low=low):
                        receipt = fixtures.clean_receipts()[0 if before_write else 3]
                        receipt[phase]["root_free_inodes"] = low
                        with self.assertRaisesRegex(RuntimeError,
                                                    phase + " free-inode floor failed"):
                            transport.validate_probe(receipt, before_write=before_write)
            receipt = fixtures.clean_receipts()[0 if before_write else 3]
            receipt["identity_before"]["root_free_inodes"] = 10000
            receipt["identity_after"]["root_free_inodes"] = 10000
            self.assertIs(transport.validate_probe(receipt,
                                                   before_write=before_write), receipt)
            for invalid in (None, True, 10000.0, "10000"):
                with self.subTest(before_write=before_write, invalid=invalid):
                    bad = copy.deepcopy(receipt)
                    bad["identity_after"]["root_free_inodes"] = invalid
                    with self.assertRaisesRegex(RuntimeError,
                                                "identity_after free-inode floor failed"):
                        transport.validate_probe(bad, before_write=before_write)

    def test_mutation_during_inherited_validation_and_between_gates_holds(self):
        inherited = transport.base.validate_probe
        for before_write in (True, False):
            with self.subTest(before_write=before_write):
                receipt = fixtures.clean_receipts()[0 if before_write else 3]

                def mutate_after_inherited(item, *, before_write):
                    inherited(item, before_write=before_write)
                    item["identity_after"]["root_free_inodes"] = 9999
                    return item

                with mock.patch.object(transport.base, "validate_probe",
                                       side_effect=mutate_after_inherited):
                    with self.assertRaisesRegex(RuntimeError,
                                                "identity_after free-inode floor failed"):
                        transport.validate_probe(receipt, before_write=before_write)
        pre, write, _, post = fixtures.clean_receipts()
        transport.validate_probe(pre, before_write=True)
        pre["identity_after"]["root_free_inodes"] = 9999
        with mock.patch.object(transport, "checked_ssh_identity",
                               side_effect=AssertionError("SSH identity reached")):
            with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
                transport.build_ssh_argv("write", preflight=pre)
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.compare_preflight_to_write(pre, write)
        clean_pre = fixtures.clean_receipts()[0]
        transport.validate_probe(post, before_write=False)
        post["identity_after"]["root_free_inodes"] = 9999
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.compare_postflight(clean_pre, post)

    def test_nested_writer_probe_and_unchanged_remote_binding(self):
        pre, write, verify, post = fixtures.clean_receipts()
        self.assertEqual(hashlib.sha256(operator.FINAL_REMOTE.read_bytes()).hexdigest(),
                         operator.PRIOR_R4_REMOTE_SHA256)
        for phase in ("identity_before", "identity_after"):
            with self.subTest(phase=phase):
                bad_write = copy.deepcopy(write)
                bad_write["prewrite_probe"][phase]["root_free_inodes"] = 9999
                with self.assertRaises(RuntimeError) as first:
                    transport.compare_preflight_to_write(pre, bad_write)
                self.assertIn("inode", str(first.exception))
                with self.assertRaises(RuntimeError) as second:
                    transport.compare_sessions(pre, bad_write, verify)
                self.assertIn("inode", str(second.exception))
        persistence = transport.compare_sessions(pre, write, verify)
        self.assertEqual(persistence["status"], "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        self.assertEqual(transport.compare_postflight(pre, post,
                         persistence=persistence)["status"], "READ_ONLY_POSTFLIGHT_PASS")


if __name__ == "__main__":
    unittest.main()
