"""Offline negatives for disabled trn21 final-r5 full-probe inode repair."""

import copy
import sys
from pathlib import Path
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_final_r5_once as operator
import stage_r4_transport_final_r5 as transport
import test_operator_transfer_final_r4_once as r4_fixtures


class FinalR5InodeFloorTests(unittest.TestCase):
    def test_distinct_disabled_request_stops_before_authority_marker_or_ssh(self):
        request, binding = operator._read_frozen_request_and_code()
        self.assertEqual(request["schema"],
                         "trn21-c5c-a0-r4-transfer-final-request-v5-disabled")
        self.assertIs(request["dispatch_authorized"], False)
        self.assertIs(request["source_transfer_authorized"], False)
        self.assertIs(request["full_probe_before_and_after_inode_floor_required"], True)
        self.assertEqual(request["prior_final_r4_independent_hold_sha256"],
                         operator.PRIOR_R4_INDEPENDENT_HOLD_SHA256)
        self.assertEqual(binding["request_sha256"], operator.REQUEST_SHA256)
        with (mock.patch.object(operator, "_read_required",
                                side_effect=AssertionError("authority reached")),
              mock.patch.object(operator, "_consume_fixed_marker",
                                side_effect=AssertionError("marker reached")),
              mock.patch.object(operator, "_run_ssh_final",
                                side_effect=AssertionError("SSH reached")),
              mock.patch.object(operator.subprocess, "Popen",
                                side_effect=AssertionError("process spawned"))):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()

    def test_verifier_9999_reproducer_now_holds_preflight_and_postflight(self):
        pre, write, verify, post = r4_fixtures.clean_receipts()
        pre["identity_after"]["root_free_inodes"] = transport.MIN_INODES - 1
        post["identity_after"]["root_free_inodes"] = transport.MIN_INODES - 1
        for probe, before_write in ((pre, True), (post, False)):
            with self.subTest(before_write=before_write):
                with self.assertRaisesRegex(RuntimeError,
                                            "identity_after free-inode floor failed"):
                    transport.validate_probe(probe, before_write=before_write)
        with mock.patch.object(transport, "checked_ssh_identity",
                               side_effect=AssertionError("SSH argv gate reached")):
            with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
                transport.build_ssh_argv("write", preflight=pre)
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.compare_sessions(pre, write, verify)
        clean_pre = r4_fixtures.clean_receipts()[0]
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.compare_postflight(clean_pre, post)

    def test_both_samples_are_required_for_each_full_probe_and_boundary_passes(self):
        for before_write in (True, False):
            for phase in ("identity_before", "identity_after"):
                with self.subTest(before_write=before_write, phase=phase):
                    probe = r4_fixtures.clean_receipts()[0 if before_write else 3]
                    probe[phase]["root_free_inodes"] = transport.MIN_INODES - 1
                    with self.assertRaisesRegex(RuntimeError,
                                                phase + " free-inode floor failed"):
                        transport.validate_probe(probe, before_write=before_write)
            boundary = r4_fixtures.clean_receipts()[0 if before_write else 3]
            boundary["identity_before"]["root_free_inodes"] = transport.MIN_INODES
            boundary["identity_after"]["root_free_inodes"] = transport.MIN_INODES
            self.assertIs(transport.validate_probe(boundary,
                                                  before_write=before_write), boundary)

    def test_missing_or_noninteger_terminal_count_is_not_inferred_safe(self):
        for value in (None, True, "10000", 10000.0, -1):
            with self.subTest(value=value):
                probe = r4_fixtures.clean_receipts()[0]
                probe["identity_after"]["root_free_inodes"] = value
                with self.assertRaisesRegex(RuntimeError,
                                            "identity_after free-inode floor failed"):
                    transport.validate_probe(probe, before_write=True)
        probe = r4_fixtures.clean_receipts()[3]
        del probe["identity_after"]["root_free_inodes"]
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.validate_probe(probe, before_write=False)

    def test_inode_drop_between_preflight_writer_and_postflight_stages_holds(self):
        pre, write, verify, post = r4_fixtures.clean_receipts()
        self.assertIs(transport.validate_probe(pre, before_write=True), pre)
        late_pre = copy.deepcopy(pre)
        late_pre["identity_after"]["root_free_inodes"] = 9999
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.compare_preflight_to_write(late_pre, write)
        late_write = copy.deepcopy(write)
        late_write["prewrite_probe"]["identity_after"]["root_free_inodes"] = 9999
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.compare_preflight_to_write(pre, late_write)
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.compare_sessions(pre, late_write, verify)
        self.assertEqual(transport.compare_sessions(pre, write, verify)["status"],
                         "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        late_post = copy.deepcopy(post)
        late_post["identity_after"]["root_free_inodes"] = 9999
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.compare_postflight(pre, late_post)

    def test_mutation_during_inherited_validator_is_rechecked(self):
        inherited = transport.base.validate_probe
        for before_write in (True, False):
            with self.subTest(before_write=before_write):
                probe = r4_fixtures.clean_receipts()[0 if before_write else 3]

                def lower_terminal_after_inherited_check(item, *, before_write):
                    result = inherited(item, before_write=before_write)
                    item["identity_after"]["root_free_inodes"] = 9999
                    return result

                with mock.patch.object(transport.base, "validate_probe",
                                       side_effect=lower_terminal_after_inherited_check):
                    with self.assertRaisesRegex(RuntimeError,
                                                "identity_after free-inode floor failed"):
                        transport.validate_probe(probe, before_write=before_write)

    def test_unchanged_lock_mount_label_and_readback_gates_still_hold(self):
        pre, write, verify, post = r4_fixtures.clean_receipts()
        persistence = transport.compare_sessions(pre, write, verify)
        self.assertEqual(transport.compare_postflight(pre, post,
                                                      persistence=persistence)["status"],
                         "READ_ONLY_POSTFLIGHT_PASS")
        bad_lock = copy.deepcopy(pre)
        bad_lock["host_lock_identity"]["trust"]["acl"]["cache"][
            "system.posix_acl_access"] = "present"
        with self.assertRaises(RuntimeError):
            transport.validate_probe(bad_lock, before_write=True)
        bad_mount = copy.deepcopy(post)
        bad_mount["host_lock_identity"]["trust"]["mounts"]["unsafe_aliases"] = ["alias"]
        with self.assertRaises(RuntimeError):
            transport.validate_probe(bad_mount, before_write=False)
        bad_label = copy.deepcopy(write)
        bad_label["final_creation_gate"]["science_label_scan"]["match_count"] = 1
        with self.assertRaises(RuntimeError):
            transport.compare_preflight_to_write(pre, bad_label)
        bad_readback = copy.deepcopy(verify)
        bad_readback["host_lock_identity"]["file"]["inode"] += 1
        with self.assertRaises(RuntimeError):
            transport.compare_sessions(pre, write, bad_readback)


if __name__ == "__main__":
    unittest.main()
