"""Offline adversarial checks for disabled trn21 final-r6 post-label repair."""

import contextlib
import copy
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_final_r6_once as operator
import stage_r4_remote_final_r6 as remote
import stage_r4_transport_final_r6 as transport
import test_operator_transfer_final_r3_once as r3_fixtures
import test_operator_transfer_final_r4_once as r4_fixtures
import test_stage_r4_transfer as base_fixtures


EMPTY = hashlib.sha256(b"").hexdigest()
ACTIVITY_KEYS = ("targets", "stops", "processes", "checkpoint_fusers",
                 "shared_lock_fuser", "neuron0_fuser", "neuron_ls")


def clean_r6_receipts():
    pre, write, verify, post = r4_fixtures.clean_receipts()
    final = write["final_creation_gate"]
    lock_identity = copy.deepcopy(pre["host_lock_identity"])
    first_label = copy.deepcopy(final["science_label_scan"])
    post_first = {key: copy.deepcopy(final[key]) for key in ACTIVITY_KEYS}
    post_first["utc"] = base_fixtures.utc(-1.35)
    post_final = {
        "utc": base_fixtures.utc(-1.1),
        "monotonic_completed_ns": 1_200_000_000,
        "targets_before": copy.deepcopy(final["targets"]),
        "stops_before": copy.deepcopy(final["stops"]),
        "processes_before": copy.deepcopy(final["processes"]),
        "neuron0_fuser": copy.deepcopy(final["neuron0_fuser"]),
        "neuron_ls": copy.deepcopy(final["neuron_ls"]),
        "targets_after": copy.deepcopy(final["targets"]),
        "stops_after": copy.deepcopy(final["stops"]),
        "processes_after": copy.deepcopy(final["processes"]),
    }
    final["pre_postscan_science_label_scan"] = first_label
    final["post_first_label_gate"] = post_first
    final["post_first_label_lock_identity"] = copy.deepcopy(lock_identity)
    final["science_label_scan"] = r3_fixtures.label_scan(-1.3, -1.2)
    final["post_final_label_lock_identity"] = copy.deepcopy(lock_identity)
    final["post_final_label_gate"] = post_final
    final["final_label_completed_monotonic_ns"] = 1_000_000_000
    final["target_open_started_monotonic_ns"] = 1_450_000_000
    final["final_label_to_open_ns"] = 450_000_000
    final["target_open_completed_monotonic_ns"] = 1_450_100_000
    final["target_open_duration_ns"] = 100_000
    write["science_label_scan"] = copy.deepcopy(final["science_label_scan"])
    return pre, write, verify, post


@contextlib.contextmanager
def simulated_r6_writer(root, activity=None):
    """Adapt the historical disposable writer fixture to repeated held fuser."""
    with (mock.patch.object(r3_fixtures, "remote", remote),
          mock.patch.object(remote, "_lock_mount_proof",
                            return_value=r4_fixtures.mount_proof())):
        with r3_fixtures.simulated_writer(root, activity) as context:
            inherited_fuser = remote._fuser
            calls = {"lock": 0}

            def compatible_fuser(path):
                if path == remote.LOCK_PATH:
                    calls["lock"] += 1
                    if calls["lock"] <= 2:
                        return r3_fixtures.idle_fuser()
                    return r3_fixtures.self_fuser(os.getpid())
                return inherited_fuser(path)

            with mock.patch.object(remote, "_fuser", side_effect=compatible_fuser):
                yield context


class FinalR6PostLabelTests(unittest.TestCase):
    def test_disabled_request_pins_held_r5_and_stops_before_authority_or_io(self):
        request, binding = operator._read_frozen_request_and_code()
        self.assertEqual(request["schema"],
                         "trn21-c5c-a0-r4-transfer-final-request-v6-disabled")
        self.assertIs(request["dispatch_authorized"], False)
        self.assertIs(request["source_transfer_authorized"], False)
        self.assertEqual(request["prior_final_r5_enabled_independent_hold_sha256"],
                         operator.PRIOR_R5_ENABLED_HOLD_SHA256)
        self.assertEqual(request["prior_final_r5_enabled_independent_tests_sha256"],
                         operator.PRIOR_R5_ENABLED_INDEPENDENT_TESTS_SHA256)
        self.assertEqual(binding["request_sha256"], operator.REQUEST_SHA256)
        with (mock.patch.object(operator, "_read_required",
                                side_effect=AssertionError("authority reached")),
              mock.patch.object(operator, "_consume_fixed_marker",
                                side_effect=AssertionError("marker reached")),
              mock.patch.object(operator.core, "Journal",
                                side_effect=AssertionError("journal reached")),
              mock.patch.object(operator, "_run_ssh_final",
                                side_effect=AssertionError("SSH reached")),
              mock.patch.object(operator.subprocess, "Popen",
                                side_effect=AssertionError("process spawned"))):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()

    def test_persistent_stop_during_either_label_scan_blocks_o_excl(self):
        for scan_number in (1, 2):
            with self.subTest(scan_number=scan_number), tempfile.TemporaryDirectory(
                    dir="/dev/shm") as temporary:
                root = Path(temporary)
                stop_file = root / "late-STOP"
                activity = {"active": False, "kind": "STOP"}
                with simulated_r6_writer(root, activity) as (pin, lock_pin, target, _, _):
                    scan = remote._label_scan
                    calls = {"scan": 0}

                    def insert_persistent_stop():
                        calls["scan"] += 1
                        if calls["scan"] == scan_number:
                            stop_file.write_text("STOP", encoding="ascii")
                            activity["active"] = True
                        return scan()

                    with (mock.patch.object(remote, "_label_scan",
                                            side_effect=insert_persistent_stop),
                          mock.patch.object(remote, "BYTES", 0),
                          mock.patch.object(remote, "SHA256", EMPTY)):
                        with self.assertRaisesRegex(RuntimeError, "STOP"):
                            remote.write(io.BytesIO(b""), pin, lock_pin)
                    self.assertTrue(stop_file.is_file())
                    self.assertFalse(target.exists())

    def test_process_or_neuron_arriving_in_last_scan_blocks_o_excl(self):
        for kind in ("process", "Neuron"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(
                    dir="/dev/shm") as temporary:
                root = Path(temporary)
                activity = {"active": False, "kind": kind}
                with simulated_r6_writer(root, activity) as (pin, lock_pin, target, _, _):
                    scan = remote._label_scan
                    calls = {"scan": 0}

                    def activate_on_last_scan():
                        calls["scan"] += 1
                        if calls["scan"] == 2:
                            activity["active"] = True
                        return scan()

                    with (mock.patch.object(remote, "_label_scan",
                                            side_effect=activate_on_last_scan),
                          mock.patch.object(remote, "BYTES", 0),
                          mock.patch.object(remote, "SHA256", EMPTY)):
                        with self.assertRaisesRegex(RuntimeError, kind):
                            remote.write(io.BytesIO(b""), pin, lock_pin)
                    self.assertFalse(target.exists())

    def test_stop_arriving_during_postscan_neuron_command_blocks_o_excl(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            activity = {"active": False, "kind": "STOP"}
            with simulated_r6_writer(root, activity) as (pin, lock_pin, target, _, _):
                original_scan = remote._label_scan
                original_neuron = remote._neuron_ls
                calls = {"scan": 0}

                def counted_scan():
                    calls["scan"] += 1
                    return original_scan()

                def late_stop_during_neuron():
                    if calls["scan"] == 2:
                        activity["active"] = True
                    return original_neuron()

                with (mock.patch.object(remote, "_label_scan", side_effect=counted_scan),
                      mock.patch.object(remote, "_neuron_ls",
                                        side_effect=late_stop_during_neuron),
                      mock.patch.object(remote, "BYTES", 0),
                      mock.patch.object(remote, "SHA256", EMPTY)):
                    with self.assertRaisesRegex(RuntimeError, "STOP"):
                        remote.write(io.BytesIO(b""), pin, lock_pin)
                self.assertFalse(target.exists())

    def test_stale_final_label_expires_before_o_excl(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            with simulated_r6_writer(root) as (pin, lock_pin, target, _, _):
                with (mock.patch.object(remote.time, "monotonic_ns",
                                        side_effect=[100, 200, 1_000_000_101]),
                      mock.patch.object(remote, "BYTES", 0),
                      mock.patch.object(remote, "SHA256", EMPTY)):
                    with self.assertRaisesRegex(RuntimeError,
                                                "science-label scan expired"):
                        remote.write(io.BytesIO(b""), pin, lock_pin)
                self.assertFalse(target.exists())

    def test_label_appearing_during_post_first_activity_is_rescanned(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            late_label = root / ("late-" + remote.LABEL)
            with simulated_r6_writer(root) as (pin, lock_pin, target, _, _):
                activity_gate = remote._activity_gate_snapshot
                calls = {"activity": 0}

                def add_label_after_first_postscan_gate(held_lock_pid=None):
                    result = activity_gate(held_lock_pid=held_lock_pid)
                    calls["activity"] += 1
                    if calls["activity"] == 4:
                        late_label.write_text("late label", encoding="ascii")
                    return result

                with (mock.patch.object(remote, "_activity_gate_snapshot",
                                        side_effect=add_label_after_first_postscan_gate),
                      mock.patch.object(remote, "BYTES", 0),
                      mock.patch.object(remote, "SHA256", EMPTY)):
                    with self.assertRaisesRegex(RuntimeError, "science label"):
                        remote.write(io.BytesIO(b""), pin, lock_pin)
                self.assertTrue(late_label.is_file())
                self.assertFalse(target.exists())

    def test_clean_local_zero_byte_fixture_records_both_postscan_gates(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            with simulated_r6_writer(root) as (pin, lock_pin, target, lock, _):
                old_lock = lock.read_bytes()
                with (mock.patch.object(remote, "BYTES", 0),
                      mock.patch.object(remote, "SHA256", EMPTY)):
                    receipt = remote.write(io.BytesIO(b""), pin, lock_pin)
                self.assertEqual(receipt["status"],
                                 "HASH_VERIFIED_IN_THIS_SESSION_ONLY")
                self.assertTrue(target.is_file())
                self.assertEqual(lock.read_bytes(), old_lock)
                final = receipt["final_creation_gate"]
                self.assertEqual(final["science_label_scan"]["match_count"], 0)
                self.assertEqual(final["pre_postscan_science_label_scan"][
                    "match_count"], 0)
                self.assertTrue(all(not item["exists"] for item in
                                    final["post_final_label_gate"]["stops_after"].values()))

    def test_receipt_validator_requires_temporal_postscan_and_lock_binding(self):
        pre, write, verify, post = clean_r6_receipts()
        self.assertIs(transport.validate_probe(pre, before_write=True), pre)
        continuity = transport.compare_preflight_to_write(pre, write)
        self.assertEqual(continuity["status"],
                         "PREFLIGHT_TO_POST_LABEL_ACTIVITY_GATE_PASS")
        persistence = transport.compare_sessions(pre, write, verify)
        self.assertEqual(transport.compare_postflight(pre, post,
                                                      persistence=persistence)["status"],
                         "READ_ONLY_POSTFLIGHT_PASS")
        mutations = {
            "missing_post_final": lambda item: item["final_creation_gate"].pop(
                "post_final_label_gate"),
            "stop_before": lambda item: item["final_creation_gate"][
                "post_final_label_gate"]["stops_before"][remote.STOP_PATHS[0]].update(
                    {"exists": True}),
            "stop_after": lambda item: item["final_creation_gate"][
                "post_final_label_gate"]["stops_after"][remote.STOP_PATHS[0]].update(
                    {"exists": True}),
            "neuron": lambda item: item["final_creation_gate"][
                "post_final_label_gate"]["neuron_ls"].update(
                    {"non_diagnostic_pids": [987]}),
            "stale_time": lambda item: item["final_creation_gate"][
                "post_final_label_gate"].update({
                    "utc": item["final_creation_gate"][
                        "pre_postscan_science_label_scan"]["completed_utc"]}),
            "wrong_lock": lambda item: item["final_creation_gate"][
                "post_final_label_lock_identity"]["file"].update({"inode": 99}),
            "expired_scan": lambda item: item["final_creation_gate"].update({
                "target_open_started_monotonic_ns": 2_000_000_001,
                "final_label_to_open_ns": 1_000_000_001,
                "target_open_completed_monotonic_ns": 2_000_100_001}),
            "slow_open": lambda item: item["final_creation_gate"].update({
                "target_open_completed_monotonic_ns": 2_450_000_001,
                "target_open_duration_ns": 1_000_000_001}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(write)
                mutate(changed)
                with self.assertRaises(RuntimeError):
                    transport.compare_preflight_to_write(pre, changed)

    def test_prior_inode_acl_mount_and_readback_gates_remain_fail_closed(self):
        pre, write, verify, post = clean_r6_receipts()
        bad_inode = copy.deepcopy(pre)
        bad_inode["identity_after"]["root_free_inodes"] = 9999
        with self.assertRaisesRegex(RuntimeError, "identity_after free-inode floor failed"):
            transport.validate_probe(bad_inode, before_write=True)
        bad_acl = copy.deepcopy(pre)
        bad_acl["host_lock_identity"]["trust"]["acl"]["cache"][
            "system.posix_acl_access"] = "unknown"
        with self.assertRaises(RuntimeError):
            transport.validate_probe(bad_acl, before_write=True)
        bad_mount = copy.deepcopy(post)
        bad_mount["host_lock_identity"]["trust"]["mounts"][
            "unsafe_aliases"] = ["alias"]
        with self.assertRaises(RuntimeError):
            transport.validate_probe(bad_mount, before_write=False)
        bad_readback = copy.deepcopy(verify)
        bad_readback["host_lock_identity"]["file"]["inode"] += 1
        with self.assertRaises(RuntimeError):
            transport.compare_sessions(pre, write, bad_readback)

    def test_pinned_r6_argv_uses_only_new_remote_without_ssh(self):
        pre = clean_r6_receipts()[0]
        self.assertEqual(hashlib.sha256(transport.REMOTE.read_bytes()).hexdigest(),
                         transport.REMOTE_SHA256)
        with mock.patch.object(transport, "checked_ssh_identity", return_value={}):
            argv = transport.build_ssh_argv("write", preflight=pre)
        self.assertEqual(argv[-2], transport.ENDPOINT)
        self.assertIn("stage_r4_remote_final_r6.py", str(transport.REMOTE))
        self.assertNotEqual(transport.REMOTE_SHA256,
                            operator.PRIOR_R5_REMOTE_SHA256)


if __name__ == "__main__":
    unittest.main()
