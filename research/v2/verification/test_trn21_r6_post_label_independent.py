"""Independent local negatives for the disabled trn21 final-r6 transfer writer."""

import contextlib
import copy
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock


HERE = (Path(__file__).resolve().parents[1] / "authorizations" /
        "trn21-c5c-a0-r4-transfer-only")
sys.path.insert(0, str(HERE))

import operator_transfer_final_r6_once as operator
import stage_r4_remote_final_r6 as remote
import stage_r4_transport_final_r6 as transport
import test_operator_transfer_final_r3_once as r3_fixtures
import test_operator_transfer_final_r4_once as r4_fixtures
import test_operator_transfer_final_r6_once as r6_fixtures


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


@contextlib.contextmanager
def local_writer(root, activity):
    """Use a disposable root while executing the actual r6 write path."""
    with (mock.patch.object(r3_fixtures, "remote", remote),
          mock.patch.object(remote, "_lock_mount_proof",
                            return_value=r4_fixtures.mount_proof())):
        with r3_fixtures.simulated_writer(root, activity) as fixture:
            inherited_fuser = remote._fuser
            lock_calls = 0

            def held_lock_fuser(path):
                nonlocal lock_calls
                if path == remote.LOCK_PATH:
                    lock_calls += 1
                    return (r3_fixtures.idle_fuser() if lock_calls <= 2 else
                            r3_fixtures.self_fuser(os.getpid()))
                return inherited_fuser(path)

            with mock.patch.object(remote, "_fuser", side_effect=held_lock_fuser):
                yield fixture


class IndependentR6PostLabelTests(unittest.TestCase):
    def test_disabled_entrypoint_never_reaches_authority_marker_or_ssh(self):
        request, binding = operator._read_frozen_request_and_code()
        self.assertIs(request["dispatch_authorized"], False)
        self.assertIs(request["source_transfer_authorized"], False)
        self.assertEqual(binding["request_sha256"], operator.REQUEST_SHA256)
        with (mock.patch.object(operator, "_read_required",
                                side_effect=AssertionError("authority read")),
              mock.patch.object(operator, "_consume_fixed_marker",
                                side_effect=AssertionError("marker created")),
              mock.patch.object(operator.core, "Journal",
                                side_effect=AssertionError("journal created")),
              mock.patch.object(operator, "_run_ssh_final",
                                side_effect=AssertionError("SSH reached")),
              mock.patch.object(operator.subprocess, "Popen",
                                side_effect=AssertionError("process spawned"))):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()

    def test_persistent_stop_during_each_scan_and_postscan_neuron(self):
        for phase in ("first_scan", "second_scan", "postscan_neuron"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory(
                    dir="/dev/shm") as directory:
                root = Path(directory)
                stop = root / "persistent-STOP"
                activity = {"active": False, "kind": "STOP"}
                with local_writer(root, activity) as (identity, lock, target, _, _):
                    original_scan = remote._label_scan
                    original_neuron = remote._neuron_ls
                    scans = 0

                    def scanned():
                        nonlocal scans
                        scans += 1
                        if ((phase == "first_scan" and scans == 1) or
                                (phase == "second_scan" and scans == 2)):
                            stop.write_text("STOP", encoding="ascii")
                            activity["active"] = True
                        return original_scan()

                    def neuron():
                        if phase == "postscan_neuron" and scans == 2:
                            stop.write_text("STOP", encoding="ascii")
                            activity["active"] = True
                        return original_neuron()

                    with (mock.patch.object(remote, "_label_scan",
                                            side_effect=scanned),
                          mock.patch.object(remote, "_neuron_ls",
                                            side_effect=neuron),
                          mock.patch.object(remote, "BYTES", 0),
                          mock.patch.object(remote, "SHA256", EMPTY_SHA256)):
                        with self.assertRaisesRegex(RuntimeError, "STOP"):
                            remote.write(io.BytesIO(b""), identity, lock)
                    self.assertTrue(stop.is_file())
                    self.assertFalse(target.exists())

    def test_real_monotonic_delay_over_one_second_prevents_target_open(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            with local_writer(root, {}) as (identity, lock, target, _, _):
                original_gate = remote._post_final_label_gate

                def delayed_gate():
                    result = original_gate()
                    time.sleep(1.05)
                    return result

                with (mock.patch.object(remote, "_post_final_label_gate",
                                        side_effect=delayed_gate),
                      mock.patch.object(remote, "BYTES", 0),
                      mock.patch.object(remote, "SHA256", EMPTY_SHA256)):
                    with self.assertRaisesRegex(RuntimeError,
                                                "science-label scan expired"):
                        remote.write(io.BytesIO(b""), identity, lock)
                self.assertFalse(target.exists())

    def test_lock_or_host_identity_drift_prevents_target_open(self):
        for lock_check in ("post_first_scan", "post_second_scan"):
            with self.subTest(lock_check=lock_check), tempfile.TemporaryDirectory(
                    dir="/dev/shm") as directory:
                root = Path(directory)
                with local_writer(root, {}) as (identity, lock, target, _, _):
                    original_identity = remote._host_lock_identity
                    calls = 0

                    def drifted_lock(parent_fd):
                        nonlocal calls
                        calls += 1
                        result = original_identity(parent_fd)
                        if calls == (2 if lock_check == "post_first_scan" else 3):
                            result = copy.deepcopy(result)
                            result["file"]["inode"] += 1
                        return result

                    with (mock.patch.object(remote, "_host_lock_identity",
                                            side_effect=drifted_lock),
                          mock.patch.object(remote, "BYTES", 0),
                          mock.patch.object(remote, "SHA256", EMPTY_SHA256)):
                        with self.assertRaisesRegex(RuntimeError,
                                                    "host lock pathname changed"):
                            remote.write(io.BytesIO(b""), identity, lock)
                    self.assertFalse(target.exists())

        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            with local_writer(root, {}) as (identity, lock, target, _, _):
                original_identity = remote._identity

                def drifted_host(*args, **kwargs):
                    result = copy.deepcopy(original_identity(*args, **kwargs))
                    result["boot_id"] = "different-boot"
                    return result

                with mock.patch.object(remote, "_identity",
                                       side_effect=drifted_host):
                    with self.assertRaisesRegex(RuntimeError,
                                                "host identity changed"):
                        remote.write(io.BytesIO(b""), identity, lock)
                self.assertFalse(target.exists())

    def test_transport_rejects_stale_or_mutated_postscan_evidence(self):
        pre, written, _, _ = r6_fixtures.clean_r6_receipts()
        self.assertEqual(transport.compare_preflight_to_write(pre, written)["status"],
                         "PREFLIGHT_TO_POST_LABEL_ACTIVITY_GATE_PASS")
        exact_limit = copy.deepcopy(written)
        final = exact_limit["final_creation_gate"]
        final["target_open_started_monotonic_ns"] = 2_000_000_000
        final["final_label_to_open_ns"] = 1_000_000_000
        final["target_open_completed_monotonic_ns"] = 2_000_100_000
        self.assertEqual(transport.compare_preflight_to_write(pre, exact_limit)["status"],
                         "PREFLIGHT_TO_POST_LABEL_ACTIVITY_GATE_PASS")
        mutations = {
            "over_limit": lambda final: final.update({
                "target_open_started_monotonic_ns": 2_000_000_001,
                "final_label_to_open_ns": 1_000_000_001,
                "target_open_completed_monotonic_ns": 2_000_100_001}),
            "stop_after": lambda final: final["post_final_label_gate"][
                "stops_after"][remote.STOP_PATHS[0]].update({"exists": True}),
            "lock_after_first": lambda final: final[
                "post_first_label_lock_identity"]["file"].update({"inode": 1}),
            "neuron_after_scan": lambda final: final["post_final_label_gate"][
                "neuron_ls"].update({"non_diagnostic_pids": [12345]}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(exact_limit)
                mutate(changed["final_creation_gate"])
                with self.assertRaises(RuntimeError):
                    transport.compare_preflight_to_write(pre, changed)


if __name__ == "__main__":
    unittest.main()
