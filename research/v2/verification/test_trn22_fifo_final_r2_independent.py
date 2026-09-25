"""Independent, host-free adversarial checks of exact trn22 final-r2 transfer.

All source files, markers, journals and SSH spawns used here are synthetic.
The fixed real authority files remain absent; this test never contacts a host.
"""

import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


V2 = Path(__file__).resolve().parents[1]
PACKAGE = V2 / "authorizations/trn22-c5c-a0-r4-transfer-lock-final-r2"
DISABLED_TEST = (V2 / "authorizations/trn22-c5c-a0-r4-transfer-lock-nonblock-hold"
                 / "test_lock_final_scan_hold.py")
sys.path.insert(0, str(PACKAGE))
import operator_transfer_lock_final_r2 as final

spec = importlib.util.spec_from_file_location("trn22_r2_disabled_fixtures", DISABLED_TEST)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class MemoryJournal:
    def __init__(self, on_emit=None):
        self.events = []
        self.on_emit = on_emit

    def emit(self, name, **details):
        self.events.append((name, details))
        if self.on_emit is not None:
            self.on_emit(name)

    def close(self):
        self.events.append(("CLOSED", {}))


class IndependentFifoFinalR2(unittest.TestCase):
    def test_exact_role_hash_chain_and_no_self_attestation(self):
        binding = final._frozen_binding()
        self.assertEqual(binding["request_sha256"], final.REQUEST_SHA256)
        self.assertEqual(binding["final_wrapper_sha256"], hashlib.sha256(
            (PACKAGE / "operator_transfer_lock_final_r2.py").read_bytes()).hexdigest())
        review = final._expected_final_review(binding)
        review_bytes = packed(review)
        assignment = final._expected_assignment(
            binding, hashlib.sha256(review_bytes).hexdigest())
        accepted = final._validate_authority_files(
            binding, review_bytes, packed(assignment))
        self.assertEqual(accepted["final_review_sha256"],
                         hashlib.sha256(review_bytes).hexdigest())

        for key, value in (("verdict", "HOLD"),
                           ("reviewer_role", "implementer"),
                           ("final_wrapper_sha256", "0" * 64),
                           ("historical_final_r1_hold_sha256_not_authority", "0" * 64),
                           ("remote_sha256", "0" * 64),
                           ("one_shot_marker", "/tmp/wrong")):
            with self.subTest(review_field=key):
                changed = copy.deepcopy(review)
                changed[key] = value
                with self.assertRaises(RuntimeError):
                    final._validate_authority_files(
                        binding, packed(changed), packed(assignment))
        for key, value in (("issuer_role", "implementer"),
                           ("dispatch_authorized", False),
                           ("final_independent_review_sha256", "0" * 64),
                           ("source_checkout_authorized", True),
                           ("target", "/tmp/wrong"),
                           ("one_attempt", False)):
            with self.subTest(assignment_field=key):
                changed = copy.deepcopy(assignment)
                changed[key] = value
                with self.assertRaises(RuntimeError):
                    final._validate_authority_files(
                        binding, review_bytes, packed(changed))

    def test_actual_fixed_files_and_nofollow_are_required(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r2-auth-offline-") as root:
            missing = Path(root) / "missing.json"
            with mock.patch.object(final, "FINAL_REVIEW", missing), \
                 mock.patch.object(final, "ASSIGNMENT", missing), \
                 mock.patch.object(final, "_consume_fixed_marker") as marker, \
                 mock.patch.object(final.core, "Journal") as journal, \
                 mock.patch.object(final.core, "_run_ssh") as ssh:
                with self.assertRaisesRegex(RuntimeError, "fixed-path file is absent"):
                    final.run_once()
                marker.assert_not_called()
                journal.assert_not_called()
                ssh.assert_not_called()
            target = Path(root) / "target.json"
            target.write_bytes(b"{}")
            link = Path(root) / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "unsafe fixed-path"):
                final.core._read_evidence(link)

    def test_stop_in_second_authority_blocks_marker(self):
        state = {"stop": False, "auth": 0, "checks": 0}

        def authorize():
            state["auth"] += 1
            if state["auth"] == 2:
                state["stop"] = True
            return {"synthetic": "assigned"}

        def breakers():
            state["checks"] += 1
            if state["stop"]:
                raise RuntimeError("synthetic persistent STOP")
            return {"local_stops_clear": True}

        with mock.patch.object(final, "authorize_final", side_effect=authorize), \
             mock.patch.object(final.core.transport, "checked_local_source",
                               return_value={"synthetic": "source"}), \
             mock.patch.object(final.core, "_check_controller_breakers",
                               side_effect=breakers), \
             mock.patch.object(final, "_consume_fixed_marker") as marker, \
             mock.patch.object(final.core, "Journal") as journal, \
             mock.patch.object(final.core, "_run_ssh") as ssh:
            with self.assertRaisesRegex(RuntimeError, "persistent STOP"):
                final.run_once()
            self.assertEqual(state["checks"], 2)
            marker.assert_not_called()
            journal.assert_not_called()
            ssh.assert_not_called()

    def test_stop_in_marker_ancestry_blocks_exclusive_create(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r2-marker-offline-") as root:
            parent = Path(root)
            marker = parent / "synthetic.marker"
            state = {"stop": False}

            def slow_parent():
                state["stop"] = True
                return os.open(parent, os.O_RDONLY | os.O_DIRECTORY)

            def breakers():
                if state["stop"]:
                    raise RuntimeError("synthetic persistent STOP")
                return {"local_stops_clear": True}

            with mock.patch.object(final, "MARKER", marker), \
                 mock.patch.object(final, "_open_marker_parent", side_effect=slow_parent), \
                 mock.patch.object(final.core, "_check_controller_breakers",
                                   side_effect=breakers):
                with self.assertRaisesRegex(RuntimeError, "persistent STOP"):
                    final._consume_fixed_marker({"synthetic": "assigned"})
            self.assertFalse(marker.exists())

    def test_stop_during_slow_ssh_setup_or_fd_rehash_blocks_popen(self):
        for arrival in ("build", "rehash-journal"):
            with self.subTest(arrival=arrival), \
                 tempfile.TemporaryDirectory(prefix="trn22-r2-write-offline-") as root:
                source = Path(root) / "source.tar.gz"
                source.write_bytes(b"synthetic source")
                source.chmod(0o600)
                info = source.stat()
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                binding = {"path": str(source), "bytes": info.st_size,
                           "sha256": digest, "device": info.st_dev,
                           "inode": info.st_ino, "mode": "0o600"}
                state = {"stop": False, "checks": 0}

                def breakers():
                    state["checks"] += 1
                    if state["stop"]:
                        raise RuntimeError("synthetic persistent STOP")
                    return {"local_stops_clear": True}

                def build(operation, *, preflight=None):
                    self.assertEqual(operation, "write")
                    if arrival == "build":
                        state["stop"] = True
                    return ["ssh", "synthetic-only"]

                def emitted(name):
                    if arrival == "rehash-journal" and name == "SOURCE_FD_REHASH_PASS":
                        state["stop"] = True

                journal = MemoryJournal(emitted)
                with mock.patch.object(final.core.transport, "SOURCE", source), \
                     mock.patch.object(final.core.transport, "SOURCE_BYTES", info.st_size), \
                     mock.patch.object(final.core.transport, "SOURCE_SHA256", digest), \
                     mock.patch.object(final.core.transport, "build_ssh_argv",
                                       side_effect=build), \
                     mock.patch.object(final.core, "_check_controller_breakers",
                                       side_effect=breakers), \
                     mock.patch.object(final.core.subprocess, "Popen") as spawn:
                    with self.assertRaisesRegex(RuntimeError, "persistent STOP"):
                        final.core._run_ssh("write", journal, preflight={},
                                            source_binding=binding)
                    spawn.assert_not_called()
                self.assertEqual(state["checks"], 1)
                self.assertIn("SOURCE_FD_REHASH_PASS", [x[0] for x in journal.events])
                self.assertIn("RAW_SSH_TRANSPORT", [x[0] for x in journal.events])

    def test_write_source_fd_is_bound_to_pre_marker_inode_and_hash(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r2-source-offline-") as root:
            source = Path(root) / "source.tar.gz"
            source.write_bytes(b"first contents")
            source.chmod(0o600)
            info = source.stat()
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            binding = {"path": str(source), "bytes": info.st_size,
                       "sha256": digest, "device": info.st_dev,
                       "inode": info.st_ino, "mode": "0o600"}
            replacement = Path(root) / "replacement.tar.gz"
            replacement.write_bytes(b"first contents")
            replacement.chmod(0o600)
            replacement.replace(source)
            journal = MemoryJournal()
            with mock.patch.object(final.core.transport, "SOURCE", source), \
                 mock.patch.object(final.core.transport, "SOURCE_BYTES", info.st_size), \
                 mock.patch.object(final.core.transport, "SOURCE_SHA256", digest), \
                 mock.patch.object(final.core.transport, "build_ssh_argv",
                                   return_value=["ssh", "synthetic-only"]), \
                 mock.patch.object(final.core.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "differs from hashed source"):
                    final.core._run_ssh("write", journal, preflight={},
                                        source_binding=binding)
                spawn.assert_not_called()

            source.write_bytes(b"wrong contents")  # Same inode and length.
            changed = source.stat()
            changed_binding = {**binding, "device": changed.st_dev,
                               "inode": changed.st_ino}
            with mock.patch.object(final.core.transport, "SOURCE", source), \
                 mock.patch.object(final.core.transport, "SOURCE_BYTES", info.st_size), \
                 mock.patch.object(final.core.transport, "SOURCE_SHA256", digest), \
                 mock.patch.object(final.core.transport, "build_ssh_argv",
                                   return_value=["ssh", "synthetic-only"]), \
                 mock.patch.object(final.core.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "bytes/inode drift"):
                    final.core._run_ssh("write", journal, preflight={},
                                        source_binding=changed_binding)
                spawn.assert_not_called()
            with mock.patch.object(final.core.transport, "build_ssh_argv") as build:
                with self.assertRaisesRegex(RuntimeError, "requires exact"):
                    final.core._run_ssh("write", journal, preflight={})
                build.assert_not_called()

    def test_remote_receipt_mutations_fail_continuity_and_safety(self):
        transport = final.core.transport
        pre, write, verify = fixtures.compared_receipts()
        self.assertEqual(transport.compare_sessions(
            write, verify, preflight=pre)["status"],
            "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        mutations = (
            ("lock-inode", lambda e: e["cooperative_lock_after_create"]["file"].update(
                inode=12345678)),
            ("late-label", lambda e: e["final_label_observation"]["scan"].update(
                match_count=1)),
            ("slow-open", lambda e: e.update(
                target_open_completed_monotonic_ns=2_000_000_000)),
            ("STOP", lambda e: next(iter(e["last_stop_process_check"]["stops"].values()))
                .update(exists=True)),
            ("process", lambda e: e["last_stop_process_check"]["processes"].update(
                matched=[{"pid": 42}])),
            ("Neuron", lambda e: e["last_live_precreate_safety"]["neuron_ls"]
                .update(non_diagnostic_pids=[42])),
        )
        for name, mutate in mutations:
            with self.subTest(write_mutation=name):
                changed = copy.deepcopy(write)
                mutate(changed["prewrite_evidence"])
                with self.assertRaises(RuntimeError):
                    transport.compare_sessions(changed, verify, preflight=pre)
        pre_mutations = (
            ("boot", lambda p: p["identity_before"].update(boot_id="other")),
            ("mount", lambda p: p["identity_before"]["mount"].update(
                major_minor="1:1")),
            ("private-cache", lambda p: p["host_lock_identity"]["cache_directory"]
                .update(mode="0o775")),
            ("private-acl", lambda p: p["host_lock_identity"]["cache_directory"]
                ["acl"].update(access="present")),
            ("disk", lambda p: p["identity_before"].update(root_free_bytes=1)),
            ("old-inventory", lambda p: p["old_inventory"].update(catalog_complete=False)),
        )
        for name, mutate in pre_mutations:
            with self.subTest(preflight_mutation=name):
                changed = copy.deepcopy(pre)
                mutate(changed)
                with self.assertRaises(RuntimeError):
                    transport.compare_sessions(write, verify, preflight=changed)

    def test_remote_stop_during_final_label_scan_reaches_exclusive_create(self):
        # The reviewed remote source leaves the last STOP/process observation
        # before a potentially 45-second label scan. A persistent STOP which
        # appears inside that scan is not sampled again before target O_EXCL.
        remote = fixtures.remote
        state = {"stop": False, "stop_checks": 0}
        with fixtures.LockFixture() as fixture:
            target = fixture.root / "new-archive.tar.gz"

            def stop_check():
                state["stop_checks"] += 1
                if state["stop"]:
                    raise RuntimeError("synthetic persistent remote STOP")
                return {}

            def final_label_scan():
                state["stop"] = True
                return {"completed_monotonic_ns": time.monotonic_ns()}

            supplied_fixture = fixtures.LockFinalScanTests()
            patches = supplied_fixture._writer_fixture(
                fixture, final_label=final_label_scan)
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                stack.enter_context(mock.patch.object(
                    remote, "_require_no_stops_or_processes", side_effect=stop_check))
                # The synthetic payload intentionally mismatches the pinned
                # capsule, so it stops after the O_EXCL proof with no success.
                with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                    remote.write(io.BytesIO(b"payload"), {})
            self.assertTrue(state["stop"])
            self.assertEqual(state["stop_checks"], 1)
            self.assertTrue(target.exists(), "O_EXCL should expose the STOP gap")
            self.assertEqual(target.read_bytes(), b"payload")

    def test_optimized_python_still_holds_without_fixed_authority(self):
        child = (
            "from pathlib import Path\nimport sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import operator_transfer_lock_final_r2 as final\n"
            "final.FINAL_REVIEW=Path(sys.argv[2])\n"
            "final.ASSIGNMENT=Path(sys.argv[3])\n"
            "final.run_once()\n"
        )
        with tempfile.TemporaryDirectory(prefix="trn22-r2-opt-offline-") as root:
            result = subprocess.run(
                [sys.executable, "-O", "-B", "-c", child, str(PACKAGE),
                 str(Path(root) / "missing-review.json"),
                 str(Path(root) / "missing-assignment.json")],
                capture_output=True, text=True, timeout=5, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FIFO FINAL REVIEW HOLD", result.stderr)


if __name__ == "__main__":
    unittest.main()
