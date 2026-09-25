"""Offline verifier checks for the exact continuity final-r2 transfer request.

The remote writer is exercised only against a temporary directory. These tests
never open SSH, read the private key, or contact the host.
"""

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
import operator_transfer_continuity_final_r2 as final
import operator_transfer_final_once as marker_support
import stage_r4_remote as remote
import stage_r4_transport as transport
import test_stage_r4_transfer as fixtures


class IndependentContinuityFinalR2Tests(unittest.TestCase):
    def test_fixed_marker_and_journal_are_private_exclusive_and_fsynced(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            marker, journal_path = parent / "marker", parent / "journal.jsonl"
            original_fsync = os.fsync
            with (mock.patch.object(marker_support, "MARKER", marker),
                  mock.patch.object(marker_support, "_open_marker_parent",
                                    side_effect=lambda: os.open(
                                        parent, os.O_RDONLY | os.O_DIRECTORY |
                                        os.O_NOFOLLOW | os.O_CLOEXEC)),
                  mock.patch.object(final.core, "JOURNAL", journal_path),
                  mock.patch.object(os, "fsync", wraps=original_fsync) as fsync):
                digest = marker_support._consume_fixed_marker({"binding": "synthetic"})
                self.assertEqual(digest, final.core._sha(marker.read_bytes()))
                journal = final.core.Journal()
                journal.emit("SYNTHETIC_LOCAL_ONLY")
                journal.close()
                self.assertGreaterEqual(fsync.call_count, 6)
                with self.assertRaises(FileExistsError):
                    marker_support._consume_fixed_marker({"binding": "synthetic"})
                with self.assertRaises(FileExistsError):
                    final.core.Journal()
            self.assertEqual(marker.stat().st_mode & 0o777, 0o600)
            self.assertEqual(journal_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(journal_path.read_bytes().splitlines()), 1)

    def test_literal_ssh_argv_requires_pinned_local_host_identity(self):
        proof = transport.checked_ssh_identity()
        self.assertEqual(proof["host_fingerprint"], transport.HOST_FINGERPRINT)
        argv = transport.build_ssh_argv("probe")
        self.assertEqual(argv[:4], ["ssh", "-F", "/dev/null", "-T"])
        self.assertIn("UserKnownHostsFile=" + str(transport.KNOWN_HOSTS), argv)
        self.assertIn("HostKeyAlgorithms=ssh-ed25519", argv)
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertEqual(argv[-2], transport.ENDPOINT)
        with mock.patch.object(transport, "KNOWN_HOSTS_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "known-hosts bytes changed"):
                transport.build_ssh_argv("probe")

    def test_actual_file_authority_missing_or_mutated_stops_before_marker_and_ssh(self):
        _request, binding = final._read_frozen_request_and_code()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review_path = root / "review.json"
            assignment_path = root / "assignment.json"
            with (mock.patch.object(final, "FINAL_REVIEW", review_path),
                  mock.patch.object(final, "ASSIGNMENT", assignment_path),
                  mock.patch.object(final.marker_support, "_consume_fixed_marker",
                                    side_effect=AssertionError("marker reached")),
                  mock.patch.object(final.core, "_run_ssh",
                                    side_effect=AssertionError("SSH reached"))):
                with self.assertRaisesRegex(RuntimeError, "FINAL REVIEW HOLD"):
                    final.run_once()
                review_path.write_bytes(final.core._json_bytes(
                    final._expected_final_review(binding)))
                with self.assertRaisesRegex(RuntimeError, "SUPERVISOR ASSIGNMENT HOLD"):
                    final.run_once()
                review_bytes = review_path.read_bytes()
                assignment = final._expected_assignment(
                    binding, final.core._sha(review_bytes))
                assignment["final_independent_review_sha256"] = "0" * 64
                assignment_path.write_bytes(final.core._json_bytes(assignment))
                with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
                    final.run_once()

    def test_preflight_identity_and_old_path_mutations_are_rejected(self):
        pre = fixtures.probe()
        binding = transport.write_binding(pre)
        changed_identity = copy.deepcopy(pre["identity_after"])
        changed_identity["mount_namespace"] = "mnt:[other]"
        with mock.patch.object(remote, "_process_identity",
                               return_value={"pid": 456, "start_ticks": 789}):
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                remote._require_preflight_binding(binding, changed_identity)
            for key, wrong in (("source_commit", "0" * 40),
                               ("capsule_sha256", "0" * 64)):
                changed = copy.deepcopy(binding)
                changed[key] = wrong
                with self.subTest(key=key):
                    with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                        remote._require_preflight_binding(changed,
                                                          pre["identity_after"])
        incomplete = fixtures.probe()
        incomplete["old_inventory"]["paths"].pop(remote.CHECKPOINT)
        with self.assertRaisesRegex(RuntimeError, "inventory incomplete"):
            transport.validate_probe(incomplete, before_write=True)
        protected_drift = fixtures.probe()
        protected_drift["protected"]["six_caches"]["compiler"]["inode"] += 1
        with self.assertRaisesRegex(RuntimeError, "warm-cache child changed"):
            transport.validate_probe(protected_drift, before_write=True)

    def test_old_worktree_git_file_is_classified_without_aborting_full_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "old-checkout"
            checkout.mkdir()
            (checkout / ".git").write_text("gitdir: /unused/local-fixture\n",
                                            encoding="ascii")
            nested_head = checkout / ".git" / "HEAD"
            try:
                result = remote._safe_path(str(nested_head))
            except NotADirectoryError as error:
                self.fail("file-backed old .git/HEAD aborted inventory: " + repr(error))
            self.assertIs(result.get("exists"), False)

    def test_neuron_becoming_busy_after_live_safety_cannot_create_target(self):
        """A newly busy Neuron must gate the last write boundary, as STOP does."""
        pre = fixtures.probe()
        identity = pre["identity_after"]
        payload = b"independent-verifier-neuron-boundary"
        digest = hashlib.sha256(payload).hexdigest()
        busy = {"value": False}
        identity_calls = {"count": 0}
        neuron_calls = {"count": 0}

        def identity_at_boundary(_parent_fd, _ancestry):
            identity_calls["count"] += 1
            if identity_calls["count"] == 2:
                # Another process acquires Neuron after the earlier device
                # probe, while the writer checks identity before O_EXCL.
                busy["value"] = True
            return identity

        def fuser(path):
            if path == "/dev/neuron0" and busy["value"]:
                return {"path_exists": True, "rc": 0, "pids": [777],
                        "stderr_empty": True}
            return copy.deepcopy(pre["neuron0_fuser"] if path == "/dev/neuron0"
                                 else pre["checkpoint_fuser"] if path == remote.CHECKPOINT
                                 else pre["shared_lock_fuser"])

        def neuron_ls():
            neuron_calls["count"] += 1
            result = copy.deepcopy(pre["neuron_ls"])
            if busy["value"]:
                result["non_diagnostic_pids"] = [777]
            return result

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / remote.TARGET_NAME
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with (mock.patch.object(remote, "BYTES", len(payload)),
                      mock.patch.object(remote, "SHA256", digest),
                      mock.patch.object(transport, "SOURCE_SHA256", digest),
                      mock.patch.object(remote, "_open_parent",
                                        side_effect=lambda: (os.dup(parent_fd),
                                                             identity["ancestry"])),
                      mock.patch.object(remote, "_identity",
                                        side_effect=identity_at_boundary),
                      mock.patch.object(remote, "_process_identity",
                                        return_value={"pid": 456, "start_ticks": 789}),
                      mock.patch.object(remote, "_safe_path",
                                        side_effect=lambda path: {"exists": target.exists()
                                                                  if path == remote.TARGET
                                                                  else False}),
                      mock.patch.object(remote, "_processes",
                                        return_value={"matched": [],
                                                      "unreadable_cmdlines": 0}),
                      mock.patch.object(remote, "_fuser", side_effect=fuser),
                      mock.patch.object(remote, "_neuron_ls", side_effect=neuron_ls),
                      mock.patch.object(remote, "_label_scan",
                                        return_value=copy.deepcopy(
                                            pre["science_label_scan"]))):
                    binding = transport.write_binding(pre)
                    try:
                        remote.write(io.BytesIO(payload), binding)
                    except RuntimeError:
                        pass
                    self.assertTrue(busy["value"])
                    self.assertGreaterEqual(neuron_calls["count"], 1)
                    self.assertFalse(target.exists(),
                                     "Neuron became busy before O_EXCL, but the exact target was created")
            finally:
                os.close(parent_fd)


if __name__ == "__main__":
    unittest.main()
