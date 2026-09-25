"""Local-only negatives for the exact trn22 FIFO-lock final request candidate."""

import base64
import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_lock_final_r3 as final

DISABLED_TEST = (HERE.parent / "trn22-c5c-a0-r4-transfer-lock-nonblock-hold"
                 / "test_lock_final_scan_hold.py")
_fixture_spec = importlib.util.spec_from_file_location(
    "trn22_r3_remote_receipt_fixtures", DISABLED_TEST)
fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(fixtures)


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class FinalFifoRequestTest(unittest.TestCase):
    def test_subject_and_request_bind_actual_bytes(self):
        binding = final._frozen_binding()
        self.assertEqual(binding["request_sha256"], final.REQUEST_SHA256)
        self.assertEqual(binding["remote_sha256"], final.REMOTE_SHA256)
        self.assertEqual(binding["disabled_subject_file_sha256"], final.SUBJECT_SHA256)
        with mock.patch.object(final, "CORE_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "drift"):
                final._frozen_binding()
        with mock.patch.object(final, "STATIC_REVIEW_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "drift"):
                final._frozen_binding()

    def test_absent_future_files_hold_before_marker_journal_or_ssh(self):
        self.assertFalse(final.FINAL_REVIEW.exists())
        self.assertFalse(final.ASSIGNMENT.exists())
        self.assertFalse(final.MARKER.exists())
        self.assertFalse(final.JOURNAL.exists())
        with mock.patch.object(final, "_consume_fixed_marker") as marker, \
             mock.patch.object(final.core, "_run_ssh") as ssh, \
             mock.patch.object(final.core, "Journal") as journal:
            with self.assertRaisesRegex(RuntimeError, "fixed-path file is absent"):
                final.run_once()
            marker.assert_not_called()
            journal.assert_not_called()
            ssh.assert_not_called()

    def test_review_assignment_are_distinct_exact_actual_file_gates(self):
        binding = final._frozen_binding()
        review = final._expected_final_review(binding)
        review_bytes = packed(review)
        assignment = final._expected_assignment(binding, final.core._sha(review_bytes))
        assignment_bytes = packed(assignment)
        with tempfile.TemporaryDirectory(prefix="trn22-final-auth-") as root:
            review_path = Path(root) / "review.json"
            assignment_path = Path(root) / "assignment.json"
            review_path.write_bytes(review_bytes)
            with mock.patch.object(final, "FINAL_REVIEW", review_path), \
                 mock.patch.object(final, "ASSIGNMENT", assignment_path), \
                 mock.patch.object(final.core.transport, "checked_ssh_identity") as identity:
                with self.assertRaisesRegex(RuntimeError, "fixed-path file is absent"):
                    final.authorize_final()
                identity.assert_not_called()
                assignment_path.write_bytes(assignment_bytes)
                actual = final.authorize_final()
                self.assertEqual(actual["request_sha256"], final.REQUEST_SHA256)
                identity.assert_called_once()
                bad_review = copy.deepcopy(review)
                bad_review["reviewer_role"] = "implementer"
                review_path.write_bytes(packed(bad_review))
                with self.assertRaisesRegex(RuntimeError, "not exact PASS"):
                    final.authorize_final()
                review_path.write_bytes(review_bytes)
                bad_assignment = copy.deepcopy(assignment)
                bad_assignment["final_independent_review_sha256"] = "0" * 64
                assignment_path.write_bytes(packed(bad_assignment))
                with self.assertRaisesRegex(RuntimeError, "assignment binding mismatch"):
                    final.authorize_final()

    def test_argv_is_exact_source_and_pinned_ssh_without_contact(self):
        transport = final.core.transport
        with mock.patch.object(transport, "checked_ssh_identity"):
            argv = transport.build_ssh_argv("probe")
        self.assertEqual(argv[:7], ["ssh", "-F", "/dev/null", "-T", "-i",
                                    transport.IDENTITY_FILE, "-o"])
        self.assertEqual(argv[-2], transport.ENDPOINT)
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertIn("HostKeyAlgorithms=ssh-ed25519", argv)
        words = shlex.split(argv[-1])
        self.assertEqual(words[:3], ["exec", "python3", "-c"])
        self.assertEqual(words[-1], "probe")
        self.assertEqual(base64.b64decode(words[-2]), transport.REMOTE.read_bytes())
        compile(words[3], "<local-reviewed-dispatch>", "exec")
        with self.assertRaises(RuntimeError):
            transport.build_ssh_argv("write")
        with self.assertRaises(RuntimeError):
            transport.build_ssh_argv("probe", preflight={})
        with self.assertRaises(ValueError):
            transport.build_ssh_argv("checkout")

    def test_remote_dispatch_shape_with_safe_synthetic_source_only(self):
        # This executes only a tiny local fixture, never the real host writer.
        fixture = ("def probe(): return {'safe':True}\n"
                   "if __name__ == '__main__': raise RuntimeError('wrong entry')\n")
        runner = final.core.transport._REVIEWED_REMOTE_DISPATCH
        command = [sys.executable, "-B", "-c", runner,
                   base64.b64encode(fixture.encode()).decode(), "probe"]
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=2, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"safe": True})
        wrong = subprocess.run(command[:-1] + ["checkout"], capture_output=True,
                               text=True, timeout=2, check=False)
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("wrong exact remote operation shape", wrong.stderr)

    def test_one_shot_marker_is_exclusive_without_actual_marker(self):
        with tempfile.TemporaryDirectory(prefix="trn22-final-marker-") as root:
            parent = Path(root)
            marker = parent / "test-attempt.marker"
            with mock.patch.object(final, "MARKER", marker), \
                 mock.patch.object(final, "_open_marker_parent",
                                   side_effect=lambda: os.open(parent, os.O_RDONLY |
                                                               os.O_DIRECTORY)), \
                 mock.patch.object(final.core, "_check_controller_breakers",
                                   return_value={"stops_clear": True}):
                digest = final._consume_fixed_marker({"test": "reviewed-assigned"})
                self.assertEqual(final.core._sha(marker.read_bytes()), digest)
                self.assertEqual(marker.stat().st_mode & 0o777, 0o600)
                with self.assertRaises(FileExistsError):
                    final._consume_fixed_marker({"test": "reviewed-assigned"})

    def test_write_failure_still_attempts_separate_read_only_postflight(self):
        class MemoryJournal:
            def __init__(self):
                self.events = []

            def emit(self, event, **details):
                self.events.append(event)

            def close(self):
                self.events.append("CLOSED")

        journal = MemoryJournal()
        pre = {"old_inventory": {"paths": {}}}
        post = {"operation": "probe"}
        with mock.patch.object(final, "authorize_final", return_value={"ok": True}), \
             mock.patch.object(final.core.transport, "checked_local_source",
                               return_value={"ok": True}), \
             mock.patch.object(final.core, "_check_controller_breakers",
                               return_value={"tripped": False}), \
             mock.patch.object(final, "_consume_fixed_marker", return_value="1" * 64), \
             mock.patch.object(final.core, "Journal", return_value=journal), \
             mock.patch.object(final.core, "_pinned", return_value=b""), \
             mock.patch.object(final.core.transport, "validate_probe") as validate, \
             mock.patch.object(final.core.transport, "compare_postflight") as compare, \
             mock.patch.object(final.core, "_run_ssh",
                               side_effect=[pre, RuntimeError("write failed"), post]) as ssh:
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                final.run_once()
            self.assertEqual([call.args[0] for call in ssh.call_args_list],
                             ["probe", "write", "probe"])
            self.assertEqual(validate.call_count, 2)
            compare.assert_called_once()
            self.assertIn("FINAL_READ_ONLY_POSTFLIGHT", journal.events)
            self.assertIn("TRANSFER_HOLD", journal.events)

    def test_same_source_fd_is_rehashed_before_any_ssh_spawn(self):
        class MemoryJournal:
            def __init__(self):
                self.events = []

            def emit(self, event, **details):
                self.events.append(event)

        with tempfile.TemporaryDirectory(prefix="trn22-source-fd-") as root:
            source = Path(root) / "source.tar.gz"
            source.write_bytes(b"synthetic-source")
            source.chmod(0o600)
            journal = MemoryJournal()
            transport = final.core.transport
            info = source.stat()

            def source_binding(digest):
                return {"path": str(source), "bytes": len(b"synthetic-source"),
                        "sha256": digest, "device": info.st_dev,
                        "inode": info.st_ino, "mode": "0o600"}

            with mock.patch.object(transport, "SOURCE", source), \
                 mock.patch.object(transport, "SOURCE_BYTES", len(b"synthetic-source")), \
                 mock.patch.object(transport, "SOURCE_SHA256", "0" * 64), \
                 mock.patch.object(transport, "build_ssh_argv", return_value=["ssh"]), \
                 mock.patch.object(final.core.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "bytes/inode drift"):
                    final.core._run_ssh("write", journal, preflight={},
                                        source_binding=source_binding("0" * 64))
                spawn.assert_not_called()
            expected = hashlib.sha256(b"synthetic-source").hexdigest()
            with mock.patch.object(transport, "SOURCE", source), \
                 mock.patch.object(transport, "SOURCE_BYTES", len(b"synthetic-source")), \
                 mock.patch.object(transport, "SOURCE_SHA256", expected), \
                 mock.patch.object(transport, "build_ssh_argv", return_value=["ssh"]), \
                 mock.patch.object(final.core, "_check_controller_breakers",
                                   return_value={"stops_clear": True}), \
                 mock.patch.object(final.core.subprocess, "Popen",
                                   side_effect=OSError("no host")) as spawn:
                with self.assertRaisesRegex(OSError, "no host"):
                    final.core._run_ssh("write", journal, preflight={},
                                        source_binding=source_binding(expected))
                spawn.assert_called_once()
                self.assertIn("SOURCE_FD_REHASH_PASS", journal.events)

    def test_stop_during_slow_reauthorization_blocks_before_marker(self):
        state = {"stop": False, "auth_calls": 0, "breaker_calls": 0}
        authority = {"synthetic": "authorized"}

        def reauthorize():
            state["auth_calls"] += 1
            if state["auth_calls"] == 2:
                state["stop"] = True
            return authority

        def breakers():
            state["breaker_calls"] += 1
            if state["stop"]:
                raise RuntimeError("synthetic local STOP")
            return {"stops_clear": True}

        with mock.patch.object(final, "authorize_final", side_effect=reauthorize), \
             mock.patch.object(final.core.transport, "checked_local_source",
                               return_value={"source": "synthetic"}), \
             mock.patch.object(final.core, "_check_controller_breakers",
                               side_effect=breakers), \
             mock.patch.object(final, "_consume_fixed_marker") as marker, \
             mock.patch.object(final.core, "Journal") as journal, \
             mock.patch.object(final.core, "_run_ssh") as ssh:
            with self.assertRaisesRegex(RuntimeError, "synthetic local STOP"):
                final.run_once()
            self.assertEqual(state["breaker_calls"], 2)
            marker.assert_not_called()
            journal.assert_not_called()
            ssh.assert_not_called()

    def test_stop_during_marker_ancestry_blocks_o_excl(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r3-marker-stop-") as root:
            parent = Path(root)
            marker = parent / "synthetic.marker"
            state = {"stop": False}

            def slow_parent():
                state["stop"] = True
                return os.open(parent, os.O_RDONLY | os.O_DIRECTORY)

            def breakers():
                if state["stop"]:
                    raise RuntimeError("synthetic local STOP")
                return {"stops_clear": True}

            with mock.patch.object(final, "MARKER", marker), \
                 mock.patch.object(final, "_open_marker_parent",
                                   side_effect=slow_parent), \
                 mock.patch.object(final.core, "_check_controller_breakers",
                                   side_effect=breakers):
                with self.assertRaisesRegex(RuntimeError, "synthetic local STOP"):
                    final._consume_fixed_marker({"synthetic": "assigned"})
            self.assertTrue(state["stop"])
            self.assertFalse(marker.exists())

    def test_stop_after_source_fd_rehash_blocks_write_spawn_and_keeps_postflight(self):
        class MemoryJournal:
            def __init__(self, state):
                self.state = state
                self.events = []

            def emit(self, event, **details):
                self.events.append(event)
                if event == "SOURCE_FD_REHASH_PASS":
                    self.state["stop"] = True

            def close(self):
                self.events.append("CLOSED")

        with tempfile.TemporaryDirectory(prefix="trn22-r3-write-stop-") as root:
            source = Path(root) / "synthetic.tar.gz"
            source.write_bytes(b"synthetic source")
            source.chmod(0o600)
            info = source.stat()
            digest = hashlib.sha256(b"synthetic source").hexdigest()
            binding = {"path": str(source), "bytes": len(b"synthetic source"),
                       "sha256": digest, "device": info.st_dev,
                       "inode": info.st_ino, "mode": "0o600"}
            state = {"stop": False, "breaker_calls": 0, "probe_calls": 0}
            journal = MemoryJournal(state)
            real_run_ssh = final.core._run_ssh

            def breakers():
                state["breaker_calls"] += 1
                if state["stop"]:
                    raise RuntimeError("synthetic local STOP")
                return {"stops_clear": True}

            def run_ssh(operation, local_journal, *, preflight=None,
                        source_binding=None):
                if operation == "probe":
                    state["probe_calls"] += 1
                    if state["probe_calls"] == 1:
                        return {"old_inventory": {"paths": {}}}
                    return {"operation": "probe"}
                return real_run_ssh(operation, local_journal,
                                    preflight=preflight,
                                    source_binding=source_binding)

            with mock.patch.object(final, "authorize_final",
                                   return_value={"synthetic": "authorized"}), \
                 mock.patch.object(final.core.transport, "checked_local_source",
                                   return_value=binding), \
                 mock.patch.object(final.core, "_check_controller_breakers",
                                   side_effect=breakers), \
                 mock.patch.object(final, "_consume_fixed_marker",
                                   return_value="1" * 64), \
                 mock.patch.object(final.core, "Journal", return_value=journal), \
                 mock.patch.object(final.core, "_pinned", return_value=b""), \
                 mock.patch.object(final.core.transport, "validate_probe"), \
                 mock.patch.object(final.core.transport, "compare_postflight"), \
                 mock.patch.object(final.core, "_run_ssh", side_effect=run_ssh), \
                 mock.patch.object(final.core.transport, "SOURCE", source), \
                 mock.patch.object(final.core.transport, "SOURCE_BYTES",
                                   len(b"synthetic source")), \
                 mock.patch.object(final.core.transport, "SOURCE_SHA256", digest), \
                 mock.patch.object(final.core.transport, "build_ssh_argv",
                                   return_value=["ssh", "synthetic-only"]), \
                 mock.patch.object(final.core.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                    final.run_once()
                spawn.assert_not_called()
            self.assertTrue(state["stop"])
            self.assertEqual(state["breaker_calls"], 4)
            self.assertEqual(state["probe_calls"], 2)
            self.assertIn("FINAL_READ_ONLY_POSTFLIGHT", journal.events)
            self.assertIn("SOURCE_FD_REHASH_PASS", journal.events)

    def _synthetic_remote_write_after_scan(self, *, arrival):
        """Exercise only the copied r3 remote on a temporary local ext4 path."""
        remote = final.core.transport.remote_constants()
        with tempfile.TemporaryDirectory(prefix="trn22-r3-remote-scan-") as root:
            parent = Path(root)
            cache = parent / ".cache"
            lock_dir = cache / "trainium_freshstart"
            cache.mkdir(mode=0o700)
            lock_dir.mkdir(mode=0o775)
            lock_dir.chmod(0o775)
            lock = lock_dir / "device0.lock"
            lock.write_bytes(b"")
            lock.chmod(0o664)
            target = parent / "synthetic-archive.tar.gz"
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            state = {"stop": False, "neuron_busy": False,
                     "slow_post": False, "scan_seen": False,
                     "lock_held_at_scan": False}
            identity = {"hostname": remote.HOST, "uid": os.geteuid(),
                        "root_free_bytes": remote.PRE_FREE + 1_000_000,
                        "root_free_inodes": remote.MIN_INODES + 100,
                        "boot_id": "synthetic", "mount_namespace": "mnt:[synthetic]",
                        "mount": {}, "block_aliases": {}, "ancestry": {}}

            def safe_path(path):
                if path in remote._all_stop_paths() and state["stop"]:
                    return {"exists": True}
                return {"exists": False}

            def idle_fuser(path):
                return {"path_exists": True, "rc": 1, "pids": [],
                        "stderr_empty": True}

            def neuron_ls():
                if state["slow_post"]:
                    time.sleep(0.01)  # Bounded local fixture, never host work.
                return {"rc": 0, "json_valid": True, "schema_valid": True,
                        "device_count": 1, "stderr_empty": True,
                        "non_diagnostic_pids": [42] if state["neuron_busy"] else []}

            def final_label_scan():
                rival_fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(rival_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    state["lock_held_at_scan"] = True
                finally:
                    os.close(rival_fd)
                state["scan_seen"] = True
                state[arrival] = True  # Simulate persistent host activity in slow scan.
                return {"completed_monotonic_ns": time.monotonic_ns()}

            with contextlib.ExitStack() as stack:
                for patch in (
                    mock.patch.object(remote, "UID", os.geteuid()),
                    mock.patch.object(remote, "HOST_LOCK", str(lock)),
                    mock.patch.object(remote, "TARGET_NAME", target.name),
                    mock.patch.object(remote, "_open_parent",
                                      return_value=(parent_fd, {})),
                    mock.patch.object(remote, "_identity", return_value=identity),
                    mock.patch.object(remote, "_stable"),
                    mock.patch.object(remote, "_require_preflight_binding",
                                      return_value="synthetic-bound"),
                    mock.patch.object(remote, "_safe_path", side_effect=safe_path),
                    mock.patch.object(remote, "_processes",
                                      return_value={"matched": [],
                                                    "unreadable_cmdlines": 0}),
                    mock.patch.object(remote, "_live_prewrite_safety",
                                      return_value={}),
                    mock.patch.object(remote, "_fuser", side_effect=idle_fuser),
                    mock.patch.object(remote, "_neuron_ls", side_effect=neuron_ls),
                    mock.patch.object(remote, "_final_label_observation",
                                      side_effect=final_label_scan),
                ):
                    stack.enter_context(patch)
                if arrival == "slow_post":
                    stack.enter_context(mock.patch.object(
                        remote, "MAX_FINAL_SCAN_TO_OPEN_NS", 1_000_000))
                with self.assertRaisesRegex(RuntimeError,
                                            "STOP|post-label checkpoint or Neuron|expired"):
                    remote.write(io.BytesIO(b"synthetic payload"), {})
            self.assertTrue(state["scan_seen"])
            self.assertTrue(state["lock_held_at_scan"])
            self.assertTrue(state[arrival])
            self.assertFalse(target.exists(), "host activity must block O_EXCL")

    def test_remote_stop_during_final_label_scan_blocks_exclusive_create(self):
        self._synthetic_remote_write_after_scan(arrival="stop")

    def test_remote_neuron_during_final_label_scan_blocks_exclusive_create(self):
        self._synthetic_remote_write_after_scan(arrival="neuron_busy")

    def test_slow_post_scan_safety_expires_label_before_exclusive_create(self):
        self._synthetic_remote_write_after_scan(arrival="slow_post")

    def test_post_final_label_receipt_is_required_and_bounded(self):
        transport = final.core.transport
        pre, write, verify = fixtures.compared_receipts()
        evidence = write["prewrite_evidence"]
        with self.assertRaisesRegex(RuntimeError, "post-final-label"):
            transport.compare_sessions(write, verify, preflight=pre)
        prior = evidence["last_live_precreate_safety"]
        stop = evidence["last_stop_process_check"]
        evidence["post_final_label_safety"] = {
            "label_completed_monotonic_ns": 3000,
            "started_monotonic_ns": 3200,
            "completed_monotonic_ns": 3500,
            "checkpoint_fuser": prior["checkpoint_fuser"],
            "neuron0_fuser": prior["neuron0_fuser"],
            "neuron_ls": prior["neuron_ls"],
            "cooperative_lock": prior["cooperative_lock"],
            "stop_process": stop, "final_stops": stop["stops"],
        }
        result = transport.compare_sessions(write, verify, preflight=pre)
        self.assertEqual(result["status"], "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        for name, mutation in (
            ("post-stop", lambda s: next(iter(s["final_stops"].values()))
             .update(exists=True)),
            ("post-process", lambda s: s["stop_process"]["processes"]
             .update(matched=[{"pid": 42}])),
            ("post-neuron", lambda s: s["neuron_ls"]
             .update(non_diagnostic_pids=[42])),
            ("post-lock", lambda s: s["cooperative_lock"]["file"]
             .update(inode=12345678)),
            ("before-scan", lambda s: s.update(started_monotonic_ns=2500)),
            ("after-open", lambda s: s.update(completed_monotonic_ns=4500)),
            ("wrong-label-binding", lambda s: s.update(
             label_completed_monotonic_ns=1234)),
        ):
            with self.subTest(mutation=name):
                changed = copy.deepcopy(write)
                mutation(changed["prewrite_evidence"]["post_final_label_safety"])
                with self.assertRaises(RuntimeError):
                    transport.compare_sessions(changed, verify, preflight=pre)


if __name__ == "__main__":
    unittest.main()
