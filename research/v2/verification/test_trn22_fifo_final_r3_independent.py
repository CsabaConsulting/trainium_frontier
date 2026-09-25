"""Independent synthetic-only trn22 FIFO final-r3 transfer checks.

The remote writer runs only against temporary local files with all host
observations mocked. No SSH, AWS, real marker, journal or archive is touched.
"""

import contextlib
import copy
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
PACKAGE = V2 / "authorizations/trn22-c5c-a0-r4-transfer-lock-final-r3"
OLD_TEST = (V2 / "authorizations/trn22-c5c-a0-r4-transfer-lock-nonblock-hold"
            / "test_lock_final_scan_hold.py")
sys.path.insert(0, str(PACKAGE))
import operator_transfer_lock_final_r3 as final

spec = importlib.util.spec_from_file_location("trn22_r3_old_fixture", OLD_TEST)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class IndependentFifoFinalR3(unittest.TestCase):
    def test_exact_hash_chain_rejects_new_remote_and_historical_hold_drift(self):
        binding = final._frozen_binding()
        self.assertEqual(binding["request_sha256"], final.REQUEST_SHA256)
        self.assertEqual(binding["remote_sha256"], final.REMOTE_SHA256)
        self.assertEqual(binding["historical_final_r2_hold_sha256_not_authority"],
                         final.HISTORICAL_R2_HOLD_SHA256)
        review = final._expected_final_review(binding)
        review_bytes = packed(review)
        assignment = final._expected_assignment(
            binding, hashlib.sha256(review_bytes).hexdigest())
        self.assertEqual(final._validate_authority_files(
            binding, review_bytes, packed(assignment))["request_sha256"],
            final.REQUEST_SHA256)
        for key, changed_value in (
            ("verdict", "HOLD"), ("reviewer_role", "implementer"),
            ("remote_sha256", "0" * 64),
            ("final_wrapper_sha256", "0" * 64),
            ("historical_final_r2_hold_sha256_not_authority", "0" * 64),
            ("one_shot_marker", "/tmp/wrong"),
        ):
            with self.subTest(review_field=key):
                changed = copy.deepcopy(review)
                changed[key] = changed_value
                with self.assertRaises(RuntimeError):
                    final._validate_authority_files(
                        binding, packed(changed), packed(assignment))
        for key, changed_value in (
            ("issuer_role", "implementer"), ("dispatch_authorized", False),
            ("final_independent_review_sha256", "0" * 64),
            ("native_run_authorized", True), ("target", "/tmp/wrong"),
            ("one_attempt", False),
        ):
            with self.subTest(assignment_field=key):
                changed = copy.deepcopy(assignment)
                changed[key] = changed_value
                with self.assertRaises(RuntimeError):
                    final._validate_authority_files(
                        binding, review_bytes, packed(changed))

    def test_missing_actual_authority_cannot_reach_marker_journal_or_ssh(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r3-auth-offline-") as root:
            missing_review = Path(root) / "missing-review.json"
            missing_assignment = Path(root) / "missing-assignment.json"
            with mock.patch.object(final, "FINAL_REVIEW", missing_review), \
                 mock.patch.object(final, "ASSIGNMENT", missing_assignment), \
                 mock.patch.object(final, "_consume_fixed_marker") as marker, \
                 mock.patch.object(final.core, "Journal") as journal, \
                 mock.patch.object(final.core, "_run_ssh") as ssh:
                with self.assertRaisesRegex(RuntimeError, "fixed-path file is absent"):
                    final.run_once()
                marker.assert_not_called()
                journal.assert_not_called()
                ssh.assert_not_called()
            target = Path(root) / "regular"
            target.write_bytes(b"{}")
            symlink = Path(root) / "symlink"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "unsafe fixed-path"):
                final.core._read_evidence(symlink)

    def _synthetic_remote_write(self, arrival):
        """Inject activity during the last scan; return before O_EXCL state."""
        remote = final.core.transport.remote_constants()
        with tempfile.TemporaryDirectory(prefix="trn22-r3-host-offline-") as root:
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
            identity = {"hostname": remote.HOST, "uid": os.geteuid(),
                        "root_free_bytes": remote.PRE_FREE + 1_000_000,
                        "root_free_inodes": remote.MIN_INODES + 100,
                        "boot_id": "synthetic", "mount_namespace": "mnt:[synthetic]",
                        "mount": {}, "block_aliases": {}, "ancestry": {}}
            state = {"arrived": False, "process_calls": 0,
                     "late_stop": False, "lock_held": False}

            def safe_path(path):
                if (path in remote._all_stop_paths() and
                        ((arrival == "stop" and state["arrived"]) or
                         state["late_stop"])):
                    return {"exists": True}
                return {"exists": False}

            def fuser(path):
                if arrival == "moderate_latency" and state["arrived"]:
                    time.sleep(0.35)
                busy = arrival == "checkpoint" and state["arrived"] and path == remote.CHECKPOINT
                return {"path_exists": True, "rc": 0 if busy else 1,
                        "pids": [42] if busy else [], "stderr_empty": True}

            def neuron_ls():
                if arrival == "slow":
                    time.sleep(0.01)
                if arrival == "moderate_latency" and state["arrived"]:
                    time.sleep(0.35)
                return {"rc": 0, "json_valid": True, "schema_valid": True,
                        "device_count": 1, "stderr_empty": True,
                        "non_diagnostic_pids": [42] if
                        arrival == "neuron" and state["arrived"] else []}

            def processes():
                state["process_calls"] += 1
                if arrival == "stop_during_process" and state["arrived"]:
                    state["late_stop"] = True
                busy = arrival == "process" and state["arrived"]
                return {"matched": [{"pid": 42}] if busy else [],
                        "unreadable_cmdlines": 0}

            def final_scan():
                # The unchanged flock is held across scan and target create.
                import fcntl
                contender = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    state["lock_held"] = True
                finally:
                    os.close(contender)
                state["arrived"] = True
                if arrival == "lock":
                    replacement = lock_dir / "replacement"
                    replacement.write_bytes(b"")
                    replacement.chmod(0o664)
                    replacement.replace(lock)
                return {"completed_monotonic_ns": time.monotonic_ns()}

            with contextlib.ExitStack() as stack:
                for patch in (
                    mock.patch.object(remote, "UID", os.geteuid()),
                    mock.patch.object(remote, "HOST_LOCK", str(lock)),
                    mock.patch.object(remote, "TARGET_NAME", target.name),
                    mock.patch.object(remote, "_open_parent", return_value=(parent_fd, {})),
                    mock.patch.object(remote, "_identity", return_value=identity),
                    mock.patch.object(remote, "_stable"),
                    mock.patch.object(remote, "_require_preflight_binding",
                                      return_value="synthetic-bound"),
                    mock.patch.object(remote, "_safe_path", side_effect=safe_path),
                    mock.patch.object(remote, "_processes", side_effect=processes),
                    mock.patch.object(remote, "_live_prewrite_safety", return_value={}),
                    mock.patch.object(remote, "_fuser", side_effect=fuser),
                    mock.patch.object(remote, "_neuron_ls", side_effect=neuron_ls),
                    mock.patch.object(remote, "_final_label_observation",
                                      side_effect=final_scan),
                ):
                    stack.enter_context(patch)
                if arrival == "slow":
                    stack.enter_context(mock.patch.object(
                        remote, "MAX_FINAL_SCAN_TO_OPEN_NS", 1_000_000))
                with self.assertRaises(RuntimeError):
                    remote.write(io.BytesIO(b"payload"), {})
            self.assertTrue(state["arrived"])
            self.assertTrue(state["lock_held"])
            self.assertFalse(target.exists(),
                             "post-scan activity or delay reached target O_EXCL")
            if arrival == "stop_during_process":
                self.assertTrue(state["late_stop"])
            return state

    def test_all_persistent_post_scan_arrivals_block_before_exclusive_create(self):
        for arrival in ("stop", "neuron", "checkpoint", "process", "lock",
                        "stop_during_process", "slow", "moderate_latency"):
            with self.subTest(arrival=arrival):
                self._synthetic_remote_write(arrival)

    def test_post_scan_receipt_required_and_rejects_tampering(self):
        transport = final.core.transport
        pre, write, verify = old.compared_receipts()
        with self.assertRaisesRegex(RuntimeError, "post-final-label"):
            transport.compare_sessions(write, verify, preflight=pre)
        evidence = write["prewrite_evidence"]
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
        self.assertEqual(transport.compare_sessions(
            write, verify, preflight=pre)["status"],
            "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        for name, mutate in (
            ("STOP", lambda s: next(iter(s["final_stops"].values()))
             .update(exists=True)),
            ("process", lambda s: s["stop_process"]["processes"]
             .update(matched=[{"pid": 42}])),
            ("neuron", lambda s: s["neuron_ls"]
             .update(non_diagnostic_pids=[42])),
            ("checkpoint", lambda s: s["checkpoint_fuser"].update(pids=[42])),
            ("lock", lambda s: s["cooperative_lock"]["file"].update(inode=7)),
            ("before-scan", lambda s: s.update(started_monotonic_ns=2500)),
            ("after-open", lambda s: s.update(completed_monotonic_ns=4500)),
            ("wrong-binding", lambda s: s.update(label_completed_monotonic_ns=7)),
        ):
            with self.subTest(mutation=name):
                changed = copy.deepcopy(write)
                mutate(changed["prewrite_evidence"]["post_final_label_safety"])
                with self.assertRaises(RuntimeError):
                    transport.compare_sessions(changed, verify, preflight=pre)

    def test_optimized_python_still_requires_actual_fixed_authority(self):
        code = (
            "from pathlib import Path\nimport sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import operator_transfer_lock_final_r3 as final\n"
            "final.FINAL_REVIEW=Path(sys.argv[2])\n"
            "final.ASSIGNMENT=Path(sys.argv[3])\n"
            "final.run_once()\n"
        )
        with tempfile.TemporaryDirectory(prefix="trn22-r3-opt-offline-") as root:
            result = subprocess.run(
                [sys.executable, "-O", "-B", "-c", code, str(PACKAGE),
                 str(Path(root) / "missing-review.json"),
                 str(Path(root) / "missing-assignment.json")],
                capture_output=True, text=True, timeout=5, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FIFO FINAL REVIEW HOLD", result.stderr)


if __name__ == "__main__":
    unittest.main()
