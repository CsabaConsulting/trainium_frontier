"""Independent offline adversarial checks of the trn22 FIFO final request.

All SSH spawns and the real marker/journal are replaced with synthetic fixtures.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


V2 = Path(__file__).resolve().parents[1]
PACKAGE = V2 / "authorizations/trn22-c5c-a0-r4-transfer-lock-final-r1"
DISABLED = V2 / "authorizations/trn22-c5c-a0-r4-transfer-lock-nonblock-hold"
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(DISABLED))
import operator_transfer_lock_final_once as final
import test_lock_final_scan_hold as disabled_fixtures


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class MemoryJournal:
    def __init__(self):
        self.events = []

    def emit(self, name, **details):
        self.events.append((name, details))

    def close(self):
        self.events.append(("CLOSED", {}))


class IndependentFifoFinalR1(unittest.TestCase):
    def test_exact_role_hash_chain_rejects_tampering(self):
        binding = final._frozen_binding()
        review = final._expected_final_review(binding)
        review_bytes = packed(review)
        assignment = final._expected_assignment(
            binding, hashlib.sha256(review_bytes).hexdigest())
        accepted = final._validate_authority_files(
            binding, review_bytes, packed(assignment))
        self.assertEqual(accepted["request_sha256"], final.REQUEST_SHA256)

        for key, value in (("verdict", "HOLD"),
                           ("reviewer_role", "implementer"),
                           ("final_wrapper_sha256", "0" * 64),
                           ("remote_sha256", "0" * 64),
                           ("independent_disabled_static_review_sha256", "0" * 64)):
            with self.subTest(review_field=key):
                changed = copy.deepcopy(review)
                changed[key] = value
                with self.assertRaises(RuntimeError):
                    final._validate_authority_files(
                        binding, packed(changed), packed(assignment))
        for key, value in (("issuer_role", "implementer"),
                           ("dispatch_authorized", False),
                           ("final_independent_review_sha256", "0" * 64),
                           ("target", "/tmp/wrong"),
                           ("one_attempt", False)):
            with self.subTest(assignment_field=key):
                changed = copy.deepcopy(assignment)
                changed[key] = value
                with self.assertRaises(RuntimeError):
                    final._validate_authority_files(
                        binding, review_bytes, packed(changed))

    def test_enabled_transport_rejects_live_receipt_mutations(self):
        transport = final.core.transport
        pre, write, verify = disabled_fixtures.compared_receipts()
        self.assertEqual(transport.compare_sessions(
            write, verify, preflight=pre)["status"],
            "CROSS_SESSION_PERSISTENCE_HASH_PASS")
        for key, mutate in (
            ("lock", lambda e: e["cooperative_lock_after_create"]["file"].update(
                inode=e["cooperative_lock_after_create"]["file"]["inode"] + 1)),
            ("late-label", lambda e: e["final_label_observation"]["scan"].update(
                match_count=1)),
            ("stale-open", lambda e: e.update(
                target_open_attempt_monotonic_ns=2_000_000_000)),
            ("STOP", lambda e: next(iter(e["last_stop_process_check"]["stops"].values()))
                .update(exists=True)),
            ("process", lambda e: e["last_stop_process_check"]["processes"].update(
                matched=[{"pid": 42}])),
            ("Neuron", lambda e: e["last_live_precreate_safety"]["neuron_ls"]
                .update(non_diagnostic_pids=[42])),
        ):
            with self.subTest(mutation=key):
                changed = copy.deepcopy(write)
                mutate(changed["prewrite_evidence"])
                with self.assertRaises(RuntimeError):
                    transport.compare_sessions(changed, verify, preflight=pre)
        for key, mutate in (
            ("boot", lambda p: p["identity_before"].update(boot_id="other")),
            ("mount", lambda p: p["identity_before"]["mount"].update(
                major_minor="1:1")),
            ("private-ACL", lambda p: p["host_lock_identity"]["cache_directory"]
                ["acl"].update(access="present")),
            ("disk", lambda p: p["identity_before"].update(root_free_bytes=1)),
        ):
            with self.subTest(preflight_mutation=key):
                changed = copy.deepcopy(pre)
                mutate(changed)
                with self.assertRaises(RuntimeError):
                    transport.compare_sessions(write, verify, preflight=changed)

    def test_stop_during_slow_reauthorization_reaches_marker_witness(self):
        # A persistent STOP can appear during the second authority rehash.
        # The current wrapper reaches the marker without reading STOP again.
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
             mock.patch.object(final, "_consume_fixed_marker",
                               side_effect=RuntimeError("marker reached")) as marker, \
             mock.patch.object(final.core, "Journal") as journal, \
             mock.patch.object(final.core, "_run_ssh") as ssh:
            with self.assertRaisesRegex(RuntimeError, "marker reached"):
                final.run_once()
            self.assertTrue(state["stop"])
            self.assertEqual(state["breaker_calls"], 1)
            marker.assert_called_once()
            journal.assert_not_called()
            ssh.assert_not_called()

    def test_stop_during_local_write_setup_reaches_ssh_spawn_witness(self):
        # Build argv and hash the 104 MB source after the wrapper's last STOP
        # check. Simulate a STOP during that interval, with Popen intercepted.
        with tempfile.TemporaryDirectory(prefix="trn22-final-stop-witness-") as root:
            source = Path(root) / "source.tar.gz"
            source.write_bytes(b"synthetic source")
            source.chmod(0o600)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            journal = MemoryJournal()
            state = {"stop": False, "breaker_calls": 0, "probe_calls": 0,
                     "spawn_attempts": 0}
            actual_run_ssh = final.core._run_ssh

            def breakers():
                state["breaker_calls"] += 1
                if state["stop"]:
                    raise RuntimeError("synthetic local STOP")
                return {"stops_clear": True}

            def build(operation, *, preflight=None):
                self.assertEqual(operation, "write")
                state["stop"] = True
                return ["ssh", "synthetic-only"]

            def intercept_spawn(*args, **kwargs):
                state["spawn_attempts"] += 1
                raise OSError("synthetic spawn blocked before host contact")

            def run_ssh(operation, local_journal, *, preflight=None):
                if operation == "probe":
                    state["probe_calls"] += 1
                    if state["probe_calls"] == 1:
                        return {"old_inventory": {"paths": {}}}
                    return {"operation": "probe"}
                return actual_run_ssh(operation, local_journal,
                                      preflight=preflight)

            with mock.patch.object(final, "authorize_final",
                                   return_value={"synthetic": "authorized"}), \
                 mock.patch.object(final.core.transport, "checked_local_source",
                                   return_value={"source": "synthetic"}), \
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
                                   side_effect=build), \
                 mock.patch.object(final.core.subprocess, "Popen",
                                   side_effect=intercept_spawn):
                with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                    final.run_once()
            self.assertTrue(state["stop"])
            self.assertEqual(state["breaker_calls"], 2)
            self.assertEqual(state["spawn_attempts"], 1)
            self.assertEqual(state["probe_calls"], 2)
            self.assertIn("FINAL_READ_ONLY_POSTFLIGHT",
                          [name for name, _ in journal.events])

    def test_optimized_python_still_requires_fixed_authority_files(self):
        child = (
            "from pathlib import Path\n"
            "import sys\n"
            "sys.path.insert(0,sys.argv[1])\n"
            "import operator_transfer_lock_final_once as final\n"
            "final.FINAL_REVIEW=Path(sys.argv[2])\n"
            "final.ASSIGNMENT=Path(sys.argv[3])\n"
            "final.run_once()\n"
        )
        with tempfile.TemporaryDirectory(prefix="trn22-final-opt-") as root:
            result = subprocess.run(
                [sys.executable, "-O", "-B", "-c", child, str(PACKAGE),
                 str(Path(root) / "missing-review.json"),
                 str(Path(root) / "missing-assignment.json")],
                capture_output=True, text=True, timeout=5, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FIFO FINAL REVIEW HOLD", result.stderr)


if __name__ == "__main__":
    unittest.main()
