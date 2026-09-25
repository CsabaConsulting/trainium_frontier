"""Offline tests: an enabled request alone is never stage authority."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REQUEST = HERE / "REQUEST.ENABLED_PENDING_FINAL_REVIEW.json"
EXPECTED_REQUEST_SHA256 = "25092856f0fef3c1e3cf85e40e45fa0d2d51971b7b8112f50dbc3761770971f2"

spec = importlib.util.spec_from_file_location(
    "trn22_r10_stage_operator_under_test", HERE / "operator_once.py"
)
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class EnabledPendingRequestTests(unittest.TestCase):
    def test_request_is_canonical_and_matches_source_authority_schema(self):
        raw = REQUEST.read_bytes()
        request = json.loads(raw)
        self.assertEqual(raw.strip(), json.dumps(request, sort_keys=True,
                                                 separators=(",", ":")).encode())
        self.assertEqual(sha256(raw), EXPECTED_REQUEST_SHA256)
        self.assertTrue(request["dispatch_authorized"])
        self.assertTrue(request["source_stage_authorized"])
        self.assertFalse(request["native_training_authorized"])
        self.assertFalse(request["evaluation_authorized"])
        # The source itself is unchanged; the request contains every exact
        # field the local authority gate compares before any marker/SSH.
        expected = {
            "schema": "trn22-r10-incremental-stage-r1-enabled-request-v1",
            "dispatch_authorized": True,
            "source_stage_authorized": True,
            "scope": operator.SCOPE,
            "host": "trn22",
            "hostname": "ip-172-31-32-253",
            "source_commit": operator.BASE_COMMIT,
            "target_commit": operator.TARGET_COMMIT,
            "source": operator.SOURCE,
            "target": operator.TARGET,
            "remote_bundle": operator.REMOTE_BUNDLE,
            "bundle_sha256": operator.BUNDLE_SHA256,
            "bundle_bytes": operator.BUNDLE_BYTES,
            "remote_source_sha256": request["remote_source_sha256"],
            "operator_source_sha256": request["operator_source_sha256"],
            "attempt_label": operator.LABEL,
            "marker": str(operator.MARKER),
            "one_attempt": True,
            "prior_r3_hold_commit": operator.PRIOR_R3_HOLD_COMMIT,
            "prior_r3_hold_sha256": operator.PRIOR_R3_HOLD_SHA256,
            "r4_static_pass_commit": operator.R4_STATIC_PASS_COMMIT,
            "r4_static_pass_sha256": operator.R4_STATIC_PASS_SHA256,
            "r10_source_pass_commit": operator.R10_SOURCE_PASS_COMMIT,
            "r10_source_pass_sha256": operator.R10_SOURCE_PASS_SHA256,
            "provisional_failure_policy": operator.PROVISIONAL_FAILURE_POLICY,
            "native_training_authorized": False,
            "evaluation_authorized": False,
        }
        self.assertEqual(request, expected)

    def test_exact_unchanged_source_bundle_and_science_root(self):
        request = json.loads(REQUEST.read_text())
        self.assertEqual(
            sha256((HERE / "remote_incremental.py").read_bytes()),
            request["remote_source_sha256"],
        )
        self.assertEqual(
            sha256((HERE / "operator_once.py").read_bytes()),
            request["operator_source_sha256"],
        )
        bundle = (HERE / "e65-to-r10.bundle").read_bytes()
        self.assertEqual((len(bundle), sha256(bundle)),
                         (request["bundle_bytes"], request["bundle_sha256"]))
        self.assertEqual(
            sha256((ROOT / "train.py").read_bytes()),
            "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586",
        )
        self.assertEqual(
            sha256((ROOT / "prepare.py").read_bytes()),
            "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb",
        )
        disabled = json.loads((HERE / "REQUEST.HOLD.json").read_text())
        self.assertFalse(disabled["dispatch_authorized"])
        self.assertFalse(disabled["source_stage_authorized"])
        self.assertEqual(
            (request["source_commit"], request["target_commit"], request["source"],
             request["target"], request["remote_bundle"], request["attempt_label"],
             request["marker"]),
            (disabled["source_commit"], disabled["target_commit"], disabled["source_path"],
             disabled["new_target_path"], disabled["remote_bundle"],
             disabled["attempt_label"], disabled["one_attempt_marker"]),
        )

    def test_new_r5_authorities_are_distinct_from_preserved_r4_evidence(self):
        prior_review = HERE / "INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW.PASS.json"
        prior_assignment = HERE / "SUPERVISOR_ONE_ATTEMPT_SOURCE_STAGE_ASSIGNMENT.json"
        self.assertEqual(sha256(prior_review.read_bytes()),
                         "c9be026c2609fbbdcfdc85005630675959a2b6661befb91268dd4cf73d59d615")
        self.assertEqual(sha256(prior_assignment.read_bytes()),
                         "ed07f024e45b5a606c4c8ed3f0ad6bc335a6bcaead548bf89c1e0cbcd2ec5bb4")
        self.assertNotEqual(operator.FINAL_REVIEW, prior_review)
        self.assertNotEqual(operator.ASSIGNMENT, prior_assignment)
        self.assertIn("R5", operator.FINAL_REVIEW.name)
        self.assertIn("R5", operator.ASSIGNMENT.name)
        self.assertEqual(operator.ENABLED_REQUEST, REQUEST)
        old_review = json.loads(prior_review.read_text())
        old_assignment = json.loads(prior_assignment.read_text())
        self.assertNotEqual(old_review["schema"],
                            "trn22-r10-incremental-stage-r1-r5-independent-final-review-v1")
        self.assertNotEqual(old_assignment["schema"],
                            "trn22-r10-incremental-stage-r1-r5-supervisor-assignment-v1")
        with self.assertRaisesRegex(RuntimeError, "actual independent final source-stage PASS"):
            operator._authority(
                json.loads(REQUEST.read_text()), sha256(REQUEST.read_bytes()),
                old_review, sha256(prior_review.read_bytes()), old_assignment,
                sha256((HERE / "remote_incremental.py").read_bytes()),
                sha256((HERE / "operator_once.py").read_bytes()),
            )

    def test_runbook_command_targets_exact_repaired_operator(self):
        designated_operator = Path(
            "/home/poojith-devan/Documents/ChatGPT/"
            "Trainium-trn22-r10-remote-timeout-repair-r1-disabled/"
            "research/v2/authorizations/"
            "trn22-c5c-a0-r10-incremental-stage-r1-hold/operator_once.py"
        )
        runbook = (HERE / "FINAL_PENDING_RUNBOOK.md").read_text(encoding="utf-8")
        commands = [line for line in runbook.splitlines()
                    if line.startswith("/usr/bin/python3 -B ")]
        self.assertEqual(commands, ["/usr/bin/python3 -B " + str(designated_operator)])
        operator_hash = json.loads(REQUEST.read_text())["operator_source_sha256"]
        self.assertEqual(operator_hash,
                         "6224763c0acdbba8bfadf4281498fe9801a8b8d4aaee3d37f8b58f2f97b412be")
        self.assertEqual(sha256((HERE / "operator_once.py").read_bytes()), operator_hash)
        self.assertEqual(sha256(designated_operator.read_bytes()), operator_hash)

    def test_enabled_request_alone_refuses_before_ssh_or_marker(self):
        remote_call = mock.Mock(side_effect=AssertionError("remote called"))
        with tempfile.TemporaryDirectory(prefix="trn22-r10-absent-r5-") as temp:
            with (
                mock.patch.object(operator, "FINAL_REVIEW", Path(temp) / "missing-review"),
                mock.patch.object(operator, "ASSIGNMENT", Path(temp) / "missing-assignment"),
                mock.patch.object(operator, "_local_stop_and_breakers") as stop,
                mock.patch.object(operator, "_ssh_identity") as ssh,
                mock.patch.object(operator, "_remote_timeout_identity") as probe,
                mock.patch.object(operator, "_marker") as marker,
            ):
                if sys.flags.optimize:
                    with self.assertRaisesRegex(RuntimeError, "optimized Python"):
                        operator.stage_once(remote_call=remote_call)
                else:
                    with self.assertRaises(FileNotFoundError):
                        operator.stage_once(remote_call=remote_call)
                stop.assert_not_called()
                ssh.assert_not_called()
                probe.assert_not_called()
                marker.assert_not_called()
                remote_call.assert_not_called()

    def test_even_an_in_memory_review_without_assignment_refuses(self):
        if sys.flags.optimize:
            self.skipTest("optimized Python refuses before any authority read")
        original_json = operator._json

        def without_assignment(path):
            if path == operator.FINAL_REVIEW:
                return {"schema": "in-memory-only"}, "0" * 64
            return original_json(path)

        remote_call = mock.Mock(side_effect=AssertionError("remote called"))
        with tempfile.TemporaryDirectory(prefix="trn22-r10-absent-assignment-") as temp:
            with (
                mock.patch.object(operator, "ASSIGNMENT", Path(temp) / "missing-assignment"),
                mock.patch.object(operator, "_json", side_effect=without_assignment),
                mock.patch.object(operator, "_local_stop_and_breakers") as stop,
                mock.patch.object(operator, "_ssh_identity") as ssh,
                mock.patch.object(operator, "_remote_timeout_identity") as probe,
                mock.patch.object(operator, "_marker") as marker,
            ):
                with self.assertRaises(FileNotFoundError):
                    operator.stage_once(remote_call=remote_call)
                stop.assert_not_called()
                ssh.assert_not_called()
                probe.assert_not_called()
                marker.assert_not_called()
                remote_call.assert_not_called()

    def _check_late_desktop_change(self, kind, during):
        with tempfile.TemporaryDirectory(prefix="trn22-r10-late-local-") as temp:
            root = Path(temp)
            desktop = root / "Desktop" / "Trainium"
            documents = root / "Documents" / "ChatGPT" / "Trainium"
            fake_here = root / "package" / "research" / "v2" / "authorizations" / "entry"
            control = desktop / "research" / "v2"
            control.mkdir(parents=True)
            state = {
                "breakers": {"global_correctness": {"tripped": False},
                             "trn22": {"tripped": False}},
                "launch_authorized": False,
                "hosts": {"trn22": {"launch_allowed": False}},
            }
            state_file = control / "state.json"
            state_file.write_text(json.dumps(state), encoding="utf-8")
            marker_path = root / "one-attempt-marker"

            def change_desktop():
                if kind == "STOP":
                    (control / "STOP").write_text("synthetic late STOP\n", encoding="utf-8")
                else:
                    state["breakers"]["trn22"]["tripped"] = True
                    state_file.write_text(json.dumps(state), encoding="utf-8")

            def identity_then_change_desktop():
                if during == "identity":
                    change_desktop()

            def probe_then_change_desktop():
                if during == "probe":
                    change_desktop()

            remote_call = mock.Mock(side_effect=AssertionError("SSH attempted"))
            # Exercise the seam under both normal and -O bytecode. The real -O
            # entrypoint remains forbidden and is checked separately below.
            synthetic_sys = types.SimpleNamespace(flags=types.SimpleNamespace(optimize=0))
            with (
                mock.patch.object(operator, "CONTROL_ROOT", desktop),
                mock.patch.object(operator, "FORMER_CONTROL_ROOT", documents),
                mock.patch.object(operator, "HERE", fake_here),
                mock.patch.object(operator, "MARKER", marker_path),
                mock.patch.object(operator, "PRIOR_MARKERS", ()),
                mock.patch.object(operator, "sys", synthetic_sys),
                mock.patch.dict(operator.os.environ, {}, clear=True),
                mock.patch.object(operator, "_json", side_effect=[
                    ({"dispatch_authorized": True}, "requesthash"),
                    ({}, "reviewhash"), ({}, "assignmenthash")]),
                mock.patch.object(operator, "_source_bytes", return_value=(b"bundle", b"remote")),
                mock.patch.object(operator, "_authority"),
                mock.patch.object(operator, "_ssh_identity",
                                  side_effect=identity_then_change_desktop) as identity,
                mock.patch.object(operator, "_remote_timeout_identity",
                                  side_effect=probe_then_change_desktop) as probe,
                mock.patch.object(operator, "_local_stop_and_breakers",
                                  wraps=operator._local_stop_and_breakers) as stop,
                mock.patch.object(operator, "_marker", wraps=operator._marker) as marker,
            ):
                reason = ("local STOP present" if kind == "STOP" else
                          "controller STOP/breaker/launch state")
                with self.assertRaisesRegex(RuntimeError, reason):
                    operator.stage_once(remote_call=remote_call)
                self.assertEqual(stop.call_count, 2 if during == "identity" else 3)
                identity.assert_called_once_with()
                if during == "identity":
                    probe.assert_not_called()
                else:
                    probe.assert_called_once_with()
                marker.assert_not_called()
                remote_call.assert_not_called()
            self.assertFalse(marker_path.exists())

    def test_late_desktop_stop_after_identity_refuses_before_marker(self):
        self._check_late_desktop_change("STOP", "identity")

    def test_late_desktop_breaker_after_identity_refuses_before_marker(self):
        self._check_late_desktop_change("breaker", "identity")

    def test_late_desktop_stop_after_remote_probe_refuses_before_marker(self):
        self._check_late_desktop_change("STOP", "probe")

    def test_late_desktop_breaker_after_remote_probe_refuses_before_marker(self):
        self._check_late_desktop_change("breaker", "probe")

    def test_direct_entrypoint_holds_in_normal_and_optimized_python(self):
        # Run from an isolated copy with no r5 authority. This remains safe
        # even after an independent review and assignment are later added.
        with tempfile.TemporaryDirectory(prefix="trn22-r10-entry-hold-") as temp:
            copy = Path(temp)
            for name in ("operator_once.py", "REQUEST.ENABLED_PENDING_FINAL_REVIEW.json",
                         "e65-to-r10.bundle", "remote_incremental.py"):
                shutil.copyfile(HERE / name, copy / name)
            for flags in (["-B"], ["-O", "-B"]):
                with self.subTest(flags=flags):
                    result = subprocess.run(
                        [sys.executable, *flags, str(copy / "operator_once.py")],
                        cwd=ROOT, stdin=subprocess.DEVNULL,
                        text=True, capture_output=True, timeout=15, check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("HOLD:", result.stderr)


if __name__ == "__main__":
    unittest.main()
