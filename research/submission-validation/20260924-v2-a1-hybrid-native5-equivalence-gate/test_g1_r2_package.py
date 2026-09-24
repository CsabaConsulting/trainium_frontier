#!/usr/bin/env python3
"""Local-only adversarial checks for the non-executable G1 r2 retry package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "research"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


LAUNCH = load("fresh_start_launch_r2_test", ROOT / "research/fresh_start_launch.py")
FINAL = load("g1_r2_finalization_test", HERE / "g1_finalization.py")


class IdentityTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identity = self.root / "controller-identity"
        self.identity.write_bytes(b"TEST-KEY-BYTES-MUST-NEVER-BE-READ")
        self.identity.chmod(0o600)

    def tearDown(self):
        self.temp.cleanup()

    def test_lstat_only_identity_validation_and_exact_argv(self):
        original_open = Path.open
        original_read_bytes = Path.read_bytes

        def guarded_open(path, *args, **kwargs):
            if path == self.identity:
                raise AssertionError("identity contents were opened")
            return original_open(path, *args, **kwargs)

        def guarded_read(path, *args, **kwargs):
            if path == self.identity:
                raise AssertionError("identity contents were read")
            return original_read_bytes(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open), \
                mock.patch.object(Path, "read_bytes", guarded_read):
            self.assertEqual(LAUNCH.validate_identity_file(self.identity), self.identity)
        common = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                  "-o", "IdentitiesOnly=yes", "-i", str(self.identity)]
        self.assertEqual(LAUNCH.transport_argv("ssh", self.identity), ["ssh", *common])
        self.assertEqual(LAUNCH.transport_argv("scp", self.identity), ["scp", *common])
        with self.assertRaises(ValueError):
            LAUNCH.transport_argv("rsync", self.identity)

    def test_rejects_absent_relative_symlink_mode_link_owner_and_nonregular(self):
        attacks = []
        attacks.append((Path("relative-key"), None))
        attacks.append((self.root / "absent", None))
        symlink = self.root / "symlink"
        symlink.symlink_to(self.identity)
        attacks.append((symlink, None))
        directory = self.root / "directory"
        directory.mkdir()
        directory.chmod(0o600)
        attacks.append((directory, None))
        wrong_mode = self.root / "wrong-mode"
        wrong_mode.write_bytes(b"x")
        wrong_mode.chmod(0o640)
        attacks.append((wrong_mode, None))
        linked = self.root / "linked"
        linked.write_bytes(b"x")
        linked.chmod(0o600)
        os.link(linked, self.root / "linked-again")
        attacks.append((linked, None))
        attacks.append((self.identity, mock.patch.object(LAUNCH.os, "getuid",
                                                         return_value=os.getuid() + 1)))
        for path, owner_patch in attacks:
            with self.subTest(path=path):
                context = owner_patch if owner_patch is not None else mock.patch.object(
                    LAUNCH.os, "getuid", wraps=os.getuid)
                with context, self.assertRaises(ValueError):
                    LAUNCH.validate_identity_file(path)

    def _request(self, root: Path) -> Path:
        request = {
            "host": "trn21", "label": "fs-r2-local-test", "expected_commit": "a" * 40,
            "remote": "/home/ubuntu/trainium_freshstart_r2_local_test",
            "command": ["timeout", "1s", "true"],
            "environment": {"NEURON_LOGICAL_NC_CONFIG": "2",
                            "NEURON_CC_FLAGS": "--optlevel=1"},
            "min_free_bytes": 6_979_321_856,
        }
        target = root / "request.json"
        target.write_text(json.dumps(request))
        return target

    def test_main_uses_exact_ssh_and_never_serializes_identity(self):
        controller = self.root / "controller"
        (controller / "research/run-logs").mkdir(parents=True)
        request = self._request(controller)
        completed = subprocess.CompletedProcess([], 0, stdout="remote-ok\n", stderr="")
        with mock.patch.object(LAUNCH, "ROOT", controller), \
                mock.patch.object(LAUNCH.subprocess, "run", return_value=completed) as run, \
                mock.patch.object(sys, "argv", ["fresh_start_launch.py", str(request),
                                                 "--identity-file", str(self.identity)]):
            LAUNCH.main()
        argv = run.call_args.args[0]
        self.assertEqual(argv[:-2], LAUNCH.transport_argv("ssh", self.identity))
        self.assertEqual(argv[-2], LAUNCH.HOSTS["trn21"])
        self.assertEqual(run.call_args.kwargs["timeout"], 90)
        self.assertEqual(json.loads(run.call_args.kwargs["input"]), json.loads(request.read_text()))
        dispatch = controller / "research/run-logs/fs-r2-local-test.dispatch.json"
        durable = dispatch.read_bytes()
        self.assertNotIn(str(self.identity).encode(), durable)
        self.assertNotIn(b"TEST-KEY-BYTES", durable)
        key_digest = hashlib.sha256(b"TEST-KEY-BYTES-MUST-NEVER-BE-READ").hexdigest().encode()
        self.assertNotIn(key_digest, durable)

    def test_timeout_fails_closed_without_identity_in_receipt_or_error(self):
        controller = self.root / "timeout-controller"
        (controller / "research/run-logs").mkdir(parents=True)
        request = self._request(controller)
        timeout = subprocess.TimeoutExpired(["ssh", "-i", str(self.identity)], 90)
        with mock.patch.object(LAUNCH, "ROOT", controller), \
                mock.patch.object(LAUNCH.subprocess, "run", side_effect=timeout), \
                mock.patch.object(sys, "argv", ["fresh_start_launch.py", str(request),
                                                 "--identity-file", str(self.identity)]):
            with self.assertRaisesRegex(RuntimeError, "SSH transport timed out") as raised:
                LAUNCH.main()
        self.assertNotIn(str(self.identity), str(raised.exception))
        durable = (controller / "research/run-logs/fs-r2-local-test.dispatch.json").read_bytes()
        self.assertNotIn(str(self.identity).encode(), durable)
        self.assertEqual(json.loads(durable)["transport_status"], "TIMEOUT")

    def test_nonzero_redacts_identity_and_missing_key_never_calls_ssh(self):
        controller = self.root / "nonzero-controller"
        (controller / "research/run-logs").mkdir(parents=True)
        request = self._request(controller)
        failed = subprocess.CompletedProcess([], 255, stdout="", stderr=(
            f"identity {self.identity} unavailable\n"))
        with mock.patch.object(LAUNCH, "ROOT", controller), \
                mock.patch.object(LAUNCH.subprocess, "run", return_value=failed), \
                mock.patch.object(sys, "argv", ["fresh_start_launch.py", str(request),
                                                 "--identity-file", str(self.identity)]):
            with self.assertRaisesRegex(RuntimeError, "nonzero status 255") as raised:
                LAUNCH.main()
        self.assertNotIn(str(self.identity), str(raised.exception))
        durable = (controller / "research/run-logs/fs-r2-local-test.dispatch.json").read_bytes()
        self.assertNotIn(str(self.identity).encode(), durable)
        self.assertIn(b"[controller-identity]", durable)
        with mock.patch.object(LAUNCH.subprocess, "run") as run, \
                mock.patch.object(sys, "argv", ["fresh_start_launch.py", str(request),
                                                 "--identity-file", str(self.root / "missing")]):
            with self.assertRaises(ValueError):
                LAUNCH.main()
            run.assert_not_called()

    def test_request_cannot_alias_or_symlink_to_identity(self):
        controller = self.root / "alias-controller"
        (controller / "research/run-logs").mkdir(parents=True)
        alias = controller / "request-alias.json"
        alias.symlink_to(self.identity)
        original_open = LAUNCH.os.open

        def guarded_open(path, *args, **kwargs):
            if Path(path) == self.identity:
                raise AssertionError("identity inode opened")
            return original_open(path, *args, **kwargs)

        for request in (self.identity, alias):
            with self.subTest(request=request), \
                    mock.patch.object(LAUNCH.os, "open", side_effect=guarded_open), \
                    mock.patch.object(LAUNCH.os, "read",
                                      side_effect=AssertionError("identity bytes read")), \
                    mock.patch.object(LAUNCH.subprocess, "run") as run, \
                    mock.patch.object(sys, "argv", ["fresh_start_launch.py", str(request),
                                                     "--identity-file", str(self.identity)]):
                with self.assertRaises(ValueError):
                    LAUNCH.main()
                run.assert_not_called()


class R2HistoricalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = json.loads((HERE / "g1-request.template.json").read_text())
        cls.pending = json.loads((HERE / "g1-r2-authorization.pending.json").read_text())
        cls.r2_auth = json.loads((HERE / "g1-r2-authorization.json").read_text())

    def test_r2_is_historical_consumed_and_r3_is_the_only_pending_template(self):
        FINAL.validate_template(self.template)
        self.assertEqual(self.template["label"], FINAL.G1_LABEL)
        self.assertTrue((HERE / "g1-r2-authorization.json").exists())
        self.assertTrue(self.r2_auth["executable"])
        self.assertEqual(self.r2_auth["label"],
                         "fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2")
        self.assertFalse(self.template["executable"])
        self.assertFalse(self.pending["executable"])
        self.assertFalse(FINAL.MATERIALIZATION_OUTPUT.exists())

    def test_consumed_r2_authority_cannot_validate_r3(self):
        with self.assertRaises(FINAL.ContractError):
            FINAL.validate_authorization(
                self.r2_auth, self.template,
                hashlib.sha256((HERE / "g1-request.template.json").read_bytes()).hexdigest())

    def test_r2_transport_amendment_remains_immutable_historical_evidence(self):
        amendment = json.loads((HERE / "G1_R2_TRANSPORT_RETRY_AMENDMENT.json").read_text())
        self.assertFalse(amendment["scientific_change"])
        self.assertTrue(amendment["r1_failure"]["historical_consumed"])
        self.assertFalse(amendment["r1_failure"]["training_started"])
        self.assertEqual(hashlib.sha256((HERE / "G1_R2_TRANSPORT_RETRY_AMENDMENT.json").read_bytes()).hexdigest(),
                         "f90813f72bb0e5a21b86b89d24550f5f9d701aa4bc076915e9b6e41beb97447d")

    def test_r2_failure_provenance_is_exact_and_zero_update(self):
        failure = self.template["r2_correctness_failure"]
        self.assertEqual(failure["failure_commit"],
                         "5b392db4bc8c01a47da2f5969cb33df0c50ec52b")
        self.assertEqual(failure["analysis_sha256"],
                         "014288adbbcfaa516cbdcf141e5f70a7dc7418fa63191f6797014ee0b5b20c3a")
        self.assertEqual(failure["request_verification_commit"],
                         "65b807dab3ea6a785e44aef91fef32c5168b3879")
        self.assertEqual(failure["rematerialized_verification_sha256"],
                         "b86b2cc04c93b89392eabe8a89cbde4a46dd06fde22428fbe739a1a7d7fbbf0c")
        self.assertEqual(failure["training_updates"], 0)
        self.assertEqual(failure["rank_reports"], 0)
        self.assertFalse(failure["checkpoint_exists"])
        self.assertFalse(failure["retry_under_r2_identity_authorized"])

    def test_r2_failure_provenance_mutation_fails_closed(self):
        for field in ("failure_commit", "analysis_sha256", "authorization_sha256",
                      "concrete_request_sha256", "runtime_result_sha256"):
            with self.subTest(field=field):
                candidate = json.loads(json.dumps(self.template))
                candidate["r2_correctness_failure"][field] = "0" * len(
                    candidate["r2_correctness_failure"][field])
                with self.assertRaises(FINAL.ContractError):
                    FINAL.validate_template(candidate)

    def test_repaired_launcher_is_still_exact_and_contains_no_request_identity(self):
        self.assertEqual(hashlib.sha256((ROOT / "research/fresh_start_launch.py").read_bytes()).hexdigest(),
                         "7620fea10bb51c54481b8347ec9e97ebd40ce8f5ee206dc3f203c9483c35ee2f")
        encoded = json.dumps(self.template, sort_keys=True)
        self.assertNotIn("--identity-file", encoded)
        self.assertNotIn(".pem", encoded)


if __name__ == "__main__":
    unittest.main()
