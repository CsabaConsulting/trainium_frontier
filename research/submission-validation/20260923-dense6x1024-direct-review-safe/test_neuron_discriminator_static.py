#!/usr/bin/env python3
"""Local, hardware-free fail-closed tests for neuron_discriminator_direct.py."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("neuron_discriminator_direct", HERE / "neuron_discriminator_direct.py")
assert SPEC is not None and SPEC.loader is not None
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class DiscriminatorStaticTests(unittest.TestCase):
    def test_frozen_source_and_direct(self):
        result = D.static_source_checks()
        self.assertTrue(result["passed"], result)
        self.assertEqual(D.sha256(D.SOURCE), D.SOURCE_SHA256)
        self.assertEqual(D.sha256(D.DIRECT), D.DIRECT_SHA256)

    def test_numerical_contract_is_literal(self):
        contract = D.expected_contract()
        self.assertEqual(contract["ce_rows"], [128, 512, 32768])
        self.assertEqual(contract["ce_repetitions"], 2)
        self.assertEqual(contract["optimizer_fixtures"], [
            "tiny_fixed", "tiny_changed", "production_width1024_fs_w62"
        ])
        self.assertEqual(contract["outer_timeout_argv"], [
            "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "5400s"
        ])

    def test_recompile_limit_warning_invalidates(self):
        self.assertTrue(D.scan_recompile_warning("WARNING: recompilation count exceeded limit"))
        self.assertTrue(D.scan_recompile_warning("too many recompiles; aborting"))
        self.assertFalse(D.scan_recompile_warning("compiled graph normally"))

    def test_result_writer_rejects_overwrite_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "result.json"
            D.exclusive_json(target, {"passed": True})
            with self.assertRaises(FileExistsError):
                D.exclusive_json(target, {"passed": False})
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(FileExistsError):
                D.exclusive_json(link, {"passed": False})

    def test_pending_request_cannot_be_full_authority(self):
        request = {
            "schema": D.AUTH_SCHEMA,
            "authorization_state": "pending_external_verifier",
            "request_contract": D.expected_contract(),
            "expected_sha256": {
                name: D.sha256(D.ROOT / name) for name in D.PROTECTED_PATHS
                if (D.ROOT / name).exists()
            },
        }
        # Unit-test the trust-state branch without weakening the real fixed path.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            original = D.REQUEST
            try:
                D.REQUEST = path
                args = argparse.Namespace(
                    authorization_request=path,
                    static_only=False,
                    authorized_head="0" * 40,
                    authorization_sha256=D.sha256(path),
                )
                with self.assertRaisesRegex(ValueError, "pending authorization request cannot launch"):
                    D.authenticate(args, full=True)
            finally:
                D.REQUEST = original

    def test_changed_contract_fails_before_fingerprints(self):
        request = {
            "schema": D.AUTH_SCHEMA,
            "authorization_state": "pending_external_verifier",
            "request_contract": {**D.expected_contract(), "ce_rows": [128, 512]},
            "expected_sha256": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            original = D.REQUEST
            try:
                D.REQUEST = path
                args = argparse.Namespace(
                    authorization_request=path,
                    static_only=True,
                    authorized_head=D.PENDING_HEAD,
                    authorization_sha256=D.PENDING_SHA,
                )
                with self.assertRaisesRegex(ValueError, "contract differs"):
                    D.authenticate(args, full=False)
            finally:
                D.REQUEST = original

    def test_fingerprint_mutation_fails(self):
        hashes = {name: D.sha256(D.ROOT / name) for name in D.PROTECTED_PATHS}
        hashes[D.PROTECTED_PATHS[0]] = "f" * 64
        request = {
            "schema": D.AUTH_SCHEMA,
            "authorization_state": "pending_external_verifier",
            "request_contract": D.expected_contract(),
            "expected_sha256": hashes,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            original = D.REQUEST
            try:
                D.REQUEST = path
                args = argparse.Namespace(
                    authorization_request=path, static_only=True,
                    authorized_head=D.PENDING_HEAD, authorization_sha256=D.PENDING_SHA,
                )
                with mock.patch.object(D, "git", return_value=""):
                    with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                        D.authenticate(args, full=False)
            finally:
                D.REQUEST = original

    def test_descendant_head_fails_with_original_literals(self):
        request = {
            "schema": D.AUTH_SCHEMA,
            "authorization_state": "finalized_by_external_verifier",
            "authorized_head": "1" * 40,
            "request_contract": D.expected_contract(),
            "expected_sha256": {name: D.sha256(D.ROOT / name) for name in D.PROTECTED_PATHS},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            original = D.REQUEST
            try:
                D.REQUEST = path
                args = argparse.Namespace(
                    authorization_request=path, static_only=False,
                    authorized_head="1" * 40, authorization_sha256=D.sha256(path),
                )
                with mock.patch.object(D, "git", side_effect=["2" * 40, ""]):
                    with self.assertRaisesRegex(ValueError, "exact externally authorized commit"):
                        D.authenticate(args, full=True)
            finally:
                D.REQUEST = original

    def test_exact_launch_argv_rejects_timeout_substitution_by_construction(self):
        args = argparse.Namespace(
            run_label="fs-direct-discriminator-verifier-supplied",
            expected_commit="1" * 40,
            authorized_head="1" * 40,
            authorization_sha256="2" * 64,
            report=D.ROOT / "research/run-logs/fs-direct-discriminator-verifier-supplied-result.json",
            expected_hostname=D.EXPECTED_HOSTNAME,
        )
        request = D.exact_launch_request(args, {"request_sha256": "2" * 64})
        self.assertEqual(request["command"][:5], D.OUTER_TIMEOUT)
        self.assertNotIn("9999s", request["command"])
        self.assertEqual(request["command"][5], D.NATIVE_PYTHON)
        self.assertEqual(request["environment"], {
            **D.STATIC_ENVIRONMENT,
            "TRAINIUM_DISCRIMINATOR_LABEL": args.run_label,
            "TRAINIUM_DISCRIMINATOR_EXPECTED_COMMIT": args.expected_commit,
            "TRAINIUM_DISCRIMINATOR_AUTHORIZED_HEAD": args.authorized_head,
            "TRAINIUM_DISCRIMINATOR_AUTHORIZATION_SHA256": args.authorization_sha256,
        })

    def test_optimizer_comparator_is_exact_and_plan_sensitive(self):
        fixtures = ("tiny_fixed", "tiny_changed", "production_width1024_fs_w62")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in ("raw", "source", "direct"):
                for rank in range(4):
                    reports = []
                    for name in fixtures:
                        state = {"g0:p0": {"step": 1}} if rank == 0 else {}
                        snapshot = {
                            "step": 1,
                            "parameters": {"0": {"sha256": "a" * 64}},
                            "state": {"g0:p0": {"step": 1}} if variant == "raw" else state,
                            "group_scalars": [{"lr": 0.1}],
                            "owner_calls": 1,
                            "cross_rank_parameter_sha256": ["a" * 64] * 4,
                            "cross_rank_exact": True,
                        }
                        reports.append({
                            "fixture": name, "passed": True, "parameter_count": 1,
                            "element_count": 1, "plan": {"entries": [{"owner": 0}], "summary": {}},
                            "snapshots": [snapshot],
                        })
                    payload = {"passed": True, "fixtures": reports}
                    (root / f"optimizer-{variant}-rank{rank}.json").write_text(json.dumps(payload))
            self.assertTrue(D.compare_optimizer_results(root)["passed"])
            path = root / "optimizer-direct-rank0.json"
            payload = json.loads(path.read_text())
            payload["fixtures"][0]["plan"]["entries"][0]["owner"] = 1
            path.write_text(json.dumps(payload))
            comparison = D.compare_optimizer_results(root)
            self.assertFalse(comparison["passed"])
            self.assertTrue(any("ownership plan" in failure for failure in comparison["failures"]))


if __name__ == "__main__":
    unittest.main()
