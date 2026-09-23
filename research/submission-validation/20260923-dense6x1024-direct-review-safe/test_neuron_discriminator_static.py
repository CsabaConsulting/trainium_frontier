#!/usr/bin/env python3
"""Local, hardware-free fail-closed tests for neuron_discriminator_direct.py."""
from __future__ import annotations

import argparse
import copy
import hashlib
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
    @staticmethod
    def launch_args():
        return argparse.Namespace(
            run_label="fs-direct-discriminator-verifier-supplied",
            expected_commit="1" * 40,
            authorized_head="1" * 40,
            authorization_sha256="2" * 64,
            report=D.ROOT / "research/run-logs/fs-direct-discriminator-verifier-supplied-result.json",
            expected_hostname=D.EXPECTED_HOSTNAME,
        )

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
            "purpose": D.EXPECTED_PURPOSE,
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
                with mock.patch.object(D, "require_request_integrity", return_value={"tracked": True, "clean": True}):
                    with self.assertRaisesRegex(ValueError, "pending authorization request cannot launch"):
                        D.authenticate(args, full=True)
            finally:
                D.REQUEST = original

    def test_changed_contract_fails_before_fingerprints(self):
        request = {
            "schema": D.AUTH_SCHEMA,
            "authorization_state": "pending_external_verifier",
            "purpose": D.EXPECTED_PURPOSE,
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
                    authorization_sha256=D.sha256(path),
                )
                with mock.patch.object(D, "require_request_integrity", return_value={"tracked": True, "clean": True}):
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
            "purpose": D.EXPECTED_PURPOSE,
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
                    authorized_head=D.PENDING_HEAD, authorization_sha256=D.sha256(path),
                )
                with mock.patch.object(D, "require_request_integrity", return_value={"tracked": True, "clean": True}):
                    with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                        D.authenticate(args, full=False)
            finally:
                D.REQUEST = original

    def test_descendant_head_fails_with_original_literals(self):
        request = {
            "schema": D.AUTH_SCHEMA,
            "authorization_state": "finalized_by_external_verifier",
            "purpose": D.EXPECTED_PURPOSE,
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
                with mock.patch.object(D, "require_request_integrity", return_value={"tracked": True, "clean": True}):
                    with mock.patch.object(D, "git", side_effect=["2" * 40, ""]):
                        with self.assertRaisesRegex(ValueError, "exact externally authorized commit"):
                            D.authenticate(args, full=True)
            finally:
                D.REQUEST = original

    def test_exact_launch_argv_rejects_timeout_substitution_by_construction(self):
        args = self.launch_args()
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
        self.assertEqual(request["environment"]["WORLD_SIZE"], "4")
        self.assertEqual(request["environment"]["LOCAL_WORLD_SIZE"], "4")

    def test_generated_environment_is_exactly_accepted_and_missing_world_rejected(self):
        args = self.launch_args()
        environment = D.expected_outer_environment(args)
        runtime = {**environment, "TRAINIUM_FRESH_START_RUN_ID": args.run_label}
        with mock.patch.dict(D.os.environ, runtime, clear=True):
            with mock.patch.object(D.socket, "gethostname", return_value=D.EXPECTED_HOSTNAME):
                with mock.patch.object(D.sys, "executable", D.NATIVE_PYTHON):
                    with mock.patch.object(D, "git", return_value=args.expected_commit):
                        D.validate_full_environment(args)
        for missing in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
            broken = dict(runtime)
            broken.pop(missing)
            with mock.patch.dict(D.os.environ, broken, clear=True):
                with self.assertRaisesRegex(ValueError, f"environment mismatch for {missing}"):
                    D.validate_full_environment(args)

    def test_launcher_default_hash_union_and_manifest_are_literal(self):
        args = self.launch_args()
        authorization = json.loads(D.REQUEST.read_text())
        authorized_request = D.exact_launch_request(args, {})
        values = dict(authorization["expected_sha256"])
        values[D.rel(D.REQUEST)] = "2" * 64
        required = {path: values[path] for path in D.launcher_fingerprint_paths()}
        self.assertEqual(
            set(D.launcher_fingerprint_paths()),
            set(D.LAUNCHER_DEFAULT_FINGERPRINT_PATHS) | set(D.PROTECTED_PATHS) | {D.rel(D.REQUEST)},
        )
        for required_default in ("train.py", "prepare.py", "research/fresh_start_backup.py"):
            self.assertIn(required_default, required)
        payload = {
            **authorized_request,
            "started_at": "2026-09-23T00:00:00+00:00",
            "hostname": D.EXPECTED_HOSTNAME,
            "git_commit": args.expected_commit,
            "source_dirty": False,
            "preflight_free_bytes": 9 * 1024**3,
            "source_sha256": required,
        }
        D.validate_launcher_payload(payload, authorized_request, required, args, "job.json")
        mutations = []
        changed_env = copy.deepcopy(payload)
        changed_env["environment"]["WORLD_SIZE"] = "999"
        mutations.append(changed_env)
        changed_timeout = copy.deepcopy(payload)
        changed_timeout["command"][4] = "9999s"
        mutations.append(changed_timeout)
        missing_default = copy.deepcopy(payload)
        missing_default["source_sha256"].pop("train.py")
        mutations.append(missing_default)
        extra_hash = copy.deepcopy(payload)
        extra_hash["source_sha256"]["extra.py"] = "0" * 64
        mutations.append(extra_hash)
        for mutation in mutations:
            with self.assertRaises(ValueError):
                D.validate_launcher_payload(mutation, authorized_request, required, args, "job.json")

    def test_authorization_schema_rejects_embedded_self_sha(self):
        request = json.loads(D.REQUEST.read_text())
        request["authorization_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            original = D.REQUEST
            try:
                D.REQUEST = path
                args = argparse.Namespace(
                    authorization_request=path, static_only=True,
                    authorized_head=D.PENDING_HEAD, authorization_sha256=D.sha256(path),
                )
                with mock.patch.object(D, "require_request_integrity", return_value={"tracked": True, "clean": True}):
                    with self.assertRaisesRegex(ValueError, "top-level key schema"):
                        D.authenticate(args, full=False)
            finally:
                D.REQUEST = original

    def test_authorization_request_integrity_rejects_symlink_untracked_and_dirty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("{}")
            link = root / "request.json"
            link.symlink_to(target)
            original = D.REQUEST
            try:
                D.REQUEST = link.absolute()
                with self.assertRaisesRegex(ValueError, "regular and non-symlink"):
                    D.require_request_integrity(link)
                D.REQUEST = target.absolute()
                with mock.patch.object(D, "rel", return_value="request.json"):
                    with mock.patch.object(D, "git", side_effect=D.subprocess.CalledProcessError(1, "git")):
                        with self.assertRaisesRegex(ValueError, "not tracked"):
                            D.require_request_integrity(target)
                    with mock.patch.object(D, "git", side_effect=["request.json", " M request.json"]):
                        with self.assertRaisesRegex(ValueError, "dirty/untracked"):
                            D.require_request_integrity(target)
            finally:
                D.REQUEST = original

    @staticmethod
    def tensor_record(label):
        return {
            "sha256": hashlib.sha256(label.encode()).hexdigest(),
            "shape": [1], "dtype": "torch.float32", "finite": True,
            "nonzero": True, "l2": 1.0, "max_abs": 1.0,
        }

    @classmethod
    def optimizer_matrix(cls):
        variants = {}
        for variant in ("raw", "source", "direct"):
            variants[variant] = []
            for rank in range(4):
                fixtures = []
                for fixture_name, spec in D.FIXTURE_SPECS.items():
                    count, total = spec["parameter_count"], spec["element_count"]
                    elements = [total - count + 1] + [1] * (count - 1)
                    entries = []
                    owner_elements = [0] * 4
                    owner_costs = [0.0] * 4
                    for index, amount in enumerate(elements):
                        owner = index % 4
                        kind = "adamw" if index % 2 == 0 else "muon"
                        cost = 1.0 + index / 1000
                        entries.append({
                            "index": index, "group": index, "position": 0, "kind": kind,
                            "shape": [amount], "elements": amount, "estimated_ms": cost,
                            "owner": owner, "offset": owner_elements[owner],
                        })
                        owner_elements[owner] += amount
                        owner_costs[owner] += cost
                    stride = (max(owner_elements) + 127) // 128 * 128
                    plan = {
                        "entries": entries,
                        "summary": {
                            "elements_by_owner": owner_elements,
                            "estimated_ms_by_owner": owner_costs,
                            "shard_elements": stride,
                            "total_padded_bytes": stride * 4 * 4,
                        },
                    }

                    def parameters(version):
                        return {
                            str(index): cls.tensor_record(f"{fixture_name}:parameter:{version}:{index}")
                            for index in range(count)
                        }

                    def state(step):
                        records = {}
                        for entry in entries:
                            if variant != "raw" and entry["owner"] != rank:
                                continue
                            key = f"g{entry['group']}:p0"
                            if entry["kind"] == "adamw":
                                records[key] = {
                                    "step": step,
                                    "exp_avg": cls.tensor_record(f"{fixture_name}:state:{step}:{key}:avg"),
                                    "exp_avg_sq": cls.tensor_record(f"{fixture_name}:state:{step}:{key}:sq"),
                                }
                            else:
                                records[key] = {
                                    "momentum_buffer": cls.tensor_record(f"{fixture_name}:state:{step}:{key}:momentum"),
                                    "second_momentum_buffer": cls.tensor_record(f"{fixture_name}:state:{step}:{key}:second"),
                                }
                        return records

                    snapshots = []
                    for step in range(1, spec["steps"] + 1):
                        parameter_version = 3 if fixture_name != "tiny_fixed" and step == 4 else step
                        parameter_records = parameters(parameter_version)
                        combined = hashlib.sha256(b"".join(
                            bytes.fromhex(parameter_records[str(index)]["sha256"])
                            for index in range(count)
                        )).hexdigest()
                        snapshot = {
                            "step": step,
                            "parameters": parameter_records,
                            "state": state(step),
                            "group_scalars": [
                                {
                                    "kind": entry["kind"],
                                    "lr": 0.0 if fixture_name != "tiny_fixed" and step == 4 else 0.01 / step,
                                    "momentum": 0.85 + 0.02 * step if entry["kind"] == "muon" else None,
                                }
                                for entry in entries
                            ],
                            "owner_calls": None if variant == "raw" else step,
                            "cross_rank_parameter_sha256": [combined] * 4,
                            "cross_rank_exact": True,
                        }
                        if fixture_name == "production_width1024_fs_w62":
                            output = cls.tensor_record(f"{fixture_name}:linear:{step}")
                            snapshot["linear_cache"] = {
                                "cache_exact": True, "output_exact": True,
                                "compiled_output": output, "eager_output": copy.deepcopy(output),
                            }
                        snapshots.append(snapshot)
                    fixtures.append({
                        "fixture": fixture_name,
                        "passed": True,
                        "parameter_count": count,
                        "element_count": total,
                        "plan": plan,
                        "initial_parameters": parameters(0),
                        "snapshots": snapshots,
                    })
                variants[variant].append({
                    "schema": D.WORKER_SCHEMA,
                    "worker": "optimizer",
                    "variant": variant,
                    "rank": rank,
                    "world_size": 4,
                    "passed": True,
                    "fixtures": fixtures,
                    "compile_cache": f"/tmp/cache-{variant}",
                })
        return variants

    @staticmethod
    def write_optimizer_matrix(root, matrix):
        for variant, reports in matrix.items():
            for rank, report in enumerate(reports):
                (root / f"optimizer-{variant}-rank{rank}.json").write_text(json.dumps(report))

    @staticmethod
    def make_first_step_parameter_noop(matrix):
        for reports in matrix.values():
            for report in reports:
                fixture = report["fixtures"][0]
                snapshot = fixture["snapshots"][0]
                snapshot["parameters"] = copy.deepcopy(fixture["initial_parameters"])
                combined = hashlib.sha256(b"".join(
                    bytes.fromhex(snapshot["parameters"][str(index)]["sha256"])
                    for index in range(fixture["parameter_count"])
                )).hexdigest()
                snapshot["cross_rank_parameter_sha256"] = [combined] * 4

    @staticmethod
    def make_second_step_state_noop(matrix):
        for reports in matrix.values():
            for report in reports:
                fixture = report["fixtures"][0]
                fixture["snapshots"][1]["state"] = copy.deepcopy(fixture["snapshots"][0]["state"])

    def test_optimizer_comparator_accepts_only_complete_exact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.optimizer_matrix()
            self.write_optimizer_matrix(root, matrix)
            comparison = D.compare_optimizer_results(root)
            self.assertTrue(comparison["passed"], comparison["failures"][:3])

    def test_optimizer_comparator_rejects_adversarial_evidence(self):
        mutations = {
            "empty_snapshots": lambda m: m["direct"][0]["fixtures"][0].__setitem__("snapshots", []),
            "extra_snapshot": lambda m: m["direct"][0]["fixtures"][0]["snapshots"].append(
                copy.deepcopy(m["direct"][0]["fixtures"][0]["snapshots"][-1])
            ),
            "nonfinite_tensor": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0]["parameters"]["0"].__setitem__("finite", False),
            "zero_tensor": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0]["parameters"]["0"].update(nonzero=False, l2=0.0, max_abs=0.0),
            "empty_state": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0].__setitem__("state", {}),
            "wrong_owner_calls": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0].__setitem__("owner_calls", 999),
            "wrong_schema": lambda m: m["direct"][0].__setitem__("schema", "wrong"),
            "wrong_rank": lambda m: m["direct"][0].__setitem__("rank", 3),
            "production_count_drift": lambda m: m["direct"][0]["fixtures"][2].__setitem__("parameter_count", 61),
            "nonfinite_scalar": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0]["group_scalars"][0].__setitem__("lr", float("nan")),
            "no_op_parameter_step": self.make_first_step_parameter_noop,
            "no_op_state_step": self.make_second_step_state_noop,
            "missing_step_key": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0].pop("step"),
            "bad_cross_rank_digest": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0]["cross_rank_parameter_sha256"].__setitem__(0, "0" * 64),
            "plan_drift": lambda m: m["direct"][0]["fixtures"][0]["plan"]["entries"][0].__setitem__("owner", 1),
        }
        base = self.optimizer_matrix()
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                matrix = copy.deepcopy(base)
                mutate(matrix)
                root = Path(directory)
                self.write_optimizer_matrix(root, matrix)
                comparison = D.compare_optimizer_results(root)
                self.assertFalse(comparison["passed"], name)


if __name__ == "__main__":
    unittest.main()
