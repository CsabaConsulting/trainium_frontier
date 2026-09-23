#!/usr/bin/env python3
"""Fail-closed local checks for the c595 warmdown-0.60 implementation."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "research/submission-validation/20260924-c595-warmdown060-trn22"
R1_LABEL = "fs-20260924-c595-warmdown060-trn22-r1"
R2_LABEL = "fs-20260924-c595-warmdown060-trn22-r2"
R1_EVIDENCE_COMMIT = "b17fb2d0438e2a8fa7cea51683b88c1c209b2932"
R1_RETRIEVED = EVIDENCE / f"retrieved/{R1_LABEL}"
RETRIEVED_RUNTIME_JOB = (
    R1_RETRIEVED / "job.json"
)
SOURCE_PACKAGE = ROOT / "submissions/dense6x1024_direct_review"
CANDIDATE_PACKAGE = ROOT / "submissions/dense6x1024_direct_review_warmdown060"
SOURCE_TRAIN_SHA = "c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537"
CANDIDATE_TRAIN_SHA = "52cf99df3e241ab8399c104b7ad4b717571e0987781b6b835303a8500baff38c"
PREPARE_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
LAUNCH_SHA = "b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd"
LAUNCH = (
    "NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py "
    "--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct "
    "--nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce "
    "--seq-len 1024 --no-eval-public"
)
CONTROL_BPB = Decimal("0.993593002710418")
KEEP_DELTA = Decimal("0.0015")
REPLICATE_BELOW_DELTA = Decimal("0.0025")


def load_job():
    path = EVIDENCE / "retune_job.py"
    spec = importlib.util.spec_from_file_location("warmdown060_retune_job", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_adapter():
    path = EVIDENCE / "launch_adapter.py"
    spec = importlib.util.spec_from_file_location("warmdown060_launch_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(), parse_float=Decimal)


def assert_exact_candidate(test: unittest.TestCase, source: str, candidate: str) -> None:
    test.assertEqual(source.count("WARMDOWN_RATIO = 0.75"), 1)
    test.assertNotIn("WARMDOWN_RATIO = 0.60", source)
    test.assertEqual(candidate.count("WARMDOWN_RATIO = 0.60"), 1)
    test.assertNotIn("WARMDOWN_RATIO = 0.75", candidate)
    restored = candidate.replace("WARMDOWN_RATIO = 0.60", "WARMDOWN_RATIO = 0.75")
    test.assertEqual(restored, source, "candidate contains a change beyond WARMDOWN_RATIO")
    for invariant in (
        "MAX_TRAIN_SECONDS = 1800",
        "MATRIX_LR = 0.02",
        "WEIGHT_DECAY = 0.2",
        "default=int(os.environ.get('NEURON_COMPETITION_R1_SEED', '42'))",
    ):
        test.assertEqual(candidate.count(invariant), 1, invariant)


class CandidateDiffTests(unittest.TestCase):
    def test_source_and_candidate_hashes_and_exact_one_line_change(self) -> None:
        source_path = SOURCE_PACKAGE / "train.py"
        candidate_path = CANDIDATE_PACKAGE / "train.py"
        self.assertEqual(sha256(source_path), SOURCE_TRAIN_SHA)
        self.assertEqual(sha256(candidate_path), CANDIDATE_TRAIN_SHA)
        assert_exact_candidate(self, source_path.read_text(), candidate_path.read_text())

    def test_exact_diff_gate_rejects_second_change(self) -> None:
        source = (SOURCE_PACKAGE / "train.py").read_text()
        candidate = (CANDIDATE_PACKAGE / "train.py").read_text()
        with self.assertRaises(AssertionError):
            assert_exact_candidate(self, source, candidate.replace("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = 0.3"))

    def test_exact_diff_gate_rejects_weight_decay_instead_of_warmdown(self) -> None:
        source = (SOURCE_PACKAGE / "train.py").read_text()
        wrong = source.replace("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = 0.6")
        with self.assertRaises(AssertionError):
            assert_exact_candidate(self, source, wrong)

    def test_candidate_package_is_minimal(self) -> None:
        self.assertEqual(
            {path.name for path in CANDIDATE_PACKAGE.iterdir() if path.is_file()},
            {"train.py", "launch-command.txt", "provenance.json"},
        )
        self.assertLessEqual(
            {path.name for path in CANDIDATE_PACKAGE.iterdir() if path.is_dir()},
            {"__pycache__"},
        )


class ConfigurationTests(unittest.TestCase):
    def test_protected_files_and_exact_launch_command(self) -> None:
        self.assertEqual(sha256(ROOT / "prepare.py"), PREPARE_SHA)
        source_launch = SOURCE_PACKAGE / "launch-command.txt"
        candidate_launch = CANDIDATE_PACKAGE / "launch-command.txt"
        self.assertEqual(sha256(source_launch), LAUNCH_SHA)
        self.assertEqual(sha256(candidate_launch), LAUNCH_SHA)
        self.assertEqual(source_launch.read_text().strip(), LAUNCH)
        self.assertEqual(candidate_launch.read_text().strip(), LAUNCH)
        argv = shlex.split(LAUNCH)
        self.assertEqual(argv[0], "NEURON_LOGICAL_NC_CONFIG=2")
        self.assertEqual(argv[1:5], ["torchrun", "--standalone", "--nproc_per_node=4", "train.py"])
        self.assertIn("--no-eval-public", argv)
        self.assertNotIn("--num-steps", argv)
        self.assertNotIn("--max-train-seconds", argv)

    def test_provenance_is_exact_and_pending_hardware(self) -> None:
        provenance = load_json(CANDIDATE_PACKAGE / "provenance.json")
        self.assertEqual(provenance["derived_from"]["train_py_sha256"], SOURCE_TRAIN_SHA)
        self.assertEqual(provenance["train_sha256"], CANDIDATE_TRAIN_SHA)
        self.assertEqual(
            provenance["single_causal_change"],
            {"constant": "WARMDOWN_RATIO", "from": Decimal("0.75"), "to": Decimal("0.60")},
        )
        self.assertEqual(provenance["unchanged_control_constant"], {"constant": "WEIGHT_DECAY", "value": Decimal("0.2")})
        self.assertEqual(provenance["launch_command"], LAUNCH)
        self.assertEqual(provenance["seed"], 42)
        self.assertEqual(provenance["max_train_seconds"], 1800)
        self.assertIs(provenance["smoke"], False)
        self.assertTrue(provenance["hardware_validation"].startswith("NOT RUN"))

    def test_proposal_pins_control_metric_and_predeclared_decision(self) -> None:
        proposal = load_json(EVIDENCE / "EXPERIMENT_PROPOSAL.json")
        self.assertEqual(proposal["proposal_id"], "20260924-c595-warmdown060-trn22-r2")
        self.assertEqual(proposal["status"], "human-approved-pending-independent-verification")
        self.assertEqual(proposal["mechanism_class"], "schedule")
        self.assertIn("not a weight-decay change", proposal["single_causal_change"])
        self.assertEqual(proposal["control"]["candidate_sha256"], SOURCE_TRAIN_SHA)
        self.assertEqual(proposal["control"]["public_val_bpb"], CONTROL_BPB)
        metric = proposal["control"]["selection_metric"]
        self.assertEqual((metric["seq_len"], metric["num_tokens"], metric["num_bytes"]), (1024, 20971520, 83462186))
        self.assertEqual(proposal["candidate"]["sha256"], CANDIDATE_TRAIN_SHA)
        self.assertEqual(proposal["candidate"]["run_label"], R2_LABEL)
        self.assertEqual(
            proposal["candidate"]["remote_checkout"],
            "/home/ubuntu/trainium_freshstart_20260924_c595_warmdown060_trn22_r2",
        )
        self.assertEqual(proposal["candidate"]["runtime_job"], f"research/run-logs/{R2_LABEL}.job.json")
        self.assertEqual(proposal["candidate"]["dispatch_receipt"], f"research/run-logs/{R2_LABEL}.dispatch.json")
        self.assertEqual(proposal["candidate"]["output_directory"], f"out/{R2_LABEL}")
        self.assertEqual(proposal["candidate"]["launch_command"], LAUNCH)
        self.assertEqual(proposal["candidate"]["seed"], 42)
        self.assertEqual(proposal["candidate"]["max_train_seconds"], 1800)
        self.assertIs(proposal["candidate"]["smoke_authorized"], False)
        policy = proposal["replication_policy"]
        self.assertEqual(policy["keep_bpb_maximum"], CONTROL_BPB - KEEP_DELTA)
        self.assertEqual(policy["replicate_keep_bpb_interval"]["lower_exclusive"], CONTROL_BPB - REPLICATE_BELOW_DELTA)
        self.assertIs(proposal["gates"]["independent_verification"], False)
        self.assertIs(proposal["gates"]["training_launched"], False)
        self.assertIs(proposal["gates"]["submission_authorized"], False)
        self.assertIsNone(proposal["verdict"])
        retry = proposal["retry_provenance"]
        self.assertEqual(retry["previous_label"], R1_LABEL)
        self.assertEqual(retry["evidence_commit"], R1_EVIDENCE_COMMIT)
        self.assertIs(retry["training_started"], False)
        self.assertEqual(retry["failure_stage"], "runtime_contract_before_training")

    def test_finalization_files_are_absent_together_or_exact_child_only(self) -> None:
        auth_path = EVIDENCE / "authorization.json"
        request_path = EVIDENCE / "final-request.json"
        self.assertEqual(auth_path.exists(), request_path.exists())
        if not auth_path.exists():
            return
        job = load_job()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        core = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=ROOT, text=True).strip()
        changed = set(subprocess.check_output(["git", "diff", "--name-only", core, head], cwd=ROOT, text=True).splitlines())
        self.assertEqual(changed, {job.AUTH_REL, job.REQUEST_REL})
        auth = json.loads(auth_path.read_text())
        request = json.loads(request_path.read_text())
        request_sha = job.validate_authorization_request(auth, request, core)
        self.assertEqual(request_sha, hashlib.sha256(job.canonical_json(request)).hexdigest())

    def test_control_result_matches_pinned_full20m_metric(self) -> None:
        result = load_json(
            ROOT
            / "research/submission-validation/20260923-c595-dual-host-seed42-controls/retrieved"
            / "fs-20260924-c595-eval-recovery-trn22-r1/result.json"
        )
        self.assertEqual(result["host"], "trn22")
        self.assertEqual(result["hostname"], "ip-172-31-32-253")
        self.assertEqual(result["candidate_sha256"], SOURCE_TRAIN_SHA)
        self.assertIs(result["checkpoint_preserved"], True)
        metrics = result["secondary"]["metrics"]
        self.assertEqual(metrics["val_bpb"], CONTROL_BPB)
        self.assertEqual((metrics["num_tokens"], metrics["num_bytes"]), (20971520, 83462186))
        self.assertIs(metrics["eval_over_timeout"], False)

    def test_r1_failure_evidence_is_unchanged_from_evidence_commit(self) -> None:
        paths = [
            str(R1_RETRIEVED.relative_to(ROOT)),
            f"research/run-logs/{R1_LABEL}.dispatch.json",
        ]
        subprocess.run(
            ["git", "diff", "--quiet", R1_EVIDENCE_COMMIT, "--", *paths],
            cwd=ROOT,
            check=True,
        )
        self.assertEqual(
            sha256(R1_RETRIEVED / "operator-failure-receipt.json"),
            "b2ab112abe99fc6226428e48cd9cd645173fcd2a4e2c91fec66a49ca3bd3ab9a",
        )
        self.assertEqual(
            sha256(RETRIEVED_RUNTIME_JOB),
            "fccc470e6a44e232e5731ed00948a50810b4d80ea62db296c4be5d05e3f1ba09",
        )

    def test_r2_runtime_namespace_is_unique_and_unused(self) -> None:
        job = load_job()
        self.assertEqual(job.LABEL, R2_LABEL)
        self.assertEqual(job.RUNTIME_JOB_REL, f"research/run-logs/{R2_LABEL}.job.json")
        self.assertEqual(job.DISPATCH_REL, f"research/run-logs/{R2_LABEL}.dispatch.json")
        self.assertEqual(job.OUTPUT, ROOT / f"out/{R2_LABEL}")
        self.assertEqual(
            job.CONCRETE_REQUEST,
            Path(f"/tmp/{R2_LABEL}.request.json"),
        )
        self.assertNotEqual(job.LABEL, R1_LABEL)
        self.assertFalse(job.OUTPUT.exists())
        self.assertFalse(job.CONCRETE_REQUEST.exists())
        for suffix in ("dispatch.json", "job.json", "launch.json", "log", "preflight.json", "postflight.json", "process.json"):
            self.assertFalse((ROOT / f"research/run-logs/{R2_LABEL}.{suffix}").exists())

    def test_r2_executable_config_has_no_r1_identity(self) -> None:
        paths = [EVIDENCE / "retune_job.py", EVIDENCE / "launch_adapter.py"]
        for name in ("authorization.json", "final-request.json"):
            path = EVIDENCE / name
            if path.exists():
                paths.append(path)
        for path in paths:
            text = path.read_text()
            self.assertNotIn(R1_LABEL, text, path.name)
            self.assertNotIn(
                "/home/ubuntu/trainium_freshstart_20260924_c595_warmdown060_trn22_r1",
                text,
                path.name,
            )


class SecureEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.job = load_job()
        cls.adapter = load_adapter()

    def test_finite_receipt_requires_exact_positive_schema(self) -> None:
        good = '{"finite": true, "map_location": "cpu", "tensor_count": 62, "weights_only": true}\n'
        self.assertEqual(self.job.parse_finite_receipt(good)["tensor_count"], 62)
        bad = (
            '{"finite": true, "map_location": "cpu", "tensor_count": 0, "weights_only": true}',
            '{"finite": false, "map_location": "cpu", "tensor_count": 62, "weights_only": true}',
            '{"finite": true, "tensor_count": 62, "weights_only": true}',
            good + good,
        )
        for receipt in bad:
            with self.subTest(receipt=receipt), self.assertRaises(SystemExit):
                self.job.parse_finite_receipt(receipt)

    def test_full20m_metric_parser_rejects_nonfinite_and_count_drift(self) -> None:
        total_nats = 57480924.0
        good = {
            "val_bpb": total_nats / (83462186 * math.log(2.0)),
            "num_tokens": 20971520,
            "num_bytes": 83462186,
            "total_nats": total_nats,
            "eval_seconds": 700.0,
            "eval_timeout": 1200,
            "eval_over_timeout": False,
        }
        self.assertEqual(self.job.parse_metrics(json.dumps(good))["num_tokens"], 20971520)
        for bad in (
            {**good, "num_tokens": 20971519},
            {**good, "num_bytes": 83462185},
            {**good, "val_bpb": float("nan")},
            {**good, "eval_over_timeout": True},
            {**good, "eval_seconds": 1200.0001},
        ):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                self.job.parse_metrics(json.dumps(bad))

    def test_decimal_keep_and_replication_boundaries_are_exact(self) -> None:
        keep_boundary = self.job.decide_bpb(str(CONTROL_BPB - KEEP_DELTA))
        self.assertEqual(keep_boundary["decision"], "KEEP")
        self.assertIs(keep_boundary["replication_required"], True)
        just_below_keep = self.job.decide_bpb(str(CONTROL_BPB - KEEP_DELTA + Decimal("0.000000000000001")))
        self.assertEqual(just_below_keep["decision"], "DISCARD")
        replicate_boundary = self.job.decide_bpb(str(CONTROL_BPB - REPLICATE_BELOW_DELTA))
        self.assertEqual(replicate_boundary["decision"], "KEEP")
        self.assertIs(replicate_boundary["replication_required"], False)
        just_inside_replicate = self.job.decide_bpb(
            str(CONTROL_BPB - REPLICATE_BELOW_DELTA + Decimal("0.000000000000001"))
        )
        self.assertEqual(just_inside_replicate["decision"], "KEEP")
        self.assertIs(just_inside_replicate["replication_required"], True)
        for value in ("NaN", "Infinity", "-1", "0", "not-a-number"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.job.decide_bpb(value)

    def test_durable_artifacts_are_no_overwrite_nofollow_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, evidence = self.job.create_output(root, "unit-evidence")
            value = {"schema": "unit-v1", "finite": True}
            self.job.durable_json(evidence, "receipt.json", value)
            self.assertEqual(self.job.persisted_json(evidence, "receipt.json", value, "unit-v1"), value)
            members = self.job.inspect_artifacts(evidence, {"receipt.json"})
            self.job.revalidate_members(evidence, members)
            with self.assertRaises(FileExistsError):
                self.job.durable_json(evidence, "receipt.json", value)
            (evidence / "receipt.json").write_bytes(b"mutated")
            with self.assertRaises(SystemExit):
                self.job.revalidate_members(evidence, members)
            outside = root / "outside"
            outside.write_bytes(b"unsafe")
            os.symlink(outside, evidence / "link")
            with self.assertRaises(SystemExit):
                self.job.artifact_bytes(evidence, "link")
            with self.assertRaises(SystemExit):
                self.job.inspect_artifacts(evidence, {"receipt.json"})

    def test_driver_declares_sealed_success_lifecycle(self) -> None:
        source = (EVIDENCE / "retune_job.py").read_text()
        for required in (
            "os.O_EXCL", "os.O_NOFOLLOW", "os.fsync", "weights_only=True",
            "artifact-manifest.json", "ready-to-retrieve.json", "checkpoint_preserved",
            "process_group_quiescent", "BASE_ARTIFACTS",
        ):
            self.assertIn(required, source)
        self.assertNotIn(".write_text(", source)

    def fixture(self):
        core = "1" * 40
        final = "2" * 40
        source_sha = {path: "a" * 64 for path in self.job.CORE_FINGERPRINT_PATHS}
        request = {
            "schema": "c595-warmdown060-final-request-v1",
            "host": self.job.HOST,
            "expected_hostname": self.job.HOSTNAME,
            "remote": self.job.REMOTE,
            "label": self.job.LABEL,
            "purpose": "exact single-factor warmdown full run and fresh full20m evaluation",
            "verified_core_commit": core,
            "authorization_path": self.job.AUTH_REL,
            "request_path": self.job.REQUEST_REL,
            "source_sha256": source_sha,
            "environment": self.job.MINIMAL_ENVIRONMENT,
            "command": [
                "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "4200s", self.job.PYTHON,
                self.job.DRIVER_REL, "--authorization", self.job.AUTH_REL, "--request", self.job.REQUEST_REL,
                "--runtime-job", self.job.RUNTIME_JOB_REL,
            ],
            "runtime_job_schema": "c595-warmdown060-runtime-job-v1",
            "runner_owns_manifest": False,
            "non_overwrite": True,
        }
        request_sha = hashlib.sha256(self.job.canonical_json(request)).hexdigest()
        auth = {
            "schema": "c595-warmdown060-authorization-v2", "arm": "candidate",
            "host": self.job.HOST, "expected_hostname": self.job.HOSTNAME, "label": self.job.LABEL,
            "package": self.job.PACKAGE_REL, "base_train_sha256": self.job.SOURCE_SHA,
            "train_sha256": self.job.CANDIDATE_SHA,
            "single_causal_change": "WARMDOWN_RATIO 0.75 -> 0.60", "unchanged_weight_decay": 0.2,
            "seed": 42, "max_train_seconds": 1800, "smoke_authorized": False,
            "training_authorized": True, "evaluation_authorized": True,
            "checkpoint_prune_authorized": False, "verified_core_commit": core,
            "request_path": self.job.REQUEST_REL, "request_sha256": request_sha,
            "request_content": request, "state": "human_approved_pending_operator",
        }
        authorization_sha = "d" * 64
        concrete = self.job.build_concrete_request(request, authorization_sha, request_sha, final)
        runtime = {
            **concrete,
            "request_file_sha256": hashlib.sha256(self.job.canonical_json(concrete)).hexdigest(),
            "started_at": "2026-09-24T12:00:00+00:00",
            "hostname": self.job.HOSTNAME,
            "git_commit": final,
            "source_dirty": False,
            "preflight_free_bytes": 9 * 1024**3,
            "source_sha256": concrete["expected_sha256"],
        }
        runtime_bytes = self.job.standard_launcher_job_bytes(runtime, concrete)
        return core, final, request, request_sha, auth, authorization_sha, concrete, runtime, runtime_bytes

    def test_strict_minimal_environment_rejects_hostile_inheritance(self) -> None:
        clean = dict(self.job.MINIMAL_ENVIRONMENT)
        self.assertEqual(self.job.build_subprocess_environment(clean), self.job.MINIMAL_ENVIRONMENT)
        for key, value in (
            ("LD_PRELOAD", "/tmp/evil.so"),
            ("XLA_FLAGS", "--xla_force_host_platform_device_count=99"),
            ("NEURON_RT_VISIBLE_CORES", "0"),
            ("NEURON_STOCHASTIC_ROUNDING_EN", "1"),
            ("TORCH_LOGS", "+dynamo"),
        ):
            with self.subTest(key=key), self.assertRaises(SystemExit):
                self.job.build_subprocess_environment({**clean, key: value})
        inherited_benign = {**clean, "TERM": "xterm", "SHELL": "/bin/bash"}
        self.assertEqual(self.job.build_subprocess_environment(inherited_benign), self.job.MINIMAL_ENVIRONMENT)

    def test_request_auth_and_runtime_contract_reject_forgery(self) -> None:
        core, final, request, request_sha, auth, authorization_sha, _concrete, runtime, runtime_bytes = self.fixture()
        self.assertEqual(self.job.validate_authorization_request(auth, request, core), request_sha)
        accepted, _ = self.job.validate_runtime_job_bytes(
            runtime_bytes, request, request_sha, authorization_sha, core, final
        )
        self.assertEqual(accepted, runtime)
        mutations = []
        for field, value in (
            ("finalization_head", "3" * 40),
            ("verified_core_commit", "4" * 40),
            ("request_file_sha256", "b" * 64),
            ("command", ["python", "evil.py"]),
            ("environment", {**request["environment"], "TORCH_LOGS": "+all"}),
            ("source_sha256", {**runtime["source_sha256"], "prepare.py": "c" * 64}),
            ("expected_sha256", {**runtime["expected_sha256"], "prepare.py": "c" * 64}),
            ("git_commit", "6" * 40),
        ):
            mutations.append({**runtime, field: value})
        for forged in mutations:
            with self.subTest(field=next(key for key in forged if forged[key] != runtime.get(key))), self.assertRaises(SystemExit):
                self.job.validate_runtime_contract(
                    forged, request, request_sha, authorization_sha, core, final
                )

    def test_alternate_child_and_request_template_mutation_fail(self) -> None:
        core, final, request, request_sha, auth, authorization_sha, _concrete, runtime, _runtime_bytes = self.fixture()
        with self.assertRaises(SystemExit):
            self.job.validate_runtime_contract(
                runtime, request, request_sha, authorization_sha, core, "5" * 40
            )
        mutated = {**request, "purpose": "mutated same-core request"}
        with self.assertRaises(SystemExit):
            self.job.validate_authorization_request(auth, mutated, core)

    def test_adapter_materializes_standard_launcher_request_without_overwrite(self) -> None:
        _core, final, request, request_sha, _auth, authorization_sha, concrete, _runtime, _runtime_bytes = self.fixture()
        self.adapter.validate_standard_request(concrete, self.job)
        self.assertEqual(concrete["expected_commit"], final)
        self.assertEqual(concrete["template_sha256"], request_sha)
        self.assertEqual(concrete["authorization_sha256"], authorization_sha)
        self.assertEqual(concrete["fingerprint_paths"], sorted(concrete["expected_sha256"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "concrete.json"
            expected_sha = hashlib.sha256(self.job.canonical_json(concrete)).hexdigest()
            self.assertEqual(self.adapter.durable_concrete(path, concrete, self.job), expected_sha)
            self.assertEqual(json.loads(path.read_bytes()), concrete)
            with self.assertRaises(FileExistsError):
                self.adapter.durable_concrete(path, concrete, self.job)

    def test_adapter_rejects_generated_request_field_attacks(self) -> None:
        _core, final, request, request_sha, _auth, authorization_sha, concrete, _runtime, _runtime_bytes = self.fixture()
        attacks = (
            {**concrete, "expected_commit": "7" * 40},
            {**concrete, "command": ["python", "evil.py"]},
            {**concrete, "environment": {**concrete["environment"], "LD_PRELOAD": "/tmp/evil.so"}},
            {**concrete, "fingerprint_paths": concrete["fingerprint_paths"][:-1]},
        )
        for value in attacks:
            with self.assertRaises(SystemExit):
                self.adapter.validate_standard_request(value, self.job)
        with self.assertRaises(SystemExit):
            self.job.build_concrete_request(request, authorization_sha, request_sha, "not-a-head")
        self.assertEqual(concrete["expected_commit"], final)

    def test_adapter_pins_r2_concrete_path_for_non_dry_run(self) -> None:
        source = (EVIDENCE / "launch_adapter.py").read_text()
        self.assertIn("concrete_path != driver.CONCRETE_REQUEST", source)
        self.assertIn("non-dry-run concrete request path mismatch", source)
        self.assertEqual(self.job.CONCRETE_REQUEST, Path(f"/tmp/{R2_LABEL}.request.json"))

    def validate_retrieved_serialization(self, data: bytes):
        runtime = json.loads(data)
        concrete = {
            key: value for key, value in runtime.items()
            if key not in self.job.RUNTIME_DYNAMIC_KEYS
        }
        if self.job.standard_launcher_job_bytes(runtime, concrete) != data:
            raise SystemExit("historical launcher serialization mismatch")
        return runtime, runtime["request_file_sha256"]

    def test_retrieved_standard_launcher_fixture_pins_r1_serialization(self) -> None:
        data = RETRIEVED_RUNTIME_JOB.read_bytes()
        self.assertEqual(len(data), 12220)
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            "fccc470e6a44e232e5731ed00948a50810b4d80ea62db296c4be5d05e3f1ba09",
        )
        runtime, concrete_sha = self.validate_retrieved_serialization(data)
        self.assertEqual(
            concrete_sha,
            "c6d1ab20a1b5f13c98341aeccd1407230c0b5c569c52704cc505f00fa99fd281",
        )
        self.assertEqual(runtime["label"], R1_LABEL)

    def test_retrieved_runtime_rejects_whitespace_mutation(self) -> None:
        with self.assertRaises(SystemExit):
            self.validate_retrieved_serialization(RETRIEVED_RUNTIME_JOB.read_bytes() + b"\n")

    def test_retrieved_runtime_rejects_key_order_mutation(self) -> None:
        runtime = json.loads(RETRIEVED_RUNTIME_JOB.read_bytes())
        reordered = json.dumps(runtime, indent=2, sort_keys=True).encode()
        self.assertNotEqual(reordered, RETRIEVED_RUNTIME_JOB.read_bytes())
        with self.assertRaises(SystemExit):
            self.validate_retrieved_serialization(reordered)

    def test_current_runtime_rejects_content_mutation(self) -> None:
        core, final, request, request_sha, _auth, authorization_sha, _concrete, runtime, _data = self.fixture()
        runtime["hostname"] = "ip-172-31-32-254"
        mutated = json.dumps(runtime, indent=2).encode()
        with self.assertRaises(SystemExit):
            self.job.validate_runtime_job_bytes(
                mutated, request, request_sha, authorization_sha, core, final
            )

    def test_finalized_adapter_cli_dry_run_without_ssh(self) -> None:
        if not (EVIDENCE / "authorization.json").exists():
            return
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as directory:
            concrete = Path(directory).resolve() / "concrete-request.json"
            done = subprocess.run(
                [
                    "python3", str(EVIDENCE / "launch_adapter.py"),
                    "--concrete-request", str(concrete), "--dry-run",
                    "--dry-run-origin-head", head,
                ],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            )
            receipt = json.loads(done.stdout)
            self.assertIs(receipt["dry_run"], True)
            self.assertEqual(receipt["finalization_head"], head)
            self.assertEqual(receipt["concrete_request_sha256"], sha256(concrete))
            self.adapter.validate_standard_request(json.loads(concrete.read_bytes()), self.job)


if __name__ == "__main__":
    unittest.main(verbosity=2)
