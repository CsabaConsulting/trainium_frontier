#!/usr/bin/env python3
"""Static/adversarial Track A checks; no Neuron work or network access."""

import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import artifact_io as aio
import contracts as c
import launch_adapter as adapter


class Reader:
    def __init__(self, values):
        self.values = values

    def read(self, name):
        if name not in self.values:
            raise SystemExit("fixture member missing")
        return self.values[name]


def metric(bpb=0.994):
    return {
        "val_bpb": float(bpb), "num_tokens": c.SCREEN_TOKENS, "num_bytes": c.SCREEN_BYTES,
        "total_nats": float(bpb) * c.SCREEN_BYTES * math.log(2.0), "eval_seconds": 12.5,
        "eval_timeout": 1200, "eval_over_timeout": False,
    }


def control_fixture(mutator=None):
    checkpoint_sha = "c" * 64
    metrics = metric()
    finite = {
        "schema": "track-a-finite-v1", "finite": True, "weights_only": True,
        "map_location": "cpu", "tensor_count": 62, "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": 123456, "argv": ["python", "scan"], "wall_seconds": 1.0,
    }
    result = {
        "schema": "track-a-result-v1", "arm": "control-trn21", "role": "control",
        "label": c.ARMS["control-trn21"]["label"], "host": "trn21",
        "hostname": c.ARMS["control-trn21"]["hostname"], "source_commit": c.SOURCE_COMMIT,
        "source_path": c.SOURCE_PATH, "source_sha256": c.SOURCE_SHA, "prepare_sha256": c.PREPARE_SHA,
        "launch_command": c.ARMS["control-trn21"]["launch"], "seed": 42, "world_size": 4,
        "lnc": 2, "max_train_seconds": 1800, "smoke": False,
        "checkpoint_path": "/tmp/control/final.pt", "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": 123456, "checkpoint_preserved": True,
        "checkpoint_prune_authorized": False, "cache_cleanup_authorized": False,
        "finite_scan": finite, "screen2m": metrics, "screening_only": True,
        "keep_authorized": False, "adoption_authorized": False, "a1_decision": None,
        "ready_to_retrieve": False,
    }
    receipt = {"schema": "track-a-screen2m-metrics-v1", "argv": ["eval"], "metrics": metrics, "wall_seconds": 12.6}
    values = {name: (name + "\n").encode() for name in c.ARCHIVE_MEMBERS}
    if mutator:
        mutator(result, receipt, values)
    values["result.json"] = c.canonical_json(result)
    values["screen2m.metrics.json"] = c.canonical_json(receipt)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in sorted(c.ARCHIVE_MEMBERS):
            data = values[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    values["evidence.tar.gz"] = buffer.getvalue()
    members = {name: aio.identity(data) for name, data in sorted(values.items())}
    manifest = {"schema": "track-a-artifact-manifest-v1", "label": result["label"], "members": members}
    manifest_bytes = c.canonical_json(manifest)
    completion = {
        "schema": "track-a-completion-v1", "arm": "control-trn21", "role": "control",
        "label": result["label"], "host": "trn21", "hostname": c.ARMS["control-trn21"]["hostname"],
        "finalization_head": "a" * 40, "verified_core_commit": "b" * 40,
        "authorization_sha256": "1" * 64, "request_sha256": "2" * 64,
        "runtime_job_sha256": "3" * 64, "source_sha256": c.SOURCE_SHA,
        "prepare_sha256": c.PREPARE_SHA, "artifact_manifest_sha256": aio.digest(manifest_bytes),
        "artifact_members": members, "archive_sha256": aio.digest(values["evidence.tar.gz"]),
        "archive_bytes": len(values["evidence.tar.gz"]), "checkpoint_path": result["checkpoint_path"],
        "checkpoint_sha256": checkpoint_sha, "checkpoint_bytes": 123456,
        "checkpoint_preserved": True, "checkpoint_prune_authorized": False,
        "cache_cleanup_authorized": False, "screen2m": result["screen2m"], "ready_to_retrieve": True,
    }
    values["artifact-manifest.json"] = manifest_bytes
    values["ready-to-retrieve.json"] = c.canonical_json(completion)
    identity = {
        "archive_commit": "d" * 40, "directory": "research/evidence/control/artifacts",
        "completion_sha256": aio.digest(values["ready-to-retrieve.json"]),
        "manifest_sha256": aio.digest(values["artifact-manifest.json"]),
        "result_sha256": aio.digest(values["result.json"]),
        "metrics_sha256": aio.digest(values["screen2m.metrics.json"]),
    }
    return Reader(values), identity


def request_fixture(arm, core="a" * 40):
    rel = str(HERE.relative_to(ROOT))
    source_map = {
        c.SOURCE_PATH: c.SOURCE_SHA,
        "prepare.py": c.PREPARE_SHA,
        "README.md": c.RULES_SHA,
        f"{rel}/track_a_job.py": "1" * 64,
        f"{rel}/contracts.py": "2" * 64,
        f"{rel}/artifact_io.py": "3" * 64,
        f"{rel}/launch_adapter.py": "4" * 64,
        f"{rel}/EXPERIMENT_PROPOSAL.json": "5" * 64,
        "research/fresh_start_launch.py": "6" * 64,
        "research/fresh_start_host.py": "7" * 64,
        "research/fresh_start_guard.py": "8" * 64,
        "research/fresh_start_backup.py": "9" * 64,
        "research/fresh_start_sync.py": "a" * 64,
        "research/run_experiment.py": "b" * 64,
    }
    spec = c.ARMS[arm]
    return {
        "schema": "track-a-request-template-v1", "arm": arm, "host": spec["host"],
        "expected_hostname": spec["hostname"], "remote": spec["remote"],
        "origin": "https://github.com/CsabaConsulting/trainium_frontier.git", "label": spec["label"],
        "purpose": f"authorized {spec['label']} exact arm", "min_free_bytes": c.MIN_FREE_BYTES,
        "environment": c.ENVIRONMENT, "command": c.expected_command(arm),
        "source_sha256": source_map, "request_path": f"{rel}/{arm}-request.json",
        "authorization_path": f"{rel}/{arm}-authorization.json", "verified_core_commit": core,
        "runner_owns_manifest": False, "non_overwrite": True,
    }


def authorization_fixture(arm, request, core="a" * 40):
    spec = c.ARMS[arm]
    control_evidence = None
    state = "human_approved_pending_operator"
    if spec["role"] == "a1":
        state = "human_approved_control_bound_pending_operator"
        control_evidence = {
            "archive_commit": "d" * 40,
            "directory": "research/evidence/control/artifacts",
            "completion_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "result_sha256": "3" * 64,
            "metrics_sha256": "4" * 64,
        }
    return {
        "schema": "track-a-authorization-v1", "state": state, "arm": arm,
        "role": spec["role"], "host": spec["host"], "expected_hostname": spec["hostname"],
        "label": spec["label"], "source_commit": c.SOURCE_COMMIT, "source_path": c.SOURCE_PATH,
        "source_sha256": c.SOURCE_SHA, "prepare_sha256": c.PREPARE_SHA, "seed": 42,
        "max_train_seconds": 1800, "world_size": 4, "lnc": 2, "smoke_authorized": False,
        "screen_tokens": c.SCREEN_TOKENS, "screen_bytes": c.SCREEN_BYTES,
        "checkpoint_prune_authorized": False, "cache_cleanup_authorized": False,
        "verified_core_commit": core, "request_path": request["request_path"],
        "request_sha256": c.digest(c.canonical_json(request)), "request_content": request,
        "control_evidence": control_evidence,
    }


class SourceAndCommandTests(unittest.TestCase):
    def test_introduction_object_and_working_source_exact(self):
        blob = subprocess.check_output(["git", "show", f"{c.SOURCE_COMMIT}:{c.SOURCE_PATH}"], cwd=ROOT)
        self.assertEqual(hashlib.sha256(blob).hexdigest(), c.SOURCE_SHA)
        self.assertEqual((ROOT / c.SOURCE_PATH).read_bytes(), blob)
        self.assertEqual(hashlib.sha256((ROOT / "prepare.py").read_bytes()).hexdigest(), c.PREPARE_SHA)
        self.assertEqual(hashlib.sha256((ROOT / "README.md").read_bytes()).hexdigest(), c.RULES_SHA)
        parent_blob = subprocess.check_output(
            ["git", "show", f"{c.SOURCE_COMMIT}^:{c.SOURCE_PATH}"], cwd=ROOT
        )
        self.assertNotEqual(hashlib.sha256(parent_blob).hexdigest(), c.SOURCE_SHA)

    def test_control_commands_exact_and_only_host_identity_differs(self):
        expected = c.PREFIX + c.BASE_TAIL
        self.assertEqual(c.ARMS["control-trn21"]["launch"], expected)
        self.assertEqual(c.ARMS["control-trn22"]["launch"], expected)
        self.assertEqual(c.train_argv("control-trn21"), c.train_argv("control-trn22"))

    def test_a1_replaces_only_ngram_subsystem_configuration(self):
        control = c.train_argv("control-trn21")
        a1 = c.train_argv("a1-trn21")
        self.assertEqual(control[:control.index("--no-ngram-ve")], a1[:a1.index("--ngram-ve")])
        control_remainder = control[control.index("--no-ngram-ve") + 1:]
        end = a1.index("--no-qk-shift")
        self.assertEqual(a1[end:], control_remainder)
        self.assertEqual(a1[a1.index("--ngram-ve"):end], [
            "--ngram-ve", "--ngram-ve-layers=-1", "--ngram-ve-table-mult", "64",
            "--ngram-ve-dim", "128", "--ngram-ve-trigram", "--ngram-ve-opt", "rmsprop",
            "--ngram-ve-flat-lr", "--ngram-ve-bf16", "--ngram-ve-clip-exclude",
        ])

    def test_proposal_has_no_candidate_edit_or_keep_authority(self):
        proposal = json.loads((HERE / "EXPERIMENT_PROPOSAL.json").read_text())
        self.assertEqual(proposal["immutable_source"]["sha256"], c.SOURCE_SHA)
        self.assertFalse(proposal["immutable_source"]["modified"])
        self.assertEqual(proposal["candidate"]["files_allowed"], [])
        self.assertIn("never KEEP", proposal["screening"]["selection_authority"])


class DecisionDiskMetricTests(unittest.TestCase):
    def test_a1_exact_nomination_boundary_and_no_keep(self):
        result = c.a1_decision("1.000", "0.999")
        self.assertEqual(result["decision"], "NOMINATE_OTHER_HOST_FULL20M_REPLICATE")
        self.assertFalse(result["keep_authorized"])
        self.assertFalse(result["adoption_authorized"])
        self.assertTrue(result["other_host_full20m_replicate_required"])

    def test_a1_below_boundary_discards_and_nonfinite_fails(self):
        self.assertEqual(c.a1_decision("1.000", "0.9990000001")["decision"], "DISCARD")
        for bad in ("NaN", "Infinity", "0"):
            with self.assertRaises(SystemExit):
                c.a1_decision("1.0", bad)

    def test_disk_boundaries(self):
        self.assertEqual(c.disk_disposition(c.HARD_ABORT_BYTES - 1), "HARD_ABORT")
        self.assertEqual(c.disk_disposition(c.HARD_ABORT_BYTES), "BLOCKED")
        self.assertEqual(c.disk_disposition(c.MIN_FREE_BYTES - 1), "BLOCKED")
        self.assertEqual(c.disk_disposition(c.MIN_FREE_BYTES), "ALLOWED")

    def test_metric_exact_counts_and_math(self):
        self.assertEqual(c.validate_metric(metric())["num_bytes"], c.SCREEN_BYTES)
        for key, bad in (("num_tokens", 1), ("num_bytes", 1), ("val_bpb", float("nan"))):
            value = metric()
            value[key] = bad
            with self.assertRaises(SystemExit):
                c.validate_metric(value)


class EvidenceTests(unittest.TestCase):
    def test_exact_control_archive_passes(self):
        reader, identity = control_fixture()
        value = c.validate_control_evidence(reader, identity)
        self.assertEqual(value["label"], c.ARMS["control-trn21"]["label"])
        self.assertEqual(value["screen2m_val_bpb"], 0.994)

    def test_member_append_missing_and_digest_mutation_fail(self):
        reader, identity = control_fixture()
        reader.values["training.log"] += b"late"
        with self.assertRaises(SystemExit):
            c.validate_control_evidence(reader, identity)
        reader, identity = control_fixture()
        del reader.values["contract-check.log"]
        with self.assertRaises(SystemExit):
            c.validate_control_evidence(reader, identity)
        reader, identity = control_fixture()
        identity["result_sha256"] = "0" * 64
        with self.assertRaises(SystemExit):
            c.validate_control_evidence(reader, identity)

    def test_schema_host_metric_and_checkpoint_mutations_fail(self):
        reader, identity = control_fixture(lambda result, _receipt, _values: result.__setitem__("host", "trn22"))
        with self.assertRaises(SystemExit):
            c.validate_control_evidence(reader, identity)
        reader, identity = control_fixture(lambda result, _receipt, _values: result.__setitem__("extra", True))
        with self.assertRaises(SystemExit):
            c.validate_control_evidence(reader, identity)
        reader, identity = control_fixture(lambda result, receipt, _values: receipt["metrics"].__setitem__("num_bytes", 1))
        with self.assertRaises(SystemExit):
            c.validate_control_evidence(reader, identity)
        reader, identity = control_fixture(lambda result, _receipt, _values: result.__setitem__("checkpoint_preserved", False))
        with self.assertRaises(SystemExit):
            c.validate_control_evidence(reader, identity)

    def test_no_follow_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").write_bytes(b"x")
            (root / "link").symlink_to(root / "real")
            with self.assertRaises(SystemExit):
                aio.regular_bytes(root / "link")


class RequestRuntimeArtifactTests(unittest.TestCase):
    def test_request_exact_and_mutations_fail(self):
        request = request_fixture("control-trn21")
        self.assertRegex(c.validate_request(request, "control-trn21", "a" * 40), r"^[0-9a-f]{64}$")
        for attack in ("command", "environment", "remote", "min_free_bytes", "source_sha256"):
            value = copy.deepcopy(request)
            if attack == "command":
                value[attack][-1] = "0.02"
            elif attack == "environment":
                value[attack]["NEURON_CC_FLAGS"] = "--optlevel=2"
            elif attack == "remote":
                value[attack] += "-other"
            elif attack == "min_free_bytes":
                value[attack] -= 1
            else:
                value[attack][c.SOURCE_PATH] = "f" * 64
            with self.assertRaises(SystemExit):
                c.validate_request(value, "control-trn21", "a" * 40)

    def test_authorization_exact_and_mutations_fail(self):
        for arm in c.ARMS:
            request = request_fixture(arm)
            auth = authorization_fixture(arm, request)
            self.assertEqual(c.validate_authorization(auth, request, arm, "a" * 40), auth["request_sha256"])
            for key, bad in (
                ("smoke_authorized", True), ("checkpoint_prune_authorized", True),
                ("cache_cleanup_authorized", True), ("source_sha256", "e" * 64),
                ("max_train_seconds", 1799),
            ):
                attacked = copy.deepcopy(auth)
                attacked[key] = bad
                with self.assertRaises(SystemExit):
                    c.validate_authorization(attacked, request, arm, "a" * 40)
        request = request_fixture("a1-trn21")
        pending = authorization_fixture("a1-trn21", request)
        pending["state"] = "human_approved_pending_control_archive"
        with self.assertRaises(SystemExit):
            c.validate_authorization(pending, request, "a1-trn21", "a" * 40)

    def test_concrete_schema_command_and_floor_are_exact(self):
        arm = "control-trn21"
        request = request_fixture(arm)
        concrete = c.build_concrete(request, "b" * 64, c.digest(c.canonical_json(request)), "c" * 40)
        adapter.validate_concrete(concrete, arm)
        self.assertEqual(concrete["command"][:5], [
            "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "4200s",
        ])
        for key, bad in (("min_free_bytes", c.MIN_FREE_BYTES - 1), ("label", "wrong")):
            attacked = copy.deepcopy(concrete)
            attacked[key] = bad
            with self.assertRaises(SystemExit):
                adapter.validate_concrete(attacked, arm)
        attacked = copy.deepcopy(concrete)
        attacked["command"][-1] = "changed"
        with self.assertRaises(SystemExit):
            adapter.validate_concrete(attacked, arm)

    def test_runtime_exact_and_adversarial_fields(self):
        request = request_fixture("control-trn21")
        concrete = c.build_concrete(request, "b" * 64, c.digest(c.canonical_json(request)), "c" * 40)
        runtime = {**copy.deepcopy(concrete), "started_at": "now", "hostname": c.ARMS["control-trn21"]["hostname"],
                   "git_commit": "c" * 40, "source_dirty": False, "preflight_free_bytes": c.MIN_FREE_BYTES,
                   "source_sha256": copy.deepcopy(request["source_sha256"])}
        raw = json.dumps(runtime, indent=2).encode()
        self.assertEqual(c.validate_runtime(runtime, concrete, raw), hashlib.sha256(raw).hexdigest())
        for key, bad in (("hostname", "wrong"), ("source_dirty", True), ("preflight_free_bytes", c.MIN_FREE_BYTES - 1)):
            attacked = copy.deepcopy(runtime)
            attacked[key] = bad
            with self.assertRaises(SystemExit):
                c.validate_runtime(attacked, concrete, json.dumps(attacked).encode())
        attacked = copy.deepcopy(runtime)
        attacked["source_sha256"][c.SOURCE_PATH] = "e" * 64
        with self.assertRaises(SystemExit):
            c.validate_runtime(attacked, concrete, json.dumps(attacked).encode())

    def test_store_no_overwrite_and_symlink_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            store = aio.Store(Path(directory), "label")
            store.create()
            store.write("x", b"one")
            with self.assertRaises(FileExistsError):
                store.write("x", b"two")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").mkdir()
            (root / "out").symlink_to(root / "real")
            with self.assertRaises(OSError):
                aio.Store(root, "label").create()


if __name__ == "__main__":
    unittest.main()
