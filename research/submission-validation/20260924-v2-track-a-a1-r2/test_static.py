#!/usr/bin/env python3
"""CPU/static fail-closed tests for the A1 r2 infrastructure retry."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import artifact_io as aio
import contracts as c
import launch_adapter as adapter
import stage_contract as stage


CONTROL_IDENTITY = {
    "archive_commit": "b0b23cbc2e6d16e8bc8d88be9cb042140494206d",
    "directory": "research/submission-validation/20260924-v2-track-a-35c9-controls-a1/operator-evidence/fs-20260924-v2-35c9-lr015-control-trn21-r1/runtime/out/fs-20260924-v2-35c9-lr015-control-trn21-r1",
    "completion_sha256": "41a27cd7c7a0747a302c0c5e034f9fe6a302d0650efbe91624291ef284d3aacb",
    "manifest_sha256": "d067f57d83301c27f35a4aea63d508eb230dc0ef9d656839d71e08b2c48af625",
    "result_sha256": "4020d2a9a82ca4e8c05bb158ae0881da2464e3ecad55c9dfef6c8847973645dc",
    "metrics_sha256": "c174d2807efdd401fe336b5bf835a7288acb15d987b630f907dc7bb609d9472b",
}
R1_FAILURE_IDENTITY = {
    "archive_commit": "65d70f5956ad2c68c37560604c8f322cbbecf338",
    "directory": "research/submission-validation/20260924-v2-track-a-35c9-controls-a1/operator-evidence/fs-20260924-v2-a1-lastlayer-ngram64d128-trn21-r1",
    "audit_sha256": "825753a9c627d257af2573844af7c227274a33653420b299f888b600e20f50ee",
    "dispatch_sha256": "0ed5f89cb0ec4a011ed6d1951e120ffa11e6fd5908263fb154a581d641f29f37",
    "concrete_request_sha256": "b171e72e15cfd8b7f777fbc0b0e7f82640453307c4bb2814733d36ee0843c887",
    "runtime_job_sha256": "6da04a89b8e3c665e5ef66ed30893012657971d3d6e60ff8a602b88570fcec38",
    "launch_sha256": "6da04a89b8e3c665e5ef66ed30893012657971d3d6e60ff8a602b88570fcec38",
    "log_sha256": "ad10343d34b54800acd107c960a7da8f798b10cb4ff9a69aca66d342131b93e2",
    "postflight_sha256": "f637a87c237ec33a447063f87ba522ec47f6ec49bf11d1f8990549c9fe7a17b4",
    "preflight_sha256": "7d1b5ed05d9b087a1e5cffb5a7fe78eb2c7da5b1182bad8352316e00b0783d84",
    "process_sha256": "70c70feb5466cae0e4eca52067d56cce352e65f5c49f11fa276bd6a8fd0c5cd1",
}


class MutatedFailureReader:
    def __init__(self, mutate_name=None, mutate=None, extra=None):
        reader = c.FailureGitReader(R1_FAILURE_IDENTITY)
        self.blobs = {name: reader.read(name) for name in reader.names()}
        if mutate_name:
            value = json.loads(self.blobs[mutate_name])
            mutate(value)
            self.blobs[mutate_name] = c.canonical_json(value)
        if extra:
            self.blobs[extra] = b"forbidden\n"

    def names(self):
        return list(self.blobs)

    def read(self, name):
        return self.blobs[name]


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


class SourceIdentityTests(unittest.TestCase):
    def test_source_prepare_and_parent_are_exact(self):
        source = subprocess.check_output(["git", "show", f"{c.SOURCE_COMMIT}:{c.SOURCE_PATH}"], cwd=ROOT)
        self.assertEqual(hashlib.sha256(source).hexdigest(), c.SOURCE_SHA)
        self.assertEqual((ROOT / c.SOURCE_PATH).read_bytes(), source)
        self.assertEqual(hashlib.sha256((ROOT / "prepare.py").read_bytes()).hexdigest(), c.PREPARE_SHA)
        self.assertEqual(git(ROOT, "merge-base", "c873b603e767bd44177f64b12335aa85103d925b", "HEAD"),
                         "c873b603e767bd44177f64b12335aa85103d925b")

    def test_r2_is_unique_and_science_is_exactly_r1(self):
        spec = c.ARMS[c.RETRY_ARM]
        self.assertEqual(spec["label"], "fs-20260924-v2-a1-lastlayer-ngram64d128-trn21-r2")
        self.assertEqual(spec["remote"],
                         "/home/ubuntu/trainium_freshstart_20260924_v2_a1_lastlayer_ngram64d128_trn21_r2")
        self.assertNotIn("trn21-r1", spec["label"])
        parent_dir = ROOT / "research/submission-validation/20260924-v2-track-a-35c9-controls-a1"
        parent = json.loads((parent_dir / "a1-trn21-request.json").read_text())
        module_spec = importlib.util.spec_from_file_location("parent_track_a_contracts", parent_dir / "contracts.py")
        parent_contracts = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(parent_contracts)
        self.assertEqual(spec["launch"], parent_contracts.ARMS["a1-trn21"]["launch"])
        self.assertEqual(spec["launch"], c.PREFIX + c.A1_TAIL)
        self.assertIn("--matrix-lr 0.015", spec["launch"])
        self.assertEqual(parent["environment"], c.ENVIRONMENT)

    def test_authenticated_control_is_exact(self):
        result = c.validate_control_evidence(c.GitReader(CONTROL_IDENTITY), CONTROL_IDENTITY)
        self.assertEqual(result["screen2m_val_bpb"], 0.9866973734526626)
        self.assertEqual(result["checkpoint_sha256"],
                         "aaa4b7c0ff58c56df564954c21f8abdd786930bfaa1cc447f6fb8e5b0da83535")

    def test_r1_failure_evidence_is_exact_fail_before_training(self):
        result = c.validate_r1_failure_evidence(c.FailureGitReader(R1_FAILURE_IDENTITY),
                                                R1_FAILURE_IDENTITY)
        self.assertEqual(result["classification"], "infrastructure_fail_before_training_detached_head")
        self.assertFalse(result["training_started"])
        self.assertTrue(result["quiescent"])

    def test_proposal_records_bound_r1_failure_and_no_science_result(self):
        proposal = json.loads((HERE / "EXPERIMENT_PROPOSAL.json").read_text())
        self.assertEqual(proposal["r1_failure"]["evidence_identity"], R1_FAILURE_IDENTITY)
        self.assertIsNone(proposal["r1_failure"]["science_result"])
        self.assertTrue(proposal["gates"]["r1_failure_evidence_bound"])
        self.assertFalse(proposal["staging_contract"]["detached_head_allowed"])
        self.assertEqual(proposal["candidate"]["files_allowed"], [])

    def test_r1_failure_mutations_and_unexpected_output_reject(self):
        cases = [
            ("operator-failure-audit.json", lambda value: value["staging"].update(checkout_was_detached=False)),
            ("operator-failure-audit.json", lambda value: value["failure"].update(training_started=True)),
            (f"runtime/run-logs/{c.R1_LABEL}.postflight.json",
             lambda value: value["cleanup"].update(remaining=[{"pid": 9}])),
            (f"runtime/run-logs/{c.R1_LABEL}.postflight.json",
             lambda value: value.update(command_returncode=0)),
        ]
        for name, mutation in cases:
            with self.subTest(name=name), self.assertRaises(SystemExit):
                c.validate_r1_failure_evidence(MutatedFailureReader(name, mutation), R1_FAILURE_IDENTITY)
        with self.assertRaises(SystemExit):
            c.validate_r1_failure_evidence(
                MutatedFailureReader(extra=f"runtime/out/{c.R1_LABEL}/training.log"), R1_FAILURE_IDENTITY)

    def test_r1_failure_external_digest_mutation_rejects(self):
        identity = dict(R1_FAILURE_IDENTITY)
        identity["log_sha256"] = "1" * 64
        with self.assertRaises(SystemExit):
            c.validate_r1_failure_evidence(c.FailureGitReader(identity), identity)


class SymbolicStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", "-b", c.BRANCH, self.repo], check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=self.repo, check=True)
        (self.repo / "x").write_text("one\n")
        subprocess.run(["git", "add", "x"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "one"], cwd=self.repo, check=True)
        self.head = git(self.repo, "rev-parse", "HEAD")

    def tearDown(self):
        self.temp.cleanup()

    def test_correct_symbolic_branch_and_head_pass(self):
        self.assertEqual(stage.validate_symbolic_checkout(self.repo, c.BRANCH, self.head),
                         {"branch": c.BRANCH, "head": self.head, "clean": True})

    def test_detached_head_rejects(self):
        subprocess.run(["git", "checkout", "-q", "--detach", self.head], cwd=self.repo, check=True)
        with self.assertRaises(SystemExit):
            stage.validate_symbolic_checkout(self.repo, c.BRANCH, self.head)

    def test_wrong_symbolic_branch_rejects(self):
        subprocess.run(["git", "checkout", "-q", "-b", "wrong-branch"], cwd=self.repo, check=True)
        with self.assertRaises(SystemExit):
            stage.validate_symbolic_checkout(self.repo, c.BRANCH, self.head)

    def test_wrong_head_and_dirty_tree_reject(self):
        (self.repo / "x").write_text("two\n")
        with self.assertRaises(SystemExit):
            stage.validate_symbolic_checkout(self.repo, c.BRANCH, self.head)
        subprocess.run(["git", "checkout", "--", "x"], cwd=self.repo, check=True)
        with self.assertRaises(SystemExit):
            stage.validate_symbolic_checkout(self.repo, c.BRANCH, "0" * 40)


class NonOverwriteAndContractTests(unittest.TestCase):
    def test_o_excl_concrete_request_rejects_reuse(self):
        target = Path("/tmp") / f"a1-r2-oexcl-{os.getpid()}.json"
        if target.exists():
            self.fail("test-owned O_EXCL path unexpectedly exists")
        try:
            adapter.durable(target, {"schema": "test"})
            with self.assertRaises(FileExistsError):
                adapter.durable(target, {"schema": "changed"})
        finally:
            target.unlink(missing_ok=True)

    def test_artifact_store_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            store = aio.Store(Path(directory), "unique-r2")
            store.create()
            store.write("x", b"one")
            with self.assertRaises(FileExistsError):
                store.write("x", b"two")

    def test_decision_is_screening_only(self):
        decision = c.a1_decision("0.9866973734526626", "0.9856973734526626")
        self.assertEqual(decision["decision"], "NOMINATE_OTHER_HOST_FULL20M_REPLICATE")
        self.assertFalse(decision["keep_authorized"])
        self.assertFalse(decision["adoption_authorized"])


if __name__ == "__main__":
    unittest.main()
