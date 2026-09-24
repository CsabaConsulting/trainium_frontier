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

    def test_proposal_records_r1_failure_as_pending_and_no_science_result(self):
        proposal = json.loads((HERE / "EXPERIMENT_PROPOSAL.json").read_text())
        self.assertEqual(proposal["r1_failure"]["evidence_identity"],
                         "PENDING_AUTHENTICATED_R1_FAILURE_EVIDENCE_COMMIT")
        self.assertIsNone(proposal["r1_failure"]["science_result"])
        self.assertFalse(proposal["staging_contract"]["detached_head_allowed"])
        self.assertEqual(proposal["candidate"]["files_allowed"], [])


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
