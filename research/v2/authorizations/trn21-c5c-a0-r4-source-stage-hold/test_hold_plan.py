"""Offline Implementer assertions for the disabled trn21 446b stage plan."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONTRACT = HERE / "GATE_CONTRACT.json"
E65 = "e65a3097a5b00f6178188a935bb2acb9a5042c6c"
R4 = "446b657695f086f9b484fa601c9a3b61c2495912"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_bytes(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def require_disabled(plan):
    assert plan["schema"] == "trn21-c5c-a0-r4-clean-source-stage-disabled-v1"
    assert plan["host"] == "trn21"
    assert plan["expected_hostname"] == "ip-172-31-34-85"
    for key in ("source_transfer_authorized", "source_checkout_authorized",
                "cpu_g2_authorized", "native_training_authorized",
                "evaluation_authorized", "submission_authorized"):
        assert plan[key] is False
    source = plan["source_capsule"]
    assert source["frozen_source_commit"] == R4
    assert source["local_bytes"] == 104154875
    assert source["local_sha256"] == "767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a"
    targets = plan["proposed_new_trn21_paths"]
    assert targets["all_path_absences_live_verified"] is False
    assert targets["dev_shm_forbidden"] is True
    assert targets["no_in_place_edit_overwrite_delete_rename_or_retry"] is True
    assert targets["persistent_ext4_capsule"] != targets["existing_e65_checkout_preserved"]
    assert targets["private_clean_checkout"] != targets["existing_e65_checkout_preserved"]
    assert targets["persistent_ext4_capsule"].startswith("/home/ubuntu/")
    assert targets["private_clean_checkout"].startswith("/home/ubuntu/")
    floor = plan["pretransfer_safety_floor"]
    assert floor["minimum_root_free_bytes_after_every_stage"] == 10 * 1024**3
    assert floor["minimum_root_free_bytes_before_transfer"] == (
        floor["minimum_root_free_bytes_after_every_stage"]
        + source["local_bytes"] + source["unpacked_regular_file_bytes"] + 2 * 1024**3
    )
    assert floor["minimum_root_free_inodes_before_transfer"] >= 10000
    assert floor["native_operational_floor_bytes"] == 13 * 1024**3 // 2
    assert all(value is None for value in plan["missing_execution_gate_receipts"].values())
    assert plan["g2_source_identity_disposition"]["source_identity_verifier_receipt"] is None
    assert plan["g2_source_identity_disposition"]["native_dispatch_from_carry_alone"] is False


class HoldPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(CONTRACT.read_text())

    def test_disabled_and_nonoverwriting(self):
        require_disabled(self.plan)

    def test_local_capsule_and_manifest_pins(self):
        source = self.plan["source_capsule"]
        capsule = Path(source["local_path"])
        metadata = capsule.stat(follow_symlinks=False)
        self.assertTrue(capsule.is_file())
        self.assertEqual(metadata.st_size, source["local_bytes"])
        self.assertEqual(metadata.st_mode & 0o777, 0o600)
        self.assertEqual(metadata.st_nlink, 1)
        self.assertEqual(digest(capsule), source["local_sha256"])
        manifest = git_bytes("HEAD", source["manifest_path"])
        self.assertEqual(len(manifest), source["manifest_bytes"])
        self.assertEqual(hashlib.sha256(manifest).hexdigest(), source["manifest_sha256"])

    def test_frozen_science_and_safety_source_hashes(self):
        source = self.plan["frozen_source_file_sha256"]
        for path, expected in source.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256(git_bytes(R4, path)).hexdigest(), expected)
        for path in ("train.py", "prepare.py", "research/fresh_start_sync.py",
                     "research/run_experiment.py"):
            self.assertEqual(git_bytes(E65, path), git_bytes(R4, path))
        self.assertNotEqual(git_bytes(E65, "research/fresh_start_launch.py"),
                            git_bytes(R4, "research/fresh_start_launch.py"))

    def test_historical_protected_evidence_rebinds_without_authority(self):
        historical = self.plan["historical_trn21_protected_assets_not_fresh_preflight"]
        path = historical["checkout_postflight_path"]
        raw = git_bytes("HEAD", path)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         historical["checkout_postflight_sha256"])
        old = json.loads(raw)
        self.assertEqual(old["root_free_bytes"], historical["root_free_bytes"])
        self.assertEqual(old["cache_root"]["inode"], historical["warm_cache_root_inode"])
        for key, inode in historical["warm_cache_child_inodes"].items():
            self.assertEqual(old["caches"][key]["inode"], inode)
        self.assertEqual([(x["device"], x["inode"], x["bytes"], x["sha256"])
                          for x in old["checkpoints"]],
                         [(x["device"], x["inode"], x["bytes"], x["sha256"])
                          for x in historical["protected_checkpoints"]])

    def test_in_memory_authority_and_floor_mutations_fail(self):
        for key in ("source_transfer_authorized", "source_checkout_authorized",
                    "cpu_g2_authorized", "native_training_authorized"):
            changed = copy.deepcopy(self.plan)
            changed[key] = True
            with self.subTest(key=key), self.assertRaises(AssertionError):
                require_disabled(changed)
        changed = copy.deepcopy(self.plan)
        changed["pretransfer_safety_floor"]["minimum_root_free_bytes_before_transfer"] -= 1
        with self.assertRaises(AssertionError):
            require_disabled(changed)
        changed = copy.deepcopy(self.plan)
        changed["missing_execution_gate_receipts"]["operator_cross_session_transfer_result"] = "self attested"
        with self.assertRaises(AssertionError):
            require_disabled(changed)
        changed = copy.deepcopy(self.plan)
        changed["proposed_new_trn21_paths"]["private_clean_checkout"] = (
            changed["proposed_new_trn21_paths"]["existing_e65_checkout_preserved"])
        with self.assertRaises(AssertionError):
            require_disabled(changed)


if __name__ == "__main__":
    unittest.main()
