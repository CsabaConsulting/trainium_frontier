"""Offline Implementer self-checks; not an independent Verifier receipt."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
E65 = "e65a3097a5b00f6178188a935bb2acb9a5042c6c"
R4 = "446b657695f086f9b484fa601c9a3b61c2495912"


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def source_bytes(commit, path):
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT
    )


def require_hold(request, contract):
    assert request["schema"] == "trn21-c5c-a0-native-request-disabled-v1"
    assert request["host"] == "trn21"
    assert request["label"] == "fs-20260925-v2-c5c-a0-control-trn21-r1"
    assert request["dispatch_authorized"] is False
    assert request["host_action_authorized"] is False
    assert request["cpu_g2_authorized"] is False
    assert request["evaluation_authorized"] is False
    assert request["submission_authorized"] is False
    assert request["executable_request"] is False
    assert request["one_attempt_only"] is True
    assert request.get("command") is None
    assert request.get("native_launcher") is None
    assert request.get("paired_monitor_binding") is None
    assert request.get("one_attempt_guard") is None
    assert request["source_checkout"]["commit"] == E65
    assert request["safety_source_gap"]["r4_source_commit"] == R4
    assert request["safety_source_gap"]["source_matched_native_safety_installed"] is False
    assert request["science"]["logical_nc_config"] == 2
    assert request["science"]["nproc_per_node"] == 4
    assert request["science"]["seed"] == 42
    assert request["science"]["max_train_seconds"] == 1800
    assert request["science"]["outer_timeout_seconds"] == 3600
    assert request["science"]["environment"] == {
        "NEURON_LOGICAL_NC_CONFIG": "2",
        "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
        "NEURON_COMPETITION_R1_SEED": "42",
        "NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS": "1800",
    }
    assert request["science"]["train_argv"] == [
        "--n-embd", "1024", "--no-ngram-ve", "--no-qk-shift",
        "--compile-sdpa-direct", "--nki-local-conv", "--bf16-norm-output",
        "--pack-factor", "32", "--clip-after-reduce", "--seq-len", "1024",
        "--no-eval-public", "--matrix-lr", "0.015",
    ]
    assert request["disk_safety"]["minimum_prelaunch_free_bytes"] == 10 * 1024**3
    assert request["disk_safety"]["operational_floor_bytes"] == 13 * 1024**3 // 2
    assert request["disk_safety"]["stop_free_bytes"] == 17 * 1024**3 // 2
    assert request["preserve"]["cache_cleanup_authorized"] is False
    assert request["preserve"]["checkpoint_prune_authorized"] is False
    assert len(request["preserve"]["warm_cache_children_inodes"]) == 6
    assert len(request["preserve"]["protected_checkpoints"]) == 2
    assert all(isinstance(p["inode"], int) and len(p["sha256"]) == 64
               for p in request["preserve"]["protected_checkpoints"])
    assert all(value is None for key, value in request["pending_gate_receipts"].items()
               if key not in {"g1_trn21_independent_pass_path", "g1_trn21_independent_pass_sha256",
                              "g2_final_request_path", "g2_final_request_sha256"})
    assert contract["status"] == "IMPLEMENTER_DRAFT_NON_DISPATCHABLE"
    assert contract["host_action_authorized"] is False
    assert contract["native_dispatch_authorized"] is False
    assert contract["source_safety_disposition"]["e65_old_launcher_is_native_safe"] is False
    assert contract["source_safety_disposition"]["in_place_e65_edit_authorized"] is False
    assert all(value is None for value in contract["current_missing_evidence"].values())


class HoldRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = json.loads((HERE / "NATIVE_REQUEST.HOLD.json").read_text())
        cls.contract = json.loads((HERE / "GATE_CONTRACT.json").read_text())

    def test_disabled_contract(self):
        require_hold(self.request, self.contract)
        rel = self.contract["request_path"]
        self.assertEqual(sha_bytes((ROOT / rel).read_bytes()),
                         self.contract["request_sha256"])

    def test_source_hashes_and_train_equivalence(self):
        identity = self.request["source_checkout"]
        safety = self.request["safety_source_gap"]
        for name, key in (("train.py", "train_sha256"),
                          ("prepare.py", "prepare_sha256")):
            old = source_bytes(E65, name)
            self.assertEqual(sha_bytes(old), identity[key])
            self.assertEqual(old, source_bytes(R4, name))
        unchanged = ["research/fresh_start_sync.py", "research/run_experiment.py"]
        for name in unchanged:
            self.assertEqual(source_bytes(E65, name), source_bytes(R4, name))
        paths = {
            "research/fresh_start_launch.py": ("e65_launcher_sha256", "r4_launcher_sha256"),
            "research/fresh_start_guard.py": ("e65_guard_sha256", "r4_guard_sha256"),
            "research/fresh_start_host.py": ("e65_host_sha256", "r4_host_sha256"),
        }
        for name, (old_key, new_key) in paths.items():
            self.assertEqual(sha_bytes(source_bytes(E65, name)), safety[old_key])
            self.assertEqual(sha_bytes(source_bytes(R4, name)), safety[new_key])
            self.assertNotEqual(safety[old_key], safety[new_key])
        for name, key in (
            ("research/fresh_start_monitor_interlock.py", "r4_interlock_sha256"),
            ("research/v2/disk-monitor/root_free_guard_monitor.py", "r4_base_monitor_sha256"),
            ("research/v2/disk-monitor/paired_root_free_guard_monitor.py", "r4_paired_monitor_sha256"),
        ):
            self.assertEqual(sha_bytes(source_bytes(R4, name)), safety[key])
            self.assertNotEqual(0, subprocess.run(
                ["git", "cat-file", "-e", f"{E65}:{name}"], cwd=ROOT,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False).returncode)

    def test_e65_launcher_cannot_enforce_pair_or_auth(self):
        old = source_bytes(E65, "research/fresh_start_launch.py")
        new = source_bytes(R4, "research/fresh_start_launch.py")
        self.assertNotIn(b"dispatch_authorized", old)
        self.assertNotIn(b"require_live_pair", old)
        self.assertIn(b"dispatch_authorized", new)
        self.assertIn(b"require_live_pair", new)

    def test_adversarial_disabled_mutations(self):
        for key, bad in (
            ("dispatch_authorized", True),
            ("host_action_authorized", True),
            ("cpu_g2_authorized", True),
            ("command", ["torchrun", "train.py"]),
            ("native_launcher", "/some/path"),
            ("paired_monitor_binding", "/some/path"),
            ("one_attempt_guard", "/some/path"),
        ):
            with self.subTest(key=key):
                altered = copy.deepcopy(self.request)
                altered[key] = bad
                with self.assertRaises(AssertionError):
                    require_hold(altered, self.contract)
        changed = copy.deepcopy(self.request)
        changed["science"]["environment"]["NEURON_COMPETITION_R1_SEED"] = "43"
        with self.assertRaises(AssertionError):
            require_hold(changed, self.contract)
        changed = copy.deepcopy(self.request)
        changed["disk_safety"]["minimum_prelaunch_free_bytes"] -= 1
        with self.assertRaises(AssertionError):
            require_hold(changed, self.contract)
        changed = copy.deepcopy(self.request)
        changed["pending_gate_receipts"]["g2_numeric_result_independent_pass"] = "self-attested"
        with self.assertRaises(AssertionError):
            require_hold(changed, self.contract)


if __name__ == "__main__":
    unittest.main()
