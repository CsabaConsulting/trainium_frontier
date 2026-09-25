"""Local, read-only consistency checks for the non-dispatchable r4 draft."""

import hashlib
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parent
SOURCE = "446b657695f086f9b484fa601c9a3b61c2495912"
STEM = "fs-20260925-v2-c5c-a0-control-trn22-r4"
OLD_STEM = "fs-20260925-v2-c5c-a0-control-trn22-r3"
REQUEST = PACKAGE / f"{STEM}.request.DISABLED.json"
BASE = PACKAGE / f"{STEM}.base-binding.STATIC-ONLY.json"
PAIR = PACKAGE / f"{STEM}.pair-binding.STATIC-ONLY.json"
CONTRACT = PACKAGE / "GATE_CONTRACT.json"
OLD_PREFIX = "research/v2/disk-monitor/"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def frozen_file(path):
    return subprocess.check_output(["git", "show", f"{SOURCE}:{path}"], cwd=ROOT)


class HoldPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = json.loads(REQUEST.read_text())
        cls.base = json.loads(BASE.read_text())
        cls.pair = json.loads(PAIR.read_text())
        cls.contract = json.loads(CONTRACT.read_text())

    def test_frozen_source_hashes_are_actual_git_blobs(self):
        self.assertEqual(self.contract["frozen_source_commit"], SOURCE)
        self.assertEqual(self.request["expected_commit"], SOURCE)
        self.assertEqual(self.base["expected_commit"], SOURCE)
        self.assertEqual(self.request["expected_sha256"], self.contract["source_sha256"])
        self.assertEqual(self.base["expected_sha256"], self.contract["source_sha256"])
        for path, expected in self.contract["source_sha256"].items():
            with self.subTest(path=path):
                self.assertEqual(sha256(frozen_file(path)), expected)
        self.assertEqual(set(self.request["fingerprint_paths"]),
                         set(self.contract["source_sha256"]))

    def test_request_to_base_to_pair_hash_chain(self):
        c, b, p = self.contract, self.base, self.pair
        self.assertEqual(sha256(REQUEST.read_bytes()), c["request_sha256"])
        self.assertEqual(sha256(BASE.read_bytes()), c["base_binding_sha256"])
        self.assertEqual(sha256(PAIR.read_bytes()), c["pair_binding_sha256"])
        self.assertEqual(b["request_sha256"], c["request_sha256"])
        self.assertEqual(p["request_sha256"], c["request_sha256"])
        self.assertEqual(p["base_binding_sha256"], c["base_binding_sha256"])
        self.assertEqual(b["monitor_script_sha256"],
                         c["source_sha256"][OLD_PREFIX + "root_free_guard_monitor.py"])
        self.assertEqual(p["wrapper_script_sha256"],
                         c["source_sha256"][OLD_PREFIX + "paired_root_free_guard_monitor.py"])

    def test_hard_hold_and_absent_authorization(self):
        c, r, b, p = self.contract, self.request, self.base, self.pair
        self.assertFalse(c["dispatch_authorized"])
        self.assertFalse(c["host_action_authorized"])
        self.assertFalse(c["source_safety_pass_claimed"])
        self.assertFalse(r["dispatch_authorized"])
        self.assertEqual(b["package_state"], "STATIC_ONLY")
        self.assertEqual(p["package_state"], "STATIC_ONLY")
        self.assertTrue(all(value is False for value in c["required_gates"].values()))
        self.assertIsNone(c["one_shot_authorization_sha256"])
        self.assertIsNone(c["source_transfer_artifact_sha256"])
        self.assertFalse(list(PACKAGE.glob("*authorization*.json")))
        self.assertTrue(r["state"].startswith("DISABLED_"))

    def test_single_new_label_host_and_thresholds(self):
        c, r, b, p = self.contract, self.request, self.base, self.pair
        self.assertEqual(c["label"], STEM)
        self.assertEqual((r["label"], b["label"], p["label"]), (STEM,) * 3)
        self.assertEqual((r["host"], c["host"]), ("trn22", "trn22"))
        self.assertEqual((r["remote"], b["remote"], p["remote"]),
                         (c["host_checkout_target"],) * 3)
        self.assertEqual((r["expected_hostname"], b["expected_hostname"],
                          p["expected_hostname"]), ("ip-172-31-32-253",) * 3)
        self.assertEqual((b["expected_uid"], p["expected_uid"]), (1000, 1000))
        self.assertEqual((r["min_free_bytes"], b["floor_free_bytes"]),
                         (6979321856, 6979321856))
        self.assertEqual((r["pre_launch_required_free_bytes"],
                          b["prelaunch_free_bytes"]), (10737418240,) * 2)
        self.assertEqual((b["stop_free_bytes"], p["stop_free_bytes"]),
                         (9126805504,) * 2)
        self.assertEqual(b["stop_free_bytes"] - b["floor_free_bytes"], 2 * 1024**3)
        self.assertLess(b["stop_free_bytes"], b["prelaunch_free_bytes"])
        self.assertNotIn(OLD_STEM, json.dumps((r, b, p)))
        self.assertEqual(r["monitor_interlock"]["pair_binding_path"],
                         p["base_binding_path"].replace(".base-binding.json", ".pair-binding.json"))

    def test_science_recipe_is_r3_with_only_new_label_and_paths(self):
        old = json.loads(frozen_file(OLD_PREFIX + OLD_STEM + ".request.DISABLED.json"))
        r = self.request
        self.assertEqual(r["command"], [s.replace("r3", "r4") for s in old["command"]])
        self.assertEqual(r["environment"], old["environment"])
        self.assertEqual(r["warm_cache_policy"], old["warm_cache_policy"])
        self.assertEqual(r["external_boundary_observer"]["script_sha256"],
                         old["external_boundary_observer"]["script_sha256"])
        self.assertEqual(r["external_boundary_observer"]["command"],
                         [s.replace("r3", "r4")
                          for s in old["external_boundary_observer"]["command"]])
        self.assertEqual(r["expected_sha256"]["train.py"],
                         old["expected_sha256"]["train.py"])
        self.assertEqual(r["expected_sha256"]["prepare.py"],
                         old["expected_sha256"]["prepare.py"])
        self.assertEqual(r["command"][-2:], ["--matrix-lr", "0.015"])

    def test_r3_binding_is_historical_not_current(self):
        old = self.contract["stale_r3_not_reusable"]
        self.assertNotEqual(old["source_commit"], SOURCE)
        for kind, suffix in (("request", "request.DISABLED"),
                             ("base_binding", "base-binding.STATIC-ONLY"),
                             ("pair_binding", "pair-binding.STATIC-ONLY")):
            with self.subTest(kind=kind):
                path = OLD_PREFIX + OLD_STEM + f".{suffix}.json"
                self.assertEqual(sha256(frozen_file(path)), old[kind + "_sha256"])
                self.assertNotEqual(old[kind + "_sha256"],
                                    self.contract[kind + "_sha256"])


if __name__ == "__main__":
    unittest.main()
