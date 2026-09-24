from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = load("trn22_storage_contract_test", HERE / "storage_contract.py")
CHECKPOINT = load("trn22_checkpoint_guard_test", HERE / "checkpoint_guard.py")
if importlib.util.find_spec("torch") is None:
    sys.modules["torch"] = types.ModuleType("torch")
OBSERVER = load("trn22_observer_test", ROOT / "research/a1_native_gate_observer.py")
COMPARATOR = load("trn22_comparator_test", HERE / "compare_gate_reports_trn22.py")


class DummyDist:
    def all_reduce(self, tensor, *args, **kwargs):
        return tensor

    def all_gather_into_tensor(self, output, input_tensor, *args, **kwargs):
        return output


def request() -> dict:
    return json.loads((HERE / "g0-request.pending.json").read_text())


def mount(target: str, mount_id: int, device: str, fs_type: str) -> dict:
    return {
        "mount_id": mount_id, "parent_id": 1, "device": device, "root": "/",
        "target": target, "mount_options": ["nodev", "nosuid", "rw"] if target == "/dev/shm" else ["rw"],
        "optional": [], "fs_type": fs_type, "source": "tmpfs" if fs_type == "tmpfs" else "/dev/root",
        "super_options": ["rw"],
    }


def evidence(arm: str = "G0") -> tuple[dict, dict]:
    mounts = {"root": mount("/", 1, "259:1", "ext4"),
              "tmpfs": mount("/dev/shm", 2, "0:42", "tmpfs")}
    caches = {path: {"path": path, "sha256": "a" * 64, "allocated": 8192}
              for path in CONTRACT.ROOT_CACHE_PATHS}
    info = CONTRACT.ARMS[arm]
    pre = {
        "host": CONTRACT.HOST, "hostname": CONTRACT.HOSTNAME, "mounts": copy.deepcopy(mounts),
        "resources": {"swap_total_bytes": 0, "swap_used_bytes": 0,
                      "mem_available_bytes": info["memory_before"],
                      "root_free_bytes": CONTRACT.ROOT_BREAKER,
                      "tmpfs_free_bytes": info["tmpfs_free_before"]},
        "root_caches": copy.deepcopy(caches),
        "fresh_paths_absent": True,
        "retained_arms": {},
    }
    post = {
        "host": CONTRACT.HOST, "hostname": CONTRACT.HOSTNAME, "mounts": copy.deepcopy(mounts),
        "resources": {"swap_total_bytes": 0, "swap_used_bytes": 0,
                      "mem_available_bytes": CONTRACT.LIVE_MEMORY_BREAKER,
                      "root_free_bytes": CONTRACT.ROOT_BREAKER,
                      "tmpfs_free_bytes": info["tmpfs_free_after"]},
        "root_caches": copy.deepcopy(caches),
        "declared_cache": {"path": CONTRACT.arm_paths(arm)["compile_cache"],
                           "allocated": 8192, "mount_device": "0:42"},
        "output": {"path": CONTRACT.arm_paths(arm)["output"], "allocated": 1024**2,
                   "mount_device": "0:42"},
        "root_log_allocated": 4096,
        "root_allowed_growth_bytes": 4096,
        "root_unallowlisted_growth_bytes": 0,
        "retained_arms": {},
        "checkpoint": {"path": CONTRACT.arm_paths(arm)["checkpoint"],
                       "mount_device": "0:42", "regular": True, "nlink": 1,
                       "same_fd_stable": True, "weights_only": True, "model_only": True,
                       "finite": True, "strict_load": True, "logits_exact": True,
                       "causality_exact": True},
        "process_quiescent": True, "device_idle": True,
    }
    return pre, post


class RequestContractTests(unittest.TestCase):
    def test_g0_pending_request_is_exact(self):
        CONTRACT.validate_request(request(), "G0")

    def test_every_request_field_mutation_fails(self):
        attacks = [
            ("host", "trn21"), ("expected_hostname", "wrong"), ("label", "consumed"),
            ("executable", True), ("world_size", 8), ("lnc", 1), ("seed", 7),
            ("optimizer_updates", 6), ("state", "finalized"),
        ]
        for key, value in attacks:
            with self.subTest(key=key):
                mutated = request()
                mutated[key] = value
                with self.assertRaises(CONTRACT.ContractError):
                    CONTRACT.validate_request(mutated, "G0")

    def test_path_environment_argv_hash_and_threshold_mutations_fail(self):
        mutations = []
        for section, key, value in (
            ("paths", "output", "/tmp/out"),
            ("environment", "NEURON_COMPILE_CACHE_DIR", "/tmp/neff_cache"),
            ("source_sha256", "prepare.py", "0" * 64),
            ("storage", "root_breaker", 1),
            ("storage", "tmpfs_free_before", 1),
            ("lifecycle", "cleanup_authorized", True),
            ("external_literals", "request_sha256", "0" * 64),
        ):
            mutated = request()
            mutated[section][key] = value
            mutations.append(mutated)
        argv = request()
        argv["train_argv"][-1] = "/tmp/report"
        mutations.append(argv)
        for index, mutated in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(CONTRACT.ContractError):
                CONTRACT.validate_request(mutated, "G0")

    def test_authority_and_downstream_are_locked(self):
        auth = json.loads((HERE / "g0-authorization.pending.json").read_text())
        state = json.loads((HERE / "requests.pending.json").read_text())
        self.assertFalse(auth["executable"])
        self.assertFalse(auth["human_authorization"])
        self.assertFalse(auth["g0_authorized"])
        self.assertFalse(state["executable"])
        self.assertIsNone(state["g1_request"])
        self.assertIsNone(state["g2_request"])
        self.assertFalse(state["materialization_authorized"])

    def test_all_arm_paths_are_unique_and_tmpfs_outputs(self):
        path_sets = [set(CONTRACT.arm_paths(arm).values()) for arm in CONTRACT.ARMS]
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertFalse(path_sets[left] & path_sets[right])
        for arm in CONTRACT.ARMS:
            paths = CONTRACT.arm_paths(arm)
            self.assertTrue(paths["output"].startswith(CONTRACT.TMPFS_PARENT + "/"))
            self.assertTrue(paths["compile_cache"].startswith(CONTRACT.TMPFS_PARENT + "/"))

    def test_frozen_science_hashes(self):
        for relative, digest in CONTRACT.SOURCE_PINS.items():
            self.assertEqual(CONTRACT.sha256_file(ROOT / relative), digest, relative)


class ObserverAndComparatorTests(unittest.TestCase):
    def test_exact_trn22_observer_cross_boundary(self):
        for arm, info in CONTRACT.ARMS.items():
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temp:
                observer = OBSERVER.NativeGateObserver(
                    arm=arm, detailed=info["observation"], expected_mode=info["mode"],
                    rank=0, world_size=4, report_dir=Path(temp) / "report", dist_module=DummyDist(),
                    label=info["label"], host="trn22", config={})
                os.close(observer._fd)

    def test_trn21_and_consumed_labels_rejected(self):
        consumed = [
            "fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1",
            "fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1",
            "fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2",
            "fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3",
        ]
        for host, label in [("trn21", CONTRACT.ARMS["G0"]["label"])] + [("trn22", item) for item in consumed]:
            with self.subTest(host=host, label=label), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):
                    OBSERVER.NativeGateObserver(
                        arm="G0", detailed=False, expected_mode="hybrid", rank=0,
                        world_size=4, report_dir=Path(temp) / "report", dist_module=DummyDist(),
                        label=label, host=host, config={})

    def test_comparator_binds_exact_new_identity(self):
        self.assertEqual(COMPARATOR.BASE_COMPARATOR.EXPECTED_HOST, "trn22")
        self.assertEqual(COMPARATOR.BASE_COMPARATOR.LABELS,
                         {arm: info["label"] for arm, info in CONTRACT.ARMS.items()})
        self.assertEqual(COMPARATOR.BASE_COMPARATOR.OWNERSHIP_MANIFEST_SHA256,
                         hashlib.sha256((ROOT / "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/a1_ownership_manifest.json").read_bytes()).hexdigest())

    def test_manifest_exact_66_entry_contract_and_new_observer_pin(self):
        manifest = COMPARATOR.BASE_COMPARATOR._load_ownership_manifest()
        self.assertEqual(len(manifest["owned_entries"]), 66)
        self.assertEqual(manifest["summary"]["elements_by_owner"],
                         [27405059, 30540035, 30408963, 29362179])
        self.assertEqual(manifest["summary"]["shard_elements"], 30540160)
        self.assertEqual(manifest["summary"]["total_padded_elements"], 122160640)
        self.assertEqual(manifest["source_pins"]["files"]["research/a1_native_gate_observer.py"],
                         hashlib.sha256((ROOT / "research/a1_native_gate_observer.py").read_bytes()).hexdigest())


class MountAndStorageTests(unittest.TestCase):
    def test_mount_parser_and_exact_options(self):
        text = ("1 0 259:1 / / rw,relatime - ext4 /dev/root rw\n"
                "2 1 0:42 / /dev/shm rw,nosuid,nodev - tmpfs tmpfs rw,size=65536k\n")
        root = CONTRACT.parse_mountinfo(text, "/")
        tmpfs = CONTRACT.parse_mountinfo(text, "/dev/shm")
        CONTRACT.validate_mounts(root, tmpfs)

    def test_wrong_mount_alias_type_or_noexec_fails(self):
        root = mount("/", 1, "259:1", "ext4")
        candidates = [mount("/dev/shm", 1, "259:1", "tmpfs"),
                      mount("/dev/shm", 2, "0:42", "ext4"),
                      mount("/dev/shm", 2, "0:42", "tmpfs")]
        candidates[-1]["mount_options"].append("noexec")
        for candidate in candidates:
            with self.assertRaises(CONTRACT.ContractError):
                CONTRACT.validate_mounts(root, candidate)

    def test_positive_storage_verdict(self):
        pre, post = evidence()
        self.assertEqual(CONTRACT.adjudicate_storage("G0", pre, post)["status"], "PASS")

    def test_cache_mutation_and_empty_cache_are_routing_invalid(self):
        pre, post = evidence()
        post["root_caches"][CONTRACT.ROOT_CACHE_PATHS[0]]["allocated"] += 4096
        self.assertEqual(CONTRACT.adjudicate_storage("G0", pre, post)["status"],
                         "CACHE_ROUTING_INVALID")
        pre, post = evidence()
        post["declared_cache"]["allocated"] = 4096
        self.assertEqual(CONTRACT.adjudicate_storage("G0", pre, post)["status"],
                         "CACHE_ROUTING_INVALID")

    def test_incomplete_cache_enumeration_is_routing_invalid(self):
        pre, post = evidence()
        pre["root_caches"].pop(CONTRACT.ROOT_CACHE_PATHS[-1])
        post["root_caches"].pop(CONTRACT.ROOT_CACHE_PATHS[-1])
        self.assertEqual(CONTRACT.adjudicate_storage("G0", pre, post)["status"],
                         "CACHE_ROUTING_INVALID")

    def test_resource_mount_log_and_checkpoint_attacks_fail(self):
        attacks = []
        for mutate in (
            lambda pre, post: post["resources"].__setitem__("mem_available_bytes", 1),
            lambda pre, post: post["resources"].__setitem__("swap_total_bytes", 1),
            lambda pre, post: post.__setitem__("root_log_allocated", CONTRACT.ROOT_LOG_CAP + 1),
            lambda pre, post: post["output"].__setitem__("path", "/tmp/out"),
            lambda pre, post: post["checkpoint"].__setitem__("same_fd_stable", False),
            lambda pre, post: post["checkpoint"].__setitem__("finite", False),
            lambda pre, post: post.__setitem__("process_quiescent", False),
            lambda pre, post: post.__setitem__("root_unallowlisted_growth_bytes", 1),
            lambda pre, post: pre.__setitem__("fresh_paths_absent", False),
        ):
            pre, post = evidence()
            mutate(pre, post)
            attacks.append((pre, post))
        for index, (pre, post) in enumerate(attacks):
            with self.subTest(index=index):
                self.assertNotEqual(CONTRACT.adjudicate_storage("G0", pre, post)["status"], "PASS")

    def test_watcher_terminates_on_cache_root_ram_swap_and_cap_violations(self):
        baseline = evidence()[0]["root_caches"]
        valid = {"root_caches": copy.deepcopy(baseline), "swap_total_bytes": 0,
                 "swap_used_bytes": 0, "mem_available_bytes": CONTRACT.LIVE_MEMORY_BREAKER,
                 "root_free_bytes": CONTRACT.ROOT_BREAKER,
                 "tmpfs_free_bytes": CONTRACT.ARMS["G0"]["tmpfs_free_after"],
                 "root_log_allocated": 0, "cache_allocated": 8192, "output_allocated": 0}
        called = []
        def terminate():
            called.append(True)
            return {"term_sent": True, "kill_sent": True, "reaped": True, "quiescent": True}
        self.assertEqual(CONTRACT.run_watcher("G0", [valid], baseline, terminate)["status"], "PASS")
        self.assertFalse(called)
        mutations = [
            lambda sample: sample["root_caches"][CONTRACT.ROOT_CACHE_PATHS[0]].__setitem__("allocated", 1),
            lambda sample: sample.__setitem__("swap_total_bytes", 1),
            lambda sample: sample.__setitem__("mem_available_bytes", 1),
            lambda sample: sample.__setitem__("root_free_bytes", 1),
            lambda sample: sample.__setitem__("tmpfs_free_bytes", 1),
            lambda sample: sample.__setitem__("root_log_allocated", CONTRACT.ROOT_LOG_CAP + 1),
            lambda sample: sample.__setitem__("cache_allocated", CONTRACT.CACHE_CAP + 1),
            lambda sample: sample.__setitem__("output_allocated", CONTRACT.OUT_CAP + 1),
        ]
        for mutation in mutations:
            sample = copy.deepcopy(valid)
            mutation(sample)
            result = CONTRACT.run_watcher("G0", [sample], baseline, terminate)
            self.assertNotEqual(result["status"], "PASS")
            self.assertTrue(result["lifecycle"]["quiescent"])

    def test_watcher_rejects_incomplete_termination_receipt(self):
        baseline = evidence()[0]["root_caches"]
        bad = {"root_caches": baseline, "swap_total_bytes": 1}
        with self.assertRaises(CONTRACT.ContractError):
            CONTRACT.run_watcher("G0", [bad], baseline,
                                 lambda: {"term_sent": True, "reaped": False, "quiescent": False})

    def test_snapshot_rejects_symlink_hardlink_and_fifo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "file").write_bytes(b"x")
            os.symlink(root / "file", root / "link")
            with self.assertRaises(CONTRACT.ContractError):
                CONTRACT.snapshot_tree(root)
            (root / "link").unlink()
            os.link(root / "file", root / "hard")
            with self.assertRaises(CONTRACT.ContractError):
                CONTRACT.snapshot_tree(root)
            (root / "hard").unlink()
            os.mkfifo(root / "fifo")
            with self.assertRaises(CONTRACT.ContractError):
                CONTRACT.snapshot_tree(root)

    def test_snapshot_detects_metadata_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "file").write_bytes(b"x")
            first = CONTRACT.snapshot_tree(root)
            (root / "file").write_bytes(b"y")
            second = CONTRACT.snapshot_tree(root)
            self.assertNotEqual(first, second)

    def test_absence_rejects_existing_symlink_and_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            CONTRACT.validate_initial_absence(root, root / "absent" / "child")
            (root / "real").mkdir()
            os.symlink(root / "real", root / "link")
            with self.assertRaises(CONTRACT.ContractError):
                CONTRACT.validate_initial_absence(root, root / "link" / "child")
            with self.assertRaises(CONTRACT.ContractError):
                CONTRACT.validate_initial_absence(root, root.parent / "escape")


class CheckpointGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.old_parent = CHECKPOINT.TMPFS_PARENT
        CHECKPOINT.TMPFS_PARENT = self.temp / "tmpfs-parent"
        self.path = CHECKPOINT.TMPFS_PARENT / "g0-r1" / "out" / "final.pt"
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"authenticated-model-only-checkpoint")
        self.device = self.path.stat().st_dev

    def tearDown(self):
        CHECKPOINT.TMPFS_PARENT = self.old_parent
        shutil.rmtree(self.temp)

    @staticmethod
    def validator(handle, proc_path):
        if handle.read() != b"authenticated-model-only-checkpoint":
            raise AssertionError("wrong bytes")
        if not proc_path.startswith("/proc/self/fd/"):
            raise AssertionError("not pinned fd")
        return {"status": "PASS", "weights_only": True, "model_only": True,
                "finite": True, "strict_load": True, "logits_exact": True,
                "causality_exact": True}

    def test_same_fd_positive(self):
        result = CHECKPOINT.verify_checkpoint(self.path, self.device, self.validator)
        self.assertTrue(result["same_fd_stable"])
        self.assertEqual(result["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_same_inode_mutation_during_validation_fails(self):
        def mutate(handle, proc_path):
            del handle, proc_path
            self.path.write_bytes(b"x" * len(b"authenticated-model-only-checkpoint"))
            return {"status": "PASS"}
        with self.assertRaises(CHECKPOINT.CheckpointError):
            CHECKPOINT.verify_checkpoint(self.path, self.device, mutate)

    def test_path_replacement_during_validation_fails(self):
        def replace(handle, proc_path):
            del handle, proc_path
            moved = self.path.with_suffix(".old")
            self.path.rename(moved)
            self.path.write_bytes(b"authenticated-model-only-checkpoint")
            return {"status": "PASS"}
        with self.assertRaises(CHECKPOINT.CheckpointError):
            CHECKPOINT.verify_checkpoint(self.path, self.device, replace)

    def test_symlink_wrong_device_and_hardlink_fail(self):
        target = self.path.with_suffix(".real")
        self.path.rename(target)
        os.symlink(target, self.path)
        with self.assertRaises((CHECKPOINT.CheckpointError, OSError)):
            CHECKPOINT.verify_checkpoint(self.path, self.device, self.validator)
        self.path.unlink()
        target.rename(self.path)
        with self.assertRaises(CHECKPOINT.CheckpointError):
            CHECKPOINT.verify_checkpoint(self.path, self.device + 1, self.validator)
        hard = self.path.with_suffix(".hard")
        os.link(self.path, hard)
        with self.assertRaises(CHECKPOINT.CheckpointError):
            CHECKPOINT.verify_checkpoint(self.path, self.device, self.validator)


class PackageStateTests(unittest.TestCase):
    def test_protected_hash_manifest_matches_exact_bytes(self):
        manifest = json.loads((HERE / "protected-hashes.json").read_text())
        self.assertFalse(manifest["executable"])
        self.assertFalse(manifest["hardware_authorized"])
        for relative, expected in manifest["files"].items():
            with self.subTest(relative=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)

    def test_proposal_has_no_authority(self):
        proposal = json.loads((HERE / "EXPERIMENT_PROPOSAL.json").read_text())
        self.assertFalse(proposal["authority"]["training_authorized"])
        self.assertFalse(proposal["authority"]["host_access_authorized"])
        self.assertFalse(proposal["authority"]["materialization_authorized"])
        self.assertIsNone(proposal["authority"]["g1_request"])
        self.assertIsNone(proposal["authority"]["g2_request"])

    def test_no_concrete_request_or_downstream_auth_exists(self):
        names = {path.name for path in HERE.iterdir()}
        self.assertNotIn("g0-request.json", names)
        self.assertFalse(any(name.startswith("g1-") and "request" in name for name in names))
        self.assertFalse(any(name.startswith("g2-") and "request" in name for name in names))
        self.assertFalse(any(name.startswith("g1-") and "authorization" in name for name in names))
        self.assertFalse(any(name.startswith("g2-") and "authorization" in name for name in names))

    def test_runbook_forbids_launch_cleanup_and_auto_advance(self):
        text = (HERE / "RUNBOOK.md").read_text()
        normalized = " ".join(text.split())
        for phrase in ("NON-EXECUTABLE", "No concrete launcher request exists",
                       "No cleanup", "advancement is never automatic"):
            self.assertIn(phrase, normalized)


if __name__ == "__main__":
    unittest.main()
