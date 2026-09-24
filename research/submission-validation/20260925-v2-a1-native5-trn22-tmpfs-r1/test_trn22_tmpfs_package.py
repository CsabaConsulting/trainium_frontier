from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock


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


def fd_mount_id(module, path: Path) -> int:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        return module._fd_mount_id(fd)
    finally:
        os.close(fd)


def mount(target: str, mount_id: int, device: str, fs_type: str) -> dict:
    major, minor = (int(value) for value in device.split(":"))
    return {
        "mount_id": mount_id, "parent_id": 1, "device": device,
        "st_dev": os.makedev(major, minor), "root": "/",
        "target": target, "mount_options": ["nodev", "nosuid", "rw"] if target == "/dev/shm" else ["rw"],
        "optional": [], "fs_type": fs_type, "source": "tmpfs" if fs_type == "tmpfs" else "/dev/root",
        "super_options": ["rw"],
        "target_identity": entry("directory", os.makedev(major, minor), mount_id * 10),
    }


def entry(kind: str, device: int, inode: int, *, size: int = 0,
          allocated: int = 4096, content: bytes = b"") -> dict:
    return {
        "kind": kind, "device": device, "inode": inode,
        "mode": 0o700 if kind == "directory" else 0o600,
        "uid": os.getuid(), "gid": os.getgid(), "nlink": 1,
        "size": size, "allocated": allocated, "mtime_ns": 100, "ctime_ns": 100,
        "content_sha256": (None if kind == "directory"
                           else hashlib.sha256(content).hexdigest()),
    }


def tree(path: str, mount_record: dict, *, inode: int, cache: bool = False) -> dict:
    payload = b"compiled-cache" if cache else b"model-artifact"
    root_entry = entry("directory", mount_record["st_dev"], inode)
    entries = {
        ".": root_entry,
        "artifact": entry("file", mount_record["st_dev"], inode + 1,
                          size=len(payload), allocated=8192, content=payload),
    }
    snapshot = {
        "schema": "a1-native-five-anchored-tree-v2", "path": path,
        "state": "present", "mount_id": mount_record["mount_id"],
        "mount_device": mount_record["st_dev"],
        "mount_root_identity_sha256": CONTRACT.canonical_sha(
            mount_record["target_identity"]),
        "ancestors": [copy.deepcopy(mount_record["target_identity"]),
                      copy.deepcopy(root_entry)],
        "ancestor_mount_ids": [mount_record["mount_id"], mount_record["mount_id"]],
        "entries": entries, "entry_count": len(entries),
        "logical_bytes": len(payload),
        "allocated_bytes": sum(item["allocated"] for item in entries.values()),
        "complete": True,
    }
    snapshot["manifest_sha256"] = CONTRACT.canonical_sha(
        CONTRACT._tree_manifest_payload(snapshot))
    return snapshot


def resource(arm: str, phase: str, *, root_free: int) -> dict:
    info = CONTRACT.ARMS[arm]
    return {
        "schema": "a1-native-five-trn22-resource-v2", "phase": phase,
        "root_free_bytes": root_free, "root_free_inodes": CONTRACT.ROOT_INODE_FLOOR,
        "tmpfs_free_bytes": (info["tmpfs_free_before"] if phase == "pre"
                             else info["tmpfs_free_after"]),
        "tmpfs_free_inodes": CONTRACT.TMPFS_INODE_FLOOR,
        "mem_available_bytes": (info["memory_before"] if phase == "pre"
                                else CONTRACT.LIVE_MEMORY_BREAKER),
        "swap_total_bytes": 0, "swap_used_bytes": 0,
    }


def watcher_receipt(arm: str, caches: dict, job: str, postflight: str,
                    terminal_manifest: str) -> dict:
    baseline = CONTRACT.canonical_sha(caches)
    samples = []
    for sequence, timestamp, phase in ((0, 90, "pre_launch"), (1, 105, "runtime"),
                                       (2, 120, "final_seal")):
        sample = {
            "schema": "a1-native-five-watch-sample-v2", "sequence": sequence,
            "monotonic_ns": timestamp, "phase": phase,
            "root_free_bytes": CONTRACT.ROOT_BREAKER,
            "root_free_inodes": CONTRACT.ROOT_INODE_FLOOR,
            "tmpfs_free_bytes": CONTRACT.ARMS[arm]["tmpfs_free_after"],
            "tmpfs_free_inodes": CONTRACT.TMPFS_INODE_FLOOR,
            "mem_available_bytes": CONTRACT.LIVE_MEMORY_BREAKER,
            "swap_total_bytes": 0, "swap_used_bytes": 0,
            "root_log_allocated": 4096, "cache_allocated": 12288,
            "output_allocated": 12288,
            "root_cache_manifest_sha256": baseline,
            "current_arm_manifest_sha256": terminal_manifest,
        }
        sample["sample_sha256"] = CONTRACT.canonical_sha(CONTRACT._sample_payload(sample))
        samples.append(sample)
    coverage = {"schema": "a1-native-five-watch-coverage-v2",
                "interval_limit_ns": CONTRACT.WATCH_INTERVAL_MAX_NS,
                "child_start_ns": 100, "child_exit_ns": 110,
                "final_seal_ns": 120, "complete": True}
    return {
        "schema": "a1-native-five-watch-receipt-v2", "status": "PASS",
        "arm": arm, "host": CONTRACT.HOST, "label": CONTRACT.ARMS[arm]["label"],
        "source_sha256": CONTRACT.GATE_SOURCE_SHA256[arm],
        "observer_sha256": CONTRACT.OBSERVER_SHA256, "job_sha256": job,
        "postflight_sha256": postflight,
        "baseline_root_caches_sha256": baseline,
        "sample_count": 3, "first_monotonic_ns": 90,
        "last_monotonic_ns": 120, "series_sha256": CONTRACT.canonical_sha(samples),
        "coverage_sha256": CONTRACT.canonical_sha(coverage), "samples": samples,
        "coverage": coverage, "complete": True,
    }


def arm_evidence(arm: str, mounts: dict, caches: dict, *, inode: int = 1000) -> dict:
    paths = CONTRACT.arm_paths(arm)
    job, postflight = "1" * 64, "2" * 64
    ranks = [f"{value:x}" * 64 for value in range(3, 7)]
    output = tree(paths["output"], mounts["tmpfs"], inode=inode)
    cache = tree(paths["compile_cache"], mounts["tmpfs"], inode=inode + 10, cache=True)
    checkpoint = {
        "schema": "a1-native-five-trn22-checkpoint-v2", "path": paths["checkpoint"],
        "mount_id": mounts["tmpfs"]["mount_id"],
        "mount_device": mounts["tmpfs"]["st_dev"],
        "device": mounts["tmpfs"]["st_dev"], "inode": inode + 2,
        "kind": "file", "mode": 0o600, "uid": os.getuid(), "gid": os.getgid(),
        "nlink": 1, "bytes": 469834849, "allocated": 469835776,
        "mtime_ns": 200, "ctime_ns": 200, "sha256": "a" * 64,
        "weights_only": True, "model_only": True, "finite": True,
        "strict_load": True, "logits_exact": True, "causality_exact": True,
        "same_fd_stable": True, "model_state_sha256": "b" * 64,
        "eval_logits_sha256": "c" * 64, "causal_digest_sha256": "d" * 64,
        "source_sha256": CONTRACT.GATE_SOURCE_SHA256[arm],
        "observer_sha256": CONTRACT.OBSERVER_SHA256,
        "rank_report_sha256": list(ranks), "job_sha256": job,
        "postflight_sha256": postflight,
        "parent_identity_sha256": CONTRACT.canonical_sha(output["entries"]["."]),
        "mount_root_identity_sha256": CONTRACT.canonical_sha(
            mounts["tmpfs"]["target_identity"]),
        "complete": True,
    }
    return {
        "schema": "a1-native-five-trn22-arm-evidence-v2", "arm": arm,
        "host": CONTRACT.HOST, "hostname": CONTRACT.HOSTNAME,
        "label": CONTRACT.ARMS[arm]["label"],
        "source_sha256": CONTRACT.GATE_SOURCE_SHA256[arm],
        "protected_source_sha256": copy.deepcopy(CONTRACT.SOURCE_PINS),
        "observer_sha256": CONTRACT.OBSERVER_SHA256,
        "train_argv_sha256": CONTRACT.canonical_sha(CONTRACT.expected_train_argv(arm)),
        "environment_sha256": CONTRACT.canonical_sha(CONTRACT._expected_environment(arm)),
        "job_sha256": job, "postflight_sha256": postflight,
        "rank_report_sha256": ranks, "output": output, "cache": cache,
        "checkpoint": checkpoint, "root_log_allocated": 4096,
        "process_quiescent": True, "device_idle": True,
        "watcher": watcher_receipt(
            arm, caches, job, postflight,
            CONTRACT.canonical_sha({"output": output["manifest_sha256"],
                                    "cache": cache["manifest_sha256"]})),
        "complete": True,
    }


def evidence(arm: str = "G0") -> tuple[dict, dict]:
    mounts = {"root": mount("/", 1, "259:1", "ext4"),
              "tmpfs": mount("/dev/shm", 2, "0:42", "tmpfs")}
    caches = {path: tree(path, mounts["root"], inode=100 + index * 10, cache=True)
              for index, path in enumerate(CONTRACT.ROOT_CACHE_PATHS)}
    pre_free = max(CONTRACT.ROOT_REQUIRED_BEFORE[arm], CONTRACT.ROOT_BREAKER) + 4096
    prior = [] if arm == "G0" else (["G0"] if arm == "G1" else ["G0", "G1"])
    retained = {name: arm_evidence(name, mounts, caches, inode=2000 + index * 100)
                for index, name in enumerate(prior)}
    pre = {
        "schema": "a1-native-five-trn22-pre-v2", "host": CONTRACT.HOST,
        "hostname": CONTRACT.HOSTNAME, "mounts": copy.deepcopy(mounts),
        "resources": resource(arm, "pre", root_free=pre_free),
        "root_caches": copy.deepcopy(caches), "fresh_paths_absent": True,
        "absence_receipt": {
            "schema": "a1-native-five-absence-set-v2", "arm": arm,
            "paths": [CONTRACT.arm_paths(arm)[key] for key in
                      ("tmpfs_arm", "output", "compile_cache", "checkpoint", "remote",
                       "bundle", "root_report", "child_log", "runtime_job",
                       "runtime_result", "dispatch", "controller_request")],
            "parent_identity_sha256": "e" * 64, "o_excl_creation_required": True,
            "authoritative": False, "complete": True,
        },
        "retained_arms": retained,
    }
    post = {
        "schema": "a1-native-five-trn22-post-v2", "host": CONTRACT.HOST,
        "hostname": CONTRACT.HOSTNAME, "mounts": copy.deepcopy(mounts),
        "resources": resource(arm, "post", root_free=pre_free - 4096),
        "root_caches": copy.deepcopy(caches), "retained_arms": retained,
        "current_arm": arm_evidence(arm, mounts, caches),
        "root_growth_bytes": 4096, "root_unallowlisted_growth_bytes": 0,
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
        root_dev, tmp_dev = os.stat("/").st_dev, os.stat("/dev/shm").st_dev
        root_token = f"{os.major(root_dev)}:{os.minor(root_dev)}"
        tmp_token = f"{os.major(tmp_dev)}:{os.minor(tmp_dev)}"
        text = (f"1 0 {root_token} / / rw,relatime - ext4 /dev/root rw\n"
                f"2 1 {tmp_token} / /dev/shm rw,nosuid,nodev - tmpfs tmpfs rw,size=65536k\n")
        root = CONTRACT.parse_mountinfo(text, "/")
        tmpfs = CONTRACT.parse_mountinfo(text, "/dev/shm")
        CONTRACT.validate_mounts(root, tmpfs)

    def test_wrong_mount_alias_type_or_noexec_fails(self):
        root = mount("/", 1, "259:1", "ext4")
        candidates = [mount("/dev/shm", 1, "259:1", "tmpfs"),
                      mount("/dev/shm", 2, "0:42", "ext4"),
                      mount("/dev/shm", 2, "0:42", "tmpfs")]
        candidates[-1]["mount_options"].append("noexec")
        wrong_identity = mount("/dev/shm", 2, "0:42", "tmpfs")
        wrong_identity["target_identity"]["inode"] += 1
        wrong_identity["target_identity"]["device"] = os.makedev(0, 41)
        candidates.append(wrong_identity)
        for candidate in candidates:
            with self.assertRaises(CONTRACT.ContractError):
                CONTRACT.validate_mounts(root, candidate)

    def test_positive_storage_verdict(self):
        pre, post = evidence()
        self.assertEqual(CONTRACT.adjudicate_storage("G0", pre, post)["status"], "PASS")

    def test_closed_schema_rejects_verifier_forged_evidence(self):
        attacks = []
        pre, post = evidence()
        pre["root_caches"][CONTRACT.ROOT_CACHE_PATHS[0]] = "forged"
        post["root_caches"][CONTRACT.ROOT_CACHE_PATHS[0]] = "forged"
        attacks.append(("bare-cache", "G0", pre, post))
        pre, post = evidence()
        post["current_arm"]["checkpoint"] = {
            "path": CONTRACT.arm_paths("G0")["checkpoint"], "finite": True,
            "same_fd_stable": True}
        attacks.append(("asserted-checkpoint", "G0", pre, post))
        pre, post = evidence("G1")
        pre["retained_arms"]["G0"] = "forged"
        post["retained_arms"]["G0"] = "forged"
        attacks.append(("forged-retained-g0", "G1", pre, post))
        pre, post = evidence()
        post["current_arm"]["unexpected"] = True
        attacks.append(("extra-arm-field", "G0", pre, post))
        pre, post = evidence()
        post["current_arm"]["checkpoint"]["rank_report_sha256"][2] = "0" * 64
        attacks.append(("rank-binding", "G0", pre, post))
        pre, post = evidence()
        post["current_arm"]["watcher"]["samples"][-1][
            "current_arm_manifest_sha256"] = "0" * 64
        post["current_arm"]["watcher"]["samples"][-1]["sample_sha256"] = CONTRACT.canonical_sha(
            CONTRACT._sample_payload(post["current_arm"]["watcher"]["samples"][-1]))
        post["current_arm"]["watcher"]["series_sha256"] = CONTRACT.canonical_sha(
            post["current_arm"]["watcher"]["samples"])
        attacks.append(("terminal-watcher-seal", "G0", pre, post))
        for name, arm, before, after in attacks:
            with self.subTest(name=name):
                self.assertNotEqual(CONTRACT.adjudicate_storage(arm, before, after)["status"],
                                    "PASS")

    def test_phase_specific_root_and_inode_reservations(self):
        for arm, required in CONTRACT.ROOT_REQUIRED_BEFORE.items():
            pre, post = evidence(arm)
            pre["resources"]["root_free_bytes"] = required - 1
            self.assertNotEqual(CONTRACT.adjudicate_storage(arm, pre, post)["status"], "PASS")
            pre, post = evidence(arm)
            pre["resources"]["root_free_inodes"] = CONTRACT.ROOT_INODE_FLOOR - 1
            self.assertNotEqual(CONTRACT.adjudicate_storage(arm, pre, post)["status"], "PASS")
            pre, post = evidence(arm)
            pre["resources"]["tmpfs_free_inodes"] = CONTRACT.TMPFS_INODE_FLOOR - 1
            self.assertNotEqual(CONTRACT.adjudicate_storage(arm, pre, post)["status"], "PASS")
        self.assertEqual(CONTRACT.ROOT_REQUIRED_BEFORE["G0"], 7_849_426_944)

    def test_cache_mutation_and_empty_cache_are_routing_invalid(self):
        pre, post = evidence()
        post["root_caches"][CONTRACT.ROOT_CACHE_PATHS[0]]["entries"]["artifact"]["size"] += 1
        self.assertEqual(CONTRACT.adjudicate_storage("G0", pre, post)["status"],
                         "CACHE_ROUTING_INVALID")
        pre, post = evidence()
        cache = post["current_arm"]["cache"]
        cache["entries"].pop("artifact")
        cache["entry_count"] = 1
        cache["logical_bytes"] = 0
        cache["allocated_bytes"] = cache["entries"]["."]["allocated"]
        cache["manifest_sha256"] = CONTRACT.canonical_sha(CONTRACT._tree_manifest_payload(cache))
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
            lambda pre, post: post["current_arm"].__setitem__(
                "root_log_allocated", CONTRACT.ROOT_LOG_CAP + 1),
            lambda pre, post: post["current_arm"]["output"].__setitem__("path", "/tmp/out"),
            lambda pre, post: post["current_arm"]["checkpoint"].__setitem__(
                "same_fd_stable", False),
            lambda pre, post: post["current_arm"]["checkpoint"].__setitem__("finite", False),
            lambda pre, post: post["current_arm"].__setitem__("process_quiescent", False),
            lambda pre, post: post.__setitem__("root_unallowlisted_growth_bytes", 1),
            lambda pre, post: pre.__setitem__("fresh_paths_absent", False),
            lambda pre, post: post["current_arm"]["checkpoint"].__setitem__("mode", 0o666),
            lambda pre, post: post["current_arm"]["checkpoint"].__setitem__(
                "source_sha256", "f" * 64),
            lambda pre, post: post["current_arm"]["watcher"].__setitem__("sample_count", 0),
        ):
            pre, post = evidence()
            mutate(pre, post)
            attacks.append((pre, post))
        for index, (pre, post) in enumerate(attacks):
            with self.subTest(index=index):
                self.assertNotEqual(CONTRACT.adjudicate_storage("G0", pre, post)["status"], "PASS")

    def test_watcher_terminates_on_cache_root_ram_swap_and_cap_violations(self):
        baseline = evidence()[0]["root_caches"]
        binding = {"source_sha256": CONTRACT.GATE_SOURCE_SHA256["G0"],
                   "observer_sha256": CONTRACT.OBSERVER_SHA256,
                   "job_sha256": "1" * 64, "postflight_sha256": "2" * 64}
        coverage = {"schema": "a1-native-five-watch-coverage-v2",
                    "interval_limit_ns": CONTRACT.WATCH_INTERVAL_MAX_NS,
                    "child_start_ns": 100, "child_exit_ns": 110,
                    "final_seal_ns": 120, "complete": True}
        def sample(sequence, timestamp, phase):
            value = {"schema": "a1-native-five-watch-sample-v2", "sequence": sequence,
                     "monotonic_ns": timestamp, "phase": phase,
                     "root_free_bytes": CONTRACT.ROOT_BREAKER,
                     "root_free_inodes": CONTRACT.ROOT_INODE_FLOOR,
                     "tmpfs_free_bytes": CONTRACT.ARMS["G0"]["tmpfs_free_after"],
                     "tmpfs_free_inodes": CONTRACT.TMPFS_INODE_FLOOR,
                     "mem_available_bytes": CONTRACT.LIVE_MEMORY_BREAKER,
                     "swap_total_bytes": 0, "swap_used_bytes": 0,
                     "root_log_allocated": 0, "cache_allocated": 8192,
                     "output_allocated": 0,
                     "root_cache_manifest_sha256": CONTRACT.canonical_sha(baseline),
                     "current_arm_manifest_sha256": "7" * 64}
            value["sample_sha256"] = CONTRACT.canonical_sha(CONTRACT._sample_payload(value))
            return value
        valid = [sample(0, 90, "pre_launch"), sample(1, 105, "runtime"),
                 sample(2, 120, "final_seal")]
        called = []
        def terminate():
            called.append(True)
            return {"term_sent": True, "kill_sent": True, "reaped": True, "quiescent": True}
        self.assertEqual(CONTRACT.run_watcher(
            "G0", valid, baseline, coverage, binding, terminate)["status"], "PASS")
        self.assertFalse(called)
        mutations = [
            lambda sample: sample.__setitem__("swap_total_bytes", 1),
            lambda sample: sample.__setitem__("mem_available_bytes", 1),
            lambda sample: sample.__setitem__("root_free_bytes", 1),
            lambda sample: sample.__setitem__("tmpfs_free_bytes", 1),
            lambda sample: sample.__setitem__("root_log_allocated", CONTRACT.ROOT_LOG_CAP + 1),
            lambda sample: sample.__setitem__("cache_allocated", CONTRACT.CACHE_CAP + 1),
            lambda sample: sample.__setitem__("output_allocated", CONTRACT.OUT_CAP + 1),
        ]
        for mutation in mutations:
            series = copy.deepcopy(valid)
            mutation(series[1])
            series[1]["sample_sha256"] = CONTRACT.canonical_sha(
                CONTRACT._sample_payload(series[1]))
            result = CONTRACT.run_watcher(
                "G0", series, baseline, coverage, binding, terminate)
            self.assertNotEqual(result["status"], "PASS")
            self.assertTrue(result["lifecycle"]["quiescent"])

        result = CONTRACT.run_watcher(
            "G0", [], baseline, coverage, binding, terminate)
        self.assertNotEqual(result["status"], "PASS")
        gap = copy.deepcopy(valid)
        gap[1]["monotonic_ns"] = CONTRACT.WATCH_INTERVAL_MAX_NS + 91
        gap[1]["sample_sha256"] = CONTRACT.canonical_sha(CONTRACT._sample_payload(gap[1]))
        result = CONTRACT.run_watcher("G0", gap, baseline, coverage, binding, terminate)
        self.assertNotEqual(result["status"], "PASS")

    def test_watcher_rejects_incomplete_termination_receipt(self):
        baseline = evidence()[0]["root_caches"]
        binding = {"source_sha256": CONTRACT.GATE_SOURCE_SHA256["G0"],
                   "observer_sha256": CONTRACT.OBSERVER_SHA256,
                   "job_sha256": "1" * 64, "postflight_sha256": "2" * 64}
        coverage = {"schema": "a1-native-five-watch-coverage-v2",
                    "interval_limit_ns": CONTRACT.WATCH_INTERVAL_MAX_NS,
                    "child_start_ns": 1, "child_exit_ns": 2, "final_seal_ns": 3,
                    "complete": True}
        with self.assertRaises(CONTRACT.ContractError):
            CONTRACT.run_watcher("G0", [], baseline, coverage, binding,
                                 lambda: {"term_sent": True, "reaped": False, "quiescent": False})

    def test_snapshot_rejects_symlink_hardlink_and_fifo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "file").write_bytes(b"x")
            os.symlink(root / "file", root / "link")
            with self.assertRaises(CONTRACT.ContractError):
                CONTRACT.snapshot_tree(root, mount_id=fd_mount_id(CONTRACT, root),
                                       expected_device=root.stat().st_dev)
            (root / "link").unlink()
            os.link(root / "file", root / "hard")
            with self.assertRaises(CONTRACT.ContractError):
                CONTRACT.snapshot_tree(root, mount_id=fd_mount_id(CONTRACT, root),
                                       expected_device=root.stat().st_dev)
            (root / "hard").unlink()
            os.mkfifo(root / "fifo")
            with self.assertRaises(CONTRACT.ContractError):
                CONTRACT.snapshot_tree(root, mount_id=fd_mount_id(CONTRACT, root),
                                       expected_device=root.stat().st_dev)

    def test_snapshot_detects_metadata_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "file").write_bytes(b"x")
            mount_id = fd_mount_id(CONTRACT, root)
            first = CONTRACT.snapshot_tree(root, mount_id=mount_id,
                                           expected_device=root.stat().st_dev)
            (root / "file").write_bytes(b"y")
            second = CONTRACT.snapshot_tree(root, mount_id=mount_id,
                                            expected_device=root.stat().st_dev)
            self.assertNotEqual(first, second)

    def test_snapshot_rejects_symlink_ancestor_and_late_nested_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = root / "real"
            real.mkdir()
            (real / "file").write_bytes(b"x")
            os.symlink(real, root / "alias")
            with self.assertRaises((CONTRACT.ContractError, OSError)):
                CONTRACT.snapshot_tree(
                    root / "alias", mount_id=fd_mount_id(CONTRACT, real),
                    expected_device=root.stat().st_dev)
            nested = root / "tree" / "nested"
            nested.mkdir(parents=True)
            (nested / "stable").write_bytes(b"stable")
            original = CONTRACT.snapshot_tree_revalidate
            def late_write(fd, expected, device, mount_id):
                (nested / "late").write_bytes(b"late")
                return original(fd, expected, device, mount_id)
            with mock.patch.object(CONTRACT, "snapshot_tree_revalidate", side_effect=late_write):
                with self.assertRaises(CONTRACT.ContractError):
                    CONTRACT.snapshot_tree(
                        root / "tree", mount_id=fd_mount_id(CONTRACT, root / "tree"),
                        expected_device=root.stat().st_dev)

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

    def test_absent_cache_snapshot_is_exact_and_anchored(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            target = Path(temp) / "missing" / "cache"
            # A verifier may place its clean worktree on a mount distinct from
            # /.  Both values must describe the fixture's actual parent mount.
            device = Path(temp).stat().st_dev
            snapshot = CONTRACT.snapshot_path(
                target, mount_id=fd_mount_id(CONTRACT, Path(temp)), expected_device=device)
            self.assertEqual(snapshot["state"], "absent")
            self.assertEqual(snapshot["entries"], {})
            mutated = copy.deepcopy(snapshot)
            mutated["entries"] = {"forged": entry("file", device, 9, content=b"x")}
            with self.assertRaises(CONTRACT.ContractError):
                CONTRACT.validate_tree_snapshot(
                    mutated, str(target), snapshot["mount_id"], device,
                    snapshot["mount_root_identity_sha256"])

    def test_root_and_tmpfs_present_and_absent_boundary_traversal(self):
        cases = [(HERE, "root"), (Path("/dev/shm"), "tmpfs")]
        for parent, label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(dir=parent) as temp:
                base = Path(temp)
                present = base / "present"
                present.mkdir()
                (present / "artifact").write_bytes(b"boundary-aware")
                mount_id = fd_mount_id(CONTRACT, base)
                device = base.stat().st_dev
                present_receipt = CONTRACT.snapshot_path(
                    present, mount_id=mount_id, expected_device=device)
                self.assertEqual(present_receipt["state"], "present")
                self.assertEqual(present_receipt["mount_id"], mount_id)
                boundary = next(index for index, (ancestor, current_mount_id) in enumerate(zip(
                    present_receipt["ancestors"], present_receipt["ancestor_mount_ids"]))
                    if ancestor["device"] == device and current_mount_id == mount_id)
                self.assertEqual(present_receipt["mount_root_identity_sha256"],
                                 CONTRACT.canonical_sha(present_receipt["ancestors"][boundary]))
                absent = base / "missing" / "cache"
                absent_receipt = CONTRACT.snapshot_path(
                    absent, mount_id=mount_id, expected_device=device)
                self.assertEqual(absent_receipt["state"], "absent")
                self.assertGreaterEqual(len(absent_receipt["ancestors"]), boundary + 1)
                with self.assertRaises(CONTRACT.ContractError):
                    CONTRACT.snapshot_path(
                        absent, mount_id=mount_id + 1, expected_device=device)
                if label == "tmpfs":
                    # Absence before reaching the authenticated /dev/shm
                    # boundary is not evidence about a tmpfs-owned target.
                    with self.assertRaises(CONTRACT.ContractError):
                        CONTRACT.snapshot_path(
                            Path("/dev/definitely-absent-before-shm-boundary"),
                            mount_id=mount_id, expected_device=device)

    def test_cross_mount_symlink_bind_and_pre_boundary_identity_attacks_fail(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            base = Path(temp)
            target = base / "target"
            target.mkdir()
            nested = target / "nested"
            nested.mkdir()
            (nested / "artifact").write_bytes(b"x")
            mount_id = fd_mount_id(CONTRACT, target)
            device = target.stat().st_dev

            os.symlink(target, base / "alias")
            with self.assertRaises((CONTRACT.ContractError, OSError)):
                CONTRACT.snapshot_path(
                    base / "alias", mount_id=mount_id, expected_device=device)

            original_mount_id = CONTRACT._fd_mount_id
            def simulated_bind(fd):
                current = original_mount_id(fd)
                try:
                    opened_path = os.readlink(f"/proc/self/fd/{fd}")
                except OSError:
                    return current
                return current + 1 if opened_path.endswith("/nested") else current
            with mock.patch.object(CONTRACT, "_fd_mount_id", side_effect=simulated_bind):
                with self.assertRaises(CONTRACT.ContractError):
                    CONTRACT.snapshot_path(
                        target, mount_id=mount_id, expected_device=device)

            # A bind reattachment of the same inode can evade stat identity;
            # the fresh name-open in revalidation must also match mount ID.
            fds, names, identities, mount_ids = CONTRACT._open_dir_chain(target)
            pinned_fds = set(fds)
            def simulated_late_bind(fd):
                current = original_mount_id(fd)
                return current if fd in pinned_fds else current + 1
            try:
                with mock.patch.object(CONTRACT, "_fd_mount_id",
                                       side_effect=simulated_late_bind):
                    with self.assertRaises(CONTRACT.ContractError):
                        CONTRACT._revalidate_chain(fds, names, identities, mount_ids)
            finally:
                for fd in reversed(fds):
                    os.close(fd)

            original_stat = CONTRACT.os.stat
            wrong_identity = original_stat("/tmp", follow_symlinks=False)
            def pre_boundary_substitution(name, *args, **kwargs):
                if name == "dev" and kwargs.get("dir_fd") is not None:
                    return wrong_identity
                return original_stat(name, *args, **kwargs)
            with mock.patch.object(CONTRACT.os, "stat", side_effect=pre_boundary_substitution):
                with self.assertRaises(CONTRACT.ContractError):
                    CONTRACT.snapshot_path(
                        target, mount_id=mount_id, expected_device=device)

    def test_cross_mount_parent_swap_after_capture_fails(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            base = Path(temp)
            target = base / "target"
            target.mkdir()
            (target / "artifact").write_bytes(b"original")
            mount_id = fd_mount_id(CONTRACT, target)
            device = target.stat().st_dev
            original_revalidate = CONTRACT._revalidate_chain
            swapped = []
            def swap_after_capture(fds, names, identities, mount_ids):
                original_revalidate(fds, names, identities, mount_ids)
                if not swapped:
                    swapped.append(True)
                    target.rename(base / "target.displaced")
                    target.mkdir()
                    (target / "artifact").write_bytes(b"replacement")
            with mock.patch.object(CONTRACT, "_revalidate_chain",
                                   side_effect=swap_after_capture):
                with self.assertRaises(CONTRACT.ContractError):
                    CONTRACT.snapshot_path(
                        target, mount_id=mount_id, expected_device=device)
            self.assertTrue(swapped)

    def test_post_reseal_same_inode_same_size_content_mutation_fails(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            target = Path(temp) / "target"
            target.mkdir()
            artifact = target / "artifact"
            artifact.write_bytes(b"AAAA")
            original_inode = artifact.stat().st_ino
            original = CONTRACT.snapshot_tree_revalidate
            calls = []

            def mutate_after_reseal(fd, expected, device, mount_id):
                result = original(fd, expected, device, mount_id)
                calls.append(True)
                if len(calls) == 1:
                    artifact.write_bytes(b"BBBB")
                    self.assertEqual(artifact.stat().st_ino, original_inode)
                    self.assertEqual(artifact.stat().st_size, 4)
                return result

            with mock.patch.object(CONTRACT, "snapshot_tree_revalidate",
                                   side_effect=mutate_after_reseal):
                with self.assertRaises(CONTRACT.ContractError):
                    CONTRACT.snapshot_path(
                        target, mount_id=fd_mount_id(CONTRACT, target),
                        expected_device=target.stat().st_dev)
            self.assertGreaterEqual(len(calls), 2)
            self.assertEqual(artifact.read_bytes(), b"BBBB")

    def test_post_reseal_target_directory_rebound_fails(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            base = Path(temp)
            target = base / "target"
            target.mkdir()
            (target / "artifact").write_bytes(b"original")
            mount_id = fd_mount_id(CONTRACT, target)
            device = target.stat().st_dev
            original = CONTRACT.snapshot_tree_revalidate
            calls = []

            def rebound_after_reseal(fd, expected, expected_device, expected_mount_id):
                result = original(fd, expected, expected_device, expected_mount_id)
                calls.append(True)
                if len(calls) == 1:
                    target.rename(base / "target.displaced")
                    target.mkdir()
                    (target / "artifact").write_bytes(b"replacement")
                return result

            with mock.patch.object(CONTRACT, "snapshot_tree_revalidate",
                                   side_effect=rebound_after_reseal):
                with self.assertRaises(CONTRACT.ContractError):
                    CONTRACT.snapshot_path(
                        target, mount_id=mount_id, expected_device=device)
            self.assertEqual(len(calls), 1)
            self.assertEqual((target / "artifact").read_bytes(), b"replacement")


class CheckpointGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.old_parent = CHECKPOINT.TMPFS_PARENT
        CHECKPOINT.TMPFS_PARENT = self.temp / "tmpfs-parent"
        self.path = CHECKPOINT.TMPFS_PARENT / "g0-r1" / "out" / "final.pt"
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"authenticated-model-only-checkpoint")
        self.path.chmod(0o600)
        self.device = self.path.stat().st_dev
        self.mount_id = fd_mount_id(CHECKPOINT, self.path)
        parent = self.path.parent.stat()
        self.bindings = {
            "source_sha256": CONTRACT.GATE_SOURCE_SHA256["G0"],
            "observer_sha256": CONTRACT.OBSERVER_SHA256,
            "rank_report_sha256": [f"{value:x}" * 64 for value in range(3, 7)],
            "job_sha256": "1" * 64, "postflight_sha256": "2" * 64,
            "parent_identity_sha256": CHECKPOINT._canonical_sha(CHECKPOINT._entry(parent)),
            "mount_root_identity_sha256": CHECKPOINT._canonical_sha(
                CHECKPOINT._entry(CHECKPOINT.TMPFS_PARENT.stat())),
        }

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
                "causality_exact": True, "model_state_sha256": "a" * 64,
                "eval_logits_sha256": "b" * 64, "causal_digest_sha256": "c" * 64}

    def test_same_fd_positive(self):
        result = CHECKPOINT.verify_checkpoint(
            self.path, self.device, self.mount_id, self.bindings, self.validator)
        self.assertTrue(result["same_fd_stable"])
        self.assertEqual(result["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_same_inode_mutation_during_validation_fails(self):
        def mutate(handle, proc_path):
            del handle, proc_path
            self.path.write_bytes(b"x" * len(b"authenticated-model-only-checkpoint"))
            return {"status": "PASS"}
        with self.assertRaises(CHECKPOINT.CheckpointError):
            CHECKPOINT.verify_checkpoint(
                self.path, self.device, self.mount_id, self.bindings, mutate)

    def test_path_replacement_during_validation_fails(self):
        def replace(handle, proc_path):
            del handle, proc_path
            moved = self.path.with_suffix(".old")
            self.path.rename(moved)
            self.path.write_bytes(b"authenticated-model-only-checkpoint")
            return {"status": "PASS"}
        with self.assertRaises(CHECKPOINT.CheckpointError):
            CHECKPOINT.verify_checkpoint(
                self.path, self.device, self.mount_id, self.bindings, replace)

    def test_symlink_wrong_device_and_hardlink_fail(self):
        target = self.path.with_suffix(".real")
        self.path.rename(target)
        os.symlink(target, self.path)
        with self.assertRaises((CHECKPOINT.CheckpointError, OSError)):
            CHECKPOINT.verify_checkpoint(
                self.path, self.device, self.mount_id, self.bindings, self.validator)
        self.path.unlink()
        target.rename(self.path)
        with self.assertRaises(CHECKPOINT.CheckpointError):
            CHECKPOINT.verify_checkpoint(
                self.path, self.device + 1, self.mount_id, self.bindings, self.validator)
        with self.assertRaises(CHECKPOINT.CheckpointError):
            CHECKPOINT.verify_checkpoint(
                self.path, self.device, self.mount_id + 1, self.bindings, self.validator)
        hard = self.path.with_suffix(".hard")
        os.link(self.path, hard)
        with self.assertRaises(CHECKPOINT.CheckpointError):
            CHECKPOINT.verify_checkpoint(
                self.path, self.device, self.mount_id, self.bindings, self.validator)

    def test_world_writable_checkpoint_and_parent_replacement_fail(self):
        self.path.chmod(0o666)
        with self.assertRaises(CHECKPOINT.CheckpointError):
            CHECKPOINT.verify_checkpoint(
                self.path, self.device, self.mount_id, self.bindings, self.validator)
        self.path.chmod(0o600)
        old = self.path.parent.with_name("out.old")
        self.path.parent.rename(old)
        self.path.parent.mkdir()
        replacement = self.path
        replacement.write_bytes(b"authenticated-model-only-checkpoint")
        replacement.chmod(0o600)
        with self.assertRaises(CHECKPOINT.CheckpointError):
            CHECKPOINT.verify_checkpoint(
                replacement, self.device, self.mount_id, self.bindings, self.validator)

    def test_checkpoint_binding_schema_and_mount_root_drift_fail(self):
        for key in ("source_sha256", "observer_sha256", "mount_root_identity_sha256"):
            mutated = copy.deepcopy(self.bindings)
            mutated[key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(CHECKPOINT.CheckpointError):
                CHECKPOINT.verify_checkpoint(
                    self.path, self.device, self.mount_id, mutated, self.validator)
        for key in ("job_sha256", "postflight_sha256"):
            malformed = copy.deepcopy(self.bindings)
            malformed[key] = "not-a-sha"
            with self.subTest(key=key), self.assertRaises(CHECKPOINT.CheckpointError):
                CHECKPOINT.verify_checkpoint(
                    self.path, self.device, self.mount_id, malformed, self.validator)
        extra = copy.deepcopy(self.bindings)
        extra["unexpected"] = True
        with self.assertRaises(CHECKPOINT.CheckpointError):
            CHECKPOINT.verify_checkpoint(
                self.path, self.device, self.mount_id, extra, self.validator)

    def test_parent_swap_at_component_open_boundary_fails(self):
        original_open = CHECKPOINT.os.open
        swapped = []
        def swap_before_open(name, flags, *args, **kwargs):
            if name == "out" and kwargs.get("dir_fd") is not None and not swapped:
                swapped.append(True)
                original_parent = self.path.parent
                original_parent.rename(original_parent.with_name("out.displaced"))
                original_parent.mkdir()
                replacement = original_parent / "final.pt"
                replacement.write_bytes(b"authenticated-model-only-checkpoint")
                replacement.chmod(0o600)
            return original_open(name, flags, *args, **kwargs)
        with mock.patch.object(CHECKPOINT.os, "open", side_effect=swap_before_open):
            with self.assertRaises(CHECKPOINT.CheckpointError):
                CHECKPOINT.verify_checkpoint(
                    self.path, self.device, self.mount_id, self.bindings, self.validator)
        self.assertTrue(swapped)


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
        self.assertEqual(proposal["repair_provenance"]["verifier_fail_commit"],
                         "7856efa86669c7a0834ade8f3656ba5464fe48ac")
        self.assertEqual(proposal["repair_provenance"]["verifier_receipt_sha256"],
                         "31651d1b71e08ebc1fc4c45e7881365133e34766575842de871114b4af7761da")
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
