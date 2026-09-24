#!/usr/bin/env python3
"""Fail-closed storage and identity contract for the trn22 native-five gate.

This module does not launch work.  A later independently authorized runtime may
use it to capture pre/post storage evidence and adjudicate cache routing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any


SCHEMA = "a1-native-five-trn22-tmpfs-request-v1"
HOST = "trn22"
HOSTNAME = "ip-172-31-32-253"
TMPFS_PARENT = "/dev/shm/trainium_native5_20260925_trn22"
ROOT_BREAKER = 7_247_757_312
ROOT_OPERATIONAL_FLOOR = 6_979_321_856
ROOT_HARD_ABORT = 5_368_709_120
ROOT_LOG_CAP = 8 * 1024 * 1024
SOURCE_STAGE_ALLOWANCE = 192_167_936
OUT_CAP = 1 * 1024**3
CACHE_CAP = 12 * 1024**3
ARM_CAP = 13 * 1024**3
TMPFS_FINAL_FLOOR = 16 * 1024**3
LIVE_MEMORY_BREAKER = 32 * 1024**3
ROOT_CACHE_PATHS = (
    "/tmp/neff_cache",
    "/tmp/neuron_cache",
    "/home/ubuntu/.cache/neuron",
    "/home/ubuntu/.cache/torch_neuronx",
)
ARMS = {
    "G0": {
        "label": "fs-20260925-v2-a1-native5-g0-hybrid-observation-off-trn22-r1",
        "mode": "hybrid", "observation": False, "tmpfs_free_before": 55 * 1024**3,
        "tmpfs_free_after": 42 * 1024**3, "memory_before": 87 * 1024**3,
        "child": "g0-r1",
    },
    "G1": {
        "label": "fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn22-r1",
        "mode": "hybrid", "observation": True, "tmpfs_free_before": 42 * 1024**3,
        "tmpfs_free_after": 29 * 1024**3, "memory_before": 74 * 1024**3,
        "child": "g1-r1",
    },
    "G2": {
        "label": "fs-20260925-v2-a1-native5-g2-ordinary-observation-on-trn22-r1",
        "mode": "ordinary", "observation": True, "tmpfs_free_before": 29 * 1024**3,
        "tmpfs_free_after": TMPFS_FINAL_FLOOR, "memory_before": 61 * 1024**3,
        "child": "g2-r1",
    },
}
COMMON_ENVIRONMENT = {
    "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
    "NEURON_COMPETITION_R1_SEED": "42",
    "NEURON_LOGICAL_NC_CONFIG": "2",
    "PYTHONUNBUFFERED": "1",
}
COMMON_TRAIN_ARGV = [
    "--num-steps", "5", "--max-train-seconds", "1800", "--n-embd", "1024",
    "--ngram-ve", "--ngram-ve-layers=-1", "--ngram-ve-table-mult", "64",
    "--ngram-ve-dim", "128", "--ngram-ve-trigram", "--ngram-ve-opt", "rmsprop",
    "--ngram-ve-flat-lr", "--ngram-ve-bf16", "--ngram-ve-clip-exclude",
    "--no-qk-shift", "--compile-sdpa-direct", "--nki-local-conv",
    "--bf16-norm-output", "--pack-factor", "32", "--clip-after-reduce",
    "--seq-len", "1024", "--no-eval-public", "--matrix-lr", "0.015",
]
SOURCE_PINS = {
    "submissions/dense6x1024_public0992851/train.py":
        "9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae",
    "prepare.py": "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_hybrid.py":
        "908431d7e1da8a9110e6e6005de5ecbe3dba13d798bf2f37f2443404dd8d7d94",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_ordinary.py":
        "6ed69d2356c28e0dd8f88f494d988c0b87f7f86426a001945bfd958ed6f8f6d0",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/verify_model_checkpoints.py":
        "e1ead481f9786ec02a3c60d88e72820f084274200bccd1f7eeb16da108cb6e2e",
    "research/fresh_start_launch.py":
        "7620fea10bb51c54481b8347ec9e97ebd40ce8f5ee206dc3f203c9483c35ee2f",
    "research/fresh_start_guard.py":
        "ac6f7d23dbe998b1d9c8ae14e209c6ffc21fac5d927895b1b92b15dc493f6de6",
}


class ContractError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    fd = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                 "protected path is not a single-link regular file")
        while True:
            try:
                chunk = os.read(fd, 1024 * 1024)
            except InterruptedError:
                continue
            if not chunk:
                break
            digest.update(chunk)
        _require(os.fstat(fd) == before, "protected file changed during hash")
        return digest.hexdigest()
    finally:
        os.close(fd)


def arm_paths(arm: str) -> dict[str, str]:
    info = ARMS[arm]
    base = f"{TMPFS_PARENT}/{info['child']}"
    label = info["label"]
    token = label.replace("-", "_")
    return {
        "tmpfs_arm": base,
        "output": f"{base}/out",
        "compile_cache": f"{base}/cache",
        "checkpoint": f"{base}/out/final.pt",
        "remote": f"/home/ubuntu/trainium_freshstart_{token}",
        "bundle": f"/home/ubuntu/{label}.bundle",
        "root_report": f"/home/ubuntu/native-gate-evidence/{label}",
        "child_log": f"/home/ubuntu/native-gate-evidence/{label}.child.log",
        "runtime_job": f"/home/ubuntu/native-gate-evidence/{label}.job.json",
        "runtime_result": f"/home/ubuntu/native-gate-evidence/{label}.runtime-result.json",
        "dispatch": f"/home/ubuntu/native-gate-evidence/{label}.dispatch.json",
        "controller_request": f"/home/ubuntu/native-gate-evidence/{label}.request.json",
    }


def expected_train_argv(arm: str) -> list[str]:
    info = ARMS[arm]
    paths = arm_paths(arm)
    source = ("research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/"
              + ("train_gate_ordinary.py" if info["mode"] == "ordinary"
                 else "train_gate_hybrid.py"))
    observation = "--native-gate-observation" if info["observation"] else "--no-native-gate-observation"
    return [source, *COMMON_TRAIN_ARGV, "--out-dir", paths["output"],
            "--native-gate-arm", arm, observation,
            "--native-gate-label", info["label"], "--native-gate-host", HOST,
            "--native-gate-report-dir", paths["root_report"] + "/native-gate"]


def validate_request(request: dict[str, Any], arm: str) -> None:
    _require(arm in ARMS, "unknown arm")
    exact_keys = {
        "schema", "state", "executable", "arm", "host", "expected_hostname", "label",
        "paths", "environment", "train_argv", "world_size", "lnc", "seed",
        "optimizer_updates", "source_sha256", "storage", "lifecycle", "downstream_locks",
        "external_literals", "authorization_path", "proposal_path", "analysis",
    }
    _require(set(request) == exact_keys, "request schema mismatch")
    info = ARMS[arm]
    _require(request["schema"] == SCHEMA and request["arm"] == arm, "request version/arm mismatch")
    _require(request["state"] in {"pending-external-verification", "locked-prerequisite"},
             "invalid request state")
    _require(request["executable"] is False, "checked-in request must be non-executable")
    _require((request["host"], request["expected_hostname"], request["label"])
             == (HOST, HOSTNAME, info["label"]), "host/label mismatch")
    _require(request["paths"] == arm_paths(arm), "path contract mismatch")
    expected_env = {**COMMON_ENVIRONMENT,
                    "NEURON_COMPILE_CACHE_DIR": arm_paths(arm)["compile_cache"],
                    "TRAINIUM_FRESH_START_RUN_ID": info["label"]}
    _require(request["environment"] == expected_env, "environment mismatch")
    _require(request["train_argv"] == expected_train_argv(arm), "train argv mismatch")
    _require((request["world_size"], request["lnc"], request["seed"],
              request["optimizer_updates"]) == (4, 2, 42, 5), "topology/science mismatch")
    _require(request["source_sha256"] == SOURCE_PINS, "frozen source pins mismatch")
    expected_storage = {
        "root_breaker": ROOT_BREAKER, "root_operational_floor": ROOT_OPERATIONAL_FLOOR,
        "root_hard_abort": ROOT_HARD_ABORT, "root_log_cap": ROOT_LOG_CAP,
        "source_stage_allowance": SOURCE_STAGE_ALLOWANCE,
        "out_cap": OUT_CAP, "cache_cap": CACHE_CAP, "arm_cap": ARM_CAP,
        "tmpfs_free_before": info["tmpfs_free_before"],
        "tmpfs_free_after": info["tmpfs_free_after"],
        "tmpfs_final_floor": TMPFS_FINAL_FLOOR,
        "memory_before": info["memory_before"], "live_memory_breaker": LIVE_MEMORY_BREAKER,
        "root_cache_paths": list(ROOT_CACHE_PATHS),
    }
    _require(request["storage"] == expected_storage, "storage thresholds mismatch")
    _require(request["lifecycle"] == {
        "one_run_per_host": True, "private_process_group": True,
        "bounded_term_kill_reap_quiescence": True, "hash_only_after_quiescence": True,
        "retain_all_tmpfs_arms": True, "cleanup_authorized": False,
        "prune_authorized": False, "root_checkpoint_copy_authorized": False,
    }, "lifecycle mismatch")
    _require(request["external_literals"] == {
        "request_bearing_head": "PENDING_EXTERNAL_VERIFIER",
        "request_sha256": "PENDING_EXTERNAL_VERIFIER",
        "authorization_sha256": "PENDING_EXTERNAL_VERIFIER",
    }, "checked-in external literals must remain pending")
    if arm == "G0":
        _require(request["state"] == "pending-external-verification", "G0 must be pending")
        _require(request["downstream_locks"] == {"G1": True, "G2": True}, "G0 downstream locks missing")
    else:
        _require(request["state"] == "locked-prerequisite", "downstream request must be locked")
        _require(request["downstream_locks"].get(arm) is True, "arm lock missing")


def parse_mountinfo(text: str, target: str) -> dict[str, Any]:
    matches = []
    for line in text.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            raise ContractError("malformed mountinfo")
        fields, post = left.split(), right.split()
        if len(fields) < 6 or len(post) < 3:
            raise ContractError("malformed mountinfo fields")
        if fields[4] == target:
            matches.append({
                "mount_id": int(fields[0]), "parent_id": int(fields[1]),
                "device": fields[2], "root": fields[3], "target": fields[4],
                "mount_options": sorted(fields[5].split(",")), "optional": fields[6:],
                "fs_type": post[0], "source": post[1],
                "super_options": sorted(post[2].split(",")),
            })
    _require(len(matches) == 1, f"expected one mount record for {target}")
    return matches[0]


def validate_mounts(root: dict[str, Any], tmpfs: dict[str, Any]) -> None:
    _require(root["target"] == "/" and tmpfs["target"] == "/dev/shm", "mount target mismatch")
    _require(root["mount_id"] != tmpfs["mount_id"] and root["device"] != tmpfs["device"],
             "root and tmpfs must be distinct")
    _require(tmpfs["fs_type"] == "tmpfs", "/dev/shm is not tmpfs")
    options = set(tmpfs["mount_options"]) | set(tmpfs["super_options"])
    _require({"rw", "nosuid", "nodev"} <= options and "noexec" not in options,
             "tmpfs mount options mismatch")


def validate_resources(arm: str, sample: dict[str, Any], *, phase: str) -> None:
    info = ARMS[arm]
    _require(sample.get("swap_total_bytes") == 0 and sample.get("swap_used_bytes") == 0,
             "swap is configured or active")
    memory_floor = info["memory_before"] if phase == "pre" else LIVE_MEMORY_BREAKER
    _require(type(sample.get("mem_available_bytes")) is int
             and sample["mem_available_bytes"] >= memory_floor, "memory floor failed")
    _require(type(sample.get("root_free_bytes")) is int
             and sample["root_free_bytes"] >= ROOT_BREAKER, "root breaker failed")
    tmp_floor = info["tmpfs_free_before"] if phase == "pre" else info["tmpfs_free_after"]
    _require(type(sample.get("tmpfs_free_bytes")) is int
             and sample["tmpfs_free_bytes"] >= tmp_floor, "tmpfs floor failed")


def validate_live_sample(arm: str, sample: dict[str, Any],
                         baseline_root_caches: dict[str, Any]) -> None:
    """Validate one bounded watcher sample while the owned group is live."""
    _require(sample.get("root_caches") == baseline_root_caches,
             "root cache snapshot changed during live run")
    _require(sorted(sample.get("root_caches", {})) == sorted(ROOT_CACHE_PATHS),
             "live root cache enumeration incomplete")
    _require(sample.get("swap_total_bytes") == 0 and sample.get("swap_used_bytes") == 0,
             "swap appeared during live run")
    _require(type(sample.get("mem_available_bytes")) is int
             and sample["mem_available_bytes"] >= LIVE_MEMORY_BREAKER,
             "live memory breaker failed")
    _require(type(sample.get("root_free_bytes")) is int
             and sample["root_free_bytes"] >= ROOT_BREAKER, "live root breaker failed")
    _require(type(sample.get("tmpfs_free_bytes")) is int
             and sample["tmpfs_free_bytes"] >= ARMS[arm]["tmpfs_free_after"],
             "live tmpfs floor failed")
    _require(type(sample.get("root_log_allocated")) is int
             and sample["root_log_allocated"] <= ROOT_LOG_CAP, "live root log cap failed")
    _require(type(sample.get("cache_allocated")) is int
             and sample["cache_allocated"] <= CACHE_CAP, "live cache cap failed")
    _require(type(sample.get("output_allocated")) is int
             and sample["output_allocated"] <= OUT_CAP, "live output cap failed")


def run_watcher(arm: str, samples: list[dict[str, Any]],
                baseline_root_caches: dict[str, Any], terminate) -> dict[str, Any]:
    """Deterministically exercise watcher policy; runtime supplies sampling cadence.

    On the first violation the caller-provided reviewed process-group terminator is
    invoked and must prove TERM/KILL/reap quiescence.  This function never launches.
    """
    for index, sample in enumerate(samples):
        try:
            validate_live_sample(arm, sample, baseline_root_caches)
        except ContractError as exc:
            lifecycle = terminate()
            _require(isinstance(lifecycle, dict)
                     and lifecycle.get("term_sent") is True
                     and lifecycle.get("reaped") is True
                     and lifecycle.get("quiescent") is True,
                     "bounded termination did not prove quiescence")
            message = str(exc)
            status = ("CACHE_ROUTING_INVALID" if "cache" in message
                      else "RUNTIME_STORAGE_INVALID")
            return {"status": status, "failed_sample": index, "error": message,
                    "lifecycle": lifecycle}
    return {"status": "PASS", "sample_count": len(samples)}


def _entry(info: os.stat_result, kind: str) -> dict[str, Any]:
    return {
        "kind": kind, "device": info.st_dev, "inode": info.st_ino,
        "mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid, "gid": info.st_gid,
        "nlink": info.st_nlink, "size": info.st_size, "allocated": info.st_blocks * 512,
        "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
    }


def snapshot_tree(path: Path) -> dict[str, Any]:
    """Capture anchored no-follow recursive metadata without reading cache bytes."""
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    root_fd = os.open(path, flags)
    root_info = os.fstat(root_fd)
    _require(stat.S_ISDIR(root_info.st_mode), "snapshot root is not a directory")
    entries: dict[str, dict[str, Any]] = {".": _entry(root_info, "directory")}

    def walk(fd: int, prefix: str) -> None:
        for name in sorted(os.listdir(fd)):
            _require(name not in (".", "..") and "/" not in name, "unsafe tree member")
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(info.st_mode):
                child = os.open(name, flags, dir_fd=fd)
                try:
                    opened = os.fstat(child)
                    _require((opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino),
                             "directory changed while opening")
                    entries[relative] = _entry(opened, "directory")
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                _require(info.st_nlink == 1, "cache member has multiple links")
                entries[relative] = _entry(info, "file")
            else:
                raise ContractError("cache tree contains non-file/non-directory entry")

    try:
        walk(root_fd, "")
        final = os.fstat(root_fd)
        _require((root_info.st_dev, root_info.st_ino, root_info.st_mtime_ns,
                  root_info.st_ctime_ns) == (final.st_dev, final.st_ino,
                                             final.st_mtime_ns, final.st_ctime_ns),
                 "cache root changed during snapshot")
    finally:
        os.close(root_fd)
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": os.fspath(path), "entries": entries,
        "allocated": sum(item["allocated"] for item in entries.values()),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def validate_initial_absence(trusted_root: Path, target: Path) -> None:
    root = Path(os.path.abspath(trusted_root))
    absolute = Path(os.path.abspath(target))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ContractError("path escapes trusted root") from exc
    _require(relative.parts and all(part not in ("", ".", "..") for part in relative.parts),
             "unsafe relative path")
    cursor = root
    root_info = os.lstat(cursor)
    _require(stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode),
             "trusted root is not a real directory")
    for part in relative.parts:
        cursor /= part
        try:
            info = os.lstat(cursor)
        except FileNotFoundError:
            return
        _require(not stat.S_ISLNK(info.st_mode), "path has symlink component")
    raise ContractError("authorized fresh path already exists")


def adjudicate_storage(arm: str, pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    """Return a machine verdict; any cache ambiguity is CACHE_ROUTING_INVALID."""
    try:
        _require(pre.get("host") == HOST and post.get("host") == HOST, "host mismatch")
        _require(pre.get("hostname") == HOSTNAME and post.get("hostname") == HOSTNAME,
                 "hostname mismatch")
        _require(pre.get("mounts") == post.get("mounts"), "mount identity changed")
        validate_mounts(post["mounts"]["root"], post["mounts"]["tmpfs"])
        validate_resources(arm, pre["resources"], phase="pre")
        validate_resources(arm, post["resources"], phase="post")
        _require(pre.get("fresh_paths_absent") is True, "fresh arm paths were not absent")
        expected_prior = [] if arm == "G0" else (["G0"] if arm == "G1" else ["G0", "G1"])
        _require(sorted(pre.get("retained_arms", {})) == expected_prior,
                 "retained prior-arm inventory missing")
        _require(pre.get("retained_arms", {}) == post.get("retained_arms", {}),
                 "retained prior arm changed")
        _require(pre.get("root_caches") == post.get("root_caches"),
                 "root cache snapshot changed")
        _require(sorted(post.get("root_caches", {})) == sorted(ROOT_CACHE_PATHS),
                 "root cache enumeration incomplete")
        cache = post.get("declared_cache", {})
        output = post.get("output", {})
        _require(cache.get("path") == arm_paths(arm)["compile_cache"], "cache path mismatch")
        _require(type(cache.get("allocated")) is int and 4096 < cache["allocated"] <= CACHE_CAP,
                 "declared tmpfs cache use is not positively proved")
        _require(cache.get("mount_device") == post["mounts"]["tmpfs"]["device"],
                 "cache is not on tmpfs")
        _require(output.get("path") == arm_paths(arm)["output"], "output path mismatch")
        _require(type(output.get("allocated")) is int and 0 < output["allocated"] <= OUT_CAP,
                 "output cap failed")
        _require(output.get("mount_device") == post["mounts"]["tmpfs"]["device"],
                 "output is not on tmpfs")
        _require(type(post.get("root_log_allocated")) is int
                 and post["root_log_allocated"] <= ROOT_LOG_CAP, "root log cap failed")
        _require(type(post.get("root_allowed_growth_bytes")) is int
                 and 0 <= post["root_allowed_growth_bytes"] <= ROOT_LOG_CAP,
                 "root allowlisted growth cap failed")
        _require(post.get("root_unallowlisted_growth_bytes") == 0,
                 "unallowlisted root growth detected")
        _require(pre["resources"]["root_free_bytes"] - post["resources"]["root_free_bytes"]
                 <= ROOT_LOG_CAP, "net root growth cap failed")
        checkpoint = post.get("checkpoint", {})
        _require(checkpoint.get("path") == arm_paths(arm)["checkpoint"], "checkpoint path mismatch")
        _require(checkpoint.get("mount_device") == post["mounts"]["tmpfs"]["device"],
                 "checkpoint is not on tmpfs")
        _require(checkpoint.get("regular") is True and checkpoint.get("nlink") == 1,
                 "checkpoint identity failed")
        _require(checkpoint.get("same_fd_stable") is True
                 and checkpoint.get("weights_only") is True
                 and checkpoint.get("model_only") is True
                 and checkpoint.get("finite") is True
                 and checkpoint.get("strict_load") is True
                 and checkpoint.get("logits_exact") is True
                 and checkpoint.get("causality_exact") is True,
                 "checkpoint scientific verification failed")
        _require(post.get("process_quiescent") is True and post.get("device_idle") is True,
                 "process/device is not quiescent")
        return {"schema": "a1-native-five-trn22-storage-verdict-v1", "arm": arm,
                "status": "PASS", "g1_unlocked": arm == "G0", "cleanup_authorized": False}
    except (ContractError, KeyError, TypeError) as exc:
        message = str(exc)
        cache_terms = ("cache", "declared tmpfs", "root cache")
        status = "CACHE_ROUTING_INVALID" if any(term in message for term in cache_terms) else "INVALID"
        return {"schema": "a1-native-five-trn22-storage-verdict-v1", "arm": arm,
                "status": status, "error": f"{type(exc).__name__}: {exc}",
                "g1_unlocked": False, "cleanup_authorized": False}
