#!/usr/bin/env python3
"""Fail-closed storage and identity contract for the trn22 native-five gate.

This module does not launch work.  A later independently authorized runtime may
use it to capture pre/post storage evidence and adjudicate cache routing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


SCHEMA = "a1-native-five-trn22-tmpfs-request-v2"
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
ROOT_INODE_FLOOR = 10_000
TMPFS_INODE_FLOOR = 10_000
WATCH_INTERVAL_MAX_NS = 15_000_000_000
ROOT_REQUIRED_BEFORE = {
    "G0": 7_849_426_944,
    "G1": 7_648_870_400,
    "G2": 7_448_313_856,
}
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
OBSERVER_SHA256 = "8f0fcd21282c54f2747b0260a392363a15825adf3c1f2a1c3357fbaf8790bacc"
GATE_SOURCE_SHA256 = {
    "G0": SOURCE_PINS["research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_hybrid.py"],
    "G1": SOURCE_PINS["research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_hybrid.py"],
    "G2": SOURCE_PINS["research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_ordinary.py"],
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
        "observation_receipt":
            f"/home/ubuntu/native-gate-evidence/{label}.tree-observation.json",
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
        "root_free_required_before": ROOT_REQUIRED_BEFORE[arm],
        "root_inode_floor": ROOT_INODE_FLOOR, "tmpfs_inode_floor": TMPFS_INODE_FLOOR,
        "watch_interval_max_ns": WATCH_INTERVAL_MAX_NS,
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
            major, minor = (int(part) for part in fields[2].split(":"))
            target_info = os.stat(target, follow_symlinks=False)
            matches.append({
                "mount_id": int(fields[0]), "parent_id": int(fields[1]),
                "device": fields[2], "st_dev": os.makedev(major, minor),
                "root": fields[3], "target": fields[4],
                "mount_options": sorted(fields[5].split(",")), "optional": fields[6:],
                "fs_type": post[0], "source": post[1],
                "super_options": sorted(post[2].split(",")),
                "target_identity": _entry(target_info, "directory"),
            })
    _require(len(matches) == 1, f"expected one mount record for {target}")
    return matches[0]


def read_mountinfo_record(target: str) -> dict[str, Any]:
    """Capture one live mount record from the current process namespace."""
    fd = os.open("/proc/self/mountinfo", os.O_RDONLY | os.O_CLOEXEC)
    try:
        payload = bytearray()
        while True:
            try:
                chunk = os.read(fd, 64 * 1024)
            except InterruptedError:
                continue
            if not chunk:
                break
            payload.extend(chunk)
            _require(len(payload) <= 4 * 1024 * 1024, "mountinfo unexpectedly large")
    finally:
        os.close(fd)
    try:
        text = bytes(payload).decode()
    except UnicodeDecodeError as exc:
        raise ContractError("mountinfo is not UTF-8") from exc
    return parse_mountinfo(text, target)


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MOUNT_KEYS = {"mount_id", "parent_id", "device", "st_dev", "root", "target",
              "mount_options", "optional", "fs_type", "source", "super_options",
              "target_identity"}
RESOURCE_KEYS = {"schema", "phase", "root_free_bytes", "root_free_inodes",
                 "tmpfs_free_bytes", "tmpfs_free_inodes", "mem_available_bytes",
                 "swap_total_bytes", "swap_used_bytes"}
ENTRY_KEYS = {"kind", "device", "inode", "mode", "uid", "gid", "nlink",
              "size", "allocated", "mtime_ns", "ctime_ns", "content_sha256"}
TREE_KEYS = {"schema", "path", "state", "mount_id", "mount_device",
             "mount_root_identity_sha256", "ancestors", "ancestor_mount_ids",
             "entries", "entry_count", "logical_bytes", "allocated_bytes",
             "manifest_sha256", "complete"}
CHECKPOINT_KEYS = {"schema", "path", "mount_id", "mount_device", "device", "inode",
                   "kind", "mode", "uid", "gid", "nlink", "bytes", "allocated",
                   "mtime_ns", "ctime_ns", "sha256", "weights_only", "model_only",
                   "finite", "strict_load", "logits_exact", "causality_exact",
                   "same_fd_stable", "model_state_sha256", "eval_logits_sha256",
                   "causal_digest_sha256", "source_sha256", "observer_sha256",
                   "rank_report_sha256", "job_sha256", "postflight_sha256",
                   "parent_identity_sha256", "mount_root_identity_sha256", "complete"}
ARM_EVIDENCE_KEYS = {"schema", "arm", "host", "hostname", "label", "source_sha256",
                     "protected_source_sha256", "observer_sha256", "train_argv_sha256",
                     "environment_sha256",
                     "job_sha256", "postflight_sha256", "rank_report_sha256", "output",
                     "cache", "checkpoint", "root_log_allocated", "process_quiescent",
                     "device_idle", "watcher", "complete"}
WATCH_SAMPLE_KEYS = {"schema", "sequence", "monotonic_ns", "phase", "root_free_bytes",
                     "root_free_inodes", "tmpfs_free_bytes", "tmpfs_free_inodes",
                     "mem_available_bytes", "swap_total_bytes", "swap_used_bytes",
                     "root_log_allocated", "cache_allocated", "output_allocated",
                     "root_cache_manifest_sha256", "current_arm_manifest_sha256",
                     "sample_sha256"}
WATCH_COVERAGE_KEYS = {"schema", "interval_limit_ns", "child_start_ns", "child_exit_ns",
                       "final_seal_ns", "complete"}
WATCH_RECEIPT_KEYS = {"schema", "status", "arm", "host", "label", "source_sha256",
                      "observer_sha256", "job_sha256", "postflight_sha256",
                      "baseline_root_caches_sha256", "sample_count",
                      "first_monotonic_ns", "last_monotonic_ns", "series_sha256",
                      "coverage_sha256", "samples", "coverage", "complete"}
LOCK_RECEIPT_KEYS = {"schema", "host", "label", "path", "owner_uid", "owner_pid",
                     "owner_start_ticks", "held", "complete"}
PROCESS_GROUP_RECEIPT_KEYS = {"schema", "label", "pgid", "term_sent", "kill_sent",
                              "reaped", "quiescent", "owned_processes",
                              "remaining_pids", "complete"}
PROCESS_IDENTITY_KEYS = {"pid", "start_ticks", "command_sha256", "exit_code"}
DEVICE_POSTFLIGHT_KEYS = {"schema", "host", "hostname", "device", "active_processes",
                          "unknown_holders", "device_idle", "complete"}
OBSERVATION_CONTEXT_KEYS = {
    "schema", "arm", "phase", "host", "hostname", "label", "source_sha256",
    "request_sha256", "job_sha256", "postflight_sha256", "lock_receipt",
    "lock_receipt_sha256", "process_group_receipt", "process_group_receipt_sha256",
    "device_postflight_receipt", "device_postflight_receipt_sha256",
    "watcher_receipt_sha256", "mount_record_sha256", "mount_root_identity_sha256",
    "checkpoint_receipt_sha256", "observed_monotonic_ns", "complete",
}
TREE_OBSERVATION_RECEIPT_KEYS = {
    "schema", "statement", "arm", "host", "hostname", "label", "tree_path",
    "tree_manifest_sha256", "tree_snapshot_sha256", "source_sha256", "request_sha256",
    "job_sha256", "postflight_sha256", "lock_receipt_sha256",
    "process_group_receipt_sha256", "device_postflight_receipt_sha256",
    "watcher_receipt_sha256", "mount_record_sha256", "mount_root_identity_sha256",
    "checkpoint_receipt_sha256", "observation_context_sha256", "complete",
}
LOCK_PATH = "/home/ubuntu/.cache/trainium_freshstart/device0.lock"
OBSERVATION_STATEMENT = "authenticated-observation-not-perpetual-current-state"


def _exact_dict(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == keys, f"{label} exact schema mismatch")
    return value


def _sha(value: Any) -> bool:
    return type(value) is str and SHA_RE.fullmatch(value) is not None


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()).hexdigest()


def _validate_mount_record(record: Any, target: str) -> dict[str, Any]:
    record = _exact_dict(record, MOUNT_KEYS, f"mount {target}")
    for key in ("mount_id", "parent_id", "st_dev"):
        _require(type(record[key]) is int and record[key] >= 0, f"mount {key} type")
    for key in ("device", "root", "target", "fs_type", "source"):
        _require(type(record[key]) is str and record[key], f"mount {key} type")
    for key in ("mount_options", "optional", "super_options"):
        _require(type(record[key]) is list and all(type(item) is str for item in record[key]),
                 f"mount {key} type")
    _require(record["target"] == target, "mount target mismatch")
    identity = _validate_entry(record["target_identity"], f"mount {target} target")
    _require(identity["kind"] == "directory" and identity["device"] == record["st_dev"],
             "mount target identity/device mismatch")
    return record


def validate_mounts(root: dict[str, Any], tmpfs: dict[str, Any]) -> None:
    root = _validate_mount_record(root, "/")
    tmpfs = _validate_mount_record(tmpfs, "/dev/shm")
    _require(root["mount_id"] != tmpfs["mount_id"] and root["st_dev"] != tmpfs["st_dev"]
             and root["device"] != tmpfs["device"], "root and tmpfs must be distinct")
    _require(tmpfs["fs_type"] == "tmpfs", "/dev/shm is not tmpfs")
    options = set(tmpfs["mount_options"]) | set(tmpfs["super_options"])
    _require({"rw", "nosuid", "nodev"} <= options and "noexec" not in options,
             "tmpfs mount options mismatch")


def validate_resources(arm: str, sample: dict[str, Any], *, phase: str) -> None:
    sample = _exact_dict(sample, RESOURCE_KEYS, "resource sample")
    _require(phase in {"pre", "post"} and sample["phase"] == phase, "resource phase mismatch")
    for key in RESOURCE_KEYS - {"schema", "phase"}:
        _require(type(sample[key]) is int and sample[key] >= 0, f"resource {key} type")
    _require(sample["schema"] == "a1-native-five-trn22-resource-v2", "resource schema version")
    _require(sample["swap_total_bytes"] == 0 and sample["swap_used_bytes"] == 0,
             "swap is configured or active")
    memory_floor = ARMS[arm]["memory_before"] if phase == "pre" else LIVE_MEMORY_BREAKER
    root_floor = ROOT_REQUIRED_BEFORE[arm] if phase == "pre" else ROOT_BREAKER
    tmp_floor = ARMS[arm]["tmpfs_free_before"] if phase == "pre" else ARMS[arm]["tmpfs_free_after"]
    _require(sample["mem_available_bytes"] >= memory_floor, "memory floor failed")
    _require(sample["root_free_bytes"] >= root_floor, "reserved root floor failed")
    _require(sample["tmpfs_free_bytes"] >= tmp_floor, "tmpfs floor failed")
    _require(sample["root_free_inodes"] >= ROOT_INODE_FLOOR, "root inode floor failed")
    _require(sample["tmpfs_free_inodes"] >= TMPFS_INODE_FLOOR, "tmpfs inode floor failed")


def _entry(info: os.stat_result, kind: str,
           content_sha256: str | None = None) -> dict[str, Any]:
    return {"kind": kind, "device": info.st_dev, "inode": info.st_ino,
            "mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid, "gid": info.st_gid,
            "nlink": info.st_nlink, "size": info.st_size, "allocated": info.st_blocks * 512,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
            "content_sha256": content_sha256}


def _identity_tuple(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _dir_flags() -> int:
    required = ("O_CLOEXEC", "O_NOFOLLOW", "O_DIRECTORY")
    _require(all(hasattr(os, item) for item in required), "platform lacks anchored directory flags")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY


def _file_flags() -> int:
    _require(hasattr(os, "O_CLOEXEC") and hasattr(os, "O_NOFOLLOW"),
             "platform lacks no-follow file flags")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def _fd_mount_id(fd: int) -> int:
    info_fd = os.open(f"/proc/self/fdinfo/{fd}", os.O_RDONLY | os.O_CLOEXEC)
    try:
        payload = bytearray()
        while True:
            try:
                chunk = os.read(info_fd, 4096)
            except InterruptedError:
                continue
            if not chunk:
                break
            payload.extend(chunk)
            _require(len(payload) <= 64 * 1024, "fdinfo unexpectedly large")
    finally:
        os.close(info_fd)
    values = [line.split(b":", 1)[1].strip() for line in bytes(payload).splitlines()
              if line.startswith(b"mnt_id:")]
    _require(len(values) == 1 and values[0].isdigit(), "fd mount id unavailable")
    return int(values[0])


def _open_dir_chain(path: Path) -> tuple[
        list[int], list[str], list[tuple[int, ...]], list[int]]:
    absolute = Path(os.path.abspath(path))
    _require(absolute.is_absolute(), "anchored path must be absolute")
    fds: list[int] = []
    names: list[str] = []
    identities: list[tuple[int, ...]] = []
    mount_ids: list[int] = []
    try:
        root_fd = os.open("/", _dir_flags())
        fds.append(root_fd)
        identities.append(_identity_tuple(os.fstat(root_fd)))
        mount_ids.append(_fd_mount_id(root_fd))
        parent_fd = root_fd
        for component in absolute.parts[1:]:
            _require(component not in ("", ".", "..") and "/" not in component,
                     "unsafe anchored component")
            child_fd = os.open(component, _dir_flags(), dir_fd=parent_fd)
            opened = os.fstat(child_fd)
            named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            _require(_identity_tuple(opened) == _identity_tuple(named),
                     "ancestor name/inode changed during open")
            fds.append(child_fd)
            names.append(component)
            identities.append(_identity_tuple(opened))
            mount_ids.append(_fd_mount_id(child_fd))
            parent_fd = child_fd
        return fds, names, identities, mount_ids
    except Exception:
        for fd in reversed(fds):
            os.close(fd)
        raise


def _revalidate_chain(fds: list[int], names: list[str], identities: list[tuple[int, ...]],
                      mount_ids: list[int]) -> None:
    _require(len(fds) == len(identities) == len(mount_ids)
             and len(names) + 1 == len(fds), "anchored chain cardinality mismatch")
    _require(_identity_tuple(os.fstat(fds[0])) == identities[0], "root identity changed")
    _require(_fd_mount_id(fds[0]) == mount_ids[0], "root mount changed")
    for index, name in enumerate(names):
        opened = os.fstat(fds[index + 1])
        named = os.stat(name, dir_fd=fds[index], follow_symlinks=False)
        reopened_fd = os.open(name, _dir_flags(), dir_fd=fds[index])
        try:
            reopened = os.fstat(reopened_fd)
            _require(_identity_tuple(opened) == identities[index + 1]
                     and _identity_tuple(named) == identities[index + 1]
                     and _identity_tuple(reopened) == identities[index + 1]
                     and _fd_mount_id(fds[index + 1]) == mount_ids[index + 1]
                     and _fd_mount_id(reopened_fd) == mount_ids[index + 1],
                     "anchored ancestor changed")
        finally:
            os.close(reopened_fd)


def _mount_boundary(identities: list[tuple[int, ...]], mount_ids: list[int],
                    expected_device: int, expected_mount_id: int) -> int:
    _require(len(identities) == len(mount_ids), "mount boundary cardinality mismatch")
    matches = [index for index, (identity, current_mount_id) in
               enumerate(zip(identities, mount_ids))
               if identity[0] == expected_device and current_mount_id == expected_mount_id]
    _require(matches, "target mount boundary was not reached")
    boundary = matches[0]
    _require(all(not (identity[0] == expected_device
                     and current_mount_id == expected_mount_id)
                 for identity, current_mount_id in
                 zip(identities[:boundary], mount_ids[:boundary])),
             "target mount appeared before boundary")
    _require(all(identity[0] == expected_device and current_mount_id == expected_mount_id
                 for identity, current_mount_id in
                 zip(identities[boundary:], mount_ids[boundary:])),
             "mount changed below authenticated boundary")
    return boundary


def _tree_manifest_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: snapshot[key] for key in
            ("schema", "path", "state", "mount_id", "mount_device",
             "mount_root_identity_sha256", "ancestors", "ancestor_mount_ids", "entries",
             "entry_count", "logical_bytes", "allocated_bytes", "complete")}


def _validate_entry(value: Any, label: str) -> dict[str, Any]:
    value = _exact_dict(value, ENTRY_KEYS, label)
    _require(value["kind"] in {"directory", "file"}, f"{label} kind")
    for key in ENTRY_KEYS - {"kind", "content_sha256"}:
        _require(type(value[key]) is int and value[key] >= 0, f"{label} {key} type")
    _require(value["nlink"] >= 1, f"{label} link count")
    if value["kind"] == "file":
        _require(value["nlink"] == 1, f"{label} file hardlink")
        _require(_sha(value["content_sha256"]), f"{label} content digest")
    else:
        _require(value["content_sha256"] is None, f"{label} directory content digest")
    return value


def validate_tree_snapshot(snapshot: Any, expected_path: str, mount_id: int,
                           mount_device: int,
                           mount_root_identity_sha256: str) -> dict[str, Any]:
    snapshot = _exact_dict(snapshot, TREE_KEYS, "tree snapshot")
    _require(snapshot["schema"] == "a1-native-five-anchored-tree-v2"
             and snapshot["state"] in {"present", "absent"} and snapshot["complete"] is True,
             "tree snapshot state/version mismatch")
    _require(snapshot["path"] == expected_path and snapshot["mount_id"] == mount_id
             and snapshot["mount_device"] == mount_device, "tree snapshot path/mount mismatch")
    _require(_sha(mount_root_identity_sha256)
             and snapshot["mount_root_identity_sha256"] == mount_root_identity_sha256,
             "tree mount-root identity mismatch")
    _require(type(snapshot["ancestors"]) is list and snapshot["ancestors"],
             "tree ancestor chain missing")
    _require(type(snapshot["ancestor_mount_ids"]) is list
             and len(snapshot["ancestor_mount_ids"]) == len(snapshot["ancestors"])
             and all(type(item) is int and item >= 0
                     for item in snapshot["ancestor_mount_ids"]),
             "tree ancestor mount identities missing")
    for index, entry in enumerate(snapshot["ancestors"]):
        _validate_entry(entry, f"ancestor {index}")
        _require(entry["kind"] == "directory", "ancestor is not a directory")
    identities = [(entry["device"], entry["inode"], entry["mode"], entry["nlink"],
                   entry["uid"], entry["gid"], entry["size"], entry["mtime_ns"],
                   entry["ctime_ns"]) for entry in snapshot["ancestors"]]
    boundary = _mount_boundary(identities, snapshot["ancestor_mount_ids"],
                               mount_device, mount_id)
    _require(snapshot["mount_root_identity_sha256"]
             == canonical_sha(snapshot["ancestors"][boundary]),
             "tree mount-root boundary identity mismatch")
    if snapshot["state"] == "absent":
        _require(snapshot["entries"] == {} and snapshot["entry_count"] == 0
                 and snapshot["logical_bytes"] == 0 and snapshot["allocated_bytes"] == 0,
                 "absent tree contains entries or bytes")
        _require(_sha(snapshot["manifest_sha256"])
                 and snapshot["manifest_sha256"]
                 == canonical_sha(_tree_manifest_payload(snapshot)),
                 "absent tree manifest digest mismatch")
        return snapshot
    _require(type(snapshot["entries"]) is dict and "." in snapshot["entries"],
             "tree entries missing root")
    for relative, entry in snapshot["entries"].items():
        _require(type(relative) is str and relative
                 and (relative == "." or (not relative.startswith("/")
                                          and ".." not in relative.split("/"))),
                 "unsafe tree entry path")
        _validate_entry(entry, f"tree entry {relative}")
        _require(entry["device"] == mount_device, "tree entry device mismatch")
    _require(snapshot["entries"]["."]["kind"] == "directory", "tree root kind mismatch")
    _require(type(snapshot["entry_count"]) is int
             and snapshot["entry_count"] == len(snapshot["entries"]), "tree count mismatch")
    logical = sum(item["size"] for item in snapshot["entries"].values()
                  if item["kind"] == "file")
    allocated = sum(item["allocated"] for item in snapshot["entries"].values())
    _require(snapshot["logical_bytes"] == logical and snapshot["allocated_bytes"] == allocated,
             "tree byte accounting mismatch")
    _require(_sha(snapshot["manifest_sha256"])
             and snapshot["manifest_sha256"] == canonical_sha(_tree_manifest_payload(snapshot)),
             "tree manifest digest mismatch")
    return snapshot


def snapshot_path(path: Path, *, mount_id: int, expected_device: int) -> dict[str, Any]:
    """Snapshot a present tree or prove absence after crossing a pinned mount boundary."""
    absolute = Path(os.path.abspath(path))
    fds: list[int] = []
    names: list[str] = []
    identities: list[tuple[int, ...]] = []
    mount_ids: list[int] = []
    try:
        root_fd = os.open("/", _dir_flags())
        fds.append(root_fd)
        root_info = os.fstat(root_fd)
        identities.append(_identity_tuple(root_info))
        mount_ids.append(_fd_mount_id(root_fd))
        parent_fd = root_fd
        for component in absolute.parts[1:]:
            _require(component not in ("", ".", "..") and "/" not in component,
                     "unsafe snapshot component")
            try:
                child = os.open(component, _dir_flags(), dir_fd=parent_fd)
            except FileNotFoundError:
                boundary = _mount_boundary(identities, mount_ids,
                                           expected_device, mount_id)
                _revalidate_chain(fds, names, identities, mount_ids)
                result = {
                    "schema": "a1-native-five-anchored-tree-v2",
                    "path": os.fspath(absolute), "state": "absent",
                    "mount_id": mount_id, "mount_device": expected_device,
                    "mount_root_identity_sha256": canonical_sha(
                        _entry(os.fstat(fds[boundary]), "directory")),
                    "ancestors": [_entry(os.fstat(fd), "directory") for fd in fds],
                    "ancestor_mount_ids": list(mount_ids),
                    "entries": {}, "entry_count": 0, "logical_bytes": 0,
                    "allocated_bytes": 0, "complete": True,
                }
                result["manifest_sha256"] = canonical_sha(_tree_manifest_payload(result))
                _revalidate_chain(fds, names, identities, mount_ids)
                return validate_tree_snapshot(
                    result, os.fspath(absolute), mount_id, expected_device,
                    result["mount_root_identity_sha256"])
            except OSError as exc:
                raise ContractError("snapshot ancestor is not a stable directory") from exc
            opened = os.fstat(child)
            named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            _require(_identity_tuple(opened) == _identity_tuple(named),
                     "snapshot ancestor identity changed")
            fds.append(child)
            names.append(component)
            identities.append(_identity_tuple(opened))
            mount_ids.append(_fd_mount_id(child))
            parent_fd = child
        _mount_boundary(identities, mount_ids, expected_device, mount_id)
        _revalidate_chain(fds, names, identities, mount_ids)
        return snapshot_tree(
            absolute, mount_id=mount_id, expected_device=expected_device,
            _chain=(fds, names, identities, mount_ids))
    finally:
        for fd in reversed(fds):
            os.close(fd)


def snapshot_tree(path: Path, *, mount_id: int, expected_device: int,
                  _chain: tuple[list[int], list[str], list[tuple[int, ...]],
                                list[int]] | None = None) -> dict[str, Any]:
    """Capture a bottom-up resealed no-follow tree through pinned ancestor dirfds."""
    owns_chain = _chain is None
    if _chain is None:
        fds, names, chain_identities, chain_mount_ids = _open_dir_chain(path)
    else:
        fds, names, chain_identities, chain_mount_ids = _chain
    root_fd = fds[-1]
    root_info = os.fstat(root_fd)
    try:
        boundary = _mount_boundary(chain_identities, chain_mount_ids,
                                   expected_device, mount_id)
        _revalidate_chain(fds, names, chain_identities, chain_mount_ids)
        _require(root_info.st_dev == expected_device and _fd_mount_id(root_fd) == mount_id,
                 "snapshot root mount identity mismatch")
        entries: dict[str, dict[str, Any]] = {".": _entry(root_info, "directory")}

        def walk(fd: int, prefix: str, parent_fd: int | None,
                 own_name: str | None, initial: os.stat_result) -> None:
            names_before = sorted(os.listdir(fd))
            for name in names_before:
                _require(name not in (".", "..") and "/" not in name, "unsafe tree member")
                named = os.stat(name, dir_fd=fd, follow_symlinks=False)
                relative = f"{prefix}/{name}" if prefix else name
                if stat.S_ISDIR(named.st_mode):
                    child = os.open(name, _dir_flags(), dir_fd=fd)
                    try:
                        opened = os.fstat(child)
                        _require(_identity_tuple(opened) == _identity_tuple(named),
                                 "nested directory changed during open")
                        _require(opened.st_dev == expected_device, "nested mount/device changed")
                        _require(_fd_mount_id(child) == mount_id, "nested mount-id changed")
                        entries[relative] = _entry(opened, "directory")
                        walk(child, relative, fd, name, opened)
                        _require(_identity_tuple(os.stat(name, dir_fd=fd, follow_symlinks=False))
                                 == _identity_tuple(opened), "nested directory rebound")
                    finally:
                        os.close(child)
                elif stat.S_ISREG(named.st_mode):
                    child = os.open(name, _file_flags(), dir_fd=fd)
                    try:
                        opened = os.fstat(child)
                        _require(_identity_tuple(opened) == _identity_tuple(named),
                                 "nested file changed during open")
                        _require(opened.st_dev == expected_device and opened.st_nlink == 1,
                                 "nested file device/link mismatch")
                        _require(_fd_mount_id(child) == mount_id, "nested file mount-id changed")
                        digest = hashlib.sha256()
                        remaining = opened.st_size
                        while remaining:
                            try:
                                chunk = os.read(child, min(1024 * 1024, remaining))
                            except InterruptedError:
                                continue
                            _require(bool(chunk), "nested file truncated during snapshot")
                            digest.update(chunk)
                            remaining -= len(chunk)
                        _require(not os.read(child, 1), "nested file appended during snapshot")
                        entries[relative] = _entry(opened, "file", digest.hexdigest())
                        _require(_identity_tuple(os.fstat(child)) == _identity_tuple(opened),
                                 "nested file changed during snapshot")
                        _require(_identity_tuple(os.stat(name, dir_fd=fd, follow_symlinks=False))
                                 == _identity_tuple(opened), "nested file rebound")
                    finally:
                        os.close(child)
                else:
                    raise ContractError("tree contains non-file/non-directory entry")
            _require(sorted(os.listdir(fd)) == names_before, "nested membership changed")
            _require(_identity_tuple(os.fstat(fd)) == _identity_tuple(initial),
                     "nested directory metadata changed")
            if parent_fd is not None and own_name is not None:
                _require(_identity_tuple(os.stat(own_name, dir_fd=parent_fd,
                                                follow_symlinks=False)) == _identity_tuple(initial),
                         "nested directory name changed")

        walk(root_fd, "", fds[-2] if len(fds) > 1 else None,
             names[-1] if names else None, root_info)
        _revalidate_chain(fds, names, chain_identities, chain_mount_ids)
        mount_root = _entry(os.fstat(fds[boundary]), "directory")
        ancestors = [_entry(os.fstat(fd), "directory") for fd in fds]
        _revalidate_chain(fds, names, chain_identities, chain_mount_ids)
        # Under the separately authenticated host/run lock and process/device
        # quiescence contract, require two identical complete observations.
        # The second observation is the terminal filesystem operation and is
        # the state from which the returned manifest is derived.  This is an
        # authenticated observation, not a perpetual-current-state claim;
        # every downstream consumer must freshly reauthenticate it.
        resealed = snapshot_tree_revalidate(root_fd, entries, expected_device, mount_id)
        _require(resealed == entries, "tree changed before final seal")
        _revalidate_chain(fds, names, chain_identities, chain_mount_ids)
        observed = snapshot_tree_revalidate(
            root_fd, entries, expected_device, mount_id)
        _require(observed == entries, "tree changed during terminal observation")
        snapshot = {
            "schema": "a1-native-five-anchored-tree-v2",
            "path": os.path.abspath(os.fspath(path)),
            "state": "present", "mount_id": mount_id, "mount_device": expected_device,
            "mount_root_identity_sha256": canonical_sha(mount_root),
            "ancestors": ancestors, "ancestor_mount_ids": list(chain_mount_ids),
            "entries": observed, "entry_count": len(observed),
            "logical_bytes": sum(item["size"] for item in observed.values()
                                 if item["kind"] == "file"),
            "allocated_bytes": sum(item["allocated"] for item in observed.values()),
            "complete": True,
        }
        snapshot["manifest_sha256"] = canonical_sha(_tree_manifest_payload(snapshot))
        return validate_tree_snapshot(
            snapshot, snapshot["path"], mount_id, expected_device,
            snapshot["mount_root_identity_sha256"])
    finally:
        if owns_chain:
            for fd in reversed(fds):
                os.close(fd)


def snapshot_tree_revalidate(root_fd: int, expected: dict[str, dict[str, Any]],
                             expected_device: int,
                             expected_mount_id: int) -> dict[str, dict[str, Any]]:
    """Rewalk an already pinned root and return its exact sealed manifest."""
    observed: dict[str, dict[str, Any]] = {}

    def walk(fd: int, prefix: str) -> None:
        here = os.fstat(fd)
        _require(here.st_dev == expected_device and _fd_mount_id(fd) == expected_mount_id,
                 "reseal directory mount changed")
        relative_root = prefix or "."
        observed[relative_root] = _entry(here, "directory")
        for name in sorted(os.listdir(fd)):
            _require(name not in (".", "..") and "/" not in name,
                     "unsafe reseal member")
            relative = f"{prefix}/{name}" if prefix else name
            named = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(named.st_mode):
                child = os.open(name, _dir_flags(), dir_fd=fd)
                try:
                    opened = os.fstat(child)
                    _require(_identity_tuple(opened) == _identity_tuple(named)
                             and opened.st_dev == expected_device
                             and _fd_mount_id(child) == expected_mount_id,
                             "reseal directory rebound")
                    walk(child, relative)
                    _require(_identity_tuple(os.fstat(child)) == _identity_tuple(opened)
                             and _identity_tuple(os.stat(
                                 name, dir_fd=fd, follow_symlinks=False))
                             == _identity_tuple(opened), "reseal directory changed")
                finally:
                    os.close(child)
            elif stat.S_ISREG(named.st_mode):
                child = os.open(name, _file_flags(), dir_fd=fd)
                try:
                    opened = os.fstat(child)
                    _require(_identity_tuple(opened) == _identity_tuple(named)
                             and opened.st_dev == expected_device and opened.st_nlink == 1
                             and _fd_mount_id(child) == expected_mount_id,
                             "reseal file rebound")
                    digest = hashlib.sha256()
                    remaining = opened.st_size
                    while remaining:
                        try:
                            chunk = os.read(child, min(1024 * 1024, remaining))
                        except InterruptedError:
                            continue
                        _require(bool(chunk), "reseal file truncated")
                        digest.update(chunk)
                        remaining -= len(chunk)
                    _require(not os.read(child, 1), "reseal file appended")
                    observed[relative] = _entry(opened, "file", digest.hexdigest())
                    _require(_identity_tuple(os.fstat(child)) == _identity_tuple(opened)
                             and _identity_tuple(os.stat(
                                 name, dir_fd=fd, follow_symlinks=False))
                             == _identity_tuple(opened), "reseal file changed")
                finally:
                    os.close(child)
            else:
                raise ContractError("reseal found non-file/non-directory entry")
        _require(_identity_tuple(os.fstat(fd)) == _identity_tuple(here),
                 "reseal directory metadata changed")

    walk(root_fd, "")
    _require(set(observed) == set(expected), "reseal membership changed")
    return observed


def _validate_observation_context(context: Any, arm: str, phase: str,
                                  watcher: dict[str, Any], mount: dict[str, Any],
                                  checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the protocol assumptions for one bounded tree observation."""
    context = _exact_dict(context, OBSERVATION_CONTEXT_KEYS, "observation context")
    _require(arm in ARMS and phase in {"observation", "consumer"},
             "observation arm/phase invalid")
    _require(context["schema"] == "a1-native-five-tree-observation-context-v1"
             and context["arm"] == arm and context["phase"] == phase
             and (context["host"], context["hostname"], context["label"])
             == (HOST, HOSTNAME, ARMS[arm]["label"])
             and context["complete"] is True, "observation context identity mismatch")
    for key in ("source_sha256", "request_sha256", "job_sha256", "postflight_sha256",
                "lock_receipt_sha256", "process_group_receipt_sha256",
                "device_postflight_receipt_sha256", "watcher_receipt_sha256",
                "mount_record_sha256", "mount_root_identity_sha256",
                "checkpoint_receipt_sha256"):
        _require(_sha(context[key]), f"observation {key} malformed")
    _require(context["source_sha256"] == GATE_SOURCE_SHA256[arm],
             "observation source mismatch")
    _require(type(context["observed_monotonic_ns"]) is int
             and context["observed_monotonic_ns"] >= 0,
             "observation time malformed")

    lock = _exact_dict(context["lock_receipt"], LOCK_RECEIPT_KEYS, "lock receipt")
    _require(lock["schema"] == "a1-native-five-host-lock-v1"
             and (lock["host"], lock["label"], lock["path"])
             == (HOST, ARMS[arm]["label"], LOCK_PATH)
             and lock["owner_uid"] == os.getuid()
             and type(lock["owner_pid"]) is int and lock["owner_pid"] > 0
             and type(lock["owner_start_ticks"]) is int and lock["owner_start_ticks"] > 0
             and lock["held"] is True and lock["complete"] is True,
             "host/run lock ownership is not exact and held")
    _require(context["lock_receipt_sha256"] == canonical_sha(lock),
             "lock receipt digest mismatch")

    process = _exact_dict(context["process_group_receipt"],
                          PROCESS_GROUP_RECEIPT_KEYS, "process-group receipt")
    _require(process["schema"] == "a1-native-five-process-group-postflight-v1"
             and process["label"] == ARMS[arm]["label"]
             and type(process["pgid"]) is int and process["pgid"] > 0
             and type(process["term_sent"]) is bool and type(process["kill_sent"]) is bool
             and process["reaped"] is True and process["quiescent"] is True
             and process["remaining_pids"] == [] and process["complete"] is True,
             "process group is not exactly reaped and quiescent")
    _require(type(process["owned_processes"]) is list and process["owned_processes"],
             "owned process birth identities missing")
    for identity in process["owned_processes"]:
        identity = _exact_dict(identity, PROCESS_IDENTITY_KEYS, "owned process identity")
        _require(type(identity["pid"]) is int and identity["pid"] > 0
                 and type(identity["start_ticks"]) is int and identity["start_ticks"] > 0
                 and _sha(identity["command_sha256"])
                 and type(identity["exit_code"]) is int,
                 "owned process identity malformed")
    _require(context["process_group_receipt_sha256"] == canonical_sha(process),
             "process-group receipt digest mismatch")

    device = _exact_dict(context["device_postflight_receipt"],
                         DEVICE_POSTFLIGHT_KEYS, "device postflight receipt")
    _require(device["schema"] == "a1-native-five-device-postflight-v1"
             and (device["host"], device["hostname"], device["device"])
             == (HOST, HOSTNAME, "/dev/neuron0")
             and device["active_processes"] == [] and device["unknown_holders"] == []
             and device["device_idle"] is True and device["complete"] is True,
             "device/unknown-holder postflight is not quiescent")
    _require(context["device_postflight_receipt_sha256"] == canonical_sha(device),
             "device postflight receipt digest mismatch")

    watcher = _exact_dict(watcher, WATCH_RECEIPT_KEYS, "bound watcher receipt")
    _require(watcher["schema"] == "a1-native-five-watch-receipt-v2"
             and watcher["status"] == "PASS" and watcher["complete"] is True
             and (watcher["arm"], watcher["host"], watcher["label"])
             == (arm, HOST, ARMS[arm]["label"]), "bound watcher receipt identity")
    _require(type(watcher["samples"]) is list and len(watcher["samples"]) >= 3
             and watcher["sample_count"] == len(watcher["samples"]),
             "bound watcher coverage missing")
    coverage_sha = _validate_coverage(watcher["coverage"], watcher["samples"])
    _require(watcher["coverage_sha256"] == coverage_sha
             and watcher["series_sha256"] == canonical_sha(watcher["samples"])
             and watcher["samples"][-1]["phase"] == "final_seal"
             and context["observed_monotonic_ns"] >= watcher["last_monotonic_ns"],
             "watcher does not cover observation")
    _require((watcher["source_sha256"], watcher["job_sha256"],
              watcher["postflight_sha256"])
             == (context["source_sha256"], context["job_sha256"],
                 context["postflight_sha256"]), "watcher observation binding mismatch")
    _require(context["watcher_receipt_sha256"] == canonical_sha(watcher),
             "watcher receipt digest mismatch")

    mount = _validate_mount_record(mount, "/dev/shm")
    _require(context["mount_record_sha256"] == canonical_sha(mount)
             and context["mount_root_identity_sha256"]
             == canonical_sha(mount["target_identity"]),
             "mount observation binding mismatch")
    checkpoint = _exact_dict(checkpoint, CHECKPOINT_KEYS, "bound checkpoint receipt")
    _require(checkpoint["complete"] is True
             and (checkpoint["source_sha256"], checkpoint["job_sha256"],
                  checkpoint["postflight_sha256"])
             == (context["source_sha256"], context["job_sha256"],
                 context["postflight_sha256"]), "checkpoint observation binding mismatch")
    _require(context["checkpoint_receipt_sha256"] == canonical_sha(checkpoint),
             "checkpoint receipt digest mismatch")
    return context


def build_tree_observation_receipt(snapshot: dict[str, Any], context: dict[str, Any],
                                   watcher: dict[str, Any], mount: dict[str, Any],
                                   checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Build the bounded immutable receipt; tree/checkpoint bytes remain on tmpfs."""
    snapshot = validate_tree_snapshot(
        snapshot, snapshot["path"], snapshot["mount_id"], snapshot["mount_device"],
        snapshot["mount_root_identity_sha256"])
    arm = context.get("arm") if type(context) is dict else None
    context = _validate_observation_context(
        context, arm, "observation", watcher, mount, checkpoint)
    _require(snapshot["state"] == "present"
             and snapshot["mount_id"] == mount["mount_id"]
             and snapshot["mount_device"] == mount["st_dev"]
             and snapshot["mount_root_identity_sha256"]
             == canonical_sha(mount["target_identity"]),
             "tree and mount receipt mismatch")
    return {
        "schema": "a1-native-five-tree-observation-receipt-v1",
        "statement": OBSERVATION_STATEMENT,
        "arm": arm, "host": HOST, "hostname": HOSTNAME,
        "label": ARMS[arm]["label"], "tree_path": snapshot["path"],
        "tree_manifest_sha256": snapshot["manifest_sha256"],
        "tree_snapshot_sha256": canonical_sha(snapshot),
        "source_sha256": context["source_sha256"],
        "request_sha256": context["request_sha256"],
        "job_sha256": context["job_sha256"],
        "postflight_sha256": context["postflight_sha256"],
        "lock_receipt_sha256": context["lock_receipt_sha256"],
        "process_group_receipt_sha256": context["process_group_receipt_sha256"],
        "device_postflight_receipt_sha256": context["device_postflight_receipt_sha256"],
        "watcher_receipt_sha256": context["watcher_receipt_sha256"],
        "mount_record_sha256": context["mount_record_sha256"],
        "mount_root_identity_sha256": context["mount_root_identity_sha256"],
        "checkpoint_receipt_sha256": context["checkpoint_receipt_sha256"],
        "observation_context_sha256": canonical_sha(context), "complete": True,
    }


def _receipt_bytes(receipt: Any) -> bytes:
    receipt = _exact_dict(receipt, TREE_OBSERVATION_RECEIPT_KEYS,
                          "tree observation receipt")
    _require(receipt["schema"] == "a1-native-five-tree-observation-receipt-v1"
             and receipt["statement"] == OBSERVATION_STATEMENT
             and receipt["arm"] in ARMS
             and (receipt["host"], receipt["hostname"], receipt["label"])
             == (HOST, HOSTNAME, ARMS[receipt["arm"]]["label"])
             and receipt["complete"] is True, "observation receipt identity mismatch")
    for key in TREE_OBSERVATION_RECEIPT_KEYS - {
            "schema", "statement", "arm", "host", "hostname", "label", "tree_path",
            "complete"}:
        _require(_sha(receipt[key]), f"observation receipt {key} malformed")
    _require(type(receipt["tree_path"]) is str and receipt["tree_path"].startswith("/"),
             "observation tree path malformed")
    return json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode() + b"\n"


def _read_exact_fd(fd: int, size: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    payload = bytearray()
    while len(payload) < size:
        try:
            chunk = os.read(fd, min(1024 * 1024, size - len(payload)))
        except InterruptedError:
            continue
        _require(bool(chunk), "receipt truncated during same-descriptor readback")
        payload.extend(chunk)
    _require(not os.read(fd, 1), "receipt appended during same-descriptor readback")
    return bytes(payload)


def _stable_directory_identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_uid, info.st_gid)


def _revalidate_publication_chain(fds: list[int], names: list[str],
                                  identities: list[tuple[int, ...]],
                                  mount_ids: list[int]) -> None:
    stable = [_stable_directory_identity(os.fstat(fd)) for fd in fds]
    _require(len(fds) == len(names) + 1 == len(identities) == len(mount_ids),
             "publication chain cardinality mismatch")
    for index, fd in enumerate(fds):
        _require(_stable_directory_identity(os.fstat(fd)) == stable[index]
                 and _fd_mount_id(fd) == mount_ids[index],
                 "publication directory identity changed")
        if index:
            named = os.stat(names[index - 1], dir_fd=fds[index - 1],
                            follow_symlinks=False)
            _require(_stable_directory_identity(named) == stable[index],
                     "publication lexical chain changed")


def publish_tree_observation_receipt(path: Path, expected_path: str,
                                     receipt: dict[str, Any]) -> dict[str, Any]:
    """Publish one small canonical receipt exclusively; never copy tree bytes."""
    absolute = Path(os.path.abspath(path))
    _require(os.fspath(absolute) == expected_path and absolute.name not in ("", ".", ".."),
             "observation receipt path differs from authorized request")
    data = _receipt_bytes(receipt)
    _require(len(data) <= 64 * 1024, "observation receipt unexpectedly large")
    digest = hashlib.sha256(data).hexdigest()
    fds, names, identities, mount_ids = _open_dir_chain(absolute.parent)
    parent_fd = fds[-1]
    write_fd = -1
    try:
        try:
            os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ContractError("observation receipt path already exists")
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        write_fd = os.open(absolute.name, flags, 0o600, dir_fd=parent_fd)
        offset = 0
        while offset < len(data):
            try:
                written = os.write(write_fd, data[offset:])
            except InterruptedError:
                continue
            _require(written > 0, "zero-byte observation receipt write")
            offset += written
        os.fsync(write_fd)
        sealed = os.fstat(write_fd)
        _require(stat.S_ISREG(sealed.st_mode) and stat.S_IMODE(sealed.st_mode) == 0o600
                 and sealed.st_uid == os.getuid() and sealed.st_gid == os.getgid()
                 and sealed.st_nlink == 1 and sealed.st_size == len(data),
                 "published observation receipt metadata mismatch")
        _require(_read_exact_fd(write_fd, len(data)) == data,
                 "published observation receipt readback mismatch")
        os.fsync(parent_fd)
        _revalidate_publication_chain(fds, names, identities, mount_ids)
        named = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        _require(_identity_tuple(named) == _identity_tuple(sealed)
                 and _identity_tuple(os.fstat(write_fd)) == _identity_tuple(sealed)
                 and hashlib.sha256(_read_exact_fd(write_fd, len(data))).hexdigest() == digest,
                 "published observation receipt final seal mismatch")
        return {"sha256": digest, "bytes": len(data),
                "identity": _identity_tuple(sealed), "receipt": receipt}
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        for fd in reversed(fds):
            os.close(fd)


def read_tree_observation_receipt(path: Path, expected_path: str,
                                  expected_sha256: str) -> dict[str, Any]:
    absolute = Path(os.path.abspath(path))
    _require(os.fspath(absolute) == expected_path and _sha(expected_sha256),
             "observation receipt read authority mismatch")
    fds, names, identities, mount_ids = _open_dir_chain(absolute.parent)
    fd = -1
    try:
        fd = os.open(absolute.name, _file_flags(), dir_fd=fds[-1])
        before = os.fstat(fd)
        named = os.stat(absolute.name, dir_fd=fds[-1], follow_symlinks=False)
        _require(_identity_tuple(before) == _identity_tuple(named)
                 and stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o600
                 and before.st_uid == os.getuid() and before.st_gid == os.getgid()
                 and before.st_nlink == 1 and before.st_size <= 64 * 1024,
                 "observation receipt file identity mismatch")
        data = _read_exact_fd(fd, before.st_size)
        _require(hashlib.sha256(data).hexdigest() == expected_sha256
                 and _identity_tuple(os.fstat(fd)) == _identity_tuple(before),
                 "observation receipt bytes changed")
        _revalidate_chain(fds, names, identities, mount_ids)
        try:
            receipt = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("observation receipt JSON invalid") from exc
        _require(_receipt_bytes(receipt) == data, "observation receipt is not canonical")
        return receipt
    finally:
        if fd >= 0:
            os.close(fd)
        for chain_fd in reversed(fds):
            os.close(chain_fd)


def consume_tree_observation_receipt(
        receipt_path: Path, expected_path: str, expected_receipt_sha256: str,
        expected_snapshot: dict[str, Any], observation_context: dict[str, Any],
        consumer_context: dict[str, Any], watcher: dict[str, Any],
        mount: dict[str, Any], checkpoint: dict[str, Any], checkpoint_reauthenticate) -> dict[str, Any]:
    """Freshly reauthenticate retained live state before advancement/comparison."""
    expected_receipt = build_tree_observation_receipt(
        expected_snapshot, observation_context, watcher, mount, checkpoint)
    observed_receipt = read_tree_observation_receipt(
        receipt_path, expected_path, expected_receipt_sha256)
    _require(observed_receipt == expected_receipt,
             "immutable observation receipt differs from archived evidence")
    arm = expected_receipt["arm"]
    fresh_checkpoint = checkpoint_reauthenticate()
    _require(type(fresh_checkpoint) is dict and fresh_checkpoint == checkpoint,
             "fresh checkpoint reauthentication drifted")
    consumer_context = _validate_observation_context(
        consumer_context, arm, "consumer", watcher, mount, fresh_checkpoint)
    for key in ("arm", "host", "hostname", "label", "source_sha256", "request_sha256",
                "job_sha256", "postflight_sha256", "watcher_receipt_sha256",
                "mount_record_sha256", "mount_root_identity_sha256",
                "checkpoint_receipt_sha256"):
        _require(consumer_context[key] == observation_context[key],
                 f"consumer context changed {key}")
    live_mount = read_mountinfo_record(mount["target"])
    _require(live_mount == mount, "live mount record/root identity drifted")
    fresh = snapshot_path(Path(expected_snapshot["path"]),
                          mount_id=mount["mount_id"], expected_device=mount["st_dev"])
    _require(fresh == expected_snapshot
             and canonical_sha(fresh) == expected_receipt["tree_snapshot_sha256"],
             "fresh retained tree reauthentication drifted")
    return {"schema": "a1-native-five-tree-consumption-v1", "arm": arm,
            "status": "PASS", "receipt_sha256": expected_receipt_sha256,
            "tree_manifest_sha256": fresh["manifest_sha256"],
            "checkpoint_receipt_sha256": canonical_sha(fresh_checkpoint),
            "lock_held": True, "complete": True}


def validate_initial_absence(trusted_root: Path, target: Path) -> dict[str, Any]:
    """Produce a preflight-only absence receipt; later O_EXCL creation is mandatory."""
    root = Path(os.path.abspath(trusted_root))
    absolute = Path(os.path.abspath(target))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ContractError("path escapes trusted root") from exc
    _require(relative.parts and all(part not in ("", ".", "..") for part in relative.parts),
             "unsafe relative path")
    fds, names, identities, mount_ids = _open_dir_chain(root)
    try:
        parent_fd = fds[-1]
        existing_parts: list[str] = []
        for part in relative.parts:
            try:
                child = os.open(part, _dir_flags(), dir_fd=parent_fd)
            except FileNotFoundError:
                _revalidate_chain(fds, names, identities, mount_ids)
                receipt = {"schema": "a1-native-five-absence-preflight-v2",
                           "trusted_root": os.fspath(root), "target": os.fspath(absolute),
                           "existing_parts": existing_parts, "first_absent": part,
                           "parent_identity_sha256": canonical_sha(
                               {"identities": identities, "mount_ids": mount_ids}),
                           "o_excl_creation_required": True, "authoritative": False}
                return receipt
            except OSError as exc:
                raise ContractError("absence ancestor is not a stable directory") from exc
            opened = os.fstat(child)
            named = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(opened.st_mode) or _identity_tuple(opened) != _identity_tuple(named):
                os.close(child)
                raise ContractError("absence ancestor changed or is not a directory")
            fds.append(child)
            names.append(part)
            identities.append(_identity_tuple(opened))
            mount_ids.append(_fd_mount_id(child))
            existing_parts.append(part)
            parent_fd = child
        raise ContractError("authorized fresh path already exists")
    finally:
        for fd in reversed(fds):
            os.close(fd)


def _validate_root_caches(caches: Any, mounts: dict[str, Any]) -> dict[str, Any]:
    _require(type(caches) is dict and set(caches) == set(ROOT_CACHE_PATHS),
             "root cache enumeration incomplete")
    root = mounts["root"]
    for path, snapshot in caches.items():
        try:
            validate_tree_snapshot(snapshot, path, root["mount_id"], root["st_dev"],
                                   canonical_sha(root["target_identity"]))
        except ContractError as exc:
            raise ContractError(f"root cache snapshot invalid: {exc}") from exc
    return caches


def _sample_payload(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: sample[key] for key in WATCH_SAMPLE_KEYS - {"sample_sha256"}}


def validate_live_sample(arm: str, sample: Any, baseline_cache_digest: str) -> None:
    sample = _exact_dict(sample, WATCH_SAMPLE_KEYS, "watch sample")
    _require(sample["schema"] == "a1-native-five-watch-sample-v2", "watch schema")
    _require(type(sample["sequence"]) is int and sample["sequence"] >= 0, "watch sequence")
    _require(type(sample["monotonic_ns"]) is int and sample["monotonic_ns"] >= 0,
             "watch timestamp")
    _require(sample["phase"] in {"pre_launch", "runtime", "final_seal"}, "watch phase")
    for key in WATCH_SAMPLE_KEYS - {"schema", "phase", "sample_sha256",
                                    "root_cache_manifest_sha256", "current_arm_manifest_sha256"}:
        _require(type(sample[key]) is int and sample[key] >= 0, f"watch {key} type")
    _require(sample["root_cache_manifest_sha256"] == baseline_cache_digest,
             "root cache snapshot changed during live run")
    _require(_sha(sample["current_arm_manifest_sha256"]), "watch arm digest malformed")
    _require(sample["sample_sha256"] == canonical_sha(_sample_payload(sample)),
             "watch sample digest mismatch")
    _require(sample["swap_total_bytes"] == 0 and sample["swap_used_bytes"] == 0,
             "swap appeared during live run")
    _require(sample["mem_available_bytes"] >= LIVE_MEMORY_BREAKER, "live memory breaker failed")
    _require(sample["root_free_bytes"] >= ROOT_BREAKER, "live root breaker failed")
    _require(sample["root_free_inodes"] >= ROOT_INODE_FLOOR, "live root inode floor failed")
    _require(sample["tmpfs_free_bytes"] >= ARMS[arm]["tmpfs_free_after"],
             "live tmpfs floor failed")
    _require(sample["tmpfs_free_inodes"] >= TMPFS_INODE_FLOOR, "live tmpfs inode floor failed")
    _require(sample["root_log_allocated"] <= ROOT_LOG_CAP, "live root log cap failed")
    _require(sample["cache_allocated"] <= CACHE_CAP and sample["output_allocated"] <= OUT_CAP
             and sample["cache_allocated"] + sample["output_allocated"] <= ARM_CAP,
             "live arm allocation cap failed")


def _validate_coverage(coverage: Any, samples: list[dict[str, Any]]) -> str:
    coverage = _exact_dict(coverage, WATCH_COVERAGE_KEYS, "watch coverage")
    _require(coverage["schema"] == "a1-native-five-watch-coverage-v2"
             and coverage["complete"] is True, "watch coverage state")
    for key in ("interval_limit_ns", "child_start_ns", "child_exit_ns", "final_seal_ns"):
        _require(type(coverage[key]) is int and coverage[key] >= 0, f"coverage {key} type")
    _require(coverage["interval_limit_ns"] == WATCH_INTERVAL_MAX_NS, "watch interval mismatch")
    _require(coverage["child_start_ns"] <= coverage["child_exit_ns"]
             <= coverage["final_seal_ns"], "watch lifecycle order")
    _require(len(samples) >= 3, "watcher requires nonzero pre/runtime/final samples")
    _require(samples[0]["phase"] == "pre_launch"
             and samples[0]["monotonic_ns"] <= coverage["child_start_ns"],
             "watcher did not start before runtime")
    _require(samples[-1]["phase"] == "final_seal"
             and samples[-1]["monotonic_ns"] >= coverage["final_seal_ns"],
             "watcher did not cover final seal")
    _require(any(sample["phase"] == "runtime"
                 and coverage["child_start_ns"] <= sample["monotonic_ns"]
                 <= coverage["child_exit_ns"] for sample in samples),
             "watcher has no authenticated runtime sample")
    for before, after in zip(samples, samples[1:]):
        _require(after["sequence"] == before["sequence"] + 1, "watch sequence gap")
        _require(0 < after["monotonic_ns"] - before["monotonic_ns"]
                 <= WATCH_INTERVAL_MAX_NS, "watch cadence gap")
    return canonical_sha(coverage)


def run_watcher(arm: str, samples: list[dict[str, Any]], baseline_root_caches: dict[str, Any],
                coverage: dict[str, Any], binding: dict[str, Any], terminate) -> dict[str, Any]:
    try:
        _require(type(binding) is dict and set(binding) == {
            "source_sha256", "observer_sha256", "job_sha256", "postflight_sha256"},
            "watcher binding schema mismatch")
        _require(binding["source_sha256"] == GATE_SOURCE_SHA256[arm]
                 and binding["observer_sha256"] == OBSERVER_SHA256
                 and _sha(binding["job_sha256"]) and _sha(binding["postflight_sha256"]),
                 "watcher binding mismatch")
        _require(type(samples) is list and samples, "zero watcher samples are invalid")
        baseline_digest = canonical_sha(baseline_root_caches)
        for sample in samples:
            validate_live_sample(arm, sample, baseline_digest)
        coverage_sha = _validate_coverage(coverage, samples)
    except ContractError as exc:
        lifecycle = terminate()
        _require(type(lifecycle) is dict and lifecycle == {
            "term_sent": True, "kill_sent": True, "reaped": True, "quiescent": True},
            "bounded termination did not prove exact quiescence")
        status = "CACHE_ROUTING_INVALID" if "cache" in str(exc) else "RUNTIME_STORAGE_INVALID"
        return {"status": status, "error": str(exc), "lifecycle": lifecycle}
    return {"schema": "a1-native-five-watch-receipt-v2", "status": "PASS",
            "arm": arm, "host": HOST, "label": ARMS[arm]["label"],
            "source_sha256": binding["source_sha256"],
            "observer_sha256": binding["observer_sha256"],
            "job_sha256": binding["job_sha256"],
            "postflight_sha256": binding["postflight_sha256"],
            "baseline_root_caches_sha256": baseline_digest,
            "sample_count": len(samples), "first_monotonic_ns": samples[0]["monotonic_ns"],
            "last_monotonic_ns": samples[-1]["monotonic_ns"],
            "series_sha256": canonical_sha(samples), "coverage_sha256": coverage_sha,
            "samples": samples, "coverage": coverage, "complete": True}


def _validate_watcher_receipt(receipt: Any, arm: str, binding: dict[str, Any],
                              baseline_root_caches_sha256: str,
                              final_arm_manifest_sha256: str) -> dict[str, Any]:
    receipt = _exact_dict(receipt, WATCH_RECEIPT_KEYS, "watcher receipt")
    _require(receipt["schema"] == "a1-native-five-watch-receipt-v2"
             and receipt["status"] == "PASS" and receipt["complete"] is True,
             "watcher receipt status")
    _require((receipt["arm"], receipt["host"], receipt["label"])
             == (arm, HOST, ARMS[arm]["label"]), "watcher identity mismatch")
    _require(receipt["source_sha256"] == GATE_SOURCE_SHA256[arm]
             and receipt["observer_sha256"] == OBSERVER_SHA256
             and receipt["job_sha256"] == binding["job_sha256"]
             and receipt["postflight_sha256"] == binding["postflight_sha256"],
             "watcher evidence binding mismatch")
    _require(receipt["baseline_root_caches_sha256"] == baseline_root_caches_sha256,
             "watcher baseline cache binding mismatch")
    _require(type(receipt["samples"]) is list
             and len(receipt["samples"]) == receipt["sample_count"],
             "watcher sample series missing")
    for sample in receipt["samples"]:
        validate_live_sample(arm, sample, baseline_root_caches_sha256)
    coverage_sha = _validate_coverage(receipt["coverage"], receipt["samples"])
    _require(type(receipt["sample_count"]) is int and receipt["sample_count"] >= 3,
             "watcher receipt zero/short series")
    for key in ("first_monotonic_ns", "last_monotonic_ns"):
        _require(type(receipt[key]) is int and receipt[key] >= 0, f"watch receipt {key}")
    for key in ("series_sha256", "coverage_sha256", "baseline_root_caches_sha256"):
        _require(_sha(receipt[key]), f"watch receipt {key}")
    _require(receipt["series_sha256"] == canonical_sha(receipt["samples"])
             and receipt["coverage_sha256"] == coverage_sha,
             "watcher series/coverage digest mismatch")
    _require(receipt["samples"][-1]["current_arm_manifest_sha256"]
             == final_arm_manifest_sha256, "watcher terminal arm seal mismatch")
    return receipt


def _expected_environment(arm: str) -> dict[str, str]:
    return {**COMMON_ENVIRONMENT, "NEURON_COMPILE_CACHE_DIR": arm_paths(arm)["compile_cache"],
            "TRAINIUM_FRESH_START_RUN_ID": ARMS[arm]["label"]}


def _validate_checkpoint(receipt: Any, arm: str, mount: dict[str, Any],
                         binding: dict[str, Any]) -> dict[str, Any]:
    receipt = _exact_dict(receipt, CHECKPOINT_KEYS, "checkpoint receipt")
    _require(receipt["schema"] == "a1-native-five-trn22-checkpoint-v2"
             and receipt["complete"] is True, "checkpoint schema/state")
    _require(receipt["path"] == arm_paths(arm)["checkpoint"], "checkpoint path mismatch")
    _require(receipt["mount_id"] == mount["mount_id"]
             and receipt["mount_device"] == mount["st_dev"]
             and receipt["device"] == mount["st_dev"], "checkpoint mount/device mismatch")
    for key in ("inode", "uid", "gid", "bytes", "allocated", "mtime_ns", "ctime_ns"):
        _require(type(receipt[key]) is int and receipt[key] >= 0, f"checkpoint {key} type")
    _require(receipt["kind"] == "file" and receipt["mode"] == 0o600
             and receipt["uid"] == os.getuid() and receipt["gid"] == os.getgid()
             and receipt["nlink"] == 1 and receipt["bytes"] > 0,
             "checkpoint exact file identity failed")
    for key in ("weights_only", "model_only", "finite", "strict_load", "logits_exact",
                "causality_exact", "same_fd_stable"):
        _require(receipt[key] is True, f"checkpoint {key} failed")
    for key in ("sha256", "model_state_sha256", "eval_logits_sha256",
                "causal_digest_sha256", "source_sha256", "observer_sha256",
                "job_sha256", "postflight_sha256"):
        _require(_sha(receipt[key]), f"checkpoint {key} malformed")
    _require(type(receipt["rank_report_sha256"]) is list
             and len(receipt["rank_report_sha256"]) == 4
             and all(_sha(item) for item in receipt["rank_report_sha256"]),
             "checkpoint rank report binding malformed")
    _require(receipt["source_sha256"] == GATE_SOURCE_SHA256[arm]
             and receipt["observer_sha256"] == OBSERVER_SHA256,
             "checkpoint source binding mismatch")
    _require(receipt["job_sha256"] == binding["job_sha256"]
             and receipt["postflight_sha256"] == binding["postflight_sha256"]
             and receipt["rank_report_sha256"] == binding["rank_report_sha256"],
             "checkpoint evidence binding mismatch")
    parent_identity = canonical_sha(binding["output"]["entries"]["."])
    _require(receipt["parent_identity_sha256"] == parent_identity,
             "checkpoint parent identity binding mismatch")
    _require(receipt["mount_root_identity_sha256"]
             == canonical_sha(mount["target_identity"]),
             "checkpoint mount-root identity binding mismatch")
    return receipt


def validate_arm_evidence(evidence: Any, arm: str, mount: dict[str, Any],
                          baseline_root_caches_sha256: str) -> dict[str, Any]:
    evidence = _exact_dict(evidence, ARM_EVIDENCE_KEYS, f"{arm} evidence")
    _require(evidence["schema"] == "a1-native-five-trn22-arm-evidence-v2"
             and evidence["complete"] is True, "arm evidence version/state")
    _require((evidence["arm"], evidence["host"], evidence["hostname"], evidence["label"])
             == (arm, HOST, HOSTNAME, ARMS[arm]["label"]), "arm identity mismatch")
    _require(evidence["source_sha256"] == GATE_SOURCE_SHA256[arm]
             and evidence["observer_sha256"] == OBSERVER_SHA256, "arm source mismatch")
    _require(evidence["protected_source_sha256"] == SOURCE_PINS,
             "arm protected source map mismatch")
    _require(evidence["train_argv_sha256"] == canonical_sha(expected_train_argv(arm))
             and evidence["environment_sha256"] == canonical_sha(_expected_environment(arm)),
             "arm argv/environment binding mismatch")
    for key in ("job_sha256", "postflight_sha256"):
        _require(_sha(evidence[key]), f"arm {key} malformed")
    _require(type(evidence["rank_report_sha256"]) is list
             and len(evidence["rank_report_sha256"]) == 4
             and all(_sha(item) for item in evidence["rank_report_sha256"]),
             "arm rank reports malformed")
    output = validate_tree_snapshot(evidence["output"], arm_paths(arm)["output"],
                                    mount["mount_id"], mount["st_dev"],
                                    canonical_sha(mount["target_identity"]))
    try:
        cache = validate_tree_snapshot(evidence["cache"], arm_paths(arm)["compile_cache"],
                                       mount["mount_id"], mount["st_dev"],
                                       canonical_sha(mount["target_identity"]))
    except ContractError as exc:
        raise ContractError(f"declared tmpfs cache snapshot invalid: {exc}") from exc
    _require(0 < output["allocated_bytes"] <= OUT_CAP, "output cap failed")
    _require(4096 < cache["allocated_bytes"] <= CACHE_CAP,
             "declared tmpfs cache use is not positively proved")
    _require(output["allocated_bytes"] + cache["allocated_bytes"] <= ARM_CAP,
             "combined arm allocation cap failed")
    _validate_checkpoint(evidence["checkpoint"], arm, mount, evidence)
    final_arm_manifest = canonical_sha({"output": output["manifest_sha256"],
                                        "cache": cache["manifest_sha256"]})
    _validate_watcher_receipt(evidence["watcher"], arm, evidence,
                              baseline_root_caches_sha256, final_arm_manifest)
    _require(type(evidence["root_log_allocated"]) is int
             and 0 <= evidence["root_log_allocated"] <= ROOT_LOG_CAP, "root log cap failed")
    _require(evidence["process_quiescent"] is True and evidence["device_idle"] is True,
             "process/device is not quiescent")
    return evidence


def _validate_absence_receipt(receipt: Any, arm: str) -> None:
    keys = {"schema", "arm", "paths", "parent_identity_sha256", "o_excl_creation_required",
            "authoritative", "complete"}
    receipt = _exact_dict(receipt, keys, "absence receipt")
    expected = [arm_paths(arm)[key] for key in ("tmpfs_arm", "output", "compile_cache",
                                                "checkpoint", "remote", "bundle",
                                                "root_report", "child_log", "runtime_job",
                                                "runtime_result", "observation_receipt",
                                                "dispatch", "controller_request")]
    _require(receipt["schema"] == "a1-native-five-absence-set-v2" and receipt["arm"] == arm
             and receipt["paths"] == expected and _sha(receipt["parent_identity_sha256"])
             and receipt["o_excl_creation_required"] is True
             and receipt["authoritative"] is False and receipt["complete"] is True,
             "absence receipt mismatch")


def adjudicate_storage(arm: str, pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    try:
        pre_keys = {"schema", "host", "hostname", "mounts", "resources",
                    "fresh_paths_absent", "absence_receipt", "root_caches", "retained_arms"}
        post_keys = {"schema", "host", "hostname", "mounts", "resources", "root_caches",
                     "retained_arms", "current_arm", "root_growth_bytes",
                     "root_unallowlisted_growth_bytes"}
        pre = _exact_dict(pre, pre_keys, "pre evidence")
        post = _exact_dict(post, post_keys, "post evidence")
        _require(pre["schema"] == "a1-native-five-trn22-pre-v2"
                 and post["schema"] == "a1-native-five-trn22-post-v2", "evidence version")
        _require((pre["host"], pre["hostname"], post["host"], post["hostname"])
                 == (HOST, HOSTNAME, HOST, HOSTNAME), "host identity mismatch")
        _require(type(pre["mounts"]) is dict and set(pre["mounts"]) == {"root", "tmpfs"}
                 and pre["mounts"] == post["mounts"], "mount evidence mismatch")
        validate_mounts(pre["mounts"]["root"], pre["mounts"]["tmpfs"])
        validate_resources(arm, pre["resources"], phase="pre")
        validate_resources(arm, post["resources"], phase="post")
        _require(pre["fresh_paths_absent"] is True, "fresh paths not absent")
        _validate_absence_receipt(pre["absence_receipt"], arm)
        pre_caches = _validate_root_caches(pre["root_caches"], pre["mounts"])
        post_caches = _validate_root_caches(post["root_caches"], post["mounts"])
        _require(pre_caches == post_caches, "root cache snapshot changed")
        prior = [] if arm == "G0" else (["G0"] if arm == "G1" else ["G0", "G1"])
        _require(type(pre["retained_arms"]) is dict and sorted(pre["retained_arms"]) == prior
                 and pre["retained_arms"] == post["retained_arms"],
                 "retained arm set/identity mismatch")
        for retained in prior:
            validate_arm_evidence(pre["retained_arms"][retained], retained,
                                  pre["mounts"]["tmpfs"], canonical_sha(pre_caches))
        current = validate_arm_evidence(post["current_arm"], arm, post["mounts"]["tmpfs"],
                                        canonical_sha(pre_caches))
        delta = max(0, pre["resources"]["root_free_bytes"]
                    - post["resources"]["root_free_bytes"])
        _require(type(post["root_growth_bytes"]) is int
                 and post["root_growth_bytes"] == delta
                 and delta == current["root_log_allocated"] and delta <= ROOT_LOG_CAP,
                 "exact root growth breaker failed")
        _require(post["root_unallowlisted_growth_bytes"] == 0,
                 "unallowlisted root growth detected")
        return {"schema": "a1-native-five-trn22-storage-verdict-v2", "arm": arm,
                "status": "PASS", "downstream_unlocked": False,
                "cleanup_authorized": False}
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        message = str(exc)
        status = ("CACHE_ROUTING_INVALID" if any(term in message for term in
                  ("cache", "declared tmpfs", "root cache")) else "INVALID")
        return {"schema": "a1-native-five-trn22-storage-verdict-v2", "arm": arm,
                "status": status, "error": f"{type(exc).__name__}: {exc}",
                "downstream_unlocked": False, "cleanup_authorized": False}
