#!/usr/bin/env python3
"""Fail-closed, local-only contract for consumed trn21 staging cleanup.

This module has no SSH or training capability.  ``inventory`` is read-only.
``execute`` is reachable only with a separately finalized request, a matching
read-only inventory, and a separately supplied executable authorization blob.
The checked-in package intentionally contains neither of those authorities.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
from typing import Any, Iterator


SCHEMA_REQUEST = "a1-native-g1-consumed-staging-cleanup-request-v1"
SCHEMA_INVENTORY = "a1-native-g1-consumed-staging-cleanup-inventory-v1"
SCHEMA_AUTH = "a1-native-g1-consumed-staging-cleanup-authorization-v1"
SCHEMA_RECEIPT = "a1-native-g1-consumed-staging-cleanup-receipt-v1"
HOST = "trn21"
HOSTNAME = "ip-172-31-34-85"
PARENT = "/home/ubuntu"
ORIGIN = "https://github.com/CsabaConsulting/trainium_frontier.git"
MIN_POST_CLEANUP_FREE_BYTES = 7_286_616_064
LOCK_PATH = "/home/ubuntu/.cache/trainium_freshstart/device0.lock"
LOCK_PARENT = "/home/ubuntu/.cache/trainium_freshstart"
NEURON_LS = "/opt/aws/neuron/bin/neuron-ls"
GIT = "/usr/bin/git"

TARGETS: tuple[dict[str, Any], ...] = (
    {
        "kind": "checkout", "role": "consumed-r1-checkout",
        "path": "/home/ubuntu/trainium_freshstart_20260924_v2_a1_native5_g1_hybrid_observation_on_trn21_r1",
        "head": "149cc9124ff0b9306eff9747182079ab708fc4ed",
        "archive_evidence_commit": "e1550203a61c53cd7c73b0c543ced8a157312612",
    },
    {
        "kind": "bundle", "role": "consumed-r1-bundle",
        "path": "/home/ubuntu/v2-a1-native5-g1-head-149cc912-trn21-r1.bundle",
        "bytes": 46_549_993,
        "sha256": "942b85ec982b1834ab44d41219523c0ffbf640464d8a847417b529436c03116e",
        "archive_evidence_commit": "e1550203a61c53cd7c73b0c543ced8a157312612",
    },
    {
        "kind": "checkout", "role": "consumed-r3-checkout",
        "path": "/home/ubuntu/trainium_freshstart_20260925_v2_a1_native5_g1_hybrid_observation_on_trn21_r3",
        "head": "c4fed0c1b307b6c22149f4d42a134ce531879815",
        "archive_evidence_commit": "5e45a564bba0d04b052345a7c9c1812ddb71ce51",
    },
    {
        "kind": "bundle", "role": "consumed-r3-bundle",
        "path": "/home/ubuntu/v2-a1-native5-g1-r3.bundle",
        "bytes": 46_547_065,
        "sha256": "e319b4d1455a31dfa84594ff3fab6a6d4a3bf8ad8179f7a39bc2fe7d2504cd4c",
        "archive_evidence_commit": "5e45a564bba0d04b052345a7c9c1812ddb71ce51",
    },
    {
        "kind": "bundle", "role": "consumed-r2-bundle-only",
        "path": "/home/ubuntu/v2-a1-native5-g1-r2.bundle",
        "bytes": 46_585_255,
        "sha256": "bf33a3fe75e67fd23018151d8ccd1e219b96e91b7dfc52e1bb9fe34a48137bc8",
        "archive_evidence_commit": "5b392db4bc8c01a47da2f5969cb33df0c50ec52b",
    },
)

R2_ROOT = "/home/ubuntu/trainium_freshstart_20260924_v2_a1_native5_g1_hybrid_observation_on_trn21_r2"
R2_LABEL = "fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2"
R2_RUNTIME_HASHES = {
    f"research/run-logs/{R2_LABEL}.launch.json": "c28904ec69b06cc683cbfc9785b75909ca62e56c29ab9ff09dba60476979c09d",
    f"research/run-logs/{R2_LABEL}.job.json": "c28904ec69b06cc683cbfc9785b75909ca62e56c29ab9ff09dba60476979c09d",
    f"research/run-logs/{R2_LABEL}.process.json": "8aba524aa6a7e34dd88532638db4a930ab2cf82244b3276ca6021ed3d32c1047",
    f"research/run-logs/{R2_LABEL}.preflight.json": "958e843b0a490e1785940db144a18d99255e403ff1446e55dc8937f6dae0d13a",
    f"research/run-logs/{R2_LABEL}.child.log": "87dedc4345bb1605e115acf995ceb5dedb56c20bfddb423343405a85632e9f4a",
    f"research/run-logs/{R2_LABEL}.runtime-result.json": "705547ff56258ac69232b0dd515d07a52056445c32ce5e513ac15613ffba8596",
    f"research/run-logs/{R2_LABEL}.postflight.json": "75daab4b25fc52ddcf92ef954eed1af0a911278b60e4f81c1e92214b5907d03c",
    f"research/run-logs/{R2_LABEL}.log": "e861eb6f418728cbdc8658adb95ac07b0b6f08a90ccd82eb9def4b1b8173618a",
}
G0_CHECKPOINT = {
    "path": "/home/ubuntu/trainium_freshstart_20260924_v2_a1_native5_g0_hybrid_observation_off_trn21_r1/"
            "out/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/final.pt",
    "bytes": 739_322_371,
    "sha256": "2d0891c569e6046bd4a3635c62c1ebdb4c6ea6ed0d9788212910a26e61ad3b13",
}
G0_ROOT = "/home/ubuntu/trainium_freshstart_20260924_v2_a1_native5_g0_hybrid_observation_off_trn21_r1"
FORBIDDEN_AUTHORITIES = {
    "training": False, "evaluation": False, "request_materialization": False,
    "aws_lifecycle": False, "cache_cleanup": False, "checkpoint_pruning": False,
    "submission": False, "r4_staging": False,
}


class ContractError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_fd(fd: int) -> bytes:
    chunks = []
    while True:
        try:
            chunk = os.read(fd, 1024 * 1024)
        except InterruptedError:
            continue
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def read_sealed(path: str, *, require_mode: int | None = None) -> tuple[bytes, dict[str, int]]:
    """Read once through O_NOFOLLOW and bind bytes to stable regular metadata."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise ContractError(f"cannot safely open {path}: {exc}") from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ContractError(f"not a single-link regular file: {path}")
        if before.st_uid != os.getuid():
            raise ContractError(f"wrong owner: {path}")
        if require_mode is not None and stat.S_IMODE(before.st_mode) != require_mode:
            raise ContractError(f"wrong mode: {path}")
        data = _read_fd(fd)
        after = os.fstat(fd)
        named = os.lstat(path)
        fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in fields):
            raise ContractError(f"file mutated while read: {path}")
        if (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino):
            raise ContractError(f"path identity changed while read: {path}")
        if len(data) != before.st_size:
            raise ContractError(f"short read: {path}")
        return data, stat_record(before)
    finally:
        os.close(fd)


def read_json(path: str, *, require_mode: int | None = None) -> tuple[dict[str, Any], bytes]:
    data, _ = read_sealed(path, require_mode=require_mode)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ContractError(f"invalid JSON: {path}") from None
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be object: {path}")
    return value, data


def read_json_immediate_parent(path: str, *, require_mode: int | None = None) -> tuple[dict[str, Any], bytes]:
    """Read an input anchored below the fixed parent; no ancestor can be a link."""
    if Path(path).parent != Path(PARENT):
        raise ContractError("input is not an immediate child of fixed parent")
    parent_fd = open_safe_dir(PARENT)
    try:
        fd = os.open(Path(path).name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                    or before.st_uid != os.getuid()
                    or (require_mode is not None and stat.S_IMODE(before.st_mode) != require_mode)):
                raise ContractError("input metadata mismatch")
            data = _read_fd(fd)
            after = os.fstat(fd)
            named = os.stat(Path(path).name, dir_fd=parent_fd, follow_symlinks=False)
            if stat_record(before) != stat_record(after) or \
                    (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino) or len(data) != after.st_size:
                raise ContractError("input changed while read")
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ContractError("input JSON is invalid") from None
    if not isinstance(value, dict):
        raise ContractError("input JSON root must be object")
    return value, data


def stat_record(value: os.stat_result) -> dict[str, int]:
    return {
        "dev": value.st_dev, "ino": value.st_ino, "mode": stat.S_IMODE(value.st_mode),
        "uid": value.st_uid, "gid": value.st_gid, "nlink": value.st_nlink,
        "size": value.st_size, "blocks": value.st_blocks,
    }


def open_safe_dir(path: str) -> int:
    """Open an absolute directory by walking every component without links."""
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ContractError(f"directory path is not canonical absolute: {path}")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in Path(path).parts[1:]:
            nxt = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                          dir_fd=fd)
            os.close(fd)
            fd = nxt
        named = os.stat(path, follow_symlinks=False)
        opened = os.fstat(fd)
        if not stat.S_ISDIR(named.st_mode) or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
            raise ContractError(f"directory identity mismatch: {path}")
        return fd
    except BaseException:
        os.close(fd)
        raise


def fixed_target_map() -> dict[str, dict[str, Any]]:
    return {item["path"]: dict(item) for item in TARGETS}


def validate_request(request: dict[str, Any]) -> None:
    exact = {
        "schema", "state", "cleanup", "host", "expected_hostname", "parent", "origin",
        "minimum_post_cleanup_free_bytes", "targets", "preserve", "authorities",
        "evidence_commits", "package_head", "inventory_output", "receipt_output",
    }
    if set(request) != exact:
        raise ContractError("request schema mismatch")
    if request["schema"] != SCHEMA_REQUEST or request["state"] != "finalized":
        raise ContractError("request is not externally finalized")
    if request["cleanup"] is not True or request["host"] != HOST or request["expected_hostname"] != HOSTNAME:
        raise ContractError("request host/cleanup mismatch")
    if request["parent"] != PARENT or request["origin"] != ORIGIN:
        raise ContractError("request path/origin mismatch")
    if request["minimum_post_cleanup_free_bytes"] != MIN_POST_CLEANUP_FREE_BYTES:
        raise ContractError("request free-space threshold mismatch")
    if request["targets"] != list(TARGETS):
        raise ContractError("request exact target set/order mismatch")
    expected_preserve = {
        "r2_checkout": R2_ROOT,
        "r2_runtime_sha256": R2_RUNTIME_HASHES,
        "g0_checkpoint": G0_CHECKPOINT,
        "all_logs_results_evidence": True,
        "all_caches": True,
        "unknown_paths": True,
        "all_other_worktrees": True,
    }
    if request["preserve"] != expected_preserve or request["authorities"] != FORBIDDEN_AUTHORITIES:
        raise ContractError("request preservation/authority mismatch")
    if request["evidence_commits"] != {
        "r1": "e1550203a61c53cd7c73b0c543ced8a157312612", "r2": "5b392db4bc8c01a47da2f5969cb33df0c50ec52b",
        "r3": "5e45a564bba0d04b052345a7c9c1812ddb71ce51",
        "analysis": "83e533546fb8b53f5c8268d84f0ccd2b5c90e1a1",
    }:
        raise ContractError("request evidence provenance mismatch")
    if not isinstance(request["package_head"], str) or len(request["package_head"]) != 40:
        raise ContractError("request package_head is not finalized")
    for key in ("inventory_output", "receipt_output"):
        path = request[key]
        if not isinstance(path, str) or not path.startswith(PARENT + "/") or Path(path).parent != Path(PARENT):
            raise ContractError(f"unsafe {key}")


def _sealed_path_record(path: str, expected_sha: str | None = None,
                        expected_size: int | None = None) -> dict[str, Any]:
    data, metadata = read_sealed(path)
    digest = sha_bytes(data)
    if expected_sha is not None and digest != expected_sha:
        raise ContractError(f"hash mismatch: {path}")
    if expected_size is not None and len(data) != expected_size:
        raise ContractError(f"size mismatch: {path}")
    return {"path": path, "sha256": digest, "bytes": len(data), "metadata": metadata}


def _git(checkout_fd: int, *argv: str) -> str:
    proc = subprocess.run([GIT, "-C", f"/proc/self/fd/{checkout_fd}", *argv],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          pass_fds=(checkout_fd,), timeout=120, check=False)
    if proc.returncode != 0:
        raise ContractError(f"git {' '.join(argv)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _scan_tree_no_links(fd: int, relative: str = "") -> dict[str, int]:
    opened = os.fstat(fd)
    counts = {"directories": 1, "regular_files": 0, "logical_bytes": 0,
              "allocated_bytes": opened.st_blocks * 512}
    for name in sorted(os.listdir(fd)):
        if name in (".", ".."):
            raise ContractError("invalid directory entry")
        item = os.stat(name, dir_fd=fd, follow_symlinks=False)
        where = f"{relative}/{name}" if relative else name
        if stat.S_ISLNK(item.st_mode):
            raise ContractError(f"symlink forbidden in checkout: {where}")
        if stat.S_ISDIR(item.st_mode):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                            dir_fd=fd)
            try:
                sub = _scan_tree_no_links(child, where)
            finally:
                os.close(child)
            for key in counts:
                counts[key] += sub[key]
        elif stat.S_ISREG(item.st_mode) and item.st_nlink == 1:
            counts["regular_files"] += 1
            counts["logical_bytes"] += item.st_size
            counts["allocated_bytes"] += item.st_blocks * 512
        else:
            raise ContractError(f"non-regular or multi-link entry forbidden: {where}")
    return counts


def verify_checkout(parent_fd: int, target: dict[str, Any]) -> dict[str, Any]:
    name = Path(target["path"]).name
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode) or before.st_uid != os.getuid():
        raise ContractError(f"checkout is not an owned real directory: {target['path']}")
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        if _git(fd, "rev-parse", "HEAD").strip() != target["head"]:
            raise ContractError(f"checkout HEAD mismatch: {target['path']}")
        if _git(fd, "remote", "get-url", "origin").strip() != ORIGIN:
            raise ContractError(f"checkout origin mismatch: {target['path']}")
        # Include ignored entries: deleting an otherwise clean checkout must not
        # silently destroy an unknown ignored cache, log, checkpoint, or result.
        if _git(fd, "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching") != "":
            raise ContractError(f"checkout is dirty: {target['path']}")
        _git(fd, "fsck", "--full", "--strict")
        tree = _scan_tree_no_links(fd)
        after = os.fstat(fd)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or \
                (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino):
            raise ContractError(f"checkout identity changed: {target['path']}")
        return {"path": target["path"], "kind": "checkout", "head": target["head"],
                "origin": ORIGIN, "metadata": stat_record(after), "tree": tree}
    finally:
        os.close(fd)


def verify_bundle(parent_fd: int, target: dict[str, Any]) -> dict[str, Any]:
    name = Path(target["path"]).name
    fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid():
            raise ContractError(f"bundle metadata mismatch: {target['path']}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(fd)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat_record(before) != stat_record(after) or \
                (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino):
            raise ContractError(f"bundle identity changed: {target['path']}")
        if size != target["bytes"] or digest.hexdigest() != target["sha256"]:
            raise ContractError(f"bundle bytes/hash mismatch: {target['path']}")
        return {"path": target["path"], "kind": "bundle", "sha256": digest.hexdigest(),
                "bytes": size, "metadata": stat_record(after)}
    finally:
        os.close(fd)


def _sealed_beneath(root: str, relative: str, expected_sha: str,
                    expected_size: int | None = None) -> dict[str, Any]:
    if os.path.isabs(relative) or ".." in Path(relative).parts:
        raise ContractError("protected relative path escapes root")
    root_fd = open_safe_dir(root)
    fd = root_fd
    opened: list[int] = []
    try:
        if os.fstat(root_fd).st_uid != os.getuid():
            raise ContractError("protected root has wrong owner")
        parts = Path(relative).parts
        for component in parts[:-1]:
            nxt = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                          dir_fd=fd)
            opened.append(nxt)
            fd = nxt
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=fd)
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid():
                raise ContractError("protected file metadata mismatch")
            data = _read_fd(file_fd)
            after = os.fstat(file_fd)
            named = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
            if stat_record(before) != stat_record(after) or \
                    (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino):
                raise ContractError("protected file identity changed")
        finally:
            os.close(file_fd)
    finally:
        for item in reversed(opened):
            os.close(item)
        os.close(root_fd)
    if len(data) != before.st_size or sha_bytes(data) != expected_sha:
        raise ContractError("protected file bytes/hash mismatch")
    if expected_size is not None and len(data) != expected_size:
        raise ContractError("protected file size mismatch")
    return {"path": f"{root}/{relative}", "sha256": expected_sha, "bytes": len(data),
            "metadata": stat_record(before)}


def protected_records() -> dict[str, Any]:
    root_fd = open_safe_dir(R2_ROOT)
    try:
        root_stat = os.fstat(root_fd)
        if root_stat.st_uid != os.getuid():
            raise ContractError("preserved r2 checkout has wrong owner")
    finally:
        os.close(root_fd)
    r2 = {relative: _sealed_beneath(R2_ROOT, relative, digest)
          for relative, digest in R2_RUNTIME_HASHES.items()}
    g0_relative = os.path.relpath(G0_CHECKPOINT["path"], G0_ROOT)
    return {
        "r2_checkout": {"path": R2_ROOT, "metadata": stat_record(root_stat)},
        "r2_runtime": r2,
        "g0_checkpoint": _sealed_beneath(G0_ROOT, g0_relative, G0_CHECKPOINT["sha256"],
                                          G0_CHECKPOINT["bytes"]),
    }


def free_record(parent: str) -> dict[str, int]:
    info = os.statvfs(parent)
    return {"free_bytes": info.f_bavail * info.f_frsize, "free_inodes": info.f_favail,
            "fragment_size": info.f_frsize}


def _relevant_processes() -> list[dict[str, Any]]:
    matches = []
    terms = ("train.py", "torchrun", "neuronx-cc", "fresh_start_guard.py")
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            raw = Path(f"/proc/{name}/cmdline").read_bytes()
            argv = [piece.decode(errors="replace") for piece in raw.split(b"\0") if piece]
            status = Path(f"/proc/{name}/stat").read_text().split()
        except (OSError, IndexError):
            continue
        if status[2] != "Z" and any(any(term in arg for term in terms) for arg in argv):
            matches.append({"pid": int(name), "argv": argv})
    return matches


@contextlib.contextmanager
def host_quiescence() -> Iterator[dict[str, Any]]:
    if socket.gethostname() != HOSTNAME:
        raise ContractError("wrong host identity")
    lock_parent_fd = open_safe_dir(LOCK_PARENT)
    try:
        lock_fd = os.open(Path(LOCK_PATH).name, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                          dir_fd=lock_parent_fd)
    finally:
        os.close(lock_parent_fd)
    try:
        lock_stat = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_uid != os.getuid() or lock_stat.st_nlink != 1:
            raise ContractError("workflow lock metadata mismatch")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ContractError("workflow lock is held") from None
        processes = _relevant_processes()
        if processes:
            raise ContractError("training/compiler process is live")
        fuser = subprocess.run(["/usr/bin/fuser", "/dev/neuron0"], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, check=False, timeout=10)
        if fuser.returncode != 1 or fuser.stdout.strip() or fuser.stderr.strip():
            raise ContractError("Neuron device is held or fuser failed")
        neuron = subprocess.run([NEURON_LS, "--json-output"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, check=False, timeout=20)
        if neuron.returncode != 0:
            raise ContractError("neuron-ls failed")
        try:
            devices = json.loads(neuron.stdout)
        except json.JSONDecodeError:
            raise ContractError("neuron-ls emitted invalid JSON") from None
        if not isinstance(devices, list) or not devices or any(d.get("neuron_processes") for d in devices):
            raise ContractError("Neuron device inventory is not free")
        yield {"hostname": HOSTNAME, "workflow_lock_free_and_held_by_cleanup": True,
               "relevant_processes": [], "device_holders": [], "neuron_process_count": 0}
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _assert_request_head(request: dict[str, Any]) -> None:
    root = Path(__file__).resolve().parents[3]
    proc = subprocess.run([GIT, "-C", str(root), "rev-parse", "HEAD"], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or proc.stdout.strip() != request["package_head"]:
        raise ContractError("cleanup package HEAD differs from finalized request")
    dirty = subprocess.run([GIT, "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if dirty.returncode != 0 or dirty.stdout:
        raise ContractError("cleanup package checkout is dirty")


def _collect_inventory_locked(request: dict[str, Any], request_sha256: str,
                              health: dict[str, Any]) -> dict[str, Any]:
    validate_request(request)
    _assert_request_head(request)
    parent_fd = open_safe_dir(PARENT)
    try:
        parent_stat = os.fstat(parent_fd)
        before = free_record(PARENT)
        records = []
        for target in TARGETS:
            record = (verify_checkout if target["kind"] == "checkout" else verify_bundle)(parent_fd, target)
            if record["metadata"]["dev"] != parent_stat.st_dev:
                raise ContractError("target is not on cleanup parent filesystem")
            records.append(record)
        protected = protected_records()
        allocated = sum(
            record["metadata"]["blocks"] * 512 if record["kind"] == "bundle"
            else record["tree"]["allocated_bytes"]
            for record in records
        )
        projected = before["free_bytes"] + allocated
        if projected < MIN_POST_CLEANUP_FREE_BYTES:
            raise ContractError("conservative post-cleanup projection is below required free bytes")
        return {
            "schema": SCHEMA_INVENTORY, "status": "VERIFIED_READ_ONLY",
            "host": HOST, "hostname": HOSTNAME, "request_sha256": request_sha256,
            "package_head": request["package_head"], "parent": PARENT,
            "parent_metadata": stat_record(parent_stat), "free_before": before,
            "target_allocated_bytes": allocated,
            "projected_post_cleanup_free_bytes": projected,
            "health": health, "targets": records, "protected": protected,
            "deletion_performed": False, "training_authorized": False,
        }
    finally:
        os.close(parent_fd)


def collect_inventory(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    with host_quiescence() as health:
        return _collect_inventory_locked(request, request_sha256, health)


def _write_exclusive(path: str, value: dict[str, Any]) -> str:
    if Path(path).parent != Path(PARENT):
        raise ContractError("output is not an immediate child of fixed parent")
    parent_fd = open_safe_dir(PARENT)
    name = Path(path).name
    data = canonical(value)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                     0o600, dir_fd=parent_fd)
        try:
            view = memoryview(data)
            while view:
                try:
                    count = os.write(fd, view)
                except InterruptedError:
                    continue
                if count <= 0:
                    raise ContractError("short output write")
                view = view[count:]
            os.fsync(fd)
            created = os.fstat(fd)
        finally:
            os.close(fd)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (created.st_dev, created.st_ino) != (named.st_dev, named.st_ino) or named.st_nlink != 1:
            raise ContractError("output path identity changed")
        os.fsync(parent_fd)
        return sha_bytes(data)
    finally:
        os.close(parent_fd)


def validate_authorization(auth: dict[str, Any], request: dict[str, Any], request_sha: str,
                           inventory: dict[str, Any], inventory_sha: str) -> None:
    exact = {"schema", "state", "executable", "cleanup", "authorized_head", "request_sha256",
             "inventory_sha256", "host", "expected_hostname", "targets", "authorities",
             "minimum_post_cleanup_free_bytes", "independent_verifier_commit", "human_authorization"}
    if set(auth) != exact or auth["schema"] != SCHEMA_AUTH or auth["state"] != "finalized":
        raise ContractError("authorization schema/state mismatch")
    if auth["executable"] is not True or auth["cleanup"] is not True or auth["human_authorization"] is not True:
        raise ContractError("authorization is non-executable")
    if auth["authorized_head"] != request["package_head"] or auth["request_sha256"] != request_sha:
        raise ContractError("authorization request/head mismatch")
    if auth["inventory_sha256"] != inventory_sha or inventory["request_sha256"] != request_sha:
        raise ContractError("authorization inventory mismatch")
    if auth["host"] != HOST or auth["expected_hostname"] != HOSTNAME or auth["targets"] != list(TARGETS):
        raise ContractError("authorization host/target mismatch")
    if auth["authorities"] != FORBIDDEN_AUTHORITIES or \
            auth["minimum_post_cleanup_free_bytes"] != MIN_POST_CLEANUP_FREE_BYTES:
        raise ContractError("authorization scope mismatch")
    if not isinstance(auth["independent_verifier_commit"], str) or len(auth["independent_verifier_commit"]) != 40:
        raise ContractError("authorization lacks independent verifier identity")


def _identity_key(record: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    m = record["metadata"]
    return m["dev"], m["ino"], m["mode"], m["uid"], m["gid"], m["nlink"]


def _unlink_tree(fd: int) -> None:
    for name in sorted(os.listdir(fd)):
        item = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISDIR(item.st_mode):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                            dir_fd=fd)
            try:
                _unlink_tree(child)
                os.fsync(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=fd)
        elif stat.S_ISREG(item.st_mode) and item.st_nlink == 1:
            os.unlink(name, dir_fd=fd)
        else:
            raise ContractError("tree changed to unsafe entry during deletion")


def _delete_verified(parent_fd: int, target: dict[str, Any], inventory_record: dict[str, Any]) -> None:
    name = Path(target["path"]).name
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino, stat.S_IMODE(current.st_mode), current.st_uid,
            current.st_gid, current.st_nlink) != _identity_key(inventory_record):
        raise ContractError(f"target identity drift before delete: {target['path']}")
    if target["kind"] == "bundle":
        os.unlink(name, dir_fd=parent_fd)
    else:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            _unlink_tree(fd)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise ContractError(f"target still exists after delete: {target['path']}")


def execute(request: dict[str, Any], request_sha: str, inventory: dict[str, Any], inventory_sha: str,
            authorization: dict[str, Any], authorization_sha: str) -> dict[str, Any]:
    validate_request(request)
    validate_authorization(authorization, request, request_sha, inventory, inventory_sha)
    _assert_request_head(request)
    if inventory.get("schema") != SCHEMA_INVENTORY or inventory.get("status") != "VERIFIED_READ_ONLY":
        raise ContractError("invalid inventory")
    with host_quiescence() as health:
        fresh = _collect_inventory_locked(request, request_sha, health)
        for key in ("parent_metadata", "targets", "protected"):
            if fresh[key] != inventory[key]:
                raise ContractError(f"read-only inventory drift: {key}")
        execution_free_before = fresh["free_before"]
        parent_fd = open_safe_dir(PARENT)
        deleted: list[str] = []
        try:
            for target, record in zip(TARGETS, inventory["targets"], strict=True):
                _delete_verified(parent_fd, target, record)
                deleted.append(target["path"])
            protected_after = protected_records()
            if protected_after != inventory["protected"]:
                raise ContractError("preserved r2/G0 identity changed")
            free_after = free_record(PARENT)
            if free_after["fragment_size"] != execution_free_before["fragment_size"]:
                raise ContractError("filesystem geometry changed during cleanup")
            if free_after["free_bytes"] < execution_free_before["free_bytes"]:
                raise ContractError("free bytes decreased after cleanup")
            if free_after["free_bytes"] < MIN_POST_CLEANUP_FREE_BYTES:
                raise ContractError("post-cleanup free bytes below r4 staging requirement")
            for target in TARGETS:
                try:
                    os.stat(Path(target["path"]).name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise ContractError(f"deleted target reappeared: {target['path']}")
            return {
                "schema": SCHEMA_RECEIPT, "status": "PASS", "cleanup_performed": True,
                "host": HOST, "hostname": HOSTNAME, "package_head": request["package_head"],
                "request_sha256": request_sha, "inventory_sha256": inventory_sha,
                "authorization_sha256": authorization_sha, "health": health,
                "deleted_exact_paths": deleted, "targets_absent": True,
                "protected_after": protected_after,
                "authorized_inventory_free_before": inventory["free_before"],
                "free_before": execution_free_before,
                "free_after": free_after, "minimum_post_cleanup_free_bytes": MIN_POST_CLEANUP_FREE_BYTES,
                "same_filesystem": True, "parent_fsynced_after_each_target": True,
                "training_authorized": False, "evaluation_authorized": False,
                "r4_staging_authorized": False, "cache_cleanup_performed": False,
                "checkpoint_pruning_performed": False,
            }
        finally:
            os.close(parent_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--request", required=True)
    exe = sub.add_parser("execute")
    exe.add_argument("--request", required=True)
    exe.add_argument("--inventory", required=True)
    exe.add_argument("--authorization", required=True)
    args = parser.parse_args(argv)
    try:
        request, request_bytes = read_json_immediate_parent(args.request, require_mode=0o600)
        request_sha = sha_bytes(request_bytes)
        if args.command == "inventory":
            value = collect_inventory(request, request_sha)
            digest = _write_exclusive(request["inventory_output"], value)
        else:
            inventory, inventory_bytes = read_json_immediate_parent(args.inventory, require_mode=0o600)
            authorization, authorization_bytes = read_json_immediate_parent(args.authorization, require_mode=0o600)
            value = execute(request, request_sha, inventory, sha_bytes(inventory_bytes), authorization,
                            sha_bytes(authorization_bytes))
            digest = _write_exclusive(request["receipt_output"], value)
        print(json.dumps({"status": value["status"], "sha256": digest}, sort_keys=True))
        return 0
    except (ContractError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
