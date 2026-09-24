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
import signal
import socket
import stat
import subprocess
import sys
from typing import Any, Callable, Iterator


SCHEMA_REQUEST = "a1-native-g1-consumed-staging-cleanup-request-v2"
SCHEMA_INVENTORY = "a1-native-g1-consumed-staging-cleanup-inventory-v2"
SCHEMA_AUTH = "a1-native-g1-consumed-staging-cleanup-authorization-v2"
SCHEMA_RECEIPT = "a1-native-g1-consumed-staging-cleanup-receipt-v2"
SCHEMA_JOURNAL = "a1-native-g1-consumed-staging-cleanup-journal-v1"
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


# Tests replace this hook to force mutations at destructive boundaries.  It is
# None in production; the operation never accepts a callback from CLI/request.
_BOUNDARY_HOOK: Callable[[str, dict[str, Any]], None] | None = None


def _boundary(name: str, **context: Any) -> None:
    if _BOUNDARY_HOOK is not None:
        _BOUNDARY_HOOK(name, context)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _first_difference(left: Any, right: Any, path: str = "$") -> str:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: keys {sorted(set(left) ^ set(right))}"
        for key in sorted(left):
            if left[key] != right[key]:
                return _first_difference(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            if a != b:
                return _first_difference(a, b, f"{path}[{index}]")
    return f"{path}: {left!r} != {right!r}"


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


def read_json_immediate_parent(path: str, *, require_mode: int | None = None) -> tuple[dict[str, Any], bytes]:
    """Read an input anchored below the fixed parent; no ancestor can be a link."""
    if Path(path).parent != Path(PARENT):
        raise ContractError("input is not an immediate child of fixed parent")
    parent_fd = open_safe_dir(PARENT)
    try:
        fd = os.open(Path(path).name,
                     os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
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


def directory_identity(value: os.stat_result) -> dict[str, int]:
    """Stable anchor identity; child creation legitimately changes size/times."""
    return {"dev": value.st_dev, "ino": value.st_ino, "mode": stat.S_IMODE(value.st_mode),
            "uid": value.st_uid, "gid": value.st_gid}


def mount_id(fd: int) -> int:
    try:
        lines = Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines()
        return int(next(line.split(":", 1)[1] for line in lines if line.startswith("mnt_id:")))
    except (OSError, StopIteration, ValueError):
        raise ContractError("cannot authenticate filesystem mount identity") from None


def parent_identity(fd: int) -> dict[str, int]:
    return {**directory_identity(os.fstat(fd)), "mount_id": mount_id(fd)}


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


def validate_request(request: dict[str, Any]) -> None:
    exact = {
        "schema", "state", "cleanup", "host", "expected_hostname", "parent", "origin",
        "minimum_post_cleanup_free_bytes", "targets", "preserve", "authorities",
        "evidence_commits", "package_head", "inventory_output", "journal_output", "receipt_output",
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
    outputs = []
    for key in ("inventory_output", "journal_output", "receipt_output"):
        path = request[key]
        if not isinstance(path, str) or not path.startswith(PARENT + "/") or Path(path).parent != Path(PARENT):
            raise ContractError(f"unsafe {key}")
        outputs.append(path)
    if len(set(outputs)) != len(outputs):
        raise ContractError("request output paths must be distinct")


def _git(checkout_fd: int, *argv: str) -> str:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    proc = subprocess.run([GIT, "-C", f"/proc/self/fd/{checkout_fd}", *argv],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          pass_fds=(checkout_fd,), timeout=120, check=False, env=env)
    if proc.returncode != 0:
        raise ContractError(f"git {' '.join(argv)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _hash_fd(fd: int) -> tuple[str, int, dict[str, int]]:
    """Hash exact bytes through one fd and reject in-read mutation."""
    os.lseek(fd, 0, os.SEEK_SET)
    before = os.fstat(fd)
    digest = hashlib.sha256()
    size = 0
    while True:
        try:
            chunk = os.read(fd, 1024 * 1024)
        except InterruptedError:
            continue
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(fd)
    if stat_record(before) != stat_record(after) or size != after.st_size:
        raise ContractError("file changed while hashing")
    return digest.hexdigest(), size, stat_record(after)


def _assert_named_inode(parent_fd: int, name: str, opened: os.stat_result) -> None:
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
        raise ContractError(f"name-to-inode binding changed: {name}")


def _inventory_tree(fd: int, relative: str = "", root_dev: int | None = None,
                    root_mount: int | None = None) -> dict[str, Any]:
    """Return the complete authenticated directory/file membership contract."""
    opened = os.fstat(fd)
    root_dev = opened.st_dev if root_dev is None else root_dev
    root_mount = mount_id(fd) if root_mount is None else root_mount
    if not stat.S_ISDIR(opened.st_mode) or opened.st_dev != root_dev or opened.st_uid != os.getuid():
        raise ContractError(f"unsafe checkout directory: {relative or '.'}")
    if mount_id(fd) != root_mount:
        raise ContractError(f"mount substitution in checkout: {relative or '.'}")
    entries: list[dict[str, Any]] = [
        {"path": relative, "kind": "directory", "metadata": stat_record(opened),
         "mount_id": root_mount}
    ]
    names = sorted(os.listdir(fd))
    if any(name in (".", "..") or "/" in name or "\x00" in name for name in names):
        raise ContractError("invalid directory entry")
    for name in names:
        where = f"{relative}/{name}" if relative else name
        try:
            child = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                            dir_fd=fd)
        except OSError as exc:
            raise ContractError(f"cannot no-follow open checkout entry {where}: {exc}") from None
        try:
            item = os.fstat(child)
            _assert_named_inode(fd, name, item)
            if item.st_dev != root_dev or item.st_uid != os.getuid():
                raise ContractError(f"mount/owner substitution in checkout: {where}")
            if stat.S_ISDIR(item.st_mode):
                entries.extend(_inventory_tree(child, where, root_dev, root_mount)["entries"])
            elif stat.S_ISREG(item.st_mode) and item.st_nlink == 1:
                digest, size, metadata = _hash_fd(child)
                _assert_named_inode(fd, name, os.fstat(child))
                if mount_id(child) != root_mount:
                    raise ContractError(f"mount substitution in checkout: {where}")
                entries.append({"path": where, "kind": "file", "metadata": metadata,
                                "mount_id": root_mount,
                                "bytes": size, "sha256": digest})
            else:
                raise ContractError(f"symlink, hardlink, or nonregular checkout entry: {where}")
        finally:
            os.close(child)
    if sorted(os.listdir(fd)) != names:
        raise ContractError(f"checkout membership changed while inventoried: {relative or '.'}")
    total = sum(entry["metadata"]["blocks"] * 512 for entry in entries)
    logical = sum(entry.get("bytes", 0) for entry in entries)
    return {"entries": entries, "directories": sum(x["kind"] == "directory" for x in entries),
            "regular_files": sum(x["kind"] == "file" for x in entries),
            "logical_bytes": logical, "allocated_bytes": total,
            "manifest_sha256": sha_bytes(canonical(entries))}


def verify_checkout(parent_fd: int, target: dict[str, Any]) -> dict[str, Any]:
    name = Path(target["path"]).name
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        before = os.fstat(fd)
        _assert_named_inode(parent_fd, name, before)
        if not stat.S_ISDIR(before.st_mode) or before.st_uid != os.getuid():
            raise ContractError(f"checkout is not an owned real directory: {target['path']}")
        if mount_id(fd) != mount_id(parent_fd):
            raise ContractError(f"checkout is a mount substitution: {target['path']}")
        if _git(fd, "rev-parse", "HEAD").strip() != target["head"]:
            raise ContractError(f"checkout HEAD mismatch: {target['path']}")
        if _git(fd, "remote", "get-url", "origin").strip() != ORIGIN:
            raise ContractError(f"checkout origin mismatch: {target['path']}")
        # Include ignored entries: deleting an otherwise clean checkout must not
        # silently destroy an unknown ignored cache, log, checkpoint, or result.
        if _git(fd, "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching") != "":
            raise ContractError(f"checkout is dirty: {target['path']}")
        _git(fd, "fsck", "--full", "--strict")
        tree = _inventory_tree(fd)
        after = os.fstat(fd)
        _assert_named_inode(parent_fd, name, after)
        if stat_record(before) != stat_record(after):
            raise ContractError(f"checkout identity changed: {target['path']}")
        return {"path": target["path"], "kind": "checkout", "head": target["head"],
                "origin": ORIGIN, "metadata": stat_record(after), "mount_id": mount_id(fd),
                "tree": tree}
    finally:
        os.close(fd)


def verify_bundle(parent_fd: int, target: dict[str, Any]) -> dict[str, Any]:
    name = Path(target["path"]).name
    fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                 dir_fd=parent_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid():
            raise ContractError(f"bundle metadata mismatch: {target['path']}")
        _assert_named_inode(parent_fd, name, before)
        if mount_id(fd) != mount_id(parent_fd):
            raise ContractError(f"bundle is a mount substitution: {target['path']}")
        digest, size, metadata = _hash_fd(fd)
        _assert_named_inode(parent_fd, name, os.fstat(fd))
        if size != target["bytes"] or digest != target["sha256"]:
            raise ContractError(f"bundle bytes/hash mismatch: {target['path']}")
        return {"path": target["path"], "kind": "bundle", "sha256": digest,
                "bytes": size, "metadata": metadata, "mount_id": mount_id(fd)}
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
        root_mount = mount_id(root_fd)
        parts = Path(relative).parts
        for component in parts[:-1]:
            nxt = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                          dir_fd=fd)
            if mount_id(nxt) != root_mount:
                os.close(nxt)
                raise ContractError("mount substitution in protected path")
            opened.append(nxt)
            fd = nxt
        file_fd = os.open(parts[-1],
                          os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid():
                raise ContractError("protected file metadata mismatch")
            if mount_id(file_fd) != root_mount:
                raise ContractError("mount substitution in protected file")
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
            "metadata": stat_record(before), "mount_id": root_mount}


def protected_records() -> dict[str, Any]:
    root_fd = open_safe_dir(R2_ROOT)
    try:
        root_stat = os.fstat(root_fd)
        root_mount = mount_id(root_fd)
        if root_stat.st_uid != os.getuid():
            raise ContractError("preserved r2 checkout has wrong owner")
    finally:
        os.close(root_fd)
    r2 = {relative: _sealed_beneath(R2_ROOT, relative, digest)
          for relative, digest in R2_RUNTIME_HASHES.items()}
    g0_relative = os.path.relpath(G0_CHECKPOINT["path"], G0_ROOT)
    return {
        "r2_checkout": {"path": R2_ROOT, "metadata": stat_record(root_stat),
                        "mount_id": root_mount},
        "r2_runtime": r2,
        "g0_checkpoint": _sealed_beneath(G0_ROOT, g0_relative, G0_CHECKPOINT["sha256"],
                                          G0_CHECKPOINT["bytes"]),
    }


def free_record_fd(parent_fd: int) -> dict[str, int]:
    info = os.fstatvfs(parent_fd)
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
                              health: dict[str, Any], parent_fd: int | None = None) -> dict[str, Any]:
    validate_request(request)
    _assert_request_head(request)
    owns_parent_fd = parent_fd is None
    if parent_fd is None:
        parent_fd = open_safe_dir(PARENT)
    try:
        parent_stat = os.fstat(parent_fd)
        before = free_record_fd(parent_fd)
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
            "parent_metadata": parent_identity(parent_fd), "free_before": before,
            "target_allocated_bytes": allocated,
            "projected_post_cleanup_free_bytes": projected,
            "health": health, "targets": records, "protected": protected,
            "deletion_performed": False, "training_authorized": False,
        }
    finally:
        if owns_parent_fd:
            os.close(parent_fd)


def collect_inventory(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    with host_quiescence() as health:
        return _collect_inventory_locked(request, request_sha256, health)


def _write_exclusive(path: str, value: dict[str, Any],
                     expected_parent_identity: dict[str, int] | None = None) -> str:
    if Path(path).parent != Path(PARENT):
        raise ContractError("output is not an immediate child of fixed parent")
    parent_fd = open_safe_dir(PARENT)
    name = Path(path).name
    data = canonical(value)
    try:
        if expected_parent_identity is not None and parent_identity(parent_fd) != expected_parent_identity:
            raise ContractError("output parent identity differs from operation anchor")
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
        _boundary("after_output_write_before_readback", output=name, parent_fd=parent_fd)
        # Seal the published receipt/inventory by reading the exact final inode.
        sealed_fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                            dir_fd=parent_fd)
        try:
            sealed = os.fstat(sealed_fd)
            if ((sealed.st_dev, sealed.st_ino) != (created.st_dev, created.st_ino)
                    or sealed.st_nlink != 1 or stat.S_IMODE(sealed.st_mode) != 0o600):
                raise ContractError("published output inode changed")
            readback = _read_fd(sealed_fd)
            final = os.fstat(sealed_fd)
            _assert_named_inode(parent_fd, name, final)
            if stat_record(sealed) != stat_record(final) or readback != data:
                raise ContractError("published output readback mismatch")
        finally:
            os.close(sealed_fd)
        return sha_bytes(readback)
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


def _full_write(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            count = os.write(fd, view)
        except InterruptedError:
            continue
        if count <= 0:
            raise ContractError("short durable write")
        view = view[count:]


class Journal:
    """Append-only, O_EXCL evidence created before the first destructive rename."""

    def __init__(self, parent_fd: int, path: str, header: dict[str, Any]):
        if Path(path).parent != Path(PARENT):
            raise ContractError("unsafe journal path")
        self.parent_fd = parent_fd
        self.name = Path(path).name
        self.fd = os.open(self.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                          0o600, dir_fd=parent_fd)
        self.created = os.fstat(self.fd)
        if not stat.S_ISREG(self.created.st_mode) or self.created.st_nlink != 1:
            raise ContractError("journal creation metadata mismatch")
        self.append("INTENT", **header)
        os.fsync(parent_fd)

    def append(self, event: str, **fields: Any) -> None:
        record = {"schema": SCHEMA_JOURNAL, "sequence": os.lseek(self.fd, 0, os.SEEK_CUR),
                  "event": event, **fields}
        _full_write(self.fd, canonical(record))
        os.fsync(self.fd)

    def seal(self) -> dict[str, Any]:
        os.fsync(self.fd)
        os.close(self.fd)
        self.fd = -1
        os.fsync(self.parent_fd)
        fd = os.open(self.name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=self.parent_fd)
        try:
            current = os.fstat(fd)
            if ((current.st_dev, current.st_ino) != (self.created.st_dev, self.created.st_ino)
                    or current.st_nlink != 1 or stat.S_IMODE(current.st_mode) != 0o600):
                raise ContractError("journal inode changed")
            data = _read_fd(fd)
            after = os.fstat(fd)
            _assert_named_inode(self.parent_fd, self.name, after)
            if stat_record(current) != stat_record(after):
                raise ContractError("journal mutated during seal")
            return {"path": f"{PARENT}/{self.name}", "sha256": sha_bytes(data),
                    "bytes": len(data), "metadata": stat_record(after)}
        finally:
            os.close(fd)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def _metadata_equal(actual: os.stat_result, expected: dict[str, int], *, renamed: bool = False) -> bool:
    observed = stat_record(actual)
    return observed == expected


def _require_absent(parent_fd: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise ContractError(f"path must be absent: {name}")


def _restore_quarantine(parent_fd: int, quarantine: str, original: str,
                        expected_dev_ino: tuple[int, int]) -> bool:
    """Restore only the quarantined inode; never remove a replacement."""
    try:
        current = os.stat(quarantine, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (current.st_dev, current.st_ino) != expected_dev_ino:
        return False
    try:
        os.stat(original, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        os.rename(quarantine, original, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
        return True
    return False


def _restore_untrusted_quarantine(parent_fd: int, quarantine: str, original: str) -> bool:
    """Undo our atomic rename without interpreting or deleting the moved entry."""
    try:
        os.stat(quarantine, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    try:
        os.stat(original, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        os.rename(quarantine, original, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
        return True
    return False


def _tree_children(tree: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    records = {entry["path"]: entry for entry in tree["entries"]}
    if "" not in records or records[""]["kind"] != "directory" or len(records) != len(tree["entries"]):
        raise ContractError("invalid authenticated checkout manifest")
    children: dict[str, list[str]] = {path: [] for path, item in records.items()
                                      if item["kind"] == "directory"}
    for path in records:
        if not path:
            continue
        parent = str(Path(path).parent)
        parent = "" if parent == "." else parent
        if parent not in children:
            raise ContractError("orphan in authenticated checkout manifest")
        children[parent].append(Path(path).name)
    for value in children.values():
        value.sort()
    return records, children


def _validate_open_file(fd: int, record: dict[str, Any], *, renamed: bool = False) -> None:
    current = os.fstat(fd)
    if (not stat.S_ISREG(current.st_mode) or current.st_nlink != 1
            or not _metadata_equal(current, record["metadata"], renamed=renamed)):
        raise ContractError(f"file identity/schema drift: {record['path']}")
    if "mount_id" in record and mount_id(fd) != record["mount_id"]:
        raise ContractError(f"file mount identity drift: {record['path']}")
    digest, size, _ = _hash_fd(fd)
    if digest != record["sha256"] or size != record["bytes"]:
        raise ContractError(f"file bytes drift: {record['path']}")


def _validate_open_dir(fd: int, record: dict[str, Any], *, renamed: bool = False) -> None:
    current = os.fstat(fd)
    if (not stat.S_ISDIR(current.st_mode)
            or not _metadata_equal(current, record["metadata"], renamed=renamed)):
        raise ContractError(f"directory identity/schema drift: {record['path']}")
    if "mount_id" in record and mount_id(fd) != record["mount_id"]:
        raise ContractError(f"directory mount identity drift: {record['path']}")


def _quarantine_name(prefix: str, identity: str) -> str:
    return f".cleanup-{prefix}-{identity}"


def _delete_manifest_dir(fd: int, relative: str, records: dict[str, dict[str, Any]],
                         children: dict[str, list[str]], journal: Journal,
                         target_state: dict[str, Any]) -> None:
    remaining = list(children[relative])
    for name in list(remaining):
        if sorted(os.listdir(fd)) != remaining:
            raise ContractError(f"unknown/missing checkout entry before deletion: {relative or '.'}")
        path = f"{relative}/{name}" if relative else name
        record = records[path]
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        if record["kind"] == "directory":
            flags |= os.O_DIRECTORY
        child = os.open(name, flags, dir_fd=fd)
        qname = _quarantine_name("entry", sha_bytes(path.encode())[:24])
        quarantined = False
        entry_deleted = False
        try:
            if record["kind"] == "directory":
                _validate_open_dir(child, record)
            else:
                _validate_open_file(child, record)
            _assert_named_inode(fd, name, os.fstat(child))
            _require_absent(fd, qname)
            _boundary("before_entry_quarantine", relative=path, original=name, quarantine=qname, dir_fd=fd)
            os.rename(name, qname, src_dir_fd=fd, dst_dir_fd=fd)
            quarantined = True
            opened = os.fstat(child)
            try:
                _assert_named_inode(fd, qname, opened)
                _require_absent(fd, name)
            except BaseException:
                _restore_untrusted_quarantine(fd, qname, name)
                raise
            os.fsync(fd)
            journal.append("ENTRY_QUARANTINED", target=target_state["path"], relative=path,
                           dev=opened.st_dev, ino=opened.st_ino)
            _boundary("after_entry_quarantine", relative=path, quarantine=qname, dir_fd=fd)
            if record["kind"] == "directory":
                _validate_open_dir(child, record, renamed=True)
                _delete_manifest_dir(child, path, records, children, journal, target_state)
                if os.listdir(child):
                    raise ContractError(f"authenticated directory did not become empty: {path}")
                _assert_named_inode(fd, qname, os.fstat(child))
                os.rmdir(qname, dir_fd=fd)
            else:
                _boundary("before_entry_final_hash", relative=path, quarantine=qname, dir_fd=fd)
                _validate_open_file(child, record, renamed=True)
                _assert_named_inode(fd, qname, os.fstat(child))
                # No callback or path lookup occurs between this binding and unlink.
                os.unlink(qname, dir_fd=fd)
            entry_deleted = True
            target_state["destructive_started"] = True
            os.fsync(fd)
            journal.append("ENTRY_DELETED", target=target_state["path"], relative=path,
                           dev=opened.st_dev, ino=opened.st_ino)
            _boundary("after_entry_delete", relative=path, dir_fd=fd)
        except BaseException:
            if quarantined and not entry_deleted and not target_state["destructive_started"]:
                _restore_quarantine(fd, qname, name, (os.fstat(child).st_dev, os.fstat(child).st_ino))
            raise
        finally:
            os.close(child)
        remaining.remove(name)
    if os.listdir(fd):
        raise ContractError(f"unknown checkout entries remain: {relative or '.'}")


def _target_outcome(parent_fd: int, state: dict[str, Any]) -> dict[str, Any]:
    result = {"path": state["path"], "deleted": state["deleted"],
              "destructive_started": state["destructive_started"]}
    for key, name in (("original", state["name"]), ("quarantine", state["quarantine"])):
        try:
            item = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            result[key] = {"present": True, "metadata": stat_record(item)}
        except FileNotFoundError:
            result[key] = {"present": False}
    if not result["original"]["present"] and not result["quarantine"]["present"]:
        state["deleted"] = True
        state["destructive_started"] = True
        result["deleted"] = True
        result["destructive_started"] = True
    return result


def _delete_target_quarantined(parent_fd: int, target: dict[str, Any], record: dict[str, Any],
                               index: int, request_sha: str, journal: Journal,
                               state: dict[str, Any]) -> dict[str, Any]:
    name = Path(target["path"]).name
    qname = _quarantine_name("target", f"{request_sha[:16]}-{index}")
    _require_absent(parent_fd, qname)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    if target["kind"] == "checkout":
        flags |= os.O_DIRECTORY
    fd = os.open(name, flags, dir_fd=parent_fd)
    quarantined = False
    try:
        if target["kind"] == "bundle":
            _validate_open_file(fd, record)
        else:
            _validate_open_dir(fd, {"path": "", "metadata": record["metadata"],
                                    "mount_id": record["mount_id"]})
            current_tree = _inventory_tree(fd)
            if current_tree != record["tree"]:
                raise ContractError(f"checkout manifest drift before quarantine: {target['path']}")
        opened = os.fstat(fd)
        _assert_named_inode(parent_fd, name, opened)
        _boundary("after_target_open_before_quarantine", target=target["path"], original=name,
                  quarantine=qname, parent_fd=parent_fd, target_fd=fd)
        os.rename(name, qname, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        quarantined = True
        try:
            _assert_named_inode(parent_fd, qname, opened)
            _require_absent(parent_fd, name)
        except BaseException:
            _restore_untrusted_quarantine(parent_fd, qname, name)
            raise
        os.fsync(parent_fd)
        journal.append("TARGET_QUARANTINED", path=target["path"], quarantine=qname,
                       dev=opened.st_dev, ino=opened.st_ino)
        _boundary("after_target_quarantine", target=target["path"], quarantine=qname,
                  parent_fd=parent_fd, target_fd=fd)
        if target["kind"] == "bundle":
            # Required post-quarantine and immediately-pre-unlink exact same-fd check.
            _validate_open_file(fd, record, renamed=True)
            _boundary("before_bundle_final_hash", target=target["path"], quarantine=qname,
                      parent_fd=parent_fd, target_fd=fd)
            _validate_open_file(fd, record, renamed=True)
            _assert_named_inode(parent_fd, qname, os.fstat(fd))
            os.unlink(qname, dir_fd=parent_fd)
        else:
            current_tree = _inventory_tree(fd)
            actual_entries = current_tree["entries"]
            expected_entries = record["tree"]["entries"]
            if actual_entries != expected_entries:
                raise ContractError(f"checkout manifest drift after quarantine: {target['path']}")
            _boundary("after_checkout_quarantine_validation", target=target["path"], quarantine=qname,
                      parent_fd=parent_fd, target_fd=fd)
            records, children = _tree_children(record["tree"])
            _delete_manifest_dir(fd, "", records, children, journal, state)
            if os.listdir(fd):
                raise ContractError("checkout quarantine is not empty")
            _assert_named_inode(parent_fd, qname, os.fstat(fd))
            os.rmdir(qname, dir_fd=parent_fd)
        state["deleted"] = True
        state["destructive_started"] = True
        os.fsync(parent_fd)
        journal.append("TARGET_DELETED", path=target["path"], dev=opened.st_dev, ino=opened.st_ino)
        _boundary("after_target_delete", target=target["path"], parent_fd=parent_fd)
        _require_absent(parent_fd, name)
        _require_absent(parent_fd, qname)
        return state
    except BaseException:
        if quarantined and not state["destructive_started"]:
            restored = _restore_quarantine(parent_fd, qname, name, (os.fstat(fd).st_dev, os.fstat(fd).st_ino))
            if restored:
                journal.append("TARGET_RESTORED_AFTER_FAILURE", path=target["path"])
        raise
    finally:
        os.close(fd)


def _best_effort_failure_state(parent_fd: int, inventory: dict[str, Any], states: list[dict[str, Any]],
                               error: BaseException) -> dict[str, Any]:
    outcomes = [_target_outcome(parent_fd, item) for item in states]
    try:
        protected = protected_records()
        protected_ok = protected == inventory["protected"]
    except BaseException as protected_error:
        protected = {"error": f"{type(protected_error).__name__}: {protected_error}"}
        protected_ok = False
    try:
        free = free_record_fd(parent_fd)
    except BaseException as free_error:
        free = {"error": f"{type(free_error).__name__}: {free_error}"}
    return {"error": f"{type(error).__name__}: {error}",
            "target_outcomes": outcomes,
            "protected_postflight": protected, "protected_exact": protected_ok,
            "free_after_failure": free}


def execute(request: dict[str, Any], request_sha: str, inventory: dict[str, Any], inventory_sha: str,
            authorization: dict[str, Any], authorization_sha: str) -> dict[str, Any]:
    validate_request(request)
    validate_authorization(authorization, request, request_sha, inventory, inventory_sha)
    _assert_request_head(request)
    if inventory.get("schema") != SCHEMA_INVENTORY or inventory.get("status") != "VERIFIED_READ_ONLY":
        raise ContractError("invalid inventory")
    with host_quiescence() as health:
        parent_fd = open_safe_dir(PARENT)
        journal: Journal | None = None
        states: list[dict[str, Any]] = []
        try:
            fresh = _collect_inventory_locked(request, request_sha, health, parent_fd=parent_fd)
            for key in ("parent_metadata", "targets", "protected"):
                if fresh[key] != inventory[key]:
                    raise ContractError(f"read-only inventory drift: {key}: "
                                        f"{_first_difference(fresh[key], inventory[key])}")
            execution_free_before = fresh["free_before"]
            _require_absent(parent_fd, Path(request["journal_output"]).name)
            _require_absent(parent_fd, Path(request["receipt_output"]).name)
            for index in range(len(TARGETS)):
                _require_absent(parent_fd, _quarantine_name("target", f"{request_sha[:16]}-{index}"))
            journal = Journal(parent_fd, request["journal_output"], {
                "package_head": request["package_head"], "request_sha256": request_sha,
                "inventory_sha256": inventory_sha, "authorization_sha256": authorization_sha,
                "exact_targets": [item["path"] for item in TARGETS],
                "free_before": execution_free_before,
            })
            for index, (target, record) in enumerate(zip(TARGETS, inventory["targets"], strict=True)):
                state = {"path": target["path"], "name": Path(target["path"]).name,
                         "quarantine": _quarantine_name("target", f"{request_sha[:16]}-{index}"),
                         "deleted": False, "destructive_started": False}
                states.append(state)
                result = _delete_target_quarantined(parent_fd, target, record, index, request_sha,
                                                    journal, state)
                states[-1] = result
            protected_after = protected_records()
            if protected_after != inventory["protected"]:
                raise ContractError("preserved r2/G0 identity changed")
            for target in TARGETS:
                _require_absent(parent_fd, Path(target["path"]).name)
            os.fsync(parent_fd)
            free_after = free_record_fd(parent_fd)
            if free_after["fragment_size"] != execution_free_before["fragment_size"]:
                raise ContractError("filesystem geometry changed during cleanup")
            if free_after["free_bytes"] < execution_free_before["free_bytes"]:
                raise ContractError("free bytes decreased after cleanup")
            if free_after["free_bytes"] < MIN_POST_CLEANUP_FREE_BYTES:
                raise ContractError("post-cleanup free bytes below r4 staging requirement")
            journal.append("PASS_READY", deleted_exact_paths=[item["path"] for item in TARGETS],
                           free_after=free_after, protected_exact=True)
            journal_seal = journal.seal()
            journal = None
            return {
                "schema": SCHEMA_RECEIPT, "status": "PASS", "cleanup_performed": True,
                "host": HOST, "hostname": HOSTNAME, "package_head": request["package_head"],
                "request_sha256": request_sha, "inventory_sha256": inventory_sha,
                "authorization_sha256": authorization_sha, "health": health,
                "journal": journal_seal, "parent_metadata": inventory["parent_metadata"],
                "deleted_exact_paths": [item["path"] for item in TARGETS],
                "target_outcomes": [_target_outcome(parent_fd, item) for item in states],
                "targets_absent": True, "protected_after": protected_after,
                "authorized_inventory_free_before": inventory["free_before"],
                "free_before": execution_free_before, "free_after": free_after,
                "minimum_post_cleanup_free_bytes": MIN_POST_CLEANUP_FREE_BYTES,
                "same_filesystem": True, "parent_fsynced_after_each_target": True,
                "training_authorized": False, "evaluation_authorized": False,
                "r4_staging_authorized": False, "cache_cleanup_performed": False,
                "checkpoint_pruning_performed": False,
            }
        except BaseException as exc:
            if journal is None:
                raise
            failure = _best_effort_failure_state(parent_fd, inventory, states, exc)
            try:
                journal.append("PARTIAL_FAILURE", **failure)
                journal_seal = journal.seal()
                journal = None
            except BaseException as journal_error:
                journal_seal = {"error": f"{type(journal_error).__name__}: {journal_error}",
                                "intent_was_fsynced": True}
            return {
                "schema": SCHEMA_RECEIPT, "status": "PARTIAL_FAILURE",
                "cleanup_performed": any(item["deleted"] for item in states),
                "host": HOST, "hostname": HOSTNAME, "package_head": request["package_head"],
                "request_sha256": request_sha, "inventory_sha256": inventory_sha,
                "authorization_sha256": authorization_sha, "health": health,
                "journal": journal_seal, "parent_metadata": inventory["parent_metadata"], **failure,
                "training_authorized": False, "evaluation_authorized": False,
                "r4_staging_authorized": False, "cache_cleanup_performed": False,
                "checkpoint_pruning_performed": False,
            }
        finally:
            if journal is not None:
                journal.close()
            os.close(parent_fd)


def _raise_sigterm(_signum: int, _frame: Any) -> None:
    raise ContractError("SIGTERM")


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
            digest = _write_exclusive(request["inventory_output"], value,
                                      expected_parent_identity=value["parent_metadata"])
        else:
            inventory, inventory_bytes = read_json_immediate_parent(args.inventory, require_mode=0o600)
            authorization, authorization_bytes = read_json_immediate_parent(args.authorization, require_mode=0o600)
            previous_term = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _raise_sigterm)
            try:
                value = execute(request, request_sha, inventory, sha_bytes(inventory_bytes), authorization,
                                sha_bytes(authorization_bytes))
            finally:
                signal.signal(signal.SIGTERM, previous_term)
            digest = _write_exclusive(request["receipt_output"], value,
                                      expected_parent_identity=value["parent_metadata"])
        output = {"status": value["status"], "sha256": digest}
        stream = sys.stdout if value["status"] in {"PASS", "VERIFIED_READ_ONLY"} else sys.stderr
        print(json.dumps(output, sort_keys=True), file=stream)
        return 0 if value["status"] in {"PASS", "VERIFIED_READ_ONLY"} else 2
    except (ContractError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
