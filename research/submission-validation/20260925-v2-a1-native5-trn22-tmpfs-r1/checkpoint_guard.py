#!/usr/bin/env python3
"""Same-fd model-only checkpoint verifier for retained tmpfs gate artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


MAX_CHECKPOINT_BYTES = 1024**3
TMPFS_PARENT = Path("/dev/shm/trainium_native5_20260925_trn22")
ARMS = {"g0-r1", "g1-r1", "g2-r1"}
OBSERVER_SHA256 = "8f0fcd21282c54f2747b0260a392363a15825adf3c1f2a1c3357fbaf8790bacc"
SOURCE_SHA256 = {
    "g0-r1": "908431d7e1da8a9110e6e6005de5ecbe3dba13d798bf2f37f2443404dd8d7d94",
    "g1-r1": "908431d7e1da8a9110e6e6005de5ecbe3dba13d798bf2f37f2443404dd8d7d94",
    "g2-r1": "6ed69d2356c28e0dd8f88f494d988c0b87f7f86426a001945bfd958ed6f8f6d0",
}
VALIDATION_KEYS = {"status", "weights_only", "model_only", "finite", "strict_load",
                   "logits_exact", "causality_exact", "model_state_sha256",
                   "eval_logits_sha256", "causal_digest_sha256"}
BINDING_KEYS = {"source_sha256", "observer_sha256", "rank_report_sha256",
                "job_sha256", "postflight_sha256", "parent_identity_sha256",
                "mount_root_identity_sha256"}


class CheckpointError(ValueError):
    pass


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _entry(info: os.stat_result) -> dict[str, Any]:
    return {"kind": "directory", "device": info.st_dev, "inode": info.st_ino,
            "mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid, "gid": info.st_gid,
            "nlink": info.st_nlink, "size": info.st_size,
            "allocated": info.st_blocks * 512, "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns, "content_sha256": None}


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()).hexdigest()


def _dir_flags() -> int:
    required = ("O_CLOEXEC", "O_NOFOLLOW", "O_DIRECTORY")
    if not all(hasattr(os, item) for item in required):
        raise CheckpointError("platform lacks anchored directory flags")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY


def _file_flags() -> int:
    if not hasattr(os, "O_CLOEXEC") or not hasattr(os, "O_NOFOLLOW"):
        raise CheckpointError("platform lacks no-follow file flags")
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
            if len(payload) > 64 * 1024:
                raise CheckpointError("fdinfo unexpectedly large")
    finally:
        os.close(info_fd)
    values = [line.split(b":", 1)[1].strip() for line in bytes(payload).splitlines()
              if line.startswith(b"mnt_id:")]
    if len(values) != 1 or not values[0].isdigit():
        raise CheckpointError("fd mount id unavailable")
    return int(values[0])


def _revalidate_chain(fds: list[int], names: list[str], identities: list[tuple[int, ...]]) -> None:
    if _identity(os.fstat(fds[0])) != identities[0]:
        raise CheckpointError("checkpoint root identity changed")
    for index, name in enumerate(names):
        opened = os.fstat(fds[index + 1])
        named = os.stat(name, dir_fd=fds[index], follow_symlinks=False)
        if _identity(opened) != identities[index + 1] or _identity(named) != identities[index + 1]:
            raise CheckpointError("checkpoint ancestor changed")


def _hash_fd(fd: int, size: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    remaining = size
    digest = hashlib.sha256()
    while remaining:
        try:
            chunk = os.read(fd, min(1024 * 1024, remaining))
        except InterruptedError:
            continue
        if not chunk:
            raise CheckpointError("checkpoint truncated during same-fd read")
        remaining -= len(chunk)
        digest.update(chunk)
    if os.read(fd, 1):
        raise CheckpointError("checkpoint appended during same-fd read")
    return digest.hexdigest()


def _safe_open(path: Path, expected_device: int,
               expected_mount_id: int, expected_parent_identity_sha256: str,
               expected_mount_root_identity_sha256: str
               ) -> tuple[list[int], list[str], list[tuple[int, ...]], int, str, os.stat_result]:
    absolute = Path(os.path.abspath(path))
    if (absolute.name != "final.pt" or absolute.parent.name != "out"
            or absolute.parent.parent.name not in ARMS
            or absolute.parent.parent.parent != Path(os.path.abspath(TMPFS_PARENT))):
        raise CheckpointError("checkpoint is outside exact tmpfs arm layout")
    fds: list[int] = []
    names: list[str] = []
    identities: list[tuple[int, ...]] = []
    try:
        root_fd = os.open("/", _dir_flags())
        fds.append(root_fd)
        identities.append(_identity(os.fstat(root_fd)))
        parent_fd = root_fd
        cursor = Path("/")
        tmpfs_root = Path(os.path.abspath(TMPFS_PARENT))
        within_tmpfs = False
        for component in absolute.parent.parts[1:]:
            if component in ("", ".", "..") or "/" in component:
                raise CheckpointError("unsafe checkpoint path component")
            child = os.open(component, _dir_flags(), dir_fd=parent_fd)
            opened = os.fstat(child)
            named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if _identity(opened) != _identity(named):
                os.close(child)
                raise CheckpointError("checkpoint ancestor rebound during open")
            cursor /= component
            within_tmpfs = within_tmpfs or cursor == tmpfs_root
            if within_tmpfs and opened.st_dev != expected_device:
                os.close(child)
                raise CheckpointError("checkpoint ancestor mount/device mismatch")
            if within_tmpfs and _fd_mount_id(child) != expected_mount_id:
                os.close(child)
                raise CheckpointError("checkpoint ancestor mount-id mismatch")
            if (cursor == tmpfs_root
                    and _canonical_sha(_entry(opened))
                    != expected_mount_root_identity_sha256):
                os.close(child)
                raise CheckpointError("checkpoint tmpfs root identity mismatch")
            fds.append(child)
            names.append(component)
            identities.append(_identity(opened))
            parent_fd = child
        before_parent = os.fstat(parent_fd)
        if _canonical_sha(_entry(before_parent)) != expected_parent_identity_sha256:
            raise CheckpointError("checkpoint parent identity mismatch")
        fd = os.open(absolute.name, _file_flags(), dir_fd=parent_fd)
        info = os.fstat(fd)
        named = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(named) != _identity(info):
            os.close(fd)
            raise CheckpointError("checkpoint name changed during open")
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or info.st_gid != os.getgid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_dev != expected_device or _fd_mount_id(fd) != expected_mount_id
                or not 0 < info.st_size <= MAX_CHECKPOINT_BYTES):
            os.close(fd)
            raise CheckpointError("checkpoint file identity/size/device mismatch")
        _revalidate_chain(fds, names, identities)
        return fds, names, identities, fd, absolute.name, info
    except Exception:
        for open_fd in reversed(fds):
            os.close(open_fd)
        raise


def verify_checkpoint(path: Path, expected_device: int, expected_mount_id: int,
                      bindings: dict[str, Any], validator) -> dict[str, Any]:
    """Hash and validate one stable descriptor; validator receives a duplicate file object."""
    if not isinstance(bindings, dict) or set(bindings) != BINDING_KEYS:
        raise CheckpointError("checkpoint binding schema mismatch")
    for key in BINDING_KEYS - {"rank_report_sha256"}:
        if not _sha(bindings[key]):
            raise CheckpointError(f"checkpoint binding {key} malformed")
    if (not isinstance(bindings["rank_report_sha256"], list)
            or len(bindings["rank_report_sha256"]) != 4
            or not all(_sha(value) for value in bindings["rank_report_sha256"])):
        raise CheckpointError("checkpoint rank report binding malformed")
    arm_key = Path(os.path.abspath(path)).parent.parent.name
    if (bindings["source_sha256"] != SOURCE_SHA256.get(arm_key)
            or bindings["observer_sha256"] != OBSERVER_SHA256):
        raise CheckpointError("checkpoint frozen source/observer binding mismatch")
    fds, names, identities, fd, name, before = _safe_open(
        path, expected_device, expected_mount_id, bindings["parent_identity_sha256"],
        bindings["mount_root_identity_sha256"])
    parent_fd = fds[-1]
    try:
        first_hash = _hash_fd(fd, before.st_size)
        _revalidate_chain(fds, names, identities)
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb", closefd=True) as duplicate:
            validation = validator(duplicate, f"/proc/self/fd/{fd}")
        if not isinstance(validation, dict) or set(validation) != VALIDATION_KEYS:
            raise CheckpointError("scientific validator schema mismatch")
        if (validation["status"] != "PASS"
                or any(validation[key] is not True for key in
                       ("weights_only", "model_only", "finite", "strict_load",
                        "logits_exact", "causality_exact"))
                or not all(_sha(validation[key]) for key in
                           ("model_state_sha256", "eval_logits_sha256",
                            "causal_digest_sha256"))):
            raise CheckpointError("scientific validator did not return exact PASS")
        after_validation = os.fstat(fd)
        if _identity(after_validation) != _identity(before):
            raise CheckpointError("checkpoint changed during scientific validation")
        if _identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != _identity(before):
            raise CheckpointError("checkpoint path changed during scientific validation")
        _revalidate_chain(fds, names, identities)
        second_hash = _hash_fd(fd, before.st_size)
        if second_hash != first_hash:
            raise CheckpointError("checkpoint bytes changed during validation")
        final = os.fstat(fd)
        if _identity(final) != _identity(before):
            raise CheckpointError("checkpoint metadata changed after final hash")
        if _identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != _identity(before):
            raise CheckpointError("checkpoint name changed after final hash")
        _revalidate_chain(fds, names, identities)
        return {
            "schema": "a1-native-five-trn22-checkpoint-v2",
            "path": os.path.abspath(os.fspath(path)), "mount_id": expected_mount_id,
            "mount_device": expected_device, "sha256": first_hash, "bytes": before.st_size,
            "allocated": before.st_blocks * 512, "device": before.st_dev,
            "inode": before.st_ino, "kind": "file", "nlink": before.st_nlink,
            "mode": stat.S_IMODE(before.st_mode), "uid": before.st_uid,
            "gid": before.st_gid, "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns, "same_fd_stable": True,
            "source_sha256": bindings["source_sha256"],
            "observer_sha256": bindings["observer_sha256"],
            "rank_report_sha256": bindings["rank_report_sha256"],
            "job_sha256": bindings["job_sha256"],
            "postflight_sha256": bindings["postflight_sha256"],
            "parent_identity_sha256": bindings["parent_identity_sha256"],
            "mount_root_identity_sha256": bindings["mount_root_identity_sha256"],
            "complete": True, **validation,
        }
    finally:
        os.close(fd)
        for open_fd in reversed(fds):
            os.close(open_fd)


def _tensor_bytes(tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    if not bool(cpu.isfinite().all().item()):
        raise CheckpointError("nonfinite checkpoint/model tensor")
    return cpu.view(__import__("torch").uint8).numpy().tobytes()


def _state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _read_regular_source(source: Path, expected_sha256: str) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    fd = os.open(source, flags)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.getuid() or before.st_gid != os.getgid()
                or not 0 < before.st_size <= 8 * 1024 * 1024):
            raise CheckpointError("unsafe gate source identity")
        data = bytearray()
        while len(data) < before.st_size:
            try:
                chunk = os.read(fd, before.st_size - len(data))
            except InterruptedError:
                continue
            if not chunk:
                raise CheckpointError("gate source truncated during read")
            data.extend(chunk)
        if os.read(fd, 1):
            raise CheckpointError("gate source appended during read")
        after = os.fstat(fd)
        if _identity(after) != _identity(before):
            raise CheckpointError("gate source changed during read")
        payload = bytes(data)
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise CheckpointError("gate source hash mismatch")
        return payload
    finally:
        os.close(fd)


def scientific_validator(source: Path, expected_sha256: str):
    import torch

    source_bytes = _read_regular_source(source, expected_sha256)
    module_name = "a1_trn22_checkpoint_source"
    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=os.fspath(source))
    if spec is None:
        raise CheckpointError("cannot construct source module")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = os.fspath(source)
    sys.modules[module_name] = module
    exec(compile(source_bytes, os.fspath(source), "exec"), module.__dict__)

    def validate(file_object, proc_fd_path: str) -> dict[str, Any]:
        payload = torch.load(file_object, map_location="cpu", weights_only=True)
        if set(payload) != {"model_state", "model_config", "toggles", "meta"}:
            raise CheckpointError("checkpoint is not exact model-only schema")
        state = payload["model_state"]
        if not isinstance(state, dict) or not state:
            raise CheckpointError("empty checkpoint model state")
        state_sha = _state_digest(state)
        model = module.load_for_eval(proc_fd_path, torch.device("cpu"))
        model.eval()
        live_sha = _state_digest(model.state_dict())
        if live_sha != state_sha:
            raise CheckpointError("strict-loaded state differs from checkpoint")
        generator = torch.Generator(device="cpu").manual_seed(42)
        first = torch.randint(0, module.TOKENIZER_VOCAB_SIZE, (2, 1024),
                              generator=generator, dtype=torch.int32)
        second = first.clone()
        second[:, 513:] = torch.randint(0, module.TOKENIZER_VOCAB_SIZE,
                                        second[:, 513:].shape, generator=generator,
                                        dtype=torch.int32)
        with torch.no_grad():
            logits_a, logits_b = model(first), model(second)
        if not torch.equal(logits_a[:, :513], logits_b[:, :513]):
            raise CheckpointError("causal-prefix identity failed")
        return {
            "status": "PASS", "weights_only": True, "model_only": True,
            "finite": True, "strict_load": True, "logits_exact": True,
            "causality_exact": True, "model_state_sha256": state_sha,
            "eval_logits_sha256": hashlib.sha256(_tensor_bytes(logits_a)).hexdigest(),
            "causal_digest_sha256": hashlib.sha256(
                _tensor_bytes(logits_a[:, :513])).hexdigest(),
        }

    return validate
