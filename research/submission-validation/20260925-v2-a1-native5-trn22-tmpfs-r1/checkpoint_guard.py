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


class CheckpointError(ValueError):
    pass


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


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


def _safe_open(path: Path, expected_device: int) -> tuple[int, int, str, os.stat_result]:
    absolute = Path(os.path.abspath(path))
    if absolute.parent.parent.parent != TMPFS_PARENT:
        raise CheckpointError("checkpoint is outside exact tmpfs arm layout")
    components = [Path("/")]
    cursor = Path("/")
    for part in absolute.parent.parts[1:]:
        cursor /= part
        components.append(cursor)
    for component in components:
        info = os.lstat(component)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise CheckpointError(f"unsafe checkpoint ancestor: {component}")
    parent_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    parent_fd = os.open(absolute.parent, parent_flags)
    before_parent = os.fstat(parent_fd)
    if _identity(os.lstat(absolute.parent)) != _identity(before_parent):
        os.close(parent_fd)
        raise CheckpointError("checkpoint parent changed during open")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        fd = os.open(absolute.name, flags, dir_fd=parent_fd)
        info = os.fstat(fd)
        named = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(named) != _identity(info):
            raise CheckpointError("checkpoint name changed during open")
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or info.st_gid != os.getgid()
                or info.st_dev != expected_device or not 0 < info.st_size <= MAX_CHECKPOINT_BYTES):
            raise CheckpointError("checkpoint file identity/size/device mismatch")
        return parent_fd, fd, absolute.name, info
    except Exception:
        os.close(parent_fd)
        raise


def verify_checkpoint(path: Path, expected_device: int, validator) -> dict[str, Any]:
    """Hash and validate one stable descriptor; validator receives a duplicate file object."""
    parent_fd, fd, name, before = _safe_open(path, expected_device)
    try:
        first_hash = _hash_fd(fd, before.st_size)
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb", closefd=True) as duplicate:
            validation = validator(duplicate, f"/proc/self/fd/{fd}")
        after_validation = os.fstat(fd)
        if _identity(after_validation) != _identity(before):
            raise CheckpointError("checkpoint changed during scientific validation")
        if _identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != _identity(before):
            raise CheckpointError("checkpoint path changed during scientific validation")
        second_hash = _hash_fd(fd, before.st_size)
        if second_hash != first_hash:
            raise CheckpointError("checkpoint bytes changed during validation")
        final = os.fstat(fd)
        if _identity(final) != _identity(before):
            raise CheckpointError("checkpoint metadata changed after final hash")
        if _identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != _identity(before):
            raise CheckpointError("checkpoint name changed after final hash")
        if not isinstance(validation, dict) or validation.get("status") != "PASS":
            raise CheckpointError("scientific validator did not return PASS")
        return {
            "path": os.fspath(path), "sha256": first_hash, "bytes": before.st_size,
            "allocated": before.st_blocks * 512, "device": before.st_dev,
            "inode": before.st_ino, "nlink": before.st_nlink,
            "mode": stat.S_IMODE(before.st_mode), "uid": before.st_uid,
            "gid": before.st_gid, "mtime_ns": before.st_mtime_ns,
            "same_fd_stable": True, **validation,
        }
    finally:
        os.close(fd)
        os.close(parent_fd)


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
        }

    return validate
