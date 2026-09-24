#!/usr/bin/env python3
"""Fail-closed exact comparator for the three native A1 five-step gate arms."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OWNERSHIP_MANIFEST_PATH = Path(__file__).with_name("a1_ownership_manifest.json")
OWNERSHIP_MANIFEST_SHA256 = "5fd3b2f075191210b7e373c8321c68b2d8f880db8cc7a788d60a90db4a382007"
OWNERSHIP_MANIFEST_SCHEMA = "a1-native-five-ownership-manifest-v1"
SCHEMA = "a1-native-five-equivalence-v1"
TABLES = {"ngram_embeds.b5.weight", "ngram_embeds.t5.weight"}
ARMS = {
    "G0": (False, "hybrid"),
    "G1": (True, "hybrid"),
    "G2": (True, "ordinary"),
}
LABELS = {
    "G0": "fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1",
    "G1": "fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1",
    "G2": "fs-20260924-v2-a1-native5-g2-ordinary-observation-on-trn21-r1",
}
WORLD_SIZE = 4
EXPECTED_STEPS = 5
# Exact A1 ownership contract reconstructed from the pinned source and flags.  The
# two BF16 RMSProp tables are replicated and excluded; all 66 ordinary FP32
# AdamW/Muon parameters, including both projections and both n-gram gates, are owned.
OWNERSHIP_ELEMENTS_BY_OWNER = [27_405_059, 30_540_035, 30_408_963, 29_362_179]
OWNERSHIP_ESTIMATED_MS_BY_OWNER = [
    12.991316601116363, 12.99464796558983, 12.984647965589831, 12.988706969361804,
]
OWNERSHIP_ENTRY_COUNT = 66
OWNERSHIP_OWNED_ELEMENTS = 117_716_236
OWNERSHIP_SHARD_ELEMENTS = 30_540_160
OWNERSHIP_PADDED_ELEMENTS = 122_160_640
OWNERSHIP_PADDED_BYTES = 488_642_560
MAX_AUTHENTICATED_FILE_BYTES = 64 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024


class InvalidGate(ValueError):
    pass


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _directory_identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid, info.st_gid)


def _validate_directory(info: os.stat_result, label: str) -> None:
    # The checked-out evidence tree is intentionally group writable (0775).  Pin
    # the complete accepted permission set rather than silently weakening the
    # check to "not world writable".
    accepted_modes = {0o700, 0o750, 0o755, 0o770, 0o775}
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or info.st_gid != os.getgid()
            or stat.S_IMODE(info.st_mode) not in accepted_modes):
        raise InvalidGate(f"unsafe authenticated directory: {label}")


def _validate_regular_file(info: os.stat_result, label: str, max_bytes: int) -> None:
    accepted_modes = {0o600, 0o640, 0o644, 0o660, 0o664}
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or info.st_gid != os.getgid()
            or stat.S_IMODE(info.st_mode) not in accepted_modes
            or info.st_size < 1 or info.st_size > max_bytes):
        raise InvalidGate(f"unsafe authenticated file: {label}")


def _required_open_flags(*, directory: bool = False) -> int:
    required = ("O_CLOEXEC", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise InvalidGate("platform lacks required no-follow/cloexec file-open support")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if directory:
        if not hasattr(os, "O_DIRECTORY"):
            raise InvalidGate("platform lacks required directory-open support")
        flags |= os.O_DIRECTORY
    elif hasattr(os, "O_NONBLOCK"):
        # A path swapped to a FIFO immediately before open must fail in fstat,
        # never block the comparator waiting for an attacker-controlled writer.
        flags |= os.O_NONBLOCK
    return flags


def _path_parts(trusted_root: Path, path: Path) -> tuple[Path, tuple[str, ...]]:
    root = Path(os.path.abspath(os.fspath(trusted_root)))
    target = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise InvalidGate(f"authenticated path escapes trusted root: {path}") from exc
    parts = relative.parts
    if not parts or any(part in ("", ".", "..") or "/" in part for part in parts):
        raise InvalidGate(f"invalid authenticated path: {path}")
    return root, parts


def _open_directory_chain(trusted_root: Path, path: Path) -> tuple[list[int], str, Path]:
    root, parts = _path_parts(trusted_root, path)
    fds: list[int] = []
    try:
        root_fd = os.open(root, _required_open_flags(directory=True))
        fds.append(root_fd)
        root_info = os.fstat(root_fd)
        _validate_directory(root_info, os.fspath(root))
        if _directory_identity(os.lstat(root)) != _directory_identity(root_info):
            raise InvalidGate("trusted root changed during open")
        for part in parts[:-1]:
            child_fd = os.open(part, _required_open_flags(directory=True), dir_fd=fds[-1])
            fds.append(child_fd)
            child_info = os.fstat(child_fd)
            _validate_directory(child_info, part)
            named = os.stat(part, dir_fd=fds[-2], follow_symlinks=False)
            if _directory_identity(named) != _directory_identity(child_info):
                raise InvalidGate(f"authenticated directory name changed: {part}")
        return fds, parts[-1], root
    except Exception:
        for fd in reversed(fds):
            os.close(fd)
        raise


def _revalidate_directory_chain(fds: list[int], root: Path,
                                parts: tuple[str, ...]) -> None:
    if _directory_identity(os.lstat(root)) != _directory_identity(os.fstat(fds[0])):
        raise InvalidGate("trusted root identity changed")
    for index, part in enumerate(parts):
        parent_fd, child_fd = fds[index], fds[index + 1]
        named = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
        if (_directory_identity(named) != _directory_identity(os.fstat(child_fd))):
            raise InvalidGate(f"authenticated directory path changed: {part}")


def _read_exact_fd(fd: int, expected_size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        try:
            chunk = os.read(fd, min(READ_CHUNK_BYTES, remaining))
        except InterruptedError:
            continue
        if not chunk:
            raise InvalidGate("authenticated file truncated during read")
        if len(chunk) > remaining:
            raise InvalidGate("authenticated read exceeded frozen size")
        chunks.append(chunk)
        remaining -= len(chunk)
    while True:
        try:
            extra = os.read(fd, 1)
            break
        except InterruptedError:
            continue
    if extra:
        raise InvalidGate("authenticated file appended during read")
    return b"".join(chunks)


def _sha256_bytes(data: bytes, path: Path | None = None) -> str:
    del path  # retained for deterministic path-specific race-injection regressions
    return hashlib.sha256(data).hexdigest()


def _capture_file(trusted_root: Path, path: Path, *,
                  max_bytes: int = MAX_AUTHENTICATED_FILE_BYTES,
                  expected_sha256: str | None = None,
                  parser: Any = None) -> dict[str, Any]:
    root, parts = _path_parts(trusted_root, path)
    fds, name, opened_root = _open_directory_chain(root, path)
    file_fd = -1
    try:
        file_fd = os.open(name, _required_open_flags(), dir_fd=fds[-1])
        before = os.fstat(file_fd)
        _validate_regular_file(before, os.fspath(path), max_bytes)
        named_before = os.stat(name, dir_fd=fds[-1], follow_symlinks=False)
        if _identity(named_before) != _identity(before):
            raise InvalidGate("authenticated file name changed during open")
        data = _read_exact_fd(file_fd, before.st_size)
        digest = _sha256_bytes(data, path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise InvalidGate(f"authenticated file digest mismatch: {path}")
        # Parsing occurs while the authenticated descriptor and every parent
        # descriptor remain open.  The final fstat/name/parent checks below are
        # deliberately after both hashing and parsing.
        parsed = parser(data) if parser is not None else None
        after = os.fstat(file_fd)
        if _identity(after) != _identity(before):
            raise InvalidGate("authenticated file changed during read")
        named_after = os.stat(name, dir_fd=fds[-1], follow_symlinks=False)
        if _identity(named_after) != _identity(before):
            raise InvalidGate("authenticated file path changed during read")
        _revalidate_directory_chain(fds, opened_root, parts[:-1])
        return {
            "bytes": data,
            "sha256": digest,
            "identity": _identity(before),
            "parsed": parsed,
        }
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        for fd in reversed(fds):
            os.close(fd)


def _file_sha(path: Path, trusted_root: Path = ROOT) -> str:
    return _capture_file(trusted_root, path)["sha256"]


def _load_ownership_manifest(*, validate_sources: bool = True) -> dict[str, Any]:
    captured = _capture_file(
        ROOT, OWNERSHIP_MANIFEST_PATH, max_bytes=1024 * 1024,
        expected_sha256=OWNERSHIP_MANIFEST_SHA256, parser=json.loads)
    manifest = captured["parsed"]
    if not isinstance(manifest, dict) or set(manifest) != {
            "schema", "source_pins", "configuration", "groups", "owned_entries",
            "replicated_rmsprop_parameters", "summary", "owner_exchange"}:
        raise InvalidGate("A1 ownership manifest schema mismatch")
    if manifest["schema"] != OWNERSHIP_MANIFEST_SCHEMA:
        raise InvalidGate("A1 ownership manifest version mismatch")
    summary = manifest["summary"]
    if summary != {
            "elements_by_owner": OWNERSHIP_ELEMENTS_BY_OWNER,
            "estimated_ms_by_owner": OWNERSHIP_ESTIMATED_MS_BY_OWNER,
            "owned_elements": OWNERSHIP_OWNED_ELEMENTS,
            "owned_entry_count": OWNERSHIP_ENTRY_COUNT,
            "shard_elements": OWNERSHIP_SHARD_ELEMENTS,
            "total_padded_bytes": OWNERSHIP_PADDED_BYTES,
            "total_padded_elements": OWNERSHIP_PADDED_ELEMENTS}:
        raise InvalidGate("A1 ownership manifest frozen summary mismatch")
    if (manifest["configuration"].get("world_size") != WORLD_SIZE
            or manifest["configuration"].get("owned_dtype") != "torch.float32"
            or len(manifest["owned_entries"]) != OWNERSHIP_ENTRY_COUNT
            or any(entry.get("dtype") != "torch.float32"
                   for entry in manifest["owned_entries"])):
        raise InvalidGate("A1 ownership manifest configuration mismatch")
    pins = manifest["source_pins"]
    if (pins.get("accepted_gate_implementation") !=
            "7528ca8dd9d9915097929217c2a2b0d9cfc17107"
            or pins.get("analysis_commit") !=
            "2ee8c5c0283c760683aff3de48f243845a803d0c"
            or pins.get("evidence_commit") !=
            "37aa3644e63e20a1427cfa69b44fd6de7e25cd81"
            or pins.get("failed_verifier_commit") !=
            "01d3cc04b3d55ed4d453fc1aff0e0eddad581682"):
        raise InvalidGate("A1 ownership manifest provenance mismatch")
    if validate_sources:
        files = pins.get("files")
        if not isinstance(files, dict) or not files:
            raise InvalidGate("A1 ownership manifest source pins missing")
        for relative, expected in files.items():
            if type(relative) is not str or not _sha(expected):
                raise InvalidGate("A1 ownership manifest source pin malformed")
            _capture_file(ROOT, ROOT / relative, expected_sha256=expected)
    return manifest


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return True


def _sha(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _tensor(record: Any) -> bool:
    return (isinstance(record, dict) and set(record) ==
            {"sha256", "shape", "dtype", "numel", "finite", "norm"}
            and _sha(record["sha256"]) and isinstance(record["shape"], list)
            and type(record["dtype"]) is str and type(record["numel"]) is int
            and record["numel"] > 0 and record["finite"] is True
            and type(record["norm"]) is float and math.isfinite(record["norm"]))


def _optimizer_snapshot(items: Any) -> bool:
    if type(items) is not list:
        return False
    names = []
    allowed = {
        "adamw": {"step", "exp_avg", "exp_avg_sq"},
        "rmsprop": {"step", "exp_avg_sq", "bias2"},
        "muon": {"momentum_buffer", "second_momentum_buffer"},
    }
    for item in items:
        if not isinstance(item, dict) or set(item) != {"name", "kind", "fields"}:
            return False
        name, kind, fields = item["name"], item["kind"], item["fields"]
        if type(name) is not str or kind not in allowed or not isinstance(fields, dict):
            return False
        if not set(fields).issubset(allowed[kind]):
            return False
        if "step" in fields and (type(fields["step"]) is not int or fields["step"] < 1):
            return False
        if "bias2" in fields and (type(fields["bias2"]) is not float
                                  or not math.isfinite(fields["bias2"])):
            return False
        if any(not _tensor(value) for key, value in fields.items()
               if key not in ("step", "bias2")):
            return False
        names.append(name)
    return len(names) == len(set(names))


def _parse_rank_bytes(data: bytes, arm: str, rank: int) -> list[dict[str, Any]]:
    records = []
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise InvalidGate(f"{arm} rank {rank}: unterminated JSONL record")
        record = json.loads(line)
        if not isinstance(record, dict) or type(record.get("kind")) is not str:
            raise InvalidGate(f"{arm} rank {rank}: malformed record")
        if not _finite_tree(record):
            raise InvalidGate(f"{arm} rank {rank}: nonfinite report value")
        records.append(record)
    expected = {"header": 1, "inventory": 1, "loss": 5, "update": 10,
                "collectives": 1, "complete": 1}
    if ARMS[arm][0]:
        expected.update(gradients=15, clip=5)
    counts = {kind: sum(record["kind"] == kind for record in records)
              for kind in set(record["kind"] for record in records)}
    if counts != expected:
        raise InvalidGate(f"{arm} rank {rank}: exact record cardinality mismatch {counts}")
    header = records[0]
    if header.get("kind") != "header" or header.get("schema") != SCHEMA:
        raise InvalidGate(f"{arm} rank {rank}: header/schema mismatch")
    detailed, mode = ARMS[arm]
    if (header.get("arm"), header.get("detailed"), header.get("optimizer_mode"),
            header.get("rank"), header.get("world_size"), header.get("host"),
            header.get("label")) != (arm, detailed, mode, rank, 4, "trn21", LABELS[arm]):
        raise InvalidGate(f"{arm} rank {rank}: identity mismatch")
    if header.get("config") != {
            "clip_after_reduce": True, "matrix_lr": 0.015,
            "max_train_seconds": 1800, "n_embd": 1024, "num_steps": 5,
            "pack_factor": 32, "seed": 42, "seq_len": 1024}:
        raise InvalidGate(f"{arm} rank {rank}: config mismatch")
    for kind, count in (("loss", 5), ("update", 10)):
        selected = [record for record in records if record["kind"] == kind]
        steps = ([record["step"] for record in selected] if kind == "loss"
                 else [(record["step"], record["phase"]) for record in selected])
        expected_steps = (list(range(count)) if kind == "loss"
                          else [(step, phase) for step in range(5) for phase in ("pre", "post")])
        if steps != expected_steps:
            raise InvalidGate(f"{arm} rank {rank}: {kind} order mismatch")
    if any(type(record.get("value")) is not float or not _sha(record.get("scalar_sha256"))
           for record in _by(records, "loss")):
        raise InvalidGate(f"{arm} rank {rank}: malformed loss record")
    updates = _by(records, "update")
    for record in updates:
        if set(record.get("tables", {})) != TABLES or not all(
                _tensor(item) for item in record["tables"].values()):
            raise InvalidGate(f"{arm} rank {rank}: malformed table update")
        model_state = record.get("model_state", {})
        if set(model_state) != {"sha256", "tensor_count"} or not _sha(
                model_state["sha256"]) or type(model_state["tensor_count"]) is not int:
            raise InvalidGate(f"{arm} rank {rank}: malformed model digest")
        if not _optimizer_snapshot(record.get("optimizer_state")) or type(
                record.get("rms_call_count")) is not int:
            raise InvalidGate(f"{arm} rank {rank}: malformed optimizer snapshot")
    if detailed:
        gradients = _by(records, "gradients")
        expected_gradient_order = [(step, stage) for step in range(5) for stage in
                                   ("local_pre_avg", "post_avg_preclip", "postclip")]
        if [(record.get("step"), record.get("stage")) for record in gradients] != expected_gradient_order:
            raise InvalidGate(f"{arm} rank {rank}: gradient stage/order mismatch")
        if any(set(record.get("tables", {})) != TABLES or not all(
                _tensor(item) for item in record["tables"].values()) for record in gradients):
            raise InvalidGate(f"{arm} rank {rank}: malformed gradient record")
        clips = _by(records, "clip")
        if [record.get("step") for record in clips] != list(range(5)) or any(
                not _sha(record.get("scalar_bytes_sha256")) for record in clips):
            raise InvalidGate(f"{arm} rank {rank}: malformed clip records")
    collective = _by(records, "collectives")[0]
    sequence = collective.get("sequence")
    if type(sequence) is not list or collective.get("count") != len(sequence):
        raise InvalidGate(f"{arm} rank {rank}: malformed collective sequence")
    sequence_bytes = json.dumps(sequence, sort_keys=True, separators=(",", ":")).encode()
    if collective.get("sequence_sha256") != hashlib.sha256(sequence_bytes).hexdigest():
        raise InvalidGate(f"{arm} rank {rank}: collective sequence digest mismatch")
    if [item.get("index") for item in sequence] != list(range(len(sequence))):
        raise InvalidGate(f"{arm} rank {rank}: collective index mismatch")
    complete = _by(records, "complete")[0]
    if complete.get("expected_steps") != 5 or not _sha(
            complete.get("final_model_state", {}).get("sha256")):
        raise InvalidGate(f"{arm} rank {rank}: malformed completion")
    return records


def _by(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [record for record in records if record["kind"] == kind]


def _strip(record: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in keys}


def _table_state(update: dict[str, Any]) -> dict[str, Any]:
    return {record["name"]: record for record in update["optimizer_state"]
            if record["name"] in TABLES}


def _exact(left: Any, right: Any, message: str, failures: list[str]) -> None:
    if left != right:
        failures.append(message)


def _owned_union(ranks: list[list[dict[str, Any]]], update_index: int) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for records in ranks:
        for item in _by(records, "update")[update_index]["optimizer_state"]:
            prior = by_name.get(item["name"])
            if prior is not None and prior != item:
                raise InvalidGate(f"hybrid rank disagreement for optimizer state {item['name']}")
            by_name[item["name"]] = item
    return sorted(by_name.values(), key=lambda item: item["name"])


def _ownership_shard_elements(records: list[dict[str, Any]], arm: str, rank: int,
                              manifest: dict[str, Any]) -> int:
    inventory = _by(records, "inventory")[0]
    if set(inventory) != {"kind", "groups", "ownership", "model_state"}:
        raise InvalidGate(f"{arm} rank {rank}: ownership inventory schema mismatch")
    if inventory["groups"] != manifest["groups"]:
        raise InvalidGate(f"{arm} rank {rank}: exact named A1 optimizer groups mismatch")
    if inventory["model_state"] != manifest["configuration"]["model_state"]:
        raise InvalidGate(f"{arm} rank {rank}: initial model digest mismatch")
    ownership = inventory["ownership"]
    if not isinstance(ownership, dict) or set(ownership) != {
            "entries", "summary", "replicated_rmsprop_groups"}:
        raise InvalidGate(f"{arm} rank {rank}: ownership record schema mismatch")
    entries, summary = ownership["entries"], ownership["summary"]
    if not isinstance(entries, list) or not isinstance(summary, dict) or set(summary) != {
            "elements_by_owner", "estimated_ms_by_owner", "shard_elements",
            "total_padded_bytes", "replicated_rmsprop_groups"}:
        raise InvalidGate(f"{arm} rank {rank}: ownership summary schema mismatch")
    stride = summary["shard_elements"]
    counts, costs = summary["elements_by_owner"], summary["estimated_ms_by_owner"]
    if (type(stride) is not int or stride != OWNERSHIP_SHARD_ELEMENTS
            or counts != OWNERSHIP_ELEMENTS_BY_OWNER
            or costs != OWNERSHIP_ESTIMATED_MS_BY_OWNER
            or stride != ((max(counts) + 127) // 128) * 128
            or summary["total_padded_bytes"] != OWNERSHIP_PADDED_BYTES
            or stride * WORLD_SIZE != OWNERSHIP_PADDED_ELEMENTS):
        raise InvalidGate(f"{arm} rank {rank}: frozen ownership summary mismatch")
    entry_keys = {"index", "group", "position", "kind", "shape", "elements",
                  "estimated_ms", "owner", "offset"}
    calculated_counts = [0] * WORLD_SIZE
    intervals = [[] for _ in range(WORLD_SIZE)]
    named_entries = []
    groups = inventory["groups"]
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != entry_keys:
            raise InvalidGate(f"{arm} rank {rank}: ownership entry schema mismatch")
        shape = entry["shape"]
        if (entry["index"] != index or type(entry["group"]) is not int
                or entry["group"] < 0 or type(entry["position"]) is not int
                or entry["position"] < 0 or entry["kind"] not in ("adamw", "muon")
                or type(shape) is not list or not shape
                or any(type(value) is not int or value < 1 for value in shape)
                or type(entry["elements"]) is not int
                or entry["elements"] != math.prod(shape)
                or type(entry["estimated_ms"]) is not float
                or not math.isfinite(entry["estimated_ms"]) or entry["estimated_ms"] < 0
                or type(entry["owner"]) is not int or entry["owner"] not in range(WORLD_SIZE)
                or type(entry["offset"]) is not int or entry["offset"] < 0
                or entry["offset"] + entry["elements"] > stride):
            raise InvalidGate(f"{arm} rank {rank}: malformed ownership entry")
        if entry["group"] >= len(groups) or groups[entry["group"]]["index"] != entry["group"]:
            raise InvalidGate(f"{arm} rank {rank}: ownership group mapping mismatch")
        names = groups[entry["group"]]["parameter_names"]
        if entry["position"] >= len(names):
            raise InvalidGate(f"{arm} rank {rank}: ownership parameter mapping mismatch")
        named_entries.append({**entry, "name": names[entry["position"]],
                              "dtype": "torch.float32"})
        calculated_counts[entry["owner"]] += entry["elements"]
        intervals[entry["owner"]].append((entry["offset"], entry["offset"] + entry["elements"]))
    if named_entries != manifest["owned_entries"]:
        raise InvalidGate(f"{arm} rank {rank}: exact named A1 ownership manifest mismatch")
    actual_summary = {
        "elements_by_owner": counts,
        "estimated_ms_by_owner": costs,
        "owned_elements": sum(entry["elements"] for entry in entries),
        "owned_entry_count": len(entries),
        "shard_elements": stride,
        "total_padded_bytes": summary["total_padded_bytes"],
        "total_padded_elements": stride * WORLD_SIZE,
    }
    if actual_summary != manifest["summary"] or calculated_counts != counts or any(
            any(left[1] > right[0] for left, right in zip(sorted(owner_intervals),
                                                           sorted(owner_intervals)[1:]))
            for owner_intervals in intervals):
        raise InvalidGate(f"{arm} rank {rank}: ownership coverage mismatch")
    replicated = summary["replicated_rmsprop_groups"]
    parameter_keys = {"position", "shape", "elements", "dtype", "contiguous",
                      "requires_grad", "device_type", "device_index"}
    if (not isinstance(replicated, list) or len(replicated) != 1
            or not isinstance(replicated[0], dict)
            or set(replicated[0]) != {"group", "parameters"}
            or type(replicated[0]["group"]) is not int
            or not isinstance(replicated[0]["parameters"], list)
            or len(replicated[0]["parameters"]) != len(TABLES)
            or ownership["replicated_rmsprop_groups"] != [replicated[0]["group"]]):
        raise InvalidGate(f"{arm} rank {rank}: replicated RMS ownership mismatch")
    normalized_replicated = []
    replicated_group = replicated[0]["group"]
    replicated_names = groups[replicated_group]["parameter_names"]
    for position, parameter in enumerate(replicated[0]["parameters"]):
        if (not isinstance(parameter, dict) or set(parameter) != parameter_keys
                or parameter["position"] != position
                or parameter["shape"] != [524_288, 128]
                or parameter["elements"] != 67_108_864
                or parameter["dtype"] != "torch.bfloat16"
                or parameter["contiguous"] is not True
                or parameter["requires_grad"] is not True
                or parameter["device_type"] != "neuron"
                or parameter["device_index"] != rank):
            raise InvalidGate(f"{arm} rank {rank}: replicated RMS parameter schema mismatch")
        normalized_replicated.append({
            **parameter, "group": replicated_group,
            "name": replicated_names[position], "device_index": "rank",
        })
    if normalized_replicated != manifest["replicated_rmsprop_parameters"]:
        raise InvalidGate(f"{arm} rank {rank}: exact replicated RMS manifest mismatch")
    return stride


def _load_rank(directory: Path, arm: str, rank: int, *,
               expected_sha256: str | None = None) -> list[dict[str, Any]]:
    path = directory / f"rank-{rank}.jsonl"
    sealed = _capture_file(
        directory, path, expected_sha256=expected_sha256,
        parser=lambda data: _parse_rank_bytes(data, arm, rank))
    records = sealed.get("parsed")
    if not isinstance(records, list):
        raise InvalidGate(f"{arm} rank {rank}: missing sealed report records")
    return records


def _validate_exact_groups(records: list[dict[str, Any]], manifest: dict[str, Any],
                           arm: str, rank: int) -> None:
    inventory = _by(records, "inventory")[0]
    if (inventory.get("groups") != manifest["groups"]
            or inventory.get("model_state") != manifest["configuration"]["model_state"]):
        raise InvalidGate(f"{arm} rank {rank}: exact A1 inventory/source identity mismatch")


def _validate_owner_exchange_records(records: list[dict[str, Any]], arm: str, rank: int,
                                     manifest: dict[str, Any]) -> None:
    contract = manifest["owner_exchange"]
    gathers = [item for item in _by(records, "collectives")[0]["sequence"]
               if item.get("op") == "all_gather_into_tensor"]
    expected = {
        "op": "all_gather_into_tensor",
        "output_shape": contract["output_shape"],
        "input_shape": contract["input_shape"],
        "output_dtype": contract["output_dtype"],
        "input_dtype": contract["input_dtype"],
        "output_numel": contract["output_numel"],
        "input_numel": contract["input_numel"],
    }
    if len(gathers) != contract["count_per_rank"] or any(
            {key: value for key, value in item.items() if key != "index"} != expected
            for item in gathers):
        raise InvalidGate(f"{arm} rank {rank}: exact A1 owner exchange mismatch")


def validate_immutable_g0(directory: Path) -> dict[str, Any]:
    """Authenticate the archived G0 reports against source and the exact A1 manifest."""
    manifest = _load_ownership_manifest()
    report_pins = manifest["source_pins"].get("immutable_g0_report_sha256")
    if not isinstance(report_pins, dict) or set(report_pins) != {
            f"rank-{rank}.jsonl" for rank in range(WORLD_SIZE)}:
        raise InvalidGate("immutable G0 report pin set mismatch")
    for rank in range(WORLD_SIZE):
        name = f"rank-{rank}.jsonl"
        records = _load_rank(
            directory, "G0", rank, expected_sha256=report_pins[name])
        _ownership_shard_elements(records, "G0", rank, manifest)
        _validate_owner_exchange_records(records, "G0", rank, manifest)
    return {
        "schema": "a1-native-five-g0-ownership-validation-v1",
        "status": "PASS",
        "manifest_sha256": OWNERSHIP_MANIFEST_SHA256,
        "owned_entry_count": OWNERSHIP_ENTRY_COUNT,
        "elements_by_owner": OWNERSHIP_ELEMENTS_BY_OWNER,
        "shard_elements": OWNERSHIP_SHARD_ELEMENTS,
        "total_padded_elements": OWNERSHIP_PADDED_ELEMENTS,
        "owner_exchanges_per_rank": EXPECTED_STEPS,
    }


def _validated_without_owner_exchange(records: list[dict[str, Any]], ordinary: list[dict[str, Any]],
                                      arm: str, rank: int,
                                      manifest: dict[str, Any]) -> list[dict[str, Any]]:
    stride = _ownership_shard_elements(records, arm, rank, manifest)
    sequence = _by(records, "collectives")[0]["sequence"]
    if any(item.get("op") == "all_gather_into_tensor" for item in ordinary):
        raise InvalidGate("ordinary collective sequence contains an owner exchange")
    if len(ordinary) % EXPECTED_STEPS:
        raise InvalidGate("ordinary collective sequence cannot be partitioned into five updates")
    shared_per_step = len(ordinary) // EXPECTED_STEPS
    if shared_per_step < 2:
        raise InvalidGate("ordinary collective sequence is missing per-step work")
    for step in range(EXPECTED_STEPS):
        control = ordinary[(step + 1) * shared_per_step - 1]
        if (set(control) != {"index", "op", "reduce_op", "shape", "dtype", "numel"}
                or control["op"] != "all_reduce" or not control["reduce_op"].endswith("MAX")
                or control["shape"] != [2] or control["dtype"] != "torch.int32"
                or control["numel"] != 2):
            raise InvalidGate("ordinary collective sequence lacks exact terminal step control")
    result = []
    gather_count = 0
    for item in sequence:
        if item.get("op") == "all_gather_into_tensor":
            expected = {
                "index": item.get("index"), "op": "all_gather_into_tensor",
                "output_shape": [WORLD_SIZE * stride], "input_shape": [stride],
                "output_dtype": "torch.float32", "input_dtype": "torch.float32",
                "output_numel": WORLD_SIZE * stride, "input_numel": stride,
            }
            if item != expected or gather_count >= EXPECTED_STEPS:
                raise InvalidGate(f"{arm} rank {rank}: malformed owner exchange")
            # The owner exchange occurs after gradient normalization/update and immediately
            # before the shared terminal loss/stop control all-reduce for this step.
            if len(result) != (gather_count + 1) * shared_per_step - 1:
                raise InvalidGate(f"{arm} rank {rank}: misplaced owner exchange")
            gather_count += 1
            continue
        normalized = dict(item)
        normalized["index"] = len(result)
        result.append(normalized)
    if gather_count != EXPECTED_STEPS or result != ordinary:
        raise InvalidGate(f"{arm} rank {rank}: owner exchange sequence mismatch")
    return result


def compare(g0_dir: Path, g1_dir: Path, g2_dir: Path) -> dict[str, Any]:
    manifest = _load_ownership_manifest()
    loaded = {arm: [_load_rank(directory, arm, rank) for rank in range(4)]
              for arm, directory in (("G0", g0_dir), ("G1", g1_dir), ("G2", g2_dir))}
    instrumentation_failures: list[str] = []
    rms_failures: list[str] = []
    integrated_failures: list[str] = []

    for arm, ranks in loaded.items():
        for rank, records in enumerate(ranks):
            _validate_exact_groups(records, manifest, arm, rank)

    ordinary_sequences = [_by(records, "collectives")[0]["sequence"]
                          for records in loaded["G2"]]
    for rank, sequence in enumerate(ordinary_sequences):
        if any(item.get("op") == "all_gather_into_tensor" for item in sequence):
            raise InvalidGate(f"G2 rank {rank}: unexpected owner exchange")
    for arm in ("G0", "G1"):
        for rank, records in enumerate(loaded[arm]):
            _validated_without_owner_exchange(
                records, ordinary_sequences[rank], arm, rank, manifest)

    # Observation OFF versus ON must not change any shared record byte semantics.
    for rank in range(4):
        g0, g1 = loaded["G0"][rank], loaded["G1"][rank]
        _exact(_strip(_by(g0, "header")[0], "arm", "detailed", "label"),
               _strip(_by(g1, "header")[0], "arm", "detailed", "label"),
               f"rank {rank}: observation changed header contract", instrumentation_failures)
        for kind in ("inventory", "loss", "update", "collectives", "complete"):
            _exact(_by(g0, kind), _by(g1, kind),
                   f"rank {rank}: observation changed {kind}", instrumentation_failures)

    # The ownership inventory must exclude the two replicated RMS tables and perform
    # exactly one parameter all-gather per update.  Gradient AVG collectives are shared
    # with ordinary mode; no extra table/RMS collective is permitted.
    for rank, records in enumerate(loaded["G1"]):
        inventory = _by(records, "inventory")[0]
        groups = inventory["groups"]
        rms_names = {name for group in groups if group["kind"] == "rmsprop"
                     for name in group["parameter_names"]}
        if rms_names != TABLES:
            rms_failures.append(f"rank {rank}: exact RMS table inventory mismatch")
        ownership = inventory["ownership"]
        if ownership is None or any(entry.get("kind") == "rmsprop"
                                    for entry in ownership.get("entries", [])):
            rms_failures.append(f"rank {rank}: RMS table entered ownership")
        calls = [record["rms_call_count"] for record in _by(records, "update")
                 if record["phase"] == "post"]
        if calls != [1, 2, 3, 4, 5]:
            rms_failures.append(f"rank {rank}: RMS call cardinality mismatch")
        collectives = _by(records, "collectives")[0]["sequence"]
        if sum(item["op"] == "all_gather_into_tensor" for item in collectives) != 5:
            integrated_failures.append(f"rank {rank}: owner exchange count mismatch")

    # Detailed table path is exact between hybrid and ordinary for the same rank.
    for rank in range(4):
        g1, g2 = loaded["G1"][rank], loaded["G2"][rank]
        _exact(_validated_without_owner_exchange(
                   g1, _by(g2, "collectives")[0]["sequence"], "G1", rank, manifest),
               _by(g2, "collectives")[0]["sequence"],
               f"rank {rank}: non-owner collective sequence differs", rms_failures)
        for kind in ("gradients", "clip"):
            _exact(_by(g1, kind), _by(g2, kind),
                   f"rank {rank}: table {kind} differs hybrid/ordinary", rms_failures)
        for index, (h_update, o_update) in enumerate(zip(_by(g1, "update"),
                                                         _by(g2, "update"))):
            _exact(h_update["tables"], o_update["tables"],
                   f"rank {rank} update {index}: table params differ", rms_failures)
            _exact(_table_state(h_update), _table_state(o_update),
                   f"rank {rank} update {index}: table optimizer state differs", rms_failures)
        _exact(_by(g1, "loss"), _by(g2, "loss"),
               f"rank {rank}: losses differ hybrid/ordinary", integrated_failures)
        _exact(_by(g1, "complete")[0]["final_model_state"],
               _by(g2, "complete")[0]["final_model_state"],
               f"rank {rank}: final model differs hybrid/ordinary", integrated_failures)
        for index, (h_update, o_update) in enumerate(zip(_by(g1, "update"),
                                                         _by(g2, "update"))):
            _exact(h_update["model_state"], o_update["model_state"],
                   f"rank {rank} update {index}: model differs", integrated_failures)

    # Once AVG has occurred, table gradients, clip bytes, params and state must also
    # agree across every hybrid rank.  Local pre-AVG gradients are rank-specific by design.
    reference = loaded["G1"][0]
    ref_grads = [record for record in _by(reference, "gradients")
                 if record["stage"] != "local_pre_avg"]
    for rank in range(1, 4):
        records = loaded["G1"][rank]
        _exact(ref_grads, [record for record in _by(records, "gradients")
                           if record["stage"] != "local_pre_avg"],
               f"rank {rank}: post-AVG/clip table gradients disagree", rms_failures)
        _exact(_by(reference, "clip"), _by(records, "clip"),
               f"rank {rank}: joint clip scalar bytes disagree", rms_failures)
        for index, (ref_update, update) in enumerate(zip(_by(reference, "update"),
                                                         _by(records, "update"))):
            _exact(ref_update["tables"], update["tables"],
                   f"rank {rank} update {index}: table params disagree", rms_failures)
            _exact(_table_state(ref_update), _table_state(update),
                   f"rank {rank} update {index}: table state disagrees", rms_failures)
            _exact(ref_update["model_state"], update["model_state"],
                   f"rank {rank} update {index}: persistent model state disagrees",
                   integrated_failures)
        _exact(_by(reference, "complete")[0]["final_model_state"],
               _by(records, "complete")[0]["final_model_state"],
               f"rank {rank}: final persistent model state disagrees", integrated_failures)

    # Owner shards are a representation detail.  Their exact named union must equal the
    # full ordinary state after every pre/post snapshot.
    ordinary_reference = loaded["G2"][0]
    for update_index in range(10):
        owner_union = _owned_union(loaded["G1"], update_index)
        ordinary = _by(ordinary_reference, "update")[update_index]["optimizer_state"]
        _exact(owner_union, ordinary,
               f"update {update_index}: owned optimizer state union differs", integrated_failures)
        for rank in range(1, 4):
            _exact(_by(loaded["G2"][rank], "update")[update_index]["optimizer_state"], ordinary,
                   f"ordinary rank {rank} update {update_index}: optimizer state differs",
                   integrated_failures)

    if instrumentation_failures:
        raise InvalidGate("; ".join(instrumentation_failures))
    rms_exact = not rms_failures
    integrated_exact = rms_exact and not integrated_failures
    status = ("PASS" if integrated_exact else
              "RMS_PASS_INTEGRATED_BLOCKED" if rms_exact else "RMS_FAIL")
    return {
        "schema": "a1-native-five-comparison-v1",
        "status": status,
        "rms_repair_verdict": "RMS_PASS" if rms_exact else "RMS_FAIL",
        "integrated_ownership_verdict": "INTEGRATED_PASS" if integrated_exact else "INTEGRATED_BLOCKED",
        "rms_failures": rms_failures,
        "integrated_failures": integrated_failures,
        "timing_is_equality_gate": False,
    }


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        try:
            count = os.write(fd, view[written:])
        except InterruptedError:
            continue
        if type(count) is not int or count <= 0 or count > len(view) - written:
            raise InvalidGate("short/invalid authenticated result write")
        written += count


def _file_inode(info: os.stat_result) -> tuple[int, int]:
    return (info.st_dev, info.st_ino)


def _absolute_output(path: Path) -> Path:
    raw = Path(path)
    if any(part == ".." for part in raw.parts):
        raise InvalidGate("output path contains a lexical parent escape")
    absolute = Path(os.path.abspath(os.fspath(raw)))
    if absolute.name in ("", ".", "..") or absolute.parent == absolute:
        raise InvalidGate("output path has no safe final component")
    return absolute


def _open_output_directory_chain(path: Path) -> tuple[list[int], list[str], list[tuple[int, ...]]]:
    """Open every lexical ancestor from / through the output parent without following links."""
    absolute = _absolute_output(path)
    names = list(absolute.parent.parts[1:])
    fds: list[int] = []
    identities: list[tuple[int, ...]] = []
    try:
        root_fd = os.open("/", _required_open_flags(directory=True))
        fds.append(root_fd)
        root_info = os.fstat(root_fd)
        if not stat.S_ISDIR(root_info.st_mode):
            raise InvalidGate("filesystem root is not a directory")
        identities.append(_directory_identity(root_info))
        if _directory_identity(os.lstat("/")) != identities[0]:
            raise InvalidGate("filesystem root identity changed")
        for component in names:
            child_fd = os.open(component, _required_open_flags(directory=True), dir_fd=fds[-1])
            fds.append(child_fd)
            child_info = os.fstat(child_fd)
            if not stat.S_ISDIR(child_info.st_mode):
                raise InvalidGate(f"output ancestor is not a directory: {component}")
            named = os.stat(component, dir_fd=fds[-2], follow_symlinks=False)
            identity = _directory_identity(child_info)
            if _directory_identity(named) != identity:
                raise InvalidGate(f"output ancestor changed during open: {component}")
            identities.append(identity)
        _validate_directory(os.fstat(fds[-1]), os.fspath(absolute.parent))
        return fds, names, identities
    except Exception:
        for fd in reversed(fds):
            os.close(fd)
        raise


def _revalidate_output_directory_chain(fds: list[int], names: list[str],
                                       identities: list[tuple[int, ...]]) -> None:
    if len(fds) != len(identities) or len(fds) != len(names) + 1:
        raise InvalidGate("output directory-chain cardinality changed")
    if (_directory_identity(os.fstat(fds[0])) != identities[0]
            or _directory_identity(os.lstat("/")) != identities[0]):
        raise InvalidGate("filesystem root binding changed")
    for index, component in enumerate(names, start=1):
        child = os.fstat(fds[index])
        named = os.stat(component, dir_fd=fds[index - 1], follow_symlinks=False)
        if (_directory_identity(child) != identities[index]
                or _directory_identity(named) != identities[index]):
            raise InvalidGate(f"output lexical ancestor binding changed: {component}")


def _assert_final_absent(parent_fd: int, final_name: str) -> None:
    try:
        os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError(errno.EEXIST, "exclusive output already exists", final_name)


def _private_temp_name() -> str:
    return f".a1-comparator-{os.getpid()}-{secrets.token_hex(16)}.tmp"


def _validate_result_info(info: os.stat_result, expected_inode: tuple[int, int],
                          expected_size: int, label: str) -> None:
    if (not stat.S_ISREG(info.st_mode) or _file_inode(info) != expected_inode
            or info.st_nlink != 1 or info.st_uid != os.getuid()
            or info.st_gid != os.getgid() or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size != expected_size):
        raise InvalidGate(f"unsafe {label} result identity")


def _read_validate_result(parent_fd: int, name: str, expected_inode: tuple[int, int],
                          expected_data: bytes, expected_digest: str,
                          payload: dict[str, Any], label: str) -> dict[str, Any]:
    fd = os.open(name, _required_open_flags(), dir_fd=parent_fd)
    try:
        before = os.fstat(fd)
        _validate_result_info(before, expected_inode, len(expected_data), label)
        named_before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(named_before) != _identity(before):
            raise InvalidGate(f"{label} result name/inode changed before read")
        data = _read_exact_fd(fd, before.st_size)
        digest = _sha256_bytes(data)
        parsed = json.loads(data)
        after = os.fstat(fd)
        named_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (_identity(after) != _identity(before)
                or _identity(named_after) != _identity(before)):
            raise InvalidGate(f"{label} result changed during readback")
        if data != expected_data or digest != expected_digest or parsed != payload:
            raise InvalidGate(f"{label} result content mismatch")
        return {"sha256": digest, "identity": _identity(before)}
    finally:
        os.close(fd)


def _rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise InvalidGate("platform lacks renameat2 no-replace publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(parent_fd, os.fsencode(source), parent_fd,
                       os.fsencode(destination), 1)  # RENAME_NOREPLACE
    if result != 0:
        value = ctypes.get_errno()
        raise OSError(value, os.strerror(value), destination)


def _unlink_dedicated_name(parent_fd: int, name: str) -> None:
    """Remove one dedicated output entry without following it or any target."""
    try:
        os.unlink(name, dir_fd=parent_fd)
    except FileNotFoundError:
        return
    except IsADirectoryError:
        # A race-created directory is still an entry in the dedicated output
        # namespace.  rmdir never follows a symlink and fails closed if nonempty.
        os.rmdir(name, dir_fd=parent_fd)


def _cleanup_result_entries(parent_fd: int, names_to_remove: tuple[str, str],
                            chain_fds: list[int], chain_names: list[str],
                            chain_identities: list[tuple[int, ...]]) -> None:
    # Once final-name absence has been established, every later entry at either
    # dedicated name is race-created.  Remove the lexical names unconditionally;
    # unlink/rmdir never follows symlink or hardlink targets.
    for _ in range(8):
        for name in names_to_remove:
            _unlink_dedicated_name(parent_fd, name)
        os.fsync(parent_fd)
        _revalidate_output_directory_chain(chain_fds, chain_names, chain_identities)
        remaining = []
        for name in names_to_remove:
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            remaining.append(name)
        if not remaining:
            # The final anchored absence check is intentionally after the last
            # directory-chain callback so an injection at that boundary loops.
            _revalidate_output_directory_chain(chain_fds, chain_names, chain_identities)
            remaining_after_chain = []
            for name in names_to_remove:
                try:
                    os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                remaining_after_chain.append(name)
            if not remaining_after_chain:
                return
    raise InvalidGate("could not clear dedicated comparator output names")


def _write_exclusive(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    absolute = _absolute_output(path)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode() + b"\n"
    digest = _sha256_bytes(data)
    fds, names, identities = _open_output_directory_chain(absolute)
    parent_fd = fds[-1]
    temp_name = _private_temp_name()
    final_name = absolute.name
    write_fd = -1
    created_inode: tuple[int, int] | None = None
    dedicated_names_active = False
    try:
        _revalidate_output_directory_chain(fds, names, identities)
        _assert_final_absent(parent_fd, final_name)
        # From this exact point onward the final name is a dedicated empty
        # publication slot.  Any later entry at final/temp is race-created and
        # is removed unconditionally on failure through this pinned parent fd.
        dedicated_names_active = True
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        write_fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
        initial = os.fstat(write_fd)
        created_inode = _file_inode(initial)
        _validate_result_info(initial, created_inode, 0, "new temporary")
        named_initial = os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(named_initial) != _identity(initial):
            raise InvalidGate("temporary result name/inode changed after create")
        _write_all(write_fd, data)
        os.fsync(write_fd)
        written = os.fstat(write_fd)
        _validate_result_info(written, created_inode, len(data), "written temporary")
        os.close(write_fd)
        write_fd = -1

        _read_validate_result(parent_fd, temp_name, created_inode, data, digest,
                              payload, "temporary")
        _revalidate_output_directory_chain(fds, names, identities)
        _assert_final_absent(parent_fd, final_name)
        # The last temp seal is adjacent to the atomic no-replace publication.
        _read_validate_result(parent_fd, temp_name, created_inode, data, digest,
                              payload, "pre-publication temporary")
        _rename_noreplace(parent_fd, temp_name, final_name)
        os.fsync(parent_fd)
        _revalidate_output_directory_chain(fds, names, identities)

        _read_validate_result(parent_fd, final_name, created_inode, data, digest,
                              payload, "published")
        _revalidate_output_directory_chain(fds, names, identities)
        # Close/reopen the final file for the last content seal.  No writable file
        # descriptor has survived publication.
        _read_validate_result(
            parent_fd, final_name, created_inode, data, digest, payload, "pre-final")
        # Absolute last pre-success filesystem boundary: first anchor every
        # lexical directory, then freshly open and seal the final entry.  No
        # callback or filesystem operation follows except closing that read-only
        # fd inside _read_validate_result and returning its digest/identity.
        _revalidate_output_directory_chain(fds, names, identities)
        return _read_validate_result(
            parent_fd, final_name, created_inode, data, digest, payload, "final")
    except Exception as original:
        if write_fd >= 0:
            os.close(write_fd)
            write_fd = -1
        if dedicated_names_active:
            try:
                _cleanup_result_entries(
                    parent_fd, (temp_name, final_name), fds, names, identities)
            except Exception as cleanup:
                raise InvalidGate(f"result publication and cleanup failed: {original}; {cleanup}") from cleanup
        raise
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        for fd in reversed(fds):
            os.close(fd)


def _verify_publication_receipt(payload: dict[str, Any], receipt: dict[str, Any]) -> None:
    """Bind caller success to the exact final seal returned by _write_exclusive."""
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode() + b"\n"
    if not isinstance(receipt, dict) or set(receipt) != {"sha256", "identity"}:
        raise InvalidGate("publication receipt schema mismatch")
    identity = receipt["identity"]
    if (receipt["sha256"] != _sha256_bytes(data)
            or not isinstance(identity, tuple) or len(identity) != 9
            or not stat.S_ISREG(identity[2]) or identity[3] != 1
            or identity[4] != os.getuid() or identity[5] != os.getgid()
            or stat.S_IMODE(identity[2]) != 0o600 or identity[6] != len(data)):
        raise InvalidGate("publication receipt does not bind the requested result")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-g0-only", type=Path)
    parser.add_argument("--g0", type=Path)
    parser.add_argument("--g1", type=Path)
    parser.add_argument("--g2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full = all((args.g0, args.g1, args.g2))
    partial = any((args.g0, args.g1, args.g2)) and not full
    if partial or bool(args.validate_g0_only) == full:
        parser.error("choose exactly --validate-g0-only or the complete --g0/--g1/--g2 set")
    try:
        payload = (validate_immutable_g0(args.validate_g0_only)
                   if args.validate_g0_only else compare(args.g0, args.g1, args.g2))
    except Exception as exc:
        payload = {"schema": "a1-native-five-comparison-v1", "status": "INVALID",
                   "error": f"{type(exc).__name__}: {exc}"}
        receipt = _write_exclusive(args.output, payload)
        _verify_publication_receipt(payload, receipt)
        return 2
    receipt = _write_exclusive(args.output, payload)
    _verify_publication_receipt(payload, receipt)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
