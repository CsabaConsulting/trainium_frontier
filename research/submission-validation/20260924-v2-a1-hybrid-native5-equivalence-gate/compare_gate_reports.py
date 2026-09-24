#!/usr/bin/env python3
"""Fail-closed exact comparator for the three native A1 five-step gate arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

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


class InvalidGate(ValueError):
    pass


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


def _load_rank(directory: Path, arm: str, rank: int) -> list[dict[str, Any]]:
    path = directory / f"rank-{rank}.jsonl"
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise InvalidGate(f"{arm} rank {rank}: report is not one regular unlinked file")
    records = []
    with path.open("rb") as handle:
        for line in handle:
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


def _without_owner_exchange(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sequence = _by(records, "collectives")[0]["sequence"]
    result = []
    for item in sequence:
        if item["op"] == "all_gather_into_tensor":
            continue
        normalized = dict(item)
        normalized["index"] = len(result)
        result.append(normalized)
    return result


def compare(g0_dir: Path, g1_dir: Path, g2_dir: Path) -> dict[str, Any]:
    loaded = {arm: [_load_rank(directory, arm, rank) for rank in range(4)]
              for arm, directory in (("G0", g0_dir), ("G1", g1_dir), ("G2", g2_dir))}
    instrumentation_failures: list[str] = []
    rms_failures: list[str] = []
    integrated_failures: list[str] = []

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
        _exact(_without_owner_exchange(g1), _without_owner_exchange(g2),
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


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise InvalidGate("output parent is not a real directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode() + b"\n"
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--g2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = compare(args.g0, args.g1, args.g2)
    except Exception as exc:
        payload = {"schema": "a1-native-five-comparison-v1", "status": "INVALID",
                   "error": f"{type(exc).__name__}: {exc}"}
        _write_exclusive(args.output, payload)
        return 2
    _write_exclusive(args.output, payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
