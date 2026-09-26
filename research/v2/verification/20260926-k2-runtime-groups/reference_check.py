#!/usr/bin/env python3
"""Independent stdlib model of only the supplied runtime-group delta.

No candidate/source/harness imports. The unchanged inner arithmetic is represented
as ordered FP32 addition, not claimed to emulate every NKI instruction. Addresses
from old slices are compared with a generic contiguous reshape/select view. Receipt
validation is evidence authentication, not an independent rerun of those gates.
"""

import hashlib
import json
import math
from pathlib import Path
import platform
import random
import struct
import sys
from datetime import datetime, timezone


SOURCE = "f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e"
PARENT = "662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00"
LANE = Path("/home/poojith-devan/Desktop/Trainium/research/v2/lanes/k2-runtime-groups")
RECEIPTS = (
    "SOURCE.diff", "SPEC.md", "cpu-result.json",
    "compile-production-result-1790417895251858239.json",
    "simulate-result-1790417911971128655.json",
    "simulate-result-1790417933199676192.json", "artifacts-manifest.json",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class View:
    """Contiguous row-major layout, with dimension selection preserving strides."""

    def __init__(self, shape, strides=None, offset=0):
        self.shape = tuple(shape)
        self.strides = tuple(strides) if strides is not None else tuple(
            math.prod(self.shape[i + 1:]) for i in range(len(self.shape))
        )
        self.offset = offset

    def select(self, axis, index):
        require(0 <= index < self.shape[axis], "select out of bounds")
        return View(self.shape[:axis] + self.shape[axis + 1:],
                    self.strides[:axis] + self.strides[axis + 1:],
                    self.offset + index * self.strides[axis])

    def address(self, *indices):
        require(len(indices) == len(self.shape), "rank mismatch")
        require(all(0 <= i < n for i, n in zip(indices, self.shape)), "index out of bounds")
        return self.offset + sum(i * s for i, s in zip(indices, self.strides))


def runtime_groups(count):
    # A sequential runtime induction variable, independently expressed from range.
    index = 0
    while index < count:
        yield index
        index += 1


def production_addresses():
    groups, partition, channels, owners = 64, 128, 1024, 2
    features = channels // owners
    metadata = View((groups, partition, 1))
    maxima = View((groups, 1, 1))
    output = View((groups, partition, channels))
    visited = bytearray(groups * partition * channels)
    output_hash = hashlib.sha256()
    writes = 0
    for owner in range(owners):
        observed_groups = []
        for group in runtime_groups(groups):
            observed_groups.append(group)
            require(maxima.select(0, group).address(0, 0) == group, "maximum address")
            row_view = metadata.select(0, group)
            out_view = output.select(0, group)
            for row in range(partition):
                original_row = group * partition + row
                require(row_view.address(row, 0) == original_row, "counts/starts address")
                # Enumerate every feature, deriving the new row base from the view.
                new_base = out_view.address(row, owner * features)
                addresses = []
                for feature in range(features):
                    original = original_row * channels + owner * features + feature
                    selected = new_base + feature * out_view.strides[1]
                    require(original == selected, "output address differs")
                    require(visited[selected] == 0, "cross-owner or row overlap")
                    visited[selected] = 1
                    addresses.append(selected)
                    writes += 1
                output_hash.update(struct.pack("<" + "I" * features, *addresses))
        require(observed_groups == list(range(groups)), "group order differs")
    require(all(visited), "output hole")
    return {"geometry": {"G": groups, "P": partition, "C": channels, "owners": owners},
            "metadata_rows_checked_per_owner": groups * partition,
            "maxima_checked_per_owner": groups, "output_addresses_checked": writes,
            "every_element_written_once": True, "owner_overlap": False,
            "last_address": len(visited) - 1, "ordered_address_sha256": output_hash.hexdigest()}


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(values):
    return struct.pack("<" + "f" * len(values), *values)


def metadata_for(ids, vocab, partition):
    # Independently build stable row-local encounter lists from original tokens.
    encounters = [[i for i, row in enumerate(ids) if row == target] for target in range(vocab)]
    counts = [len(row) for row in encounters]
    starts, positions = [], []
    for row in encounters:
        starts.append(len(positions))
        positions.extend(row)
    maxima = [max(counts[first:first + partition]) for first in range(0, vocab, partition)]
    return counts, starts, positions, maxima, encounters


def execute_group_model(ids, gradient, geometry, dynamic):
    groups, partition = geometry["groups"], geometry["partition"]
    channels, owners = geometry["channels"], geometry["owners"]
    vocab, features = groups * partition, channels // owners
    counts, starts, positions, maxima, encounters = metadata_for(ids, vocab, partition)
    out = [None] * (vocab * channels)
    writes = [0] * len(out)
    trace = []
    metadata_view, maximum_view = View((groups, partition, 1)), View((groups, 1, 1))
    output_view = View((groups, partition, channels))
    for owner in range(owners):
        for group in runtime_groups(groups) if dynamic else range(groups):
            if dynamic:
                row_addresses = [metadata_view.select(0, group).address(p, 0) for p in range(partition)]
                maximum_address = maximum_view.select(0, group).address(0, 0)
            else:
                first = group * partition
                row_addresses = list(range(first, first + partition))
                maximum_address = group
            remaining = [counts[address] for address in row_addresses]
            pointer = [starts[address] for address in row_addresses]
            accumulator = [[0.0] * features for _ in range(partition)]
            for encounter in range(maxima[maximum_address]):
                for p in range(partition):
                    if remaining[p]:
                        token = positions[pointer[p]]
                        require(0 <= token < len(ids), "invalid active token")
                        trace.append((owner, group, p, encounter, pointer[p], token))
                        for f in range(features):
                            accumulator[p][f] = fp32(accumulator[p][f] + gradient[token][owner * features + f])
                        remaining[p] -= 1
                        pointer[p] += 1
            for p in range(partition):
                for f in range(features):
                    if dynamic:
                        address = output_view.select(0, group).address(p, owner * features + f)
                    else:
                        address = (group * partition + p) * channels + owner * features + f
                    out[address] = accumulator[p][f]
                    writes[address] += 1
    require(all(count == 1 for count in writes), "output ownership/write count")
    for owner in range(owners):
        for row in range(vocab):
            observed = [event[-1] for event in trace if event[0] == owner and event[1] * partition + event[2] == row]
            require(observed == encounters[row], "duplicate encounter order differs")
    return out, trace, counts, maxima


def check_case(case, geometry, multipliers):
    ids = case["ids"]
    require(len(ids) == len(case["values"]), "case dimensions")
    gradient = [[fp32(v * multiplier) for multiplier in multipliers] for v in case["values"]]
    # BF16-exact input values: low 16 bits in IEEE binary32 are all zero.
    require(all(struct.unpack("<I", bits([v]))[0] & 0xffff == 0 for row in gradient for v in row), "non-BF16 test input")
    old, old_trace, counts, maxima = execute_group_model(ids, gradient, geometry, False)
    new, new_trace, _, _ = execute_group_model(ids, gradient, geometry, True)
    require(bits(old) == bits(new) and old_trace == new_trace, "static/dynamic difference")
    channels = geometry["channels"]
    direct = [0.0] * len(old)
    # The oracle visits original token order and has no row-group loop or metadata.
    for token, row in enumerate(ids):
        for feature in range(channels):
            address = row * channels + feature
            direct[address] = fp32(direct[address] + gradient[token][feature])
    require(bits(new) == bits(direct), "direct encounter oracle difference")
    for row, count in enumerate(counts):
        if count == 0:
            require(bits(new[row * channels:(row + 1) * channels]) == bytes(4 * channels), "empty row is not positive zero")
    return {"name": case["name"], "tokens": len(ids), "counts": counts, "group_maxima": maxima,
            "output_rows": [new[i:i + channels] for i in range(0, len(new), channels)],
            "trace_sha256": digest(json.dumps(new_trace, separators=(",", ":")).encode()),
            "exact_static_dynamic_and_direct": True, "empty_rows_positive_zero": True}


def authenticate_evidence():
    data = {name: (LANE / name).read_bytes() for name in RECEIPTS}
    file_hashes = {name: {"bytes": len(value), "sha256": digest(value)} for name, value in data.items()}
    parsed = {name: json.loads(value) for name, value in data.items() if name.endswith(".json")}
    diff = data["SOURCE.diff"].decode()
    adds = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    removes = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    require(len(adds) == 7 and len(removes) == 8, "delta size changed")
    require(SOURCE in data["SPEC.md"].decode() and PARENT in data["SPEC.md"].decode(), "SPEC source pins")
    cpu = parsed["cpu-result.json"]
    require(cpu["hashes"][str(LANE / "train.py")] == SOURCE, "CPU source pin")
    require(cpu["hashes"][str(LANE.parent / "k2-compiler-local-constants/train.py")] == PARENT, "CPU parent pin")
    cpu_summary = {}
    for dtype, body in cpu["by_index_dtype"].items():
        require(all(case["actual_bits"] == case["expected_bits"] for case in body["operation"]), "CPU receipt bit mismatch")
        cpu_summary[dtype] = {"finite_cases": sum(c["verdict"] == "EXACT_FINITE" for c in body["operation"]),
                              "nonfinite_diagnostics_not_accepted_as_finite": sum(c["verdict"] == "FAIL_NONFINITE_DIAGNOSTIC" for c in body["operation"]),
                              "all_reported_bits_equal": True}
    compile_name = RECEIPTS[3]
    comp, manifest = parsed[compile_name], parsed["artifacts-manifest.json"]
    require(manifest["compile_receipt_sha256"] == digest(data[compile_name]), "compile receipt hash")
    require(comp["source_sha256"] == manifest["source_sha256"] == SOURCE, "compile/manifest source pins")
    require(comp["status"] == manifest["compile_status"] == "FULL_BACKWARD_COMPILE_PASS", "compile status")
    require(comp["stage"] == "NEFF_WRITTEN" and comp["bir_pass"] is True, "compile stage")
    require(comp["production_default_options"] is True and comp["source_unmodified"] is True, "compile options/source declaration")
    require(comp["native_execution"] is False and comp["forward_framework_compile"] is False, "compile scope")
    require(comp["target"] == "trn2" and comp["lnc"] == 2, "compile target")
    artifact_root = Path(comp["artifacts_dir"])
    require(artifact_root == LANE / "scratch/compile-1790417891961756099", "artifact directory differs")
    mapping = {"k2.neff": "k2.neff", "backend.log": "log-neuron-cc.txt",
               "core0-instruction-stats.txt": "nc00/sg00/instruction_stats.txt",
               "core1-instruction-stats.txt": "nc01/sg00/instruction_stats.txt",
               "kernel-info.json": "kernel_info.json", "mempressure.txt": "mempressure.txt"}
    artifacts = {}
    for name, relative in mapping.items():
        raw = (artifact_root / relative).read_bytes()
        actual = {"bytes": len(raw), "sha256": digest(raw)}
        require(actual == manifest["artifacts"][name], "artifact mismatch: " + name)
        artifacts[name] = {"path": str(artifact_root / relative), **actual}
    require(comp["neff_sha256"] == artifacts["k2.neff"]["sha256"] and comp["neff_bytes"] == artifacts["k2.neff"]["bytes"], "NEFF binding")
    sims = []
    for name in RECEIPTS[4:6]:
        sim = parsed[name]
        require(sim["source_sha256"] == SOURCE and sim["check_backend_sha256"] == comp["check_backend_sha256"], "sim source/harness binding")
        require(sim["status"] == "SIMULATOR_NUMERICS_PASS" and sim["exact_equal"] is True and sim["max_abs_error"] == 0, "sim numerical result")
        require(sim["actual_shape"] == [8192, 1024] and sim["actual_dtype"] == "float32", "sim output geometry")
        require(sim["native_execution"] is False, "sim scope")
        sims.append({"receipt": name, "fixture": sim["fixture"], "maximum_count": sim["maximum_count"], "untouched_rows": sim["untouched_rows"], "exact_equal": True})
    return {"source_pin": SOURCE, "parent_pin": PARENT, "delta": {"added": len(adds), "removed": len(removes)},
            "files": file_hashes, "cpu_receipt": cpu_summary, "compiler_artifacts": artifacts, "simulator_receipts": sims,
            "source_bytes_independently_rehashed": False,
            "source_binding_limit": "Whitelist excludes full candidate, parent and gate harness; pins are supplied receipt declarations."}


def main():
    start = datetime.now(timezone.utc).isoformat()
    own = Path(__file__).resolve()
    cases_path = own.with_name("reference_cases.json")
    cases_raw = cases_path.read_bytes()
    cases = json.loads(cases_raw)
    receipt_evidence = authenticate_evidence()
    address_result = production_addresses()
    result = [check_case(case, cases["geometry"], cases["channel_multipliers"]) for case in cases["cases"]]
    require(result[1]["output_rows"][-1][0] == 0 and result[2]["output_rows"][-1][0] == 1, "order discriminator ineffective")
    config = cases["random_cases"]
    rng = random.Random(config["seed"])
    random_hash = hashlib.sha256()
    vocab = cases["geometry"]["groups"] * cases["geometry"]["partition"]
    for i in range(config["count"]):
        length = rng.randrange(config["maximum_tokens"] + 1)
        case = {"name": "seed42_" + str(i), "ids": [rng.randrange(vocab) for _ in range(length)],
                "values": [rng.choice(config["value_choices"]) for _ in range(length)]}
        outcome = check_case(case, cases["geometry"], cases["channel_multipliers"])
        random_hash.update(json.dumps({"case": case, "result": outcome}, sort_keys=True).encode())
    print(json.dumps({"status": "PASS_CHANGED_SOURCE_SEMANTICS_ONLY", "start_utc": start,
                      "end_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                      "executable": sys.executable, "platform": platform.platform(), "cwd": str(Path.cwd()),
                      "argv": sys.argv, "script_sha256": digest(own.read_bytes()), "reference_cases_sha256": digest(cases_raw),
                      "production_addresses": address_result, "reference_cases": result,
                      "random_cases": {"count": config["count"], "seed": config["seed"], "all_passed": True, "evidence_sha256": random_hash.hexdigest()},
                      "supplied_evidence": receipt_evidence,
                      "native_execution": False, "framework_imports": False, "host_contact": False,
                      "performance_or_launch_acceptance": False}, indent=2))


if __name__ == "__main__":
    main()
