#!/usr/bin/env python3
"""Read pinned e1e2 graph counters and reconstruct ownership; stdlib only.

No torch import, compiler execution, host access, training, or file writes.
Graph-to-shape assignment is explicitly inferred from source call order. It
must not be presented as an observed IR signature or a compiler wall interval.
"""
import argparse
import ast
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import types


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "workflow/candidates/confirmed-clock36-116188/train.py"
SOURCE_SHA = "e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43"
COLD = ROOT / "research/v2/operator/20260926-trn22-eval-boundary-0557/completed/results-bundle.json"
COLD_SHA = "20e1e5f920dff44a4d0a047790e02397ed916945e975a6822d6385efe2f9151d"
WARM_SHA = "9289bf8a94f164dccedbc1898408e7b8a02a4e3ed7cb927a44ad501b203e22d9"


def check(value, message):
    if not value:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ownership():
    source = SOURCE.read_bytes()
    check(sha(source) == SOURCE_SHA, "source hash changed")
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_plan")
    namespace = {"torch": types.SimpleNamespace(float32="fp32")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<pinned make_plan metadata only>", "exec"), namespace)

    class ParameterShape:
        dtype = "fp32"

        def __init__(self, shape):
            self.shape = tuple(shape)

        def numel(self):
            return math.prod(self.shape)

        def is_contiguous(self):
            return True

    # Exact e1e2 caller: five attention blocks, eight MLPs, twenty scalar
    # parameters, five local convolutions and only the last-layer VE gate.
    definitions = [
        ("adamw", (8192, 1024), 1),
        ("adamw", (8192, 1024), 1),
        ("adamw", (1,), 20),
        ("adamw", (8192, 1024), 1),
        ("adamw", (1024, 1, 2), 5),
        ("muon", (8, 32), 1),
        ("muon", (1024, 1024), 20),
        ("muon", (1024, 4096), 8),
        ("muon", (4096, 1024), 8),
    ]
    groups = [{"kind": kind, "params": [ParameterShape(shape) for _ in range(count)]}
              for kind, shape, count in definitions]
    entries, parameters, summary = namespace["make_plan"](groups, 4)
    check(len(parameters) == 65, "parameter count")
    check(sum(p.numel() for p in parameters) == 113256724, "parameter elements")
    stacks = []
    for rank in range(4):
        ordered = []
        for group_index, group in enumerate(groups):
            if group["kind"] != "muon":
                continue
            count = sum(e["owner"] == rank and e["group"] == group_index for e in entries)
            if count:
                ordered.append({"matrix_shape": list(group["params"][0].shape),
                                "real_count": count, "stack_count": max(2, count)})
        stacks.append(ordered)
    return stacks, summary


def analyze(path, expected_hash, stacks, cache_status):
    raw = path.read_bytes()
    check(sha(raw) == expected_hash, "bundle hash changed: " + str(path))
    bundle = json.loads(raw)
    files = bundle["files"]
    check(files["package/train.py"]["sha256"] == SOURCE_SHA, "run source identity")
    rows = []
    csv_hashes = {}
    for rank in range(4):
        name = f"metrics-training/metric_table_graph_stats_rank{rank}.csv"
        entry = files[name]
        encoded = entry["text"].encode()
        check(sha(encoded) == entry["sha256"] and len(encoded) == entry["bytes"], name)
        csv_hashes[name] = entry["sha256"]
        parsed = list(csv.DictReader(io.StringIO(entry["text"])))
        inference = [r for r in parsed if "_inference_" in r["graph_name"]]
        check(len(inference) == len(stacks[rank]), "ordered inference count differs")
        for order, (row, shape) in enumerate(zip(inference, stacks[rank])):
            check(int(row["graph_node_count"]) == 194, "unexpected inference node count")
            rows.append({**row, **shape, "rank": rank, "inference_order": order + 1})
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["graph_id"]].append(row)
    graphs = []
    totals_us = collections.Counter()
    for graph_id, occurrences in sorted(grouped.items()):
        shape_signatures = {(tuple(r["matrix_shape"]), r["stack_count"]) for r in occurrences}
        check(len(shape_signatures) == 1, "same graph assigned conflicting inferred shapes")
        shape, count = shape_signatures.pop()
        nonzero = [r for r in occurrences if int(r["torch_neuronx_compile_us"]) > 0]
        check(len(nonzero) == 1, "one reported nonzero compile counter per unique graph required")
        owner = nonzero[0]
        duration_us = int(owner["torch_neuronx_compile_us"])
        totals_us["x".join(map(str, shape))] += duration_us
        graphs.append({
            "graph_id": graph_id,
            "inferred_matrix_shape": list(shape),
            "inferred_stack_count": count,
            "occurrence_ranks": [r["rank"] for r in occurrences],
            "counter_rank": owner["rank"],
            "counter_graph_name": owner["graph_name"],
            "reported_compile_seconds": duration_us / 1e6,
            "graph_record_timestamp": owner["timestamp"],
            "counter_raw_key": f"files/metrics-training/metric_table_graph_stats_rank{owner['rank']}.csv",
        })
    check(len(graphs) == 8 and len(rows) == 13, "eight unique graphs and thirteen occurrences")
    rehearsal = json.loads(files["train/rehearsal.json"]["text"])
    return {
        "bundle": str(path), "bundle_sha256": sha(raw), "remote_root": bundle["root"],
        "source_sha256": SOURCE_SHA, "cache_status": cache_status,
        "csv_sha256": csv_hashes, "unique_graph_count": len(graphs),
        "rank_occurrences": len(rows), "graphs": graphs,
        "reported_counter_seconds_by_inferred_shape": {k: v / 1e6 for k, v in sorted(totals_us.items())},
        "reported_compile_counter_total_seconds": sum(totals_us.values()) / 1e6,
        "actual_muon_elapsed_compile_seconds": None,
        "compiler_cpu_work_seconds": None,
        "observed_full_startup_seconds": rehearsal["startup_allowance_observed"],
        "attribution": "source order + exact shape-only ownership + identical graph IDs across ranks; individual IR absent",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-bundle", type=Path)
    args = parser.parse_args()
    stacks, summary = ownership()
    result = {"status": "COUNTERS_MEASURED_SHAPE_ATTRIBUTION_INFERRED",
              "source_sha256": SOURCE_SHA, "owner_local_stacks_in_call_order": stacks,
              "ownership_summary": summary,
              "cold": analyze(COLD, COLD_SHA, stacks, "fresh empty SDK caches recorded"),
              "limits": ["counter sum is not concurrent wall elapsed or CPU work",
                         "zero rank counters do not establish cache hits",
                         "shape attribution lacks direct per-graph IR",
                         "no Torch numerical test, native run or source patch executed"]}
    if args.warm_bundle:
        result["reused_cache_control"] = analyze(args.warm_bundle, WARM_SHA, stacks, "historical research cache reused; cache hits unknown")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
