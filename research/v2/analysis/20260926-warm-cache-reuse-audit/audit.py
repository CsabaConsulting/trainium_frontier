#!/usr/bin/env python3
"""Bounded, read-only archive audit. Stdlib only; no host/compiler/Torch calls.

Prints authenticated selected inputs, graph IDs and counters; writes nothing.
"""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OP = "research/v2/operator/"
RUNS = {
    "e1e2_cold_trn22": ("20260926-trn22-eval-boundary-0557/completed/results-bundle.json", "20e1e5f920dff44a4d0a047790e02397ed916945e975a6822d6385efe2f9151d"),
    "e1e2_reused_trn21": ("20260926-final-sprint-successors/trn21-e1e2-control-completed/results-bundle.json", "9289bf8a94f164dccedbc1898408e7b8a02a4e3ed7cb927a44ad501b203e22d9"),
    "e1e2_reused_trn22": ("20260926-final-sprint-successors/trn22-e1e2-control-completed/results-bundle.json", "f17ea4ebbbef720cee5d02f810fcb103fc887a5d56fa7c906910f28346dae6c7"),
    "8c6_lr015_trn22": ("20260926-final-sprint-successors/trn22-p0-completed/results-bundle.json", "cc54af74150b6512cac5861c04b3908e5b21fecd2b4f85e8aa9b1e9e7b3d99a8"),
    "8c6_lr021_trn21": ("20260926-final-sprint-successors/trn21-batch-lr14-completed/results-bundle.json", "30826ff24d91bf6fac6747b6dac618ceee543993e87575ad0effce1617bdc962"),
    "8c6_lr021_trn22": ("20260926-final-sprint-successors/trn22-batch-lr14-completed/results-bundle.json", "c05eaa57272719cb7ba3e2d24dae67f11ba6ca6e2de170ba7d0fbffe163bd91f"),
}
ENV_KEYS = ["NEURON_COMPILE_CACHE_DIR", "NEURON_COMPILE_CACHE_URL",
            "TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
            "TORCH_NEURONX_HLO_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
            "TORCH_NEURONX_DEBUG_DIR", "NEURON_CC_FLAGS", "NEURON_LOGICAL_NC_CONFIG",
            "TORCH_NEURONX_NEFF_KEY_FROM_HLO", "TORCH_NEURONX_NEFF_DISABLE_CACHE",
            "TORCH_NEURONX_HLO_CACHE", "NKI_DUMP_PERSISTENT_CACHE"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_run(path, expected):
    raw = (ROOT / OP / path).read_bytes()
    check(digest(raw) == expected, path)
    bundle = json.loads(raw)
    files = bundle["files"]
    verified = {}

    def member(name):
        entry = files[name]
        data = entry["text"].encode()
        check(digest(data) == entry["sha256"] and len(data) == entry["bytes"], name)
        verified[name] = entry["sha256"]
        return entry["text"]

    source = member("package/train.py")
    command = member("package/launch-command.txt").strip()
    launch = json.loads(member("training-launch.json"))
    rehearsal = json.loads(member("train/rehearsal.json"))
    env = launch["environment"]
    rows = []
    for rank in range(4):
        name = f"metrics-training/metric_table_graph_stats_rank{rank}.csv"
        for row in csv.DictReader(io.StringIO(member(name))):
            row = dict(row, rank=rank, archive_member=name)
            row["compile_seconds"] = int(row["torch_neuronx_compile_us"]) / 1e6
            rows.append(row)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["graph_id"]].append(row)
    graphs = []
    totals = collections.Counter()
    for key, occurrences in sorted(groups.items()):
        kind = next(k for k in ["forward", "backward", "inference"] if "_" + k + "_" in occurrences[0]["graph_name"])
        microseconds = sum(int(r["torch_neuronx_compile_us"]) for r in occurrences)
        totals[kind] += microseconds
        check(sum(r["compile_seconds"] > 0 for r in occurrences) == 1, "one nonzero timer per primary graph")
        graphs.append({"graph_id": key, "kind": kind,
                       "node_counts": sorted({int(r["graph_node_count"]) for r in occurrences}),
                       "rank_occurrences": len(occurrences),
                       "nonzero_counter_occurrences": sum(r["compile_seconds"] > 0 for r in occurrences),
                       "compile_counter_seconds_sum": microseconds / 1e6,
                       "rows": [{k: r[k] for k in ["rank", "archive_member", "graph_name", "timestamp", "compile_seconds"]} for r in occurrences]})
    check(len(graphs) == 10 and len(rows) == 21, "expected ten graph IDs / twenty-one rank rows")
    training_log = member("training.log")
    isolation = [json.loads(line.removeprefix("ISOLATED_SUBMISSION ")) for line in training_log.splitlines()
                 if line.startswith("ISOLATED_SUBMISSION ")]
    check(len(isolation) == 1, "one sandbox receipt")
    check(isolation[0]["train_sha256"] == digest(source.encode()), "executed source hash")
    effective_caches = {k: rehearsal.get("fresh_empty_sdk_caches", {}).get(k, env.get(k))
                        for k in ENV_KEYS if "CACHE_DIR" in k or k == "NEURON_COMPILE_CACHE_URL" or k == "TORCH_NEURONX_DEBUG_DIR"}
    log_lines = [line for line in training_log.splitlines() if
                 any(s in line for s in ["micro_tokens=", "grad_accum_steps=", "cache hit", "cache miss", "Cache hit", "Cache miss", "Loaded NEFF", "loaded NEFF"])][:40]
    return {"bundle": OP + path, "bundle_sha256": expected,
            "remote_root": bundle["root"], "source_sha256": digest(source.encode()),
            "executed_source_path": isolation[0]["sandbox"] + "/train.py",
            "launch_command": command, "training_harness_sha256": launch.get("harness_sha256"),
            "launch_time": launch.get("utc"), "selected_launch_environment": {k: env.get(k) for k in ENV_KEYS},
            "selected_environment_scope": "archived launch environment, not a newly sampled worker environment; null means absent from launch receipt",
            "fresh_empty_sdk_caches_receipt": rehearsal.get("fresh_empty_sdk_caches"),
            "effective_cache_paths_from_launch_plus_fresh_harness_receipt": effective_caches,
            "startup_seconds": rehearsal["observations"][4]["observed_seconds"],
            "wall_seconds": rehearsal["wall_seconds"],
            "verified_members_sha256": verified,
            "unique_graphs": len(graphs), "rank_rows": len(rows),
            "compile_counter_seconds_sum_by_kind": {k: v / 1e6 for k, v in totals.items()},
            "compile_counter_seconds_sum": sum(totals.values()) / 1e6,
            "sdk_global_version_suffixes": sorted({g["graph_id"].split(":")[-1] for g in graphs}),
            "selected_training_log_lines": log_lines, "graphs": graphs}


def historical_stable_repeat():
    cases = [
        ("S1", "20260926-trn22-stable-source-pair/completed/results-bundle.json", "6424cfcfbb311a49a6e39080a88afa6d05b027ce951a09a506b2547e49742b0c"),
        ("W1", "20260926-trn22-stablepath-warm/completed/results-bundle.json", "e50418bffad38eaabf4d2b38be6c7d2e4d511bbe82b5591a5feec77d02436b5e"),
    ]
    out = {}
    for phase, path, expected in cases:
        raw = (ROOT / OP / path).read_bytes()
        check(digest(raw) == expected, path)
        files = json.loads(raw)["files"]
        verified = {}

        def member(name):
            entry = files[name]
            check(digest(entry["text"].encode()) == entry["sha256"] and len(entry["text"].encode()) == entry["bytes"], name)
            verified[name] = entry["sha256"]
            return entry["text"]

        launch = json.loads(member(phase + "-launch.json"))
        identity = json.loads(member(phase + "-pre-source-identity.json"))["executed_train"]
        after = json.loads(member(phase + "-post-source-identity.json"))["executed_train"]
        check(identity == after, "stable historical source identity")
        rehearsal = json.loads(member(phase.lower() + "/train/rehearsal.json"))
        graphs = collections.defaultdict(lambda: {"compile_us": 0, "rank_rows": 0})
        for name in sorted(files):
            if "metric_table_graph_stats_rank" not in name:
                continue
            for row in csv.DictReader(io.StringIO(member(name))):
                graph = graphs[row["graph_id"]]
                graph["compile_us"] += int(row["torch_neuronx_compile_us"])
                graph["rank_rows"] += 1
                graph["kind"] = next(k for k in ["forward", "backward", "inference"] if "_" + k + "_" in row["graph_name"])
        out[phase] = {"bundle": OP + path, "bundle_sha256": expected,
                      "executed_source_identity": identity, "training_command": launch["training_command"],
                      "selected_cache_paths": {k: v for k, v in launch["environment"].items() if k in ENV_KEYS},
                      "startup_seconds": rehearsal["observations"][4]["observed_seconds"],
                      "graphs": dict(graphs), "verified_members_sha256": verified,
                      "actual_import_origin_observed": launch["actual_import_origin_observed"]}
    first, second = out["S1"], out["W1"]
    out["comparison"] = {
        "same_executed_source_identity": first["executed_source_identity"] == second["executed_source_identity"],
        "same_training_command": first["training_command"] == second["training_command"],
        "same_cache_paths": first["selected_cache_paths"] == second["selected_cache_paths"],
        "identical_graph_ids": sorted(set(first["graphs"]) & set(second["graphs"])),
        "S1_compiler_counter_sum_seconds": sum(v["compile_us"] for v in first["graphs"].values()) / 1e6,
        "W1_compiler_counter_sum_seconds": sum(v["compile_us"] for v in second["graphs"].values()) / 1e6,
        "scope": "different 3e79 source, five steps, not accepted e1e2 or 8c6; timing plus identical frontend keys is consistent with reuse, not explicit NEFF-hit proof or isolated source-path causality",
    }
    return out


def main():
    runs = {name: read_run(*pin) for name, pin in RUNS.items()}
    pairs = []
    selected_pairs = [("e1e2_cold_trn22", "e1e2_reused_trn22"),
                      ("e1e2_cold_trn22", "e1e2_reused_trn21"),
                      ("e1e2_reused_trn21", "e1e2_reused_trn22"),
                      ("8c6_lr015_trn22", "8c6_lr021_trn22"),
                      ("8c6_lr021_trn21", "8c6_lr021_trn22"),
                      ("e1e2_reused_trn22", "8c6_lr021_trn22")]
    for a, b in selected_pairs:
        x, y = runs[a], runs[b]
        gx, gy = {g["graph_id"] for g in x["graphs"]}, {g["graph_id"] for g in y["graphs"]}
        pairs.append({"first": a, "second": b,
                      "same_source": x["source_sha256"] == y["source_sha256"],
                      "same_launch_command": x["launch_command"] == y["launch_command"],
                      "same_neff_namespace": x["effective_cache_paths_from_launch_plus_fresh_harness_receipt"]["TORCH_NEURONX_NEFF_CACHE_DIR"] == y["effective_cache_paths_from_launch_plus_fresh_harness_receipt"]["TORCH_NEURONX_NEFF_CACHE_DIR"],
                      "shared_graph_ids": sorted(gx & gy),
                      "first_minus_second_startup_seconds": x["startup_seconds"] - y["startup_seconds"],
                      "first_minus_second_compile_counter_seconds": x["compile_counter_seconds_sum"] - y["compile_counter_seconds_sum"]})
    print(json.dumps({"scope": "six pinned archived full-run bundles; no new host activity",
                      "runs": runs, "comparisons": pairs,
                      "historical_stable_source_repeat": historical_stable_repeat(),
                      "limits": ["compiler counter sum is neither concurrent wall time nor CPU work",
                                 "zero counters on sharing ranks are not proof of persistent cache hits",
                                 "graph_id is frontend serialized-MLIR content hash plus SDK suffix; not final NEFF filename",
                                 "no direct matched MLIR/HLO/NEFF payload is read or compared"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
