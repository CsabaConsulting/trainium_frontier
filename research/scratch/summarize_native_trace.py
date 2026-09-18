#!/usr/bin/env python3
"""Summarize native execution spans without summing overlapping runtime waits.

Chrome trace timestamps/durations are microseconds. This reports driver execution
spans grouped by NEFF identity and correlates them with the CPU operation's
External id. CPU/runtime/driver views overlap; their durations are never added.
The result is attribution evidence, not an unprofiled throughput measurement.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def merged_duration_us(events):
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    total = end = 0.0
    for lo, hi in intervals:
        total += max(0.0, hi - max(lo, end))
        end = max(end, hi)
    return total


def summarize(path):
    events = json.loads(path.read_text())["traceEvents"]
    cpu = defaultdict(set)
    for event in events:
        if event.get("cat") == "cpu_op" and event.get("ph") == "X":
            cpu[event.get("args", {}).get("External id")].add(event["name"])
    groups = defaultdict(list)
    driver_events = []
    for event in events:
        if (event.get("ph") == "X" and event.get("cat") == "privateuse1_driver"
                and event.get("name") == "model_default"):
            driver_events.append(event)
            groups[event["args"]["persistent_cache_key"]].append(event)
    rows = []
    for key, group in groups.items():
        spans = [event["dur"] / 1000 for event in group]
        rows.append(dict(
            persistent_cache_key=key,
            neff_cache_keys=sorted({e["args"]["neff_cache_key"] for e in group}),
            correlated_cpu_operations=sorted(set().union(*(
                cpu.get(e["args"].get("External id"), set()) for e in group))),
            external_ids=[e["args"].get("External id") for e in group],
            calls=len(group), execution_ms=spans, median_execution_ms=statistics.median(spans),
            union_execution_ms=merged_duration_us(group) / 1000,
        ))
    return dict(graphs=sorted(rows, key=lambda r: r["union_execution_ms"], reverse=True),
                all_graph_execution_union_ms=merged_duration_us(driver_events) / 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    files = []
    traces = []
    for path in sorted(args.root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(args.root))
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(data)
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=h.hexdigest()))
        if path.name == "trace.json":
            traces.append(dict(path=relative, **summarize(path)))
    assert traces, "No trace.json files found"
    result = dict(
        profile_root=str(args.root), files=files, total_bytes=sum(f["bytes"] for f in files),
        profiled_latency_is_performance_evidence=False,
        interpretation="Correlate driver spans by External id. Never sum CPU, runtime and "
                       "driver durations or treat queue wait as device operation time.",
        traces=traces,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Summarized {len(traces)} traces; hashed {len(files)} artifacts, "
          f"{result['total_bytes']:,} bytes -> {args.output}")


if __name__ == "__main__":
    main()
