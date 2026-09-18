#!/usr/bin/env python3
"""Compare saved full-forward graphs for SBUF spills and layout work.

This reads existing Parquet only. Byte counters are reported with their
population rates and installed schema; missing DMA metadata prevents an exact
HBM bandwidth claim. Instruction durations are interval-merged per physical
core/engine before being called active time. Source groups overlap each other
and cannot be added to reconstruct the critical path.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


COUNTERS = ("spill_save_bytes", "spill_reload_bytes", "hbm_read_bytes",
            "hbm_write_bytes", "sbuf_read_bytes", "sbuf_write_bytes")
COLUMNS = ["start_ts", "end_ts", "pcore_idx", "engine", "opcode", "duration_ns",
           "nki_source_location", "bir_debug_info_source_location", "layer",
           "bir_instruction_name", "penguin_id", "tensor_instruction_type",
           "raw_flops", "adjusted_flops", "adjusted_flops_transpose", *COUNTERS]


def union_ns(starts, ends, lower, upper):
    starts, ends = np.maximum(starts, lower), np.minimum(ends, upper)
    order = np.argsort(starts)
    total, previous = 0, lower
    for i in order:
        start, end = int(starts[i]), int(ends[i])
        if end > max(start, previous):
            total += end - max(start, previous)
            previous = end
    return total


def merged_activity(table, lower, upper):
    rows = []
    for key in table.select(["pcore_idx", "engine"]).group_by(
            ["pcore_idx", "engine"]).aggregate([]).to_pylist():
        part = table.filter(pc.and_(pc.equal(table["pcore_idx"], key["pcore_idx"]),
                                    pc.equal(table["engine"], key["engine"])))
        rows.append(dict(**key, instruction_rows=len(part),
                         active_ms=union_ns(part["start_ts"].to_numpy(),
                                            part["end_ts"].to_numpy(), lower, upper) / 1e6))
    return sorted(rows, key=lambda r: (r["pcore_idx"], r["engine"]))


def totals(table):
    result = {}
    for name in COUNTERS:
        values = table[name]
        result[name] = dict(nonnull_rows=len(values) - values.null_count,
                            positive_rows=int(pc.sum(pc.cast(pc.fill_null(
                                pc.greater(values, 0), False), pa.int64())).as_py() or 0),
                            sum=int(pc.sum(values).as_py() or 0))
    return result


def top_groups(table, fields, lower, upper, limit=20):
    if not len(table):
        return []
    grouped = table.group_by(fields).aggregate(
        [("duration_ns", "sum"), ("start_ts", "count"),
         ("start_ts", "min"), ("end_ts", "max"),
         *[(name, "sum") for name in COUNTERS]])
    rows = sorted(grouped.to_pylist(), key=lambda r: r["duration_ns_sum"], reverse=True)[:limit]
    for row in rows:
        mask = None
        for field in fields:
            cond = (pc.is_null(table[field]) if row[field] is None
                    else pc.equal(table[field], row[field]))
            mask = cond if mask is None else pc.and_(mask, cond)
        part = table.filter(mask)
        row["merged_activity_by_core_engine"] = merged_activity(part, lower, upper)
        row["examples"] = part.select(["nki_source_location", "layer",
                                        "bir_instruction_name", "opcode"]).slice(0, 3).to_pylist()
    return rows


def analyze(root, neff):
    def read(name, **kwargs):
        return pq.read_table(root / (name + ".parquet"), **kwargs)
    executions = [r for r in read("ExecutionInfo").to_pylist() if r["neff_name"] == neff]
    assert len(executions) == 1, (neff, executions)
    execution = executions[0]
    lower, upper = execution["execution_start_ts"], execution["execution_end_ts"]
    all_ins = read("Instruction", columns=COLUMNS)
    ins = all_ins.filter(pc.and_(pc.greater_equal(all_ins["start_ts"], lower),
                                  pc.less(all_ins["start_ts"], upper)))
    positive_spill = pc.or_(pc.fill_null(pc.greater(ins["spill_save_bytes"], 0), False),
                            pc.fill_null(pc.greater(ins["spill_reload_bytes"], 0), False))
    named_spill = pc.fill_null(pc.match_substring_regex(ins["bir_instruction_name"],
                                                       "(?i)spill"), False)
    mlp = pc.fill_null(pc.match_substring(ins["nki_source_location"],
                                         "mlp_fused_bf16.py"), False)
    transpose = pc.fill_null(pc.equal(ins["tensor_instruction_type"], "TRANSPOSE"), False)
    subsets = {}
    for name, mask in (("positive_spill_counter", positive_spill), ("named_spill", named_spill),
                       ("custom_mlp", mlp), ("tensor_transpose", transpose)):
        sub = ins.filter(mask)
        subsets[name] = dict(instruction_rows=len(sub), counters=totals(sub),
                             activity=merged_activity(sub, lower, upper),
                             source_groups=top_groups(sub, ["nki_source_location"], lower, upper),
                             layer_groups=top_groups(sub, ["layer"], lower, upper, 12))
    active = read("ActiveTime")
    active = active.filter(pc.and_(pc.less(active["start_ts"], upper),
                                    pc.greater(active["end_ts"], lower)))
    metadata = read("Metadata").to_pylist()
    schema = [r for r in read("SchemaFields").to_pylist()
              if r["table_name"] == "Instruction" and r["field_name"] in COUNTERS]
    flop_groups = ins.filter(pc.equal(ins["opcode"], "MATMUL")).group_by(
        ["pcore_idx", "tensor_instruction_type"]).aggregate(
            [("raw_flops", "sum"), ("adjusted_flops", "sum"),
             ("adjusted_flops_transpose", "sum")]).to_pylist()
    return dict(directory=str(root), execution=execution, wall_ms=(upper - lower) / 1e6,
                instruction_rows=len(ins), counters=totals(ins), metadata=metadata,
                installed_counter_schema=schema, warnings=read("Warning").to_pylist(),
                active_by_core_engine=merged_activity(active, lower, upper),
                subsets=subsets, matmul_flops_by_core_type=flop_groups,
                top_sources=top_groups(ins, ["nki_source_location"], lower, upper),
                input_sha256={name: hashlib.file_digest((root / (name + ".parquet")).open("rb"),
                                                       "sha256").hexdigest()
                              for name in ("Instruction", "ExecutionInfo", "ActiveTime", "Warning")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--reference-neff", required=True)
    ap.add_argument("--candidate-neff", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    # Guard the interval arithmetic against overlap/double-counting mistakes.
    assert union_ns(np.array([0, 2, 8, 15]), np.array([4, 6, 10, 18]), 1, 16) == 8
    result = dict(reference=analyze(args.reference, args.reference_neff),
                  candidate=analyze(args.candidate, args.candidate_neff),
                  caveats=["Profiled wall times diagnose execution, not unprofiled throughput.",
                           "Instruction group durations overlap: only merged per-core/engine intervals are active times.",
                           "Spill instructions may overlap useful work; active time is not a causal slowdown estimate.",
                           "Counter coverage and warnings determine whether byte totals are complete.",
                           "Exact HBM bounds are unavailable when DMA notifications or dynamic metadata are missing."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({kind: dict(wall_ms=result[kind]["wall_ms"],
                                  counters=result[kind]["counters"],
                                  named_spill_rows=result[kind]["subsets"]["named_spill"]["instruction_rows"])
                      for kind in ("reference", "candidate")}), flush=True)


if __name__ == "__main__":
    main()
