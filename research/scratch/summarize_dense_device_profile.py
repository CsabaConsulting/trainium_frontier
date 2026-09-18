#!/usr/bin/env python3
"""Summarize an ingested native device trace without summing overlapping engines.

Run on the copied Parquet directory, not on an accelerator. ExecutionInfo defines
the measured graph windows. ActiveTime is already merged per engine/physical core;
clip those intervals to each graph and normalize by that graph's wall time. Raw
instruction-duration sums are reported separately and are NOT wall-time costs.
Keep missing-metadata warnings alongside all results. In particular, incomplete
DMA metadata cannot establish a reliable memory-bandwidth lower bound.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.directory
    read = lambda name, **kw: pq.read_table(root / (name + ".parquet"), **kw)
    metadata = read("Metadata").to_pylist()
    executions = sorted(read("ExecutionInfo").to_pylist(), key=lambda x: x["execution_index"])
    active = read("ActiveTime")
    columns = ["start_ts", "end_ts", "engine", "pcore_idx", "opcode", "duration_ns",
               "performance_mode", "tensor_instruction_type", "raw_flops",
               "adjusted_flops", "flops_transpose", "adjusted_flops_transpose",
               "hlo_name", "layer", "nki_source_location", "kernel_instruction_name",
               "penguin_id", "bir_instruction_name"]
    instructions = read("Instruction", columns=columns)
    attribution = {}
    for name in ["hlo_name", "layer", "nki_source_location", "kernel_instruction_name"]:
        values = instructions[name]
        nonempty = pc.fill_null(pc.not_equal(values, ""), False)
        attribution[name] = int(pc.sum(pc.cast(nonempty, pa.int64())).as_py())
    graphs = []
    for execution in executions:
        start, end = execution["execution_start_ts"], execution["execution_end_ts"]
        duration = end - start
        if duration < 1_000_000:
            continue
        selected = active.filter(pc.and_(pc.less(active["start_ts"], end),
                                          pc.greater(active["end_ts"], start)))
        clipped = pc.subtract(pc.min_element_wise(selected["end_ts"], end),
                              pc.max_element_wise(selected["start_ts"], start))
        selected = selected.append_column("clipped_ns", clipped)
        busy = selected.group_by(["pcore_idx", "engine"]).aggregate([("clipped_ns", "sum")]).to_pylist()
        for row in busy:
            row["active_ms"] = row.pop("clipped_ns_sum") / 1e6
            row["fraction_of_graph_wall_time"] = row["active_ms"] / (duration / 1e6)
        ins = instructions.filter(pc.and_(pc.greater_equal(instructions["start_ts"], start),
                                           pc.less(instructions["start_ts"], end)))
        counts = ins.group_by(["engine", "pcore_idx", "performance_mode",
                               "tensor_instruction_type"]).aggregate(
            [("duration_ns", "sum"), ("raw_flops", "sum"), ("adjusted_flops", "sum"),
             ("flops_transpose", "sum"), ("adjusted_flops_transpose", "sum"),
             ("start_ts", "count")]).to_pylist()
        opcodes = ins.group_by(["engine", "opcode"]).aggregate(
            [("duration_ns", "sum"), ("start_ts", "count")]).to_pylist()
        sources = ins.group_by(["nki_source_location"]).aggregate(
            [("duration_ns", "sum"), ("start_ts", "count")]).to_pylist()
        dma = ins.filter(pc.equal(ins["opcode"], "DMA_DIRECT2D"))
        dma_groups = dma.group_by(["penguin_id"]).aggregate(
            [("duration_ns", "sum"), ("start_ts", "count"),
             ("start_ts", "min"), ("end_ts", "max")]).to_pylist()
        dma_groups = sorted(dma_groups, key=lambda x: x["duration_ns_sum"], reverse=True)[:10]
        for row in dma_groups:
            samples = dma.filter(pc.equal(dma["penguin_id"], row["penguin_id"]))
            row["example_bir_instruction"] = samples["bir_instruction_name"][0].as_py()
        graphs.append(dict(**execution, wall_ms=duration / 1e6,
                           active_by_physical_core=sorted(busy, key=lambda x: (x["pcore_idx"], x["engine"])),
                           instructions_by_type=counts,
                           top_instruction_duration_sums=sorted(opcodes, key=lambda x: x["duration_ns_sum"], reverse=True)[:20],
                           source_instruction_duration_sums=sorted(sources, key=lambda x: x["duration_ns_sum"], reverse=True)[:20],
                           top_dma_instruction_groups=dma_groups))
    files = []
    for path in sorted(root.glob("*.parquet")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        files.append(dict(name=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest()))
    result = dict(directory=str(root), metadata=metadata,
                  warnings=read("Warning").to_pylist(), summary_as_reported=read("Summary").to_pylist(),
                  caveats=["Profiled execution is for attribution; use unprofiled phase timing for throughput.",
                           "Active fractions are independently derived for each physical core and graph.",
                           "Instruction-duration sums overlap and cannot be treated as operation wall times.",
                           "Missing DMA notifications and dropped packets preclude an exact HBM bound.",
                           "HLO/kernel attribution requires nonempty source fields; no names are inferred from DMA counts."],
                  source_attribution_nonempty_rows=attribution,
                  annotations=read("TimelineAnnotation").to_pylist(),
                  executions=executions, graphs=graphs, files=files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), metadata=metadata,
                          attribution=attribution, warnings=result["warnings"],
                          graphs=[dict(neff=g["neff_name"], wall_ms=g["wall_ms"],
                                       active=g["active_by_physical_core"]) for g in graphs])))


if __name__ == "__main__":
    main()
