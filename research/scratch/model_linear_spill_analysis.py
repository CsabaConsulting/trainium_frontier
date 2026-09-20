#!/usr/bin/env python3
"""Separate arithmetic from spill instructions attributed to ordinary Linear.

Requires the already identity-matched caller analyses. Instruction intervals
can overlap; reported unions are attribution, not additive wall-time shares.
Dropped DMA notifications still prohibit exact byte or bandwidth conclusions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from fp8_mlp_profile_analysis import merged_activity


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def analyze(path):
    report = json.loads(path.read_text())
    assert report['all_runtime_dispatches_match_ingested_binaries']
    assert len(report['graph_analyses']) == 1
    graph = report['graph_analyses'][0]
    lo, hi = (graph['execution'][k] for k in ('execution_start_ts', 'execution_end_ts'))
    parquet = Path(report['parquet']) / 'Instruction.parquet'
    columns = ['start_ts', 'end_ts', 'duration_ns', 'pcore_idx', 'engine', 'opcode',
               'tensor_instruction_type', 'raw_flops', 'adjusted_flops',
               'spill_reload_bytes', 'spill_save_bytes', 'nki_source_location', 'hlo_name', 'layer']
    table = pq.read_table(parquet, columns=columns)
    mask = pc.and_(pc.greater_equal(table['start_ts'], lo), pc.less(table['start_ts'], hi))
    mask = pc.and_(mask, pc.fill_null(pc.match_substring(table['nki_source_location'],
                                                       'train.py:2853'), False))
    table = table.filter(mask)
    source = next(row for row in graph['top_source_groups']
                  if row['nki_source_location'] and 'train.py:2853' in row['nki_source_location'])
    assert table.num_rows == source['start_ts_count']
    keys = ['pcore_idx', 'engine', 'opcode', 'tensor_instruction_type']
    rows = table.group_by(keys).aggregate([
        ('start_ts', 'count'), ('duration_ns', 'sum'), ('raw_flops', 'sum'),
        ('adjusted_flops', 'sum'), ('spill_reload_bytes', 'sum'), ('spill_save_bytes', 'sum')
    ]).to_pylist()
    for row in rows:
        mask = pa.array([True] * table.num_rows)
        for key in keys:
            mask = pc.and_(mask, pc.is_null(table[key]) if row[key] is None
                           else pc.equal(table[key], row[key]))
        activity = merged_activity(table.filter(mask), lo, hi)
        assert len(activity) == 1
        row['active_union_ms'] = activity[0]['active_ms']
    rows.sort(key=lambda row: (row['pcore_idx'], -row['active_union_ms']))
    regular = table.filter(pc.and_(pc.equal(table['opcode'], 'MATMUL'),
                                  pc.equal(table['tensor_instruction_type'], 'REGULAR')))
    buckets = regular.group_by(['pcore_idx', 'raw_flops', 'adjusted_flops']).aggregate([
        ('start_ts', 'count'), ('duration_ns', 'sum')]).to_pylist()
    return dict(analysis=str(path), analysis_sha256=digest(path),
                instruction_parquet=str(parquet), instruction_sha256=digest(parquet),
                execution=graph['execution'], linear_instruction_count=table.num_rows,
                opcode_groups=rows, regular_matmul_work_buckets=buckets,
                hlo_names=pc.unique(table['hlo_name']).to_pylist(),
                layer_names=pc.unique(table['layer']).to_pylist(), warnings=report['warnings'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pa.set_cpu_count(4)
    result = dict(analysis_script_sha256=digest(Path(__file__)),
                  native=analyze(args.native), candidate=analyze(args.candidate),
                  caveats=['Instruction/source intervals overlap; unions are not additive wall-time shares.',
                           'Dropped DMA notifications and missing dynamic metadata make byte counters incomplete.',
                           'Empty HLO/layer metadata cannot identify individual Linear module names.',
                           'Source attribution does not establish the cause of scheduling changes.'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for arm in ('native', 'candidate'):
        print(json.dumps(dict(arm=arm, groups=result[arm]['opcode_groups'][:8])), flush=True)


if __name__ == '__main__':
    main()
