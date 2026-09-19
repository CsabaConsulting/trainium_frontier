#!/usr/bin/env python3
"""Separate shared Linear math, transpose and load activity in preserved callers.

Runs locally on already verified Parquet data; no device or mutable experiment
state. This shared source line cannot identify an individual projection layer.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', nargs=3, action='append', required=True,
                        metavar=('NAME', 'ANALYSIS_JSON', 'LOCAL_PARQUET'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pa.set_cpu_count(4)
    arms = {}
    columns = ['nki_source_location', 'pcore_idx', 'engine', 'opcode',
               'tensor_instruction_type', 'start_ts', 'end_ts', 'raw_flops',
               'adjusted_flops', 'spill_save_bytes', 'spill_reload_bytes']
    for name, analysis_path, directory in args.arm:
        assert name not in arms
        authority = json.loads(Path(analysis_path).read_text())
        assert authority['all_runtime_dispatches_match_ingested_binaries']
        paths = list(Path(directory).rglob('Instruction.parquet'))
        assert len(paths) == 1
        expected = [value for key, value in authority['sha256'].items()
                    if key.endswith('/Instruction.parquet')]
        assert len(expected) == 1 and digest(paths[0]) == expected[0]
        graphs = []
        for graph in authority['graph_analyses']:
            execution = graph['execution']
            lo, hi = execution['execution_start_ts'], execution['execution_end_ts']
            table = pq.read_table(paths[0], columns=columns, filters=[
                ('start_ts', '>=', lo), ('start_ts', '<', hi)])
            table = table.filter(pc.fill_null(pc.ends_with(
                table['nki_source_location'], '/train.py:2853'), False))
            table = table.append_column('duration_ns', pc.subtract(table['end_ts'], table['start_ts']))
            for field in ('engine', 'opcode', 'tensor_instruction_type'):
                table = table.set_column(table.schema.get_field_index(field), field,
                                         pc.fill_null(table[field], '<missing>'))
            grouped = table.group_by(['pcore_idx', 'engine', 'opcode',
                                      'tensor_instruction_type']).aggregate([
                ('start_ts', 'count'), ('duration_ns', 'sum'), ('raw_flops', 'sum'),
                ('adjusted_flops', 'sum'), ('spill_save_bytes', 'sum'),
                ('spill_reload_bytes', 'sum')]).to_pylist()
            for row in grouped:
                mask = None
                for field in ('pcore_idx', 'engine', 'opcode', 'tensor_instruction_type'):
                    term = pc.equal(table[field], row[field])
                    mask = term if mask is None else pc.and_(mask, term)
                selected = table.filter(mask)
                row['merged_activity'] = merged_activity(selected, lo, hi)
            grouped.sort(key=lambda row: row['duration_ns_sum'] or 0, reverse=True)
            regular = table.filter(pc.and_(pc.equal(table['opcode'], 'MATMUL'),
                                            pc.equal(table['tensor_instruction_type'], 'REGULAR')))
            size_groups = regular.group_by(['pcore_idx', 'raw_flops', 'adjusted_flops']).aggregate([
                ('start_ts', 'count'), ('duration_ns', 'sum'), ('duration_ns', 'mean')]).to_pylist()
            size_groups.sort(key=lambda row: row['start_ts_count'], reverse=True)
            graphs.append(dict(execution=execution, shared_linear_rows=len(table),
                               opcode_groups=grouped, regular_matmul_work_groups=size_groups))
            print('LINEAR_OPCODES ' + json.dumps(dict(arm=name,
                execution_index=execution['execution_index'], top_groups=grouped[:8])), flush=True)
            del table
        arms[name] = dict(authority=analysis_path, authority_sha256=digest(Path(analysis_path)),
                          instruction_sha256=expected[0], graphs=graphs,
                          warnings=authority['warnings'])
    result = dict(analysis_source_sha256=digest(Path(__file__)), arms=arms,
                  caveats=[
                      'Shared Linear source attribution does not identify QKV, head, or a specific layer.',
                      'Source attribution and metadata can be incomplete; preserve all capture warnings.',
                      'Activity unions overlap and are not additive wall-time shares.',
                      'FP8 capture dropped EVENT/DMA notifications; native capture dropped DMA notifications.',
                      'This is attribution, not a performance comparison or proof of allocator causality.'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), arms=list(arms))), flush=True)


if __name__ == '__main__':
    main()
