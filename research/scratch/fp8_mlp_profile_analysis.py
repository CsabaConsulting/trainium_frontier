#!/usr/bin/env python3
"""Attribute an FP8 component trace with verified binary and phase mappings.

Read-only Parquet analysis. Match every runtime dispatch to the ingested NEFF
before attributing an execution to its enclosing annotation. Report merged
engine intervals and populated byte counters; profiling and missing dynamic
DMA metadata prohibit interpreting these values as unprofiled speed/bandwidth.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

COUNTERS = ('spill_save_bytes', 'spill_reload_bytes', 'hbm_read_bytes',
            'hbm_write_bytes', 'sbuf_read_bytes', 'sbuf_write_bytes')
COLUMNS = ['start_ts', 'end_ts', 'pcore_idx', 'engine', 'opcode', 'duration_ns',
           'nki_source_location', 'bir_instruction_name', 'tensor_instruction_type', *COUNTERS]


def merged_activity(table, lower, upper):
    """Union intervals separately per core/engine; use existing Arrow only."""
    result = []
    for key in table.select(['pcore_idx', 'engine']).group_by(['pcore_idx', 'engine']).aggregate([]).to_pylist():
        part = table.filter(pc.and_(pc.equal(table['pcore_idx'], key['pcore_idx']),
                                    pc.equal(table['engine'], key['engine'])))
        intervals = part.select(['start_ts', 'end_ts']).sort_by('start_ts')
        total, previous = 0, lower
        for start, end in zip(intervals['start_ts'].to_pylist(), intervals['end_ts'].to_pylist()):
            start, end = max(start, lower), min(end, upper)
            if end > max(start, previous):
                total += end - max(start, previous)
                previous = end
        result.append(dict(**key, instruction_rows=len(part), active_ms=total / 1e6))
    return sorted(result, key=lambda row: (row['pcore_idx'], row['engine']))


def totals(table):
    return {name: dict(nonnull_rows=len(table[name]) - table[name].null_count,
                       positive_rows=int(pc.sum(pc.cast(pc.fill_null(
                           pc.greater(table[name], 0), False), pa.int64())).as_py() or 0),
                       sum=int(pc.sum(table[name]).as_py() or 0))
            for name in COUNTERS}


def top_groups(table, fields, lower, upper, limit=18):
    grouped = table.group_by(fields).aggregate([
        ('duration_ns', 'sum'), ('start_ts', 'count'),
        *[(field, 'sum') for field in COUNTERS]])
    rows = sorted(grouped.to_pylist(), key=lambda row: row['duration_ns_sum'], reverse=True)[:limit]
    for row in rows:
        mask = None
        for field in fields:
            cond = pc.is_null(table[field]) if row[field] is None else pc.equal(table[field], row[field])
            mask = cond if mask is None else pc.and_(mask, cond)
        part = table.filter(mask)
        row['merged_activity_by_core_engine'] = merged_activity(part, lower, upper)
        row['examples'] = part.select(['bir_instruction_name', 'opcode']).slice(0, 3).to_pylist()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-pattern', default='fp8_mlp_',
                        help='NKI source substring for per-line attribution; defaults to the FP8 MLP')
    parser.add_argument('--all-phases', action='store_true',
                        help='Also attribute current-cache preparation and complete backward/accumulation')
    args = parser.parse_args()
    pa.set_cpu_count(4)
    root = next(args.parquet.glob('*nc_0_session_0'))
    events = json.loads(next(args.trace.rglob('trace.json')).read_text())['traceEvents']
    manifest = json.loads((args.trace / 'neffs/manifest.json').read_text())
    logical = {entry['cache_key']: entry['file'] for entry in manifest['exported']}
    dispatches = sorted((event for event in events if event.get('ph') == 'X'
                        and event.get('cat') == 'privateuse1_runtime'
                        and event.get('args', {}).get('persistent_cache_key')),
                        key=lambda event: event['args']['nrt_sequence_id'])
    executions = sorted(pq.read_table(root / 'ExecutionInfo.parquet').to_pylist(),
                        key=lambda row: row['execution_index'])
    assert len(executions) == len(dispatches)
    for execution, event in zip(executions, dispatches):
        attrs = event['args']
        binary = logical.get(attrs.get('neff_cache_key'), attrs['persistent_cache_key'] + '.neff')
        assert binary == execution['neff_name'], (execution, binary)
        parents = [annotation for annotation in events
                   if annotation.get('cat') == 'user_annotation'
                   and annotation['ts'] <= event['ts'] < annotation['ts'] + annotation['dur']]
        execution['phase'] = min(parents, key=lambda annotation: annotation['dur'])['name']
        execution['device_wall_ms'] = (execution['execution_end_ts'] - execution['execution_start_ts']) / 1e6
        execution['nrt_sequence_id'] = attrs['nrt_sequence_id']

    ins = pq.read_table(root / 'Instruction.parquet', columns=COLUMNS)
    active = pq.read_table(root / 'ActiveTime.parquet')
    usage = pq.read_table(root / 'SbufUsage.parquet')
    analyses = []
    for execution in executions:
        if not args.all_phases and not execution['phase'].endswith('/forward'):
            continue
        lo, hi = execution['execution_start_ts'], execution['execution_end_ts']
        overlaps = [other['execution_index'] for other in executions
                    if other['execution_index'] != execution['execution_index']
                    and other['execution_start_ts'] < hi and other['execution_end_ts'] > lo]
        assert not overlaps, (execution, overlaps)
        part = ins.filter(pc.and_(pc.greater_equal(ins['start_ts'], lo), pc.less(ins['start_ts'], hi)))
        sampled = usage.filter(pc.and_(pc.greater_equal(usage['timestamp'], lo),
                                       pc.less(usage['timestamp'], hi)))
        busy = active.filter(pc.and_(pc.less(active['start_ts'], hi), pc.greater(active['end_ts'], lo)))
        transpose = part.filter(pc.fill_null(pc.equal(part['tensor_instruction_type'], 'TRANSPOSE'), False))
        compiler_transpose = transpose.filter(pc.fill_null(pc.match_substring(
            transpose['nki_source_location'], 'private_nkl/transpose.py'), False))
        source_rows = part.filter(pc.fill_null(pc.match_substring(
            part['nki_source_location'], args.source_pattern), False))
        source = (dict(top=top_groups(source_rows, ['nki_source_location'], lo, hi, limit=18))
                  if len(source_rows) else None)
        ops = part.group_by(['engine', 'opcode']).aggregate([
            ('duration_ns', 'sum'), ('start_ts', 'count'),
            *[(field, 'sum') for field in COUNTERS]])
        analysis = dict(execution=execution, instruction_rows=len(part),
                        interval_overlaps_other_executions=overlaps,
                        counters=totals(part), engine_activity=merged_activity(busy, lo, hi),
                        tensor_transpose=dict(instruction_rows=len(transpose),
                                              activity=merged_activity(transpose, lo, hi)),
                        compiler_private_transpose=dict(instruction_rows=len(compiler_transpose),
                            activity=merged_activity(compiler_transpose, lo, hi)),
                        source_groups=source,
                        top_opcodes=sorted(ops.to_pylist(), key=lambda row: row['duration_ns_sum'], reverse=True)[:20],
                        sbuf_sampled_peaks=sampled.group_by('pcore_idx').aggregate([
                            ('avg_util_percent', 'max'), ('timestamp', 'count')]).to_pylist())
        analyses.append(analysis)
    schema = [row for row in pq.read_table(root / 'SchemaFields.parquet').to_pylist()
              if (row['table_name'] == 'Instruction' and row['field_name'] in COUNTERS)
              or row['table_name'] == 'SbufUsage']
    report = dict(trace=str(args.trace), parquet=str(root), source_pattern=args.source_pattern,
                  all_runtime_dispatches_match_ingested_binaries=True,
                  executions=executions, all_phases_requested=args.all_phases,
                  phase_analyses=analyses,
                  forward_analyses=[a for a in analyses
                                    if a['execution']['phase'].endswith('/forward')],
                  warnings=pq.read_table(root / 'Warning.parquet').to_pylist(),
                  installed_schema=schema, profiled_times_are_not_speed_benchmarks=True,
                  source_group_activity_overlaps_and_is_not_additive=True,
                  dynamic_dma_warning_prevents_exact_bandwidth_claim=True,
                  hashes={name: hashlib.file_digest((root / name).open('rb'), 'sha256').hexdigest()
                          for name in ('ExecutionInfo.parquet', 'Instruction.parquet', 'ActiveTime.parquet',
                                       'Warning.parquet', 'SchemaFields.parquet')})
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for analysis in analyses:
        print(json.dumps(dict(phase=analysis['execution']['phase'],
              device_wall_ms=analysis['execution']['device_wall_ms'],
              counters=analysis['counters'], engine_activity=analysis['engine_activity'],
              transpose=analysis['tensor_transpose'],
              sampled_sbuf=analysis['sbuf_sampled_peaks'])), flush=True)


if __name__ == '__main__':
    main()
