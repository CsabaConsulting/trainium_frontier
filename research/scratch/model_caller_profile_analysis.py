#!/usr/bin/env python3
"""Attribute large caller graphs only after matching every runtime dispatch.

Use the installed Explorer schema, retained NEFF manifests and device intervals.
Runtime API durations include asynchronous enqueue/queue effects and are never
summed as device time. Engine/source intervals overlap: their unions are useful
for attribution but are not additive shares of execution time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from fp8_mlp_profile_analysis import COLUMNS, COUNTERS, merged_activity, top_groups, totals


def present_match(column, text):
    return pc.fill_null(pc.match_substring(column, text), False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True,
                        help='Rank directory containing neffs/manifest.json and trace.json')
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-graph-ms', type=float, default=1.0)
    args = parser.parse_args()
    pa.set_cpu_count(4)
    roots = list(args.parquet.glob('*nc_0_session_0'))
    if len(roots) != 1:
        raise ValueError(f'Expected one rank/core session, found {roots}')
    root = roots[0]
    traces = list(args.trace.rglob('trace.json'))
    if len(traces) != 1:
        raise ValueError(f'Expected one runtime trace, found {traces}')
    events = json.loads(traces[0].read_text())['traceEvents']
    manifest_path = args.trace / 'neffs/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    logical = {entry['cache_key']: entry['file'] for entry in manifest['exported']}
    dispatches = sorted((event for event in events if event.get('ph') == 'X'
                         and event.get('cat') == 'privateuse1_runtime'
                         and event.get('args', {}).get('persistent_cache_key')),
                        key=lambda event: event['args']['nrt_sequence_id'])
    executions = sorted(pq.read_table(root / 'ExecutionInfo.parquet').to_pylist(),
                        key=lambda row: row['execution_index'])
    auxiliary = []
    if len(executions) != len(dispatches):
        # phase_timer aligns all ranks AFTER enabling the profiler and BEFORE
        # the annotated microstep. Its c10d barrier has a real device binary
        # but no ordinary privateuse1_runtime persistent-key dispatch. Accept
        # only this explicit one-to-one, ordered, preserved startup case.
        assert len(executions) == len(dispatches) + 1
        first = executions[0]
        barriers = [event for event in events if event.get('ph') == 'X'
                    and event.get('cat') == 'cpu_op' and event.get('name') == 'c10d::barrier']
        annotations = [event for event in events if event.get('cat') == 'user_annotation']
        assert len(barriers) == 1 and annotations
        assert first['execution_index'] == 0 and first['execution_start_ts'] == 0
        assert first['num_subgraphs'] == 1
        assert re.fullmatch(r'barrier_[0-9a-f]{32}\.neff', first['neff_name'])
        barrier, first_annotation = barriers[0], min(annotations, key=lambda event: event['ts'])
        assert barrier['ts'] + barrier['dur'] <= first_annotation['ts']
        assert first['execution_end_ts'] <= executions[1]['execution_start_ts']
        binary = args.trace / 'neffs' / first['neff_name']
        assert binary.is_file(), binary
        with binary.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        auxiliary.append(dict(device_execution=first, cpu_event=barrier,
                              binary=str(binary), sha256=digest,
                              classification='preserved c10d startup barrier before timed annotation'))
        executions = executions[1:]
    assert len(executions) == len(dispatches), (len(executions), len(dispatches))
    for execution, event in zip(executions, dispatches):
        attrs = event['args']
        binary = logical.get(attrs.get('neff_cache_key'), attrs['persistent_cache_key'] + '.neff')
        assert execution['neff_name'] == binary, (execution['neff_name'], binary)
        binary_path = args.trace / 'neffs' / binary
        assert binary_path.is_file(), binary_path
        parents = [annotation for annotation in events
                   if annotation.get('cat') == 'user_annotation'
                   and annotation['ts'] <= event['ts'] < annotation['ts'] + annotation['dur']]
        execution.update(runtime_name=event['name'], nrt_sequence_id=attrs['nrt_sequence_id'],
                         logical_cache_key=attrs.get('neff_cache_key'),
                         enclosing_annotation=(min(parents, key=lambda a: a['dur'])['name']
                                               if parents else None),
                         device_wall_ms=(execution['execution_end_ts']
                                         - execution['execution_start_ts']) / 1e6)

    columns = list(dict.fromkeys(COLUMNS + ['raw_flops', 'adjusted_flops']))
    ins = pq.read_table(root / 'Instruction.parquet', columns=columns)
    active = pq.read_table(root / 'ActiveTime.parquet')
    usage = pq.read_table(root / 'SbufUsage.parquet')
    analyses = []
    for execution in executions:
        if execution['device_wall_ms'] < args.minimum_graph_ms:
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
        private_transpose = transpose.filter(present_match(
            transpose['nki_source_location'], 'private_nkl/transpose.py'))
        regular = part.filter(pc.and_(
            pc.and_(pc.equal(part['opcode'], 'MATMUL'), pc.equal(part['tensor_instruction_type'], 'REGULAR')),
            pc.and_(pc.greater(part['raw_flops'], 0), pc.less_equal(part['end_ts'], hi))))
        regular = regular.append_column('bf16_cost_multiplier',
                                         pc.divide(pc.cast(regular['adjusted_flops'], pa.float64()),
                                                   pc.cast(regular['raw_flops'], pa.float64())))
        precision = regular.group_by(['pcore_idx', 'bf16_cost_multiplier']).aggregate([
            ('raw_flops', 'sum'), ('adjusted_flops', 'sum'), ('start_ts', 'count')]).to_pylist()
        fp32 = regular.filter(pc.equal(regular['bf16_cost_multiplier'], 4.0))
        fp32_sources = fp32.group_by('nki_source_location').aggregate([
            ('raw_flops', 'sum'), ('adjusted_flops', 'sum'), ('start_ts', 'count')]).to_pylist()
        sources = {}
        for name, match in (
                ('fp8_forward', 'fp8_mlp_token_rows_nki.py'),
                ('fp8_surrogate', 'fp8_mlp_island_autograd.py'),
                ('bf16_mlp', 'mlp_fused_bf16.py'),
                ('softcap_ce', 'softcap_ce_parallel_nki.py'),
                ('attention', '/attention/'),
                ('functional', '/torch/nn/functional.py'),
                ('private_transpose', 'private_nkl/transpose.py')):
            selected = part.filter(present_match(part['nki_source_location'], match))
            sources[name] = dict(instruction_rows=len(selected), counters=totals(selected),
                                 activity=merged_activity(selected, lo, hi))
        analysis = dict(execution=execution, instruction_rows=len(part),
                        counters=totals(part), engine_activity=merged_activity(busy, lo, hi),
                        regular_matmul_precision_by_core=precision,
                        fp32_matmul_sources=sorted(fp32_sources, key=lambda r: r['raw_flops_sum'], reverse=True),
                        transpose=dict(instruction_rows=len(transpose), activity=merged_activity(transpose, lo, hi)),
                        private_transpose=dict(instruction_rows=len(private_transpose),
                                               activity=merged_activity(private_transpose, lo, hi)),
                        sampled_sbuf_peaks=sampled.group_by('pcore_idx').aggregate([
                            ('avg_util_percent', 'max'), ('timestamp', 'count')]).to_pylist(),
                        source_families=sources,
                        top_source_groups=top_groups(part, ['nki_source_location'], lo, hi, limit=24))
        analyses.append(analysis)
        print('CALLER_GRAPH ' + json.dumps(dict(execution=execution,
              counters=analysis['counters'], engines=analysis['engine_activity'],
              matmul_precision=precision, private_transpose=analysis['private_transpose'],
              sampled_sbuf=analysis['sampled_sbuf_peaks'])), flush=True)

    files = [root / name for name in ('ExecutionInfo.parquet', 'Instruction.parquet',
             'ActiveTime.parquet', 'SbufUsage.parquet', 'Warning.parquet', 'SchemaFields.parquet')]
    files += [traces[0], manifest_path]
    hashes = {}
    for path in files:
        with path.open('rb') as stream:
            hashes[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(trace=str(args.trace), parquet=str(root),
                  all_runtime_dispatches_match_ingested_binaries=True,
                  executions=executions, graph_analyses=analyses,
                  explicitly_matched_auxiliary_executions=auxiliary,
                  total_device_executions_accounted_for=len(executions) + len(auxiliary),
                  warnings=pq.read_table(root / 'Warning.parquet').to_pylist(),
                  metric_schema=[row for row in pq.read_table(root / 'SchemaFields.parquet').to_pylist()
                      if (row['table_name'] == 'Instruction' and row['field_name'] in
                          (*COUNTERS, 'raw_flops', 'adjusted_flops', 'tensor_instruction_type'))
                      or row['table_name'] == 'SbufUsage'],
                  caveats=['Use unprofiled same-host phase runs for speed; traces are attribution.',
                           'Runtime API durations contain asynchronous queueing and overlap.',
                           'Source/engine activity unions overlap and are not additive wall-time shares.',
                           'Missing dynamic DMA metadata prevents exact traffic or bandwidth claims.',
                           'Sampled SBUF utilization is not a complete allocator/live-memory proof.',
                           'BF16-cost multipliers follow the installed Explorer schema, not assumed FLOP peaks.'],
                  sha256=hashes)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), matched_dispatches=len(executions),
                          matched_auxiliary_executions=len(auxiliary),
                          analyzed_graphs=len(analyses))), flush=True)


if __name__ == '__main__':
    main()
