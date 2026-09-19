#!/usr/bin/env python3
"""Inspect operand strings and timing of shared Linear forward MATMUL records."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from caller_linear_opcode_analysis import digest


def main():
    pa.set_cpu_count(4)
    root = Path(__file__).resolve().parents[2]
    prior_path = root / 'research/run-logs/fs-w140-shared-linear-opcode-attribution-r2.json'
    prior = json.loads(prior_path.read_text())
    result = dict(source_sha256=digest(Path(__file__)),
                  opcode_analysis_sha256=digest(prior_path), arms={},
                  caveats=prior['caveats'] + [
                      'Operand geometry strings are observations, not an inferred layer identity.',
                      'Counts/durations can be incomplete; time bins are not wall-time shares.'])
    for name, arm in prior['arms'].items():
        stem = 'native' if name == 'native' else 'full-fp8'
        paths = list((root / ('out/fresh-start-profiles/fs-w140-' + stem
                             + '-pack32-caller-remote-ingest-r1-parquet')).rglob('Instruction.parquet'))
        assert len(paths) == 1 and digest(paths[0]) == arm['instruction_sha256']
        execution = arm['graphs'][0]['execution']
        lo, hi = execution['execution_start_ts'], execution['execution_end_ts']
        table = pq.read_table(paths[0], columns=['nki_source_location', 'start_ts', 'end_ts',
            'raw_flops', 'adjusted_flops', 'operands'], filters=[('start_ts', '>=', lo),
            ('start_ts', '<', hi), ('pcore_idx', '=', 0), ('opcode', '=', 'MATMUL'),
            ('tensor_instruction_type', '=', 'REGULAR')])
        table = table.filter(pc.fill_null(pc.ends_with(table['nki_source_location'], '/train.py:2853'), False))
        groups = {}
        for row in table.to_pylist():
            key = str((row['raw_flops'], row['adjusted_flops']))
            group = groups.setdefault(key, dict(count=0, start_min_ns=hi, end_max_ns=lo,
                duration_sum_ns=0, operand_patterns=Counter(), bins_10ms=Counter(), examples=[]))
            group['count'] += 1
            group['start_min_ns'] = min(group['start_min_ns'], row['start_ts'])
            group['end_max_ns'] = max(group['end_max_ns'], row['end_ts'])
            group['duration_sum_ns'] += row['end_ts'] - row['start_ts']
            group['bins_10ms'][str((row['start_ts'] - lo) // 10_000_000)] += 1
            operands = row['operands'] or '<missing>'
            # Keep dtype, stride/extent and array geometry; remove addresses and
            # accumulation flags so identical tiling is not split by placement.
            pattern = re.sub(r'0x[0-9a-fA-F]+', '<address>', operands)
            pattern = re.sub(r'acc_flags=\d+', 'acc_flags=<value>', pattern)
            group['operand_patterns'][pattern] += 1
            if len(group['examples']) < 2:
                group['examples'].append(row)
        result['arms'][name] = dict(execution=execution, instruction_sha256=arm['instruction_sha256'],
                                   groups=groups, warnings=arm['warnings'])
    output = root / 'research/run-logs/fs-w140-shared-linear-forward-tiles-r1.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), arms=list(result['arms']))), flush=True)


if __name__ == '__main__':
    main()
