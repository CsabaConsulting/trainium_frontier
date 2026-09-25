#!/usr/bin/env python3
"""Executable stdlib tests: scalar arithmetic and static contract, NOT NKI proof."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tracemalloc

from consumer_nki import build_kernel
from reference import C, P, T, V, CASE_NAMES, consumer, dense, fixture, grouped_wrong, raw, require

HERE = Path(__file__).resolve().parent


def main():
    try:
        build_kernel()
    except RuntimeError as error:
        require(str(error).startswith('EXPERIMENT_DISABLED'), 'default OFF')
    else:
        raise RuntimeError('native factory must refuse by default')
    require('nki' not in sys.modules and 'torch' not in sys.modules, 'OFF imported native/tensor module')
    tracemalloc.start()
    results = []
    for name in CASE_NAMES:
        for dtype in ('bf16', 'fp32'):
            ids, gradients, meta = fixture(name, dtype)
            expected = dense(ids, gradients)
            actual, trace = consumer(ids, gradients, meta)
            require(raw(actual) == raw(expected), name + '/' + dtype + ': byte mismatch')
            require(trace['active_gathers'] == len(ids) and trace['output_rows_stored'] == V, 'visit/store count')
            require(trace['accumulator_hbm_reads'] == 0, 'state HBM reads')
            for row in range(V):
                require([t for r, t in trace['visits'] if r == row] == [t for t, r in enumerate(ids) if r == row], 'ordered recurrence')
            wrong = None
            if name.startswith('cross_tile_'):
                wrong = grouped_wrong(ids, gradients)[3][0]
                require(raw(grouped_wrong(ids, gradients)) != raw(expected), 'retained rejection disappeared')
            results.append(dict(case=name, dtype=dtype, byte_exact=True,
                                sha256=hashlib.sha256(raw(actual)).hexdigest(),
                                group_rounds=trace['group_rounds'], active_gathers=len(ids),
                                inactive_gathers_skipped=trace['inactive_gathers_skipped'],
                                row3=actual[3][0], rejected_grouped_row3=wrong))
            del ids, gradients, meta, expected, actual, trace
    from reference import metadata, validate_metadata
    for invalid in (-1, 256):
        try:
            metadata([invalid])
        except ValueError:
            pass
        else:
            raise RuntimeError('invalid token accepted')
    ids, _, meta = fixture('cross_tile_large')
    for kind in ('order', 'bound', 'offset'):
        bad = [values.copy() for values in meta]
        if kind == 'order':
            bad[2][0], bad[2][1] = bad[2][1], bad[2][0]
        elif kind == 'bound':
            bad[3][0] -= 1
        else:
            bad[1][3] += 1
        try:
            validate_metadata(ids, bad)
        except ValueError:
            pass
        else:
            raise RuntimeError('invalid metadata accepted: ' + kind)
    tree = ast.parse((HERE / 'consumer_nki.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    require(sum(isinstance(n.func, ast.Attribute) and n.func.attr == 'dynamic_range' for n in calls) == 1, 'one dynamic loop body')
    require(not any(isinstance(n.func, ast.Attribute) and n.func.attr in ('nc_matmul', 'tensor_reduce', 'tensor_tensor_scan', 'atomic_rmw') for n in calls), 'gradient reassociation primitive')
    require(not any(isinstance(n.func, ast.Attribute) and n.func.attr in ('cpu', 'numpy', 'item') for n in calls), 'hidden device-to-host')
    for optimized in (False, True):
        command = [sys.executable] + (['-O'] if optimized else []) + ['-B', str(HERE / 'check_native.py')]
        off = subprocess.run(command, text=True, capture_output=True, check=True, timeout=10)
        require(json.loads(off.stdout) == {'status': 'DISABLED', 'native_imported': False, 'native_executed': False}, 'native CLI default OFF')
    peak = tracemalloc.get_traced_memory()[1]
    require(peak < 8 * 1024**2, 'bounded scalar memory')
    print(json.dumps(dict(status='PASS_STDLIB_REFERENCE_ONLY', cases=results, peak_traced_bytes=peak,
                          default_off_refuses_before_native_import=True, production_files_changed=False,
                          torch_imported='torch' in sys.modules, native_executed=False,
                          native_legality='UNPROVEN', native_exactness='UNMEASURED',
                          independent_review='PENDING', launch_authorized=False), indent=2))


if __name__ == '__main__':
    main()
