"""Exact P2 table source through real NKI backend or official CPU simulator.

No training import, source transformation, device execution, or native claim.
Commands run sequentially under one local owner. Machine outputs are receipts.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from dataclasses import asdict, replace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / 'train.py'
SOURCE_SHA = '414cdd76df85450659375c0928b7f0c882b963088529ca049c19be32814fec34'
NKI_VERSION = '0.5.0+28631259367.ga768afa6'
NCC_VERSION = '2.26.6360.0+6f180f47'


def kernel_from_exact_source():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    tree = ast.parse(raw, filename=str(SOURCE))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_k2_table_build_kernel')
    namespace = {'__name__': 'p2_exact_source_backend_check'}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    kernel = namespace['_k2_table_build_kernel'](enabled=True)
    assert not kernel.func.__code__.co_freevars
    return kernel


def compile_backward(info):
    import nki.language as nl
    from nki.language.tensor import NkiTensor
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    from nki.framework.compiled import CompileKernel
    assert importlib.metadata.version('neuronx-cc') == NCC_VERSION
    shapes = {'gradient': ((32768, 128), nl.bfloat16),
              'counts': ((524288, 1), nl.int32), 'starts': ((524288, 1), nl.int32),
              'positions': ((32768, 1), nl.int32),
              'group_maximums': ((4096, 1), nl.int32)}
    tensors = {name: NkiTensor(shape=shape, dtype=dtype, buffer=nl.shared_hbm,
                              storage=None, name=name)
               for name, (shape, dtype) in shapes.items()}
    artifact = HERE / 'scratch' / ('compile-' + str(time.time_ns()))
    artifact.mkdir(parents=True)
    kernel = kernel_from_exact_source()
    if info['mode'] == 'compile-production':
        # Match the exact @nki.jit callable's production pipeline policy.
        # In this SDK default compile opts DISABLE experimental MLIR allocation
        # and scheduling; the neuronx-cc backend handles them instead.
        kernel = kernel._to_subclass(CompileKernel, target='trn2')
        opts = replace(kernel._compile_opts(), artifacts_dir=str(artifact),
                       output_path=str(artifact / 'tables.neff'), verbose=True,
                       verbose_neuronx_cc=True)
    else:
        # Retain the initial diagnostic's NON-PRODUCTION allocator policy.
        opts = CompileOptions(target='trn2', lnc=2, artifacts_dir=str(artifact),
                              output_path=str(artifact / 'tables.neff'),
                              verbose=True, verbose_neuronx_cc=True,
                              neuronx_cc_args=('--optlevel=1', '--auto-cast', 'matmult',
                                               '--auto-cast-type', 'bf16'))
    info['compile_options'] = asdict(opts)
    info['kernel_enable_backend_opt'] = getattr(kernel, '_enable_backend_opt', None)
    info['production_default_options'] = info['mode'] == 'compile-production'
    info.update(artifacts_dir=str(artifact), output_shape=[524288, 128],
                output_dtype='float32', source_unmodified=True,
                shapes={k: {'shape': list(s), 'dtype': str(d)} for k, (s, d) in shapes.items()})
    info['stage'] = 'PARSER_TO_BIR'
    bir = compile_to_bir(kernel, frontend=ParserFrontend(),
                         inputs=tensors, compile_opts=opts, enable_cache=False)
    info['bir_pass'] = True
    info['stage'] = 'NEURONX_CC_BACKEND'
    result = compile_bir_to_neff(opts, bir, input_arrays=[],
                                 argument_names=[s.name for s in bir.descriptor.input_specs],
                                 output_arg_names=[s.name for s in bir.descriptor.output_specs])
    neff = Path(result.neff_path)
    info.update(status='FULL_BACKWARD_COMPILE_PASS', stage='NEFF_WRITTEN',
                neff_path=str(neff), neff_bytes=neff.stat().st_size,
                neff_sha256=hashlib.sha256(neff.read_bytes()).hexdigest())


def simulate_backward(info):
    import numpy as np
    import ml_dtypes
    import nki
    from dataclasses import replace
    # Exact dimensions. Four collisions per row, stable encounter ordering;
    # nontrivial BF16 signed inputs, exactly representable in FP32 accumulation.
    vocab, tokens, channels = 524288, 32768, 128
    sparse = info.get('fixture') == 'sparse'
    # Both fixtures have four collisions per active row and 516096 unused
    # rows. Sparse spreads two active rows (63 and127) across every128-row
    # group, including the final vocabulary row. Uniform clusters activity
    # in the first8192 rows and exercises4032 fully empty runtime groups.
    ids = np.arange(tokens, dtype=np.int32) % 8192
    if sparse:
        ids = ids * 64 + 63
    positions = np.argsort(ids, kind='stable').astype(np.int32).reshape(tokens, 1)
    counts = np.bincount(ids, minlength=vocab).astype(np.int32).reshape(vocab, 1)
    starts = (np.cumsum(counts, axis=0, dtype=np.int32) - counts).reshape(vocab, 1)
    maximums = counts.reshape(4096, 128).max(axis=1).reshape(4096, 1)
    gradient = np.empty((tokens, channels), dtype=ml_dtypes.bfloat16)
    channel = np.arange(channels, dtype=np.int32)
    for begin in range(0, tokens, 128):
        row = np.arange(begin, begin + 128, dtype=np.int32)[:, None]
        gradient[begin:begin + 128] = (((row * 13 + channel * 7) % 61) - 30) / 16.0
    reference = np.zeros((vocab, channels), dtype=np.float32)
    for position in range(tokens):
        reference[ids[position]] += gradient[position].astype(np.float32)
    info.update(stage='OFFICIAL_NKI_SIMULATOR', shape=[tokens, channels],
                metadata_fixture=('four collisions on two rows per group; all4096 groups active' if sparse
                                  else 'four collisions per first8192 rows;4032 empty groups'),
                untouched_rows=int(np.count_nonzero(counts == 0)),
                maximum_count=int(counts.max()),
                reference='ordered FP32 row accumulation of exact BF16 inputs')
    sim = replace(nki.simulate(kernel_from_exact_source()), target='trn2')
    actual = sim(gradient, counts, starts, positions, maximums)
    actual = np.asarray(actual)
    same = actual.dtype == np.float32 and actual.shape == reference.shape
    max_error = 0.0
    # Chunk comparison avoids extra full268MiB difference/absolute arrays.
    if same:
        for begin in range(0, vocab, 8192):
            left, right = actual[begin:begin+8192], reference[begin:begin+8192]
            same = bool(np.array_equal(left, right)) and same
            max_error = max(max_error, float(np.max(np.abs(left-right))))
    info.update(status='SIMULATOR_NUMERICS_PASS' if same else 'SIMULATOR_NUMERICS_FAIL',
                exact_equal=bool(same), max_abs_error=max_error,
                actual_dtype=str(actual.dtype), actual_shape=list(actual.shape))
    assert same, 'official simulator differs from ordered FP32 reference'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['compile-production', 'simulate'])
    parser.add_argument('--fixture', choices=['uniform', 'sparse'], default='uniform')
    args = parser.parse_args()
    # Keep a memory-limited compiler from exhausting this already busy laptop.
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (900, 920))
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'
    os.environ['TMPDIR'] = str(HERE / 'pip-tmp')
    (HERE / 'pip-tmp').mkdir(exist_ok=True)
    os.environ['NEURON_CC_FLAGS'] = '--optlevel=1 --auto-cast matmult --auto-cast-type bf16'
    os.environ['NEURON_LOGICAL_NC_CONFIG'] = '2'
    started = time.monotonic()
    info = {'source_sha256': SOURCE_SHA, 'mode': args.mode, 'status': 'STARTED',
            'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'fixture': args.fixture,
            'nki_version': importlib.metadata.version('nki'),
            'python': sys.version.split()[0], 'target': 'trn2', 'lnc': 2,
            'address_space_limit_bytes': 2 * 1024**3, 'thread_env_limit': 1,
            'native_execution': False, 'forward_framework_compile': False,
            'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'command': [sys.executable, *sys.argv]}
    assert info['nki_version'] == NKI_VERSION
    try:
        if args.mode.startswith('compile'):
            compile_backward(info)
        else:
            simulate_backward(info)
    except Exception as exc:
        info.update(status='FAIL', error=str(exc), exception_type=type(exc).__name__,
                    traceback=traceback.format_exc())
    info.update(elapsed_s=time.monotonic() - started,
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    receipt = HERE / (args.mode + '-result-' + str(time.time_ns()) + '.json')
    receipt.write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2), flush=True)
    return 0 if info['status'].endswith('_PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
