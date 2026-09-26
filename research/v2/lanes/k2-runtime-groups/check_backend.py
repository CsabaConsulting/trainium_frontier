"""Exact K2 source through real NKI backend or official CPU simulator.

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
SOURCE = HERE / 'train.py'
SOURCE_SHA = 'f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e'
NKI_VERSION = '0.5.0+28631259367.ga768afa6'
NCC_VERSION = '2.26.6360.0+6f180f47'


def kernel_from_exact_source():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    tree = ast.parse(raw, filename=str(SOURCE))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_k2_build_kernel')
    namespace = {'__name__': 'k2_exact_source_backend_check'}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    kernel = namespace['_k2_build_kernel'](enabled=True)
    assert not kernel.func.__code__.co_freevars
    return kernel


def compile_backward(info):
    import nki.language as nl
    from nki.language.tensor import NkiTensor
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import compile_bir_to_neff
    from nki.framework.compiled import CompileKernel
    assert importlib.metadata.version('neuronx-cc') == NCC_VERSION
    shapes = {'gradient': ((32768, 1024), nl.bfloat16),
              'counts': ((8192, 1), nl.int32), 'starts': ((8192, 1), nl.int32),
              'positions': ((32768, 1), nl.int32),
              'group_maximums': ((64, 1), nl.int32)}
    tensors = {name: NkiTensor(shape=shape, dtype=dtype, buffer=nl.shared_hbm,
                              storage=None, name=name)
               for name, (shape, dtype) in shapes.items()}
    artifact = HERE / 'scratch' / ('compile-' + str(time.time_ns()))
    artifact.mkdir(parents=True)
    kernel = kernel_from_exact_source()
    # Use authenticated production options, not raw experimental allocator
    # defaults. This is the same policy as the accepted parent backend check.
    kernel = kernel._to_subclass(CompileKernel, target='trn2')
    opts = replace(kernel._compile_opts(), artifacts_dir=str(artifact),
                   output_path=str(artifact / 'k2.neff'), verbose=True,
                   verbose_neuronx_cc=True)
    info['compile_options'] = asdict(opts)
    info['kernel_enable_backend_opt'] = getattr(kernel, '_enable_backend_opt', None)
    info['production_default_options'] = info['mode'] == 'compile-production'
    info.update(artifacts_dir=str(artifact), output_shape=[8192, 1024],
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
    vocab, tokens, channels = 8192, 32768, 1024
    sparse = info.get('fixture') == 'sparse'
    ids = (np.arange(tokens, dtype=np.int32) % (vocab // 2)) * 2 if sparse else np.arange(tokens, dtype=np.int32) % vocab
    positions = np.argsort(ids, kind='stable').astype(np.int32).reshape(tokens, 1)
    counts = np.bincount(ids, minlength=vocab).astype(np.int32).reshape(vocab, 1)
    starts = (np.cumsum(counts, axis=0, dtype=np.int32) - counts).reshape(vocab, 1)
    maximums = counts.reshape(64, 128).max(axis=1).reshape(64, 1)
    gradient = np.empty((tokens, channels), dtype=ml_dtypes.bfloat16)
    channel = np.arange(channels, dtype=np.int32)
    for begin in range(0, tokens, 128):
        row = np.arange(begin, begin + 128, dtype=np.int32)[:, None]
        gradient[begin:begin + 128] = (((row * 13 + channel * 7) % 61) - 30) / 16.0
    reference = np.zeros((vocab, channels), dtype=np.float32)
    for position in range(tokens):
        reference[ids[position]] += gradient[position].astype(np.float32)
    info.update(stage='OFFICIAL_NKI_SIMULATOR', shape=[tokens, channels],
                metadata_fixture=('eight collisions per even row; odd rows untouched' if sparse
                                  else 'stable uniform four collisions per row'),
                untouched_rows=int(np.count_nonzero(counts == 0)),
                maximum_count=int(counts.max()),
                reference='ordered FP32 row accumulation of exact BF16 inputs')
    sim = replace(nki.simulate(kernel_from_exact_source()), target='trn2')
    actual = sim(gradient, counts, starts, positions, maximums)
    actual = np.asarray(actual)
    same = actual.dtype == np.float32 and actual.shape == reference.shape and np.array_equal(actual, reference)
    info.update(status='SIMULATOR_NUMERICS_PASS' if same else 'SIMULATOR_NUMERICS_FAIL',
                exact_equal=bool(same), max_abs_error=float(np.max(np.abs(actual - reference))),
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
    info = {'source_sha256': SOURCE_SHA,
            'check_backend_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'mode': args.mode, 'status': 'STARTED',
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
