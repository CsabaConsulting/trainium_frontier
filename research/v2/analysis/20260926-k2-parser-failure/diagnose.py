"""CPU-only parser diagnosis. Never imports train.py or calls a device runtime.

The actual factory is AST-extracted for the original case. Prospective binding
variants exist only in memory under diagnostic virtual filenames. No candidate
file is generated or changed. The parser ends at verified MLIR, before BIR,
neuronx-cc, NEFF generation, and device execution.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import linecache
from pathlib import Path
import subprocess
import sys

import nki
import nki.isa as nisa
import nki.language as nl
from nki.compiler.frontend import FrontendKind, source_to_mlir
from nki.language.tensor import NkiTensor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'research/v2/lanes/k2-native-index-repair/train.py'
SOURCE_SHA = 'a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71'
GEOMETRY = {'tokens': 32768, 'channels': 1024, 'vocab': 8192,
            'groups': 64, 'features': 512, 'partition': 128}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def tensor(name, shape, dtype):
    return NkiTensor(shape=shape, dtype=dtype, buffer=nl.shared_hbm,
                     storage=None, name=name)


def virtual_module(text, label):
    filename = str(HERE / ('virtual_' + label + '.py'))
    # inspect.getsource in the unmodified NKI parser reads these source lines.
    # This cache entry is only in this diagnostic child process.
    linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
    namespace = {'__name__': 'diagnostic_' + label}
    exec(compile(text, filename, 'exec'), namespace)
    return namespace


def production_case(local_count):
    raw = SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA, 'SOURCE_HASH_MISMATCH')
    tree = ast.parse(raw, filename=str(SOURCE))
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_k2_build_kernel')
    module = ast.Module(body=[factory], type_ignores=[])
    if local_count == 0:
        # Exact original AST and filename: source/parser lines match native.
        namespace = {'__name__': 'diagnostic_original'}
        exec(compile(module, str(SOURCE), 'exec'), namespace)
    else:
        inner = next(n for n in factory.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'resident_consumer')
        assignments = [ast.Assign(targets=[ast.Name(id=k, ctx=ast.Store())],
                                  value=ast.Constant(value=v))
                       for k, v in list(GEOMETRY.items())[:local_count]]
        inner.body = assignments + inner.body
        text = ast.unparse(ast.fix_missing_locations(module)) + '\n'
        namespace = virtual_module(text, 'production_local_' + str(local_count))
    kernel = namespace['_k2_build_kernel'](enabled=True)
    inputs = {
        'gradient': tensor('gradient', (32768, 1024), nl.bfloat16),
        'counts': tensor('counts', (8192, 1), nl.int32),
        'starts': tensor('starts', (8192, 1), nl.int32),
        'positions': tensor('positions', (32768, 1), nl.int32),
        'group_maximums': tensor('group_maximums', (64, 1), nl.int32),
    }
    return kernel, inputs, 2


def minimal_case(local):
    assignment = '        tokens = 128\n' if local else ''
    text = '''import nki
import nki.language as nl
import nki.isa as nisa
def factory():
    tokens = 128
    @nki.jit
    def copy_kernel(x):
''' + assignment + '''        assert x.shape == (tokens, 1)
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        tile = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=x)
        nisa.dma_copy(dst=out, src=tile)
        return out
    return copy_kernel
'''
    namespace = virtual_module(text, 'minimal_local' if local else 'minimal_closure')
    return namespace['factory'](), {'x': tensor('x', (128, 1), nl.float32)}, 1


def run_case(case):
    if case == 'original':
        kernel, inputs, lnc = production_case(0)
    elif case.startswith('production_local_'):
        kernel, inputs, lnc = production_case(int(case.rsplit('_', 1)[1]))
    else:
        kernel, inputs, lnc = minimal_case(case == 'minimal_local')
    function = kernel.func
    info = {
        'case': case, 'source_sha256': SOURCE_SHA,
        'nki_version': importlib.metadata.version('nki'),
        'python': sys.version.split()[0], 'target': 'trn2', 'lnc': lnc,
        'closure': {name: cell.cell_contents for name, cell in
                    zip(function.__code__.co_freevars, function.__closure__ or ())},
        'no_device_execution': True,
    }
    try:
        result = source_to_mlir(kernel, frontend=FrontendKind.PARSER,
                                target='trn2', lnc=lnc, **inputs)
        result.verify()
        text = str(result)
        info.update(status='PARSER_MLIR_PASS', mlir_bytes=len(text.encode()),
                    mlir_sha256=hashlib.sha256(text.encode()).hexdigest())
    except Exception as exc:
        info.update(status='PARSER_FAIL', exception_type=type(exc).__name__,
                    error=str(exc))
    print(json.dumps(info, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.case:
        run_case(args.case)
        return
    rows = []
    cases = ['original', 'minimal_closure', 'minimal_local']
    cases += ['production_local_' + str(i) for i in range(1, 7)]
    for case in cases:
        child = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                '--case', case], capture_output=True, text=True,
                               timeout=60, check=False, cwd=HERE)
        row = {'case': case, 'returncode': child.returncode,
               'stdout': child.stdout, 'stderr': child.stderr}
        rows.append(row)
    print(json.dumps({'scope': 'real matching NKI parser; CPU only; no training',
                      'cases': rows}, indent=2))


if __name__ == '__main__':
    main()
