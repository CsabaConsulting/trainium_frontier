"""Real version-matched NKI parser check of the exact candidate, no device call.

Only the existing factory AST is extracted to avoid importing the training
module. The kernel body, constants, source filename and source lines are not
rewritten. Compilation stops at verified MLIR, before BIR/backend/runtime.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'train.py'
SOURCE_SHA = 'f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e'
NKI_VERSION = '0.5.0+28631259367.ga768afa6'


def run_child():
    import nki
    import nki.language as nl
    from nki.compiler.frontend import FrontendKind, source_to_mlir
    from nki.language.tensor import NkiTensor

    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA, 'SOURCE_HASH_MISMATCH'
    assert importlib.metadata.version('nki') == NKI_VERSION, 'NKI_VERSION_MISMATCH'
    tree = ast.parse(raw, filename=str(SOURCE))
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == '_k2_build_kernel')
    namespace = {'__name__': 'exact_candidate_parser_check'}
    exec(compile(ast.Module(body=[factory], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    kernel = namespace['_k2_build_kernel'](enabled=True)
    function = kernel.func
    assert function.__code__.co_freevars == (), 'UNEXPECTED_CLOSURE'
    shapes = {'gradient': ((32768, 1024), nl.bfloat16),
              'counts': ((8192, 1), nl.int32),
              'starts': ((8192, 1), nl.int32),
              'positions': ((32768, 1), nl.int32),
              'group_maximums': ((64, 1), nl.int32)}
    inputs = {name: NkiTensor(shape=shape, dtype=dtype, buffer=nl.shared_hbm,
                             storage=None, name=name)
              for name, (shape, dtype) in shapes.items()}
    info = {'source_sha256': SOURCE_SHA, 'nki_version': NKI_VERSION,
            'python': sys.version.split()[0], 'target': 'trn2', 'lnc': 2,
            'closure': {}, 'exact_unmodified_factory_ast': True,
            'production_geometry': {name: {'shape': list(shape), 'dtype': str(dtype)}
                                    for name, (shape, dtype) in shapes.items()},
            'no_device_execution': True, 'backend_executed': False}
    try:
        result = source_to_mlir(kernel, frontend=FrontendKind.PARSER,
                                target='trn2', lnc=2, **inputs)
        result.verify()
        mlir = str(result).encode()
        info.update(status='PARSER_MLIR_PASS', mlir_verified=True,
                    mlir_bytes=len(mlir), mlir_sha256=hashlib.sha256(mlir).hexdigest())
    except Exception as exc:
        info.update(status='PARSER_FAIL', exception_type=type(exc).__name__, error=str(exc))
    print(json.dumps(info, sort_keys=True))
    return 0 if info['status'] == 'PARSER_MLIR_PASS' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.child:
        return run_child()
    process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--child'],
                             cwd=HERE, capture_output=True, text=True, timeout=60, check=False)
    info = {'scope': 'EXACT_CANDIDATE_REAL_NKI_PARSER_ONLY',
            'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'test_parser_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'command': [sys.executable, '-B', str(Path(__file__).resolve())],
            'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}
    (HERE / 'parser-result.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
