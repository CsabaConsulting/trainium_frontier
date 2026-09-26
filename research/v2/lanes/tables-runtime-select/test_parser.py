"""One exact production-shape NKI parser check, no backend or device claim.

This is intentionally one process and one kernel per invocation. The caller
must own the Supervisor's laptop process slot before executing it.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'train.py'
SOURCE_SHA = '414cdd76df85450659375c0928b7f0c882b963088529ca049c19be32814fec34'
NKI_VERSION = '0.5.0+28631259367.ga768afa6'
GEOMETRIES = {
    'embedding': ('_k2_build_kernel', 8192, 32768, 1024, 64),
    'table': ('_k2_table_build_kernel', 524288, 32768, 128, 4096),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', choices=tuple(GEOMETRIES), required=True)
    args = parser.parse_args()
    started = time.monotonic()
    raw = SOURCE.read_bytes()
    receipt = {
        'scope': 'EXACT_CANDIDATE_REAL_NKI_PARSER_ONLY',
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'test_parser_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'command': [sys.executable, '-B', str(Path(__file__).resolve()),
                    '--kernel', args.kernel],
        'kernel': args.kernel,
        'backend_executed': False,
        'simulator_executed': False,
        'device_executed': False,
        'training_forward_compiled': False,
    }
    try:
        import nki
        import nki.language as nl
        from nki.compiler.frontend import FrontendKind, source_to_mlir
        from nki.language.tensor import NkiTensor

        assert receipt['source_sha256'] == SOURCE_SHA, 'SOURCE_HASH_MISMATCH'
        version = importlib.metadata.version('nki')
        assert version == NKI_VERSION, 'NKI_VERSION_MISMATCH'
        factory_name, vocab, tokens, channels, groups = GEOMETRIES[args.kernel]
        tree = ast.parse(raw, filename=str(SOURCE))
        factory = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == factory_name)
        namespace = {'__name__': 'exact_tables_candidate_parser_check'}
        exec(compile(ast.Module(body=[factory], type_ignores=[]),
                     str(SOURCE), 'exec'), namespace)
        kernel = namespace[factory_name](enabled=True)
        assert kernel.func.__code__.co_freevars == (), 'UNEXPECTED_CLOSURE'
        shapes = {
            'gradient': ((tokens, channels), nl.bfloat16),
            'counts': ((vocab, 1), nl.int32),
            'starts': ((vocab, 1), nl.int32),
            'positions': ((tokens, 1), nl.int32),
            'group_maximums': ((groups, 1), nl.int32),
        }
        inputs = {
            name: NkiTensor(shape=shape, dtype=dtype, buffer=nl.shared_hbm,
                            storage=None, name=name)
            for name, (shape, dtype) in shapes.items()
        }
        receipt.update(
            nki_version=version, python=sys.version.split()[0], target='trn2', lnc=2,
            closure={}, exact_unmodified_factory_ast=True,
            production_geometry={
                name: {'shape': list(shape), 'dtype': str(dtype)}
                for name, (shape, dtype) in shapes.items()
            },
        )
        result = source_to_mlir(kernel, frontend=FrontendKind.PARSER,
                                target='trn2', lnc=2, **inputs)
        result.verify()
        mlir = str(result).encode()
        receipt.update(status='PARSER_MLIR_PASS', mlir_verified=True,
                       mlir_bytes=len(mlir), mlir_sha256=hashlib.sha256(mlir).hexdigest())
    except Exception as exc:
        receipt.update(status='PARSER_FAIL', exception_type=type(exc).__name__,
                       error=str(exc))
    receipt['elapsed_s'] = time.monotonic() - started
    path = HERE / ('parser-result-' + args.kernel + '.json')
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'PARSER_MLIR_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
