#!/usr/bin/env python3
"""Run source181's unchanged ROW MLP checks with the gated byte transpose.

Only the two weight-layout expressions change. Keep the original source file,
all arithmetic references, numerical bounds, fixtures and timing boundaries.
The explicit transpose and online producers remain inside the timed forward.
"""
import ast
import hashlib
from pathlib import Path

import torch_neuronx

import nki_library_row_mlp_probe as original
from nki_library_byte_transpose import transpose_bytes
from nki_library_probe_common import emit


def install():
    path = Path(original.__file__)
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_case')

    class ReplaceByteTranspose(ast.NodeTransformer):
        def __init__(self):
            self.changed = []

        def visit_Call(self, value):
            if (isinstance(value.func, ast.Attribute) and value.func.attr == 'contiguous'
                    and isinstance(value.func.value, ast.Attribute) and value.func.value.attr == 'T'
                    and isinstance(value.func.value.value, ast.Name)
                    and value.func.value.value.id in ('qu', 'qd')):
                operand = value.func.value.value.id
                self.changed.append(operand)
                return ast.copy_location(ast.parse(f'byte_transpose[2]({operand})', mode='eval').body, value)
            return self.generic_visit(value)

    change = ReplaceByteTranspose()
    node = change.visit(node)
    assert change.changed == ['qu', 'qd']
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    env = dict(vars(original), byte_transpose=torch_neuronx.wrap_nki(transpose_bytes))
    exec(compile(module, str(path), 'exec'), env)
    original.run_case = env['run_case']
    emit('NKILIB_BYTE_SAFE_PREPARATION', dict(original_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
         transformed_ast_sha256=hashlib.sha256(ast.dump(module).encode()).hexdigest(),
         changed_expressions=change.changed, all_original_arithmetic_and_fixtures_retained=True,
         byte_transposes_inside_timed_path=True, original_failure_source_unchanged=True))


if __name__ == '__main__':
    install()
    original.main()
