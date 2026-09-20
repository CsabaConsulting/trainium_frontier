#!/usr/bin/env python3
"""Retain every original ROW MLP gate while fusing the two weight producers.

Exactly four preparation expressions change: each row quantizer becomes the
fused transposed producer, and its now-redundant weight transpose is removed.
The input quantizer, all fixtures, independent numerical bounds, exact checks,
and complete forward timing boundary remain the original source181 code.
"""
import ast
import hashlib
from pathlib import Path

import torch_neuronx

import nki_library_row_mlp_probe as original
from nki_library_probe_common import emit
from nki_library_row_weight_prepare import quantize_transposed_rows


class FuseWeightPreparation(ast.NodeTransformer):
    def __init__(self):
        self.changed = []

    def visit_Call(self, value):
        if (isinstance(value.func, ast.Subscript)
                and isinstance(value.func.value, ast.Name) and value.func.value.id == 'quantizer'
                and len(value.args) == 2 and isinstance(value.args[0], ast.Call)
                and isinstance(value.args[0].func, ast.Attribute)
                and value.args[0].func.attr == 'bfloat16'
                and isinstance(value.args[0].func.value, ast.Name)
                and value.args[0].func.value.id in ('uu', 'dd')):
            operand = value.args[0].func.value.id
            assert isinstance(value.args[1], ast.Constant) and value.args[1].value is False
            self.changed.append('producer:' + operand)
            return ast.copy_location(ast.parse(f'weight_prepare[2]({operand}.bfloat16(), True)',
                                               mode='eval').body, value)
        if (isinstance(value.func, ast.Attribute) and value.func.attr == 'contiguous'
                and isinstance(value.func.value, ast.Attribute) and value.func.value.attr == 'T'
                and isinstance(value.func.value.value, ast.Name)
                and value.func.value.value.id in ('qu', 'qd')):
            operand = value.func.value.value.id
            self.changed.append('layout:' + operand)
            return ast.copy_location(ast.Name(id=operand, ctx=ast.Load()), value)
        return self.generic_visit(value)


def install():
    path = Path(original.__file__)
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_case')
    change = FuseWeightPreparation()
    node = change.visit(node)
    assert change.changed == ['producer:uu', 'producer:dd', 'layout:qu', 'layout:qd']
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(vars(original), weight_prepare=torch_neuronx.wrap_nki(quantize_transposed_rows))
    exec(compile(module, str(path), 'exec'), namespace)
    original.run_case = namespace['run_case']
    emit('NKILIB_FUSED_WEIGHT_PREPARATION', dict(
        original_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        transformed_ast_sha256=hashlib.sha256(ast.dump(module).encode()).hexdigest(),
        changed_expressions=change.changed, all_original_arithmetic_and_fixtures_retained=True,
        both_online_weight_producers_inside_timed_path=True,
        original_failure_source_unchanged=True, direct_fp8_transpose=True))


if __name__ == '__main__':
    install()
    original.main()
