#!/usr/bin/env python3
"""Run every source175 training gate with an explicit smaller SBUF budget."""
import argparse
import ast
import hashlib
from pathlib import Path
import sys

import nki_library_training_mlp_reuse_transpose_probe as original
from nki_library_probe_common import emit
from nki_library_sbuf_budget import saved_relu_mlp_budgeted


class AddBudget(ast.NodeTransformer):
    def __init__(self, budget_bytes):
        self.budget_bytes = budget_bytes
        self.changed = 0

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'wrapped'):
            assert isinstance(node.func.slice, ast.Constant) and node.func.slice.value == 2
            assert len(node.args) == 3 and not node.keywords
            node.args.append(ast.Constant(value=self.budget_bytes))
            self.changed += 1
        return self.generic_visit(node)


def install(budget_kib):
    assert budget_kib in (64, 128)
    path = Path(original.__file__)
    source = path.read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'run_case')
    unchanged_ast = ast.dump(node)
    change = AddBudget(budget_kib * 1024)
    node = change.visit(node)
    assert change.changed == 1
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(vars(original), saved_relu_mlp_reuse_transpose=saved_relu_mlp_budgeted)
    exec(compile(module, str(path), 'exec'), namespace)
    original.run_case = namespace['run_case']
    for call in ast.walk(node):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Subscript)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == 'wrapped'):
            assert call.args.pop().value == budget_kib * 1024
    assert ast.dump(node) == unchanged_ast
    emit('NKILIB_SBUF_BUDGET', dict(
        original_probe_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        budget_kib_per_partition=budget_kib, explicit_static_kernel_argument=True,
        all_original_fixtures_and_bounds_retained=True,
        allocator_safety_checks_preserved=True, changed_call_sites=change.changed,
        model_scheduling_hypothesis=True, component_speedup_required=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--budget-kib', type=int, choices=(64, 128), required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    install(args.budget_kib)
    original.main()
