#!/usr/bin/env python3
"""Retain every source194 gate and compare matched Scalar/Vector saved forwards."""
import ast
import copy
import hashlib
from pathlib import Path

import nki_library_row_training_forward_probe as saved
from nki_library_probe_common import emit
from nki_library_row_vector_square import install as install_vector, row_mlp_vector_square_forward

original_transform = saved.transform_run_case


def transform_run_case(node):
    node = original_transform(node)
    original_ast = ast.dump(node)
    training = next(n for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name == 'training_complete')
    scalar = copy.deepcopy(training)
    scalar.name = 'scalar_training_complete'
    calls = [n for n in ast.walk(scalar) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'training_mlp']
    assert len(calls) == 1 and len(calls[0].args) == 5
    calls[0].args.append(ast.Constant(value=False))
    vector_calls = [n for n in ast.walk(training) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == 'training_mlp']
    assert len(vector_calls) == 1 and len(vector_calls[0].args) == 5
    vector_flag = ast.Constant(value=True)
    vector_calls[0].args.append(vector_flag)
    extra = [scalar, *ast.parse('''
scalar_training_fn = torch.compile(scalar_training_complete, backend='neuron', fullgraph=True, dynamic=False)
''').body]
    at = node.body.index(training) + 1
    node.body[at:at] = extra
    inserted = list(extra)
    at = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
              and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id == 'training_result') + 1
    check = ast.parse('''
exact(kind + ':vector_vs_scalar_all_six_outputs', training_result, scalar_training_fn(*values))
emit('NKILIB_VECTOR_SQUARE_CASE', dict(passed=True, kind=kind, shape=list(shape),
    all_six_outputs_exact_against_scalar_control=True,
    original_source194_checks_and_bounds_retained=True,
    all_weight_preparation_and_saves_charged=True, backward_or_quality_claim=False))
''').body
    node.body[at:at] = check
    inserted.extend(check)
    timings = [n for n in ast.walk(node) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Name) and n.func.id == 'dict'
               and any(k.arg == 'row_fp8_training_forward' for k in n.keywords)]
    assert len(timings) == 1
    scalar_arm = ast.keyword(arg='row_fp8_scalar_training_forward',
                            value=ast.parse('lambda: scalar_training_fn(*values)', mode='eval').body)
    timings[0].keywords.append(scalar_arm)
    full_body = node.body
    full_keywords = timings[0].keywords
    node.body = [n for n in node.body if n not in inserted]
    timings[0].keywords = [k for k in full_keywords if k is not scalar_arm]
    assert vector_calls[0].args.pop() is vector_flag
    assert ast.dump(node) == original_ast
    vector_calls[0].args.append(vector_flag)
    node.body = full_body
    timings[0].keywords = full_keywords
    return node


def install():
    # The source194 driver still builds its real namespace, emits its original
    # identity record and runs every unchanged fixture and bound. Only its
    # constructor and the additional matched control are selected here.
    saved.transform_run_case = transform_run_case
    saved.row_mlp_training_forward = row_mlp_vector_square_forward
    saved.install_training = install_vector
    saved.install()
    emit('NKILIB_VECTOR_SQUARE_REWRITE', dict(
        source194_probe_sha256=hashlib.sha256(Path(saved.__file__).read_bytes()).hexdigest(),
        all_source194_tests_and_original_three_forward_controls_retained=True,
        matched_scalar_and_vector_training_forward_both_charged=True,
        ordinary_training_forward_timing_arm_uses_vector_square=True,
        ten_additional_exact_comparisons=True, backward_or_quality_claim=False))


if __name__ == '__main__':
    install()
    saved.inline.original.main()
