#!/usr/bin/env python3
"""Retain the complete ROW screen and audit the fused loader's actual bytes."""
import ast
import copy
import hashlib
from pathlib import Path

import torch_neuronx

import nki_library_row_mlp_probe as original
from nki_library_probe_common import emit
from nki_library_row_mlp_fused_weight_probe import FuseWeightPreparation
from nki_library_row_weight_prepare import quantize_transposed_rows
from nki_library_row_fused_input import install as install_inline, row_mlp_bf16_input


def statements(source):
    return ast.parse(source).body


def transform_run_case(node):
    weights = FuseWeightPreparation()
    node = weights.visit(node)
    assert weights.changed == ['producer:uu', 'producer:dd', 'layout:qu', 'layout:qd']
    fused_weight_ast = ast.dump(node)
    before_complete = next(n for n in node.body
                           if isinstance(n, ast.FunctionDef) and n.name == 'complete')
    old_complete = copy.deepcopy(before_complete)
    prepared_complete = copy.deepcopy(before_complete)
    prepared_complete.name = 'prepared_complete'
    new_complete = statements('''
def complete(xx, uu, dd):
    qu, um = weight_prepare[2](uu.bfloat16(), True)
    qd, dm = weight_prepare[2](dd.bfloat16(), True)
    su = um[:, 1].reshape(1, hidden).expand(128, hidden).contiguous()
    sd = dm[:, 1].reshape(1, channels).expand(128, channels).contiguous()
    return inline_mlp[2](xx, qu, qd, su, sd, False)
''')[0]
    index = node.body.index(before_complete)
    node.body[index] = new_complete
    additions = [prepared_complete, *statements('''
def inline_diagnostic(*values):
    return inline_mlp[2](*values, True)
prepared_complete_fn = torch.compile(prepared_complete, backend='neuron', fullgraph=True, dynamic=False)
inline_debug_fn = torch.compile(inline_diagnostic, backend='neuron', fullgraph=True, dynamic=False)
''')]
    node.body[index + 1:index + 1] = additions
    inserted = list(additions)

    def after_assignment(name, source):
        matches = [i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == name]
        assert len(matches) == 1, name
        extra = statements(source)
        at = matches[0] + 1
        node.body[at:at] = extra
        inserted.extend(extra)

    after_assignment('result', '''
inline_result = tuple(v.detach().cpu() for v in inline_debug_fn(values[0], *prepared[1:5]))
exact(kind + ':inline_original_stages', inline_result[:5], result)
quantizer_check(kind + ':inline_input', x, inline_result[5:], False)
exact(kind + ':inline_input_bytes', inline_result[5], px[:, :channels])
exact(kind + ':inline_input_metadata', inline_result[6], xm)
''')
    after_assignment('plain', '''
exact(kind + ':prepared_control_output', prepared_complete_fn(*values), output)
''')
    after_assignment('other', '''
changed_inline = tuple(v.detach().cpu() for v in inline_debug_fn(changed.to('neuron'), *prepared[1:5]))
quantizer_check(kind + ':inline_changed_input', changed, changed_inline[5:], False)
exact(kind + ':inline_changed_output', changed_inline[0], other)
assert not torch.equal(changed_inline[6][1], inline_result[6][1]), kind
emit('NKILIB_INLINE_INPUT_GATE', dict(passed=True, kind=kind, shape=list(shape),
    actual_input_bytes_and_metadata_checked=True, independent_original_and_changed_input=True,
    exact_all_original_mlp_stages=True, exact_prepared_control_output=True,
    original_fixtures_and_bounds_retained=True, no_input_quantized_hbm_in_plain_path=True))
''')
    timing_dicts = [n for n in ast.walk(node) if isinstance(n, ast.Dict)
                    and any(isinstance(k, ast.Constant) and k.value == 'row_fp8_with_online_preparation'
                            for k in n.keys)]
    # The original uses dict(keyword=...), not a literal dict.
    assert not timing_dicts
    timing_calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'dict'
                    and any(k.arg == 'row_fp8_with_online_preparation' for k in n.keywords)]
    assert len(timing_calls) == 1
    timing_extra = ast.keyword(arg='row_fp8_prepared_control',
        value=ast.parse('lambda: prepared_complete_fn(*values)', mode='eval').body)
    timing_calls[0].keywords.append(timing_extra)

    # Reversing only these additions and the complete() replacement must
    # recover source189's exact AST. Every original bound/check/fixture stays.
    saved_body = node.body
    saved_keywords = timing_calls[0].keywords
    node.body = [old_complete if n is new_complete else n for n in node.body if n not in inserted]
    timing_calls[0].keywords = [n for n in timing_calls[0].keywords if n is not timing_extra]
    assert ast.dump(node) == fused_weight_ast
    node.body = saved_body
    timing_calls[0].keywords = saved_keywords
    return node


def install():
    path = Path(original.__file__)
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'run_case')
    module = ast.fix_missing_locations(ast.Module(body=[transform_run_case(node)], type_ignores=[]))
    namespace = dict(vars(original), weight_prepare=torch_neuronx.wrap_nki(quantize_transposed_rows),
                     inline_mlp=torch_neuronx.wrap_nki(row_mlp_bf16_input))
    exec(compile(module, str(path), 'exec'), namespace)
    original.run_case = namespace['run_case']
    original.install = install_inline
    emit('NKILIB_INLINE_INPUT_REWRITE', dict(
        original_probe_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        transformed_ast_sha256=hashlib.sha256(ast.dump(module).encode()).hexdigest(),
        all_original_numerical_checks_fixtures_and_bounds_retained=True,
        fused_weight_control_retained=True, online_weight_preparation_inside_both_timings=True,
        actual_fused_input_bytes_and_all_original_stages_exposed=True,
        independent_changed_input_check=True, training_or_bpb_claim=False))


if __name__ == '__main__':
    install()
    original.main()
