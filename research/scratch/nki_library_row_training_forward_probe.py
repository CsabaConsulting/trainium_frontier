#!/usr/bin/env python3
"""Keep every source191 gate and compare all training saves to actual stages."""
import ast
import copy
import hashlib
from pathlib import Path

import torch_neuronx

import nki_library_row_fused_input_probe as inline
from nki_library_probe_common import emit
from nki_library_training_mlp_hxt_contiguous_probe import reference_forward as native_saved_forward
from nki_library_row_training_forward import install as install_training, row_mlp_training_forward


def transform_run_case(node):
    node = inline.transform_run_case(node)
    original_ast = ast.dump(node)
    complete = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'complete')
    training = copy.deepcopy(complete)
    training.name = 'training_complete'
    calls = [n for n in ast.walk(training) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'inline_mlp']
    assert len(calls) == 1 and isinstance(calls[0].args[-1], ast.Constant)
    assert calls[0].args[-1].value is False
    calls[0].func.value.id = 'training_mlp'
    calls[0].args.pop()
    inserted = [training, *ast.parse('''
training_fn = torch.compile(training_complete, backend='neuron', fullgraph=True, dynamic=False)
native_saved_fn = torch.compile(native_saved_forward, backend='neuron', fullgraph=True, dynamic=False)
''').body]
    at = node.body.index(complete) + 1
    node.body[at:at] = inserted

    def after_assignment(name, source):
        found = [i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                 and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == name]
        assert len(found) == 1, name
        extra = ast.parse(source).body
        node.body[found[0] + 1:found[0] + 1] = extra
        inserted.extend(extra)

    after_assignment('inline_result', '''
training_expected = (inline_result[0], inline_result[2], inline_result[5],
                     inline_result[6][:, 1:2], inline_result[3], inline_result[4])
training_result = tuple(v.detach().cpu() for v in training_fn(*values))
exact(kind + ':training_all_actual_forward_saves', training_result, training_expected)
exact(kind + ':training_all_saved_repeat', training_fn(*values), training_result)
exact(kind + ':training_input_unchanged', values[0], x)
native_saved_result = native_saved_fn(*values)
assert all(bool(torch.isfinite(v).all()) for v in native_saved_result)
assert native_saved_result[1].shape == (hidden, tokens)
''')
    after_assignment('changed_inline', '''
changed_training = tuple(v.detach().cpu() for v in training_fn(changed.to('neuron'), *values[1:]))
changed_expected = (changed_inline[0], changed_inline[2], changed_inline[5],
                    changed_inline[6][:, 1:2], changed_inline[3], changed_inline[4])
exact(kind + ':training_all_changed_saves', changed_training, changed_expected)
unchanged_rows = torch.arange(tokens) != 1
exact(kind + ':training_all_saved_row_independence',
      tuple(v[unchanged_rows] for v in changed_training),
      tuple(v[unchanged_rows] for v in training_result))
saved_bytes = sum(v.numel() * v.element_size() for v in training_result[1:])
assert saved_bytes == tokens * (3 * hidden + channels + 8)
emit('NKILIB_ROW_TRAINING_SAVE_CASE', dict(passed=True, kind=kind, shape=list(shape),
    exact_all_six_actual_forward_outputs=True, original_source191_gates_retained=True,
    repeat_changed_input_and_all_saved_row_independence=True,
    native_control_saved_relu_layout='HxT', saved_tensor_bytes=saved_bytes,
    actual_input_and_hidden_bytes_scales_and_relu_saved=True,
    all_online_master_casts_weight_preparation_and_saves_charged=True,
    raw_up_diagnostic_store_excluded=True, backward_implemented=False,
    full_training_or_bpb_claim=False))
''')
    timings = [n for n in ast.walk(node) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Name) and n.func.id == 'dict'
               and any(k.arg == 'row_fp8_prepared_control' for k in n.keywords)]
    assert len(timings) == 1
    additions = [ast.keyword(arg=name, value=ast.parse(value, mode='eval').body)
                 for name, value in (
                     ('native_bf16_training_forward', 'lambda: native_saved_fn(*values)'),
                     ('row_fp8_training_forward', 'lambda: training_fn(*values)'))]
    timings[0].keywords.extend(additions)
    full_body = node.body
    full_keywords = timings[0].keywords
    node.body = [n for n in node.body if n not in inserted]
    timings[0].keywords = [n for n in full_keywords if n not in additions]
    assert ast.dump(node) == original_ast
    node.body = full_body
    timings[0].keywords = full_keywords
    return node


def install():
    path = Path(inline.original.__file__)
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'run_case')
    module = ast.fix_missing_locations(ast.Module(body=[transform_run_case(node)], type_ignores=[]))
    namespace = dict(vars(inline.original),
        weight_prepare=torch_neuronx.wrap_nki(inline.quantize_transposed_rows),
        inline_mlp=torch_neuronx.wrap_nki(inline.row_mlp_bf16_input),
        training_mlp=torch_neuronx.wrap_nki(row_mlp_training_forward),
        native_saved_forward=native_saved_forward)
    exec(compile(module, str(path), 'exec'), namespace)
    inline.original.run_case = namespace['run_case']
    inline.original.install = install_training
    emit('NKILIB_ROW_TRAINING_SAVE_REWRITE', dict(
        original_probe_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source191_probe_sha256=hashlib.sha256(Path(inline.__file__).read_bytes()).hexdigest(),
        transformed_ast_sha256=hashlib.sha256(ast.dump(module).encode()).hexdigest(),
        all_source191_numerical_checks_fixtures_and_bounds_retained=True,
        original_three_timing_controls_retained=True,
        native_hxt_and_row_fp8_training_saves_both_charged=True,
        backward_implemented=False, training_or_bpb_claim=False))


if __name__ == '__main__':
    install()
    inline.original.main()
