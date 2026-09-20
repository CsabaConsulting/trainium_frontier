#!/usr/bin/env python3
"""Reproduce failed ROW preparation and isolate byte transpose vs producer.

Keep the source181 preparation function AST as the original diagnostic control.
Only two weight transpose expressions change in alternative functions. The
original failure is recorded, never promoted or hidden by a relaxed FP8 decoder.
"""
import ast
import copy
import hashlib
from pathlib import Path

import torch
import torch_neuronx

from fp8_mlp_island_probe import make_inputs
from mlp_fused_bf16_probe import timing
from nki_library_byte_transpose import transpose_bytes
from nki_library_probe_common import emit, exact, verify_sources
from nki_library_row_quantize import quantize_rows
from nki_library_row_quantize_probe import check as quantizer_check


def record(name, actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    bad = actual != expected
    indices = bad.nonzero()[:12]
    examples = [dict(index=idx.tolist(), actual=int(actual[tuple(idx)]),
                     expected=int(expected[tuple(idx)])) for idx in indices]
    info = dict(name=name, shape=list(actual.shape), dtype=str(actual.dtype),
                exact=not bool(bad.any()), mismatches=int(bad.sum()), examples=examples,
                diagnostic_only=True, permits_promotion=False)
    if actual.dtype == torch.uint8:
        info.update(actual_sha256=hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest(),
                    expected_sha256=hashlib.sha256(expected.contiguous().numpy().tobytes()).hexdigest())
    emit('NKILIB_BYTE_TRANSPOSE_DIAGNOSTIC', info)
    return info


def preparation_function(mode, quantizer, transpose, channels, hidden):
    path = Path(__file__).with_name('nki_library_row_mlp_probe.py')
    source = ast.parse(path.read_text())
    run = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'run_case')
    original = next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    node = copy.deepcopy(original)

    class ChangeWeightTranspose(ast.NodeTransformer):
        def __init__(self):
            self.changed = []

        def visit_Call(self, value):
            if (isinstance(value.func, ast.Attribute) and value.func.attr == 'contiguous'
                    and isinstance(value.func.value, ast.Attribute) and value.func.value.attr == 'T'
                    and isinstance(value.func.value.value, ast.Name)
                    and value.func.value.value.id in ('qu', 'qd')):
                operand = value.func.value.value.id
                self.changed.append(operand)
                expressions = dict(original=f'{operand}.T.contiguous()', raw=operand,
                                   float_carrier=f'{operand}.float().T.contiguous().to(torch.uint8)',
                                   nki=f'transpose[2]({operand})')
                return ast.copy_location(ast.parse(expressions[mode], mode='eval').body, value)
            return self.generic_visit(value)

    transform = ChangeWeightTranspose()
    node = transform.visit(node)
    assert transform.changed == ['qu', 'qd']
    node.name = 'prepare_' + mode
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    env = dict(torch=torch, quantizer=quantizer, transpose=transpose, channels=channels, hidden=hidden)
    exec(compile(module, str(path), 'exec'), env)
    emit('NKILIB_PREPARATION_AST', dict(mode=mode, original_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
         transformed_ast_sha256=hashlib.sha256(ast.dump(module).encode()).hexdigest(),
         changed_expressions=transform.changed, original_transpose_retained=mode == 'original'))
    return torch.compile(env[node.name], backend='neuron', fullgraph=True, dynamic=False)


def main():
    torch.set_num_threads(4)
    verify_sources()
    quantizer = torch_neuronx.wrap_nki(quantize_rows)
    transpose = torch_neuronx.wrap_nki(transpose_bytes)
    x, up, down = make_inputs((512, 256, 1024), 'random')
    values = x.to('neuron'), up.float().to('neuron'), down.float().to('neuron')
    # Direct unchanged producer before any transpose supplies checked bytes.
    raw_up, um = quantizer[2](up.to('neuron'), False)
    raw_down, dm = quantizer[2](down.to('neuron'), False)
    quantizer_check('isolated_up', up, (raw_up, um), False)
    quantizer_check('isolated_down', down, (raw_down, dm), False)
    results = {}
    observed_failure = False
    for mode in ('original', 'raw', 'float_carrier', 'nki'):
        torch._dynamo.reset()
        fn = preparation_function(mode, quantizer, transpose, 256, 1024)
        result = tuple(v.detach().cpu() for v in fn(*values))
        results[mode] = result
        for index, raw in ((1, raw_up), (2, raw_down)):
            expected = raw.detach().cpu() if mode == 'raw' else raw.detach().cpu().T.contiguous()
            row = record(f'composed:{mode}:weight{index}', result[index], expected)
            if mode == 'original':
                observed_failure |= not row['exact']
            else:
                assert row['exact'], row
        for index in (0, 3, 4, 5, 6, 7):
            if mode != 'original':
                exact(f'composed:{mode}:unchanged_output{index}', result[index], results['original'][index])
    emit('NKILIB_ORIGINAL_PREPARATION_DIAGNOSTIC', dict(original_failure_reproduced=observed_failure,
         original_source_unchanged=True, all_three_alternative_weight_layouts_exact=True,
         other_six_outputs_unchanged=True, diagnostic_only=True, permits_promotion=False))
    assert observed_failure, 'The original source181 failure must reproduce before claiming its cause'
    known_byte_tests(transpose)
    emit('NKILIB_BYTE_TRANSPOSE_RESULT', dict(passed=True, original_failure_reproduced=True,
         known_byte_cases=5, all_three_alternatives_exact=True, forward_or_training_claim=False))


def known_byte_tests(transpose):
    for rows, columns in ((256, 256), (1024, 256), (256, 1024), (4096, 1024), (1024, 4096)):
        torch._dynamo.reset()
        # Every byte code, with different row/block offsets so neither identity
        # nor a missed contraction tile can accidentally satisfy the reference.
        value = ((torch.arange(rows)[:, None] * 17 + torch.arange(columns)[None, :] * 31) % 256).to(torch.uint8)
        device = value.to('neuron')

        def ordinary(v):
            return v.T.contiguous()

        def float_carrier(v):
            return v.float().T.contiguous().to(torch.uint8)

        def nki(v):
            return transpose[2](v)

        functions = {name: torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
                     for name, fn in dict(ordinary=ordinary, float_carrier=float_carrier, nki=nki).items()}
        for name, fn in functions.items():
            output = fn(device)
            row = record(f'known:{rows}x{columns}:{name}', output, value.T.contiguous())
            if name != 'ordinary':
                assert row['exact'], row
                exact(f'known:{rows}x{columns}:{name}:repeat', fn(device), output)
        measured = timing({name: lambda fn=fn: fn(device) for name, fn in functions.items()},
                          20, torch.neuron.synchronize)
        emit('NKILIB_KNOWN_BYTE_CASE', dict(shape=[rows, columns], passed=True,
             all_256_codes=True, float_carrier_and_nki_exact=True,
             component_timing_only=True, **measured))


if __name__ == '__main__':
    main()
