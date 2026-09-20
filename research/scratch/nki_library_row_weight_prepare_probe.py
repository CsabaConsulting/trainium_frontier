#!/usr/bin/env python3
"""Gate fused ROW weight producers against independent arithmetic and exact bytes.

Compare the already gated quantizer plus exact NKI byte transpose with two
fused producers. All arms include the FP32 master cast, scale calculation,
quantization, transpose, and metadata stores. No MLP/backward/BPB claim follows.
"""
import argparse

import torch
import torch_neuronx

from mlp_fused_bf16_probe import timing
from nki_library_byte_transpose import transpose_bytes
from nki_library_probe_common import emit, exact, verify_sources
from nki_library_row_quantize import quantize_rows
from nki_library_row_quantize_probe import check
from nki_library_row_weight_prepare import quantize_transposed_rows


def run(name, source, steps):
    torch._dynamo.reset()
    quantizer, transpose, fused = (torch_neuronx.wrap_nki(kernel) for kernel in
                                 (quantize_rows, transpose_bytes, quantize_transposed_rows))

    def separate(master):
        raw, metadata = quantizer[2](master.bfloat16(), False)
        return transpose[2](raw), metadata

    def direct(master):
        return fused[2](master.bfloat16(), True)

    def carrier(master):
        return fused[2](master.bfloat16(), False)

    arms = {key: torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
            for key, fn in (('separate', separate), ('direct_fp8', direct), ('bf16_carrier', carrier))}
    master = source.float().to('neuron')
    outputs = {key: tuple(v.detach().cpu() for v in fn(master)) for key, fn in arms.items()}
    for key, result in outputs.items():
        check(name + ':' + key, source, (result[0].T.contiguous(), result[1]), False)
        exact(name + ':' + key + ':repeat', arms[key](master), result)
        if key != 'separate':
            exact(name + ':' + key + ':reference_bytes_and_metadata', result, outputs['separate'])
    exact(name + ':master_unchanged', master, source.float())
    changed = source.clone()
    changed[1] = (changed[1].float() * .5 + .03125).bfloat16()
    altered = {key: tuple(v.detach().cpu() for v in fn(changed.float().to('neuron')))
               for key, fn in arms.items()}
    keep = torch.arange(source.shape[0]) != 1
    for key, result in altered.items():
        check(name + ':' + key + ':changed', changed, (result[0].T.contiguous(), result[1]), False)
        exact(name + ':' + key + ':other_rows',
              (result[0][:, keep], result[1][keep]),
              (outputs[key][0][:, keep], outputs[key][1][keep]))
        if key != 'separate':
            exact(name + ':' + key + ':changed_reference', result, altered['separate'])
        assert not torch.equal(result[1][1], outputs[key][1][1])
    measured = {}
    if name.startswith('random'):
        measured = timing({key: lambda fn=fn: fn(master) for key, fn in arms.items()},
                          steps, torch.neuron.synchronize)
    emit('NKILIB_WEIGHT_PREPARE_CASE', dict(name=name, passed=True, shape=list(source.shape),
        independent_quantizer_checks=True, exact_all_bytes_and_metadata=True,
        changed_inputs_and_other_rows_checked=True, original_control_retained=True,
        fp32_master_cast_quantization_transpose_and_metadata_charged=True,
        intermediate_quantized_hbm_tensor_eliminated=True, **measured))
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify_sources()
    generator = torch.Generator().manual_seed(189021)
    base = torch.randn(512, 256, generator=generator).bfloat16()
    cases = []
    for name in ('random_small', 'zero', 'tiny', 'outliers', 'row_scale_range'):
        value = base.clone()
        if name == 'zero':
            value.zero_()
        elif name == 'tiny':
            value = (value.float() * 2.0**-30).bfloat16()
        elif name == 'outliers':
            value[127, 128], value[128, 127] = 2.0**40, -(2.0**40)
        elif name == 'row_scale_range':
            value = torch.ldexp(value.float(), torch.linspace(-60, 60, 512).round().int()[:, None]).bfloat16()
        cases.append((name, value))
    alphabet = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    valid = torch.isfinite(alphabet) & (alphabet.float().abs() <= 2.0**100)
    cases.append(('finite_bf16_alphabet', torch.where(valid, alphabet, torch.zeros_like(alphabet)).reshape(256, 256)))
    for rows, columns in ((4096, 1024), (1024, 4096)):
        value = (torch.randn(rows, columns, generator=generator) / columns ** .5).bfloat16()
        cases.append((f'random_{rows}x{columns}', value))
    production = {}
    for name, source in cases:
        emit('NKILIB_WEIGHT_PREPARE_START', dict(name=name, shape=list(source.shape)))
        measured = run(name, source, args.steps)
        if name in ('random_4096x1024', 'random_1024x4096'):
            production[name] = measured['median_ms']
    emit('NKILIB_WEIGHT_PREPARE_RESULT', dict(passed=True, cases=len(cases),
        direct_fp8_and_bf16_carrier=True, exact_bytes_scales_and_metadata=True,
        independent_quantization_and_reciprocal_gates=True,
        all_preparation_costs_charged=True, no_mlp_backward_or_quality_claim=True))
    assert len(production) == 2
    emit('NKILIB_WEIGHT_PREPARE_ADVANCE', dict(
        warranted=all(row['direct_fp8'] <= .95 * row['separate'] for row in production.values()),
        production_median_ms=production, maximum_ratio=.95,
        next_test='unchanged complete ROW forward gate',
        training_or_quality_promotion=False))


if __name__ == '__main__':
    main()
