#!/usr/bin/env python3
"""Gate the explicit NKI byte transpose after source182's float-carrier failure.

The original source182 experiment remains failed. Its framework float-carrier
arm produced exactly the same corrupt bytes as the original transpose, before
the NKI alternative ran. This separate gate keeps the original preparation and
raw producer controls and requires the unchanged NKI alternative to be exact.
No numeric tolerance or decoder rule is changed; no MLP is run here.
"""
import torch
import torch_neuronx

from fp8_mlp_island_probe import make_inputs
from mlp_fused_bf16_probe import timing
from nki_library_byte_transpose import transpose_bytes
from nki_library_byte_transpose_diagnostic import preparation_function, record
from nki_library_probe_common import emit, exact, verify_sources
from nki_library_row_quantize import quantize_rows
from nki_library_row_quantize_probe import check as quantizer_check


def main():
    torch.set_num_threads(4)
    verify_sources()
    quantizer = torch_neuronx.wrap_nki(quantize_rows)
    transpose = torch_neuronx.wrap_nki(transpose_bytes)
    x, up, down = make_inputs((512, 256, 1024), 'random')
    values = x.to('neuron'), up.float().to('neuron'), down.float().to('neuron')
    raw_up, um = quantizer[2](up.to('neuron'), False)
    raw_down, dm = quantizer[2](down.to('neuron'), False)
    quantizer_check('isolated_up', up, (raw_up, um), False)
    quantizer_check('isolated_down', down, (raw_down, dm), False)
    results = {}
    observed_failure = False
    for mode in ('original', 'raw', 'nki'):
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
        if mode != 'original':
            for index in (0, 3, 4, 5, 6, 7):
                exact(f'composed:{mode}:unchanged_output{index}', result[index], results['original'][index])
    assert observed_failure, 'Original source181 failure must reproduce'
    emit('NKILIB_NKI_PREPARATION_DIAGNOSTIC', dict(original_failure_reproduced=True,
         raw_and_nki_weights_exact=True, other_six_outputs_unchanged=True,
         original_source_unchanged=True, failed_float_carrier_not_promoted=True))
    for rows, columns in ((256, 256), (1024, 256), (256, 1024), (4096, 1024), (1024, 4096)):
        torch._dynamo.reset()
        value = ((torch.arange(rows)[:, None] * 17 + torch.arange(columns)[None, :] * 31) % 256).to(torch.uint8)
        device = value.to('neuron')

        def ordinary(v):
            return v.T.contiguous()

        def nki(v):
            return transpose[2](v)

        functions = {name: torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
                     for name, fn in dict(ordinary=ordinary, nki=nki).items()}
        for name, fn in functions.items():
            output = fn(device)
            row = record(f'known:{rows}x{columns}:{name}', output, value.T.contiguous())
            if name == 'nki':
                assert row['exact'], row
                exact(f'known:{rows}x{columns}:nki:repeat', fn(device), output)
        measured = timing({name: lambda fn=fn: fn(device) for name, fn in functions.items()},
                          20, torch.neuron.synchronize)
        emit('NKILIB_KNOWN_BYTE_NKI_CASE', dict(shape=[rows, columns], passed=True,
             all_256_codes=True, nki_exact=True, component_timing_only=True, **measured))
    emit('NKILIB_BYTE_TRANSPOSE_NKI_RESULT', dict(passed=True, original_failure_reproduced=True,
         known_byte_cases=5, raw_and_nki_weights_exact=True, forward_or_training_claim=False))


if __name__ == '__main__':
    main()
