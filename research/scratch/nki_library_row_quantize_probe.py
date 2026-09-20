#!/usr/bin/env python3
"""Gate raw FP8 bytes, packed FP32 scales and independent row quantization.

This is a producer prerequisite, not an MLP or training-speed comparison.
FP32 reciprocal error must be <=4 FP32 unit roundoffs relative to independent
FP64 division. This is a predeclared experimental screen, not an AWS accuracy
guarantee. Given that separately checked reciprocal, every decoded quantized
value must equal CPU round-to-nearest-even legacy E4M3, including subnormals.
No acceptance bound is inferred from candidate output errors.
"""
import argparse

import torch
import torch_neuronx

from fresh_start_fp8_probe import e4m3_legacy
from fp8_packed_weight_cache_probe import decode_legacy
from mlp_fused_bf16_probe import timing
from nki_library_probe_common import emit, exact, verify_sources
from nki_library_row_quantize import quantize_rows


def check(name, source, result, packed):
    raw, metadata = (t.detach().cpu() for t in result)
    rows, columns = source.shape
    assert raw.dtype == torch.uint8 and raw.shape == (rows, columns + 4 * packed)
    assert metadata.dtype == torch.float32 and metadata.shape == (rows, 3)
    assert torch.isfinite(metadata).all() and (metadata[:, :2] > 0).all()
    maximum = source.float().abs().amax(1, keepdim=True)
    scale = (maximum * torch.tensor(1 / 240, dtype=torch.float32)).clamp_min(1e-6)
    exact(name + ':absmax', metadata[:, 2:3], maximum)
    exact(name + ':scale', metadata[:, 1:2], scale)
    expected_inverse = 1.0 / scale.double()
    inverse_error = (metadata[:, 0:1].double() / expected_inverse - 1).abs()
    limit = 4 * 2.0**-24
    inverse_passed = bool((inverse_error <= limit).all())
    emit('NKILIB_ROW_RECIPROCAL', dict(name=name, passed=inverse_passed,
         max_relative_error=float(inverse_error.max()), limit=limit,
         reference='independent FP64 division; predeclared screen'))
    assert inverse_passed, name
    decoded = decode_legacy(raw[:, :columns])
    expected = e4m3_legacy(source.float() * metadata[:, 0:1])
    exact(name + ':legacy_decode', decoded, expected)
    if packed:
        # CPU byte reinterpretation independently checks the library's C+4
        # layout; it does not numerically convert scale bytes through FP8.
        scales_from_bytes = raw[:, columns:].contiguous().view(torch.float32)
        exact(name + ':appended_fp32_scale', scales_from_bytes, scale)
    noninvariant = not torch.equal(expected, expected.flip(-1))
    wrong_lane_failures = int((decoded.flip(-1) != expected).sum())
    assert not noninvariant or wrong_lane_failures > 0
    emit('NKILIB_ROW_QUANT_GATE', dict(name=name, passed=True, shape=list(source.shape),
         packed_input=packed, independently_decoded_legacy_e4m3=True,
         scale_exact=True, reciprocal_checked_separately=True,
         wrong_channel_negative_control_failures=wrong_lane_failures,
         signed_zero_encodings_equivalent=True))
    return raw, metadata


def run_case(name, source, packed, steps):
    torch._dynamo.reset()
    kernel = torch_neuronx.wrap_nki(quantize_rows)

    def call(values):
        return kernel[2](values, packed)

    compiled = torch.compile(call, backend='neuron', fullgraph=True, dynamic=False)
    device = source.to('neuron')
    result = compiled(device)
    check(name, source, result, packed)
    exact(name + ':repeat', compiled(device), result)
    exact(name + ':input_unchanged', device, source)
    # A second invocation with changed row1 must not reuse old metadata or
    # bytes, and may not alter any other row. Row1 is on the first core.
    changed = source.clone()
    changed[1] = (changed[1].float() * .5 + .03125).bfloat16()
    changed_result = compiled(changed.to('neuron'))
    check(name + ':changed_row', changed, changed_result, packed)
    keep = torch.arange(source.shape[0]) != 1
    for index in range(2):
        exact(name + f':other_rows:{index}', changed_result[index].cpu()[keep], result[index].cpu()[keep])
    assert not torch.equal(changed_result[1].cpu()[1], result[1].cpu()[1]), name
    measurement = {}
    if name.startswith('random'):
        measurement = timing({'online_row_quantizer': lambda: compiled(device)}, steps,
                             torch.neuron.synchronize)
    emit('NKILIB_ROW_QUANT_CASE', dict(name=name, passed=True, packed_input=packed,
         shape=list(source.shape), changed_input_and_row_independence=True,
         calibration_and_metadata_stores_charged=True, **measurement))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify_sources()
    gen = torch.Generator().manual_seed(179021)
    cases = []
    base = torch.randn(512, 256, generator=gen).bfloat16()
    for name in ('random_small', 'zero', 'tiny', 'outliers', 'row_scale_range'):
        value = base.clone()
        if name == 'zero':
            value.zero_()
        elif name == 'tiny':
            value = (value.float() * 2.0**-30).bfloat16()
        elif name == 'outliers':
            value[127, 128] = 2.0**40
            value[128, 127] = -(2.0**40)
        elif name == 'row_scale_range':
            value = torch.ldexp(value.float(), torch.linspace(-60, 60, 512).round().int()[:, None]).bfloat16()
        for packed in (False, True):
            cases.append((name + ('_packed' if packed else '_weights'), value, packed))
    alphabet = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    valid = torch.isfinite(alphabet) & (alphabet.float().abs() <= 2.0**100)
    alphabet = torch.where(valid, alphabet, torch.zeros_like(alphabet)).reshape(256, 256)
    cases.append(('finite_bf16_alphabet', alphabet, True))
    for rows, columns, packed in ((32768, 1024, True), (4096, 1024, False), (1024, 4096, False)):
        value = (torch.randn(rows, columns, generator=gen) / (1 if packed else columns**.5)).bfloat16()
        cases.append((f'random_{rows}x{columns}', value, packed))
    for name, value, packed in cases:
        emit('NKILIB_ROW_QUANT_START', dict(name=name, shape=list(value.shape), packed_input=packed))
        run_case(name, value, packed, args.steps)
    emit('NKILIB_ROW_QUANT_RESULT', dict(passed=True, cases=len(cases),
         original_and_changed_inputs=True, byte_and_scale_gate=True,
         no_forward_backward_or_model_claim=True))


if __name__ == '__main__':
    main()
