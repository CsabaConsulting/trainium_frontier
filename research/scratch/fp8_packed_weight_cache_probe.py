#!/usr/bin/env python3
"""Gate packed FP8 HBM bytes, matrix layout and the paired BF16 backward cache.

An independent CPU decoder checks the legacy format explicitly. Signed zero
encodings are allowed; every nonzero value and all layout positions are exact.
This probe is a prerequisite, not evidence of a complete training speed gain.
"""
import argparse
import gc
import json

import torch

from fp8_packed_weight_cache_nki import build_kernel
from fp8_weight_cache_nki import build_kernels
from fp8_weight_cache_probe import calibration_reference, require_exact
from fresh_start_fp8_probe import e4m3_legacy


def selected_lanes(rows):
    groups = rows // min(512, rows)
    return 2 if groups % 2 == 0 else 1


def decode_legacy(values):
    codes = values.detach().cpu().int()
    exponent, fraction = (codes >> 3) & 15, codes & 7
    assert bool((exponent < 15).all()), 'Packed weights contain legacy Inf/NaN encodings'
    normal = torch.ldexp(1.0 + fraction.float() / 8.0, exponent - 7)
    magnitude = torch.where(exponent == 0, fraction.float() * 2.0**-9, normal)
    return torch.where((codes & 128) != 0, -magnitude, magnitude)


def pack_gate(name, weight, calibrate, quantize, prepare):
    assert weight.device.type == 'cpu' and weight.dtype == torch.bfloat16
    rows, columns = weight.shape
    width, lanes = min(512, rows), selected_lanes(rows)
    expected_meta, expected_qdq = calibration_reference(weight)
    device_weight = weight.to('neuron')
    meta = calibrate[1](device_weight)
    require_exact(name + ':metadata', meta, expected_meta)
    old_cache = quantize[2](device_weight, meta, 2)
    restored, packed = prepare[lanes](device_weight, meta, lanes)
    repeated_qdq, repeated_packed = prepare[lanes](device_weight, meta, lanes)
    assert packed.dtype == torch.uint8
    require_exact(name + ':backward_cache_reference', restored, expected_qdq)
    require_exact(name + ':backward_cache_old_kernel', restored, old_cache)
    require_exact(name + ':backward_cache_repeat', repeated_qdq, restored)
    require_exact(name + ':packed_bytes_repeat', repeated_packed, packed)
    raw = e4m3_legacy(weight.float() * expected_meta[0, 0])
    expected = raw.reshape(rows // width, width, columns // 128, 128).permute(0, 3, 2, 1).contiguous()
    decoded = decode_legacy(packed)
    require_exact(name + ':independent_legacy_decode_and_layout', decoded, expected)
    require_exact(name + ':input_unchanged', device_weight, weight)
    require_exact(name + ':metadata_unchanged', meta, expected_meta)
    # A contraction-lane permutation must be detectable for a non-invariant
    # input. This catches a transpose that looks plausible by shape alone.
    wrong = decoded.flip(1)
    negative_failures = int((wrong != expected).sum())
    expected_noninvariant = not torch.equal(expected, expected.flip(1))
    assert not expected_noninvariant or negative_failures > 0
    record = dict(name=name, shape=list(weight.shape), packed_shape=list(packed.shape),
                  pack_lanes=lanes, dtype='uint8 raw legacy E4M3 bytes',
                  independent_decode_exact=True, bf16_backward_cache_exact=True,
                  wrong_contraction_lane_negative_control_failures=negative_failures,
                  signed_zero_encodings_equivalent=True, passed=True,
                  whole_model_or_speed_claim=False)
    print('FP8_PACKED_CACHE_GATE ' + json.dumps(record), flush=True)
    return meta, restored, packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    args = parser.parse_args()
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    prepare = wrap_nki(build_kernel())
    calibrate, quantize = (wrap_nki(kernel) for kernel in build_kernels())
    generator = torch.Generator().manual_seed(8391)
    cases = []
    for name in ('random', 'zero', 'tiny', 'outliers', 'changed_weights'):
        weight = (torch.randn(512, 256, generator=generator) / 16).bfloat16()
        if name == 'zero': weight.zero_()
        if name == 'tiny': weight = (weight.float() * 2.0**-110).bfloat16()
        if name == 'outliers': weight[127, 128] = 2.0**40; weight[128, 127] = -(2.0**40)
        if name == 'changed_weights': weight = (weight.float() * 7 + .123).bfloat16()
        cases.append((name, weight))
    alphabet = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    valid = torch.isfinite(alphabet) & (alphabet.float().abs() <= 2.0**100)
    alphabet = torch.where(valid, alphabet, torch.zeros_like(alphabet))
    cases.append(('finite_bf16_declared_domain_alphabet', alphabet.reshape(256, 256)))
    if args.real_shape:
        for name, shape in [('real_up', (4096, 1024)), ('real_down', (1024, 4096))]:
            cases.append((name, (torch.randn(shape, generator=generator) / shape[1]**.5).bfloat16()))
    print('FP8_PACKED_CACHE_CONTRACT ' + json.dumps(dict(
        decoded_legacy_values_and_layout='exact', signed_zero_sign_ignored=True,
        finite_bf16_weight_amax_max=2.0**100,
        calibration='existing online exponent-bit scale; no validation data',
        layout='[output group, contraction inner128, contraction block, output within group]',
        full_forward_and_backward_gate_still_required=True)), flush=True)
    for name, weight in cases:
        pack_gate(name, weight, calibrate, quantize, prepare)
        gc.collect()
    print('FP8_PACKED_CACHE_RESULT ' + json.dumps(dict(
        passed=True, cases=len(cases), real_shape=args.real_shape,
        online_preparation_not_free=True, complete_speed_not_measured=True)), flush=True)


if __name__ == '__main__':
    main()
