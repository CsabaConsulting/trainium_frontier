#!/usr/bin/env python3
"""Exact two-core calibration/quantization and recomputed BF16 scale transfer.

Every partial maximum, final scale and quantized byte is checked before timing.
Timing compares complete gradient preparation against both original calibration
and the faster, separately gated fused abs/max. Activation byte preparation,
matrix products, other gradients and training quality remain outside this probe.
"""
import argparse
import gc
import json
import torch

from fp8_gradient_prepare_nki import build_kernels
from fp8_weight_gradient_nki import build_kernels as gradient_kernels
from fp8_weight_cache_nki import build_kernels as calibration_kernels
from fp8_weight_cache_probe import calibration_reference, require_exact
from fp8_packed_weight_cache_probe import decode_legacy
from fresh_start_fp8_probe import e4m3_legacy
from mlp_fused_bf16_probe import timing


def run(name, shape, transfer, functions, benchmark, steps):
    partial, finish, multiply, quantize, calibrate = functions
    rows, cols = shape
    generator = torch.Generator().manual_seed(94271 + rows + cols)
    value = (torch.randn(rows, cols, generator=generator) * .03125).bfloat16()
    factors = torch.ldexp(torch.ones(rows, 1),
                          torch.linspace(-15, 15, rows).round().int()[:, None])
    if name == 'zero': value.zero_()
    elif name == 'negative': value = -value.abs()
    elif name == 'tiny': value = (value.float() * 2.0**-110).bfloat16(); factors.fill_(1.0)
    elif name == 'subnormal': value.fill_(2.0**-130); factors.fill_(1.0)
    elif name in ('core0_maximum', 'core1_maximum'):
        factors.fill_(1.0)
        value[0 if name == 'core0_maximum' else 128, -1] = -(2.0**20)
    elif name == 'all_declared_codes':
        assert shape == (512, 128)
        codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
        codes[~torch.isfinite(codes) | (codes.float().abs() > 2.0**100)] = 0
        value = codes.reshape(shape)
        factors.fill_(1.0)
    rescaled = (value.double() * factors.double()).bfloat16() if transfer else value
    expected, _ = calibration_reference(rescaled)
    expected_bytes = e4m3_legacy(rescaled.float() * expected[0, 0])
    expected_partial = rescaled.float().abs().reshape(rows // 256, 2, 128, cols).amax(
        dim=(0, 3)).T.contiguous()
    vd, fd = value.to('neuron'), factors.to('neuron')
    existing_value = multiply[2](vd, fd, 2) if transfer else vd
    original_metadata = calibrate[1](existing_value, False, 0)
    fused_metadata = calibrate[1](existing_value, True, 3)
    existing_bytes = quantize[2](existing_value, original_metadata, 2, False)
    parts = partial[2](vd, fd, 2, transfer)
    actual_bytes, actual_metadata = finish[2](vd, fd, parts, 2, transfer)
    tag = name + (':rescaled' if transfer else ':unscaled')
    require_exact(tag + ':original_rescaled_bf16', existing_value, rescaled)
    require_exact(tag + ':original_cpu_metadata', original_metadata, expected)
    require_exact(tag + ':original_fused_metadata', original_metadata, fused_metadata)
    require_exact(tag + ':partial_cpu_maxima', parts, expected_partial)
    require_exact(tag + ':final_cpu_metadata', actual_metadata, expected)
    require_exact(tag + ':exact_original_bytes', actual_bytes, existing_bytes)
    require_exact(tag + ':decoded_cpu_quantizer', decode_legacy(actual_bytes.cpu()), expected_bytes)
    repeated_parts = partial[2](vd, fd, 2, transfer)
    repeated_bytes, repeated_meta = finish[2](vd, fd, repeated_parts, 2, transfer)
    require_exact(tag + ':repeat_partial', repeated_parts, parts)
    require_exact(tag + ':repeat_bytes', repeated_bytes, actual_bytes)
    require_exact(tag + ':repeat_metadata', repeated_meta, actual_metadata)
    require_exact(tag + ':values_unchanged', vd, value)
    require_exact(tag + ':factors_unchanged', fd, factors)
    require_exact(tag + ':partials_unchanged', parts, expected_partial)
    print('FP8_GRADIENT_PREPARE_GATE ' + json.dumps(dict(name=name, shape=list(shape),
          transfer_rows=transfer, passed=True, original_scale_and_byte_contract=True,
          no_current_scale_approximation=True, full_gradient_or_quality_claim=False)), flush=True)
    if benchmark:
        def reference(mode):
            current = multiply[2](vd, fd, 2) if transfer else vd
            meta = calibrate[1](current, mode != 0, mode)
            return quantize[2](current, meta, 2, False), meta

        def candidate():
            intermediate = partial[2](vd, fd, 2, transfer)
            return finish[2](vd, fd, intermediate, 2, transfer)

        measured = timing({'original_scale_calibrate_bytes': lambda: reference(0),
                           'fused_amax_scale_calibrate_bytes': lambda: reference(3),
                           'two_core_recomputed_scale_prepare': candidate},
                          steps, torch.neuron.synchronize)
        print('FP8_GRADIENT_PREPARE_TIMING ' + json.dumps(dict(shape=list(shape),
              transfer_rows=transfer, **measured, both_kernel_calls_charged=True,
              gradient_preparation_only=True, full_gradient_or_quality_claim=False)), flush=True)
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--steps', type=int, default=12)
    args = parser.parse_args()
    assert not args.benchmark or args.real_shape
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    partial, finish = (wrap_nki(k) for k in build_kernels())
    multiply, quantize, _ = (wrap_nki(k) for k in gradient_kernels())
    calibrate = wrap_nki(calibration_kernels()[0])
    functions = (partial, finish, multiply, quantize, calibrate)
    count = 0
    for name in ('random', 'zero', 'negative', 'tiny', 'subnormal',
                 'core0_maximum', 'core1_maximum', 'all_declared_codes'):
        for transfer in (False, True):
            run(name, (512, 128) if name == 'all_declared_codes' else (512, 256),
                transfer, functions, False, args.steps)
            count += 1
    if args.real_shape:
        for cols, transfer in ((1024, True), (4096, False)):
            run('real_random', (16384, cols), transfer, functions, args.benchmark, args.steps)
            count += 1
    print('FP8_GRADIENT_PREPARE_RESULT ' + json.dumps(dict(passed=True, cases=count,
          full_gradient_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
