#!/usr/bin/env python3
"""Separate FP8 storage capacity from FP8 matrix-multiply arithmetic.

The candidate retains E4M3 weights on chip but dequantizes each consumed tile
for BF16 GEMMs. The paired NKI control uses precisely the same quantized BF16
weights. This requires exact agreement, including backward. A native compiled
BF16 MLP and FP64 sampled GEMMs provide additional independent controls.

The explicit quantizer surrogate passes weight gradients to FP32 masters once;
input gradients use the quantized weights. Timings include online calibration,
weight-cache refresh and two forward/backward microbatches per update. They do
not include the optimizer, collective communication or the surrounding model.
"""
import argparse
import gc
import json
import math

import torch

from fp8_weight_cache_nki import build_kernels
from fresh_start_fp8_probe import e4m3_legacy
from mlp_fused_bf16 import build_kernel, reference_forward
from mlp_fused_bf16_probe import compare, gemm_gate, timing


def emit(kind, record):
    print(kind + " " + json.dumps(record), flush=True)


def calibration_reference(weight):
    maximum = float(weight.float().abs().max())
    assert math.isfinite(maximum) and maximum <= 2.0**100
    _, exponent = math.frexp(max(maximum, 2.0**-100))
    scale = math.ldexp(1.0, 7 - exponent)
    inverse = math.ldexp(1.0, exponent - 7)
    metadata = torch.tensor([scale, inverse, maximum]).expand(128, 3).contiguous()
    quantized = (e4m3_legacy(weight.float() * scale) * inverse).bfloat16()
    return metadata, quantized


def require_exact(name, actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    record = dict(name=name, shape=list(actual.shape), dtype=str(actual.dtype),
                  finite=bool(torch.isfinite(actual).all()),
                  exact=bool(torch.equal(actual, expected)),
                  max_abs=float((actual.float() - expected.float()).abs().max()))
    record['passed'] = record['finite'] and record['exact']
    emit('FP8_CACHE_EXACT_GATE', record)
    assert record['passed'], record


def calibration_gate(name, weight, calibrate, quantize):
    expected_metadata, expected = calibration_reference(weight)
    device_weight = weight.to('neuron')
    metadata = calibrate[1](device_weight)
    actual = quantize[2](device_weight, metadata, 2)
    require_exact(name + ':metadata', metadata, expected_metadata)
    require_exact(name + ':quantized', actual, expected)
    require_exact(name + ':metadata_repeat', calibrate[1](device_weight), metadata)
    require_exact(name + ':quantized_repeat', quantize[2](device_weight, metadata, 2), actual)
    require_exact(name + ':input_unchanged', device_weight, weight)
    # Requantizing the dequantized cache must not introduce additional drift.
    require_exact(name + ':idempotent', quantize[2](actual, metadata, 2), actual)
    emit('FP8_CACHE_CALIBRATION', dict(name=name, shape=list(weight.shape),
         maximum=float(expected_metadata[0, 2]), scale=float(expected_metadata[0, 0]),
         scaled_maximum=float(expected_metadata[0, 2].double() * expected_metadata[0, 0].double()),
         online_calibration=True, all_gates_passed=True))
    return metadata, actual


def make_cached_autograd(forward):
    """BF16 backward with declared identity surrogate to FP32 master weights."""
    class CachedMLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, up_master, down_master, up_cache, down_cache, up_meta, down_meta):
            y, relu_t = forward(x, up_cache, down_cache, up_meta, down_meta)
            ctx.save_for_backward(x, up_cache, down_cache, relu_t)
            ctx.up_dtype, ctx.down_dtype = up_master.dtype, down_master.dtype
            return y

        @staticmethod
        def backward(ctx, gradient):
            x, up, down, relu_t = ctx.saved_tensors
            activated_t = relu_t.square()
            dhidden_t = down.T @ gradient.T
            dpre_t = dhidden_t * (relu_t * 2.0)
            dx = dpre_t.T @ up
            dup = (dpre_t @ x).to(ctx.up_dtype)
            ddown = (gradient.T @ activated_t.T).to(ctx.down_dtype)
            return dx, dup, ddown, None, None, None, None
    return CachedMLP.apply


def quantization_diagnostic(name, actual, original):
    actual, original = actual.detach().cpu().double(), original.detach().cpu().double()
    rms = original.square().mean().sqrt().clamp_min(1e-30)
    emit('FP8_CACHE_QUANTIZATION_DIAGNOSTIC', dict(name=name,
         relative_rms=float((actual - original).square().mean().sqrt() / rms),
         max_abs=float((actual - original).abs().max()),
         quality_gate=False, quantized_model_differs_from_original=True))


def mlp_gate(shape, case, kernel, calibrate, quantize, steps, benchmark):
    tokens, channels, hidden = shape
    tile = min(512, tokens // 2)
    generator = torch.Generator().manual_seed(5291 + tokens + channels)
    x = torch.randn(tokens, channels, generator=generator).bfloat16()
    up = torch.randn(hidden, channels, generator=generator) / channels**0.5
    down = torch.randn(channels, hidden, generator=generator) / hidden**0.5
    gradient = (torch.randn(tokens, channels, generator=generator) * .01).bfloat16()
    if case == 'zero':
        down.zero_()
    elif case == 'changed':
        up *= 4.0
        down *= .25
        x *= .5
    up_meta, uq = calibration_gate(case + ':up', up.bfloat16(), calibrate, quantize)
    down_meta, dq = calibration_gate(case + ':down', down.bfloat16(), calibrate, quantize)
    uq_cpu, dq_cpu = uq.cpu(), dq.cpu()
    quantization_diagnostic(case + ':up_weights', uq_cpu, up.bfloat16())
    quantization_diagnostic(case + ':down_weights', dq_cpu, down.bfloat16())

    def native_call(xx, uu, dd, um, dm):
        y, relu = reference_forward(xx, uu, dd)
        return y, relu.T

    def bf16_call(xx, uu, dd, um, dm):
        return kernel[2](xx, uu, dd, 2, tile, True, True, True, True, 0, False, um, dm)

    def fp8_call(xx, uu, dd, um, dm):
        return kernel[2](xx, uu, dd, 2, tile, True, True, True, True, 0, True, um, dm)

    forward = {name: torch.compile(function, backend='neuron', fullgraph=True, dynamic=False)
               for name, function in dict(native=native_call, bf16_cache=bf16_call,
                                          fp8_cache=fp8_call).items()}
    device_x = x.to('neuron')
    values = (device_x, uq, dq, up_meta, down_meta)
    outputs = {name: tuple(a.cpu() for a in function(*values)) for name, function in forward.items()}
    for stage in range(2):
        require_exact(f'{case}:storage_only:stage{stage}', outputs['fp8_cache'][stage],
                      outputs['bf16_cache'][stage])
        compare(f'{case}:independent_native:stage{stage}', outputs['fp8_cache'][stage],
                outputs['native'][stage])
    # Bounded row samples prevent the independent FP64 reference from turning
    # the real-shape device gate into a large CPU GEMM benchmark.
    sample = min(tokens, 128)
    relu_sample = outputs['fp8_cache'][1][:, :sample].T.contiguous()
    gemm_gate(case + ':up_fp64_sample', relu_sample, x[:sample], uq_cpu.T, relu=True)
    gemm_gate(case + ':down_fp64_sample', outputs['fp8_cache'][0][:sample],
              relu_sample.square(), dq_cpu.T)
    for name, function in forward.items():
        repeated = function(*values)
        for stage in range(2):
            require_exact(f'{case}:{name}:repeat{stage}', repeated[stage], outputs[name][stage])
    require_exact(case + ':x_unchanged', device_x, x)
    require_exact(case + ':up_cache_unchanged', uq, uq_cpu)
    require_exact(case + ':down_cache_unchanged', dq, dq_cpu)

    # Independent CPU autograd audits the explicit quantizer surrogate. Every
    # device timing arm uses the same declared custom BF16 backward formulas.
    cpu_leaves = tuple(a.detach().clone().requires_grad_() for a in (x, uq_cpu, dq_cpu))
    cpu_y = reference_forward(*cpu_leaves)[0]
    cpu_y.backward(gradient)
    train = {name: torch.compile(make_cached_autograd(function), backend='neuron',
                                 fullgraph=True, dynamic=False)
             for name, function in dict(native=native_call, bf16_cache=bf16_call,
                                        fp8_cache=fp8_call).items()}
    leaves = tuple(a.to('neuron').detach().requires_grad_() for a in (x, up, down))
    gd = gradient.to('neuron')

    def backward_call(fn, cached_up=uq, cached_down=dq, um=up_meta, dm=down_meta):
        for value in leaves:
            value.grad = None
        result = fn(*leaves, cached_up, cached_down, um, dm)
        result.backward(gd)
        return result

    gradients = {}
    for name, function in train.items():
        result = backward_call(function)
        require_exact(f'{case}:{name}:training_forward', result, outputs[name][0])
        gradients[name] = tuple(a.grad.cpu() for a in leaves)
        assert leaves[1].grad.dtype == leaves[2].grad.dtype == torch.float32
        for i in range(3):
            compare(f'{case}:{name}:cpu_surrogate_gradient{i}', gradients[name][i], cpu_leaves[i].grad)
    for i in range(3):
        require_exact(f'{case}:storage_only:gradient{i}', gradients['fp8_cache'][i], gradients['bf16_cache'][i])
    # Characterize precision change separately; it is not implementation error.
    original_y = forward['native'](device_x, leaves[1].detach().bfloat16(),
                                    leaves[2].detach().bfloat16(), up_meta, down_meta)[0]
    quantization_diagnostic(case + ':forward_model_drift', outputs['fp8_cache'][0], original_y)
    emit('FP8_CACHE_MLP_GATE', dict(shape=shape, case=case, passed=True,
         fp8_matmuls=False, fp32_master_gradient_types=True,
         straight_through_weight_quantizer=True, exact_storage_control=True,
         original_model_equivalence_claimed=False))

    if benchmark:
        def update(function, quantized):
            for value in leaves:
                value.grad = None
            # Exactly once per two microbatches, including master casts and
            # all amax/exponent-bit calibration, conversion and HBM traffic.
            ub, db = leaves[1].detach().bfloat16(), leaves[2].detach().bfloat16()
            um, dm = up_meta, down_meta
            if quantized:
                um, dm = calibrate[1](ub), calibrate[1](db)
                ub, db = quantize[2](ub, um, 2), quantize[2](db, dm, 2)
            for _ in range(2):
                result = function(*leaves, ub, db, um, dm)
                result.backward(gd)
            return result

        arms = dict(native_bf16_original=lambda: update(train['native'], False),
                    native_bf16_quantized=lambda: update(train['native'], True),
                    nki_bf16_cache_quantized=lambda: update(train['bf16_cache'], True),
                    nki_fp8_cache_bf16_math=lambda: update(train['fp8_cache'], True))
        measured = timing(arms, steps, torch.neuron.synchronize)
        emit('FP8_CACHE_BENCH', dict(shape=shape, **measured, microbatches_per_update=2,
             units='ms per cache refresh plus two forward/backward microbatches',
             online_calibration_and_master_casts_included=True,
             gradient_accumulation_included=True, fp8_matmuls=False,
             optimizer_collectives_and_surrounding_model_included=False))
    del outputs, gradients, forward, train, leaves, cpu_leaves
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--calibration-only', action='store_true')
    args = parser.parse_args()
    import torch_neuronx
    torch.set_num_threads(4)
    calibrate, quantize = (torch_neuronx.wrap_nki(k) for k in build_kernels())
    generator = torch.Generator().manual_seed(5151)
    base = torch.randn(256, 256, generator=generator).bfloat16()
    cases = dict(random=base * .03, zero=torch.zeros_like(base), tiny=base * 2.0**-110)
    for name, maximum in (('upper_boundary', 2.0**100), ('lower_boundary', 2.0**-100),
                          ('outlier', 128.0)):
        value = base * (maximum / 1024)
        value[129, 137] = -maximum
        cases[name] = value
    codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    codes[~torch.isfinite(codes) | (codes.float().abs() > 2.0**100)] = 0
    cases['all_declared_finite_bf16_codes'] = codes.reshape(256, 256)
    with torch.no_grad():
        for name, value in cases.items():
            calibration_gate(name, value, calibrate, quantize)
    if not args.calibration_only:
        kernel = torch_neuronx.wrap_nki(build_kernel())
        for case in ('random', 'changed', 'zero'):
            mlp_gate((256, 256, 512), case, kernel, calibrate, quantize, args.steps, False)
        if args.real_shape:
            mlp_gate((16384, 1024, 4096), 'real_random', kernel, calibrate, quantize, args.steps, True)
    emit('FP8_CACHE_RESULT', dict(passed=True, calibration_only=args.calibration_only,
         real_shape=args.real_shape, promoted_into_training=False, fp8_matmuls=False))


if __name__ == '__main__':
    main()
