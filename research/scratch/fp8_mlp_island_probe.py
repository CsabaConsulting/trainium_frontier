#!/usr/bin/env python3
"""Stagewise gates for actual FP8 GEMMs; no forward-only training speed claim."""
import argparse
import gc
import json

import torch

from fp8_mlp_island_nki import build_kernel
from fp8_weight_cache_nki import build_kernels
from fp8_weight_cache_probe import calibration_gate, require_exact, quantization_diagnostic
from fresh_start_fp8_probe import e4m3_legacy, eleven_bit_pair_reference
from mlp_fused_bf16 import reference_forward


def emit(kind, record):
    print(kind + " " + json.dumps(record), flush=True)


def row_reference(values):
    maximum = values.float().abs().amax(dim=1, keepdim=True)
    assert bool(torch.isfinite(maximum).all()) and float(maximum.max()) <= 2.0**100
    _, exponent = torch.frexp(maximum.clamp_min(2.0**-100))
    scale = torch.ldexp(torch.ones_like(maximum), 7 - exponent)
    inverse = torch.ldexp(torch.ones_like(maximum), exponent - 7)
    quantized = (e4m3_legacy(values.float() * scale) * inverse).bfloat16()
    return torch.cat((scale, inverse, maximum), dim=1), quantized


def pair_gate(name, actual, left, right, inverse_rows, inverse_weight, full):
    """Validate corresponding128-block pairs under the measured11-bit model.

    This empirical pair rule was characterized independently on this SDK.
    It is not an AWS guarantee and not ordinary FP32 dot-product accuracy.
    """
    rows, contraction = left.shape
    columns = right.shape[1]
    if full:
        selected_rows = torch.arange(rows)
        selected_columns = torch.arange(columns)
    else:
        selected_rows = torch.unique(torch.cat((torch.linspace(0, rows - 1, 12).long(),
                                                torch.tensor([0, 127, 128, rows - 1]))))
        selected_columns = torch.unique(torch.cat((torch.linspace(0, columns - 1, 128).long(),
                                                   torch.tensor([0, 127, 128, columns - 1]))))
    order = torch.arange(contraction).reshape(-1, 2, 128).transpose(1, 2).reshape(-1)
    a = left[selected_rows][:, order]
    b = right[order][:, selected_columns]
    truth, bound = eleven_bit_pair_reference(a, b, 1.0)
    scale = inverse_rows[selected_rows].double() * float(inverse_weight)
    truth, bound = truth * scale, bound * scale.abs()
    observed = actual[selected_rows][:, selected_columns].double()
    difference = (observed - truth).abs()
    finite = bool(torch.isfinite(observed).all())
    failures = int((difference > bound + 1e-300).sum())
    negative_failures = int(((observed * .5 - truth).abs() > bound + 1e-300).sum())
    nonzero = bool((truth != 0).any())
    record = dict(name=name, sampled_shape=list(observed.shape), full_coordinates=full,
                  finite=finite, bound_failures=failures, max_abs=float(difference.max()),
                  relative_rms=float(difference.square().mean().sqrt()
                                     / truth.square().mean().sqrt().clamp_min(1e-300)),
                  max_error_over_bound=float((difference / bound.clamp_min(1e-300)).max()),
                  half_output_negative_control_failures=negative_failures,
                  contraction_pair_order="corresponding positions in adjacent128-channel blocks",
                  reference="empirical11-significand-bit pair sums, FP32 gamma_(K/2) accumulation",
                  passed=finite and failures == 0 and (not nonzero or negative_failures > 0))
    emit("FP8_ISLAND_PAIR_GATE", record)
    assert record["passed"], record


def run_case(shape, case, kernel, calibrate, quantize):
    tokens, channels, hidden = shape
    tile = min(512, tokens // 2)
    generator = torch.Generator().manual_seed(8107 + channels + tokens)
    x = torch.randn(tokens, channels, generator=generator).bfloat16()
    up = (torch.randn(hidden, channels, generator=generator) / channels**.5).bfloat16()
    down = (torch.randn(channels, hidden, generator=generator) / hidden**.5).bfloat16()
    if case == "zero_input":
        x.zero_()
    elif case == "zero_up":
        up.zero_()
    elif case == "zero_down":
        down.zero_()
    elif case == "outliers":
        x[127, 1] = 64; x[128, 255] = -64
        up[3, 128] = 8; down[128, 255] = -4
    elif case == "changed_weights":
        up = (up.float() + torch.randn(up.shape, generator=generator) * .07).bfloat16()
        down = (down.float() + torch.randn(down.shape, generator=generator) * .04).bfloat16()
    elif case == "token_scale_range":
        exponents = torch.linspace(-30, 30, tokens).round().int()
        x = torch.ldexp(x.float(), exponents[:, None]).bfloat16()
    meta_up, uq = calibration_gate(case + ':up', up, calibrate, quantize)
    meta_down, dq = calibration_gate(case + ':down', down, calibrate, quantize)
    inputs = x.to("neuron"), uq, dq, meta_up, meta_down
    outputs = tuple(value.cpu() for value in kernel[2](*inputs, 2, tile, True))
    y, relu_t, xq, hq, up_stage_t, down_stage, x_meta, h_meta = outputs
    expected_x_meta, expected_xq = row_reference(x)
    require_exact(case + ':per_token_x_metadata', x_meta, expected_x_meta)
    require_exact(case + ':per_token_x_quantizer', xq, expected_xq)
    cpu_up, cpu_down = uq.cpu(), dq.cpu()
    up_meta, down_meta = meta_up.cpu(), meta_down.cpu()
    raw_x = xq.float() * x_meta[:, 0:1]
    raw_up = cpu_up.float() * up_meta[0, 0]
    pair_gate(case + ':up', up_stage_t.T, raw_x, raw_up.T,
              x_meta[:, 1:2], up_meta[0, 1], full=tokens <= 512)
    require_exact(case + ':relu_rounding', relu_t, up_stage_t.bfloat16().relu())
    expected_h_meta, expected_hq = row_reference(relu_t.T.contiguous().square())
    require_exact(case + ':per_token_hidden_metadata', h_meta, expected_h_meta)
    require_exact(case + ':per_token_hidden_quantizer', hq, expected_hq)
    raw_h = hq.float() * h_meta[:, 0:1]
    raw_down = cpu_down.float() * down_meta[0, 0]
    pair_gate(case + ':down', down_stage, raw_h, raw_down.T,
              h_meta[:, 1:2], down_meta[0, 1], full=tokens <= 512)
    require_exact(case + ':output_rounding', y, down_stage.bfloat16())
    normal = tuple(value.cpu() for value in kernel[2](*inputs, 2, tile, False))
    repeat = tuple(value.cpu() for value in kernel[2](*inputs, 2, tile, False))
    for index, (value, expected, again) in enumerate(zip(normal, outputs[:4], repeat)):
        require_exact(case + f':diagnostic_vs_normal{index}', value, expected)
        require_exact(case + f':repeat{index}', again, value)
    require_exact(case + ':input_unchanged', inputs[0], x)
    original, _ = reference_forward(x, up, down)
    quantization_diagnostic(case + ':original_bf16_model', y, original)
    # Changing either side of a128-token boundary must not affect other rows.
    if case in ('random', 'real_random'):
        changed = x.clone(); changed[127] = -changed[127] * 7; changed[128] += 11
        varied = tuple(value.cpu() for value in kernel[2](changed.to('neuron'), *inputs[1:], 2, tile, False))
        keep = torch.ones(tokens, dtype=torch.bool); keep[127:129] = False
        for index, (before, after) in enumerate(zip(normal, varied)):
            if index == 1:
                before, after = before.T, after.T
            require_exact(case + f':row_independence{index}', after[keep], before[keep])
    emit('FP8_ISLAND_CASE', dict(case=case, shape=list(shape), passed=True,
         actual_fp8_up_and_down_matmuls=True, per_token_activation_scales=True,
         backward_gated=False, training_speed_claimed=False))
    del inputs, outputs, normal, repeat
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    args = parser.parse_args()
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    kernel = wrap_nki(build_kernel())
    calibrate, quantize = (wrap_nki(k) for k in build_kernels())
    emit('FP8_ISLAND_CONTRACT', dict(
        pair_significand_bits=11, accumulation_bound='FP32 gamma_(K/2)',
        diagnostic_intermediates=True, backward_gated=False, timing_allowed=False,
        activation_scales='per-token, same-row channels only',
        weight_scales='per-matrix online calibration charged once per update',
        accuracy_gate_is_not_a_quality_gate=True))
    for case in ('random', 'zero_input', 'zero_up', 'zero_down', 'outliers',
                 'changed_weights', 'token_scale_range'):
        run_case((512, 256, 512), case, kernel, calibrate, quantize)
    if args.real_shape:
        run_case((16384, 1024, 4096), 'real_random', kernel, calibrate, quantize)
    emit('FP8_ISLAND_RESULT', dict(passed=True, real_shape=args.real_shape,
         actual_fp8_matmuls=True, backward_gated=False, promoted_into_training=False))


if __name__ == '__main__':
    main()
