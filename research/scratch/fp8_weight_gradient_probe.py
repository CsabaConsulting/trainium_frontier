#!/usr/bin/env python3
"""Gate long-token FP8 weight products and charge backward preparation.

Down gradient transfers the saved hidden inverse scale into incoming G before
one global gradient calibration. Up gradient uses the existing fixed input
scale. This declares a new quantized backward; exact previous-VJP or acceptable
quality is not inferred from implementation accuracy. Saved forward scales are
assumed inputs; their extra forward stores and shared data products are absent.
"""
import argparse
import gc
import json

import torch

from fp8_weight_gradient_nki import build_kernels
from fp8_weight_cache_nki import build_kernels as build_calibration
from fp8_weight_cache_probe import calibration_reference, require_exact, quantization_diagnostic
from fp8_packed_weight_cache_probe import decode_legacy
from fp8_mlp_island_probe import row_reference, pair_gate
from fresh_start_fp8_probe import e4m3_legacy
from mlp_fused_bf16_probe import timing


def run(kind, shape, case, functions, calibrate, benchmark, steps, grouped_io=False,
        stage_timings=False, parallel_calibration=False, balanced_io=False, calibration_mode=0,
        gradient_prepare=None):
    tokens, outputs, inputs = shape
    multiply, quantize, product = functions
    generator = torch.Generator().manual_seed(9231 + outputs + inputs)
    gradient = (torch.randn(tokens, outputs, generator=generator) * .03125).bfloat16()
    activations = torch.randn(tokens, inputs, generator=generator).bfloat16()
    if kind == 'down':
        activations = activations.relu().square()
    if case == 'zero_gradient': gradient.zero_()
    if case == 'zero_activation': activations.zero_()
    if case == 'tiny_gradient': gradient = (gradient.float() * 2.0**-110).bfloat16()
    if case == 'row_scale_range':
        activations = torch.ldexp(activations.float(), torch.linspace(-30, 30, tokens).round().int()[:, None]).bfloat16()
    if case == 'outliers':
        gradient[127, 0] = 2.0**20; gradient[128, -1] = -(2.0**20)
        activations[127, 0] = 2.0**20; activations[128, -1] = 2.0**20
    if case == 'signed_cancellation':
        gradient[128:256] = -gradient[:128]; gradient[256:] = 0
        activations[128:256] = activations[:128]; activations[256:] = 0
    if kind == 'down':
        right_meta, saved = row_reference(activations)
        factors = right_meta[:, 1:2].contiguous()
        rescaled = (gradient.double() * factors.double()).bfloat16()
        right_inverse = 1.0
        product_right_meta = torch.tensor([1.0, 1.0, 1.0]).expand(128, 3).contiguous()
    else:
        scale = 4.0 if inputs == 1024 else 8.0
        right_meta = torch.tensor([scale, 1.0 / scale, float(activations.float().abs().max())]).expand(128, 3).contiguous()
        saved = (e4m3_legacy(activations.float() * scale) / scale).bfloat16()
        rescaled, factors = gradient, None
        right_inverse = 1.0 / scale
        product_right_meta = right_meta
    left_meta, _ = calibration_reference(rescaled)
    expected_left = e4m3_legacy(rescaled.float() * left_meta[0, 0])
    expected_right = e4m3_legacy(saved.float() * (right_meta[:, 0:1] if kind == 'down' else right_meta[0, 0]))
    gdev, hdev = gradient.to('neuron'), saved.to('neuron')
    rmeta_dev = right_meta.to('neuron')
    product_meta_dev = product_right_meta.to('neuron')
    fdev = factors.to('neuron') if factors is not None else None
    gprime = multiply[2](gdev, fdev, 2) if kind == 'down' else gdev
    require_exact(case + ':' + kind + ':scale_transfer_bf16_boundary', gprime, rescaled)
    metadata = calibrate[1](gprime, parallel_calibration, calibration_mode)
    require_exact(case + ':' + kind + ':gradient_metadata', metadata, left_meta)
    if parallel_calibration or calibration_mode:
        require_exact(case + ':' + kind + ':candidate_vs_original_calibration',
                      metadata, calibrate[1](gprime, False))
    left_bytes = quantize[2](gprime, metadata, 2, False)
    prepare_factors = fdev if kind == 'down' else rmeta_dev

    def prepare_gradient():
        partial, finish = gradient_prepare
        parts = partial[2](gdev, prepare_factors, 2, kind == 'down')
        return finish[2](gdev, prepare_factors, parts, 2, kind == 'down')

    if gradient_prepare is not None:
        prepared_bytes, prepared_metadata = prepare_gradient()
        require_exact(case + ':' + kind + ':two_core_exact_original_metadata', prepared_metadata, metadata)
        require_exact(case + ':' + kind + ':two_core_exact_original_bytes', prepared_bytes, left_bytes)
        left_bytes, metadata = prepared_bytes, prepared_metadata
    right_bytes = quantize[2](hdev, rmeta_dev, 2, kind == 'down')
    raw_left, raw_right = decode_legacy(left_bytes.cpu()), decode_legacy(right_bytes.cpu())
    require_exact(case + ':' + kind + ':gradient_quantizer_decoded', raw_left, expected_left)
    require_exact(case + ':' + kind + ':saved_activation_quantizer_decoded', raw_right, expected_right)
    actual = product[2](left_bytes, right_bytes, metadata, product_meta_dev,
                        2, grouped_io, balanced_io).cpu()
    assert actual.dtype == torch.float32
    inverse_rows = torch.full((outputs, 1), float(left_meta[0, 1]))
    pair_gate(case + ':' + kind + ':weight_product', actual, raw_left.T, raw_right,
              inverse_rows, right_inverse, full=tokens <= 512)
    repeated = product[2](left_bytes, right_bytes, metadata, product_meta_dev,
                          2, grouped_io, balanced_io)
    require_exact(case + ':' + kind + ':repeat', repeated, actual)
    if grouped_io:
        first_layout = product[2](left_bytes, right_bytes, metadata, product_meta_dev, 2, False)
        require_exact(case + ':' + kind + ':grouped_vs_original_product', actual, first_layout)
    require_exact(case + ':' + kind + ':gradient_unchanged', gdev, gradient)
    require_exact(case + ':' + kind + ':saved_activation_unchanged', hdev, saved)
    require_exact(case + ':' + kind + ':left_bytes_unchanged', decode_legacy(left_bytes.cpu()), raw_left)
    require_exact(case + ':' + kind + ':right_bytes_unchanged', decode_legacy(right_bytes.cpu()), raw_right)
    # Bound the CPU work for real shapes; the hardware pair gate samples
    # independently chosen full-token contractions above.
    oi = torch.unique(torch.cat((torch.linspace(0, outputs - 1, min(32, outputs)).long(), torch.tensor([127, 128]))))
    ii = torch.unique(torch.cat((torch.linspace(0, inputs - 1, min(64, inputs)).long(), torch.tensor([127, 128]))))
    original = gradient[:, oi].double().T @ saved[:, ii].double()
    quantization_diagnostic(case + ':' + kind + ':original_bf16_surrogate_fp64', actual[oi][:, ii], original)
    record = dict(kind=kind, case=case, shape=list(shape), passed=True, grouped_io=grouped_io,
        parallel_calibration=parallel_calibration, balanced_io=balanced_io, calibration_mode=calibration_mode,
        two_core_gradient_prepare=gradient_prepare is not None,
        fp32_master_output=True, exact_previous_bf16_vjp=False,
        scale_transfer_underflow_count=int(((gradient != 0) & (rescaled == 0)).sum()),
        gradient_quantizer_zeroed_nonzero_count=int(((rescaled != 0) & (raw_left == 0)).sum()),
        gradient_elements=rescaled.numel(), saved_forward_scales_assumed=True,
        shared_data_gradient_products_or_autograd=False, quality_gate=False)
    print('FP8_WEIGHT_GRADIENT_GATE ' + json.dumps(record), flush=True)
    if benchmark:
        native = torch.compile(lambda gg, hh: (gg.T @ hh).float(),
                               backend='neuron', fullgraph=True, dynamic=False)

        def native_call():
            return native(gdev, hdev)

        def candidate_call():
            # Charge current-gradient scale transfer, calibration, both byte
            # preparations, long reduction and FP32 master stores. No hidden
            # assumption that forward already returns our byte representation.
            if gradient_prepare is not None:
                a, meta = prepare_gradient()
            else:
                current = multiply[2](gdev, fdev, 2) if kind == 'down' else gdev
                meta = calibrate[1](current, parallel_calibration, calibration_mode)
                a = quantize[2](current, meta, 2, False)
            b = quantize[2](hdev, rmeta_dev, 2, kind == 'down')
            return product[2](a, b, meta, product_meta_dev, 2, grouped_io, balanced_io)

        native_result = native_call().cpu()
        assert native_result.dtype == torch.float32 and bool(torch.isfinite(native_result).all())
        quantization_diagnostic(case + ':' + kind + ':native_vs_fp64_master', native_result[oi][:, ii], original)
        require_exact(case + ':' + kind + ':composed_vs_gated_stages', candidate_call(), actual)
        measured = timing({'native_bf16_weight_product': native_call,
                           'fp8_scale_calibrate_pack_weight_product': candidate_call},
                          steps, torch.neuron.synchronize)
        print('FP8_WEIGHT_GRADIENT_BENCH ' + json.dumps(dict(
            kind=kind, shape=list(shape), grouped_io=grouped_io, **measured,
            parallel_calibration=parallel_calibration, balanced_io=balanced_io, calibration_mode=calibration_mode,
            two_core_gradient_prepare=gradient_prepare is not None,
            current_gradient_scale_and_both_byte_preparations_charged=True,
            saved_forward_scales_assumed=True, extra_forward_scale_stores_not_included=True,
            data_gradients_activation_derivative_optimizer_or_whole_model_included=False)), flush=True)
        if stage_timings:
            stages = {
                'gradient_calibration': lambda: calibrate[1](gprime, parallel_calibration, calibration_mode),
                'gradient_bytes': lambda: quantize[2](gprime, metadata, 2, False),
                'activation_bytes': lambda: quantize[2](hdev, rmeta_dev, 2, kind == 'down'),
                'fp8_product_prepared_operands': lambda: product[2](
                    left_bytes, right_bytes, metadata, product_meta_dev, 2, grouped_io, balanced_io),
                'native_bf16_product': native_call,
            }
            if gradient_prepare is not None:
                del stages['gradient_calibration'], stages['gradient_bytes']
                stages['two_core_gradient_preparation'] = prepare_gradient
            elif kind == 'down':
                stages['gradient_row_scale_transfer'] = lambda: multiply[2](gdev, fdev, 2)
            separated = timing(stages, steps, torch.neuron.synchronize)
            print('FP8_WEIGHT_GRADIENT_STAGE_TIMINGS ' + json.dumps(dict(
                kind=kind, shape=list(shape), grouped_io=grouped_io, **separated,
                parallel_calibration=parallel_calibration, balanced_io=balanced_io, calibration_mode=calibration_mode,
                two_core_gradient_prepare=gradient_prepare is not None,
                diagnostic_only=True, prepared_inputs_exclude_producer_cost=True,
                isolated_times_not_additive=True, full_component_above_is_speed_authority=True)), flush=True)
    del actual, repeated, gprime, left_bytes, right_bytes, gdev, hdev
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--grouped-io', action='store_true')
    parser.add_argument('--stage-timings', action='store_true')
    parser.add_argument('--parallel-calibration', action='store_true')
    parser.add_argument('--balanced-io', action='store_true')
    parser.add_argument('--calibration-mode', type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument('--two-core-gradient-prepare', action='store_true')
    args = parser.parse_args()
    assert not args.benchmark or args.real_shape
    assert not args.stage_timings or args.benchmark
    assert not args.balanced_io or args.grouped_io
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    functions = tuple(wrap_nki(k) for k in build_kernels())
    calibrate = wrap_nki(build_calibration()[0])
    gradient_prepare = None
    if args.two_core_gradient_prepare:
        from fp8_gradient_prepare_nki import build_kernels as build_prepare
        gradient_prepare = tuple(wrap_nki(k) for k in build_prepare())
    print('FP8_WEIGHT_GRADIENT_CONTRACT ' + json.dumps(dict(
        current_gradient_scale='global, online exponent-bit calibration; clamp amax at2^-100',
        transferred_gradient='BF16(G*saved_inverse_hidden_scale), then legacy E4M3',
        product='FP32 long-token reduction under empirical11-bit pair model; sequential inverse scales',
        finite_bf16_rescaled_gradient_amax_max=2.0**100,
        signed_zero_encodings_equivalent=True, numerical_bounds_unchanged_from_pair_reference=True,
        new_quantized_backward_algorithm=True, exact_previous_vjp_or_quality_claim=False)), flush=True)
    count = 0
    for case in ('random', 'zero_gradient', 'zero_activation', 'tiny_gradient',
                 'row_scale_range', 'outliers', 'signed_cancellation'):
        run('down', (512, 256, 512), case, functions, calibrate, False, args.steps,
            args.grouped_io, False, args.parallel_calibration, args.balanced_io, args.calibration_mode,
            gradient_prepare)
        count += 1
    run('up', (512, 512, 256), 'random', functions, calibrate, False, args.steps,
        args.grouped_io, False, args.parallel_calibration, args.balanced_io, args.calibration_mode,
        gradient_prepare)
    count += 1
    if args.real_shape:
        for kind, shape in [('down', (16384, 1024, 4096)), ('up', (16384, 4096, 1024))]:
            run(kind, shape, 'real_random', functions, calibrate, args.benchmark,
                args.steps, args.grouped_io, args.stage_timings,
                args.parallel_calibration, args.balanced_io, args.calibration_mode, gradient_prepare)
            count += 1
    print('FP8_WEIGHT_GRADIENT_RESULT ' + json.dumps(dict(
        passed=True, cases=count, real_shape=args.real_shape, grouped_io=args.grouped_io,
        parallel_calibration=args.parallel_calibration, balanced_io=args.balanced_io, calibration_mode=args.calibration_mode,
        two_core_gradient_prepare=args.two_core_gradient_prepare,
        full_fp8_backward_or_autograd_integration=False, quality_gain_established=False)), flush=True)


if __name__ == '__main__':
    main()
