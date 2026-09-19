#!/usr/bin/env python3
"""Independent first-stage FP8 backward checks and charged incremental timing.

Only G@Wdown (and separately dPre@Wup) is approximated here. Weight-gradient
products, ReLU derivative fusion and autograd integration are not implemented.
Stage references distinguish implementation error from gradient-quantization
error, using the existing independent CPU quantizer and empirical pair model.
"""
import argparse
import gc
import json

import torch

from fp8_gradient_gemm_nki import build_kernel
from fp8_mlp_island_probe import row_reference, pair_gate
from fp8_packed_weight_cache_nki import build_kernel as build_packer
from fp8_packed_weight_cache_probe import pack_gate, selected_lanes
from fp8_weight_cache_nki import build_kernels
from fp8_weight_cache_probe import require_exact, quantization_diagnostic
from mlp_fused_bf16_probe import timing


def run(shape, case, kernel, calibrate, quantize, prepare, steps, benchmark):
    tokens, contraction, outputs = shape
    generator = torch.Generator().manual_seed(9103 + contraction + outputs)
    gradient = (torch.randn(tokens, contraction, generator=generator) * .03125).bfloat16()
    weight = (torch.randn(outputs, contraction, generator=generator) / contraction**.5).bfloat16()
    if case == 'zero_gradient': gradient.zero_()
    if case == 'zero_weight': weight.zero_()
    if case == 'tiny_gradient': gradient = (gradient.float() * 2.0**-110).bfloat16()
    if case == 'row_scale_range':
        gradient = torch.ldexp(gradient.float(), torch.linspace(-80, 30, tokens).round().int()[:, None]).bfloat16()
    if case == 'outliers':
        gradient[127, 0] = 2.0**20; gradient[128, -1] = -(2.0**20)
        weight[0, 127] = 8; weight[-1, 128] = -8
    if case == 'signed_cancellation':
        weight[:, 128:256] = -weight[:, :128]
        gradient[:, 128:256] = gradient[:, :128]
    meta, qdq, packed = pack_gate(case + ':gradient_weight', weight, calibrate, quantize, prepare)
    input_device = gradient.to('neuron')
    output, gq, gm, dot = (v.cpu() for v in kernel[2](input_device, packed, meta, 2, True))
    expected_meta, expected_gradient = row_reference(gradient)
    require_exact(case + ':gradient_metadata', gm, expected_meta)
    require_exact(case + ':gradient_quantizer', gq, expected_gradient)
    raw_g = gq.float() * gm[:, 0:1]
    raw_w = qdq.cpu().float() * meta.cpu()[0, 0]
    pair_gate(case + ':backward_dot', dot, raw_g, raw_w.T,
              gm[:, 1:2], meta.cpu()[0, 1], full=tokens <= 512)
    require_exact(case + ':output_rounding', output, dot.bfloat16())
    normal = kernel[2](input_device, packed, meta, 2, False).cpu()
    repeated = kernel[2](input_device, packed, meta, 2, False).cpu()
    require_exact(case + ':normal_vs_diagnostic', normal, output)
    require_exact(case + ':repeat', repeated, output)
    require_exact(case + ':input_unchanged', input_device, gradient)
    surrogate = gradient.float() @ qdq.cpu().float().T
    quantization_diagnostic(case + ':original_bf16_surrogate_dot', output, surrogate)
    if case in ('random', 'real_down_gradient', 'real_input_gradient'):
        varied = gradient.clone(); varied[127] *= -7; varied[128] += 3
        actual = kernel[2](varied.to('neuron'), packed, meta, 2, False).cpu()
        keep = torch.ones(tokens, dtype=torch.bool); keep[127:129] = False
        require_exact(case + ':other_gradient_rows_unchanged', actual[keep], output[keep])
    print('FP8_GRADIENT_GEMM_GATE ' + json.dumps(dict(
        case=case, shape=list(shape), passed=True, online_row_calibration=True,
        pair_arithmetic='empirical11-bit paired sum then FP32 gamma_(K/2)',
        approximation_changes_existing_bf16_vjp=True,
        weight_gradients_or_autograd_integration=False)), flush=True)
    if benchmark:
        # Current BF16 backward already has QDQ weights from forward. The FP8
        # alternative adds one transpose/packed-cache preparation per update;
        # reuse existing scale metadata rather than charge a fictitious free
        # transpose or rederive scales from already rounded weights.
        ordinary_down = qdq.T.contiguous()
        native = torch.compile(lambda g, w: g @ w, backend='neuron', fullgraph=True, dynamic=False)
        lanes = selected_lanes(outputs)

        def native_update():
            for _ in range(2): value = native(input_device, ordinary_down)
            return value

        def fp8_update():
            converted = ordinary_down.T.contiguous()
            restored, current_packed = prepare[lanes](converted, meta, lanes)
            for _ in range(2): value = kernel[2](input_device, current_packed, meta, 2, False)
            return value

        # Verify the cache built from the existing QDQ representation uses the
        # same legacy bytes/values as the independently checked initial cache.
        restored, again = prepare[lanes](ordinary_down.T.contiguous(), meta, lanes)
        require_exact(case + ':backward_cache_repack', restored, qdq)
        require_exact(case + ':backward_packed_bytes', again, packed)
        measured = timing({'existing_bf16_two_dots': native_update,
                           'fp8_extra_cache_plus_two_dots': fp8_update}, steps, torch.neuron.synchronize)
        print('FP8_GRADIENT_GEMM_BENCH ' + json.dumps(dict(shape=list(shape), **measured,
            units='ms per incremental transpose/pack plus two gradient GEMMs',
            shared_forward_weight_calibration_already_paid=True,
            gradient_row_calibration_and_quantization_inside_kernel=True,
            complete_mlp_backward_optimizer_or_model_included=False)), flush=True)
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--steps', type=int, default=15)
    args = parser.parse_args()
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    kernel, prepare = wrap_nki(build_kernel()), wrap_nki(build_packer())
    calibrate, quantize = (wrap_nki(k) for k in build_kernels())
    print('FP8_GRADIENT_GEMM_CONTRACT ' + json.dumps(dict(
        finite_bf16_gradient_and_weight_max=2.0**100, scale_floor=2.0**-100,
        gradient_scale='same-row channels only', weight_scale='existing matrix metadata',
        independent_reference='CPU legacy quantizer; empirical pair rule with fixed gamma bound',
        no_tolerance_fitting=True, full_fp8_backward=False)), flush=True)
    for case in ('random', 'zero_gradient', 'zero_weight', 'tiny_gradient',
                 'row_scale_range', 'outliers', 'signed_cancellation'):
        run((512, 256, 512), case, kernel, calibrate, quantize, prepare, args.steps, False)
    if args.real_shape:
        for shape, case in [((16384, 1024, 4096), 'real_down_gradient'),
                            ((16384, 4096, 1024), 'real_input_gradient')]:
            run(shape, case, kernel, calibrate, quantize, prepare, args.steps, True)
    print('FP8_GRADIENT_GEMM_RESULT ' + json.dumps(dict(passed=True, real_shape=args.real_shape,
          integrated_backward=False, quality_gain_established=False)), flush=True)


if __name__ == '__main__':
    main()
