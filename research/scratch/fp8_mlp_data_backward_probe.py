#!/usr/bin/env python3
"""Stagewise fused FP8 data-gradient checks; weight gradients remain separate."""
import argparse
import gc
import json

import torch

from fp8_mlp_data_backward_nki import build_kernel
from fp8_gradient_gemm_nki import build_kernel as build_dot
from fp8_mlp_island_probe import row_reference, pair_gate
from fp8_packed_weight_cache_nki import build_kernel as build_packer
from fp8_packed_weight_cache_probe import pack_gate, selected_lanes, decode_legacy
from fp8_weight_cache_nki import build_kernels
from fp8_weight_cache_probe import require_exact, quantization_diagnostic
from mlp_backward_rounding_bounds import gamma
from mlp_fused_bf16_probe import timing


def native_gate(gradient, down, up, relu, actual):
    """Independent FP64 chain with propagated BF16/FP32 dot-rounding bounds."""
    g, d, u, r = (v.double() for v in (gradient, down, up, relu))
    unit = 2.0**-8
    first = g @ d
    acc = gamma(g.shape[1]) * (g.abs() @ d.abs())
    first_error = acc + unit * (first.abs() + acc)
    pre = first * (r * 2)
    incoming = first_error * (r * 2)
    pre_error = incoming + unit * (pre.abs() + incoming)
    dx = pre @ u
    propagated = pre_error @ u.abs()
    acc = gamma(u.shape[0]) * ((pre.abs() + pre_error) @ u.abs())
    dx_error = propagated + acc + unit * (dx.abs() + propagated + acc)
    for field, value, truth, bound in zip(('dx', 'dpre'), actual, (dx, pre), (dx_error, pre_error)):
        value = value.cpu().double()
        error = (value - truth).abs()
        failures = int((error > bound + 1e-30).sum())
        negative = int(((value * .5 - truth).abs() > bound + 1e-30).sum())
        record = dict(field=field, finite=bool(torch.isfinite(value).all()),
                      bound_failures=failures, half_output_negative_control_failures=negative,
                      max_abs=float(error.max()), passed=failures == 0)
        print('FP8_DATA_BACKWARD_NATIVE_BOUND ' + json.dumps(record), flush=True)
        assert record['finite'] and record['passed'] and (not bool(truth.abs().any()) or negative > 0)


def repack_gate(name, value, packed):
    a, b = value.cpu(), packed.cpu()
    changed = a != b
    zeros = ((a & 127) == 0) & ((b & 127) == 0)
    assert not bool((changed & ~zeros).any())
    require_exact(name + ':decoded', decode_legacy(a), decode_legacy(b))
    print('FP8_DATA_BACKWARD_REPACK ' + json.dumps(dict(
        name=name, signed_zero_changes=int((changed & zeros).sum()), nonzero_changes=0, passed=True)), flush=True)


def run(shape, case, kernel, dot, calibrate, quantize, prepare, steps, benchmark,
        fused_abs=False):
    tokens, channels, hidden = shape
    generator = torch.Generator().manual_seed(9391 + channels + tokens)
    g = (torch.randn(tokens, channels, generator=generator) * .03125).bfloat16()
    r = torch.randn(tokens, hidden, generator=generator).relu().bfloat16()
    # First operand is Wdown.T; second is Wup.T. Use the original matrix-wide
    # calibration policy: transpose does not change its maximum or exponent.
    dt = (torch.randn(hidden, channels, generator=generator) / hidden**.5).bfloat16()
    ut = (torch.randn(channels, hidden, generator=generator) / channels**.5).bfloat16()
    if case == 'zero_gradient': g.zero_()
    if case == 'zero_relu': r.zero_()
    if case == 'zero_down': dt.zero_()
    if case == 'zero_up': ut.zero_()
    if case == 'tiny_gradient': g = (g.float() * 2.0**-110).bfloat16()
    if case == 'subnormal_gradient':
        bits = (torch.arange(tokens * channels) % 127 + 1).short().reshape(tokens, channels)
        bits[1::2] |= -32768
        g = bits.view(torch.bfloat16)
    if case == 'signed_zero_gradient':
        bits = torch.zeros((tokens, channels), dtype=torch.int16)
        bits[1::2] = -32768
        g = bits.view(torch.bfloat16)
    if case == 'row_scale_range':
        g = torch.ldexp(g.float(), torch.linspace(-60, 30, tokens).round().int()[:, None]).bfloat16()
    if case == 'outliers':
        g[127, 0] = 2.0**20; g[128, -1] = -(2.0**20)
        r[127, 255] = 32; dt[3, 128] = 8; ut[128, 255] = -4
    dm, dq_t, pd = pack_gate(case + ':down_transpose', dt, calibrate, quantize, prepare)
    um, uq_t, pu = pack_gate(case + ':up_transpose', ut, calibrate, quantize, prepare)
    gd, rd = g.to('neuron'), r.to('neuron')
    diagnostic = kernel[2](gd, rd, pd, pu, dm, um, 2, 128, True, fused_abs)
    dx, pre, gq, pq, first, second, gm, pm = (v.cpu() for v in diagnostic)
    expected_gm, expected_gq = row_reference(g)
    require_exact(case + ':gradient_metadata', gm, expected_gm)
    require_exact(case + ':gradient_quantizer', gq, expected_gq)
    pair_gate(case + ':first_dot', first, gq.float() * gm[:, 0:1],
              (dq_t.cpu().float() * dm.cpu()[0, 0]).T,
              gm[:, 1:2], dm.cpu()[0, 1], full=tokens <= 512)
    expected_pre = first.bfloat16() * (r * 2.0)
    require_exact(case + ':bf16_relu_derivative', pre, expected_pre)
    expected_pm, expected_pq = row_reference(pre)
    require_exact(case + ':pre_metadata', pm, expected_pm)
    require_exact(case + ':pre_quantizer', pq, expected_pq)
    pair_gate(case + ':second_dot', second, pq.float() * pm[:, 0:1],
              (uq_t.cpu().float() * um.cpu()[0, 0]).T,
              pm[:, 1:2], um.cpu()[0, 1], full=tokens <= 512)
    require_exact(case + ':dx_rounding', dx, second.bfloat16())
    if fused_abs:
        original = kernel[2](gd, rd, pd, pu, dm, um, 2, 128, True, False)
        for field, observed, expected in zip(
                ('dx', 'dpre', 'gqdq', 'preqdq', 'first_dot', 'second_dot',
                 'gradient_metadata', 'pre_metadata'), diagnostic, original):
            require_exact(case + ':original_algorithm:' + field, observed, expected)
    normal = kernel[2](gd, rd, pd, pu, dm, um, 2, 128, False, fused_abs)
    repeated = kernel[2](gd, rd, pd, pu, dm, um, 2, 128, False, fused_abs)
    for field, observed, again, expected in zip(('dx', 'dpre'), normal, repeated, (dx, pre)):
        require_exact(case + ':diagnostic_vs_normal:' + field, observed, expected)
        require_exact(case + ':repeat:' + field, again, expected)
    separate_first = dot[2](gd, pd, dm, 2, False)
    separate_pre = separate_first * (rd * 2.0)
    separate_dx = dot[2](separate_pre, pu, um, 2, False)
    require_exact(case + ':separate_stage_pre', pre, separate_pre)
    require_exact(case + ':separate_stage_dx', dx, separate_dx)
    require_exact(case + ':gradient_unchanged', gd, g)
    require_exact(case + ':relu_unchanged', rd, r)
    plain_pre = (g.float() @ dq_t.cpu().float().T) * (r.float() * 2.0)
    plain_dx = plain_pre @ uq_t.cpu().float().T
    quantization_diagnostic(case + ':bf16_surrogate_pre', pre, plain_pre)
    quantization_diagnostic(case + ':bf16_surrogate_dx', dx, plain_dx)
    if case in ('random', 'real_random'):
        changed_g, changed_r = g.clone(), r.clone()
        changed_g[127] *= -7; changed_r[128] *= 3
        changed = kernel[2](changed_g.to('neuron'), changed_r.to('neuron'),
                            pd, pu, dm, um, 2, 128, False, fused_abs)
        keep = torch.ones(tokens, dtype=torch.bool); keep[127:129] = False
        for field, observed, expected in zip(('dx', 'dpre'), changed, (dx, pre)):
            require_exact(case + ':other_rows_unchanged:' + field, observed.cpu()[keep], expected[keep])
    print('FP8_DATA_BACKWARD_STAGE_GATE ' + json.dumps(dict(
        case=case, shape=list(shape), passed=True, separate_and_fused_algorithm_exact=True,
        independent_quantizer_and_pair_stages=True, bf16_pre_saved_for_weight_gradient=True,
        fused_abs=fused_abs, original_algorithm_exact=fused_abs,
        approximation_changes_vjp=True, autograd_or_weight_gradient_integration=False)), flush=True)
    if benchmark:
        down, up = dq_t.T.contiguous(), uq_t.T.contiguous()

        def native_data(g, r, d, u):
            pre = (g @ d) * (r * 2.0)
            return pre @ u, pre

        native = torch.compile(native_data, backend='neuron', fullgraph=True, dynamic=False)
        native_gate(g, down.cpu(), up.cpu(), r, native(gd, rd, down, up))
        dl, ul = selected_lanes(hidden), selected_lanes(channels)
        rebuilt_d, packed_d = prepare[dl](down.T.contiguous(), dm, dl)
        rebuilt_u, packed_u = prepare[ul](up.T.contiguous(), um, ul)
        require_exact('real:down_repack_qdq', rebuilt_d, dq_t)
        require_exact('real:up_repack_qdq', rebuilt_u, uq_t)
        repack_gate('real:down', packed_d, pd); repack_gate('real:up', packed_u, pu)

        def update(mode):
            if mode != 'native':
                _, current_d = prepare[dl](down.T.contiguous(), dm, dl)
                _, current_u = prepare[ul](up.T.contiguous(), um, ul)
            for _ in range(2):
                if mode == 'native': value = native(gd, rd, down, up)
                elif mode == 'separate':
                    first = dot[2](gd, current_d, dm, 2, False)
                    pre = first * (rd * 2.0)
                    value = dot[2](pre, current_u, um, 2, False), pre
                else:
                    value = kernel[2](gd, rd, current_d, current_u, dm, um, 2, 128,
                                      False, mode == 'fused_abs')
            return value

        methods = dict((name, (lambda mode=mode: update(mode))) for name, mode in (
            ('native_bf16_chain', 'native'), ('fp8_separate_dots', 'separate'),
            ('fp8_fused_data_backward', 'fused')))
        if fused_abs:
            methods['fp8_fused_abs_data_backward'] = lambda: update('fused_abs')
        measured = timing(methods, steps, torch.neuron.synchronize)
        print('FP8_DATA_BACKWARD_BENCH ' + json.dumps(dict(shape=list(shape), **measured,
            units='ms per extra transposes/packing plus two data-gradient chains',
            all_extra_backward_cache_work_charged=True, shared_forward_calibration_already_paid=True,
            fused_abs=fused_abs,
            weight_gradients_fp32_accumulation_optimizer_model_excluded=True)), flush=True)
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--fused-abs', action='store_true')
    args = parser.parse_args()
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    kernel, dot, prepare = wrap_nki(build_kernel()), wrap_nki(build_dot()), wrap_nki(build_packer())
    calibrate, quantize = (wrap_nki(k) for k in build_kernels())
    print('FP8_DATA_BACKWARD_CONTRACT ' + json.dumps(dict(
        algorithm='row-quantized G dot Wdown; BF16 multiply by2*ReLU; row-quantized dPre dot Wup',
        arithmetic='unchanged legacy quantizer, paired11-bit sums and FP32 gamma bounds',
        calibration='signed per-row maxima at both gradient boundaries',
        fused_abs=args.fused_abs,
        original_bf16_vjp_exact=False, full_fp8_backward=False)), flush=True)
    for case in ('random', 'zero_gradient', 'zero_relu', 'zero_down', 'zero_up',
                 'tiny_gradient', 'subnormal_gradient', 'signed_zero_gradient',
                 'row_scale_range', 'outliers'):
        run((512, 256, 512), case, kernel, dot, calibrate, quantize, prepare,
            args.steps, False, args.fused_abs)
    if args.real_shape:
        run((16384, 1024, 4096), 'real_random', kernel, dot, calibrate, quantize,
            prepare, args.steps, True, args.fused_abs)
    print('FP8_DATA_BACKWARD_RESULT ' + json.dumps(dict(passed=True, real_shape=args.real_shape,
          fused_abs=args.fused_abs,
          integrated_backward=False, full_model_speed_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
