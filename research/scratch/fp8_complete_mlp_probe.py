#!/usr/bin/env python3
"""Gate and time the complete FP8 MLP, including current cache preparation.

Independent CPU quantizers and the established empirical pair-arithmetic bounds
precede exact compiled-autograd comparisons. Two different microbatches share
current caches and accumulate into FP32 masters. Timing includes all forward
and backward products, casts, calibration, transposes, packing and saved data.
This is a component experiment, not full-model integration or a quality result.
"""
import argparse
import gc
import json

import torch

from fresh_start_fp8_probe import e4m3_legacy
from fp8_complete_mlp_autograd import make_autograd
from fp8_forward_saved_bytes_probe import run as saved_forward_gate
from fp8_gradient_prepare_nki import build_kernels as build_gradient_prepare
from fp8_mlp_data_backward_nki import build_kernel as build_data_backward
from fp8_mlp_island_autograd import make_autograd as make_mixed_autograd
from fp8_mlp_island_probe import make_inputs, pair_gate, row_reference
from fp8_mlp_token_rows_nki import build_kernel as build_forward
from fp8_packed_weight_cache_nki import build_kernel as build_pack
from fp8_packed_weight_cache_probe import decode_legacy, pack_gate, selected_lanes
from fp8_rms_scale_contract import normalize, policy
from fp8_weight_cache_nki import build_kernels as build_calibration
from fp8_weight_cache_probe import calibration_reference, make_cached_autograd, require_exact
from fp8_weight_gradient_nki import build_kernels as build_weight_gradient
from mlp_fused_bf16 import reference_forward
from mlp_fused_bf16_probe import timing


def emit(kind, **record):
    print(kind + ' ' + json.dumps(record), flush=True)


def gate_saved_bytes(name, values, saved):
    _, relu, xq, hq, xb, hb, hm = saved
    scale = policy(values.shape[1])[0]
    raw_x = e4m3_legacy(values.float() * scale)
    expected_hm, expected_hq = row_reference(relu.cpu().square())
    require_exact(name + ':input_bytes', decode_legacy(xb), raw_x)
    require_exact(name + ':input_qdq', xq, (raw_x / scale).bfloat16())
    require_exact(name + ':hidden_metadata', hm, expected_hm)
    require_exact(name + ':hidden_qdq', hq, expected_hq)
    require_exact(name + ':hidden_bytes', decode_legacy(hb),
                  e4m3_legacy(relu.cpu().square().float() * expected_hm[:, 0:1]))


def gated_backward(name, g, saved, caches, functions):
    """Independently gate all four products on the actual forward saves."""
    backward, partial, finish, product = functions
    _, relu, _, _, xb, hb, hm = saved
    _, _, um, dm, _, _, put, pdt, xm, one = caches
    gd = g.to('neuron')
    diagnostic = backward[2](gd, relu, pdt, put, dm, um, 2, 128, True, True)
    dx, pre, gq, pq, first, second, gm, pm = (v.cpu() for v in diagnostic)
    expected_gm, expected_gq = row_reference(g)
    require_exact(name + ':data_g_metadata', gm, expected_gm)
    require_exact(name + ':data_g_quantizer', gq, expected_gq)
    uq, dq = caches[0].cpu(), caches[1].cpu()
    umc, dmc = um.cpu(), dm.cpu()
    pair_gate(name + ':data_down', first, gq.float() * gm[:, 0:1],
              dq.float() * dmc[0, 0], gm[:, 1:2], dmc[0, 1], full=g.shape[0] <= 512)
    expected_pre = first.bfloat16() * (relu.cpu() * 2.0)
    require_exact(name + ':relu_derivative', pre, expected_pre)
    expected_pm, expected_pq = row_reference(pre)
    require_exact(name + ':data_pre_metadata', pm, expected_pm)
    require_exact(name + ':data_pre_quantizer', pq, expected_pq)
    pair_gate(name + ':data_up', second, pq.float() * pm[:, 0:1],
              uq.float() * umc[0, 0], pm[:, 1:2], umc[0, 1], full=g.shape[0] <= 512)
    require_exact(name + ':dx_bf16_rounding', dx, second.bfloat16())
    normal = backward[2](gd, relu, pdt, put, dm, um, 2, 128, False, True)
    require_exact(name + ':normal_dx', normal[0], dx)
    require_exact(name + ':normal_pre', normal[1], pre)
    weight_results = []
    for kind, values, bytes_right, factors, right_meta, transferred in (
            ('down', gd, hb, hm[:, 1:2].contiguous(), one, True),
            ('up', normal[1], xb, one, xm, False)):
        cpu_values = values.cpu()
        if transferred:
            cpu_values = (cpu_values.double() * factors.cpu().double()).bfloat16()
        expected_meta, _ = calibration_reference(cpu_values)
        parts = partial[2](values, factors, 2, transferred)
        byte_values, meta = finish[2](values, factors, parts, 2, transferred)
        expected_parts = cpu_values.float().abs().reshape(
            g.shape[0] // 256, 2, 128, cpu_values.shape[1]).amax(dim=(0, 3)).T.contiguous()
        require_exact(name + ':' + kind + ':partial_maxima', parts, expected_parts)
        require_exact(name + ':' + kind + ':global_metadata', meta, expected_meta)
        decoded = decode_legacy(byte_values)
        require_exact(name + ':' + kind + ':global_quantizer', decoded,
                      e4m3_legacy(cpu_values.float() * expected_meta[0, 0]))
        actual = product[2](byte_values, bytes_right, meta, right_meta, 2, True, True)
        assert actual.dtype == torch.float32
        inverse_rows = torch.full((cpu_values.shape[1], 1), float(expected_meta[0, 1]))
        pair_gate(name + ':weight_' + kind, actual.cpu(), decoded.T,
                  decode_legacy(bytes_right), inverse_rows, right_meta.cpu()[0, 1],
                  full=g.shape[0] <= 512)
        weight_results.append(actual.cpu())
    expected = dx, weight_results[1], weight_results[0]
    emit('FP8_COMPLETE_STAGES_GATE', name=name, passed=True,
         all_four_products_independently_bounded=True,
         actual_forward_saves_used=True, exact_previous_bf16_vjp=False)
    return expected


def build_functions(channels, kernel, backward, partial, finish, product):
    scale = policy(channels)[0]

    def saved_forward(x, u, d, um, dm, pu, pd):
        return kernel[2](x, u, d, um, dm, 2, 128, False, scale, pu, pd, True)

    def mixed_forward(x, u, d, um, dm, pu, pd):
        return kernel[2](x, u, d, um, dm, 2, 128, False, scale, pu, pd)

    def native_forward(x, u, d, um, dm):
        y, relu = reference_forward(x, u, d)
        return y, relu.T

    compile_fn = lambda fn: torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
    return (saved_forward,
            compile_fn(make_autograd(saved_forward, backward, partial, finish, product)),
            compile_fn(make_mixed_autograd(mixed_forward, True, True, packed_weights=True)),
            compile_fn(make_cached_autograd(native_forward)))


def prepare_caches(masters, calibrate, pack, mode, fixed):
    up, down = (value.detach().bfloat16() for value in masters)
    if mode == 'native':
        return up, down, fixed[0], fixed[1]
    um, dm = calibrate[1](up), calibrate[1](down)
    ul, dl = selected_lanes(up.shape[0]), selected_lanes(down.shape[0])
    uq, pu = pack[ul](up, um, ul)
    dq, pd = pack[dl](down, dm, dl)
    result = uq, dq, um, dm, pu, pd
    if mode == 'full':
        # Transposes retain the SAME global metadata; all movement is charged.
        _, put = pack[dl](up.T.contiguous(), um, dl)
        _, pdt = pack[ul](down.T.contiguous(), dm, ul)
        result += put, pdt, *fixed
    return result


def run(shape, case, kernels, applications, steps, benchmark):
    kernel, backward, partial, finish, product, calibrate, quantize, pack, byte_pack = kernels
    forward, complete, mixed, native = applications
    tokens, channels, hidden = shape
    # The existing independent forward/byte gates are retained unchanged.
    forward_case = case if case not in ('zero_gradient', 'tiny_gradient') else 'random'
    saved_forward_gate(shape, forward_case, kernel, calibrate, quantize, pack,
                       byte_pack, steps, False)
    x, up, down = make_inputs(shape, forward_case, True)
    generator = torch.Generator().manual_seed(97151 + channels + tokens)
    xs = (x, normalize(torch.randn(x.shape, generator=generator).bfloat16()))
    if case == 'zero_input': xs = (x, x.clone())
    gradients = [(torch.randn(x.shape, generator=generator) * .03125).bfloat16()
                 for _ in range(2)]
    if case == 'zero_gradient':
        for gradient in gradients: gradient.zero_()
    elif case == 'tiny_gradient':
        gradients = [(g.float() * 2.0**-110).bfloat16() for g in gradients]
    masters = (up.float().to('neuron').requires_grad_(),
               down.float().to('neuron').requires_grad_())
    inputs = [v.to('neuron').requires_grad_() for v in xs]
    scale, bound = policy(channels)
    xm = torch.tensor([scale, 1.0 / scale, bound]).repeat(128, 1).to('neuron')
    one = torch.ones(128, 3).to('neuron')
    fixed = xm, one
    caches = prepare_caches(masters, calibrate, pack, 'full', fixed)
    assert all(not v.requires_grad for v in caches)
    # Gate the actual backward cache orientation independently, including bytes.
    for name, weight, meta, packed in (
            ('up_transpose', up.T.contiguous(), caches[2], caches[6]),
            ('down_transpose', down.T.contiguous(), caches[3], caches[7])):
        reference_meta, _, reference_packed = pack_gate(case + ':' + name, weight, calibrate, quantize, pack)
        require_exact(case + ':' + name + ':same_global_scale', meta, reference_meta)
        require_exact(case + ':' + name + ':same_packed_bytes', packed, reference_packed)
    before_caches = tuple(v.cpu().clone() for v in caches)
    individual = []
    for index, (values, gradient) in enumerate(zip(inputs, gradients)):
        saved = forward(values.detach(), *caches[:6])
        gate_saved_bytes(case + f':micro{index}', xs[index], saved)
        expected = gated_backward(case + f':micro{index}', gradient, saved, caches,
                                  (backward, partial, finish, product))
        leaves = values, *masters
        for leaf in leaves: leaf.grad = None
        y = complete(*leaves, *caches)
        y.backward(gradient.to('neuron'))
        require_exact(case + f':compiled_y{index}', y, saved[0])
        actual = tuple(leaf.grad.cpu().clone() for leaf in leaves)
        for field, value, truth in zip(('dx', 'dup', 'ddown'), actual, expected):
            require_exact(case + f':compiled_{field}{index}', value, truth)
        assert actual[0].dtype == torch.bfloat16 and actual[1].dtype == actual[2].dtype == torch.float32
        individual.append(actual)
        if index == 0:
            for leaf in leaves: leaf.grad = None
            repeated = complete(*leaves, *caches)
            repeated.backward(gradient.to('neuron'))
            require_exact(case + ':repeat_y', repeated, y)
            for field, leaf, truth in zip(('dx', 'dup', 'ddown'), leaves, actual):
                require_exact(case + ':repeat_' + field, leaf.grad, truth)
        del saved, expected
    for leaf in (*inputs, *masters): leaf.grad = None
    for values, gradient in zip(inputs, gradients):
        complete(values, *masters, *caches).backward(gradient.to('neuron'))
    for index, master in enumerate(masters):
        require_exact(case + f':fp32_master_accumulation{index}', master.grad,
                      individual[0][index + 1] + individual[1][index + 1])
    for index, value in enumerate(inputs):
        require_exact(case + f':distinct_micro_input_gradient{index}', value.grad, individual[index][0])
        require_exact(case + f':input_unchanged{index}', value, xs[index])
    for index, (master, original) in enumerate(zip(masters, (up.float(), down.float()))):
        require_exact(case + f':master_unchanged{index}', master, original)
    for index, (current, before) in enumerate(zip(caches, before_caches)):
        require_exact(case + f':cache_unchanged{index}', current, before)
    emit('FP8_COMPLETE_AUTOGRAD_GATE', case=case, shape=list(shape), passed=True,
         independent_all_four_product_gates=True, compiled_vs_staged_exact=True,
         two_distinct_microbatches=True, fp32_master_accumulation_exact=True,
         forward_activation_bytes_reused=True, exact_previous_bf16_vjp=False,
         full_model_or_quality_gain_established=False)

    # Refresh every cache after an explicit master update, then recheck one
    # full derivative. This is a cache lifecycle gate, not optimizer tuning.
    if case in ('random', 'real_random'):
        with torch.no_grad():
            masters[0].mul_(.75).add_(.00390625)
            masters[1].mul_(1.25).sub_(.001953125)
        refreshed = prepare_caches(masters, calibrate, pack, 'full', fixed)
        assert not torch.equal(refreshed[0].cpu(), caches[0].cpu())
        assert not torch.equal(refreshed[1].cpu(), caches[1].cpu())
        saved = forward(inputs[0].detach(), *refreshed[:6])
        gate_saved_bytes(case + ':updated_master', xs[0], saved)
        expected = gated_backward(case + ':updated_master', gradients[0], saved, refreshed,
                                  (backward, partial, finish, product))
        for leaf in (inputs[0], *masters): leaf.grad = None
        output = complete(inputs[0], *masters, *refreshed)
        output.backward(gradients[0].to('neuron'))
        require_exact(case + ':updated_forward', output, saved[0])
        for field, leaf, truth in zip(('dx', 'dup', 'ddown'), (inputs[0], *masters), expected):
            require_exact(case + ':updated_' + field, leaf.grad, truth)
        emit('FP8_COMPLETE_CACHE_REFRESH_GATE', case=case, passed=True,
             current_master_refresh_verified=True)
        with torch.no_grad():
            masters[0].copy_(up.float().to('neuron'))
            masters[1].copy_(down.float().to('neuron'))
        restored = prepare_caches(masters, calibrate, pack, 'full', fixed)
        for index, (current, original) in enumerate(zip(restored, caches)):
            require_exact(case + f':restored_cache{index}', current, original)
        del refreshed, restored, saved, expected

    if benchmark:
        # Keep the native control's independent derivative gate. Small cases
        # and the actual shape use the original FP64 propagated-rounding bound.
        from mlp_backward_rounding_bounds import reference_and_bounds, audit
        for leaf in (inputs[0], *masters): leaf.grad = None
        nc = prepare_caches(masters, calibrate, pack, 'native', fixed)
        native(inputs[0], *masters, *nc).backward(gradients[0].to('neuron'))
        truth, bounds = reference_and_bounds(x, up, down, gradients[0])
        audit(case + ':native_timing_control', tuple(v.grad for v in (inputs[0], *masters)), truth, bounds)
        del truth, bounds
        device_gradients = [v.to('neuron') for v in gradients]

        def update(mode):
            for leaf in (*inputs, *masters): leaf.grad = None
            current = prepare_caches(masters, calibrate, pack, mode, fixed)
            fn = {'native': native, 'mixed': mixed, 'full': complete}[mode]
            for values, gradient in zip(inputs, device_gradients):
                result = fn(values, *masters, *current)
                result.backward(gradient)
            return result

        for mode in ('native', 'mixed', 'full'):
            update(mode)
            for leaf in (*inputs, *masters):
                assert leaf.grad is not None and bool(torch.isfinite(leaf.grad.cpu()).all())
        for index, master in enumerate(masters):
            require_exact('timed_full:master_accumulation' + str(index), master.grad,
                          individual[0][index + 1] + individual[1][index + 1])
        measured = timing({name: (lambda selected=mode: update(selected)) for name, mode in (
            ('native_bf16', 'native'), ('fp8_forward_bf16_backward', 'mixed'),
            ('fp8_forward_all_four_backward_products', 'full'))}, steps, torch.neuron.synchronize)
        emit('FP8_COMPLETE_MLP_BENCH', shape=list(shape), **measured,
             units='ms per current online caches plus two distinct FW/BW microbatches',
             all_master_casts_calibration_transposes_packing_saves_charged=True,
             fp32_master_gradient_accumulation_charged=True,
             bf16_relu_derivative=True, new_approximate_vjp=True,
             optimizer_collectives_surrounding_model_excluded=True,
             whole_model_speed_or_quality_gain_established=False)

    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    kernel, backward, pack = (wrap_nki(fn()) for fn in (build_forward, build_data_backward, build_pack))
    partial, finish = (wrap_nki(k) for k in build_gradient_prepare())
    _, byte_pack, product = (wrap_nki(k) for k in build_weight_gradient())
    calibrate, quantize = (wrap_nki(k) for k in build_calibration())
    kernels = kernel, backward, partial, finish, product, calibrate, quantize, pack, byte_pack
    small = build_functions(256, kernel, backward, partial, finish, product)
    cases = ('random', 'zero_input', 'zero_up', 'zero_down', 'zero_gradient',
             'tiny_gradient', 'outliers', 'changed_weights', 'norm_spikes', 'norm_tiny')
    emit('FP8_COMPLETE_MLP_CONTRACT', gradient_data_scale='current per row',
         gradient_weight_scale='current matrix wide after declared BF16 hidden scale transfer',
         derivative='BF16 ReLU squared', master_gradients='FP32',
         quantizer='legacy E4M3', arithmetic='independently characterized paired 11-bit plus FP32 accumulation',
         exact_original_bf16_vjp=False, model_or_quality_claim=False)
    for case in cases:
        run((512, 256, 512), case, kernels, small, args.steps, False)
    if args.real_shape:
        large = build_functions(1024, kernel, backward, partial, finish, product)
        run((16384, 1024, 4096), 'real_random', kernels, large, args.steps, True)
    emit('FP8_COMPLETE_MLP_RESULT', passed=True, cases=len(cases) + int(args.real_shape),
         real_shape=args.real_shape, full_model_speed_or_quality_claim=False)


if __name__ == '__main__':
    main()
