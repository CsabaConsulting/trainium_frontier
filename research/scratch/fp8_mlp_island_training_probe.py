#!/usr/bin/env python3
"""Gate FP8-forward/BF16-backward STE and charge complete training-component costs."""
import argparse
import gc
import json

import torch

from fp8_mlp_island_nki import build_kernel
from fp8_mlp_island_probe import make_inputs, run_case
from fp8_mlp_island_autograd import make_autograd, reference_and_bounds, audit
from fp8_weight_cache_nki import build_kernels
from fp8_weight_cache_probe import calibration_gate, require_exact, make_cached_autograd
from mlp_fused_bf16 import build_kernel as build_bf16, reference_forward
from mlp_fused_bf16_probe import timing


def emit(kind, record):
    print(kind + ' ' + json.dumps(record), flush=True)


def gate(shape, case, island, calibrate, quantize, steps, benchmark,
         token_rows=False, rms_fixed_input=False, token_row_backward=False):
    tokens, channels, hidden = shape
    tile = 128 if token_rows else min(512, tokens // 2)
    assert not rms_fixed_input or token_rows
    fixed_args = ()
    if rms_fixed_input:
        from fp8_rms_scale_contract import policy
        fixed_args = (policy(channels)[0],)
    x, up, down = make_inputs(shape, case, rms_fixed_input)
    generator = torch.Generator().manual_seed(9073 + tokens + channels)
    gradients = [(torch.randn(x.shape, generator=generator) * .03125).bfloat16()
                 for _ in range(2)]
    up_meta, up_cache = calibration_gate(case + ':training_up', up, calibrate, quantize)
    down_meta, down_cache = calibration_gate(case + ':training_down', down, calibrate, quantize)
    leaves = (x.to('neuron').requires_grad_(), up.float().to('neuron').requires_grad_(),
              down.float().to('neuron').requires_grad_())
    caches = up_cache, down_cache, up_meta, down_meta

    def call(xx, uu, dd, um, dm):
        return island[2](xx, uu, dd, um, dm, 2, tile, False, *fixed_args)

    apply = torch.compile(make_autograd(call, relu_token_major=token_rows,
                                       token_row_backward=token_row_backward),
                          backend='neuron', fullgraph=True, dynamic=False)
    with torch.no_grad():
        saved = tuple(value.cpu() for value in call(leaves[0], *caches))
    y, relu_t, xq, hq = saved
    if token_rows:
        relu_t = relu_t.T
    individual = []
    for index, gradient in enumerate(gradients):
        for leaf in leaves:
            leaf.grad = None
        output = apply(*leaves, *caches)
        output.backward(gradient.to('neuron'))
        require_exact(case + f':compiled_training_forward{index}', output, y)
        actual = tuple(leaf.grad.cpu() for leaf in leaves)
        assert actual[0].dtype == torch.bfloat16
        assert actual[1].dtype == actual[2].dtype == torch.float32
        reference, bounds = reference_and_bounds(xq, up_cache.cpu(), down_cache.cpu(),
                                                relu_t, hq, gradient)
        audit(case + f':microbatch{index}', actual, reference, bounds)
        individual.append(actual)
        del reference, bounds
    # A second backward must accumulate into each original FP32 master once.
    for leaf in leaves:
        leaf.grad = None
    for gradient in gradients:
        apply(*leaves, *caches).backward(gradient.to('neuron'))
    for index, leaf in enumerate(leaves):
        expected = (individual[0][index] + individual[1][index]).to(leaf.dtype)
        require_exact(case + f':two_microbatch_accumulation{index}', leaf.grad, expected)
    for name, leaf, original in zip(('x', 'up_master', 'down_master'), leaves, (x, up.float(), down.float())):
        require_exact(case + ':' + name + '_unchanged', leaf, original)
    assert not any(t.requires_grad for t in caches)
    emit('FP8_ISLAND_TRAINING_GATE', dict(shape=list(shape), case=case, passed=True,
         fp32_master_gradients=True, two_microbatch_accumulation_exact=True,
         token_row_output_layout=token_rows,
         fixed_rms_input_scale=rms_fixed_input,
         token_row_backward=token_row_backward,
         fp8_forward=True, bf16_backward=True, identity_quantizer_surrogate=True,
         forward_quality_not_established=True))

    if benchmark:
        bf16_kernel = build_bf16()
        from torch_neuronx import wrap_nki
        bf16_kernel = wrap_nki(bf16_kernel)
        control_tile = min(512, tokens // 2)

        def native_forward(xx, uu, dd, um, dm):
            result, relu = reference_forward(xx, uu, dd)
            return result, relu.T

        def bf16_forward(xx, uu, dd, um, dm):
            return bf16_kernel[2](xx, uu, dd, 2, control_tile, True, True, True, True, 0)

        def fp8_storage_forward(xx, uu, dd, um, dm):
            return bf16_kernel[2](xx, uu, dd, 2, control_tile, True, True, True, True, 0, True, um, dm)

        control = {name: torch.compile(make_cached_autograd(function), backend='neuron',
                                      fullgraph=True, dynamic=False)
                   for name, function in (('native', native_forward), ('bf16_cache', bf16_forward),
                                          ('fp8_storage', fp8_storage_forward))}
        device_gradient = gradients[0].to('neuron')
        from mlp_backward_rounding_bounds import reference_and_bounds as bf16_reference
        from mlp_backward_rounding_bounds import audit as bf16_audit
        for name, function in control.items():
            for leaf in leaves:
                leaf.grad = None
            if name == 'fp8_storage':
                ub, db = up_cache, down_cache
            else:
                ub, db = leaves[1].detach().bfloat16(), leaves[2].detach().bfloat16()
            function(*leaves, ub, db, up_meta, down_meta).backward(device_gradient)
            reference, bounds = bf16_reference(x, ub.cpu(), db.cpu(), gradients[0])
            bf16_audit(case + ':timing_control:' + name,
                       tuple(leaf.grad for leaf in leaves), reference, bounds)
            del reference, bounds

        if token_rows:
            # Retain a fresh original FP8 orientation as a paired timing arm.
            # Its unchanged kernel passed the wave73 forward-stage contract.
            original = wrap_nki(build_kernel())

            def original_call(xx, uu, dd, um, dm):
                return original[2](xx, uu, dd, um, dm, 2, control_tile, False)

            original_apply = torch.compile(make_autograd(original_call), backend='neuron',
                                           fullgraph=True, dynamic=False)
            with torch.no_grad():
                original_saved = tuple(v.cpu() for v in original_call(leaves[0], *caches))
            for leaf in leaves:
                leaf.grad = None
            original_output = original_apply(*leaves, *caches)
            original_output.backward(device_gradient)
            require_exact(case + ':original_fp8_timing_control_forward', original_output, original_saved[0])
            reference, bounds = reference_and_bounds(original_saved[2], up_cache.cpu(), down_cache.cpu(),
                                                    original_saved[1], original_saved[3], gradients[0])
            audit(case + ':original_fp8_timing_control', tuple(leaf.grad for leaf in leaves), reference, bounds)
            del reference, bounds, original_saved, original_output

        if rms_fixed_input:
            # Pair with the same token-row kernel under its old adaptive input
            # scale, using identical normalized inputs and weight-cache policy.
            def adaptive_call(xx, uu, dd, um, dm):
                return island[2](xx, uu, dd, um, dm, 2, tile, False)

            adaptive_apply = torch.compile(make_autograd(adaptive_call, relu_token_major=True),
                                           backend='neuron', fullgraph=True, dynamic=False)
            with torch.no_grad():
                adaptive_saved = tuple(v.cpu() for v in adaptive_call(leaves[0], *caches))
            for leaf in leaves:
                leaf.grad = None
            adaptive_output = adaptive_apply(*leaves, *caches)
            adaptive_output.backward(device_gradient)
            require_exact(case + ':adaptive_fp8_timing_control_forward', adaptive_output, adaptive_saved[0])
            reference, bounds = reference_and_bounds(adaptive_saved[2], up_cache.cpu(), down_cache.cpu(),
                                                    adaptive_saved[1].T, adaptive_saved[3], gradients[0])
            audit(case + ':adaptive_fp8_timing_control', tuple(leaf.grad for leaf in leaves), reference, bounds)
            del reference, bounds, adaptive_saved, adaptive_output

        if token_row_backward:
            previous_apply = torch.compile(make_autograd(call, relu_token_major=True),
                                           backend='neuron', fullgraph=True, dynamic=False)
            for leaf in leaves:
                leaf.grad = None
            previous_output = previous_apply(*leaves, *caches)
            previous_output.backward(device_gradient)
            require_exact(case + ':previous_backward_timing_control_forward', previous_output, y)
            reference, bounds = reference_and_bounds(xq, up_cache.cpu(), down_cache.cpu(),
                                                    relu_t, hq, gradients[0])
            audit(case + ':previous_backward_timing_control', tuple(leaf.grad for leaf in leaves), reference, bounds)
            del reference, bounds, previous_output

        def update(function, quantized):
            for leaf in leaves:
                leaf.grad = None
            ub, db = leaves[1].detach().bfloat16(), leaves[2].detach().bfloat16()
            um, dm = up_meta, down_meta
            if quantized:
                um, dm = calibrate[1](ub), calibrate[1](db)
                ub, db = quantize[2](ub, um, 2), quantize[2](db, dm, 2)
            for _ in range(2):
                result = function(*leaves, ub, db, um, dm)
                result.backward(device_gradient)
            return result

        arms = {
            'native_bf16_original': lambda: update(control['native'], False),
            'nki_bf16_cache_original': lambda: update(control['bf16_cache'], False),
            'nki_fp8_storage_bf16_math': lambda: update(control['fp8_storage'], True),
            ('nki_fp8_token_row_backward' if token_row_backward else
             'nki_fp8_rms_fixed_input_bf16_backward' if rms_fixed_input else
             'nki_fp8_token_rows_bf16_backward' if token_rows else
             'nki_fp8_island_bf16_backward'): lambda: update(apply, True),
        }
        if token_rows:
            arms['nki_fp8_original_layout_bf16_backward'] = lambda: update(original_apply, True)
        if rms_fixed_input:
            arms['nki_fp8_token_rows_adaptive_bf16_backward'] = lambda: update(adaptive_apply, True)
        if token_row_backward:
            arms['nki_fp8_previous_backward_same_forward'] = lambda: update(previous_apply, True)
        measured = timing(arms, steps, torch.neuron.synchronize)
        emit('FP8_ISLAND_TRAINING_BENCH', dict(shape=list(shape), **measured,
             microbatches_per_update=2, units='ms per weight-cache refresh plus two FW/BW microbatches',
             online_weight_calibration_and_master_casts_included=True,
             activation_scaling_and_all_saved_values_included=True,
             gradient_accumulation_included=True, fp8_forward=True, bf16_backward=True,
             token_row_output_layout=token_rows,
             fixed_rms_input_scale=rms_fixed_input,
             token_row_backward=token_row_backward,
             normalization_is_a_shared_caller_precondition_not_in_component_timing=rms_fixed_input,
             optimizer_collectives_and_surrounding_model_included=False,
             timing_control_backward_bounds_gated=True,
             control_arithmetic_differs_because_quantization_is_part_of_candidate=True))
    del leaves, saved, individual
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--token-rows', action='store_true')
    parser.add_argument('--rms-fixed-input', action='store_true')
    parser.add_argument('--token-row-backward', action='store_true')
    args = parser.parse_args()
    assert not args.rms_fixed_input or args.token_rows
    assert not args.token_row_backward or args.token_rows
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    selected_builder = build_kernel
    if args.token_rows:
        from fp8_mlp_token_rows_nki import build_kernel as selected_builder
    island = wrap_nki(selected_builder())
    calibrate, quantize = (wrap_nki(kernel) for kernel in build_kernels())
    emit('FP8_ISLAND_BACKWARD_CONTRACT', dict(
        forward_gates_required_before_launch=True, surrogate='identity for casts, quantizers and pair rounding',
        token_row_output_layout=args.token_rows,
        fixed_rms_input_scale=args.rms_fixed_input,
        token_row_backward=args.token_row_backward,
        reference='independent FP64 autograd with forward-audited saved values',
        error_bound='FP32 gamma_K accumulation and propagated BF16 u=2**-8 boundaries',
        gradients='BF16 input; FP32 master weights exactly once; detached caches'))
    cases = ('random', 'zero_up', 'zero_down', 'outliers', 'changed_weights')
    if args.rms_fixed_input:
        cases += ('norm_spikes', 'norm_tiny')
    for case in cases:
        gate((512, 256, 512), case, island, calibrate, quantize, args.steps,
             False, args.token_rows, args.rms_fixed_input, args.token_row_backward)
    if args.real_shape:
        # The larger forward has not been authorized by small-shape evidence
        # alone. Audit every new stage shape before any real-shape backward/timing.
        run_case((16384, 1024, 4096), 'real_random', island, calibrate, quantize,
                 args.token_rows, args.rms_fixed_input)
        gate((16384, 1024, 4096), 'real_random', island, calibrate, quantize, args.steps,
             True, args.token_rows, args.rms_fixed_input, args.token_row_backward)
    emit('FP8_ISLAND_TRAINING_RESULT', dict(passed=True, real_shape=args.real_shape,
         token_row_output_layout=args.token_rows,
         fixed_rms_input_scale=args.rms_fixed_input,
         token_row_backward=args.token_row_backward,
         fp8_forward=True, bf16_backward=True, promoted_into_training=False))


if __name__ == '__main__':
    main()
