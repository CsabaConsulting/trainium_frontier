#!/usr/bin/env python3
"""Gate packed-cache MLP stages and its complete charged training component.

The independent legacy-byte/layout gate precedes this probe. Forward stages
reuse the independent quantizer and empirical pair-arithmetic contract. Backward
uses the existing FP64 surrogate bounds, exact paired gradients and two-microbatch
FP32-master accumulation. Timing charges calibration, both caches and all saves.
"""
import argparse
import gc
import json

import torch

from fp8_mlp_token_rows_nki import build_kernel
from fp8_mlp_island_probe import make_inputs, run_case
from fp8_mlp_island_autograd import make_autograd, reference_and_bounds, audit
from fp8_packed_weight_cache_nki import build_kernel as build_packer
from fp8_packed_weight_cache_probe import pack_gate, selected_lanes
from fp8_weight_cache_nki import build_kernels
from fp8_weight_cache_probe import require_exact, make_cached_autograd
from fp8_rms_scale_contract import policy
from mlp_fused_bf16 import reference_forward
from mlp_fused_bf16_probe import timing


def training_gate(shape, case, kernel, calibrate, quantize, prepare, steps, benchmark):
    tokens, channels, hidden = shape
    x, up, down = make_inputs(shape, case, True)
    um, ub, packed_up = pack_gate(case + ':training_up', up, calibrate, quantize, prepare)
    dm, db, packed_down = pack_gate(case + ':training_down', down, calibrate, quantize, prepare)
    caches = ub, db, um, dm
    packed_caches = *caches, packed_up, packed_down
    leaves = (x.to('neuron').requires_grad_(), up.float().to('neuron').requires_grad_(),
              down.float().to('neuron').requires_grad_())
    fixed_scale = policy(channels)[0]

    def old_forward(a, u, d, us, ds):
        return kernel[2](a, u, d, us, ds, 2, 128, False, fixed_scale)

    def packed_forward(a, u, d, us, ds, pu, pd):
        return kernel[2](a, u, d, us, ds, 2, 128, False, fixed_scale, pu, pd)

    old_apply = torch.compile(make_autograd(old_forward, True, True),
                              backend='neuron', fullgraph=True, dynamic=False)
    new_apply = torch.compile(make_autograd(packed_forward, True, True, packed_weights=True),
                              backend='neuron', fullgraph=True, dynamic=False)
    with torch.no_grad():
        saved = tuple(v.cpu() for v in packed_forward(leaves[0], *packed_caches))
        old_saved = tuple(v.cpu() for v in old_forward(leaves[0], *caches))
    for index, (actual, previous) in enumerate(zip(saved, old_saved)):
        require_exact(case + f':eager_packed_vs_old_saved{index}', actual, previous)
    y, relu_rows, xq, hq = saved
    generator = torch.Generator().manual_seed(9073 + tokens + channels)
    gradients = [(torch.randn(x.shape, generator=generator) * .03125).bfloat16() for _ in range(2)]
    individual = []
    for index, gradient in enumerate(gradients):
        observations = []
        for function, cache in ((old_apply, caches), (new_apply, packed_caches)):
            for leaf in leaves: leaf.grad = None
            output = function(*leaves, *cache)
            output.backward(gradient.to('neuron'))
            require_exact(case + f':compiled_forward{index}', output, y)
            observations.append(tuple(leaf.grad.cpu() for leaf in leaves))
        for field, actual, previous in zip(('dx', 'dup', 'ddown'), observations[1], observations[0]):
            require_exact(case + f':packed_vs_old_gradient{index}:' + field, actual, previous)
        actual = observations[1]
        assert actual[0].dtype == torch.bfloat16 and actual[1].dtype == actual[2].dtype == torch.float32
        reference, bounds = reference_and_bounds(xq, ub.cpu(), db.cpu(), relu_rows.T, hq, gradient)
        audit(case + f':packed_microbatch{index}', actual, reference, bounds)
        individual.append(actual)
        del reference, bounds, observations
    for leaf in leaves: leaf.grad = None
    for gradient in gradients:
        new_apply(*leaves, *packed_caches).backward(gradient.to('neuron'))
    for index, leaf in enumerate(leaves):
        expected = (individual[0][index] + individual[1][index]).to(leaf.dtype)
        require_exact(case + f':packed_accumulation{index}', leaf.grad, expected)
    for name, leaf, original in zip(('x', 'up_master', 'down_master'), leaves, (x, up.float(), down.float())):
        require_exact(case + ':' + name + '_unchanged', leaf, original)
    assert not any(v.requires_grad for v in packed_caches)
    print('FP8_PACKED_MLP_TRAINING_GATE ' + json.dumps(dict(
        case=case, shape=list(shape), passed=True, packed_vs_previous_exact=True,
        independent_fp64_surrogate_bounds=True, fp32_master_accumulation_exact=True,
        detached_caches=True, full_fp8_backward=False, training_quality_not_established=True)), flush=True)

    if benchmark:
        def native_forward(a, u, d, us, ds):
            result, relu = reference_forward(a, u, d)
            return result, relu.T

        native_apply = torch.compile(make_cached_autograd(native_forward),
                                     backend='neuron', fullgraph=True, dynamic=False)
        device_gradient = gradients[0].to('neuron')
        from mlp_backward_rounding_bounds import reference_and_bounds as native_reference
        from mlp_backward_rounding_bounds import audit as native_audit
        for leaf in leaves: leaf.grad = None
        native_ub, native_db = leaves[1].detach().bfloat16(), leaves[2].detach().bfloat16()
        native_apply(*leaves, native_ub, native_db, um, dm).backward(device_gradient)
        reference, bounds = native_reference(x, up, down, gradients[0])
        native_audit(case + ':native_timing_control', tuple(leaf.grad for leaf in leaves), reference, bounds)
        del reference, bounds

        def update(mode):
            for leaf in leaves: leaf.grad = None
            current_up, current_down = leaves[1].detach().bfloat16(), leaves[2].detach().bfloat16()
            up_meta, down_meta = um, dm
            if mode == 'native':
                function, extra = native_apply, ()
            else:
                up_meta, down_meta = calibrate[1](current_up), calibrate[1](current_down)
                if mode == 'packed':
                    ul, dl = selected_lanes(hidden), selected_lanes(channels)
                    current_up, pu = prepare[ul](current_up, up_meta, ul)
                    current_down, pd = prepare[dl](current_down, down_meta, dl)
                    function, extra = new_apply, (pu, pd)
                else:
                    current_up = quantize[2](current_up, up_meta, 2)
                    current_down = quantize[2](current_down, down_meta, 2)
                    function, extra = old_apply, ()
            for _ in range(2):
                output = function(*leaves, current_up, current_down, up_meta, down_meta, *extra)
                output.backward(device_gradient)
            return output

        measured = timing({name: (lambda mode=mode: update(mode)) for name, mode in (
            ('native_bf16_original', 'native'), ('nki_fp8_token_row_backward', 'old'),
            ('nki_fp8_packed_weights', 'packed'))}, steps, torch.neuron.synchronize)
        print('FP8_PACKED_MLP_TRAINING_BENCH ' + json.dumps(dict(
            shape=list(shape), **measured,
            units='ms per online cache refresh plus two FW/BW microbatches',
            microbatches_per_update=2, online_calibration_master_casts_and_packing_charged=True,
            bf16_backward_cache_and_saved_values_charged=True,
            normalization_shared_caller_precondition_not_in_component=True,
            optimizer_collectives_and_surrounding_model_included=False,
            numerical_gates_precede_timing=True)), flush=True)
    del leaves, saved, old_saved, individual
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('forward', 'training'), default='forward')
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    kernel, prepare = wrap_nki(build_kernel()), wrap_nki(build_packer())
    calibrate, quantize = (wrap_nki(k) for k in build_kernels())
    cases = ('random', 'zero_input', 'zero_up', 'zero_down', 'outliers', 'changed_weights',
             'token_scale_range', 'norm_spikes', 'norm_tiny')
    if args.stage == 'training':
        cases = ('random', 'zero_up', 'zero_down', 'outliers', 'changed_weights', 'norm_spikes', 'norm_tiny')
    for case in cases:
        if args.stage == 'forward':
            run_case((512, 256, 512), case, kernel, calibrate, quantize,
                     token_rows=True, rms_fixed_input=True, packed_prepare=prepare)
        else:
            training_gate((512, 256, 512), case, kernel, calibrate, quantize, prepare, args.steps, False)
    if args.real_shape:
        shape = (16384, 1024, 4096)
        if args.stage == 'forward':
            run_case(shape, 'real_random', kernel, calibrate, quantize,
                     token_rows=True, rms_fixed_input=True, packed_prepare=prepare)
        else:
            training_gate(shape, 'real_random', kernel, calibrate, quantize, prepare, args.steps, True)
    print('FP8_PACKED_MLP_RESULT ' + json.dumps(dict(passed=True, stage=args.stage,
          real_shape=args.real_shape, cases=len(cases) + int(args.real_shape),
          full_model_or_quality_promoted=False)), flush=True)


if __name__ == '__main__':
    main()
