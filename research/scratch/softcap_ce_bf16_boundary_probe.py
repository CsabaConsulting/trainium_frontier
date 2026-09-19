#!/usr/bin/env python3
"""Test eliminating FP32 HBM logits/gradients at an existing BF16 head boundary.

All loss arithmetic remains FP32. The candidate must exactly reproduce the
original loss and the original derivative rounded by autograd to its BF16
input. This probe does not modify GPT.forward, train.py or the scorer. A
component win still needs head/caller integration and whole-step measurement.
"""
import argparse
import gc
import json

import torch
from torch_neuronx import wrap_nki

from softcap_ce_parallel_nki import build_kernels
from softcap_ce_parallel_integration import selected_lanes
from softcap_ce_parallel_integration_check import native_loss_factory, T
from mlp_fused_bf16_probe import timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 32
    assert T.LOGIT_SOFTCAP == 15.0 and not T.LOSS_BYTE_WEIGHT
    original = native_loss_factory(T._softcap_ce_fwd, T._softcap_ce_bwd)
    forward, backward = (wrap_nki(kernel) for kernel in build_kernels())

    def fwd(z, target):
        lanes = selected_lanes(z.shape[0])
        return forward[lanes](z, target, lanes, True)

    def bwd(z, target, lse, gradient):
        lanes = selected_lanes(z.shape[0])
        return backward[lanes](z, target, lse, gradient, lanes)

    adaptive = native_loss_factory(fwd, bwd)

    def original_boundary(z, target, reduction):
        return original(z.float(), target, reduction)

    def adaptive_fp32_boundary(z, target, reduction):
        return adaptive(z.float(), target, reduction)

    functions = {
        'original_fp32_boundary': original_boundary,
        'adaptive_fp32_boundary': adaptive_fp32_boundary,
        'adaptive_bf16_boundary': adaptive,
    }
    compiled = {name: torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
                for name, fn in functions.items()}
    generator = torch.Generator().manual_seed(7813)
    cases = [(rows, 'random') for rows in (128, 256, 384, 512, 1024)]
    cases += [(256, case) for case in ('tiny', 'saturation', 'uniform', 'confident_correct',
                                      'confident_wrong', 'zero_gradient')]
    if args.real_shape:
        cases.append((16384, 'real_bf16_head_values'))
    print('CE_BF16_BOUNDARY_CONTRACT ' + json.dumps(dict(
        input='BF16 logits already rounded by head', computation='unchanged FP32',
        gradient='FP32 internal, then exact existing BF16 boundary rounding',
        gate='bit-exact loss and BF16 gradient versus protected original',
        exact_lse_and_shape_dispatch=True, approximate_functions=False,
        head_and_full_model_integration=False)), flush=True)
    count = 0
    for rows, case in cases:
        z = torch.randn(rows, 8192, generator=generator) * 6
        target = torch.randint(8192, (rows,), generator=generator)
        target[:4] = torch.tensor([0, 8191, 4095, 4096])
        gradient = torch.randn(rows, generator=generator) * .5
        if case == 'tiny':
            z *= .0001
        elif case == 'saturation':
            z *= 1000
        elif case == 'uniform':
            z.fill_(.7)
        elif case.startswith('confident'):
            z.zero_()
            z[torch.arange(rows), target] = 500 if case == 'confident_correct' else -500
            if case == 'confident_wrong':
                z[torch.arange(rows), (target + 1) % 8192] = 500
        elif case == 'zero_gradient':
            gradient.zero_()
        z = z.bfloat16()
        dz = z.to('neuron').requires_grad_()
        dt, dg = target.to('neuron'), gradient.to('neuron')
        target_column = dt.reshape(-1, 1).float()
        with torch.no_grad():
            reference_loss, reference_lse = T._softcap_ce_fwd(dz.float(), target_column)
            candidate_loss, candidate_lse = fwd(dz, target_column)
            exact_lse = torch.equal(reference_lse.cpu(), candidate_lse.cpu())
            exact_loss = torch.equal(reference_loss.cpu(), candidate_loss.cpu())
            print('CE_BF16_BOUNDARY_SAVED_GATE ' + json.dumps(dict(
                rows=rows, case=case, exact_loss=exact_loss, exact_saved_lse=exact_lse,
                passed=exact_loss and exact_lse)), flush=True)
            assert exact_loss and exact_lse
        del reference_loss, reference_lse, candidate_loss, candidate_lse
        reductions = ('none', 'mean') if case in ('random', 'real_bf16_head_values') else ('none',)
        for mode, arms in [('eager', functions), ('compiled', compiled)]:
            for reduction in reductions:
                observed = {}
                for name, function in arms.items():
                    value = function(dz, dt, reduction)
                    upstream = dg if reduction == 'none' else dg[0]
                    derivative, = torch.autograd.grad(value, dz, grad_outputs=upstream)
                    observed[name] = value.detach().cpu(), derivative.cpu()
                reference = observed['original_fp32_boundary']
                for name in ('adaptive_fp32_boundary', 'adaptive_bf16_boundary'):
                    actual = observed[name]
                    exact = [torch.equal(a, b) for a, b in zip(actual, reference)]
                    finite = all(bool(torch.isfinite(v).all()) for v in (*actual, *reference))
                    record = dict(rows=rows, case=case, mode=mode, reduction=reduction, arm=name,
                                  exact_loss_gradient=exact, finite=finite,
                                  gradient_dtype=str(actual[1].dtype),
                                  max_abs_loss_gradient=[float((a.float()-b.float()).abs().max())
                                                        for a, b in zip(actual, reference)],
                                  passed=all(exact) and finite and actual[1].dtype == torch.bfloat16)
                    print('CE_BF16_BOUNDARY_GATE ' + json.dumps(record), flush=True)
                    assert record['passed'], record
                    count += 1
                del observed, value, derivative
        assert torch.equal(dz.detach().cpu(), z) and torch.equal(dt.cpu(), target)
        if case == 'real_bf16_head_values':
            def step(function):
                loss = function(dz, dt, 'mean')
                derivative, = torch.autograd.grad(loss, dz)
                return loss, derivative

            measured = timing({name: (lambda fn=fn: step(fn)) for name, fn in compiled.items()},
                              args.steps, torch.neuron.synchronize)
            print('CE_BF16_BOUNDARY_BENCH ' + json.dumps(dict(
                shape=[rows, 8192], **measured, units='ms per loss FW/BW',
                fp32_boundary_casts_charged=True, bf16_input_and_gradient=True,
                caller_matmul_and_surrounding_model_included=False,
                speed_claim_is_component_only=True)), flush=True)
        del dz, dt, dg
        gc.collect()
    print('CE_BF16_BOUNDARY_RESULT ' + json.dumps(dict(
        passed=True, comparisons=count, real_shape=args.real_shape,
        integrated_into_training=False, protected_trainer_unchanged=True)), flush=True)


if __name__ == '__main__':
    main()
