#!/usr/bin/env python3
"""Exact real Linear-head/loss gates before whole-model BF16-boundary timing."""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
from softcap_ce_bf16_integration import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--gradient-boundary-only', action='store_true',
                        help='Keep original FP32 head logits; change only the backward output boundary')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    import torch_neuronx
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 32
    T.USE_WEIGHT_CACHE = True
    before_rng = torch.get_rng_state().clone()
    if args.gradient_boundary_only:
        from softcap_ce_parallel_integration import install as install_adaptive, selected_lanes
        from softcap_ce_parallel_integration_check import native_loss_factory
        from softcap_ce_parallel_nki import build_kernels
        from torch_neuronx import wrap_nki
        install_adaptive(T)
        _, backward = (wrap_nki(kernel) for kernel in build_kernels())

        def head_backward(z, target, lse, gradient):
            lanes = selected_lanes(z.shape[0])
            return backward[lanes](z, target, lse, gradient, lanes, True)

        head_loss = native_loss_factory(T._softcap_ce_fwd, head_backward)
        installed = dict(source_edit='none; original FP32 head logits retained')
    else:
        installed = install(T)
    assert torch.equal(before_rng, torch.get_rng_state())
    shapes = [(256, 256)]
    if args.real_shape:
        shapes.append((16384, 1024))
    generator = torch.Generator().manual_seed(8091)
    count = 0
    for rows, channels in shapes:
        head = T.Linear(channels, 8192, bias=False).to('neuron')
        h_cpu = torch.randn(rows, channels, generator=generator).bfloat16()
        h = h_cpu.to('neuron').requires_grad_()
        target_cpu = torch.randint(8192, (rows,), generator=generator)
        target = target_cpu.to('neuron')
        gradient = (torch.randn(rows, generator=generator) * .5).to('neuron')

        def original(x, targets, reduction):
            return T.softcap_cross_entropy(head(x).float(), targets, reduction)

        def candidate(x, targets, reduction):
            if args.gradient_boundary_only:
                return head_loss(head(x).float(), targets, reduction)
            return T.softcap_cross_entropy(head(x), targets, reduction)

        functions = {'eager': (original, candidate),
                     'compiled': tuple(torch.compile(fn, backend='neuron', fullgraph=True,
                                                     dynamic=False) for fn in (original, candidate))}
        for scale in (0.0, .001, .03, .3):
            weight_cpu = torch.randn(8192, channels, generator=generator) * scale
            with torch.no_grad():
                head.weight.copy_(weight_cpu.to('neuron'))
            head.refresh_weight_cache(torch.bfloat16)
            for mode, pair in functions.items():
                for reduction in ('none', 'mean'):
                    outputs = []
                    for function in pair:
                        value = function(h, target, reduction)
                        upstream = gradient if reduction == 'none' else gradient[0]
                        gradients = torch.autograd.grad(value, (h, head.weight), grad_outputs=upstream)
                        outputs.append((value.detach().cpu(), *(v.cpu() for v in gradients)))
                    exact = [torch.equal(a, b) for a, b in zip(*outputs)]
                    finite = all(bool(torch.isfinite(v).all()) for row in outputs for v in row)
                    record = dict(rows=rows, channels=channels, weight_std=scale, mode=mode,
                                  gradient_boundary_only=args.gradient_boundary_only,
                                  reduction=reduction, exact_loss_input_master_gradient=exact,
                                  max_abs=[float((a.float()-b.float()).abs().max()) for a, b in zip(*outputs)],
                                  fp32_master_gradient=outputs[1][2].dtype == torch.float32,
                                  finite=finite, passed=all(exact) and finite)
                    print('CE_HEAD_INTEGRATION_GATE ' + json.dumps(record), flush=True)
                    assert record['passed'] and record['fp32_master_gradient'], record
                    count += 1
                    del outputs, value, gradients
            assert torch.equal(head.weight.detach().cpu(), weight_cpu)
            assert torch.equal(h.detach().cpu(), h_cpu) and torch.equal(target.cpu(), target_cpu)
        if args.benchmark and rows == 16384:
            from mlp_fused_bf16_probe import timing
            # Re-establish an already-gated representative weight scale before
            # timing. No optimizer or full-model costs are claimed here.
            with torch.no_grad():
                head.weight.copy_((weight_cpu * .1).to('neuron'))
            head.refresh_weight_cache(torch.bfloat16)

            def step(function):
                value = function(h, target, 'mean')
                derivatives = torch.autograd.grad(value, (h, head.weight))
                return value, derivatives

            checked = [step(fn) for fn in functions['compiled']]
            exact = [torch.equal(a.detach().cpu(), b.detach().cpu())
                     for a, b in zip((checked[0][0], *checked[0][1]),
                                     (checked[1][0], *checked[1][1]))]
            print('CE_HEAD_BENCH_BOUNDARY_GATE ' + json.dumps(dict(
                exact_loss_input_master_gradient=exact, passed=all(exact))), flush=True)
            assert all(exact), exact
            del checked
            measured = timing({name: (lambda fn=fn: step(fn))
                               for name, fn in zip(('original', 'candidate'), functions['compiled'])},
                              args.steps, torch.neuron.synchronize)
            print('CE_HEAD_INTEGRATION_BENCH ' + json.dumps(dict(
                shape=[rows, channels, 8192], **measured, units='ms per head/loss FW/BW',
                gradient_boundary_only=args.gradient_boundary_only,
                whole_model_and_optimizer_included=False)), flush=True)
        del head, h, target
        gc.collect()
    print('CE_HEAD_INTEGRATION_RESULT ' + json.dumps(dict(
        passed=True, comparisons=count, real_shape=args.real_shape,
        gradient_boundary_only=args.gradient_boundary_only,
        original_linear_and_bf16_cache=True, installation_rng_unchanged=True,
        actual_gpt_body_edit=installed['source_edit'],
        whole_model_gradient_equivalence_claimed=False)), flush=True)


if __name__ == '__main__':
    main()
