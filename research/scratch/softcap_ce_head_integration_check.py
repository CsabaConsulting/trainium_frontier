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
    args = parser.parse_args()
    import torch_neuronx
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 32
    T.USE_WEIGHT_CACHE = True
    before_rng = torch.get_rng_state().clone()
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
        del head, h, target
        gc.collect()
    print('CE_HEAD_INTEGRATION_RESULT ' + json.dumps(dict(
        passed=True, comparisons=count, real_shape=args.real_shape,
        original_linear_and_bf16_cache=True, installation_rng_unchanged=True,
        actual_gpt_body_edit=installed['source_edit'],
        whole_model_gradient_equivalence_claimed=False)), flush=True)


if __name__ == '__main__':
    main()
