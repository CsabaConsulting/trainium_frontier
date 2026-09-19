#!/usr/bin/env python3
"""Fair BF16 controls for the FP8 caller-layout finding, before model integration.

Both controls use native BF16 forward and the same cached weights. Compare the
deployed contiguous HxT save, the component control's HxT view, and a TxH save
with matching backward. Use independent FP64/BF16 error bounds fixed before
measurement; report exact pair comparisons separately, without fitting tolerance.
"""
import argparse
import gc
import json

import torch

from fp8_mlp_island_probe import make_inputs
from fp8_weight_cache_probe import make_cached_autograd, require_exact
from mlp_fused_bf16 import reference_forward
from mlp_fused_bf16_probe import timing
from mlp_backward_rounding_bounds import gamma, reference_and_bounds, audit


def forward_reference_and_bound(x, up, down):
    x, up, down = (v.double() for v in (x, up, down))
    u = 2.0**-8
    hidden = (x @ up.T).relu()
    accum = gamma(x.shape[1]) * (x.abs() @ up.abs().T)
    hidden_error = accum + u * (hidden.abs() + accum)
    activated = hidden.square()
    incoming = 2 * hidden * hidden_error + hidden_error.square()
    activated_error = incoming + u * (activated + incoming)
    result = activated @ down.T
    propagated = activated_error @ down.abs().T
    accum = gamma(up.shape[0]) * ((activated + activated_error) @ down.abs().T)
    bound = propagated + accum + u * (result.abs() + propagated + accum)
    return result, bound


def gate(shape, case, steps, benchmark):
    x, up, down = make_inputs(shape, case, True)
    leaves = (x.to('neuron').requires_grad_(), up.float().to('neuron').requires_grad_(),
              down.float().to('neuron').requires_grad_())
    caches = leaves[1].detach().bfloat16(), leaves[2].detach().bfloat16()
    metadata = torch.zeros(128, 3).to('neuron')  # Unused by every native arm.

    def view_forward(a, u, d, um, dm):
        result, relu = reference_forward(a, u, d)
        return result, relu.T

    def contiguous_forward(a, u, d, um, dm):
        result, relu = reference_forward(a, u, d)
        return result, relu.T.contiguous()

    def rows_forward(a, u, d, um, dm):
        return reference_forward(a, u, d)

    functions = {name: torch.compile(make_cached_autograd(fn, token_rows=rows),
                                     backend='neuron', fullgraph=True, dynamic=False)
                 for name, fn, rows in [('component_hxt_view', view_forward, False),
                                       ('deployed_hxt_contiguous', contiguous_forward, False),
                                       ('candidate_txh', rows_forward, True)]}
    generator = torch.Generator().manual_seed(8601 + shape[0])
    gradients = [(torch.randn(x.shape, generator=generator) * .03125).bfloat16() for _ in range(2)]
    f_truth, f_bound = forward_reference_and_bound(x, up, down)
    individual = {name: [] for name in functions}
    for index, gradient in enumerate(gradients):
        truth, bounds = reference_and_bounds(x, up, down, gradient)
        output_pairs = {}
        for name, fn in functions.items():
            for leaf in leaves: leaf.grad = None
            output = fn(*leaves, *caches, metadata, metadata)
            output.backward(gradient.to('neuron'))
            actual_y = output.detach().cpu()
            error = (actual_y.double() - f_truth).abs()
            failures = int((error > f_bound + 1e-30).sum())
            negative = int(((actual_y.double() * .5 - f_truth).abs() > f_bound + 1e-30).sum())
            record = dict(case=case, arm=name, microbatch=index, finite=bool(torch.isfinite(actual_y).all()),
                          bound_failures=failures, half_output_negative_control_failures=negative,
                          max_abs=float(error.max()), passed=failures == 0)
            print('NATIVE_MLP_TOKEN_ROWS_FORWARD_BOUND ' + json.dumps(record), flush=True)
            assert record['passed'] and record['finite']
            assert not bool(f_truth.abs().any()) or negative > 0
            actual = tuple(leaf.grad.cpu() for leaf in leaves)
            assert actual[0].dtype == torch.bfloat16 and actual[1].dtype == actual[2].dtype == torch.float32
            audit(case + ':' + name + f':microbatch{index}', actual, truth, bounds)
            individual[name].append(actual)
            output_pairs[name] = actual_y
        print('NATIVE_MLP_TOKEN_ROWS_PAIR_DIAGNOSTIC ' + json.dumps(dict(
            case=case, microbatch=index,
            exact_outputs={name: torch.equal(value, output_pairs['deployed_hxt_contiguous'])
                           for name, value in output_pairs.items()},
            exact_gradients={name: [torch.equal(a, b) for a, b in zip(values[index],
                              individual['deployed_hxt_contiguous'][index])]
                             for name, values in individual.items()},
            numerical_gate='independent FP64 propagated BF16/FP32 bounds')), flush=True)
        del truth, bounds
    for name, fn in functions.items():
        for leaf in leaves: leaf.grad = None
        for gradient in gradients:
            fn(*leaves, *caches, metadata, metadata).backward(gradient.to('neuron'))
        for index, leaf in enumerate(leaves):
            require_exact(case + ':' + name + f':accumulation{index}', leaf.grad,
                          (individual[name][0][index] + individual[name][1][index]).to(leaf.dtype))
    for leaf, original in zip(leaves, (x, up.float(), down.float())):
        assert torch.equal(leaf.detach().cpu(), original)
    if benchmark:
        device_gradient = gradients[0].to('neuron')

        def update(fn):
            for leaf in leaves: leaf.grad = None
            ub, db = leaves[1].detach().bfloat16(), leaves[2].detach().bfloat16()
            for _ in range(2):
                output = fn(*leaves, ub, db, metadata, metadata)
                output.backward(device_gradient)
            return output

        measured = timing({name: (lambda fn=fn: update(fn)) for name, fn in functions.items()},
                          steps, torch.neuron.synchronize)
        print('NATIVE_MLP_TOKEN_ROWS_BENCH ' + json.dumps(dict(shape=list(shape), **measured,
              units='ms per BF16 master cast/cache refresh plus two FW/BW microbatches',
              full_model_included=False, forward_precision='BF16 unchanged',
              all_numerical_and_accumulation_gates_passed=True)), flush=True)
    print('NATIVE_MLP_TOKEN_ROWS_CASE ' + json.dumps(dict(case=case, shape=list(shape), passed=True)), flush=True)
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--steps', type=int, default=15)
    args = parser.parse_args()
    import torch_neuronx
    torch.set_num_threads(4)
    print('NATIVE_MLP_TOKEN_ROWS_CONTRACT ' + json.dumps(dict(
        forward='FP64 real-valued ReLU-squared MLP plus propagated FP32 gamma_K/BF16 u=2^-8',
        backward='unchanged independent FP64 autograd and propagated BF16/FP32 bounds',
        paired_exactness='diagnostic; neither fitted nor used to replace the independent bound',
        accumulation='exact two-microbatch sum into FP32 masters')), flush=True)
    for case in ('random', 'zero_up', 'zero_down', 'outliers'):
        gate((512, 256, 512), case, args.steps, False)
    if args.real_shape:
        gate((16384, 1024, 4096), 'real_random', args.steps, True)
    print('NATIVE_MLP_TOKEN_ROWS_RESULT ' + json.dumps(dict(passed=True, real_shape=args.real_shape,
          training_or_full_model_promotion=False)), flush=True)


if __name__ == '__main__':
    main()
