#!/usr/bin/env python3
"""Gate forward/backward scale fusion against the unchanged complete FP8 MLP.

Independent original stages remain the authority. The new compact forward and
actual autograd consumer must retain identical values, saves and gradients.
Complete-component timing charges current caches and FP32 master gradients.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from fp8_complete_mlp_autograd import make_autograd
from fp8_complete_mlp_probe import build_functions, run
from fp8_full_mlp_integration import make_ops
from fp8_gradient_prepare_nki import build_kernels as build_gradient
from fp8_mlp_data_backward_engine_nki import build_kernel as build_new_backward
from fp8_mlp_data_backward_nki import build_kernel as build_backward
from fp8_mlp_forward_engine_nki import build_kernel as build_new_forward
from fp8_mlp_island_probe import make_inputs
from fp8_mlp_token_rows_nki import build_kernel as build_forward
from fp8_packed_weight_cache_nki import build_kernel as build_pack
from fp8_rms_scale_contract import policy
from fp8_weight_cache_nki import build_kernels as build_calibration
from fp8_weight_cache_probe import require_exact
from fp8_weight_gradient_nki import build_kernels as build_weight_gradient
from mlp_backward_rounding_bounds import reference_and_bounds, audit
from mlp_fused_bf16_probe import timing


def prerequisite(root):
    root = Path(root)
    label = 'fs-w163-fp8-mlp-scale-component-r1'
    p = root / 'research/run-logs' / (label + '.postflight.json')
    raw = root / 'research/run-logs' / (label + '.log')
    post = json.loads(p.read_text())
    assert post['source_commit'] == '1f3890c91650c5ea36dae5cf996d8dcc1964df16'
    assert post['host_clean'] and post['command_returncode'] == 0 and post['label'] == label
    assert not post['failures'] and not post['guard_error'] and not post['interrupted_signals']
    records = {}
    for line in raw.read_text().splitlines():
        if line.startswith(('FP8_MLP_ENGINE_RESULT ', 'FP8_DATA_BACKWARD_STAGE_GATE ',
                            'FP8_DATA_BACKWARD_NATIVE_BOUND ')):
            marker, value = line.split(' ', 1)
            records.setdefault(marker, []).append(json.loads(value))
    result, = records['FP8_MLP_ENGINE_RESULT']
    assert result['passed'] and result['cases'] == 12 and result['production_tokens'] == 32768
    assert result['all_diagnostic_and_normal_outputs_exact_original']
    stages = records['FP8_DATA_BACKWARD_STAGE_GATE']
    assert len(stages) == 12 and all(x['passed'] and x['original_algorithm_exact'] for x in stages)
    assert {x['case'] for x in stages} == {'random', 'zero_gradient', 'zero_relu', 'zero_down',
            'zero_up', 'tiny_gradient', 'subnormal_gradient', 'signed_zero_gradient',
            'row_scale_range', 'outliers', 'real_random'}
    assert {tuple(x['shape']) for x in stages if x['case'] == 'real_random'} == {
        (16384, 1024, 4096), (32768, 1024, 4096)}
    bounds = records['FP8_DATA_BACKWARD_NATIVE_BOUND']
    assert len(bounds) == 2 and {x['field'] for x in bounds} == {'dx', 'dpre'}
    assert all(x['passed'] and x['finite'] and not x['bound_failures'] for x in bounds)
    current, hashes = Path(__file__).resolve().parent, {}
    for name in ('fp8_mlp_data_backward_engine_nki.py', 'fp8_mlp_data_backward_engine_probe.py',
                 'fp8_mlp_data_backward_nki.py', 'fp8_mlp_data_backward_probe.py',
                 'fp8_gradient_gemm_nki.py', 'fp8_mlp_island_probe.py',
                 'fp8_packed_weight_cache_nki.py', 'fp8_packed_weight_cache_probe.py',
                 'fp8_weight_cache_nki.py', 'fp8_weight_cache_probe.py',
                 'mlp_backward_rounding_bounds.py', 'mlp_fused_bf16_probe.py'):
        a, b = (current / name).read_bytes(), (root / 'research/scratch' / name).read_bytes()
        assert a == b, 'Changed after independent gate: ' + name
        hashes[name] = hashlib.sha256(a).hexdigest()
    print('FP8_MLP_ENGINE_COMPONENT_PREREQUISITE ' + json.dumps(dict(
        passed=True, cases=12, source_commit=post['source_commit'], source_sha256=hashes,
        postflight_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        log_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())), flush=True)


def applications(channels, old_forward, old_backward, new_forward, new_backward,
                 partial, finish, product):
    legacy, _, mixed, native = build_functions(
        channels, old_forward, old_backward, partial, finish, product)
    scale = policy(channels)[0]

    def compact(x, u, d, um, dm, pu, pd):
        return new_forward[2](x, u, d, um, dm, 2, 128, False, scale, pu, pd,
                              True, False, True, False)

    def paired_forward(x, u, d, um, dm, pu, pd):
        original = legacy(x, u, d, um, dm, pu, pd)
        candidate = compact(x, u, d, um, dm, pu, pd)
        repeated = compact(x, u, d, um, dm, pu, pd)
        assert len(original) == 7 and len(candidate) == len(repeated) == 5
        for i, index in enumerate((0, 1, 4, 5, 6)):
            require_exact('engine:retained_forward' + str(i), candidate[i], original[index])
            require_exact('engine:repeat_forward' + str(i), repeated[i], candidate[i])
        print('FP8_MLP_ENGINE_FORWARD_GATE ' + json.dumps(dict(
            passed=True, tokens=x.shape[0], channels=channels,
            retained_outputs_exact_original=True, repeat_exact=True)), flush=True)
        return original

    apply = make_autograd(compact, new_backward, partial, finish, product,
                          compact_forward_saves=True)
    complete = torch.compile(apply, backend='neuron', fullgraph=True, dynamic=False)
    return paired_forward, complete, mixed, native


def benchmark(native, steps):
    shape = (32768, 1024, 4096)
    x, up, down = make_inputs(shape, 'real_random', True)
    generator = torch.Generator().manual_seed(165071)
    gradient = (torch.randn(x.shape, generator=generator) * .03125).bfloat16()
    values = x.to('neuron').requires_grad_()
    masters = [w.float().to('neuron').requires_grad_() for w in (up, down)]
    grad = gradient.to('neuron')
    scale, bound = policy(1024)
    fixed = (torch.tensor([scale, 1.0 / scale, bound]).repeat(128, 1).to('neuron'),
             torch.ones(128, 3).to('neuron'))
    old_prepare, old_apply = make_ops(compact_forward_saves=True)
    new_prepare, new_apply = make_ops(compact_forward_saves=True, engine_scaling='all')
    old_complete, new_complete = [torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
                                  for fn in (old_apply, new_apply)]

    def update(mode):
        for leaf in (values, *masters):
            leaf.grad = None
        if mode == 'native':
            caches = (masters[0].detach().bfloat16(), masters[1].detach().bfloat16(), *fixed)
            output = native(values, *masters, *caches)
        else:
            prepare, apply = ((old_prepare, old_complete) if mode == 'old'
                              else (new_prepare, new_complete))
            u, um, uf, ub = prepare(masters[0])
            d, dm, df, db = prepare(masters[1])
            output = apply(values, *masters, u, d, um, dm, uf, df, ub, db, *fixed)
        output.backward(grad)
        return output

    update('native')
    truth, bounds = reference_and_bounds(x, up, down, gradient)
    audit('engine:native_complete_timing', tuple(v.grad for v in (values, *masters)), truth, bounds)
    del truth, bounds
    old_y = update('old').detach().cpu()
    old_grads = [v.grad.cpu().clone() for v in (values, *masters)]
    new_y = update('new')
    require_exact('engine:timing_forward', new_y, old_y)
    for field, leaf, expected in zip(('dx', 'dup', 'ddown'), (values, *masters), old_grads):
        require_exact('engine:timing_' + field, leaf.grad, expected)
    calls = {name: (lambda mode=mode: update(mode)) for name, mode in (
        ('native_bf16_complete', 'native'), ('old_fp8_complete_compact', 'old'),
        ('scale_fused_fp8_complete_compact', 'new'))}
    measured = timing(calls, steps, torch.neuron.synchronize)
    assert all(bool(torch.isfinite(torch.tensor(v)).all()) and min(v) > 0
               for v in measured['samples_ms'].values())
    return dict(**measured, production_tokens=32768, microbatches_per_update=1,
                all_current_master_casts_calibration_and_dual_packing_charged=True,
                all_forward_and_four_backward_products_charged=True,
                fp32_master_gradients_charged=True, exact_old_complete_outputs_and_gradients=True,
                surrounding_model_optimizer_collectives_excluded=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component-root', required=True)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    assert args.steps > 0
    prerequisite(args.component_root)
    import torch_neuronx  # noqa: F401
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    old_forward, old_backward, new_forward, new_backward, pack = [wrap_nki(b()) for b in
        (build_forward, build_backward, build_new_forward, build_new_backward, build_pack)]
    partial, finish = [wrap_nki(k) for k in build_gradient()]
    _, byte_pack, product = [wrap_nki(k) for k in build_weight_gradient()]
    calibrate, quantize = [wrap_nki(k) for k in build_calibration()]
    kernels = old_forward, old_backward, partial, finish, product, calibrate, quantize, pack, byte_pack
    small = applications(256, old_forward, old_backward, new_forward, new_backward,
                         partial, finish, product)
    cases = ('random', 'zero_input', 'zero_up', 'zero_down', 'zero_gradient',
             'tiny_gradient', 'outliers', 'changed_weights', 'norm_spikes', 'norm_tiny')
    for case in cases:
        run((512, 256, 512), case, kernels, small, 1, False)
    torch._dynamo.reset()
    large = applications(1024, old_forward, old_backward, new_forward, new_backward,
                         partial, finish, product)
    run((32768, 1024, 4096), 'real_random', kernels, large, 1, False)
    torch._dynamo.reset()
    measured = benchmark(large[3], args.steps)
    print('FP8_MLP_ENGINE_COMPLETE_RESULT ' + json.dumps(dict(
        passed=True, cases=11, engine_scaling="all", original_independent_forward_and_four_product_gates=True,
        exact_retained_forward_tensors_and_compiled_gradients=True,
        cache_updates_and_fp32_accumulation=True,
        **measured, whole_model_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
