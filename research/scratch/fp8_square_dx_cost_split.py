#!/usr/bin/env python3
"""Distinguish square FP8 dX arithmetic cost from current cache preparation.

Requires the completed independent source158 screen with unchanged code bytes.
Cached calls are optimistic cost controls, not a claim that preparation is free.
Separate measurements need not add because compiler scope and overlap differ.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fp8_dual_weight_cache_nki import build_kernel as build_prepare
from fp8_gradient_gemm_nki import build_kernel as build_gradient
from fp8_square_dx_probe import functions, inputs
from fp8_weight_cache_probe import require_exact
from mlp_fused_bf16_probe import timing


def prerequisite(root):
    root = Path(root)
    label = 'fs-w158-fp8-square-dx-component-r1'
    post_path = root / 'research/run-logs' / (label + '.postflight.json')
    raw_path = root / 'research/run-logs' / (label + '.log')
    post = json.loads(post_path.read_text())
    assert post['source_commit'] == '959495b462a824904e2f7d91a136da74f4ba94c0'
    assert post['label'] == label and post['host_clean'] and post['command_returncode'] == 0
    assert not post['failures'] and not post['guard_error'] and not post['interrupted_signals']
    raw = raw_path.read_text()
    records = {}
    for line in raw.splitlines():
        if line.startswith(('FP8_SQUARE_DX_', 'SERIAL_QKV_PRODUCT_GATE ')):
            marker, payload = line.split(' ', 1)
            records.setdefault(marker, []).append(json.loads(payload))
    final, = records['FP8_SQUARE_DX_RESULT']
    assert final['passed'] and final['cases'] == 8 and final['real_tokens'] == 32768
    assert final['real_channels'] == 1024 and final['native_forward_and_weight_product_exact']
    cases = records['FP8_SQUARE_DX_CASE']
    assert len(cases) == 8 and all(c['passed'] for c in cases)
    assert {c['case'] for c in cases} == {'random', 'zero_gradient', 'zero_weight',
           'tiny_gradient', 'outliers', 'changed_weight', 'cancellation', 'real_random'}
    products = records['SERIAL_QKV_PRODUCT_GATE']
    assert len(products) == 40 and all(c['passed'] and not c['bound_failures'] for c in products)
    current = Path(__file__).resolve().parent
    hashes = {}
    for name in ('fp8_square_dx_probe.py', 'fp8_dual_weight_cache_nki.py', 'fp8_gradient_gemm_nki.py',
                 'fp8_mlp_island_probe.py', 'fp8_packed_weight_cache_probe.py',
                 'fp8_weight_cache_probe.py', 'fresh_start_fp8_probe.py',
                 'mlp_fused_bf16_probe.py', 'serial_qkv_probe.py'):
        a, b = (current / name).read_bytes(), (root / 'research/scratch' / name).read_bytes()
        assert a == b, 'Source changed after independent gate: ' + name
        hashes[name] = hashlib.sha256(a).hexdigest()
    print('FP8_DX_COST_PREREQUISITE ' + json.dumps(dict(passed=True,
          source_commit=post['source_commit'], postflight_sha256=hashlib.sha256(post_path.read_bytes()).hexdigest(),
          log_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(), source_sha256=hashes)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component-root', required=True)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    assert args.steps > 0
    prerequisite(args.component_root)
    import torch_neuronx  # noqa: F401
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    prepare, gradient = wrap_nki(build_prepare()), wrap_nki(build_gradient())
    original = functions(prepare, gradient)
    x, master, g = [v.to('neuron') for v in inputs(32768, 1024, 'random')]
    bf16_weight = master.bfloat16()
    _, meta, _, packed = prepare[2](bf16_weight, 2, True)

    def cached_three(xx, ww, gg, pp, mm):
        dx = gradient[2](gg, pp, mm, 2, False)
        return F.linear(xx, ww), dx, (gg.T @ xx).float()

    def native_dx(gg, ww):
        return gg @ ww

    def fp8_dx(gg, pp, mm):
        return gradient[2](gg, pp, mm, 2, False)

    def current_prepare(weight):
        return prepare[2](weight.bfloat16(), 2, True)

    compile_call = lambda fn: torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
    cached, ndx, qdx, prep = map(compile_call, (cached_three, native_dx, fp8_dx, current_prepare))
    native = original['native'](x, master, g)
    candidate = original['fp8_dx'](x, master, g)
    optimistic = cached(x, bf16_weight, g, packed, meta)
    for i in range(3):
        require_exact(f'cost_split:cached_product{i}', optimistic[i], candidate[i])
    require_exact('cost_split:native_dx', ndx(g, bf16_weight), native[1])
    require_exact('cost_split:fp8_dx', qdx(g, packed, meta), candidate[1])
    prepared = prep(master)
    require_exact('cost_split:metadata', prepared[1], meta)
    require_exact('cost_split:backward_packing', prepared[3], packed)
    calls = {
        'native_three_fresh': lambda: original['native'](x, master, g),
        'fp8_three_fresh': lambda: original['fp8_dx'](x, master, g),
        'fp8_three_cached_optimistic': lambda: cached(x, bf16_weight, g, packed, meta),
        'native_dx_cached': lambda: ndx(g, bf16_weight),
        'fp8_dx_cached': lambda: qdx(g, packed, meta),
        'current_dual_prepare': lambda: prep(master),
    }
    result = timing(calls, args.steps, torch.neuron.synchronize)
    assert all(torch.isfinite(torch.tensor(values)).all() and min(values) > 0
               for values in result['samples_ms'].values())
    print('FP8_DX_COST_RESULT ' + json.dumps(dict(passed=True, tokens=32768, channels=1024,
          numerical_controls_exact=True, cases=len(calls), **result,
          cached_controls_exclude_preparation=True, separate_times_not_additive=True,
          autograd_or_model_gate=False, whole_model_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
