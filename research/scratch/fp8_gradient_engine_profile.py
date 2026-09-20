#!/usr/bin/env python3
"""Profile the gated source161 square dX fusion against both original controls.

Independent source158 and source161 numerical evidence and byte identity are prerequisites.
Native device execution is the timed comparator; the independent CPU arithmetic
checks remain in that prerequisite. Cached weights deliberately exclude their
preparation. Profile intervals are attribution, not a full-step speed claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
import torch_neuronx
from torch.profiler import ProfilerActivity, profile, record_function
from torch_neuronx import wrap_nki
from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

from fp8_dual_weight_cache_nki import build_kernel as build_prepare
from fp8_gradient_gemm_nki import build_kernel as build_gradient
from fp8_gradient_gemm_engine_nki import build_kernel as build_engine
from fp8_gradient_engine_probe import ConfiguredGradient
from fp8_square_dx_cost_split import prerequisite
from fp8_square_dx_probe import functions, inputs
from fp8_weight_cache_probe import require_exact
from mlp_fused_bf16_probe import timing
from preserve_native_trace_neffs import preserve_trace_neffs


def engine_prerequisite(root):
    root = Path(root)
    label = 'fs-w161-fp8-gradient-both-r1'
    post_path = root / 'research/run-logs' / (label + '.postflight.json')
    raw_path = root / 'research/run-logs' / (label + '.log')
    post = json.loads(post_path.read_text())
    assert post['source_commit'] == '252880f3bd3fea062a68df3451876e0a77321275'
    assert post['label'] == label and post['host_clean'] and post['command_returncode'] == 0
    assert not post['failures'] and not post['guard_error'] and not post['interrupted_signals']
    records = {}
    for line in raw_path.read_text().splitlines():
        if line.startswith(('FP8_GRADIENT_GEMM_GATE ', 'FP8_SQUARE_DX_CASE ',
                            'SERIAL_QKV_PRODUCT_GATE ', 'FP8_DX_ENGINE_RESULT ')):
            marker, payload = line.split(' ', 1)
            records.setdefault(marker, []).append(json.loads(payload))
    result, = records['FP8_DX_ENGINE_RESULT']
    assert result['passed'] and result['variant'] == 'both' and result['cases'] == 15
    assert result['domain_cases'] == 7 and result['square_cases'] == 8
    assert result['square_outputs_exact_old_fp8'] and result['independent_declared_fp8_gates']
    domain = records['FP8_GRADIENT_GEMM_GATE']
    assert len(domain) == 7 and all(x['passed'] for x in domain)
    assert {x['case'] for x in domain} == {'random', 'zero_gradient', 'zero_weight',
            'tiny_gradient', 'row_scale_range', 'outliers', 'signed_cancellation'}
    square = records['FP8_SQUARE_DX_CASE']
    assert len(square) == 8 and all(x['passed'] for x in square)
    assert {x['case'] for x in square} == {'random', 'zero_gradient', 'zero_weight',
            'tiny_gradient', 'outliers', 'changed_weight', 'cancellation', 'real_random'}
    products = records['SERIAL_QKV_PRODUCT_GATE']
    assert len(products) == 40 and all(x['passed'] and not x['bound_failures'] for x in products)
    current, hashes = Path(__file__).resolve().parent, {}
    for name in ('fp8_gradient_gemm_engine_nki.py', 'fp8_gradient_engine_probe.py',
                 'fp8_gradient_gemm_probe.py', 'fp8_packed_weight_cache_nki.py',
                 'fp8_weight_cache_nki.py', 'fp8_square_dx_probe.py',
                 'fp8_dual_weight_cache_nki.py', 'fp8_gradient_gemm_nki.py',
                 'fp8_mlp_island_probe.py', 'fp8_packed_weight_cache_probe.py',
                 'fp8_weight_cache_probe.py', 'fresh_start_fp8_probe.py',
                 'mlp_fused_bf16_probe.py', 'serial_qkv_probe.py'):
        a, b = (current / name).read_bytes(), (root / 'research/scratch' / name).read_bytes()
        assert a == b, 'Source changed after independent gate: ' + name
        hashes[name] = hashlib.sha256(a).hexdigest()
    print('FP8_ENGINE_PROFILE_PREREQUISITE ' + json.dumps(dict(passed=True,
          source_commit=post['source_commit'], variant='both', cases=15,
          postflight_sha256=hashlib.sha256(post_path.read_bytes()).hexdigest(),
          log_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(), source_sha256=hashes)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component-root', required=True)
    parser.add_argument('--engine-root', required=True)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--dge-notifications', choices=('on', 'off'), default='on')
    args = parser.parse_args()
    assert args.steps > 0
    prerequisite(args.component_root)
    engine_prerequisite(args.engine_root)
    dge_enabled = args.dge_notifications == 'on'
    assert (os.environ.get('NEURON_RT_ENABLE_DGE_NOTIFICATIONS') == '1') == dge_enabled
    torch.set_num_threads(4)
    trace = args.trace_dir.resolve()
    trace.parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(trace.parent).free >= 9 * 1024**3
    trace.mkdir(exist_ok=False)

    prepare, gradient = wrap_nki(build_prepare()), wrap_nki(build_gradient())
    original = functions(prepare, gradient)
    candidate = ConfiguredGradient(wrap_nki(build_engine()), 'both')
    x, master, g = [v.to('neuron') for v in inputs(32768, 1024, 'random')]
    weight = master.bfloat16()
    _, metadata, _, packed = prepare[2](weight, 2, True)

    def native_dx(gg, ww):
        return gg @ ww

    def fp8_dx(gg, pp, mm):
        return gradient[2](gg, pp, mm, 2, False)

    def candidate_dx(gg, pp, mm):
        return candidate[2](gg, pp, mm, 2, False)

    ndx, qdx, cdx = [torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
                for fn in (native_dx, fp8_dx, candidate_dx)]
    require_exact('profile:native_dx', ndx(g, weight), original['native'](x, master, g)[1])
    require_exact('profile:fp8_dx', qdx(g, packed, metadata), original['fp8_dx'](x, master, g)[1])
    require_exact('profile:candidate_dx', cdx(g, packed, metadata), qdx(g, packed, metadata))
    calls = dict(native_dx=lambda: ndx(g, weight), old_fp8_dx=lambda: qdx(g, packed, metadata),
                 candidate_fp8_dx=lambda: cdx(g, packed, metadata))
    saved = {name: fn().cpu() for name, fn in calls.items()}
    assert all(bool(torch.isfinite(v).all()) for v in saved.values())
    measured = timing(calls, args.steps, torch.neuron.synchronize)
    print('FP8_ENGINE_PROFILE_TIMING ' + json.dumps(dict(
        **measured, cached_weights=True, preparation_excluded=True,
        dge_notifications_enabled=dge_enabled, performance_authority='fs-w161-fp8-gradient-both-r1',
        timing_under_instrumentation_is_speed_evidence=False,
        whole_model_or_quality_claim=False)), flush=True)
    torch.neuron.synchronize()
    assert shutil.disk_usage(trace.parent).free >= 9 * 1024**3
    config = NeuronConfig(modes=[ProfileMode.RUNTIME, ProfileMode.DEVICE],
                          profile_output_dir=str(trace), max_events_per_nc=1000000,
                          precache_node_info=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                 experimental_config=config) as prof:
        for name, fn in calls.items():
            with record_function(name + '/cached_product'):
                fn()
                torch.neuron.synchronize()
    NeuronProfiler(config).export_trace(prof)
    for name, fn in calls.items():
        require_exact('profile:repeat:' + name, fn(), saved[name])

    from torch_neuronx.neuron_dynamo_backend.cache_artifact import ArtifactType
    from torch_neuronx.neuron_dynamo_backend.compile import get_compiled_cache_keys
    destination = trace / 'neffs'
    destination.mkdir(exist_ok=True)
    entries, missing = [], []
    for key in sorted(get_compiled_cache_keys(ArtifactType.NEURON_NEFF)):
        info = torch_neuronx._C.get_neff_info(key)
        if info is None:
            missing.append(key)
            continue
        persistent_key, data = info
        path = destination / (persistent_key + '.neff')
        if path.exists():
            assert path.read_bytes() == data
        else:
            path.write_bytes(data)
        entries.append(dict(cache_key=key, persistent_cache_key=persistent_key,
                            file=path.name, bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest()))
    (destination / 'manifest.json').write_text(json.dumps(
        dict(exported=entries, missing=missing), indent=2) + '\n')
    eager = preserve_trace_neffs(trace, Path('/tmp/neff_cache'), destination)
    assert not missing and not eager['unresolved']
    print('FP8_ENGINE_PROFILE_RESULT ' + json.dumps(dict(
        passed=True, trace=str(trace), tokens=32768, channels=1024,
        arms=list(calls), compiled_neffs=len(entries), eager_neffs=len(eager['files']),
        numerical_controls_exact=True, phase_synchronization=True,
        cached_weights=True, preparation_excluded=True,
        numerical_authority=['fs-w158-fp8-square-dx-component-r1', 'fs-w161-fp8-gradient-both-r1'],
        variant='both', dge_notifications_enabled=dge_enabled,
        profiled_latency_is_speed_evidence=False,
        whole_model_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
