#!/usr/bin/env python3
"""Locate the measured square FP8 dX loss without changing its arithmetic.

Independent source158 numerical evidence and byte identity are prerequisites.
Native device execution is the timed comparator; the independent CPU arithmetic
checks remain in that prerequisite. Cached weights deliberately exclude their
preparation. Profile intervals are attribution, not a full-step speed claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
import torch_neuronx
from torch.profiler import ProfilerActivity, profile, record_function
from torch_neuronx import wrap_nki
from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

from fp8_dual_weight_cache_nki import build_kernel as build_prepare
from fp8_gradient_gemm_nki import build_kernel as build_gradient
from fp8_square_dx_cost_split import prerequisite
from fp8_square_dx_probe import functions, inputs
from fp8_weight_cache_probe import require_exact
from mlp_fused_bf16_probe import timing
from preserve_native_trace_neffs import preserve_trace_neffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component-root', required=True)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    assert args.steps > 0
    prerequisite(args.component_root)
    torch.set_num_threads(4)
    trace = args.trace_dir.resolve()
    trace.parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(trace.parent).free >= 9 * 1024**3
    trace.mkdir(exist_ok=False)

    prepare, gradient = wrap_nki(build_prepare()), wrap_nki(build_gradient())
    original = functions(prepare, gradient)
    x, master, g = [v.to('neuron') for v in inputs(32768, 1024, 'random')]
    weight = master.bfloat16()
    _, metadata, _, packed = prepare[2](weight, 2, True)

    def native_dx(gg, ww):
        return gg @ ww

    def fp8_dx(gg, pp, mm):
        return gradient[2](gg, pp, mm, 2, False)

    ndx, qdx = [torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
                for fn in (native_dx, fp8_dx)]
    require_exact('profile:native_dx', ndx(g, weight), original['native'](x, master, g)[1])
    require_exact('profile:fp8_dx', qdx(g, packed, metadata), original['fp8_dx'](x, master, g)[1])
    calls = dict(native_dx=lambda: ndx(g, weight), fp8_dx=lambda: qdx(g, packed, metadata))
    saved = {name: fn().cpu() for name, fn in calls.items()}
    assert all(bool(torch.isfinite(v).all()) for v in saved.values())
    measured = timing(calls, args.steps, torch.neuron.synchronize)
    print('FP8_DX_PROFILE_TIMING ' + json.dumps(dict(
        **measured, cached_weights=True, preparation_excluded=True,
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
    print('FP8_DX_PROFILE_RESULT ' + json.dumps(dict(
        passed=True, trace=str(trace), tokens=32768, channels=1024,
        arms=list(calls), compiled_neffs=len(entries), eager_neffs=len(eager['files']),
        numerical_controls_exact=True, phase_synchronization=True,
        cached_weights=True, preparation_excluded=True,
        numerical_authority='fs-w158-fp8-square-dx-component-r1',
        profiled_latency_is_speed_evidence=False,
        whole_model_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
