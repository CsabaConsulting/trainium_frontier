#!/usr/bin/env python3
"""Profile one wave124-gated MLP arm with all cache work and saved gradients.

Wave124's same-host unprofiled comparison remains the timing authority. These
instrumented captures locate preparation, memory and engine costs; profiles
from different hosts are not a replacement for that controlled benchmark.
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

from fp8_complete_mlp_probe import build_functions, prepare_caches
from fp8_gradient_prepare_nki import build_kernels as build_gradient_prepare
from fp8_mlp_data_backward_nki import build_kernel as build_backward
from fp8_mlp_island_probe import make_inputs
from fp8_mlp_token_rows_nki import build_kernel as build_forward
from fp8_packed_weight_cache_nki import build_kernel as build_pack
from fp8_rms_scale_contract import normalize, policy
from fp8_weight_cache_nki import build_kernels as build_calibration
from fp8_weight_gradient_nki import build_kernels as build_weight_gradient
from preserve_native_trace_neffs import preserve_trace_neffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('native', 'full'), required=True)
    parser.add_argument('--trace-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    trace = args.trace_dir.resolve()
    trace.parent.mkdir(parents=True, exist_ok=True)
    # Previous six-arm component capture was500MB. A single native arm gets
    #384MiB headroom; the larger full FP8 arm gets512MiB, above the8GiB reserve.
    headroom = (384 if args.mode == 'native' else 512) * 1024**2
    assert shutil.disk_usage(trace.parent).free >= 8 * 1024**3 + headroom
    trace.mkdir(exist_ok=False)
    shape = (16384, 1024, 4096)
    x, up, down = make_inputs(shape, 'real_random', True)
    generator = torch.Generator().manual_seed(97151 + shape[0] + shape[1])
    xs = (x, normalize(torch.randn(x.shape, generator=generator).bfloat16()))
    gradients = [(torch.randn(x.shape, generator=generator) * .03125).bfloat16().to('neuron')
                 for _ in range(2)]
    inputs = [v.to('neuron').requires_grad_() for v in xs]
    masters = (up.float().to('neuron').requires_grad_(), down.float().to('neuron').requires_grad_())
    scale, bound = policy(shape[1])
    fixed = (torch.tensor([scale, 1.0 / scale, bound]).repeat(128, 1).to('neuron'),
             torch.ones(128, 3).to('neuron'))
    kernel, backward, pack = (wrap_nki(fn()) for fn in (build_forward, build_backward, build_pack))
    partial, finish = (wrap_nki(k) for k in build_gradient_prepare())
    product = wrap_nki(build_weight_gradient()[2])
    calibrate = wrap_nki(build_calibration()[0])
    _, complete, _, native = build_functions(shape[1], kernel, backward, partial, finish, product)
    function = native if args.mode == 'native' else complete

    def update(instrument=False):
        for leaf in (*inputs, *masters):
            leaf.grad = None
        with record_function(args.mode + '/current_master_cache_preparation'):
            caches = prepare_caches(masters, calibrate, pack, args.mode, fixed)
            if instrument:
                torch.neuron.synchronize()
        for index, (values, gradient) in enumerate(zip(inputs, gradients)):
            with record_function(args.mode + f'/micro{index}/forward'):
                output = function(values, *masters, *caches)
                if instrument:
                    torch.neuron.synchronize()
            with record_function(args.mode + f'/micro{index}/backward_and_accumulation'):
                output.backward(gradient)
                if instrument:
                    torch.neuron.synchronize()
        return output

    for _ in range(3):
        output = update()
    torch.neuron.synchronize()
    assert bool(torch.isfinite(output.cpu()).all())
    assert all(leaf.grad is not None and bool(torch.isfinite(leaf.grad.cpu()).all())
               for leaf in (*inputs, *masters))
    assert shutil.disk_usage(trace.parent).free >= 8 * 1024**3 + headroom
    config = NeuronConfig(modes=[ProfileMode.RUNTIME, ProfileMode.DEVICE],
                          profile_output_dir=str(trace), max_events_per_nc=1000000,
                          precache_node_info=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                 experimental_config=config) as prof:
        with record_function(args.mode + '/complete_component'):
            update(instrument=True)
    NeuronProfiler(config).export_trace(prof)
    from torch_neuronx.neuron_dynamo_backend.cache_artifact import ArtifactType
    from torch_neuronx.neuron_dynamo_backend.compile import get_compiled_cache_keys
    exported, missing = [], []
    destination = trace / 'neffs'
    destination.mkdir(exist_ok=True)
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
        exported.append(dict(cache_key=key, persistent_cache_key=persistent_key,
                             file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (destination / 'manifest.json').write_text(json.dumps(dict(exported=exported, missing=missing), indent=2) + '\n')
    eager = preserve_trace_neffs(trace, Path('/tmp/neff_cache'), destination)
    assert not missing and not eager['unresolved']
    print('FP8_COMPLETE_PROFILE_RESULT ' + json.dumps(dict(
        mode=args.mode, trace=str(trace), shape=list(shape), compiled_neffs=len(exported),
        eager_neffs=len(eager['files']), microbatches=2, two_different_inputs=True,
        current_caches_and_accumulation_included=True, phase_synchronization=True,
        numerical_and_timing_authority='fs-w124-fp8-complete-mlp-component-r1',
        profiled_latency_or_cross_host_comparison_is_speed_evidence=False,
        whole_model_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
