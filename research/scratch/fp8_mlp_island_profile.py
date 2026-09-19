#!/usr/bin/env python3
"""Attribute the already-gated complete FP8 MLP component, including preparation.

The wave73 probe is the numerical and unprofiled timing authority. This trace
adds synchronization at phase boundaries for attribution; its wall times must
not replace that benchmark. It exports all binaries before the process exits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_neuronx
from torch.profiler import ProfilerActivity, profile, record_function
from torch_neuronx import wrap_nki
from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

from fp8_mlp_island_nki import build_kernel
from fp8_mlp_island_probe import make_inputs
from fp8_mlp_island_autograd import make_autograd
from fp8_weight_cache_nki import build_kernels
from fp8_weight_cache_probe import make_cached_autograd
from mlp_fused_bf16 import build_kernel as build_bf16, reference_forward
from preserve_native_trace_neffs import preserve_trace_neffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--token-rows', action='store_true',
                        help='Also attribute the wave77-gated token-row layout')
    parser.add_argument('--rms-fixed-input', action='store_true')
    parser.add_argument('--token-row-backward', action='store_true')
    args = parser.parse_args()
    assert not (args.rms_fixed_input or args.token_row_backward) or args.token_rows
    torch.set_num_threads(4)
    trace = args.trace_dir.resolve()
    trace.mkdir(parents=True, exist_ok=False)
    shape = (16384, 1024, 4096)
    tile = 512
    x, up, down = make_inputs(shape, 'real_random', args.rms_fixed_input)
    generator = torch.Generator().manual_seed(9073 + shape[0] + shape[1])
    gradient = (torch.randn(x.shape, generator=generator) * .03125).bfloat16().to('neuron')
    leaves = (x.to('neuron').requires_grad_(), up.float().to('neuron').requires_grad_(),
              down.float().to('neuron').requires_grad_())
    island, bf16 = wrap_nki(build_kernel()), wrap_nki(build_bf16())
    calibrate, quantize = (wrap_nki(kernel) for kernel in build_kernels())

    def native_forward(xx, uu, dd, um, dm):
        result, relu = reference_forward(xx, uu, dd)
        return result, relu.T

    def bf16_forward(xx, uu, dd, um, dm):
        return bf16[2](xx, uu, dd, 2, tile, True, True, True, True, 0)

    def fp8_forward(xx, uu, dd, um, dm):
        return island[2](xx, uu, dd, um, dm, 2, tile, False)

    arms = {
        'native_bf16': (make_cached_autograd(native_forward), False),
        'nki_bf16_cache': (make_cached_autograd(bf16_forward), False),
        'nki_fp8_island': (make_autograd(fp8_forward), True),
    }
    if args.token_rows:
        from fp8_mlp_token_rows_nki import build_kernel as build_token_rows
        token_rows = wrap_nki(build_token_rows())
        fixed_args = ()
        if args.rms_fixed_input:
            from fp8_rms_scale_contract import policy
            fixed_args = (policy(shape[1])[0],)

        def token_rows_forward(xx, uu, dd, um, dm):
            return token_rows[2](xx, uu, dd, um, dm, 2, 128, False, *fixed_args)

        arms['nki_fp8_token_rows'] = (
            make_autograd(token_rows_forward, relu_token_major=True), True)
        if args.token_row_backward:
            arms['nki_fp8_token_row_backward'] = (
                make_autograd(token_rows_forward, relu_token_major=True,
                              token_row_backward=True), True)
    arms = {name: (torch.compile(function, backend='neuron', fullgraph=True, dynamic=False), q)
            for name, (function, q) in arms.items()}
    # Native/BF16 arms ignore these arguments, as in the gated timing probe.
    dummy_meta = torch.ones(128, 3, device='neuron')

    def update(name, instrument=False):
        function, quantized = arms[name]
        for leaf in leaves:
            leaf.grad = None
        with record_function(name + '/master_cast_and_cache_refresh'):
            ub, db = leaves[1].detach().bfloat16(), leaves[2].detach().bfloat16()
            um, dm = dummy_meta, dummy_meta
            if quantized:
                um, dm = calibrate[1](ub), calibrate[1](db)
                ub, db = quantize[2](ub, um, 2), quantize[2](db, dm, 2)
            if instrument:
                torch.neuron.synchronize()
        for index in range(2):
            with record_function(name + f'/microbatch{index}/forward'):
                output = function(*leaves, ub, db, um, dm)
                if instrument:
                    torch.neuron.synchronize()
            with record_function(name + f'/microbatch{index}/backward_and_accumulation'):
                output.backward(gradient)
                if instrument:
                    torch.neuron.synchronize()
        return output

    for name in arms:
        for _ in range(3):
            output = update(name)
        torch.neuron.synchronize()
        assert torch.isfinite(output.cpu()).all()
        assert all(torch.isfinite(leaf.grad.cpu()).all() for leaf in leaves)
    config = NeuronConfig(modes=[ProfileMode.RUNTIME, ProfileMode.DEVICE],
                          profile_output_dir=str(trace), max_events_per_nc=1000000,
                          precache_node_info=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                 experimental_config=config) as prof:
        for name in arms:
            with record_function(name + '/complete_update'):
                update(name, instrument=True)
    NeuronProfiler(config).export_trace(prof)
    from torch_neuronx.neuron_dynamo_backend.cache_artifact import ArtifactType
    from torch_neuronx.neuron_dynamo_backend.compile import get_compiled_cache_keys
    export = trace / 'neffs'
    export.mkdir(exist_ok=True)
    entries, missing = [], []
    for key in sorted(get_compiled_cache_keys(ArtifactType.NEURON_NEFF)):
        info = torch_neuronx._C.get_neff_info(key)
        if info is None:
            missing.append(key)
            continue
        persistent_key, data = info
        destination = export / (persistent_key + '.neff')
        if destination.exists():
            assert destination.read_bytes() == data
        else:
            destination.write_bytes(data)
        entries.append(dict(cache_key=key, persistent_cache_key=persistent_key,
                            file=destination.name, bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest()))
    (export / 'manifest.json').write_text(json.dumps(dict(exported=entries, missing=missing), indent=2) + '\n')
    eager = preserve_trace_neffs(trace, Path('/tmp/neff_cache'), export)
    assert not missing and not eager['unresolved']
    print('FP8_ISLAND_PROFILE_ARTIFACT ' + json.dumps(dict(
        trace=str(trace), shape=list(shape), arms=list(arms), microbatches=2,
        compiled_neffs=len(entries), eager_neffs=len(eager['files']),
        phase_boundary_synchronization=True, profiled_latency_is_speed_evidence=False,
        fixed_rms_input_scale=args.rms_fixed_input, token_row_backward=args.token_row_backward,
        numerical_authority=('fs-w81-fp8-token-row-backward-training-component-gate-r1'
                             if args.token_row_backward else
                             'fs-w79-fp8-rms-fixed-input-training-component-gate-r1'
                             if args.rms_fixed_input else
                             'fs-w77-fp8-token-rows-training-component-gate-r1'
                             if args.token_rows else
                             'fs-w73-fp8-causal-island-training-component-gate-r1'),
        whole_model_or_optimizer_included=False)), flush=True)


if __name__ == '__main__':
    main()
