#!/usr/bin/env python3
"""Capture one native and one fused MLP forward after unprofiled warmup.

The small shape already reproduces the large regression and avoids creating
an unnecessarily large first diagnostic trace. Trace latency is attribution
evidence, not a replacement for the unprofiled paired benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_neuronx
from torch.profiler import ProfilerActivity, profile, record_function
from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

from mlp_fused_bf16 import build_kernel, reference_forward
from preserve_native_trace_neffs import preserve_trace_neffs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace-dir', type=Path, required=True)
    ap.add_argument('--contiguous-weights', action='store_true')
    ap.add_argument('--tiled-io', action='store_true')
    ap.add_argument('--cached-bf16-weights', action='store_true')
    ap.add_argument('--real-shape', action='store_true')
    ap.add_argument('--transposed-relu', action='store_true')
    ap.add_argument('--grouped-io', action='store_true')
    args = ap.parse_args()
    if args.grouped_io and not (args.contiguous_weights and args.tiled_io
                               and args.cached_bf16_weights and args.transposed_relu):
        ap.error('--grouped-io requires contiguous/tiled I/O, BF16 cached weights and transposed ReLU')
    torch.set_num_threads(4)
    trace = args.trace_dir.resolve()
    trace.mkdir(parents=True, exist_ok=False)
    config = NeuronConfig(modes=[ProfileMode.RUNTIME, ProfileMode.DEVICE],
                          profile_output_dir=str(trace), max_events_per_nc=1000000,
                          precache_node_info=True)
    torch.manual_seed(941)
    tokens, channels, hidden = (8192, 1024, 4096) if args.real_shape else (256, 256, 512)
    tile = min(512, tokens // 2)
    weight_dtype = torch.bfloat16 if args.cached_bf16_weights else torch.float32
    x = torch.randn(tokens, channels).bfloat16().to('neuron')
    up = (torch.randn(hidden, channels)/channels**0.5).to(dtype=weight_dtype,device='neuron')
    down = (torch.randn(channels, hidden)/hidden**0.5).to(dtype=weight_dtype,device='neuron')
    wrapped = torch_neuronx.wrap_nki(build_kernel())

    def call(xx, uu, dd):
        return wrapped[2](xx, uu, dd, 2, tile, args.contiguous_weights, args.tiled_io,
                           args.transposed_relu, args.grouped_io)

    fused = torch.compile(call, backend='neuron', fullgraph=True, dynamic=False)
    native = torch.compile(reference_forward, backend='neuron', fullgraph=True, dynamic=False)
    for _ in range(3):
        fused(x, up, down)
        native(x, up, down)
    torch.neuron.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                 experimental_config=config) as prof:
        with record_function('native_mlp_saved_relu'):
            native(x, up, down)
            torch.neuron.synchronize()
        with record_function('fused_mlp_saved_relu'):
            fused(x, up, down)
            torch.neuron.synchronize()
    NeuronProfiler(config).export_trace(prof)
    from torch_neuronx.neuron_dynamo_backend.cache_artifact import ArtifactType
    from torch_neuronx.neuron_dynamo_backend.compile import get_compiled_cache_keys
    export = trace/'neffs'
    export.mkdir(exist_ok=True)
    entries, missing = [], []
    for key in sorted(get_compiled_cache_keys(ArtifactType.NEURON_NEFF)):
        info = torch_neuronx._C.get_neff_info(key)
        if info is None:
            missing.append(key)
            continue
        persistent_key, data = info
        destination = export/(persistent_key+'.neff')
        if destination.exists():
            assert destination.read_bytes() == data
        else:
            destination.write_bytes(data)
        entries.append(dict(cache_key=key, persistent_cache_key=persistent_key,
                            file=destination.name, bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest()))
    (export/'manifest.json').write_text(json.dumps(dict(exported=entries,missing=missing),indent=2)+'\n')
    eager = preserve_trace_neffs(trace, Path('/tmp/neff_cache'), export)
    assert not missing and not eager['unresolved']
    print('MLP_PROFILE_ARTIFACT '+json.dumps(dict(trace=str(trace),shape=[tokens,channels,hidden],
          contiguous_weights=args.contiguous_weights, tiled_io=args.tiled_io,
          cached_bf16_weights=args.cached_bf16_weights,
          transposed_relu=args.transposed_relu,
          grouped_io=args.grouped_io,
          compiled_neffs=len(entries),eager_neffs=len(eager['files']),
          profiled_latency_is_performance_evidence=False)),flush=True)


if __name__=='__main__':
    main()
