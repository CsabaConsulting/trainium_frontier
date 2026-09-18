#!/usr/bin/env python3
"""Capture the small-shape fused-embedding transfer regression on device.

The component's full numerical suite is a separate prerequisite. This records
one warmed native and fused backward, with no metadata work hidden in a speed
claim; profile times are diagnostic only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_neuronx
from torch.profiler import ProfilerActivity, profile, record_function
from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

from embedding_fused_dma import build_kernel
from embedding_segmented_dma import dma_metadata
from preserve_native_trace_neffs import preserve_trace_neffs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace-dir', type=Path, required=True)
    ap.add_argument('--contiguous-metadata', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    trace = args.trace_dir.resolve()
    trace.mkdir(parents=True, exist_ok=False)
    config = NeuronConfig(modes=[ProfileMode.RUNTIME, ProfileMode.DEVICE],
                          profile_output_dir=str(trace), max_events_per_nc=1000000,
                          precache_node_info=True)
    torch.manual_seed(731)
    tokens = channels = vocabulary = 256
    ids = torch.randint(vocabulary, (tokens,), dtype=torch.int32)
    gradient = (torch.randn(tokens, channels) * .01).to('neuron')
    permutation, continuation, mapping = dma_metadata(ids, vocabulary)
    if args.contiguous_metadata:
        mapping = mapping.T.contiguous()
    metadata = tuple(t.to('neuron') for t in (permutation, continuation, mapping))
    device_ids = ids.to('neuron')
    kernel = torch_neuronx.wrap_nki(build_kernel())

    def call(g, p, c, m):
        return kernel[2](g, p, c, m, vocabulary, 2, False, args.contiguous_metadata)

    def ordinary(g, i):
        return torch.ops.aten.embedding_dense_backward.default(g, i, vocabulary, -1, False)

    fused = torch.compile(call, backend='neuron', fullgraph=True, dynamic=False)
    native = torch.compile(ordinary, backend='neuron', fullgraph=True, dynamic=False)
    for _ in range(3):
        actual, expected = fused(gradient, *metadata), native(gradient, device_ids)
    torch.neuron.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-7)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                 experimental_config=config) as prof:
        with record_function('native_embedding_backward'):
            native(gradient, device_ids)
            torch.neuron.synchronize()
        with record_function('fused_embedding_backward'):
            fused(gradient, *metadata)
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
    print('EMBED_PROFILE_ARTIFACT '+json.dumps(dict(trace=str(trace),
          shape=[tokens,channels,vocabulary], compiled_neffs=len(entries),
          contiguous_metadata=args.contiguous_metadata,
          eager_neffs=len(eager['files']), profiled_latency_is_performance_evidence=False)),flush=True)


if __name__=='__main__':
    main()
