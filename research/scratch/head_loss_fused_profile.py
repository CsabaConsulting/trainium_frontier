#!/usr/bin/env python3
"""Attribute the gated head/CE forward and its charged weight transpose.

Wave101 independent numerical gates and unprofiled timing are the authority.
This synchronized device capture diagnoses the26.79% forward-only regression;
it is not a whole-model benchmark or a backward/autograd implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_neuronx
from torch.profiler import ProfilerActivity, profile, record_function
from torch_neuronx import wrap_nki
from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

from head_loss_fused_nki import build_kernel
from softcap_ce_parallel_nki import build_kernels
from preserve_native_trace_neffs import preserve_trace_neffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    trace = args.trace_dir.resolve()
    trace.mkdir(parents=True, exist_ok=False)
    shape = (16384, 1024, 8192)
    generator = torch.Generator().manual_seed(4917)
    x = torch.randn(shape[0], shape[1], generator=generator).bfloat16().to('neuron')
    w = (torch.randn(shape[2], shape[1], generator=generator) * .03).bfloat16().to('neuron')
    target = torch.randint(shape[2], (shape[0], 1), generator=generator).float().to('neuron')
    fused = wrap_nki(build_kernel())
    ce, _ = (wrap_nki(kernel) for kernel in build_kernels())

    def native(xx, ww, yy):
        z = F.linear(xx, ww).float()
        loss, lse = ce[2](z, yy, 2, True)
        return z, loss, lse

    def candidate(xx, wt, yy):
        return fused[2](xx, wt, yy, 2)

    arms = {name: torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
            for name, fn in (('native_head', native), ('fused_head', candidate))}
    with torch.no_grad():
        wt = w.T.contiguous()
        original = arms['native_head'](x, w, target)
        result = arms['fused_head'](x, wt, target)
        exact, finite = [], []
        for a, b in zip(original, result):
            ac, bc = a.cpu(), b.cpu()
            exact.append(torch.equal(ac, bc))
            finite.append(bool(torch.isfinite(ac).all() and torch.isfinite(bc).all()))
        assert all(finite)
        frozen = ce[2](result[0], target, 2, True)
        assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(result[1:], frozen))
        print('FUSED_HEAD_PROFILE_BOUNDARY ' + json.dumps(dict(
            exact_native_logits_loss_lse_diagnostic=exact, finite=finite,
            exact_frozen_logit_loss_lse=True,
            numerical_authority='fs-w101-fused-head-loss-forward-component-r1')), flush=True)
        del original, result, frozen
        for _ in range(3):
            arms['native_head'](x, w, target)
            wt = w.T.contiguous()
            arms['fused_head'](x, wt, target)
        torch.neuron.synchronize()
        config = NeuronConfig(modes=[ProfileMode.RUNTIME, ProfileMode.DEVICE],
                              profile_output_dir=str(trace), max_events_per_nc=1000000,
                              precache_node_info=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                     experimental_config=config) as prof:
            with record_function('native_head/forward'):
                arms['native_head'](x, w, target)
                torch.neuron.synchronize()
            with record_function('fused_head/weight_transpose/forward'):
                wt = w.T.contiguous()
                torch.neuron.synchronize()
            with record_function('fused_head/forward'):
                arms['fused_head'](x, wt, target)
                torch.neuron.synchronize()
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
    print('FUSED_HEAD_PROFILE_ARTIFACT ' + json.dumps(dict(
        trace=str(trace), shape=list(shape), compiled_neffs=len(entries),
        eager_neffs=len(eager['files']), phase_boundary_synchronization=True,
        numerical_authority='fs-w101-fused-head-loss-forward-component-r1',
        native_and_candidate_logits_saved=True, weight_transpose_separately_captured=True,
        profiled_latency_is_speed_evidence=False, backward_or_whole_model_included=False)), flush=True)


if __name__ == '__main__':
    main()
