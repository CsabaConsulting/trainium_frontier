#!/usr/bin/env python3
"""Attribute the already gated branch-norm component regression on hardware."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_neuronx
from torch.profiler import ProfilerActivity, profile, record_function
from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

from branch_norm_nki import make_function, reference
from mlp_separate_diagnose import autograd_nodes
from preserve_native_trace_neffs import preserve_trace_neffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--inverse-mode", type=int, choices=(1, 2), default=1)
    parser.add_argument("--packed-inverse", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(784)
    trace = args.trace_dir.resolve()
    trace.mkdir(parents=True, exist_ok=False)
    config = NeuronConfig(modes=[ProfileMode.RUNTIME, ProfileMode.DEVICE],
                          profile_output_dir=str(trace), max_events_per_nc=1000000,
                          precache_node_info=True)
    base = torch.randn(8192, 1024)
    gradient = torch.randn_like(base).bfloat16().to("neuron")
    functions = {"native": torch.compile(reference, backend="neuron", fullgraph=True, dynamic=False),
                 "nki": torch.compile(make_function(args.inverse_mode, args.packed_inverse), backend="neuron",
                                      fullgraph=True, dynamic=False)}
    leaves = {name: base.to("neuron").requires_grad_() for name in functions}

    def forward(name):
        leaves[name].grad = None
        y = functions[name](leaves[name])
        assert autograd_nodes(y).get("CompiledFunctionBackward", 0) >= 1
        return y

    for _ in range(3):
        for name in functions:
            forward(name).backward(gradient)
    torch.neuron.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                 experimental_config=config) as prof:
        for name in functions:
            with record_function(name + "_branch_norm_forward"):
                y = forward(name)
                torch.neuron.synchronize()
            with record_function(name + "_branch_norm_backward"):
                y.backward(gradient)
                torch.neuron.synchronize()
    NeuronProfiler(config).export_trace(prof)
    from torch_neuronx.neuron_dynamo_backend.cache_artifact import ArtifactType
    from torch_neuronx.neuron_dynamo_backend.compile import get_compiled_cache_keys
    export = trace / "neffs"
    export.mkdir(exist_ok=True)
    entries, missing = [], []
    for key in sorted(get_compiled_cache_keys(ArtifactType.NEURON_NEFF)):
        info = torch_neuronx._C.get_neff_info(key)
        if info is None:
            missing.append(key)
            continue
        persistent_key, data = info
        destination = export / (persistent_key + ".neff")
        if destination.exists():
            assert destination.read_bytes() == data
        else:
            destination.write_bytes(data)
        entries.append(dict(cache_key=key, persistent_cache_key=persistent_key,
                            file=destination.name, bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest()))
    (export / "manifest.json").write_text(json.dumps(dict(exported=entries, missing=missing), indent=2) + "\n")
    eager = preserve_trace_neffs(trace, Path("/tmp/neff_cache"), export)
    assert not missing and not eager["unresolved"]
    print("BRANCH_NORM_PROFILE_ARTIFACT " + json.dumps(dict(
        trace=str(trace), shape=[8192, 1024], inverse_mode=args.inverse_mode,
        packed_inverse=args.packed_inverse,
        compiled_neffs=len(entries), eager_neffs=len(eager["files"]),
        profiled_latency_is_performance_evidence=False)), flush=True)


if __name__ == "__main__":
    main()
