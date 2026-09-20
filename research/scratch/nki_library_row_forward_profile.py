#!/usr/bin/env python3
"""Capture all three gated ROW forward arms on one clean host.

Function bodies are extracted unchanged from the source191 probe transform.
All input/weight preparation remains inside each compiled call. This capture
diagnoses engine work and spills; instrumented latency is not a speed result.
No training saves, backward, optimizer or model-quality claim is implied.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import torch
import torch_neuronx
from torch.profiler import ProfilerActivity, profile, record_function
from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

import nki_library_row_fused_input_probe as fused_probe
from fp8_mlp_island_probe import make_inputs
from mlp_fused_bf16_probe import timing
from nki_library_probe_common import emit, exact, verify_sources
from preserve_native_trace_neffs import preserve_trace_neffs


def build_functions(shape):
    path = Path(fused_probe.original.__file__)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == 'c19bd4b3ff2856d3aeb77512777f9e1c73b4a1fce50728b89887804324c9d4f4'
    original_node = next(n for n in ast.parse(path.read_text()).body
                         if isinstance(n, ast.FunctionDef) and n.name == 'run_case')
    transformed = fused_probe.transform_run_case(original_node)
    names = ('prepare', 'prepared_complete', 'complete')
    nodes = {n.name: n for n in transformed.body
             if isinstance(n, ast.FunctionDef) and n.name in names}
    assert set(nodes) == set(names)
    module = ast.fix_missing_locations(ast.Module(body=[nodes[n] for n in names], type_ignores=[]))
    # install() supplies the identical wrapped NKI producers and entry kernels.
    fused_probe.install()
    namespace = dict(fused_probe.original.run_case.__globals__)
    namespace.update(tokens=shape[0], channels=shape[1], hidden=shape[2])
    namespace['quantizer'] = torch_neuronx.wrap_nki(fused_probe.original.quantize_rows)
    namespace['mlp'] = torch_neuronx.wrap_nki(fused_probe.original.row_mlp)
    exec(compile(module, str(path), 'exec'), namespace)
    functions = dict(native=fused_probe.original.native,
                     prepared=namespace['prepared_complete'], inline=namespace['complete'])
    emit('NKILIB_ROW_PROFILE_FUNCTIONS', dict(
        original_probe_sha256=digest,
        extracted_ast_sha256=hashlib.sha256(ast.dump(module).encode()).hexdigest(),
        original_function_bodies_unchanged=True,
        all_online_preparation_inside_compiled_functions=True,
        source191_transform_and_native_control_reused=True))
    return {name: torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
            for name, fn in functions.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    assert 1 <= args.steps <= 100
    torch.set_num_threads(4)
    trace = args.trace_dir.resolve()
    trace.parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(trace.parent).free >= 10 * 1024**3
    trace.mkdir(exist_ok=False)
    verify_sources()
    shape = (32768, 1024, 4096)
    functions = build_functions(shape)
    fused_probe.original.install()
    x, up, down = make_inputs(shape, 'real_random')
    values = tuple(v.to('neuron') for v in (x, up.float(), down.float()))
    outputs = {name: fn(*values) for name, fn in functions.items()}
    torch.neuron.synchronize()
    assert all(bool(torch.isfinite(value.cpu()).all()) for value in outputs.values())
    exact('profile_inline_prepared', outputs['inline'], outputs['prepared'])
    calls = {name: (lambda fn=fn: fn(*values)) for name, fn in functions.items()}
    measured = timing(calls, args.steps, torch.neuron.synchronize)
    emit('NKILIB_ROW_PROFILE_UNINSTRUMENTED', dict(shape=list(shape), **measured,
        online_input_and_current_weight_preparation_included=True,
        saved_activations_and_backward_excluded=True, training_or_bpb_claim=False))
    assert shutil.disk_usage(trace.parent).free >= 10 * 1024**3
    config = NeuronConfig(modes=[ProfileMode.RUNTIME, ProfileMode.DEVICE],
        profile_output_dir=str(trace), max_events_per_nc=1000000, precache_node_info=True)
    captured = {}
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                 experimental_config=config) as prof:
        for name, fn in calls.items():
            with record_function('library_row/' + name):
                captured[name] = fn()
                torch.neuron.synchronize()
    NeuronProfiler(config).export_trace(prof)
    for name in captured:
        exact('profile_capture_repeat:' + name, captured[name], outputs[name])

    from torch_neuronx.neuron_dynamo_backend.cache_artifact import ArtifactType
    from torch_neuronx.neuron_dynamo_backend.compile import get_compiled_cache_keys
    destination = trace / 'neffs'
    destination.mkdir(exist_ok=True)
    exported, missing = [], []
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
    (destination / 'manifest.json').write_text(json.dumps(
        dict(exported=exported, missing=missing), indent=2) + '\n')
    eager = preserve_trace_neffs(trace, Path('/tmp/neff_cache'), destination)
    assert not missing and not eager['unresolved']
    emit('NKILIB_ROW_PROFILE_RESULT', dict(passed=True, trace=str(trace), shape=list(shape),
        arms=list(calls), profiled_forwards=3, same_host_and_inputs=True,
        online_preparation_inside_each_arm=True, captured_outputs_exact=True,
        compiled_neffs=len(exported), missing_neffs=missing,
        unresolved_trace_neffs=eager['unresolved'], phase_synchronization=True,
        numerical_authority='fs-w191-nkilib-inline-input-forward-r1',
        profiled_latency_is_speed_evidence=False, full_training_or_bpb_claim=False,
        dma_notification_completeness_not_assumed=True))


if __name__ == '__main__':
    main()
