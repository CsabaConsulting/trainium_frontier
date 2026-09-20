#!/usr/bin/env python3
"""Measure unchanged source194 saved forwards with one/four concurrent ranks.

The complete source194 numerical proof is a frozen launch prerequisite. This
screen adds exact actual-stage, repeat and changed-input checks on every rank.
All online master casts, weight preparation and real state stores stay inside
the extracted original functions. Backward/model/quality are not measured.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import statistics
import sys
import types
import time

import torch
import torch.distributed as dist
import torch_neuronx

from fp8_mlp_island_probe import make_inputs
from nki_library_probe_common import emit, exact, verify_sources
import nki_library_row_training_forward_probe as original
import train as T


def build_functions(shape):
    path = Path(original.inline.original.__file__)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == 'c19bd4b3ff2856d3aeb77512777f9e1c73b4a1fce50728b89887804324c9d4f4'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'run_case')
    transformed = original.transform_run_case(node)
    names = ('prepare', 'training_complete')
    nodes = {n.name: n for n in transformed.body
             if isinstance(n, ast.FunctionDef) and n.name in names}
    assert set(nodes) == set(names)
    module = ast.fix_missing_locations(ast.Module(
        body=[nodes[n] for n in names], type_ignores=[]))
    original.install()
    # Dynamo resolves closure-function globals through their declared module.
    # A copied dict that still names the original probe misdirects its guards:
    # that module has no hidden/channels/weight_prepare attributes. Register
    # this isolated namespace before creating the unchanged function bodies.
    module_name = '_trainium_saved_row_source194_w199'
    assert module_name not in sys.modules
    scope = types.ModuleType(module_name)
    namespace = vars(scope)
    namespace.update(original.inline.original.run_case.__globals__)
    namespace.update(__name__=module_name, __package__='', __spec__=None,
                     tokens=shape[0], channels=shape[1], hidden=shape[2])
    namespace['quantizer'] = torch_neuronx.wrap_nki(original.inline.original.quantize_rows)
    sys.modules[module_name] = scope
    exec(compile(module, str(path), 'exec'), namespace)
    assert all(namespace[name].__globals__ is namespace
               and namespace[name].__module__ == module_name for name in names)
    # The reference exposes actual source191 stages, as in the original194
    # gate. It is outside timing and is not an unquantized BF16 reference.
    prepare = namespace['prepare']
    inline_mlp = namespace['inline_mlp']

    def actual_stages(xx, uu, dd):
        prepared = prepare(xx, uu, dd)
        values = inline_mlp[2](xx, *prepared[1:5], True)
        return (values[0], values[2], values[5], values[6][:, 1:2],
                values[3], values[4])

    emit('NKILIB_SAVED_CONTENTION_FUNCTIONS', dict(
        original_probe_sha256=digest,
        source194_probe_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
        extracted_ast_sha256=hashlib.sha256(ast.dump(module).encode()).hexdigest(),
        original_function_bodies_unchanged=True,
        python_module_globals_registered=True,
        native_control_saved_relu_layout='HxT',
        all_online_preparation_and_saves_inside_compiled_functions=True))
    original.inline.original.install()
    functions = dict(native=original.native_saved_forward,
                     row_fp8=namespace['training_complete'], actual_stages=actual_stages)
    return {name: torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
            for name, fn in functions.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=6)
    args = parser.parse_args()
    assert 30 <= args.steps <= 200 and 6 <= args.rounds <= 10 and args.rounds % 2 == 0
    torch.set_num_threads(4)
    ddp, rank, local_rank, world, device = T.init_runtime('neuron', seed=42)
    assert world in (1, 4)
    try:
        verify_sources()
        shape = (32768, 1024, 4096)
        functions = build_functions(shape)
        x, up, down = make_inputs(shape, 'real_random')
        values = tuple(v.to(device) for v in (x, up.float(), down.float()))
        names = ('native', 'row_fp8')
        snapshots = {}
        for name in names:
            snapshots[name] = tuple(v.detach().cpu() for v in functions[name](*values))
            assert all(bool(torch.isfinite(v).all()) for v in snapshots[name])
            exact(f'contention:{world}:{rank}:{name}:repeat',
                  functions[name](*values), snapshots[name])
        expected = tuple(v.detach().cpu() for v in functions['actual_stages'](*values))
        exact(f'contention:{world}:{rank}:all_actual_stages', snapshots['row_fp8'], expected)
        saved_bytes = sum(v.numel() * v.element_size() for v in snapshots['row_fp8'][1:])
        assert saved_bytes == shape[0] * (3 * shape[2] + shape[1] + 8)
        changed_x = x.clone()
        changed_x[1] *= -2
        changed_values = (changed_x.to(device), *values[1:])
        changed = tuple(v.detach().cpu() for v in functions['row_fp8'](*changed_values))
        exact(f'contention:{world}:{rank}:changed_actual_stages',
              changed, functions['actual_stages'](*changed_values))
        assert not torch.equal(changed[0][1], snapshots['row_fp8'][0][1])
        keep = torch.arange(shape[0]) != 1
        exact(f'contention:{world}:{rank}:all_saved_row_independence',
              tuple(v[keep] for v in changed),
              tuple(v[keep] for v in snapshots['row_fp8']))
        changed_weights = (values[0], (up.float() * .5).to(device),
                           (down.float() * 1.5).to(device))
        changed = tuple(v.detach().cpu() for v in functions['row_fp8'](*changed_weights))
        exact(f'contention:{world}:{rank}:changed_weights_actual_stages',
              changed, functions['actual_stages'](*changed_weights))
        assert not torch.equal(changed[0], snapshots['row_fp8'][0])
        del snapshots, expected, changed, changed_values, changed_weights
        from torch_neuronx.neuron_dynamo_backend.cache_artifact import ArtifactType
        from torch_neuronx.neuron_dynamo_backend.compile import get_compiled_cache_keys
        compiled_keys = list(get_compiled_cache_keys(ArtifactType.NEURON_NEFF))
        assert len(compiled_keys) >= 3
        for name in names:
            for _ in range(5):
                functions[name](*values)
            T.synchronize(device)

        def barrier():
            if ddp:
                dist.barrier()
            T.synchronize(device)

        samples = {name: [] for name in names}
        for repeat in range(args.rounds):
            order = names if repeat % 2 == 0 else tuple(reversed(names))
            for name in order:
                barrier()
                start = time.perf_counter()
                for _ in range(args.steps):
                    functions[name](*values)
                T.synchronize(device)
                local_ms = 1000 * (time.perf_counter() - start) / args.steps
                maximum = torch.tensor(local_ms, dtype=torch.float32, device=device)
                if ddp:
                    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                maximum_ms = float(maximum.cpu())
                assert 0 < local_ms <= maximum_ms + .001
                samples[name].append(maximum_ms)
                emit('NKILIB_SAVED_CONTENTION_TIMING', dict(
                    world_size=world, rank=rank, arm=name, repeat=repeat,
                    steps=args.steps, local_ms=local_ms, maximum_rank_ms=maximum_ms,
                    collectives_outside_timing=True, paired_alternating_order=True,
                    online_preparation_and_all_saves_charged=True))
        barrier()
        emit('NKILIB_SAVED_CONTENTION_RANK_GATE', dict(
            passed=True, world_size=world, rank=rank, shape=list(shape),
            exact_actual_stages_repeat_changed_inputs_and_weights=True,
            exact_all_saved_row_independence=True, saved_tensor_bytes=saved_bytes,
            original_source194_numerical_prerequisite_required=True,
            original_forward_function_bodies_unchanged=True,
            master_casts_preparation_and_all_training_saves_charged=True,
            compiled_neffs=len(compiled_keys), backward_and_model_excluded=True))
        if rank == 0:
            medians = {name: statistics.median(v) for name, v in samples.items()}
            emit('NKILIB_SAVED_CONTENTION_RESULT', dict(
                passed=True, world_size=world, shape=list(shape),
                steps=args.steps, rounds=args.rounds, maximum_rank_samples_ms=samples,
                median_ms=medians,
                fp8_regression_percent=100 * (medians['row_fp8'] / medians['native'] - 1),
                source194_function_bodies_reused=True,
                original_numerical_gates_not_replaced=True,
                no_backward_model_or_bpb_claim=True))
    finally:
        T.cleanup_runtime()


if __name__ == '__main__':
    main()
