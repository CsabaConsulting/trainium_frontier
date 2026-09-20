#!/usr/bin/env python3
"""Compare the unchanged training MLP with one versus four concurrent ranks.

The frozen source175 numerical proof remains a prerequisite. This diagnostic
adds full native comparisons and exact repeat checks on every participating
rank, and rejects compiler fallback on every measured invocation. It measures
the complete MLP including master casts, layouts, saved ReLU, and backward;
it does not measure optimizer, model context, or training quality.
"""
import argparse
import statistics
import time

import torch
import torch.distributed as dist
import torch_neuronx

from mlp_fused_bf16 import make_autograd
from mlp_fused_bf16_probe import compare
from nki_library_probe_common import emit, exact, verify_sources
from nki_library_training_mlp_reuse_transpose import install, saved_relu_mlp_reuse_transpose
from nki_library_training_mlp_reuse_transpose_probe import reference_forward
import train as T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    assert args.steps >= 20 and args.rounds >= 5
    torch.set_num_threads(4)
    ddp, rank, local_rank, world, device = T.init_runtime('neuron', seed=42)
    assert world in (1, 4)
    try:
        verify_sources()
        install()
        tokens, channels = 32768, 1024
        hidden = 4 * channels
        generator = torch.Generator().manual_seed(171021 + tokens + channels)
        x = torch.randn(tokens, channels, generator=generator).bfloat16()
        up = torch.randn(hidden, channels, generator=generator) / channels ** .5
        down = torch.randn(channels, hidden, generator=generator) / hidden ** .5
        grad = (torch.randn(x.shape, generator=generator) * .01).bfloat16().to(device)
        wrapped = torch_neuronx.wrap_nki(saved_relu_mlp_reuse_transpose)

        def kernel_call(xx, uu, dd):
            return wrapped[2](xx, uu.bfloat16().T.contiguous(), dd.bfloat16().T.contiguous())

        native_apply = make_autograd(reference_forward, transposed_relu=True)
        library_apply = make_autograd(kernel_call, transposed_relu=True)

        # Distinct root code objects prevent shared compiler skip/cache state.
        def native_root(xx, uu, dd):
            return native_apply(xx, uu, dd)

        def library_root(xx, uu, dd):
            return library_apply(xx, uu, dd)

        arms = {
            'native': torch.compile(native_root, backend='neuron', fullgraph=True, dynamic=False),
            'library': torch.compile(library_root, backend='neuron', fullgraph=True, dynamic=False),
        }
        values = {name: tuple(v.to(device).detach().clone().requires_grad_()
                             for v in (x, up, down)) for name in arms}

        def train(name):
            leaves = values[name]
            for leaf in leaves:
                leaf.grad = None
            output = arms[name](*leaves)
            assert type(output.grad_fn).__name__ == 'CompiledFunctionBackward', (name, type(output.grad_fn))
            output.backward(grad)
            assert leaves[0].grad.dtype == torch.bfloat16
            assert all(v.grad.dtype == torch.float32 for v in leaves[1:])
            return (output, *(v.grad for v in leaves))

        expected, actual = train('native'), train('library')
        for field, got, want in zip(('output', 'dx', 'dup', 'ddown'), actual, expected):
            compare(f'chip:{world}:{rank}:{field}', got, want)
        for name, outputs in (('native', expected), ('library', actual)):
            snapshot = tuple(v.detach().cpu() for v in outputs)
            exact(f'chip:{world}:{rank}:{name}:repeat', train(name), snapshot)
        del expected, actual, snapshot
        for name in arms:
            for _ in range(3):
                train(name)
            T.synchronize(device)

        def barrier():
            if ddp:
                dist.barrier()
            T.synchronize(device)

        samples = {name: [] for name in arms}
        for repeat in range(args.rounds):
            order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
            for name in order:
                barrier()
                start = time.perf_counter()
                for _ in range(args.steps):
                    train(name)
                T.synchronize(device)
                local_ms = 1000 * (time.perf_counter() - start) / args.steps
                # Collective and scalar read are deliberately outside timing.
                maximum = torch.tensor(local_ms, dtype=torch.float32, device=device)
                if ddp:
                    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                maximum_ms = float(maximum.cpu())
                assert 0 < local_ms <= maximum_ms + .001
                samples[name].append(maximum_ms)
                emit('NKILIB_CHIP_TIMING', dict(world_size=world, rank=rank,
                    arm=name, repeat=repeat, steps=args.steps, local_ms=local_ms,
                    maximum_rank_ms=maximum_ms, collectives_outside_timing=True,
                    compiled_backward_asserted=True))
        barrier()
        emit('NKILIB_CHIP_RANK_GATE', dict(passed=True, world_size=world, rank=rank,
            shape=[tokens, channels], full_native_output_and_gradients=True,
            exact_both_arm_repeat=True, fp32_master_gradients=True,
            compiled_backward_on_every_invocation=True,
            master_casts_layouts_saved_activation_and_backward_charged=True,
            independent_fp64_proof_reused_from_source175=True,
            model_optimizer_communication_and_quality_excluded=True))
        if rank == 0:
            medians = {name: statistics.median(v) for name, v in samples.items()}
            emit('NKILIB_CHIP_RESULT', dict(passed=True, world_size=world,
                shape=[tokens, channels], steps=args.steps, rounds=args.rounds,
                maximum_rank_samples_ms=samples, median_ms=medians,
                library_regression_percent=100 * (medians['library'] / medians['native'] - 1),
                source175_component_reproduced=True, fixed_seed_fixture=True,
                no_full_model_or_bpb_claim=True))
    finally:
        T.cleanup_runtime()


if __name__ == '__main__':
    main()
