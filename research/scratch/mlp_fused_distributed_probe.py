#!/usr/bin/env python3
"""Separate full-chip contention from the surrounding model's MLP regression.

All four ranks execute the same benchmark arm concurrently. Synchronization and
an explicit barrier precede each timing window; each rank reports its own local
completion time before the closing barrier. No optimizer or whole-model claim.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
import mlp_fused_bf16_probe as probe
from mlp_fused_bf16 import build_kernel


def timing(arms, steps, sync):
    for fn in arms.values():
        dist.barrier()
        for _ in range(3):
            fn()
        sync()
        dist.barrier()
    samples = {name: [] for name in arms}
    for repeat in range(3):
        names = list(arms) if repeat % 2 == 0 else list(reversed(arms))
        for name in names:
            sync()
            dist.barrier()
            start = time.perf_counter()
            for _ in range(steps):
                arms[name]()
            sync()
            samples[name].append(1000 * (time.perf_counter() - start) / steps)
            dist.barrier()
    return dict(samples_ms=samples,
                median_ms={name: statistics.median(v) for name, v in samples.items()},
                four_rank_concurrent_arms=True, closing_barrier_inside_timer=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--grouped-io', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    import torch_neuronx  # noqa:F401
    ddp, rank, _, world, _ = T.init_runtime('neuron')
    assert ddp and world == 4

    def emit(kind, record):
        print(kind+' '+json.dumps(dict(**record, rank=rank, world_size=world)), flush=True)

    probe.emit = emit
    probe.timing = timing
    try:
        probe.run_case('neuron', (8192, 1024, 4096), 'random', build_kernel(), args.steps,
                       contiguous_weights=True, tiled_io=True, cached_bf16_weights=True,
                       transposed_relu=True, grouped_io=args.grouped_io)
        dist.barrier()
        emit('DISTRIBUTED_MLP_GATE', dict(passed=True, grouped_io=args.grouped_io,
                                        interpretation='component contention diagnostic only'))
    finally:
        T.cleanup_runtime()


if __name__ == '__main__':
    main()
