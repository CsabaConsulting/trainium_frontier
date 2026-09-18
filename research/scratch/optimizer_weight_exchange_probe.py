#!/usr/bin/env python3
"""Measure exact FP32 weight exchange before implementing optimizer ownership.

No model forward, gradient, optimizer update or training result is produced.
The parameter manifest comes from the same dense6x1024 GPT/optimizer grouping.
Timing includes persistent-buffer packing and copying back into Parameters.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T


def dense_manifest(world_size):
    T.DEPTH, T.N_EMBD = 6, 1024
    T.USE_NGRAM_VE = T.USE_QK_SHIFT = False
    T.QK_SHIFT_BETA, T.QK_SHIFT_FREEZE = 0.0, True
    T.USE_LOCAL_CONV = T.TRAIN_LOCAL_CONV = True
    T.USE_FOREACH_OPTIM = T.USE_TENSOR_LR_SCALARS = True
    config = T.build_config()
    with torch.device("meta"):
        model = T.GPT(config)
        optimizer = model.setup_optimizer()
    names = {id(p): name for name, p in model.named_parameters()}
    entries, seen = [], set()
    # Per-parameter estimates derived from the existing replicated optimizer
    # trace. They choose a deterministic whole-matrix assignment, not a claim
    # that runtime will scale linearly with batch size.
    muon_ms = {(1024, 1024): 12.54413 / 24,
               (1024, 4096): 9.76316 / 6,
               (4096, 1024): 8.98251 / 6,
               (8, 32): .23355 / 3}
    for group_index, group in enumerate(optimizer.param_groups):
        for p in group["params"]:
            assert id(p) not in seen and p.dtype == torch.float32
            seen.add(id(p))
            cost = (muon_ms.get(tuple(p.shape), .01) if group["kind"] == "muon"
                    else 4.00487945 * p.numel() / (8192 * 1024) + .01)
            entries.append(dict(index=len(entries), name=names[id(p)],
                                shape=list(p.shape), elements=p.numel(),
                                group=group_index, kind=group["kind"],
                                estimated_ms=cost))
    assert len(seen) == len(list(model.parameters())), "unassigned parameter"
    assert sum(e["elements"] for e in entries) == 117453580
    costs, counts = [0.0] * world_size, [0] * world_size
    for entry in sorted(entries, key=lambda e: (-e["estimated_ms"], -e["elements"], e["index"])):
        owner = min(range(world_size), key=lambda r: (costs[r], counts[r], r))
        entry["owner"] = owner
        entry["offset"] = counts[owner]
        counts[owner] += entry["elements"]
        costs[owner] += entry["estimated_ms"]
    # Equal all-gather input sizes, with a conservative DMA-aligned tail.
    shard_elements = (max(counts) + 127) // 128 * 128
    return entries, dict(model=asdict(config), elements_by_owner=counts,
                         estimated_update_ms_by_owner=costs,
                         shard_elements=shard_elements,
                         total_padded_bytes=shard_elements * world_size * 4,
                         total_parameter_bytes=sum(counts) * 4,
                         entries=entries)


def expected_pattern(entry):
    return (torch.arange(entry["elements"], dtype=torch.float32).remainder_(251)
            .mul_(.125).add_(entry["owner"] * 10000 + entry["index"] * 40)
            .view(entry["shape"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=3)
    args = parser.parse_args()
    assert args.steps > 0 and args.blocks > 0
    import torch_neuronx
    torch.set_num_threads(4)
    _, rank, _, world_size, device = T.init_runtime("neuron")
    assert world_size == 4
    try:
        tiny = torch.arange(256, dtype=torch.float32).add_(rank * 1024).to(device)
        received_tiny = torch.empty(256 * world_size, device=device)
        dist.all_gather_into_tensor(received_tiny, tiny)
        T.synchronize(device)
        expected_tiny = torch.cat([torch.arange(256).float() + r * 1024 for r in range(world_size)])
        assert torch.equal(received_tiny.cpu(), expected_tiny)
        print("OPTIMIZER_EXCHANGE_TINY " + json.dumps(dict(rank=rank, exact=True)), flush=True)
        entries, manifest = dense_manifest(world_size)
        if rank == 0:
            print("OPTIMIZER_EXCHANGE_MANIFEST " + json.dumps(manifest), flush=True)
        stride = manifest["shard_elements"]
        send = torch.zeros(stride, dtype=torch.float32, device=device)
        received = torch.empty(stride * world_size, dtype=torch.float32, device=device)
        parameters, owned, send_views, receive_views = [], [], [], []
        for entry in entries:
            p = torch.nn.Parameter(torch.empty(entry["shape"], dtype=torch.float32, device=device))
            parameters.append(p)
            receive_views.append(received.narrow(
                0, entry["owner"] * stride + entry["offset"], entry["elements"]).view(entry["shape"]))
            with torch.no_grad():
                if entry["owner"] == rank:
                    p.copy_(expected_pattern(entry))
                    owned.append(p)
                    send_views.append(send.narrow(0, entry["offset"], entry["elements"]).view(entry["shape"]))
                else:
                    p.fill_(-1000)

        def pack():
            torch._foreach_copy_(send_views, owned)

        def gather():
            dist.all_gather_into_tensor(received, send)

        def scatter():
            torch._foreach_copy_(parameters, receive_views)

        def full():
            pack()
            gather()
            scatter()

        def local_copies():
            pack()
            scatter()

        parameter_ids = [id(p) for p in parameters]
        with torch.no_grad():
            full()
            T.synchronize(device)
            assert [id(p) for p in parameters] == parameter_ids
            assert all(torch.equal(p.cpu(), expected_pattern(e)) for e, p in zip(entries, parameters))
            assert [id(p) for p in parameters] == parameter_ids
            versions = [p._version for p in parameters]
            full()
            T.synchronize(device)
            # Report the backend's foreach mutation behavior, rather than
            # imposing Tensor.copy_'s version contract on a different op.
            version_deltas = [p._version - old for p, old in zip(parameters, versions)]
            print("OPTIMIZER_EXCHANGE_CORRECTNESS " + json.dumps(dict(
                rank=rank, all_parameter_patterns_exact=True,
                parameter_ids_preserved=True, version_deltas=sorted(set(version_deltas)),
                fp32_exchange=True, optimizer_updated=False)), flush=True)
            arms = {"all_gather_only": gather, "pack_gather_scatter": full,
                    "pack_scatter_only": local_copies}
            # Warm every native shape/collective before recording any samples.
            for call in arms.values():
                for _ in range(3):
                    call()
                    T.synchronize(device)
            samples = {name: [] for name in arms}
            order = list(arms)
            for block in range(args.blocks):
                for name in order[block % len(order):] + order[:block % len(order)]:
                    dist.barrier()
                    T.synchronize(device)
                    begin = time.perf_counter()
                    for _ in range(args.steps):
                        arms[name]()
                        T.synchronize(device)
                    samples[name].append((time.perf_counter() - begin) * 1000 / args.steps)
            assert all(torch.equal(p.cpu(), expected_pattern(e)) for e, p in zip(entries, parameters))
            print("OPTIMIZER_EXCHANGE_BENCH " + json.dumps(dict(
                rank=rank, world_size=world_size, steps_per_block=args.steps,
                blocks=args.blocks, samples_ms=samples,
                median_ms={name: statistics.median(values) for name, values in samples.items()},
                total_parameter_bytes=manifest["total_parameter_bytes"],
                total_padded_bytes=manifest["total_padded_bytes"],
                allocation_excluded=True, pack_and_copyback_included=True,
                optimizer_updated=False, whole_model=False)), flush=True)
        dist.barrier()
    finally:
        T.cleanup_runtime()


if __name__ == "__main__":
    main()
