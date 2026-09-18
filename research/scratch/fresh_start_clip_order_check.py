"""CPU check of the real clip/sync helper using two Gloo ranks.

Run from any directory with the repository's dependencies installed:
    python research/scratch/fresh_start_clip_order_check.py

Uses actual backward accumulation, then checks the resulting SGD update against
independently averaged/clipped gradients. Gloo has no AVG on some Torch versions,
so its all_reduce adapter explicitly implements AVG as SUM followed by division.
No Trainium device is initialized or used. Covers historical order, global order,
separate n-gram clipping (FP32 and BF16), flat buffers, clipping disabled, and
rejection of deferred-gradient/global-clip and flat/separate-clip combinations.
"""

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T


WORLD_SIZE = 2


class TinyModel(torch.nn.Module):
    def __init__(self, table_dtype=torch.float32):
        super().__init__()
        self.main = torch.nn.Parameter(torch.tensor([0.75, -0.5, 0.25]))
        self.table = torch.nn.Parameter(torch.tensor([[0.25, -0.5], [0.75, 1.0]],
                                                      dtype=table_dtype))


def rank_grads(rank, table_dtype):
    # Non-collinear, unequal-norm gradients make clipping order consequential.
    return [torch.tensor([10.0, 1.0, -2.0] if rank == 0 else [-1.0, 4.0, 0.5]),
            torch.tensor([[0.0, 3.0], [-1.0, 0.0]] if rank == 0 else
                         [[8.0, 0.0], [0.0, -2.0]], dtype=table_dtype)]


def accumulate(model, rank):
    # Two micro-batches whose contributions differ but whose mean is rank_grads.
    grads = rank_grads(rank, model.table.dtype)
    for factor in (0.5, 1.5):
        loss = sum((p * (g * factor)).sum() for p, g in zip(model.parameters(), grads))
        (loss / 2).backward()


def manual_clip(grads, max_norm, separate):
    out = [g.clone() for g in grads]
    if max_norm <= 0:
        return out
    groups = [[i] for i in range(len(out))] if separate else [list(range(len(out)))]
    for indices in groups:
        # Match clip_grad_norm_'s per-tensor norm dtype, while deriving the final
        # expected vector/update independently of the production clip/sync helper.
        norms = torch.stack([torch.linalg.vector_norm(out[i]) for i in indices])
        norm = torch.linalg.vector_norm(norms)
        scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
        for i in indices:
            out[i].mul_(scale)
    return out


def manual_average(per_rank):
    return [(per_rank[0][i] + per_rank[1][i]) / WORLD_SIZE
            for i in range(len(per_rank[0]))]


def legacy_clip_and_sync(model, main, ngram, max_norm):
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(main, max_norm, foreach=False)
        if ngram:
            torch.nn.utils.clip_grad_norm_(ngram, max_norm, foreach=False)
    T.sync_gradients(model)


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}",
                            rank=rank, world_size=WORLD_SIZE)
    raw_all_reduce = dist.all_reduce

    def gloo_avg(tensor, op=dist.ReduceOp.SUM, **kwargs):
        if op == dist.ReduceOp.AVG:
            raw_all_reduce(tensor, op=dist.ReduceOp.SUM, **kwargs)
            tensor.div_(WORLD_SIZE)
        else:
            raw_all_reduce(tensor, op=op, **kwargs)

    records = []
    try:
        with patch.object(T.dist, "all_reduce", side_effect=gloo_avg):
            for dtype in (torch.float32, torch.bfloat16):
                for separate in (False, True):
                    # A flat fp32 buffer is only meaningful for all-fp32 params.
                    flat_modes = (False, True) if dtype == torch.float32 and not separate else (False,)
                    for flat in flat_modes:
                        for max_norm in (0.0, 1.0):
                            for after in (False, True):
                                model = TinyModel(dtype)
                                initial = [p.detach().clone() for p in model.parameters()]
                                buffer = T.build_flat_grad_buffer(model)[0] if flat else None
                                accumulate(model, rank)
                                main = [model.main] if separate else list(model.parameters())
                                ngram = [model.table] if separate else []
                                T.clip_and_sync_gradients(
                                    model, main, ngram, max_norm=max_norm,
                                    clip_after_reduce=after, grad_flat_buffer=buffer,
                                    foreach=False,
                                )
                                per_rank = [rank_grads(r, dtype) for r in range(WORLD_SIZE)]
                                expected = (manual_clip(manual_average(per_rank), max_norm, separate)
                                            if after else manual_average(
                                                [manual_clip(g, max_norm, separate) for g in per_rank]))
                                for p, g in zip(model.parameters(), expected):
                                    torch.testing.assert_close(p.grad, g, rtol=2e-6, atol=2e-7)

                                # Compare the actual resulting parameter update, not only norms.
                                with torch.no_grad():
                                    for p, start, g in zip(model.parameters(), initial, expected):
                                        p.sub_(0.125 * p.grad)
                                        torch.testing.assert_close(p, start - 0.125 * g,
                                                                   rtol=2e-6, atol=2e-7)

                                if not after and not flat:
                                    legacy = TinyModel(dtype)
                                    accumulate(legacy, rank)
                                    legacy_clip_and_sync(legacy,
                                        [legacy.main] if separate else list(legacy.parameters()),
                                        [legacy.table] if separate else [], max_norm)
                                    for p, old in zip(model.parameters(), legacy.parameters()):
                                        assert torch.equal(p.grad, old.grad), "default path changed"
                                records.append({"dtype": str(dtype), "separate": separate,
                                                "flat": flat, "max_norm": max_norm,
                                                "clip_after_reduce": after})

            local = [rank_grads(r, torch.float32) for r in range(WORLD_SIZE)]
            before = manual_average([manual_clip(g, 1.0, True) for g in local])
            after = manual_clip(manual_average(local), 1.0, True)
            order_difference = max(float((a - b).abs().max()) for a, b in zip(before, after))
            assert order_difference > 0.1, "test gradients do not discriminate the algorithms"

            model = TinyModel()
            for kwargs in (
                {"clip_after_reduce": True, "skip_param_ids": frozenset({id(model.table)})},
                {"grad_flat_buffer": torch.zeros(7)},
            ):
                try:
                    T.clip_and_sync_gradients(model, [model.main], [model.table],
                                             max_norm=1.0, **kwargs)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"unsupported combination accepted: {kwargs}")

        assert T.CLIP_AFTER_REDUCE is False
        assert T.current_toggles()["CLIP_AFTER_REDUCE"] is False
        if rank == 0:
            print(json.dumps({"status": "PASS", "ranks": WORLD_SIZE,
                              "cases_per_rank": len(records),
                              "historical_path_bit_exact": True,
                              "order_max_abs_difference": order_difference,
                              "collective": "Gloo SUM / world_size implements AVG"}), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory(prefix="fresh-start-clip-check-") as directory:
        mp.spawn(worker, args=(str(Path(directory) / "rendezvous"),),
                 nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
