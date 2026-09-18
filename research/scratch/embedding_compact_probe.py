#!/usr/bin/env python3
"""Alternative: compact the embedding-gradient vocabulary within a batch.

The fixed tokenizer and model table stay unchanged. Map present token IDs to
local rows, use native dense backward for a smaller fixed bucket, then NKI
scatter unique rows into the ordinary dense gradient. Padding uses out-of-bound
rows with DMA skip; no two valid rows have the same destination. Metadata
creation/upload and output clearing/scatter are included in complete timings.
"""
import argparse
import json
import math

import numpy as np
import torch

from embedding_segmented_probe import (check, gradient_for, indices_for, reference, time_arms)


def compact_metadata(ids, vocabulary, minimum_bucket=128):
    unique, inverse = np.unique(ids.numpy().reshape(-1), return_inverse=True)
    assert minimum_bucket >= 128 and minimum_bucket & (minimum_bucket - 1) == 0
    bucket = min(vocabulary, max(minimum_bucket, 2 ** math.ceil(math.log2(len(unique)))))
    rows = np.full(bucket, vocabulary, dtype=np.int32)
    rows[:len(unique)] = unique
    assert len(np.unique(rows[:len(unique)])) == len(unique)
    return torch.from_numpy(inverse.astype(np.int32)), torch.from_numpy(rows).reshape(-1, 1), len(unique)


def build_scatter():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def scatter_unique(compact, row_ids, vocabulary: int, lanes: int = 2):
        bucket, channels = compact.shape
        assert bucket % 128 == 0 and vocabulary % 128 == 0 and channels % lanes == 0
        width = channels // lanes
        assert lanes in (1, 2) and row_ids.shape == (bucket, 1)
        out = nl.ndarray((vocabulary, channels), dtype=nl.float32, buffer=nl.shared_hbm)
        c0 = nl.program_id(0) * width
        zero = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zero, value=0.0)
        # Programs own disjoint channel ranges for BOTH clearing and scatter.
        # No cross-program clear-versus-scatter race or barrier is needed.
        for vi in nl.sequential_range(vocabulary // 128):
            nisa.dma_copy(dst=out[vi * 128:(vi + 1) * 128, c0:c0 + width], src=zero)
        for bi in nl.sequential_range(bucket // 128):
            values = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.sbuf)
            rows = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=values, src=compact[bi * 128:(bi + 1) * 128, c0:c0 + width])
            nisa.dma_copy(dst=rows, src=row_ids[bi * 128:(bi + 1) * 128, :])
            nisa.dma_copy(dst=out.ap(pattern=[[channels, 128], [1, width]], offset=c0,
                                     vector_offset=rows, indirect_dim=0), src=values,
                          oob_mode=nisa.oob_mode.skip, dge_mode=nisa.dge_mode.swdge)
        return out

    return scatter_unique


def run_case(args, scatter, kind, shape):
    tokens, channels, vocabulary = shape
    ids = indices_for(kind, tokens, vocabulary, 731)
    g = gradient_for(kind, tokens, channels, 732)
    inv, rows, unique = compact_metadata(ids, vocabulary, args.minimum_bucket)
    bucket = len(rows)
    exact, bound, counts = reference(g, ids, vocabulary)
    local = torch.zeros(bucket, channels).index_add_(0, inv.long(), g)
    cpu_out = torch.zeros(vocabulary, channels)
    cpu_out[rows[:unique, 0].long()] = local[:unique]
    check(cpu_out, exact, bound, counts, f"compact_cpu:{kind}:{shape}")
    print("EMBED_COMPACT_CASE " + json.dumps(dict(kind=kind, shape=shape, bucket=bucket,
                                                unique=unique, metadata_bytes=inv.numel()*4+rows.numel()*4)), flush=True)
    if args.device == "cpu":
        return
    gd, idd, invd, rowsd = (x.to("neuron") for x in (g, ids, inv, rows))
    # Check scatter independently with FP32 CPU sums, before native BF16 math.
    scatter_only = scatter[2](local.to("neuron"), rowsd, vocabulary, 2)
    check(scatter_only, exact, bound, counts, f"compact_scatter:{kind}:{shape}")
    assert torch.equal(scatter_only.cpu(), cpu_out), "unique scatter changed FP32 values"

    from torch_neuronx import wrap_nki
    wrapped = wrap_nki(scatter)

    def baseline(gg, ii):
        return torch.ops.aten.embedding_dense_backward.default(gg, ii, vocabulary, -1, False)

    def compact_path(gg, ii, rr):
        small = torch.ops.aten.embedding_dense_backward.default(gg, ii, bucket, -1, False)
        return wrapped[2](small, rr, vocabulary, 2)

    base = torch.compile(baseline, backend="neuron", fullgraph=True, dynamic=False)
    compact_fn = torch.compile(compact_path, backend="neuron", fullgraph=True, dynamic=False)
    actual = compact_fn(gd, invd, rowsd).float().cpu()
    native = base(gd, idd).float().cpu()
    # Both matmuls intentionally use the same production BF16 autocasting.
    rounded_ref, rounded_bound, _ = reference(g.bfloat16().float(), ids, vocabulary)
    check(actual, rounded_ref, rounded_bound, counts, f"compact_native_bf16:{kind}:{shape}")
    difference = (actual - native).abs()
    print("EMBED_COMPACT_NATIVE_DIFFERENCE " + json.dumps(dict(
        kind=kind, max_abs=float(difference.max()), exact=bool(torch.equal(actual, native)),
        absent_rows_exact_zero=bool((actual[counts == 0] == 0).all()))), flush=True)
    if kind in ("one_id", "zero", "cancellation"):
        return

    def complete():
        new_inv, new_rows, _ = compact_metadata(ids, vocabulary, args.minimum_bucket)
        assert len(new_rows) == bucket
        return compact_fn(gd, new_inv.to("neuron"), new_rows.to("neuron"))

    timed = time_arms(dict(native_full_vocab=lambda: base(gd, idd),
                          compact_prepared_metadata=lambda: compact_fn(gd, invd, rowsd),
                          compact_metadata_creation_upload_included=complete),
                     args.steps, torch.neuron.synchronize)
    med = timed["median_ms"]
    estimate = med["compact_prepared_metadata"] + max(
        0, med["compact_metadata_creation_upload_included"] - med["compact_prepared_metadata"]) / 4
    print("EMBED_COMPACT_BENCH " + json.dumps(dict(kind=kind, shape=shape, bucket=bucket, **timed,
        estimated_four_table_amortized_ms=estimate, estimate_is_not_whole_model_timing=True,
        provisional_speed_gate=(estimate < 0.8 * med["native_full_vocab"]),
        output_zeroing_and_scatter_included=True, training_speed_or_quality_claim=False)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "neuron"), default="neuron")
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--training-row", action="store_true")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--minimum-bucket", type=int, default=4096,
                    help="Fixed ordinary-case workspace; overflow grows to the full vocabulary.")
    args = ap.parse_args()
    torch.set_num_threads(4)
    scatter = None
    if args.device == "neuron":
        import torch_neuronx  # noqa: F401
        scatter = build_scatter()
    shapes = [(256, 256, 256)] + ([(8192, 1024, 8192)] if args.real_shape else [])
    for shape in shapes:
        kinds = ["uniform", "zipf", "one_id", "zero", "cancellation"]
        if shape[0] == 8192 and args.training_row:
            kinds.append("training")
        for kind in kinds:
            run_case(args, scatter, kind, shape)
    print("PASS: compact ordinary-embedding component probe; no training claim", flush=True)


if __name__ == "__main__":
    main()
