#!/usr/bin/env python3
"""Bounded ordinary-embedding backward comparison; no training speed claim.

Compare the native compiled backward, the proposed FP32 segmented scan, and
native embedding/indexing forward+backward controls at the current 8192x1024
shape. NKI timing includes both gradient layouts. Report metadata preparation
and upload separately and together; four-table amortization is only an estimate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F

from embedding_segmented_nki import build_kernel, cpu_scan, metadata


def reference(gradient, ids, vocabulary):
    ref = torch.zeros(vocabulary, gradient.shape[1], dtype=torch.float64)
    absolute = torch.zeros_like(ref)
    ref.index_add_(0, ids.long(), gradient.double())
    absolute.index_add_(0, ids.long(), gradient.double().abs())
    counts = torch.bincount(ids.long(), minlength=vocabulary)
    u = 2.0**-24
    nu = (counts - 1).clamp_min(0).double() * u
    bound = (nu / (1 - nu))[:, None] * absolute + 2e-7 * ref.abs() + 1e-7
    return ref, bound, counts


def check(actual, ref, bound, counts, name):
    actual = actual.detach().float().cpu()
    error = (actual.double() - ref).abs()
    result = dict(name=name, finite=bool(torch.isfinite(actual).all()),
                  max_abs=float(error.max()), bound_failures=int((error > bound).sum()),
                  absent_rows_exact_zero=bool((actual[counts == 0] == 0).all()),
                  relative_rms=float(error.square().mean().sqrt() /
                                     ref.square().mean().sqrt().clamp_min(1e-30)))
    result["passed"] = result["finite"] and not result["bound_failures"] and result["absent_rows_exact_zero"]
    print("EMBED_SEGMENT_GATE " + json.dumps(result), flush=True)
    assert result["passed"], result
    return result


def indices_for(kind, tokens, vocabulary, seed):
    gen = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        return torch.randint(vocabulary, (tokens,), generator=gen, dtype=torch.int32)
    if kind in ("one_id", "zero", "cancellation"):
        return torch.full((tokens,), vocabulary // 3, dtype=torch.int32)
    if kind == "zipf":
        p = torch.arange(1, vocabulary + 1, dtype=torch.float64).pow(-1.1)
        return torch.multinomial(p, tokens, replacement=True, generator=gen).int()
    assert kind == "training" and tokens == 8192 and vocabulary == 8192
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import prepare
    tokenizer = prepare.ensure_tokenizer(build_if_missing=False)
    row, _, _ = next(prepare.make_dataloader(tokenizer, 1, tokens, "train", "cpu", shard_ids=[0]))
    return row.reshape(-1).clone()


def gradient_for(kind, tokens, channels, seed):
    gen = torch.Generator().manual_seed(seed)
    g = torch.randn(tokens, channels, generator=gen) * 0.01
    if kind == "zero":
        g.zero_()
    if kind == "cancellation":
        g[::2] = 1e4
        g[1::2] = -1e4 + 0.001
    return g


def time_arms(arms, steps, sync):
    for fn in arms.values():
        for _ in range(3):
            fn()
        sync()
    samples = {name: [] for name in arms}
    for repeat in range(3):
        names = list(arms) if repeat % 2 == 0 else list(reversed(arms))
        for name in names:
            sync()
            start = time.perf_counter()
            for _ in range(steps):
                arms[name]()
            sync()
            samples[name].append((time.perf_counter() - start) * 1000 / steps)
    return dict(samples_ms=samples, median_ms={k: statistics.median(v) for k, v in samples.items()})


def run_case(args, kernel, kind, shape):
    tokens, channels, vocabulary = shape
    ids = indices_for(kind, tokens, vocabulary, args.seed)
    gradient = gradient_for(kind, tokens, channels, args.seed + 1)
    meta = metadata(ids, vocabulary)
    ref, bound, counts = reference(gradient, ids, vocabulary)
    literal = cpu_scan(gradient, meta)
    check(literal, ref, bound, counts, f"cpu:{kind}:{shape}")
    print("EMBED_SEGMENT_CASE " + json.dumps(dict(
        kind=kind, shape=shape, unique_ids=int((counts > 0).sum()), max_count=int(counts.max()),
        ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        metadata_bytes=sum(m.numel() * m.element_size() for m in meta),
        seed=args.seed, compiler_flags=os.environ.get("NEURON_CC_FLAGS"),
        compiled_kernel=args.compiled_kernel,
        manual_sbuf=args.manual_sbuf,
        reuse_sbuf=args.reuse_sbuf,
        aligned_sbuf=args.aligned_sbuf, gather_tile=args.gather_tile,
        gradient_dtype="float32", preparation_uses_training_only=(kind == "training"))), flush=True)
    if args.device == "cpu":
        return
    sync = torch.neuron.synchronize
    gd, idd = gradient.to("neuron"), ids.to("neuron")
    md = tuple(m.to("neuron") for m in meta)
    lanes = 2 if channels % 256 == 0 else 1
    chunk = args.split_channels or channels
    assert channels % chunk == 0 and chunk % (128 * lanes) == 0

    def invoke(gg, *metadata_args):
        parts = [kernel[lanes](gg[:, start:start + chunk].contiguous(), *metadata_args,
                              lanes, args.manual_sbuf, False, args.reuse_sbuf,
                              args.aligned_sbuf, args.gather_tile)
                 for start in range(0, channels, chunk)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    if args.compiled_kernel:
        check(invoke(gd, *md), ref, bound, counts, f"nki_eager:{kind}:{shape}")
        from torch_neuronx import wrap_nki
        wrapped = wrap_nki(kernel)

        def captured(gg, pp, cc, ee, vv):
            parts = [wrapped[lanes](gg[:, start:start + chunk].contiguous(), pp, cc, ee, vv,
                                   lanes, args.manual_sbuf, False, args.reuse_sbuf,
                                   args.aligned_sbuf, args.gather_tile)
                     for start in range(0, channels, chunk)]
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

        invoke = torch.compile(captured, backend="neuron", fullgraph=True, dynamic=False)
    candidate = lambda: invoke(gd, *md)
    result = candidate()
    check(result, ref, bound, counts, f"nki:{kind}:{shape}")
    # A second different input through the same compiled shape tests that no
    # scratch state from the preceding call enters the result.
    zero = invoke(torch.zeros_like(gd), *md).cpu()
    assert (zero == 0).all()
    assert torch.equal(result.cpu(), candidate().cpu())
    if kind in ("zero", "cancellation", "one_id"):
        return

    def backward(g, i):
        return torch.ops.aten.embedding_dense_backward.default(g, i, vocabulary, -1, False)
    native = torch.compile(backward, backend="neuron", dynamic=False)
    native_result = native(gd, idd).float().cpu()
    native_error = (native_result.double() - ref).abs()
    bf16_ref = torch.zeros_like(ref).index_add_(0, ids.long(), gradient.bfloat16().double())
    print("EMBED_NATIVE_REFERENCE " + json.dumps(dict(
        kind=kind, shape=shape, finite=bool(torch.isfinite(native_result).all()),
        max_abs_to_fp64=float(native_error.max()),
        relative_rms_to_fp64=float(native_error.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-30)),
        max_abs_to_bf16_inputs=float((native_result.double() - bf16_ref).abs().max()),
        arithmetic_may_differ_from_fp32_scan=True)), flush=True)
    assert torch.isfinite(native_result).all()

    def upload_only():
        return tuple(m.to("neuron") for m in meta)

    def complete():
        fresh = tuple(m.to("neuron") for m in metadata(ids, vocabulary))
        return invoke(gd, *fresh)

    arms = dict(native_compiled_backward=lambda: native(gd, idd),
                nki_prepared_metadata=candidate,
                metadata_cpu_only=lambda: metadata(ids, vocabulary),
                metadata_upload_only=upload_only,
                nki_metadata_creation_upload_included=complete)
    timed = time_arms(arms, args.steps, sync)
    med = timed["median_ms"]
    estimate = med["nki_prepared_metadata"] + max(0, med["nki_metadata_creation_upload_included"] - med["nki_prepared_metadata"]) / 4
    print("EMBED_SEGMENT_BENCH " + json.dumps(dict(
        kind=kind, shape=shape, **timed,
        nki_all_gradient_layouts_included=True, estimated_four_table_amortized_ms=estimate,
        amortization_is_estimate_not_training_measurement=True,
        provisional_speed_gate=(estimate < 0.8 * med["native_compiled_backward"]),
        training_speed_or_quality_claim=False)), flush=True)

    if tokens == 8192 and kind == "training":
        # Check the direct-op comparison against the real autograd formulations.
        w = torch.nn.Parameter(torch.randn(vocabulary, channels, generator=torch.Generator().manual_seed(99)).to("neuron") * 0.02)
        embed = torch.compile(lambda p, i: F.embedding(i, p), backend="neuron", dynamic=False)
        indexing = torch.compile(lambda p, i: p[i], backend="neuron", dynamic=False)

        def fwdbwd(fn):
            w.grad = None
            output = fn(w, idd)
            output.backward(gd)
            return w.grad

        for name, fn in (("embedding", embed), ("advanced_index", indexing)):
            val = fwdbwd(fn).float().cpu()
            assert torch.isfinite(val).all()
            print("EMBED_AUTOGRAD_REFERENCE " + json.dumps(dict(
                name=name, max_abs_to_fp64=float((val.double() - ref).abs().max()))), flush=True)
        pairs = time_arms(dict(native_embedding_fwdbwd=lambda: fwdbwd(embed),
                              native_advanced_index_fwdbwd=lambda: fwdbwd(indexing)), args.steps, sync)
        print("EMBED_AUTOGRAD_BENCH " + json.dumps(dict(shape=shape, **pairs)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "neuron"), default="neuron")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=731)
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--training-row", action="store_true")
    ap.add_argument("--compiled-kernel", action="store_true",
                    help="Gate eager and captured NKI, then time the captured wrapper like native.")
    ap.add_argument("--manual-sbuf", action="store_true",
                    help="Use explicit disjoint byte addresses for every scratch tile.")
    ap.add_argument("--reuse-sbuf", action="store_true",
                    help="Allocate scratch outside loops and reuse the same tensor identities.")
    ap.add_argument("--aligned-sbuf", action="store_true",
                    help="Place FP32/uint32 tiles on4*T-byte boundaries before BF16 scratch.")
    ap.add_argument("--gather-tile", type=int, default=8192,
                    help="Maximum outputs per permutation gather; source keeps full token range.")
    ap.add_argument("--split-channels", type=int, default=0,
                    help="Use independent narrow kernels, including input copies and output concatenation.")
    args = ap.parse_args()
    torch.set_num_threads(4)
    kernel = None
    if args.device == "neuron":
        import torch_neuronx  # noqa: F401
        kernel = build_kernel()
    shapes = [(256, 256, 256)]
    if args.real_shape:
        shapes.append((8192, 1024, 8192))
    for shape in shapes:
        kinds = ["uniform", "zipf", "one_id", "zero", "cancellation"]
        if shape[0] == 8192 and args.training_row:
            kinds.append("training")
        for kind in kinds:
            run_case(args, kernel, kind, shape)
    print("PASS: ordinary embedding segmented-gradient component probe; no training claim", flush=True)


if __name__ == "__main__":
    main()
