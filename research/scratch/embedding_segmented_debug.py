#!/usr/bin/env python3
"""Locate the real-shape segmented-gradient failure; diagnostic, no timing.

Instrumented stages change allocation/lifetimes, so passing a stage does not
prove the original composed kernel correct. Report errors without stopping at
the first expected failure, and compare the original kernel separately.
"""
import json

import numpy as np
import torch

from embedding_segmented_nki import build_kernel, cpu_scan, metadata
from embedding_segmented_probe import gradient_for, indices_for


def build_stages():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def stages(gradient, permutation, continuation, last, present,
               stage: int, lanes: int = 2, cont_fp32: bool = False):
        tokens, channels = gradient.shape
        vocabulary = last.shape[1]
        length = vocabulary if stage == 3 else tokens
        out = nl.ndarray((channels, length), dtype=nl.float32, buffer=nl.shared_hbm)
        if stage >= 1:
            perm = nl.ndarray((128, tokens), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=perm, src=permutation.ap(pattern=[[0, 128], [1, tokens]], offset=0))
        if stage >= 2:
            cdtype = nl.float32 if cont_fp32 else nl.bfloat16
            cont = nl.ndarray((128, tokens), dtype=cdtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=cont, src=continuation.ap(pattern=[[0, 128], [1, tokens]], offset=0))
        lane = nl.program_id(0)
        for block in nl.sequential_range(channels // (128 * lanes)):
            c0 = (block * lanes + lane) * 128
            raw = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw,
                          src=gradient.ap(pattern=[[1, 128], [channels, tokens]], offset=c0))
            if stage == 0:
                nisa.dma_copy(dst=out[c0:c0 + 128, :], src=raw)
            if stage >= 1:
                ordered = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
                nisa.nc_n_gather(dst=ordered, data=raw, indices=perm)
                if stage == 1:
                    nisa.dma_copy(dst=out[c0:c0 + 128, :], src=ordered)
            if stage >= 2:
                scanned = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor_scan(dst=scanned, data0=cont, data1=ordered,
                                        initial=0.0, op0=nl.multiply, op1=nl.add)
                if stage == 2:
                    nisa.dma_copy(dst=out[c0:c0 + 128, :], src=scanned)
            if stage == 3:
                vt = min(vocabulary, 512)
                for vi in nl.sequential_range(vocabulary // vt):
                    ends = nl.ndarray((128, vt), dtype=nl.uint32, buffer=nl.sbuf)
                    valid = nl.ndarray((128, vt), dtype=nl.bfloat16, buffer=nl.sbuf)
                    sums = nl.ndarray((128, vt), dtype=nl.float32, buffer=nl.sbuf)
                    masked = nl.ndarray((128, vt), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(dst=ends, src=last.ap(pattern=[[0, 128], [1, vt]], offset=vi * vt))
                    nisa.dma_copy(dst=valid, src=present.ap(pattern=[[0, 128], [1, vt]], offset=vi * vt))
                    nisa.nc_n_gather(dst=sums, data=scanned, indices=ends)
                    nisa.tensor_tensor(dst=masked, data1=sums, data2=valid, op=nl.multiply)
                    nisa.dma_copy(dst=out[c0:c0 + 128, vi * vt:(vi + 1) * vt], src=masked)
        return out

    return stages


def describe(name, actual, expected):
    # All comparisons are CPU [C,T] or [C,V], against a literal FP32 recurrence.
    a = actual.detach().float().cpu()
    error = (a - expected).abs()
    failed = error > (1e-7 + expected.abs() * 1e-5)
    rows = []
    for start in range(0, a.shape[0], 128):
        rows.append(dict(channel_start=start, failures=int(failed[start:start + 128].sum()),
                         max_abs=float(error[start:start + 128].max())))
    locations = failed.nonzero()[:12]
    samples = [dict(channel=int(c), offset=int(t), actual=float(a[c, t]), expected=float(expected[c, t]))
               for c, t in locations]
    record = dict(name=name, shape=list(a.shape), finite=bool(torch.isfinite(a).all()),
                  failed_elements=int(failed.sum()), max_abs=float(error.max()),
                  channel_blocks=rows,
                  offset_blocks=[int(failed[:, t:t + 512].sum()) for t in range(0, a.shape[1], 512)],
                  samples=samples, instrumented=name.startswith("stage"))
    print("EMBED_SEGMENT_DEBUG " + json.dumps(record), flush=True)
    return record


def main():
    torch.set_num_threads(4)
    import torch_neuronx  # noqa: F401
    kernel, stages = build_kernel(), build_stages()
    for tokens, channels, vocabulary in [(256, 1024, 256), (4096, 256, 4096),
                                          (8192, 256, 8192), (8192, 1024, 8192)]:
        ids = indices_for("uniform", tokens, vocabulary, 731)
        g = gradient_for("uniform", tokens, channels, 732)
        meta = metadata(ids, vocabulary)
        expected = cpu_scan(g, meta).T.contiguous()
        gd = g.to("neuron")
        md = tuple(m.to("neuron") for m in meta)
        for lanes in (2, 1):
            output = kernel[lanes](gd, *md, lanes)
            describe(f"original:T{tokens}:C{channels}:lanes{lanes}", output.T, expected)
        if tokens == 8192 and channels == 1024:
            ordered = g[meta[0].long().reshape(-1)].T.contiguous()
            scan = np.empty_like(ordered.numpy())
            state = np.zeros(channels, dtype=np.float32)
            continuation = meta[1].numpy().reshape(-1)
            for i in range(tokens):
                state = state * continuation[i] + ordered.numpy()[:, i]
                scan[:, i] = state
            refs = [g.T.contiguous(), ordered, torch.from_numpy(scan), expected]
            for stage in range(4):
                output = stages[2](gd, *md, stage, 2, False)
                describe(f"stage{stage}:contbf16", output, refs[stage])
            for stage in (2, 3):
                output = stages[2](gd, *md, stage, 2, True)
                describe(f"stage{stage}:contfp32", output, refs[stage])
    print("Completed stage diagnostics; inspect failure records, not a kernel PASS", flush=True)


if __name__ == "__main__":
    main()
