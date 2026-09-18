#!/usr/bin/env python3
"""Isolate nc_n_gather composition failures, without training or speed claims.

Data encodes its original row/channel. A post-gather dependent arithmetic tag
reads the index tile, testing whether indices still equal their input values.
Compare loop-local allocation with explicit reuse of the same SBUF tiles. The
extra instrumentation changes allocation and may hide the original failure;
passing this probe does not validate the uninstrumented segmented kernel.
"""
import argparse
import json

import torch


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def gather_probe(data, indices, lanes: int = 2, reuse: bool = False):
        tokens, channels = data.shape
        assert channels % (128 * lanes) == 0 and indices.shape == (1, tokens)
        out = nl.ndarray((channels, tokens), dtype=nl.float32, buffer=nl.shared_hbm)
        tags = nl.ndarray((channels, tokens), dtype=nl.uint32, buffer=nl.shared_hbm)
        index = nl.ndarray((128, tokens), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=index, src=indices.ap(pattern=[[0, 128], [1, tokens]], offset=0))
        if reuse:
            raw = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
            gathered = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
            dependent = nl.ndarray((128, tokens), dtype=nl.uint32, buffer=nl.sbuf)
            tagged = nl.ndarray((128, tokens), dtype=nl.uint32, buffer=nl.sbuf)
        for block in nl.sequential_range(channels // (128 * lanes)):
            c0 = (block * lanes + nl.program_id(0)) * 128
            if not reuse:
                raw = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
                gathered = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
                dependent = nl.ndarray((128, tokens), dtype=nl.uint32, buffer=nl.sbuf)
                tagged = nl.ndarray((128, tokens), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw,
                          src=data.ap(pattern=[[1, 128], [channels, tokens]], offset=c0))
            nisa.nc_n_gather(dst=gathered, data=raw, indices=index)
            # The tag's value depends on the gather output; this is not a
            # before/after copy which could legally move before the gather.
            nisa.tensor_copy(dst=dependent, src=gathered)
            nisa.tensor_tensor(dst=tagged, data1=index, data2=dependent, op=nl.add)
            nisa.dma_copy(dst=out[c0:c0 + 128, :], src=gathered)
            nisa.dma_copy(dst=tags[c0:c0 + 128, :], src=tagged)
        return out, tags

    return gather_probe


def summarize(name, outputs, expected, permutation):
    gathered, tags = (x.cpu() for x in outputs)
    actual_indices = tags.long() - gathered.to(torch.uint32).long()
    expected_indices = permutation.reshape(1, -1).long().expand_as(actual_indices)
    bad_values = gathered != expected
    bad_indices = actual_indices != expected_indices
    samples = []
    for c, t in (bad_values | bad_indices).nonzero()[:32]:
        samples.append(dict(channel=int(c), offset=int(t), actual=float(gathered[c, t]),
                            expected=float(expected[c, t]), decoded_index=int(actual_indices[c, t]),
                            expected_index=int(expected_indices[c, t])))
    record = dict(name=name, shape=list(gathered.shape), finite=bool(torch.isfinite(gathered).all()),
                  wrong_values=int(bad_values.sum()), changed_index_reads=int(bad_indices.sum()),
                  max_abs=float((gathered - expected).abs().max()), samples=samples,
                  channel_blocks=[dict(start=c, wrong_values=int(bad_values[c:c + 128].sum()),
                                       changed_index_reads=int(bad_indices[c:c + 128].sum()))
                                  for c in range(0, gathered.shape[0], 128)],
                  instrumented=True, passing_is_not_original_kernel_validation=True)
    print("GATHER_LIFETIME_DIAGNOSTIC " + json.dumps(record), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiled", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    import torch_neuronx
    kernel = build_kernel()
    wrapped = torch_neuronx.wrap_nki(kernel)
    for tokens, channels, lanes in [(256, 256, 1), (512, 256, 1), (1024, 256, 1),
                                     (8192, 256, 1), (8192, 1024, 2)]:
        permutation = torch.randperm(tokens, generator=torch.Generator().manual_seed(91)).int().reshape(1, -1)
        data = (torch.arange(tokens).float()[:, None]
                + torch.arange(channels).float()[None, :] / 2048)
        expected = data[permutation.long().reshape(-1)].T.contiguous()
        dd, ii = data.to("neuron"), permutation.to("neuron")
        for reuse in (False, True):
            name = f"T{tokens}:C{channels}:lanes{lanes}:reuse{reuse}"
            summarize("eager:" + name, kernel[lanes](dd, ii, lanes, reuse), expected, permutation)
            if args.compiled:
                # Each configuration is an independent diagnostic, not a
                # dynamic-shape workload. Avoid exhausting the default eight
                # Dynamo specializations of this shared Python code object.
                torch._dynamo.reset()
                def call(a, b):
                    return wrapped[lanes](a, b, lanes, reuse)
                compiled = torch.compile(call, backend="neuron", fullgraph=True, dynamic=False)
                summarize("compiled:" + name, compiled(dd, ii), expected, permutation)
    print("Completed gather lifetime diagnostics; inspect records, not a kernel PASS", flush=True)


if __name__ == "__main__":
    main()
