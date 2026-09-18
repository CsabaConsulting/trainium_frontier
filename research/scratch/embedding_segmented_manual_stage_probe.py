#!/usr/bin/env python3
"""Locate the remaining error in the explicitly allocated full scan kernel.

Unlike separate stage-only kernels, the instrumented arm keeps the complete
computation and returns raw/permuted/scanned intermediates alongside its dense
result. Extra DMA stores still change scheduling, so compare the uninstrumented
arm too. These are diagnostic records, not a timing or training correctness pass.
"""
import json

import numpy as np
import torch

from embedding_segmented_debug import describe
from embedding_segmented_nki import build_kernel, cpu_scan, metadata
from embedding_segmented_probe import gradient_for, indices_for


def main():
    torch.set_num_threads(4)
    import torch_neuronx
    tokens, channels, vocabulary = 8192, 1024, 8192
    ids = indices_for('uniform', tokens, vocabulary, 731)
    g = gradient_for('uniform', tokens, channels, 732)
    meta = metadata(ids, vocabulary)
    result = cpu_scan(g, meta).T.contiguous()
    raw = g.T.contiguous()
    ordered = g[meta[0].long().reshape(-1)].T.contiguous()
    values = ordered.numpy()
    flags = meta[1].numpy().reshape(-1)
    scan = np.empty_like(values)
    accumulator = np.zeros(channels, dtype=np.float32)
    for t in range(tokens):
        accumulator = accumulator * flags[t] + values[:, t]
        scan[:, t] = accumulator
    refs = (raw, ordered, torch.from_numpy(scan))
    gd = g.to('neuron')
    md = tuple(t.to('neuron') for t in meta)
    kernel = build_kernel()
    wrapped = torch_neuronx.wrap_nki(kernel)
    for debug, reuse in ((True, False), (False, False), (True, True), (False, True)):
        torch._dynamo.reset()

        def captured(gg, pp, cc, ll, vv):
            return wrapped[2](gg, pp, cc, ll, vv, 2, True, debug, reuse)

        compiled = torch.compile(captured, backend='neuron', fullgraph=True, dynamic=False)
        for mode, call in [('eager',lambda:kernel[2](gd,*md,2,True,debug,reuse)),
                           ('compiled',lambda:compiled(gd,*md))]:
            output = call()
            name = f'{mode}:manual_sbuf:debug{debug}:reuse{reuse}'
            if debug:
                final, *stages = output
                for stage, (actual, expected) in enumerate(zip(stages,refs)):
                    describe(f'stage{stage}:{name}',actual,expected)
            else:
                final = output
            describe(f'final:{name}',final.T,result)
    print('Completed manual full-scan stage diagnostics; no kernel PASS',flush=True)


if __name__=='__main__':
    main()
